#pragma once


#include <stdio.h>
#include <string.h>
#include <math.h>

#include "config.h"
//#include "util.h"






typedef float fp32_t;
typedef double fp64_t;

typedef long long int64_t;
typedef unsigned long long uint64_t;

typedef signed char int8_t;
typedef unsigned char uint8_t;

typedef int int32_t;
typedef unsigned int uint32_t;





extern char _embedded_binary_model[];

typedef struct MambaWeights MambaWeights;
typedef struct MambaConfig  MambaConfig;
typedef struct MambaState   MambaState;
typedef struct Mamba        Mamba;

static void    MambaLog(Mamba *);
static fp32_t *MambaForwardLayer(Mamba *, uint64_t);
static fp32_t *MambaForward(Mamba *, uint64_t);

#ifndef QUANTIZED
#define Tensor(NAME, X, Y, Z)  fp32_t NAME[(X) * (Y) * (Z)]

struct MambaWeights {
    Tensor(embed,			ROUNDED_VOCAB_SIZE,     D_MODEL,                    1);
    Tensor(in_proj, 		N_LAYER,        2 * D_INNER,                D_MODEL);
    Tensor(conv1d_weight,	N_LAYER,        D_INNER,                    D_CONV);
    Tensor(conv1d_bias, 	N_LAYER,        D_INNER,                    1);
    Tensor(x_proj,			N_LAYER,        DT_RANK + 2 * D_STATE, 	    D_INNER);
    Tensor(dt_proj_weight,	N_LAYER,        D_INNER,                    DT_RANK);
    Tensor(dt_proj_bias,	N_LAYER,        D_INNER,                    1);
    Tensor(A,				N_LAYER,        D_INNER,                    D_STATE);
    Tensor(D, 				N_LAYER,        D_INNER,                    1);
    Tensor(out_proj, 		N_LAYER,        D_MODEL,                    D_INNER);
    Tensor(norm, 			N_LAYER,        D_MODEL,                    1);
    Tensor(norm_f, 			D_MODEL,        1,                          1);
    Tensor(lm_head, 		ROUNDED_VOCAB_SIZE,    	D_MODEL,                    1);
};
#else
#define GS 64

#define Tensor(NAME, X, Y, Z)  fp32_t NAME[(X) * (Y) * (Z)]
#define QTensor(NAME, X, Y, Z) struct { int8_t q[(X) * (Y) * (Z)]; fp32_t s[((X) * (Y) * (Z)) / GS]; } NAME
#define QPointer(NAME, QT, P)  struct { int8_t *q; fp32_t *s; } NAME = { .q = QT.q + (P), .s = QT.s + ((P) / GS) }

struct MambaWeights {
    QTensor(embed,			ROUNDED_VOCAB_SIZE,     D_MODEL,                    1);
    QTensor(in_proj, 		N_LAYER,        2 * D_INNER,                D_MODEL);
    
	Tensor(conv1d_weight,	N_LAYER,        D_INNER,                    D_CONV);
    Tensor(conv1d_bias, 	N_LAYER,        D_INNER,                    1);
    
	QTensor(x_proj,			N_LAYER,        DT_RANK + 2 * D_STATE,      D_INNER);
    
	Tensor(dt_proj_weight,	N_LAYER,        D_INNER,                    DT_RANK);
	Tensor(dt_proj_bias,	N_LAYER,        D_INNER,                    1);
    Tensor(A,				N_LAYER,        D_INNER,                    D_STATE);
    Tensor(D, 				N_LAYER,        D_INNER,                    1);
    
	QTensor(out_proj, 		N_LAYER,        D_MODEL,                    D_INNER);
    
	Tensor(norm, 			N_LAYER,        D_MODEL,                    1);
    Tensor(norm_f, 			D_MODEL,        1,                          1);
    
	QTensor(lm_head, 		ROUNDED_VOCAB_SIZE,    	D_MODEL,                    1);
};
#endif

struct MambaConfig {
    uint64_t vocab_size; // vocabulary size, rounded to nearest multiple of 8
    uint64_t n_layer;    // number of layers
    uint64_t d_model;    // embedding dim
    uint64_t d_inner;
    uint64_t dt_rank;
    uint64_t d_state;
    uint64_t d_conv;
};

struct MambaState {
	Tensor(hidden_state,    D_MODEL,        1,               			1);
    Tensor(conv_state,      N_LAYER,        D_INNER,         			D_CONV);
    Tensor(ssm_state,       N_LAYER,        D_INNER,         			D_STATE);
};

struct Mamba {
    MambaConfig config;
    MambaState state;

    MambaWeights *weights;

	void (*log) (Mamba *);
	fp32_t *(*forward_layer) (Mamba *, uint64_t);
	fp32_t *(*forward) (Mamba *, uint64_t);
};


Mamba mamba = {
    .config = {
        .vocab_size = ROUNDED_VOCAB_SIZE,
        .n_layer = N_LAYER,
        .d_model = D_MODEL,
        .d_inner = D_INNER,
        .dt_rank = DT_RANK,
        .d_state = D_STATE,
        .d_conv = D_CONV,
    },

	.state = {},

    .weights = (MambaWeights *) _embedded_binary_model,

	.log = MambaLog,
	.forward_layer = MambaForwardLayer,
	.forward = MambaForward
};




static void rmsnorm(fp32_t* y, fp32_t* x, fp32_t* weight, uint64_t size) {
    fp32_t ss = 0.0f;
    for (uint64_t j = 0; j < size; ++j)
        ss += x[j] * x[j];

    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);

    for (uint64_t j = 0; j < size; ++j)
        y[j] = x[j] * weight[j] * ss;
}

static void softmax(fp32_t* x, uint64_t size) {
    fp32_t max_val = x[0];
    for (uint64_t i = 1; i < size; ++i)
        if (x[i] > max_val) max_val = x[i];

    fp32_t sum = 0.0f;
    for (uint64_t i = 0; i < size; ++i) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    for (uint64_t i = 0; i < size; ++i)
        x[i] /= sum;
}

static fp32_t softplus(fp32_t x) { return logf(1.0f + expf(x)); }
static fp32_t sigmoid(fp32_t x) { return 1.0f / (1.0f + expf(-x)); }
static fp32_t silu(fp32_t x) { return x * sigmoid(x); }

static void shift_and_update_last_column(fp32_t* matrix, fp32_t* x, uint64_t rows, uint64_t cols) {
    #pragma omp parallel for
    for (uint64_t i = 0; i < rows; ++i) {
        fp32_t* row = matrix + i * cols;

        for (uint64_t j = 0; j < cols - 1; ++j)
            row[j] = row[j + 1];

        row[cols - 1] = x[i];
    }
}

static void conv1d_silu(fp32_t* x, fp32_t* conv_state, fp32_t* conv1d_weight, fp32_t* conv1d_bias, uint64_t d_inner, uint64_t d_conv) {
    #pragma omp parallel for
    for (uint64_t i = 0; i < d_inner; ++i) {
        fp32_t val = 0.0f;

        for (uint64_t j = 0; j < d_conv; ++j) {
            uint64_t index = i * d_conv + j;
            val += conv_state[index] * conv1d_weight[index];
        }

        x[i] = silu(val + conv1d_bias[i]);
    }
}

static void dense_softplus(fp32_t* y, fp32_t* x, fp32_t* w, fp32_t* b, uint64_t d, uint64_t n) {

    #pragma omp parallel for
    for (uint64_t i = 0; i < d; ++i) {
        fp32_t val = 0.0f;

        for (uint64_t j = 0; j < n; ++j)
            val += w[i * n + j] * x[j];

        y[i] = softplus(val + b[i]);
    }
}

static void selective_scan(fp32_t* y, fp32_t* ssm_state, fp32_t* dt, fp32_t* A, fp32_t* B, fp32_t* C, fp32_t* D, fp32_t* x, fp32_t* z, uint64_t d_inner, uint64_t d_state) {
    #pragma omp parallel for
    for (uint64_t i = 0; i < d_inner; ++i) {
        fp32_t val = 0.0f;

        for (uint64_t j = 0; j < d_state; ++j) {
            uint64_t index = i * d_state + j;

            fp32_t dA = expf(dt[i] * A[index]);
            fp32_t dB = dt[i] * B[j];

            ssm_state[index] = ssm_state[index] * dA + x[i] * dB;

            val += ssm_state[index] * C[j];
        }

        val += D[i] * x[i];
        y[i] = val * silu(z[i]);
    }
}


#ifndef QUANTIZED
static void matmul(fp32_t* y, fp32_t* x, fp32_t* w, uint64_t d, uint64_t n) {
    #pragma omp parallel for
    for (uint64_t i = 0; i < d; ++i) {
        fp32_t val = 0.0f;

        for (uint64_t j = 0; j < n; ++j)
            val += w[i * n + j] * x[j];

        y[i] = val;
    }
}
#else
static void qmatmul(fp32_t* y, int8_t *xq, fp32_t *xs, int8_t *wq, fp32_t *ws, uint64_t d, uint64_t n) {
    #pragma omp parallel for
    for (uint64_t i = 0; i < d; i++) {

        float val = 0.0f;
        int64_t in = i * n;

        // do the matmul in groups of GS
        for (int j = 0; j <= n - GS; j += GS) {
            int32_t ival = 0;

            for (uint64_t k = 0; k < GS; k++)
                ival += ((uint64_t) xq[j + k]) * ((uint64_t) wq[in + j + k]);

            val += ((float) ival) * ws[(in + j) / GS] * xs[j / GS];
        }

        y[i] = val;
    }
}

void dequantize(fp32_t* x, int8_t *xq, fp32_t *xs, uint64_t n) {
    for (uint64_t i = 0; i < n; i++)
        x[i] = xq[i] * xs[i / GS];
}


void quantize(int8_t *xq, fp32_t *xs, fp32_t* x, uint64_t n) {
    uint64_t num_groups = n / GS;
    fp32_t Q_MAX = 127.0f;

    for (uint64_t group = 0; group < num_groups; group++) {

        // find the max absolute value in the current group
        fp32_t wmax = 0.0;
        for (uint64_t i = 0; i < GS; i++) {
            fp32_t val = fabs(x[group * GS + i]);
            if (val > wmax) {
                wmax = val;
            }
        }

        // calculate and write the scaling factor
        fp32_t scale = wmax / Q_MAX;
        xs[group] = scale;

        // calculate and write the quantized values
        for (uint64_t i = 0; i < GS; i++) {
            fp32_t quant_value = x[group * GS + i] / scale; // scale
            int8_t quantized = (int8_t) round(quant_value); // round and clamp
            xq[group * GS + i] = quantized;
        }
    }
}
#endif




#define LOG(...) fprintf(stderr, __VA_ARGS__)
#define CLOG(X, ...) if(X) LOG(__VA_ARGS__)

static char *shortScale(char *result, int64_t number) {
    char *suffixes[] = {"", "k", "m", "b"};
    uint64_t magnitude = 0;
    fp64_t num = (fp64_t)number;

    if (number < 1000) {
        sprintf(result, "%lld", number);
        return result;
    }

    while (number >= 1000 || number <= -1000) {
        magnitude++;
        number /= 1000;
        num /= 1000.0;
    }

    sprintf(result, "%.0lf%s", num, suffixes[magnitude]);
    return result;
}



#ifndef QUANTIZED
static inline void nparams(uint64_t dim) {
	char buff[16];
	shortScale(buff, dim);
	
	LOG("%12llu (%s)", dim, buff);
}
#define NPARAMS(X) nparams(sizeof(X) / sizeof(fp32_t))

static void MambaLog(Mamba *mamba) {
    MambaConfig     *p = &mamba->config;
    MambaWeights    *w = mamba->weights;
    MambaState      *s = &mamba->state;

	LOG("Mamba Config:");
    LOG("\n\tvocab_size:     %12llu",	p->vocab_size);
    LOG("\n\tn_layer:        %12llu",	p->n_layer);
    LOG("\n\td_model:        %12llu",	p->d_model);
    LOG("\n\td_inner:        %12llu",	p->d_inner);
    LOG("\n\tdt_rank:        %12llu",	p->dt_rank);
    LOG("\n\td_state:        %12llu",	p->d_state);
    LOG("\n\td_conv:         %12llu",	p->d_conv);
    printf("\n\n\n");

    LOG("Parameters Count:");
    LOG("\n\tembed:          ");		NPARAMS(w->embed);
    LOG("\n\tin_proj:        ");		NPARAMS(w->in_proj);
    LOG("\n\tconv1d_weight:  ");		NPARAMS(w->conv1d_weight);
    LOG("\n\tconv1d_bias:    ");		NPARAMS(w->conv1d_bias);
    LOG("\n\tx_proj:         ");		NPARAMS(w->x_proj);
    LOG("\n\tdt_proj_weight: ");		NPARAMS(w->dt_proj_weight);
    LOG("\n\tdt_proj_bias:   ");		NPARAMS(w->dt_proj_bias);
    LOG("\n\tA:              ");		NPARAMS(w->A);
    LOG("\n\tD:              ");		NPARAMS(w->D);
    LOG("\n\tout_proj:       ");		NPARAMS(w->out_proj);
    LOG("\n\tnorm:           ");		NPARAMS(w->norm);
    LOG("\n\tnorm_f:         ");		NPARAMS(w->norm_f);
    LOG("\n\tlm_head:        ");		NPARAMS(w->lm_head);
    LOG("\n\n\tTotal:          ");		NPARAMS(MambaWeights); //NTotal();
	printf("\n\n\n");

	LOG("Recurrent State:");
	LOG("\n\thidden_state:   ");		NPARAMS(s->hidden_state);
	LOG("\n\tconv_state:     ");		NPARAMS(s->conv_state);
	LOG("\n\tssm_state:      ");		NPARAMS(s->ssm_state);
    LOG("\n\n\tTotal:          ");		NPARAMS(MambaState);
	printf("\n\n\n");
}

static fp32_t *MambaForwardLayer(Mamba *mamba, uint64_t layer) {
    MambaConfig     *p = &mamba->config;
    MambaWeights    *w = mamba->weights;
    MambaState      *s = &mamba->state;

    uint64_t        d_model = p->d_model,
                    d_inner = p->d_inner,
                    d_conv  = p->d_conv,
                    d_state = p->d_state,
                    dt_rank = p->dt_rank;

    fp32_t          *hidden_state = s->hidden_state,
                    *conv_state   = s->conv_state + layer * d_inner * d_conv,
					*ssm_state    = s->ssm_state  + layer * d_inner * d_state;

	Tensor(			xz,   2 * D_INNER,             1, 1);
    Tensor(			x_db, DT_RANK + 2 * D_STATE,   1, 1);
    Tensor(			dt,   D_INNER,                 1, 1);
    Tensor(			y,    D_INNER,                 1, 1);
 
    fp32_t          *x = xz,
                    *z = xz	  + d_inner,
                    *B = x_db + dt_rank,
                    *C = x_db + dt_rank + d_state;


    // Proj input
    matmul(xz, hidden_state, w->in_proj + layer * 2 * d_inner * d_model, 2 * d_inner, d_model);


    // Conv
    shift_and_update_last_column(conv_state, x, d_inner, d_conv);
    conv1d_silu(x, conv_state, w->conv1d_weight + layer * d_inner * d_conv, w->conv1d_bias + layer * d_inner, d_inner, d_conv);


    // SSM
    matmul(x_db, x, w->x_proj + layer * (dt_rank + 2 * d_state) * d_inner, dt_rank + 2 * d_state, d_inner);
    dense_softplus(dt, x_db, w->dt_proj_weight + layer * d_inner * dt_rank, w->dt_proj_bias + layer * d_inner, d_inner, dt_rank); 
    selective_scan(y, ssm_state, dt, w->A + layer * d_inner * d_state, B, C, w->D + layer * d_inner, x, z, d_inner, d_state);


    // Proj output
    matmul(hidden_state, y, w->out_proj + layer * d_model * d_inner, d_model, d_inner);
	
	return hidden_state;
}

static fp32_t *MambaForward(Mamba *mamba, uint64_t token) {
    MambaConfig     *p = &mamba->config;
    MambaWeights    *w = mamba->weights;
	MambaState      *s = &mamba->state;

    uint64_t        d_model    = p->d_model,
                    n_layer    = p->n_layer,
                    vocab_size = p->vocab_size;

		   Tensor(  input,  D_MODEL,    1, 1);
	static Tensor(  logits, ROUNDED_VOCAB_SIZE, 1, 1);

	fp32_t          *hidden_state = s->hidden_state,
					*content_row  = w->embed + token * d_model;	


	memcpy(input, content_row, d_model * sizeof(fp32_t));

    for (uint64_t layer = 0; layer < n_layer; ++layer) {
        rmsnorm(hidden_state, input, w->norm + layer * d_model, d_model);
        mamba->forward_layer(mamba, layer);

        for (uint64_t i = 0; i < d_model; ++i) {
            hidden_state[i] += input[i];
            input[i] = hidden_state[i];
        }
    }

    rmsnorm(hidden_state, hidden_state, w->norm_f, d_model);
    matmul(logits, hidden_state, w->lm_head, vocab_size, d_model);

    return logits;
}
#else
uint64_t global_scale = 0;
static inline void nparams(uint64_t dim) {
	char buff[16];
	shortScale(buff, dim);
	
	global_scale += dim;

	LOG("%12llu (%s)", dim, buff);
}
#define NPARAMS(X) nparams(sizeof(X) / sizeof(fp32_t))

static inline void nqparams(uint64_t qdim) {
	uint64_t dim = qdim * 4;
	
	char buff[16]; 
	shortScale(buff, qdim);

	nparams(dim);
	LOG("\t%12llu (%s)", qdim, buff);
}
#define NQPARAMS(X) nqparams(sizeof(X.q) / sizeof(fp32_t))

static inline void ntotal(uint64_t qdim) {
	char buff[16];
	shortScale(buff, global_scale);
	LOG("%12llu (%s)", global_scale, buff);

	shortScale(buff, qdim);	
	LOG("\t%12llu (%s)", qdim, buff);

	fp32_t factor = (((fp32_t) (global_scale - qdim)) / global_scale) * 100;
	LOG("\t%12.2lf%%", factor);
}
#define NTOTAL(X) ntotal(sizeof(X) / sizeof(fp32_t))

static void MambaLog(Mamba *mamba) {
    MambaConfig     *p = &mamba->config;
    MambaWeights    *w = mamba->weights;
    MambaState      *s = &mamba->state;

	LOG("Mamba Config:");
    LOG("\n\tvocab_size:     %12llu",	p->vocab_size);
    LOG("\n\tn_layer:        %12llu",	p->n_layer);
    LOG("\n\td_model:        %12llu",	p->d_model);
    LOG("\n\td_inner:        %12llu",	p->d_inner);
    LOG("\n\tdt_rank:        %12llu",	p->dt_rank);
    LOG("\n\td_state:        %12llu",	p->d_state);
    LOG("\n\td_conv:         %12llu",	p->d_conv);
    printf("\n\n\n");

    LOG("Parameters Count:               Base               Quantized                   Factor");
    LOG("\n\tembed:          ");		NQPARAMS(w->embed);
    LOG("\n\tin_proj:        ");		NQPARAMS(w->in_proj);
    LOG("\n\tconv1d_weight:  ");		NPARAMS(w->conv1d_weight);
    LOG("\n\tconv1d_bias:    ");		NPARAMS(w->conv1d_bias);
    LOG("\n\tx_proj:         ");		NQPARAMS(w->x_proj);
    LOG("\n\tdt_proj_weight: ");		NPARAMS(w->dt_proj_weight);
    LOG("\n\tdt_proj_bias:   ");		NPARAMS(w->dt_proj_bias);
    LOG("\n\tA:              ");		NPARAMS(w->A);
    LOG("\n\tD:              ");		NPARAMS(w->D);
    LOG("\n\tout_proj:       ");		NQPARAMS(w->out_proj);
    LOG("\n\tnorm:           ");		NPARAMS(w->norm);
    LOG("\n\tnorm_f:         ");		NPARAMS(w->norm_f);
    LOG("\n\tlm_head:        ");		NQPARAMS(w->lm_head);
    LOG("\n\n\tTotal:          ");		NTOTAL(MambaWeights);
	printf("\n\n\n");

	LOG("Recurrent State:");
	LOG("\n\thidden_state:   ");		NPARAMS(s->hidden_state);
	LOG("\n\tconv_state:     ");		NPARAMS(s->conv_state);
	LOG("\n\tssm_state:      ");		NPARAMS(s->ssm_state);
    LOG("\n\n\tTotal:          ");		NPARAMS(MambaState);
	printf("\n\n\n");
}


static fp32_t *MambaForwardLayer(Mamba *mamba, uint64_t layer) {
    MambaConfig     *p = &mamba->config;
    MambaWeights    *w = mamba->weights;
    MambaState      *s = &mamba->state;

    uint64_t        d_model = p->d_model,
                    d_inner = p->d_inner,
                    d_conv  = p->d_conv,
                    d_state = p->d_state,
                    dt_rank = p->dt_rank;

    fp32_t          *hidden_state = s->hidden_state,
                    *conv_state   = s->conv_state + layer * d_inner * d_conv,
					*ssm_state    = s->ssm_state  + layer * d_inner * d_state;

	QTensor(		qhidden_state,	D_MODEL,		1,							1);


	Tensor(			xz,    2 * D_INNER,             1, 1);
	QTensor(		qx,    D_INNER,                 1, 1);
	Tensor(			x_db,  DT_RANK + 2 * D_STATE,   1, 1);
    Tensor(			dt,    D_INNER,                 1, 1);
    Tensor(			y,     D_INNER,                 1, 1);
    QTensor(		qy,    D_INNER,                 1, 1);

    fp32_t          *x = xz,
                    *z = xz	  + d_inner,
                    *B = x_db + dt_rank,
                    *C = x_db + dt_rank + d_state;

	QPointer(		in_proj,         w->in_proj,         layer * 2 * d_inner             * d_model);
	QPointer(		x_proj,	         w->x_proj,          layer * (dt_rank + 2 * d_state) * d_inner);
	QPointer(		out_proj,        w->out_proj,        layer * d_model                 * d_inner);



    // Proj input
    quantize(qhidden_state.q, qhidden_state.s, hidden_state, d_model);
	qmatmul(xz, qhidden_state.q, qhidden_state.s, in_proj.q, in_proj.s, 2 * d_inner, d_model);


    // Conv
    shift_and_update_last_column(conv_state, x, d_inner, d_conv);
    conv1d_silu(x, conv_state, w->conv1d_weight + layer * d_inner * d_conv, w->conv1d_bias + layer * d_inner, d_inner, d_conv);


    // SSM
	quantize(qx.q, qx.s, x, d_inner);
	qmatmul(x_db, qx.q, qx.s, x_proj.q, x_proj.s, dt_rank + 2 * d_state, d_inner);
    
	dense_softplus(dt, x_db, w->dt_proj_weight + layer * d_inner * dt_rank, w->dt_proj_bias + layer * d_inner, d_inner, dt_rank);
	selective_scan(y, ssm_state, dt, w->A + layer * d_inner * d_state, B, C, w->D + layer * d_inner, x, z, d_inner, d_state);


    // Proj output
	quantize(qy.q, qy.s, y, d_inner);
	qmatmul(hidden_state, qy.q, qy.s, out_proj.q, out_proj.s, d_model, d_inner);

	return hidden_state;
}


static fp32_t *MambaForward(Mamba *mamba, uint64_t token) {
    MambaConfig     *p = &mamba->config;
    MambaWeights    *w = mamba->weights;
	MambaState      *s = &mamba->state;

    uint64_t        d_model    = p->d_model,
                    n_layer    = p->n_layer,
                    vocab_size = p->vocab_size;

		   Tensor(  input,  			D_MODEL,    1, 1);
	static Tensor(  logits, 			ROUNDED_VOCAB_SIZE, 1, 1);

	QTensor(		qhidden_state, 		D_MODEL, 	1, 1);	

	fp32_t          *hidden_state = s->hidden_state;
	
	QPointer(		row, w->embed, token * d_model);


	dequantize(input, row.q, row.s, d_model);

    for (uint64_t layer = 0; layer < n_layer; ++layer) {
        rmsnorm(hidden_state, input, w->norm + layer * d_model, d_model);
        mamba->forward_layer(mamba, layer);

        for (uint64_t i = 0; i < d_model; ++i) {
            hidden_state[i] += input[i];
            input[i] = hidden_state[i];
        }
    }

    rmsnorm(hidden_state, hidden_state, w->norm_f, d_model);
    
	quantize(qhidden_state.q, qhidden_state.s, hidden_state, d_model);
	qmatmul(logits, qhidden_state.q, qhidden_state.s, w->lm_head.q, w->lm_head.s, vocab_size, d_model);

    return logits;
}
#endif


/*
#define PRINT(C) fputc((char)C, stdout), fflush(stdout)

typedef enum {false, true} bool;

static uint64_t rng_seed = 0;
static uint64_t random_u32() { 
    rng_seed ^= rng_seed >> 12;
    rng_seed ^= rng_seed << 25;
    rng_seed ^= rng_seed >> 27;
    return (rng_seed * 0x2545F4914F6CDD1Dull) >> 32;
}
static inline fp32_t random_f32() { return (random_u32() >> 8) / 16777216.0f; }

static uint64_t time_in_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

static uint64_t sample_argmax(fp32_t* probabilities, uint64_t n) {
    uint64_t max_i = 0;
    fp32_t max_p = probabilities[0];

    for (uint64_t i = 1; i < n; ++i)
        if (probabilities[i] > max_p)
            max_i = i, max_p = probabilities[i];

    return max_i;
}

static uint64_t sample_mult(fp32_t* probabilities, uint64_t n, fp32_t coin) {
    fp32_t cdf = 0.0f;

    for (uint64_t i = 0; i < n; ++i) {
        cdf += probabilities[i];

        if (coin < cdf) return i;
    }
    
    return n - 1;
}

static uint64_t sample(fp32_t* logits, fp32_t temperature, uint64_t vocab_size) {
    uint64_t next;
    
    if (temperature == 0.0f) next = sample_argmax(logits, vocab_size);
    else {
        for (uint64_t q = 0; q < vocab_size; ++q) 
            logits[q] /= temperature;

        softmax(logits, vocab_size);

        fp32_t coin = random_f32();
        next = sample_mult(logits, vocab_size, coin);
    }

    return next;
}

static bool generate(Mamba *mamba, char *prompt, uint64_t n_predict, fp32_t temperature, bool verbose) {
	uint64_t tokens[n_predict],
             len,
             time_start = time_in_ms();

    if (prompt == NULL || (len = strlen(prompt)) == 0) return EXIT_FAILURE;

    for (uint64_t i = 0; i < len; ++i) {
        tokens[i] = (uint64_t)prompt[i];
        (void) mamba->forward(mamba, tokens[i]);
    
        PRINT(prompt[i]);
    }

    for (; len < n_predict; ++len) {
        fp32_t *logits = mamba->forward(mamba, tokens[len - 1]);
        uint64_t next = sample(logits, temperature, mamba->config.vocab_size);
        tokens[len] = next;

        PRINT((char)next);
    }

    
	CLOG(verbose, "\nachieved tok/s: %f\n", n_predict / (double)(time_in_ms() - time_start) * 1000);
    
	return EXIT_SUCCESS;
}*/
