#pragma once


#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "config.h"






extern char _embedded_binary_model[];

typedef float fp32_t;

typedef struct MambaWeights MambaWeights;
typedef struct MambaConfig  MambaConfig;
typedef struct MambaState   MambaState;
typedef struct Mamba        Mamba;

static void    MambaLog(Mamba *);
static fp32_t *MambaForwardLayer(Mamba *, uint64_t);
static fp32_t *MambaForward(Mamba *, uint64_t);

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


static void matmul(fp32_t* y, fp32_t* x, fp32_t* w, uint64_t d, uint64_t n) {
    #pragma omp parallel for
    for (uint64_t i = 0; i < d; ++i) {
        fp32_t val = 0.0f;

        for (uint64_t j = 0; j < n; ++j)
            val += w[i * n + j] * x[j];

        y[i] = val;
    }
}



#define LOG(...) fprintf(stderr, __VA_ARGS__)
#define CLOG(X, ...) if(X) LOG(__VA_ARGS__)

static char *shortScale(char *result, int64_t number) {
    char *suffixes[] = {"", "k", "m", "b"};
    uint64_t magnitude = 0;
    fp32_t num = (fp32_t)number;

    if (number < 1000) {
        sprintf(result, "%lu", number);
        return result;
    }

    while (number >= 1000 || number <= -1000) {
        magnitude++;
        number /= 1000;
        num /= 1000.0;
    }

    sprintf(result, "%.0f%s", num, suffixes[magnitude]);
    return result;
}



static inline void nparams(uint64_t dim) {
	char buff[16];
	shortScale(buff, dim);
	
	LOG("%12lu (%s)", dim, buff);
}
#define NPARAMS(X) nparams(sizeof(X) / sizeof(fp32_t))

static void MambaLog(Mamba *mamba) {
    MambaConfig     *p = &mamba->config;
    MambaWeights    *w = mamba->weights;
    MambaState      *s = &mamba->state;

	LOG("Mamba Config:");
    LOG("\n\tvocab_size:     %12lu",	p->vocab_size);
    LOG("\n\tn_layer:        %12lu",	p->n_layer);
    LOG("\n\td_model:        %12lu",	p->d_model);
    LOG("\n\td_inner:        %12lu",	p->d_inner);
    LOG("\n\tdt_rank:        %12lu",	p->dt_rank);
    LOG("\n\td_state:        %12lu",	p->d_state);
    LOG("\n\td_conv:         %12lu",	p->d_conv);
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

