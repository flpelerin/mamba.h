
#include <stdlib.h>

#include <time.h>
#include <sys/time.h>
#include <stdlib.h>



#define QUANTIZED

#include "mamba.h"




#define MAX_WORD_LEN 16

typedef short		   int16_t;
typedef short unsigned uint16_t;

extern char _embedded_binary_tokenizer[];


typedef struct {
	struct __attribute__((packed)) {
		uint8_t byte;
		uint16_t prev;
	} token[VOCAB_SIZE];
} vocab_t;


vocab_t *vocab = (vocab_t *) _embedded_binary_tokenizer;

static inline uint16_t find(uint8_t byte, uint16_t prev) {
	for (uint16_t i = prev; i < VOCAB_SIZE; ++i)
		if (vocab->token[i].byte == byte && vocab->token[i].prev == prev)
			return i;

	return 0;
}

static uint16_t encode(uint8_t **str) {

	uint16_t prev = 0;
	for (; **str; ++*str) {
		uint16_t next = find(**str, prev);
		if (next == 0) break;	
		prev = next;
	}

	return prev;
}


static int8_t *decode(uint16_t token) {
        static int8_t dest[MAX_WORD_LEN + 1];
        dest[MAX_WORD_LEN] = '\0';

        uint16_t prev = token;
        uint16_t i = MAX_WORD_LEN - 1;

        for (; prev && i > 0; prev = vocab->token[prev].prev, --i)
                dest[i] = vocab->token[prev].byte;


        return dest + i + 1;
}












typedef struct Tokenizer Tokenizer;
struct Tokenizer {
	uint64_t vocab_size;

	uint64_t  (*encode) (Tokenizer *, char **);
	char     *(*decode) (Tokenizer *, uint64_t);


};


uint64_t TokenizerEncode(Tokenizer *tokenizer, char **seed_text) {	
	//uint64_t token = **seed_text;
	//*seed_text += 1;
	uint64_t token = encode(seed_text);
	return token;
}

char *TokenizerDecode(Tokenizer *tokenizer, uint64_t token) {
	/*static char dest[16];
	
	dest[0] = token;
	dest[1] = '\0';
	*/

	char *dest = decode(token);
	return dest;
}








#define PRINT(C) fputc((char)C, stdout), fflush(stdout)

typedef enum {false, true} bool;

typedef struct Sampler Sampler;
struct Sampler {
	Mamba *model;
	Tokenizer *tokenizer;

	uint64_t rng_seed;
	fp32_t temperature;
	bool verbose;

	bool (*generate) (Sampler *, char *, uint64_t);
	uint64_t (*sample) (Sampler *, fp32_t *); 
};




static uint64_t random_u32(uint64_t *rng_seed) { 
    *rng_seed ^= *rng_seed >> 12;
    *rng_seed ^= *rng_seed << 25;
    *rng_seed ^= *rng_seed >> 27;
	*rng_seed = (*rng_seed * 0x2545F4914F6CDD1Dull) >> 32;
    return *rng_seed;
}
static inline fp32_t random_f32(uint64_t *rng_seed) { return (random_u32(rng_seed) >> 8) / 16777216.0f; }

static uint64_t time_in_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

static inline uint64_t sample_argmax(fp32_t* probabilities, uint64_t n) {
    uint64_t max_i = 0;
    fp32_t max_p = probabilities[0];

    for (uint64_t i = 1; i < n; ++i)
        if (probabilities[i] > max_p)
            max_i = i, max_p = probabilities[i];

    return max_i;
}

static inline uint64_t sample_mult(fp32_t* probabilities, uint64_t n, fp32_t coin) {
    fp32_t cdf = 0.0f;

    for (uint64_t i = 0; i < n; ++i) {
        cdf += probabilities[i];

        if (coin < cdf) return i;
    }
    
    return n - 1;
}

static uint64_t SamplerSample(Sampler *sampler, fp32_t* logits) {
    uint64_t  next,
			  vocab_size = sampler->tokenizer->vocab_size,
			 *rng_seed = &sampler->rng_seed;
	
	fp32_t    temperature = sampler->temperature;

	if (temperature == 0.0f) next = sample_argmax(logits, vocab_size);
    else {
        for (uint64_t q = 0; q < vocab_size; ++q) 
            logits[q] /= temperature;

        softmax(logits, vocab_size);

        fp32_t coin = random_f32(rng_seed);
        next = sample_mult(logits, vocab_size, coin);
    }

    return next;
}

static bool SamplerGenerate(Sampler *sampler, char *seed_text, uint64_t n_predict) {
	Mamba *model = sampler->model;
	Tokenizer *tokenizer = sampler->tokenizer;
	uint64_t vocab_size = tokenizer->vocab_size;
	fp32_t temperature = sampler->temperature;
	bool verbose = sampler->verbose;

	uint64_t token; 
	fp32_t *logits;
	char *text;

    if (seed_text == NULL) return EXIT_FAILURE;

    for (; *seed_text; ) { 

		token = tokenizer->encode(tokenizer, &seed_text);	
		text = tokenizer->decode(tokenizer, token);	

		fputs(text, stdout);
		fflush(stdout);

		logits = model->forward(model, token);
    }

	uint64_t time_start;
	if (verbose) time_start = time_in_ms();

    for (int i = 0; i < n_predict; ++i) { 
		
        token = sampler->sample(sampler, logits);
		text = tokenizer->decode(tokenizer, token); 

		fputs(text, stdout);
		fflush(stdout);

		logits = model->forward(model, token);
    }
	
	CLOG(verbose, "\nachieved tok/s: %f\n", n_predict / (double)(time_in_ms() - time_start) * 1000);
    
	return EXIT_SUCCESS;
}




Tokenizer tokenizer = {
	.vocab_size = VOCAB_SIZE,

	.encode = TokenizerEncode,
	.decode = TokenizerDecode
};



Sampler sampler = {
	.model = &mamba,
	.tokenizer = &tokenizer,

	.rng_seed = 0,
	.temperature = 0.0f,
	.verbose = false,

	.generate = SamplerGenerate,
	.sample = SamplerSample
};




static void help(char *name, void *defaults[]) {
    LOG("Usage: %s [-pntsvh]\n\n", name);
    LOG("Infers a Mamba language model.\n\n");
    LOG("Options:\n");
    LOG("\t-p <seed_text>          The seed_text to start the generation with.    (default NONE)\n");
    LOG("\t-n <n_predict>       The number of tokens to predict.            (default %llu)\n", *(uint64_t *)defaults[0]);
    LOG("\t-t <temperature>     The temperature of the softmax.             (default %.1f)\n", *(fp32_t *)defaults[1]);
    LOG("\t-s <seed>            The seed for the random number generator.   (default %llu)\n", *(uint64_t *)defaults[2]);
    LOG("\t-v                   Enables verbose mode.                       (default %s)\n", *(bool *)defaults[3] ? "true" : "false");
    LOG("\t-h                   Prints this help message.\n\n");

    exit(EXIT_FAILURE);
}



int main(int argc, char *argv[]) {

	char *seed_text = NULL;
    uint64_t n_predict = 256;


    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            LOG("Invalid argument: %s\n", argv[i]);
            return 1;
        }

        switch (argv[i][1]) {
            case 'p':
                seed_text = argv[++i];
                break;
            case 'n':
                n_predict = strtoull(argv[++i], NULL, 10);
                break;
            case 't':
                sampler.temperature = strtod(argv[++i], NULL);
                break;
            case 's':
                sampler.rng_seed = strtoull(argv[++i], NULL, 10);
                break;
            case 'v':
                sampler.verbose = true;
                break;
            case 'h':
                goto help_;
                break;
            default:
                LOG("Invalid argument: %s\n", argv[i]);
                return 1;
        }
    }

    if (sampler.verbose)
        mamba.log(&mamba);

	if (sampler.generate(&sampler, seed_text, n_predict) == EXIT_FAILURE)
        goto help_;

    return 0;

    help_:
        help(argv[0], (void *[]) {&n_predict, &sampler.temperature, &sampler.rng_seed, &sampler.verbose});        
}
