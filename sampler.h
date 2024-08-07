#pragma once


//#include <stdbool.h>







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
	
	fp32_t	temperature = sampler->temperature;

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

		token = tokenizer->encode(tokenizer, (uint8_t **) &seed_text);	
		text = tokenizer->decode(tokenizer, token);	

		fputs(text, stdout);
		fflush(stdout);

		logits = model->forward(model, token);
	}

	uint64_t time_start;
	if (verbose) time_start = time_in_ms();

	for (uint64_t i = 0; i < n_predict; ++i) { 
		
		token = sampler->sample(sampler, logits);
		text = tokenizer->decode(tokenizer, token); 

		fputs(text, stdout);
		fflush(stdout);

		logits = model->forward(model, token);
	}
	
	CLOG(verbose, "\nachieved tok/s: %f\n", n_predict / (double)(time_in_ms() - time_start) * 1000);
	
	return EXIT_SUCCESS;
}



Sampler sampler = {
	.model = &mamba,
	.tokenizer = &tokenizer,

	.rng_seed = 42,
	.temperature = 0.0f,
	.verbose = false,

	.generate = SamplerGenerate,
	.sample = SamplerSample
};
