
#include <stdlib.h>

#include <time.h>
#include <sys/time.h>
#include <stdlib.h>



#include "qmamba.h"
#include "tokenizer.h"
#include "sampler.h"











static void help(char *name, void *defaults[]) {
	LOG("Usage: %s [-pntsvh]\n\n", name);
	LOG("Infers a Mamba language model.\n\n");
	LOG("Options:\n");
	LOG("\t-p <seed_text>		  The seed_text to start the generation with.	(default NONE)\n");
	LOG("\t-n <n_predict>	   The number of tokens to predict.			(default %lu)\n", *(uint64_t *)defaults[0]);
	LOG("\t-t <temperature>	 The temperature of the softmax.			 (default %.1f)\n", *(fp32_t *)defaults[1]);
	LOG("\t-s <seed>			The seed for the random number generator.   (default %lu)\n", *(uint64_t *)defaults[2]);
	LOG("\t-v				   Enables verbose mode.					   (default %s)\n", *(bool *)defaults[3] ? "true" : "false");
	LOG("\t-h				   Prints this help message.\n\n");

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
