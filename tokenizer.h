#pragma once




extern char _embedded_binary_tokenizer[];

#define MAX_WORD_LEN 24

typedef struct {
	struct __attribute__((packed)) {
		uint8_t  byte;
		uint16_t prev;
	} token[VOCAB_SIZE];
} vocab_t;

typedef struct Tokenizer Tokenizer;
struct Tokenizer {
	vocab_t *vocab;
	uint16_t vocab_size;

	uint16_t  (*find)   (Tokenizer *, uint8_t, uint16_t);
	uint16_t  (*encode) (Tokenizer *, uint8_t **);
	uint8_t  *(*decode) (Tokenizer *, uint16_t);


};


static uint16_t TokenizerFind(Tokenizer *tokenizer, uint8_t byte, uint16_t prev) {
	for (uint16_t i = prev; i < tokenizer->vocab_size; ++i)
		if (tokenizer->vocab->token[i].byte == byte && tokenizer->vocab->token[i].prev == prev)
			return i;

	return 0;
}

static uint16_t TokenizerEncode(Tokenizer *tokenizer, uint8_t **seed_text) {	

	uint16_t prev = 0;
	for (; **seed_text; ++*seed_text) {
		uint16_t next = tokenizer->find(tokenizer, **seed_text, prev);
		if (next == 0) break;	
		prev = next;
	}

	return prev;
}

static uint8_t *TokenizerDecode(Tokenizer *tokenizer, uint16_t token) {

	static uint8_t dest[MAX_WORD_LEN + 1];
	dest[MAX_WORD_LEN] = '\0';

	uint16_t prev = token;
	uint16_t i = MAX_WORD_LEN - 1;

	for (; prev && i > 0; prev = tokenizer->vocab->token[prev].prev, --i)
		dest[i] = tokenizer->vocab->token[prev].byte;

	return dest + i + 1;
}


Tokenizer tokenizer = {
	.vocab 	 	= (vocab_t *) _embedded_binary_tokenizer,
	.vocab_size = VOCAB_SIZE,

	.find   = TokenizerFind,
	.encode = TokenizerEncode,
	.decode = TokenizerDecode
};

