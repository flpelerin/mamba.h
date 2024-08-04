CC		:= gcc
CFLAGS		:= -std=c99 -O3 -Ofast -mtune=native -march=native -fopenmp -static
CLIBS		:= -lm -lc


C_TOKENIZER := tokenizer.bin
C_MODEL 	:= model.bin
C_CONFIG 	:= config.h

SRC		:= $(wildcard *.c)
TARGET		:= mamba





all:	$(TARGET)

clean:
		$(RM) $(TARGET) *.o

wipe:	
		make clean
		$(RM) $(C_TOKENIZER) $(C_MODEL) $(C_CONFIG)

run:	$(TARGET)
	OMP_NUM_THREADS=4 ./$< -t 0 -n 512 -p "One day, a little girl nam" -v





$(C_TOKENIZER):
	awk 'BEGIN {for (i = 0; i <= 255; i++) printf("%c%c%c", i, 0, 0)}' > $@





model.o:	$(C_MODEL)
	objcopy --input-target binary \
		--output-target elf64-littleaarch64 \
		--redefine-sym _binary_model_bin_start=_embedded_binary_model \
		$< $@




tokenizer.o:	$(C_TOKENIZER)
	objcopy --input-target binary \
		--output-target elf64-littleaarch64 \
		--redefine-sym _binary_tokenizer_bin_start=_embedded_binary_tokenizer \
		$< $@






$(TARGET): $(SRC) model.o tokenizer.o
	$(CC) $(CFLAGS) -o $@ $^ $(CLIBS)



.PHONY: all wipe clean run
