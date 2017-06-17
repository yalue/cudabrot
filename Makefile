.PHONY: all clean

CFLAGS := -Wall -Werror -O3 -g -fPIC

NVCCFLAGS := -g --ptxas-options=-v --compiler-options="$(CFLAGS)" \
	--cudart=shared -arch=compute_30 \
	--generate-code arch=compute_50,code=[compute_50,sm_50] \
	--generate-code arch=compute_62,code=[compute_62,sm_62]

all: cudabrot

bitmap_file.o: bitmap_file.c bitmap_file.h
	gcc $(CFLAGS) -c -o bitmap_file.o bitmap_file.c

cudabrot: cudabrot.cu bitmap_file.o
	nvcc $(NVCCFLAGS) -o cudabrot cudabrot.cu bitmap_file.o -lm

clean:
	rm -f cudabrot
