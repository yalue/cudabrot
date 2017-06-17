.PHONY: all clean

CFLAGS := -Wall -Werror -O3 -g -fPIC

NVCCFLAGS := -g --ptxas-options=-v --compiler-options="$(CFLAGS)" \
	--cudart=shared -arch=compute_30 \
	--generate-code arch=compute_50,code=[compute_50,sm_50] \
	--generate-code arch=compute_62,code=[compute_62,sm_62]

all: cudabrot

cudabrot: cudabrot.cu
	nvcc $(NVCCFLAGS) -o cudabrot cudabrot.cu -lm

clean:
	rm -f cudabrot
