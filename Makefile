.PHONY: all clean

all: cudabrot

cudabrot: cudabrot.cu
	nvcc -arch=sm_50 $(shell sdl2-config --cflags) $(shell sdl2-config --libs)\
		cudabrot.cu -o cudabrot

clean:
	rm -f cudabrot
