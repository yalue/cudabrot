#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
extern "C" {
#include <SDL2/SDL.h>
}

#define LOG2 (0.30102999566)

// The number of CUDA threads to use per block.
#define DEFAULT_BLOCK_SIZE (128)

// This macro takes a cudaError_t value and exits the program if it isn't equal
// to cudaSuccess. (Calls the ErrorCheck function, defined later).
#define CheckCUDAError(val) (InternalCUDAErrorCheck((val), #val, __FILE__, __LINE__))

// Holds globals in a single namespace.
static struct {
  SDL_Window *window;
  SDL_Renderer *renderer;
  SDL_Texture *image;
  // The maximum number of iterations to run each point.
  uint32_t max_iterations;
  // The width and height, in pixels, of the output image.
  int w;
  int h;
  // The boundaries of the fractal.
  double min_real;
  double min_imag;
  double max_real;
  double max_imag;
  // The distance between one pixel in the real and imaginary axes.
  double delta_real;
  double delta_imag;
  // Pointer to the device memory that will be iterated over.
  float *device_data;
  // The host-side copy of the data, that will receive the copy of device_data
  // after calculations have completed.
  float *host_data;
} g;

// If any globals have been initialized, this will free them. (Relies on
// globals being set to 0 at the start of the program)
static void CleanupGlobals(void) {
  if (g.renderer) SDL_DestroyRenderer(g.renderer);
  if (g.image) SDL_DestroyTexture(g.image);
  if (g.window) SDL_DestroyWindow(g.window);
  if (g.device_data) cudaFree(g.device_data);
  if (g.host_data) free(g.host_data);
  memset(&g, 0, sizeof(g));
}

// Prints an error message and exits the program if the cudaError_t value is
// not equal to cudaSuccess. Generally, this will be called via the
// CheckCudaError macro.
static void InternalCUDAErrorCheck(cudaError_t result, const char *fn,
    const char *file, int line) {
  if (result == cudaSuccess) return;
  printf("CUDA error %d in %s, line %d (%s)\n", (int) result, file, line, fn);
  exit(1);
  CleanupGlobals();
}

// Sets up the SDL window and resources. Must be called after g.w and g.h have
// been set.
static void SetupSDL(void) {
  if (SDL_Init(SDL_INIT_EVERYTHING) < 0) {
    printf("SDL error %s\n", SDL_GetError());
    CleanupGlobals();
    exit(1);
  }
  g.window = SDL_CreateWindow("Rendered image", SDL_WINDOWPOS_UNDEFINED,
    SDL_WINDOWPOS_UNDEFINED, g.w, g.h, SDL_WINDOW_SHOWN |
    SDL_WINDOW_RESIZABLE);
  if (!g.window) {
    printf("Error creating SDL window: %s\n", SDL_GetError());
    CleanupGlobals();
    exit(1);
  }
  g.renderer = SDL_CreateRenderer(g.window, -1, SDL_RENDERER_ACCELERATED);
  if (!g.renderer) {
    printf("Error creating SDL renderer: %s\n", SDL_GetError());
    CleanupGlobals();
    exit(1);
  }
  g.image = SDL_CreateTexture(g.renderer, SDL_PIXELFORMAT_RGBA8888,
    SDL_TEXTUREACCESS_STREAMING, g.w, g.h);
  if (!g.image) {
    printf("Failed creating SDL texture: %s\n", SDL_GetError());
    exit(1);
  }
}

// Allocates CUDA memory and calculates block/grid sizes. Must be called after
// g.w and g.h have been set.
static void SetupCUDA(void) {
  size_t buffer_size = g.w * g.h * sizeof(float);
  CheckCUDAError(cudaMalloc(&(g.device_data), buffer_size));
  CheckCUDAError(cudaMemset(g.device_data, 0, buffer_size));
  g.host_data = (float *) malloc(buffer_size);
  if (!g.host_data) {
    printf("Failed allocating host buffer.\n");
    CleanupGlobals();
    exit(1);
  }
}

__global__ void FractalKernel(float *data, int iterations, int w, int h,
    double min_real, double min_imag, double delta_real, double delta_imag) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int row = index / w;
  int col = index % w;
  // May desynchronize on the last block only.
  if (row >= h) return;
  double start_real = min_real + delta_real * row;
  double start_imag = min_imag + delta_imag * col;
  double current_real = start_real;
  double current_imag = start_imag;
  double tmp;
  uint32_t i;
  uint32_t escaped = 0;
  float color;
  float magnitude = (start_real * start_real) + (start_imag * start_imag);
  // Z = Z^2 + C, where C is the starting value of this point.
  for (i = 0; i < iterations; i++) {
    if (magnitude < 4) {
      tmp = (current_real * current_real) - (current_imag * current_imag);
      current_imag = 2 * current_imag * current_real + start_imag;
      current_real = tmp + start_real;
      color += expf(-magnitude);
      magnitude = (current_real * current_real) + (current_imag *
        current_imag);
    } else {
      // Smooth coloring from http://stackoverflow.com/questions/369438/
      // smooth-spectrum-for-mandelbrot-set-rendering
      //color = i - log2f(log2f(magnitude));
      color = log(magnitude) / 2;
      color = i - log(color / LOG2) / LOG2;
      data[index] = color;
      escaped = 1;
    }
  }
  if (!escaped) data[index] = 0;
}

// Renders the fractal image.
static void RenderImage(void) {
  int block_size, block_count;
  size_t data_size = g.w * g.h * sizeof(uint32_t);
  block_size = DEFAULT_BLOCK_SIZE;
  block_count = ((g.w * g.h) / block_size) + 1;
  FractalKernel<<<block_count, block_size>>>(g.device_data, g.max_iterations,
    g.w, g.h, g.min_real, g.min_imag, g.delta_real, g.delta_imag);
  CheckCUDAError(cudaGetLastError());
  CheckCUDAError(cudaMemcpy(g.host_data, g.device_data, data_size,
    cudaMemcpyDeviceToHost));
}

// Takes a floating-point value and converts it to an index into the color
// palette.
static uint8_t ColorIndex(float v) {
  return ((uint32_t) v) % 255;
}

// Copies data from the host-side data buffer to the texture drawn to the SDL
// window.
static void UpdateDisplayedImage(void) {
  int x, y;
  uint8_t *image_pixels;
  int image_pitch;
  int to_skip_per_row;
  uint8_t color_value;
  float *host_data = g.host_data;
  if (SDL_LockTexture(g.image, NULL, (void **) (&image_pixels), &image_pitch)
    < 0) {
    printf("Error locking SDL texture: %s\n", SDL_GetError());
    CleanupGlobals();
    exit(1);
  }
  // Abide by the image pitch, and skip unaffected bytes in each row.
  // (image_pitch should usually be equal to g.w * 4 anyway).
  to_skip_per_row = image_pitch - (g.w * 4);
  for (y = 0; y < g.h; y++) {
    for (x = 0; x < g.w; x++) {
      // The grey value scales linearly by the ratio of iterations to max
      // iterations.
      color_value = ColorIndex(*host_data);
      // The byte order is ABGR
      image_pixels[0] = 0xff;
      // Draw this in 255 shades of gray for now.
      image_pixels[1] = color_value;
      image_pixels[2] = color_value;
      image_pixels[3] = color_value;
      image_pixels += 4;
      host_data++;
    }
    image_pixels += to_skip_per_row;
  }
  SDL_UnlockTexture(g.image);
}

// Runs the main loop to display the SDL window. This will return when SDL
// detects an exit event.
static void SDLWindowLoop(void) {
  SDL_Event event;
  int quit = 0;
  // Update the display once every 30 ms (not really necessary for now, while
  // it doesn't change...
  while (!quit) {
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT) {
        quit = 1;
        break;
      }
    }
    UpdateDisplayedImage();
    if (SDL_RenderCopy(g.renderer, g.image, NULL, NULL) < 0) {
      printf("Error rendering image: %s\n", SDL_GetError());
      CleanupGlobals();
      exit(1);
    }
    SDL_RenderPresent(g.renderer);
    usleep(20000);
  }
}

int main(int argc, char **argv) {
  memset(&g, 0, sizeof(g));
  g.w = 1000;
  g.h = 1000;
  g.min_real = -2.0;
  g.min_imag = -2.0;
  g.max_real = 2.0;
  g.max_imag = 2.0;
  g.max_iterations = 100;
  g.delta_real = (g.max_real - g.min_real) / ((double) g.w);
  g.delta_imag = (g.max_imag - g.min_imag) / ((double) g.h);
  printf("Calculating image...\n");
  SetupCUDA();
  RenderImage();
  printf("Done!\n");
  SetupSDL();
  SDLWindowLoop();
  CleanupGlobals();
  return 0;
}
