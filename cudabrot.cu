#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
extern "C" {
#include <SDL2/SDL.h>
}

// The number of CUDA threads to use per block.
#define DEFAULT_BLOCK_SIZE (128)

// The number of iterations to record the paths of points that escape the set.
#define PATH_ITERATIONS (20000)

// This macro takes a cudaError_t value and exits the program if it isn't equal
// to cudaSuccess. (Calls the ErrorCheck function, defined later).
#define CheckCUDAError(val) (InternalCUDAErrorCheck((val), #val, __FILE__, __LINE__))

// Holds the boundaries and sizes of the fractal, in both pixels and numbers
typedef struct {
  // The width and height of the image in pixels.
  int w;
  int h;
  // The boundaries of the fractal.
  double min_real;
  double min_imag;
  double max_real;
  double max_imag;
  // The distance between pixels in the real and imaginary axes.
  double delta_real;
  double delta_imag;
} FractalDimensions;

// Holds globals in a single namespace.
static struct {
  SDL_Window *window;
  SDL_Renderer *renderer;
  SDL_Texture *image;
  // The maximum number of iterations to run each point.
  uint32_t max_iterations;
  // The size and location of the fractal and output image.
  FractalDimensions dimensions;
  // Pointer to the device memory that will contain 0 if a point is in the set,
  // and 1 if it escapes the set.
  uint8_t *device_point_escapes;
  // The host-side copy of which points escape.
  uint8_t *host_point_escapes;
} g;

// If any globals have been initialized, this will free them. (Relies on
// globals being set to 0 at the start of the program)
static void CleanupGlobals(void) {
  if (g.renderer) SDL_DestroyRenderer(g.renderer);
  if (g.image) SDL_DestroyTexture(g.image);
  if (g.window) SDL_DestroyWindow(g.window);
  if (g.device_point_escapes) cudaFree(g.device_point_escapes);
  if (g.host_point_escapes) free(g.host_point_escapes);
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
    SDL_WINDOWPOS_UNDEFINED, g.dimensions.w, g.dimensions.h, SDL_WINDOW_SHOWN |
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
    SDL_TEXTUREACCESS_STREAMING, g.dimensions.w, g.dimensions.h);
  if (!g.image) {
    printf("Failed creating SDL texture: %s\n", SDL_GetError());
    exit(1);
  }
}

// Allocates CUDA memory and calculates block/grid sizes. Must be called after
// g.w and g.h have been set.
static void SetupCUDA(void) {
  size_t buffer_size = g.dimensions.w * g.dimensions.h;
  CheckCUDAError(cudaMalloc(&(g.device_point_escapes), buffer_size));
  g.host_point_escapes = (uint8_t *) malloc(buffer_size);
  if (!g.host_point_escapes) {
    printf("Failed allocating host buffer.\n");
    CleanupGlobals();
    exit(1);
  }
}

// A basic mandelbrot set calculator which sets each element in data to 1 if
// the point escapes within the given number of iterations.
__global__ void BasicMandelbrot(uint8_t *data, int iterations,
    FractalDimensions dimensions) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int row = index / dimensions.w;
  int col = index % dimensions.w;
  // This may cause some threads to diverge on the last block only
  if (row >= dimensions.h) return;
  double start_real = dimensions.min_real + dimensions.delta_real * col;
  double start_imag = dimensions.min_imag + dimensions.delta_imag * row;
  double current_real = start_real;
  double current_imag = start_imag;
  double magnitude_squared = (start_real * start_real) + (start_imag *
    start_imag);
  uint8_t escaped = 0;
  double tmp;
  int i;
  for (i = 0; i < iterations; i++) {
    if (magnitude_squared < 4) {
      tmp = (current_real * current_real) - (current_imag * current_imag) +
        start_real;
      current_imag = 2 * current_imag * current_real + start_imag;
      current_real = tmp;
      magnitude_squared = (current_real * current_real) + (current_imag *
        current_imag);
    } else {
      escaped = 1;
    }
  }
  data[row * dimensions.w + col] = escaped;
}

// Renders the fractal image.
static void RenderImage(void) {
  int block_count;
  size_t data_size = g.dimensions.w * g.dimensions.h;
  block_count = (data_size / DEFAULT_BLOCK_SIZE) + 1;
  BasicMandelbrot<<<block_count, DEFAULT_BLOCK_SIZE>>>(g.device_point_escapes,
    g.max_iterations, g.dimensions);
  CheckCUDAError(cudaGetLastError());
  CheckCUDAError(cudaMemcpy(g.host_point_escapes, g.device_point_escapes,
    data_size, cudaMemcpyDeviceToHost));
}

// Copies data from the host-side data buffer to the texture drawn to the SDL
// window.
static void UpdateDisplayedImage(void) {
  int x, y;
  uint8_t *image_pixels;
  int image_pitch;
  int to_skip_per_row;
  uint8_t color_value;
  uint8_t *host_data = g.host_point_escapes;
  if (SDL_LockTexture(g.image, NULL, (void **) (&image_pixels), &image_pitch)
    < 0) {
    printf("Error locking SDL texture: %s\n", SDL_GetError());
    CleanupGlobals();
    exit(1);
  }
  // Abide by the image pitch, and skip unaffected bytes in each row.
  // (image_pitch should usually be equal to g.w * 4 anyway).
  to_skip_per_row = image_pitch - (g.dimensions.w * 4);
  for (y = 0; y < g.dimensions.h; y++) {
    for (x = 0; x < g.dimensions.w; x++) {
      if (*host_data) {
        color_value = 0xff;
      } else {
        color_value = 0;
      }
      // The byte order is ABGR
      image_pixels[0] = 0xff;
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
  FractalDimensions *dimensions = NULL;
  memset(&g, 0, sizeof(g));
  dimensions = &(g.dimensions);
  dimensions->w = 1000;
  dimensions->h = 1000;
  dimensions->min_real = -2.0;
  dimensions->min_imag = -2.0;
  dimensions->max_real = 2.0;
  dimensions->max_imag = 2.0;
  dimensions->delta_real = (dimensions->max_real - dimensions->min_real) /
    ((double) dimensions->w);
  dimensions->delta_imag = (dimensions->max_imag - dimensions->min_imag) /
    ((double) dimensions->h);
  g.max_iterations = 200;
  printf("Calculating image...\n");
  SetupCUDA();
  RenderImage();
  printf("Done!\n");
  SetupSDL();
  SDLWindowLoop();
  CleanupGlobals();
  return 0;
}
