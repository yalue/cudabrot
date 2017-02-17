#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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

// Tracks a single pair of points which escaped the mandelbrot set. These will
// be used as the start points of buddhabrot paths.
typedef struct {
  double real;
  double imag;
} EscapingPoint;

// Holds globals in a single namespace.
static struct {
  SDL_Window *window;
  SDL_Renderer *renderer;
  SDL_Texture *image;
  // The maximum number of iterations to run each point in the initial
  // mandelbrot calculation.
  int mandelbrot_iterations;
  // The number of iterations to track the paths of escaping points in the
  // buddhabrot.
  int buddhabrot_iterations;
  // The size and location of the fractal and output image.
  FractalDimensions dimensions;
  // Pointer to the device memory that will contain 0 if a point is in the set,
  // and 1 if it escapes the set.
  uint8_t *device_mandelbrot;
  // The host-side copy of the basic binary mandelbrot set.
  uint8_t *host_mandelbrot;
  // Lists of points which escape the mandelbrot set.
  EscapingPoint *host_escaping_points;
  EscapingPoint *device_escaping_points;
  // The number of points which escaped the mandelbrot set.
  int escaping_point_count;
  // The host and device buffers which contain the numbers of times an escaping
  // point's path crossed each point in the complex plane.
  uint32_t *device_buddhabrot;
  uint32_t *host_buddhabrot;
} g;

// If any globals have been initialized, this will free them. (Relies on
// globals being set to 0 at the start of the program)
static void CleanupGlobals(void) {
  if (g.renderer) SDL_DestroyRenderer(g.renderer);
  if (g.image) SDL_DestroyTexture(g.image);
  if (g.window) SDL_DestroyWindow(g.window);
  if (g.device_mandelbrot) cudaFree(g.device_mandelbrot);
  if (g.host_mandelbrot) free(g.host_mandelbrot);
  if (g.host_escaping_points) free(g.host_escaping_points);
  if (g.device_escaping_points) cudaFree(g.device_escaping_points);
  memset(&g, 0, sizeof(g));
}

// Returns the current time in seconds.
static double CurrentSeconds(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_REALTIME, &ts) != 0) {
    printf("Error getting time.\n");
    exit(1);
  }
  return ((double) ts.tv_sec) + (((double) ts.tv_nsec) / 1e9);
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
  CheckCUDAError(cudaFree(0));
  size_t buffer_size = g.dimensions.w * g.dimensions.h;
  CheckCUDAError(cudaMalloc(&(g.device_mandelbrot), buffer_size));
  CheckCUDAError(cudaMemset(g.device_mandelbrot, 0, buffer_size));
  g.host_mandelbrot = (uint8_t *) malloc(buffer_size);
  if (!g.host_mandelbrot) {
    printf("Failed allocating host mandelbrot buffer.\n");
    CleanupGlobals();
    exit(1);
  }
  memset(g.host_mandelbrot, 0, buffer_size);
  CheckCUDAError(cudaMalloc(&(g.device_buddhabrot), buffer_size *
    sizeof(uint32_t)));
  CheckCUDAError(cudaMemset(g.device_buddhabrot, 0, buffer_size *
    sizeof(uint32_t)));
  g.host_buddhabrot = (uint32_t *) malloc(buffer_size * sizeof(uint32_t));
  if (!g.host_buddhabrot) {
    printf("Failed allocating host buddhabrot buffer.\n");
    CleanupGlobals();
    exit(1);
  }
  memset(g.host_buddhabrot, 0, buffer_size * sizeof(uint32_t));
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

// After BasicMandelbrot has been completed, and host_mandelbrot has been
// filled in, this will allocate and populate both device_escaping_points and
// host_escaping_points.
static void GatherEscapingPoints(void) {
  int w = g.dimensions.w;
  int h = g.dimensions.h;
  int x, y;
  size_t points_size = 0;
  int points_added = 0;
  EscapingPoint *escaping_point = NULL;

  // First, get a count of the escaping points, so the correct amount of memory
  // can be allocated.
  int count = 0;
  for (y = 0; y < h; y++) {
    for (x = 0; x < w; x++) {
      if (g.host_mandelbrot[y * w + x]) count++;
    }
  }
  g.escaping_point_count = count;

  // Next, build the list of escaping points and copy it to GPU memory.
  points_size = count * sizeof(EscapingPoint);
  g.host_escaping_points = (EscapingPoint *) malloc(points_size);
  if (!g.host_escaping_points) {
    printf("Failed allocating space for escaping point list.\n");
    CleanupGlobals();
    exit(1);
  }
  CheckCUDAError(cudaMalloc(&(g.device_escaping_points), points_size));
  for (y = 0; y < h; y++) {
    for (x = 0; x < w; x++) {
      if (!g.host_mandelbrot[y * w + x]) continue;
      escaping_point = g.host_escaping_points + points_added;
      escaping_point->real = ((double) x) * g.dimensions.delta_real +
        g.dimensions.min_real;
      escaping_point->imag = ((double) y) * g.dimensions.delta_imag +
        g.dimensions.min_imag;
      points_added++;
    }
  }
  CheckCUDAError(cudaMemcpy(g.device_escaping_points, g.host_escaping_points,
    points_size, cudaMemcpyHostToDevice));
}

// This kernel takes a list of points which escape the mandelbrot set, and, for
// each iteration of the point, increments its location in the data array.
__global__ void DrawBuddhabrot(EscapingPoint *points, int point_count,
    uint32_t *data, int iterations, FractalDimensions dimensions) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= point_count) return;
  int i;
  double start_real = points[index].real;
  double start_imag = points[index].imag;
  double current_real = start_real;
  double current_imag = start_imag;
  double tmp;
  int row, col;
  // This should only happen in the final block.
  if (index > point_count) return;
  for (i = 0; i < iterations; i++) {
    tmp = (current_real * current_real) - (current_imag * current_imag) +
      start_real;
    current_imag = 2 * current_real * current_imag + start_imag;
    current_real = tmp;
    row = (current_imag - dimensions.min_imag) / dimensions.delta_imag;
    col = (current_real - dimensions.min_real) / dimensions.delta_real;
    if ((row >= 0) && (row < dimensions.h) && (col >= 0) && (col <
      dimensions.w)) {
      data[row * dimensions.w + col]++;
    }
  }
}

// Renders the fractal image.
static void RenderImage(void) {
  int block_count;
  size_t data_size = g.dimensions.w * g.dimensions.h;
  double seconds;

  printf("Calculating initial mandelbrot set.\n");
  // First, draw the basic mandelbrot to get which points escape.
  block_count = (data_size / DEFAULT_BLOCK_SIZE) + 1;
  seconds = CurrentSeconds();
  BasicMandelbrot<<<block_count, DEFAULT_BLOCK_SIZE>>>(g.device_mandelbrot,
    g.mandelbrot_iterations, g.dimensions);
  CheckCUDAError(cudaGetLastError());
  CheckCUDAError(cudaMemcpy(g.host_mandelbrot, g.device_mandelbrot,
    data_size, cudaMemcpyDeviceToHost));
  printf("Mandelbrot took %f seconds.\n", CurrentSeconds() - seconds);

  printf("Finding start points for buddhabrot.\n");
  GatherEscapingPoints();

  printf("Calculating buddhabrot.\n");
  block_count = (g.escaping_point_count / DEFAULT_BLOCK_SIZE) + 1;
  seconds = CurrentSeconds();
  DrawBuddhabrot<<<block_count, DEFAULT_BLOCK_SIZE>>>(g.device_escaping_points,
    g.escaping_point_count, g.device_buddhabrot, g.buddhabrot_iterations,
    g.dimensions);
  CheckCUDAError(cudaGetLastError());
  CheckCUDAError(cudaMemcpy(g.host_buddhabrot, g.device_buddhabrot,
    data_size * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  printf("Buddhabrot took %f seconds.\n", CurrentSeconds() - seconds);
}

static double GetColorScale(void) {
  // TODO: Figure out a color scaling system that works:
  //  - Highest value should still be 255.
  //  - Lowest value should be some pale gray
  //  - Should have a lograthmic curve that amplifies low values a lot
  return 16;
}

static uint8_t Clamp(double v) {
  if (v <= 0) return 0;
  if (v >= 255) return 255;
  return (uint8_t) v;
}

// Copies data from the host-side data buffer to the texture drawn to the SDL
// window.
static void UpdateDisplayedImage(void) {
  int x, y;
  uint8_t *image_pixels;
  int image_pitch;
  int to_skip_per_row;
  uint8_t color_value;
  double color_scale = GetColorScale();
  uint32_t *host_data = g.host_buddhabrot;
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
      color_value = Clamp(color_scale * (*host_data));
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

// Sets the resolution, scaling the complex boundaries to maintain an aspect
// ratio.
static void SetResolution(int width, int height) {
  FractalDimensions *dims = &(g.dimensions);
  double ratio = ((double) height) / ((double) width);
  // The horizontal width for which the complex plane is shown.
  double real_width = 4.0;
  double imag_width = real_width * ratio;
  dims->w = width;
  dims->h = height;
  dims->min_real = -(real_width / 2.0);
  dims->max_real = dims->min_real + real_width;
  dims->min_imag = -(imag_width / 2.0);
  dims->max_imag = dims->min_imag + imag_width;
  dims->delta_imag = imag_width / ((double) height);
  dims->delta_real = real_width / ((double) width);
}

int main(int argc, char **argv) {
  memset(&g, 0, sizeof(g));
  SetResolution(3840, 2400);
  g.mandelbrot_iterations = 100;
  g.buddhabrot_iterations = 20000;
  printf("Calculating image...\n");
  SetupCUDA();
  RenderImage();
  printf("Done!\n");
  SetupSDL();
  SDLWindowLoop();
  CleanupGlobals();
  return 0;
}
