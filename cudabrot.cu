#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
extern "C" {
#include <SDL2/SDL.h>
}

// Controls the number of threads per block to use.
#define DEFAULT_BLOCK_SIZE (1024)

// Controls the default number of blocks to use.
#define DEFAULT_BLOCK_COUNT (16)

// This macro takes a cudaError_t value and exits the program if it isn't equal
// to cudaSuccess. (Calls the ErrorCheck function, defined later).
#define CheckCUDAError(val) (InternalCUDAErrorCheck((val), #val, __FILE__, __LINE__))

// This will be a magic value instructing the program to not explicitly set a
// CUDA device.
#define USE_DEFAULT_DEVICE (-1)

// The gamma value to use for gamma correction, or 1.0 if no gamma correction
// should be applied.
#define GAMMA_CORRECTION (1.1)

// The number of color channels in the resulting image. Should be 3 for RBG.
#define COLOR_CHANNELS (3)

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

// This struct holds the parameters for different types of "iterations" needed
// when calculating the buddhabrot.
typedef struct {
  // Each CUDA thread in every block will sample this many random points.
  int samples_per_thread;
  // This is the maximum number of iterations to run to see if a point escapes.
  int max_escape_iterations;
  // If a point escapes in fewer than this many iterations, it will be ignored.
  int min_escape_iterations;
} IterationControl;

// Holds globals in a single namespace.
static struct {
  SDL_Window *window;
  SDL_Renderer *renderer;
  SDL_Texture *image;
  // The CUDA device to use. If this is -1, a device won't be set, which should
  // fall back to CUDA's normal device.
  int cuda_device;
  // This tracks the random number generator states for the GPU code.
  curandState_t *rng_states;
  // The number of threads and blocks to use when calculating the buddhabrot.
  int block_size, block_count;
  // The filename to which a bitmap image will be saved, or NULL if an image
  // should not be saved.
  char *output_image;
  // The number of iterations to check for escaping points in the buddhabrot.
  int buddhabrot_iterations;
  // The size and location of the fractal and output image.
  FractalDimensions dimensions;
  // The host and device buffers which contain the numbers of times an escaping
  // point's path crossed each point in the complex plane.
  uint32_t *device_buddhabrot;
  uint32_t *host_buddhabrot;
  // Buffers for the three different color channels, which will be calculated
  // separately and combined into the final image.
  uint8_t *color_channels[COLOR_CHANNELS];
} g;

// If any globals have been initialized, this will free them. (Relies on
// globals being set to 0 at the start of the program)
static void CleanupGlobals(void) {
  int i;
  if (g.renderer) SDL_DestroyRenderer(g.renderer);
  if (g.image) SDL_DestroyTexture(g.image);
  if (g.window) SDL_DestroyWindow(g.window);
  if (g.rng_states) cudaFree(g.rng_states);
  if (g.device_buddhabrot) cudaFree(g.device_buddhabrot);
  for (i = 0; i < COLOR_CHANNELS; i++) {
    if (g.color_channels[i]) free(g.color_channels[i]);
  }
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
  CleanupGlobals();
  exit(1);
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

// This function is used to initialize the RNG states to use when generating
// starting points in the buddhabrot calculation. The states array must hold
// one entry for every thread in every block.
__global__ void InitializeRNG(uint64_t seed, curandState_t *states) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  curand_init(seed, index, 0, states + index);
}

// Allocates CUDA memory and calculates block/grid sizes. Must be called after
// g.w and g.h have been set.
static void SetupCUDA(void) {
  int i;
  if (g.cuda_device != USE_DEFAULT_DEVICE) {
    CheckCUDAError(cudaSetDevice(g.cuda_device));
  }
  size_t buffer_size = g.dimensions.w * g.dimensions.h;

  // Initialize the host and device image buffers.
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

  // Initialize the RNG state for the device.
  CheckCUDAError(cudaMalloc(&(g.rng_states), g.block_size * g.block_count *
    sizeof(curandState_t)));
  InitializeRNG<<<g.block_size, g.block_count>>>(1337, g.rng_states);
  CheckCUDAError(cudaDeviceSynchronize());

  // Allocate the color channels for the combined image.
  for (i = 0; i < COLOR_CHANNELS; i++) {
    g.color_channels[i] = (uint8_t *) malloc(buffer_size);
    if (!g.color_channels[i]) {
      printf("Failed allocating color channel %d buffer.\n", i);
      CleanupGlobals();
      exit(1);
    }
    memset(g.color_channels[i], 0, buffer_size);
  }
}

// This should be used to update the pixel data for a point that is encountered
// in the set.
__device__ void IncrementPixelCounter(int row, int col, uint32_t *data,
    FractalDimensions *d) {
  int r, c;
  r = row;
  c = col;
  if ((r >= 0) && (r < d->h) && (c >= 0) && (c < d->h)) {
    data[r * d->w + c] += 4;
  }
}

// This kernel takes a list of points which escape the mandelbrot set, and, for
// each iteration of the point, increments its location in the data array.
__global__ void DrawBuddhabrot(FractalDimensions dimensions, uint32_t *data,
    IterationControl iterations, curandState_t *states) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  curandState_t *rng = states + index;
  int i, j, point_escaped, record_path, row, col;
  float start_real, start_imag, current_real, current_imag, tmp;
  float real_range = dimensions.max_real - dimensions.min_real;
  float imag_range = dimensions.max_imag - dimensions.min_imag;
  record_path = 0;
  point_escaped = 1;
  for (i = 0; i < iterations.samples_per_thread; i++) {
    // Calculate a new starting point only if the previous point didn't escape.
    // Otherwise, we'll use the same starting point, and record the point's
    // path.
    if (!record_path) {
      start_real = curand_uniform(rng) * real_range + dimensions.min_real;
      start_imag = curand_uniform(rng) * imag_range + dimensions.min_imag;
    }
    point_escaped = 0;
    current_real = start_real;
    current_imag = start_imag;
    for (j = 0; j < iterations.max_escape_iterations; j++) {
      tmp = (current_real * current_real) - (current_imag * current_imag) +
        start_real;
      current_imag = 2 * current_real * current_imag + start_imag;
      current_real = tmp;
      row = (current_imag - dimensions.min_imag) / dimensions.delta_imag;
      col = (current_real - dimensions.min_real) / dimensions.delta_real;
      if (record_path) {
        IncrementPixelCounter(row, col, data, &dimensions);
      }
      // If the point escapes, stop iterating and indicate the loop ended due
      // to the point escaping.
      if (((current_real * current_real) + (current_imag * current_imag)) >
        4) {
        point_escaped = 1;
        break;
      }
    }
    // Record the next path if the point didn't escape and we weren't already
    // recording.
    if (point_escaped && !record_path) {
      // Enables ignoring paths that escape too quickly.
      if (j > iterations.min_escape_iterations) record_path = 1;
    } else {
      record_path = 0;
    }
  }
}

static uint8_t Clamp(double v) {
  if (v <= 0) return 0;
  if (v >= 255) return 255;
  return (uint8_t) v;
}

// Returns the amount to multiply the original count by in order to get a value
// by which buddhabrot counts can be multiplied to get a number between 0 and
// 255.
static double GetLinearColorScale(void) {
  int x, y, index;
  uint32_t max = 0;
  index = 0;
  for (y = 0; y < g.dimensions.h; y++) {
    for (x = 0; x < g.dimensions.w; x++) {
      if (g.host_buddhabrot[index] > max) max = g.host_buddhabrot[index];
    }
  }
  return 255.0 / ((double) max);
}

// Returns the gamma-corrected 8-bit color channel value given a buddhabrot
// iteration count c.
static uint8_t DoGammaCorrection(uint32_t c, double linear_scale) {
  double scaled = ((double) c) * linear_scale;
  scaled = 255 * log(c + 1) / log(255);
  return Clamp(255 * pow(scaled / 255, 1 / GAMMA_CORRECTION));
}

// Fills in a single color channel from the current host_buddhabrot buffer.
static void SetColorChannel(uint8_t *color) {
  int x, y;
  uint8_t color_value;
  double linear_scale = GetLinearColorScale();
  uint32_t *host_data = g.host_buddhabrot;
  for (y = 0; y < g.dimensions.h; y++) {
    for (x = 0; x < g.dimensions.w; x++) {
      color_value = DoGammaCorrection(*host_data, linear_scale);
      *color = color_value;
      color++;
      host_data++;
    }
  }
}

// Renders the fractal image.
static void RenderImage(void) {
  int channel;
  size_t data_size = g.dimensions.w * g.dimensions.h;
  double seconds;
  IterationControl iterations;
  iterations.min_escape_iterations = 20;
  iterations.samples_per_thread = 100;
  iterations.max_escape_iterations = g.buddhabrot_iterations;

  for (channel = 0; channel < COLOR_CHANNELS; channel++) {
    printf("Calculating color channel %d.\n", channel);
    printf("Calculating buddhabrot.\n");
    seconds = CurrentSeconds();
    DrawBuddhabrot<<<g.block_count, g.block_size>>>(g.dimensions,
       g.device_buddhabrot, iterations, g.rng_states);
    CheckCUDAError(cudaGetLastError());
    CheckCUDAError(cudaMemcpy(g.host_buddhabrot, g.device_buddhabrot,
      data_size * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    printf("  Buddhabrot took %f seconds.\n", CurrentSeconds() - seconds);

    SetColorChannel(g.color_channels[channel]);
    // Color channels will only differ by a fixed number of iterations for now.
    iterations.max_escape_iterations /= 10;
    iterations.min_escape_iterations /= 2;
  }
}

// Copies data from the host-side data buffer to the texture drawn to the SDL
// window.
static void UpdateDisplayedImage(void) {
  int x, y;
  uint8_t *image_pixels;
  int image_pitch, to_skip_per_row, pixel_number;
  if (SDL_LockTexture(g.image, NULL, (void **) (&image_pixels), &image_pitch)
    < 0) {
    printf("Error locking SDL texture: %s\n", SDL_GetError());
    CleanupGlobals();
    exit(1);
  }
  // Abide by the image pitch, and skip unaffected bytes in each row.
  // (image_pitch should usually be equal to g.w * 4 anyway).
  to_skip_per_row = image_pitch - (g.dimensions.w * 4);
  pixel_number = 0;
  for (y = 0; y < g.dimensions.h; y++) {
    for (x = 0; x < g.dimensions.w; x++) {
      // The byte order is ABGR
      image_pixels[0] = 0xff;
      image_pixels[1] = g.color_channels[2][pixel_number];
      image_pixels[2] = g.color_channels[1][pixel_number];
      image_pixels[3] = g.color_channels[0][pixel_number];
      image_pixels += 4;
      pixel_number++;
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

// If a filename has been set for saving the image, this will attempt to save
// the image to the file.
static void SaveImage(void) {
  void *pixel_data;
  SDL_Surface *image_surface = NULL;
  int w = g.dimensions.w;
  int h = g.dimensions.h;
  // Don't do anything if the output filename wasn't set.
  if (!g.output_image) return;

  // In SDL 2, we need to copy the image from the renderer and create an SDL
  // surface in order to save a bitmap.
  pixel_data = malloc(w * h * 4);
  if (!pixel_data) {
    printf("Failed allocating space to save an image.\n");
    CleanupGlobals();
    exit(1);
  }
  if (SDL_RenderReadPixels(g.renderer, NULL, SDL_PIXELFORMAT_RGBA8888,
    pixel_data, w * 4) != 0) {
    printf("Failed getting BMP image data: %s\n", SDL_GetError());
    free(pixel_data);
    CleanupGlobals();
    exit(1);
  }
  image_surface = SDL_CreateRGBSurfaceFrom(pixel_data, w, h, 32, w * 4, 0xff,
    0xff00, 0xff0000, 0xff000000);
  if (!image_surface) {
    printf("Failed creating BMP surface: %s\n", SDL_GetError());
    free(pixel_data);
    CleanupGlobals();
    exit(1);
  }
  if (SDL_SaveBMP(image_surface, g.output_image) != 0) {
    printf("Failed saving BMP file: %s\n", SDL_GetError());
    SDL_FreeSurface(image_surface);
    free(pixel_data);
    CleanupGlobals();
    exit(1);
  }
  printf("Successfully saved %s\n", g.output_image);
  SDL_FreeSurface(image_surface);
  free(pixel_data);
}

static void PrintUsage(char *program_name) {
  printf("Usage: %s [options]\n\n", program_name);
  printf("Options may be one or more of the following:\n"
    "  --help: Prints these instructions.\n"
    "  -d <CUDA device number>: Can be used to set which GPU to use.\n"
    "     Defaults to the default GPU.\n"
    "  -s <output file name>: If provided, the rendered image will be saved\n"
    "     to a bitmap file with the given name, in addition to being\n"
    "     displayed in a window.\n"
    "  -b <buddhabrot iterations>: The number of iterations to use for the\n"
    "     buddhabrot calculation. Defaults to 1000.\n");
  exit(0);
}

// Returns an integer at the argument after index in argv. Exits if the integer
// is invalid.
static int ParseIntArg(int argc, char **argv, int index) {
  char *tmp = NULL;
  int to_return = 0;
  if ((index + 1) >= argc) {
    printf("Argument %s needs a value.\n", argv[index]);
    PrintUsage(argv[0]);
  }
  to_return = strtol(argv[index + 1], &tmp, 10);
  if (*tmp != 0) {
    printf("Invalid number given to argument %s: %s\n", argv[index],
      argv[index + 1]);
    PrintUsage(argv[0]);
  }
  return to_return;
}

// Processes command-line arguments, setting values in the globals struct as
// necessary.
static void ParseArguments(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--help") == 0) {
      PrintUsage(argv[0]);
    }
    if (strcmp(argv[i], "-d") == 0) {
      g.cuda_device = ParseIntArg(argc, argv, i);
      i++;
      continue;
    }
    if (strcmp(argv[i], "-s") == 0) {
      if ((i + 1) >= argc) {
        printf("Missing output file name.\n");
        PrintUsage(argv[0]);
      }
      i++;
      g.output_image = argv[i];
      continue;
    }
    if (strcmp(argv[i], "-b") == 0) {
      g.buddhabrot_iterations = ParseIntArg(argc, argv, i);
      i++;
      continue;
    }
    // Unrecognized argument, print the usage string.
    printf("Invalid argument: %s\n", argv[i]);
    PrintUsage(argv[0]);
  }
}

int main(int argc, char **argv) {
  memset(&g, 0, sizeof(g));
  g.buddhabrot_iterations = 1000;
  g.block_size = DEFAULT_BLOCK_SIZE;
  g.block_count = DEFAULT_BLOCK_COUNT;
  SetResolution(1000, 1000);
  g.cuda_device = USE_DEFAULT_DEVICE;
  ParseArguments(argc, argv);
  printf("Calculating image...\n");
  SetupCUDA();
  RenderImage();
  printf("Done!\n");
  SetupSDL();
  SDLWindowLoop();
  SaveImage();
  CleanupGlobals();
  return 0;
}
