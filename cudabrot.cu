#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

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
#define GAMMA_CORRECTION (1.0)

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
  // Information about the number of iterations to use.
  IterationControl iterations;
  // The size and location of the fractal and output image.
  FractalDimensions dimensions;
  // The host and device buffers which contain the numbers of times an escaping
  // point's path crossed each point in the complex plane.
  uint64_t *device_buddhabrot;
  uint64_t *host_buddhabrot;
  // This buffer will hold the 16-bit grayscale image when converted from the
  // 64-bit counters.
  uint16_t *grayscale_image;
} g;

// If any globals have been initialized, this will free them. (Relies on
// globals being set to 0 at the start of the program)
static void CleanupGlobals(void) {
  if (g.rng_states) cudaFree(g.rng_states);
  if (g.device_buddhabrot) cudaFree(g.device_buddhabrot);
  if (g.grayscale_image) free(g.grayscale_image);
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
  if (g.cuda_device != USE_DEFAULT_DEVICE) {
    CheckCUDAError(cudaSetDevice(g.cuda_device));
  }
  size_t buffer_size = g.dimensions.w * g.dimensions.h;

  // Initialize the host and device image buffers.
  CheckCUDAError(cudaMalloc(&(g.device_buddhabrot), buffer_size *
    sizeof(uint64_t)));
  CheckCUDAError(cudaMemset(g.device_buddhabrot, 0, buffer_size *
    sizeof(uint64_t)));
  g.host_buddhabrot = (uint64_t *) malloc(buffer_size * sizeof(uint64_t));
  if (!g.host_buddhabrot) {
    printf("Failed allocating host buddhabrot buffer.\n");
    CleanupGlobals();
    exit(1);
  }
  memset(g.host_buddhabrot, 0, buffer_size * sizeof(uint64_t));

  // Initialize the RNG state for the device.
  CheckCUDAError(cudaMalloc(&(g.rng_states), g.block_size * g.block_count *
    sizeof(curandState_t)));
  InitializeRNG<<<g.block_size, g.block_count>>>(1337, g.rng_states);
  CheckCUDAError(cudaDeviceSynchronize());

  // Allocate the buffer for the grayscale bitmap.
  g.grayscale_image = (uint16_t *) malloc(buffer_size * sizeof(uint16_t));
  if (!g.grayscale_image) {
    printf("Failed allocating grayscale image buffer.\n");
    CleanupGlobals();
    exit(1);
  }
  memset(g.grayscale_image, 0, buffer_size * sizeof(uint16_t));
}

// This should be used to update the pixel data for a point that is encountered
// in the set.
__device__ void IncrementPixelCounter(int row, int col, uint64_t *data,
    FractalDimensions *d) {
  int r, c;
  r = row;
  c = col;
  if ((r >= 0) && (r < d->h) && (c >= 0) && (c < d->h)) {
    data[r * d->w + c] += 1;
  }
}

// This kernel takes a list of points which escape the mandelbrot set, and, for
// each iteration of the point, increments its location in the data array.
__global__ void DrawBuddhabrot(FractalDimensions dimensions, uint64_t *data,
    IterationControl iterations, curandState_t *states) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  curandState_t *rng = states + index;
  int i, j, point_escaped, record_path, row, col;
  double start_real, start_imag, current_real, current_imag, tmp;
  double real_range = dimensions.max_real - dimensions.min_real;
  double imag_range = dimensions.max_imag - dimensions.min_imag;
  record_path = 0;
  point_escaped = 1;
  for (i = 0; i < iterations.samples_per_thread; i++) {
    // Calculate a new starting point only if the previous point didn't escape.
    // Otherwise, we'll use the same starting point, and record the point's
    // path.
    if (!record_path) {
      start_real = curand_uniform_double(rng) * real_range +
        dimensions.min_real;
      start_imag = curand_uniform_double(rng) * imag_range +
        dimensions.min_imag;
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

static uint16_t Clamp(double v) {
  if (v <= 0) return 0;
  if (v >= 0xffff) return 0xffff;
  return (uint16_t) v;
}

// Returns the amount to multiply the original count by in order to get a value
// by which buddhabrot counts can be multiplied to get a number between 0 and
// 255.
static double GetLinearColorScale(void) {
  int x, y, index;
  uint64_t max = 0;
  index = 0;
  for (y = 0; y < g.dimensions.h; y++) {
    for (x = 0; x < g.dimensions.w; x++) {
      if (g.host_buddhabrot[index] > max) max = g.host_buddhabrot[index];
    }
  }
  return ((double) 0xffff) / ((double) max);
}

// Returns the gamma-corrected 8-bit color channel value given a buddhabrot
// iteration count c.
static uint16_t DoGammaCorrection(uint64_t c, double linear_scale) {
  double max = 0xffff;
  double scaled = ((double) c) * linear_scale;
  scaled = max * log(c + 1) / log(max);
  return Clamp(max * pow(scaled / max, 1.0 / GAMMA_CORRECTION));
}

// Fills in the grayscale color image from the current host buddhabrot buffer.
static void ConvertToGrayscale(void) {
  int x, y;
  uint16_t color_value;
  double linear_scale = GetLinearColorScale();
  uint64_t *host_data = g.host_buddhabrot;
  uint16_t *output_image = g.grayscale_image;
  for (y = 0; y < g.dimensions.h; y++) {
    for (x = 0; x < g.dimensions.w; x++) {
      color_value = DoGammaCorrection(*host_data, linear_scale);
      *output_image = color_value;
      output_image++;
      host_data++;
    }
  }
}

// Renders the fractal image.
static void RenderImage(void) {
  size_t data_size = g.dimensions.w * g.dimensions.h;
  double seconds;

  printf("Calculating buddhabrot.\n");
  seconds = CurrentSeconds();
  DrawBuddhabrot<<<g.block_count, g.block_size>>>(g.dimensions,
     g.device_buddhabrot, g.iterations, g.rng_states);
  CheckCUDAError(cudaGetLastError());
  CheckCUDAError(cudaMemcpy(g.host_buddhabrot, g.device_buddhabrot,
    data_size * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  printf("Buddhabrot took %f seconds.\n", CurrentSeconds() - seconds);

  ConvertToGrayscale();
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
// If a filename has been set for an output image, this will attempt to save
// the file. The image will be in 16-bit PGM format.
static void SaveImage(void) {
  FILE *output = NULL;
  size_t buffer_size = g.dimensions.w * g.dimensions.h * sizeof(uint16_t);
  output = fopen(g.output_image, "wb");
  if (!output) {
    printf("Failed opening output file (%s).\n", g.output_image);
    CleanupGlobals();
    exit(1);
  }
  if (fprintf(output, "P5\n%d %d\n%d\n", g.dimensions.w, g.dimensions.h,
    0xffff) <= 0) {
    printf("Failed writing PGM file header.\n");
    fclose(output);
    CleanupGlobals();
    exit(1);
  }
  if (!fwrite(g.grayscale_image, buffer_size, 1, output)) {
    printf("Failed writing PGM pixel data.\n");
    fclose(output);
    CleanupGlobals();
    exit(1);
  }
  fclose(output);
  printf("Saved output image OK.\n");
}
// TODO (next): Test pgm output

static void PrintUsage(char *program_name) {
  printf("Usage: %s [options]\n\n", program_name);
  printf("Options may be one or more of the following:\n"
    "  --help: Prints these instructions.\n"
    "  -d <CUDA device number>: Can be used to set which GPU to use.\n"
    "     Defaults to the default GPU.\n"
    "  -o <output file name>: If provided, the rendered image will be saved\n"
    "     to a bitmap file with the given name, in addition to being\n"
    "     displayed in a window.\n"
    "  -m <max escape iterations>: The number of iterations to wait before\n"
    "     seeing if a point has escaped.\n"
    "  -c <min escape iterations>: If a point escpaes before iterating this\n"
    "     many times, it won't be included in the buddhabrot.\n"
    "  -r <resolution>: The number of pixels across the (square) output.\n"
    "  -s <samples per thread>: The number of samples performed by each CUDA"
    "     thread.\n");
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
  int resolution;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--help") == 0) {
      PrintUsage(argv[0]);
    }
    if (strcmp(argv[i], "-d") == 0) {
      g.cuda_device = ParseIntArg(argc, argv, i);
      i++;
      continue;
    }
    if (strcmp(argv[i], "-o") == 0) {
      if ((i + 1) >= argc) {
        printf("Missing output file name.\n");
        PrintUsage(argv[0]);
      }
      i++;
      g.output_image = argv[i];
      continue;
    }
    if (strcmp(argv[i], "-m") == 0) {
      g.iterations.max_escape_iterations = ParseIntArg(argc, argv, i);
      i++;
      continue;
    }
    if (strcmp(argv[i], "-c") == 0) {
      g.iterations.min_escape_iterations = ParseIntArg(argc, argv, i);
      i++;
      continue;
    }
    if (strcmp(argv[i], "-s") == 0) {
      g.iterations.samples_per_thread = ParseIntArg(argc, argv, i);
      i++;
      continue;
    }
    if (strcmp(argv[i], "-r") == 0) {
      resolution = ParseIntArg(argc, argv, i);
      i++;
      SetResolution(resolution, resolution);
      continue;
    }
    // Unrecognized argument, print the usage string.
    printf("Invalid argument: %s\n", argv[i]);
    PrintUsage(argv[0]);
  }
  if (!g.output_image) {
    printf("An output file name is required.\n");
    PrintUsage(argv[0]);
  }
}

int main(int argc, char **argv) {
  memset(&g, 0, sizeof(g));
  g.iterations.samples_per_thread = 1000;
  g.iterations.max_escape_iterations = 100;
  g.iterations.min_escape_iterations = 20;
  g.block_size = DEFAULT_BLOCK_SIZE;
  g.block_count = DEFAULT_BLOCK_COUNT;
  SetResolution(1000, 1000);
  g.cuda_device = USE_DEFAULT_DEVICE;
  ParseArguments(argc, argv);
  printf("Calculating image...\n");
  SetupCUDA();
  RenderImage();
  printf("Done! Saving image.\n");
  SaveImage();
  CleanupGlobals();
  return 0;
}
