#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

// Uncomment this line to render the "Buddhabrot" version of the burning ship
// fractal rather than the regular Buddhabrot.
// #define RENDER_BURNING_SHIP (1)

// Controls the number of threads per block to use.
#define DEFAULT_BLOCK_SIZE (512)

// Controls the default number of blocks to use.
#define DEFAULT_BLOCK_COUNT (512)

// The name given to the output file if one isn't specified.
#define DEFAULT_OUTPUT_NAME "output.pgm"

// This macro takes a cudaError_t value and exits the program if it isn't equal
// to cudaSuccess.
#define CheckCUDAError(val) (InternalCUDAErrorCheck((val), #val, __FILE__, __LINE__))

// Increasing this may increase efficiency, but decrease responsiveness to
// signals.
#define SAMPLES_PER_THREAD (50)

// The RNG seed used when initializing the RNG states on the GPU.
#define DEFAULT_RNG_SEED (1337)

// The type that we use to keep track of internal pixel counts. Must be a
// numerical type that will work with both IncrementPixelCounter,
// GetLinearColorScale, and DoGammaCorrection. Floating-point values *ought* to
// work here, too.
typedef uint32_t Pixel;

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
// when calculating the Buddhabrot.
typedef struct {
  // This is the maximum number of iterations to run to see if a point escapes.
  int max_escape_iterations;
  // If a point escapes in fewer than this many iterations, it will be ignored.
  int min_escape_iterations;
} IterationControl;

// Holds global state in a single struct.
static struct {
  // The CUDA device to use. Defaults to 0.
  int cuda_device;
  // This tracks the random number generator states for the GPU code.
  curandState_t *rng_states;
  // The number of threads and blocks to use when calculating the Buddhabrot.
  int block_size, block_count;
  // The filename to which a bitmap image will be saved, or NULL if an image
  // should not be saved.
  const char *output_image;
  // The filename from which to load an in-progress image buffer, or to which
  // the in-progress buffer should be stored if operation is interrupted.
  const char *inprogress_file;
  // The number of seconds to run the calculation. If negative, run
  // indefinitely until a signal is received.
  double seconds_to_run;
  // If this is nonzero, the program should save the image and quit as soon as
  // the current iteration finishes.
  int quit_signal_received;
  // Holds various iteration-related settings.
  IterationControl iterations;
  // The size and location of the fractal and output image.
  FractalDimensions dimensions;
  // The host and device buffers which contain the numbers of times an escaping
  // point's path crossed each point in the complex plane.
  Pixel *device_buddhabrot;
  Pixel *host_buddhabrot;
  // The gamma value for gamma correction.
  double gamma_correction;
  // Buffer for a single grayscale image.
  uint16_t *grayscale_image;
} g;

// Returns the size, in bytes, of the internal image buffer used to hold the
// pixel data.
static uint64_t GetImageBufferSize(void) {
  return ((uint64_t) g.dimensions.w) * ((uint64_t) g.dimensions.h) *
    sizeof(Pixel);
}

// If any globals have been initialized, this will free them. (Relies on
// globals being set to 0 at the start of the program)
static void CleanupGlobals(void) {
  cudaFree(g.rng_states);
  cudaFree(g.device_buddhabrot);
  cudaFree(g.rng_states);
  free(g.grayscale_image);
  free(g.host_buddhabrot);
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
// CheckCUDAError macro.
static void InternalCUDAErrorCheck(cudaError_t result, const char *fn,
    const char *file, int line) {
  if (result == cudaSuccess) return;
  printf("CUDA error %d (%s) in %s, line %d (%s)\n", (int) result,
    cudaGetErrorString(result), file, line, fn);
  CleanupGlobals();
  exit(1);
}

// This function is used to initialize the RNG states to use when generating
// starting points in the Buddhabrot calculation. The states array must hold
// one entry for every thread in every block.
__global__ void InitializeRNG(uint64_t seed, curandState_t *states) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  curand_init(seed, index, 0, states + index);
}

// Allocates CUDA memory and calculates block/grid sizes. Must be called after
// g.w and g.h have been set.
static void SetupCUDA(void) {
  float gpu_memory_needed, cpu_memory_needed;
  CheckCUDAError(cudaSetDevice(g.cuda_device));
  size_t pixel_count = g.dimensions.w * g.dimensions.h;
  // The GPU will need space for the image and the RNG states.
  gpu_memory_needed = GetImageBufferSize() +
    (g.block_size * g.block_count * sizeof(curandState_t));
  gpu_memory_needed /= (1024.0 * 1024.0);
  // The CPU needs space for the image and grayscale conversion.
  cpu_memory_needed = GetImageBufferSize() + (pixel_count * sizeof(uint16_t));
  cpu_memory_needed /= (1024.0 * 1024.0);
  printf("Approximate memory needed: %.03f MiB GPU, %.03f MiB CPU\n",
    gpu_memory_needed, cpu_memory_needed);

  // Initialize the host and device image buffers.
  CheckCUDAError(cudaMalloc(&(g.device_buddhabrot), GetImageBufferSize()));
  CheckCUDAError(cudaMemset(g.device_buddhabrot, 0, GetImageBufferSize()));
  g.host_buddhabrot = (Pixel *) calloc(1, GetImageBufferSize());
  if (!g.host_buddhabrot) {
    CleanupGlobals();
    exit(1);
  }

  // Initialize the RNG state for the device.
  CheckCUDAError(cudaMalloc(&(g.rng_states), g.block_size * g.block_count *
    sizeof(curandState_t)));
  InitializeRNG<<<g.block_size, g.block_count>>>(DEFAULT_RNG_SEED,
    g.rng_states);
  CheckCUDAError(cudaDeviceSynchronize());

  g.grayscale_image = (uint16_t *) calloc(pixel_count, sizeof(uint16_t));
  if (!g.grayscale_image) {
    printf("Failed allocating grayscale image.\n");
    CleanupGlobals();
    exit(1);
  }
}

// Returns the size, in bytes, of f. Exits on error.
static uint64_t GetFileSize(FILE *f) {
  int64_t to_return;
  if (fseek(f, 0, SEEK_END) != 0) {
    printf("Failed seeking file end: %s\n", strerror(errno));
    CleanupGlobals();
    exit(1);
  }
  to_return = ftell(f);
  if (to_return < 0) {
    printf("Failed reading file size: %s\n", strerror(errno));
    CleanupGlobals();
    exit(1);
  }
  if (fseek(f, 0, SEEK_SET) != 0) {
    printf("Failed seeking file start: %s\n", strerror(errno));
    CleanupGlobals();
    exit(1);
  }
  return to_return;
}

// Loads the in-progress buffer from a file, if the file exists. Exits if an
// error occurs.  Creates the file if it doesn't exist.
static void LoadInProgressBuffer(void) {
  uint64_t expected_size, file_size;
  FILE *f = NULL;
  if (!g.inprogress_file) return;

  // We won't consider it an error if the file doesn't exist, and won't try to
  // load its contents.
  f = fopen(g.inprogress_file, "rb");
  expected_size = GetImageBufferSize();
  printf("Loading previous image state from %s.\n",
    g.inprogress_file);
  if (!f) {
    if (errno == ENOENT) {
      printf("File %s doesn't exist yet. Not loading.\n",
        g.inprogress_file);
      return;
    }
    printf("Failed opening %s: %s\n", g.inprogress_file, strerror(errno));
    CleanupGlobals();
    exit(1);
  }
  file_size = GetFileSize(f);

  // Ensure the file matches the expected size of our image buffer.
  if (file_size != expected_size) {
    printf("The size of %s doesn't match the expected size of %lu bytes.\n",
      g.inprogress_file, (unsigned long) expected_size);
    fclose(f);
    CleanupGlobals();
    exit(1);
  }

  // Read the file to the local buffer, then update the device copy.
  if (fread(g.host_buddhabrot, expected_size, 1, f) != 1) {
    printf("Failed reading %s: %s\n", g.inprogress_file, strerror(errno));
    fclose(f);
    CleanupGlobals();
    exit(1);
  }
  fclose(f);
  f = NULL;
  CheckCUDAError(cudaMemcpy(g.device_buddhabrot, g.host_buddhabrot,
    expected_size, cudaMemcpyHostToDevice));
}

// Saves the in-progress buffer to a file, if the filename was specified.
// Exits if an error occurs.
static void SaveInProgressBuffer(void) {
  FILE *f = NULL;
  if (!g.inprogress_file) return;
  printf("Saving in-progress buffer to %s.\n", g.inprogress_file);
  f = fopen(g.inprogress_file, "wb");
  if (!f) {
    printf("Failed opening %s: %s\n", g.inprogress_file, strerror(errno));
    CleanupGlobals();
    exit(1);
  }
  if (fwrite(g.host_buddhabrot, GetImageBufferSize(), 1, f) != 1) {
    printf("Failed writing data to %s: %s\n", g.inprogress_file,
      strerror(errno));
    fclose(f);
    CleanupGlobals();
    exit(1);
  }
  fclose(f);
}

// This returns nonzero if the given point is in the main cardioid of the set
// and is therefore guaranteed to not escape.
inline __device__ int InMainCardioid(double real, double imag) {
  // This algorithm was taken from the Wikipedia Mandelbrot set page.
  double imag_squared = imag * imag;
  double q = (real - 0.25);
  q = q * q + imag_squared;
  return q * (q + (real - 0.25)) < (imag_squared * 0.25);
}

// This returns nonzero if the given point is in the order 2 bulb of the set
// and therefore guaranteed to not escape.
inline __device__ int InOrder2Bulb(double real, double imag) {
  double tmp = real + 1;
  tmp = tmp * tmp;
  return (tmp + (imag * imag)) < (1.0 / 16.0);
}

// This should be used to update the pixel data for a point that is encountered
// in the set.
inline __device__ void IncrementPixelCounter(double real, double imag,
    Pixel *data, FractalDimensions *d) {
  int row, col;
  // There's a small issue here with integer-dividing where values that should
  // be immediately outside of the canvas can still appear on row or col 0, so
  // just return early if we're outside the boundary.
  if ((real < d->min_real) || (imag < d->min_imag)) return;
  col = (real - d->min_real) / d->delta_real;
  row = (imag - d->min_imag) / d->delta_imag;
  if ((row >= 0) && (row < d->h) && (col >= 0) && (col < d->w)) {
    data[(row * d->w) + col] += 1;
  }
}

// Does the Mandelbrot-set iterations for the given (real, imag) point. Returns
// the number of iterations before the point escapes, or max_iterations if the
// point never escapes.
inline __device__ int IterateMandelbrot(double start_real, double start_imag,
    int max_iterations) {
  double tmp, real, imag;
  int i;
  real = start_real;
  imag = start_imag;

  for (i = 0; i < max_iterations; i++) {
#ifdef RENDER_BURNING_SHIP
    real = fabs(real);
    imag = fabs(imag);
#endif
    tmp = (real * real) - (imag * imag) + start_real;
    imag = 2 * real * imag + start_imag;
    real = tmp;
    // If the point escapes, stop iterating and indicate the loop ended due
    // to the point escaping.
    if (((real * real) + (imag * imag)) > 4) return i;
  }
  // The point didn't escape, return max_iterations.
  return max_iterations;
}

// Like IterateMandelbrot, but records the point's path. For efficiency, this
// function also has an important difference from IterateMandelbrot: *it does
// not check the max iterations*. This is important! Do not call this function
// for a point unless you're sure that it escapes in a finite number of
// iterations.
inline __device__ void IterateAndRecord(double start_real, double start_imag,
    Pixel *data, FractalDimensions *d) {
  double tmp, real, imag;
  real = start_real;
  imag = start_imag;
  while (1) {
#ifdef RENDER_BURNING_SHIP
    real = fabs(real);
    imag = fabs(imag);
#endif
    tmp = (real * real) - (imag * imag) + start_real;
    imag = 2 * real * imag + start_imag;
    real = tmp;
    IncrementPixelCounter(real, imag, data, d);
    // Stop iterating when the point escapes. This must be *guaranteed* to
    // happen by the caller performing a prior check!
    if (((real * real) + (imag * imag)) > 4) break;
  }
}


// This kernel is responsible for drawing the paths of "particles" that escape
// the mandelbrot set. It works as follows:
//
// 1. For each "sample", compute a new random starting point in the complex
//    plane
// 2. Do the normal mandelbrot iterations on the starting point, *without*
//    recording its path
// 3. If the point didn't escape the path, take a new sample (return to step 1)
// 4. If the point escaped (within the min and max iteration limits), then
//    repeat the mandelbrot iterations (e.g. step 2), except record its path
//    by incrementing the pixel value for every point it passes through.
__global__ void DrawBuddhabrot(FractalDimensions dimensions, Pixel *data,
    IterationControl iterations, curandState_t *states) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  curandState_t *rng = states + index;
  int sample, iterations_needed, max_iterations, min_iterations;
  double real, imag;
  max_iterations = iterations.max_escape_iterations;
  min_iterations = iterations.min_escape_iterations;

  // We're going to pick a number of random starting points determined by the
  // SAMPLES_PER_THREAD value.
  for (sample = 0; sample < SAMPLES_PER_THREAD; sample++) {
    // Sample across the entire domain of the set regardless of our "canvas"
    real = (curand_uniform_double(rng) * 4.0) - 2.0;
    imag = (curand_uniform_double(rng) * 4.0) - 2.0;

    // Optimization: we know ahead of time that points from the main cardioid
    // and the largest "bulb" will never escape, and it's fast to check them.
#ifndef RENDER_BURNING_SHIP
    if (InMainCardioid(real, imag) || InOrder2Bulb(real, imag)) continue;
#endif

    // Now, do the normal Mandelbrot iterations to see how quickly the point
    // escapes (if it does). However, we won't record the path yet.
    iterations_needed = IterateMandelbrot(real, imag, max_iterations);

    // Don't record the path if the point never escaped, or if it escaped too
    // quickly.
    if (iterations_needed >= max_iterations)  continue;
    if (iterations_needed < min_iterations) continue;

    // At this point, do the Mandelbrot iterations, but actually record the
    // path because we know the point is "good".
    IterateAndRecord(real, imag, data, &dimensions);
  }
}

static uint16_t Clamp(double v) {
  if (v <= 0) return 0;
  if (v >= 0xffff) return 0xffff;
  return (uint16_t) v;
}

// Returns the amount to multiply the original count by in order to get a value
// by which Buddhabrot counts can be multiplied to get a number between 0 and
// 0xffff.
static double GetLinearColorScale(void) {
  int x, y, index;
  Pixel max = 0;
  double to_return;
  index = 0;
  for (y = 0; y < g.dimensions.h; y++) {
    for (x = 0; x < g.dimensions.w; x++) {
      if (g.host_buddhabrot[index] > max) max = g.host_buddhabrot[index];
      index++;
    }
  }
  to_return = ((double) 0xffff) / ((double) max);
  printf("Max value: %lu, scale: %f\n", (unsigned long) max, to_return);
  return to_return;
}

// Returns the gamma-corrected 16-bit color channel value given a Buddhabrot
// iteration count c.
static uint16_t DoGammaCorrection(Pixel c, double linear_scale) {
  double max = 0xffff;
  double scaled = ((double) c) * linear_scale;
  // Don't do gamma correction if the gamma correction argument was negative.
  if (g.gamma_correction <= 0.0) return scaled;
  return Clamp(max * pow(scaled / max, 1 / g.gamma_correction));
}

// Converts the buffer of pixel values to a gamma-corrected grayscale image
// with 16-bit colors. The Pixel values are scaled to fill the 16-bit color
// range.
static void SetGrayscalePixels(void) {
  int x, y;
  uint16_t color_value;
  double linear_scale = GetLinearColorScale();
  Pixel *host_data = g.host_buddhabrot;
  uint16_t *grayscale = g.grayscale_image;
  for (y = 0; y < g.dimensions.h; y++) {
    for (x = 0; x < g.dimensions.w; x++) {
      color_value = DoGammaCorrection(*host_data, linear_scale);
      *grayscale = color_value;
      grayscale++;
      host_data++;
    }
  }
}

// Renders the fractal image.
static void RenderImage(void) {
  int passes_count = 0;
  double start_seconds;
  printf("Calculating Buddhabrot.\n");
  if (g.seconds_to_run < 0) {
    printf("Press ctrl+C to finish.\n");
  } else {
    printf("Running for %.03f seconds.\n", g.seconds_to_run);
  }

  // Run until either the time elapsed or we've received a SIGINT.
  start_seconds = CurrentSeconds();
  while (!g.quit_signal_received) {
    passes_count++;
    DrawBuddhabrot<<<g.block_count, g.block_size>>>(g.dimensions,
      g.device_buddhabrot, g.iterations, g.rng_states);
    CheckCUDAError(cudaDeviceSynchronize());
    if ((g.seconds_to_run >= 0) && ((CurrentSeconds() - start_seconds) >
      g.seconds_to_run)) {
      break;
    }
  }

  // Copy the resulting image to CPU memory, and convert the pixels to proper
  // grayscale values.
  CheckCUDAError(cudaMemcpy(g.host_buddhabrot, g.device_buddhabrot,
    GetImageBufferSize(), cudaMemcpyDeviceToHost));
  printf("%d Buddhabrot passes took %f seconds.\n", passes_count,
    CurrentSeconds() - start_seconds);
  SetGrayscalePixels();
}

// Recomputes the spacing between pixels in the image. Returns 0 if any image-
// dimension setting is invalid. Otherwise, returns 1.
static int RecomputePixelDeltas(void) {
  FractalDimensions *dims = &(g.dimensions);
  if (dims->w <= 0) {
    printf("Output width must be positive.\n");
    return 0;
  }
  if (dims->h <= 0) {
    printf("Output height must be positive.\n");
    return 0;
  }
  if (dims->max_real <= dims->min_real) {
    printf("Maximum real value must be greater than minimum real value.\n");
    return 0;
  }
  if (dims->max_imag <= dims->min_imag) {
    printf("Minimum imaginary value must be greater than maximum imaginary "
      "value.\n");
    return 0;
  }
  dims->delta_imag = (dims->max_imag - dims->min_imag) / ((double) dims->h);
  dims->delta_real = (dims->max_real - dims->min_real) / ((double) dims->w);
  return 1;
}

// Sets the image boundaries and dimensions to their default values.
static void SetDefaultCanvas(void) {
  FractalDimensions *dims = &(g.dimensions);
  memset(dims, 0, sizeof(*dims));
  dims->w = 1000;
  dims->h = 1000;
  dims->min_real = -2.0;
  dims->max_real = 2.0;
  dims->min_imag = -2.0;
  dims->max_imag = 2.0;
  if (!RecomputePixelDeltas()) {
    printf("Internal error setting default canvas boundaries!\n");
    exit(1);
  }
}

// If a filename has been set for saving the image, this will attempt to save
// the image to the file. This can modify the image buffer! (For changing byte
// order.)
static void SaveImage(void) {
  uint16_t tmp;
  int i;
  int pixel_count = g.dimensions.w * g.dimensions.h;
  FILE *output = fopen(g.output_image, "wb");
  if (!output) {
    printf("Failed opening output image.\n");
    return;
  }
  if (fprintf(output, "P5\n%d %d\n%d\n", g.dimensions.w, g.dimensions.h,
    0xffff) <= 0) {
    printf("Failed writing pgm header.\n");
    fclose(output);
    return;
  }
  // Flip the byte-order for the image. This assumes the program is running on
  // a little-endian architecture. I'll fix it if there's ever a demand to run
  // this on something other than Linux on x86 or ARM64 (lol).
  for (i = 0; i < pixel_count; i++) {
    tmp = g.grayscale_image[i];
    tmp = ((tmp & 0xff) << 8) | (tmp >> 8);
    g.grayscale_image[i] = tmp;
  }
  if (!fwrite(g.grayscale_image, pixel_count * sizeof(uint16_t), 1, output)) {
    printf("Failed writing pixel data.\n");
    fclose(output);
    return;
  }
  fclose(output);
}

static void PrintUsage(char *program_name) {
  printf("Usage: %s [options]\n\n", program_name);
  printf("Options may be one or more of the following:\n"
    "  --help: Prints these instructions.\n"
    "  -d <device number>: Sets which GPU to use. Defaults to GPU 0.\n"
    "  -o <output file name>: If provided, the rendered image will be saved\n"
    "     to a .pgm file with the given name. Otherwise, saves the image\n"
    "     to " DEFAULT_OUTPUT_NAME ".\n"
    "  -m <max escape iterations>: The maximum number of iterations to use\n"
    "     before giving up on seeing whether a point escapes.\n"
    "  -c <min escape iterations>: If a point escapes before this number of\n"
    "     iterations, it will be ignored.\n"
    "  -g <gamma correction>: A gamma-correction value to use on the\n"
    "     resulting image. If negative, no gamma correction will occur.\n"
    "  -t <seconds to run>: A number of seconds to run the calculation for.\n"
    "     Defaults to 10.0. If negative, the program will run continuously\n"
    "     and will terminate (saving the image) when it receives a SIGINT.\n"
    "  -w <width>: The width of the output image, in pixels. Defaults to\n"
    "     1000.\n"
    "  -h <height>: The height of the output image, in pixels. Defaults to\n"
    "     1000.\n"
    "  -s <save/load file>: If provided, this gives a file name into which\n"
    "     the rendering buffer will be saved, for future continuation.\n"
    "     If the program is loaded and the file exists, the buffer will be\n"
    "     filled with the contents of the file, but the dimensions must\n"
    "     match. Note that this file may be huge for high-resolution images.\n"
    "\n"
    "The following settings control the location of the output image on the\n"
    "complex plane, but samples are always drawn from the entire Mandelbrot-\n"
    "set domain (-2-2i to 2+2i). So these settings can be used to save\n"
    "memory or \"crop\" the output, but won't otherwise speed up rendering:\n"
    "  --min-real <min real>: The minimum value along the real axis to\n"
    "             include in the output image. Defaults to -2.0.\n"
    "  --max-real <max real>: The maximum value along the real axis to\n"
    "             include in the output image. Defaults to 2.0.\n"
    "  --min-imag <min imag>: The minimum value along the imaginary axis to\n"
    "             include in the output image. Defaults to -2.0.\n"
    "  --max-imag <max imag>: The maximum value along the imaginary axis to\n"
    "             include in the output image. Defaults to 2.0.\n"
    "");
  exit(0);
}

// Returns an integer at the argument after index in argv. Exits if the integer
// is invalid. Takes the index before the expected int value in order to print
// better error messages.
static int ParseIntArg(int argc, char **argv, int index) {
  char *tmp = NULL;
  int to_return = 0;
  if ((index + 1) >= argc) {
    printf("Argument %s needs a value.\n", argv[index]);
    PrintUsage(argv[0]);
  }
  to_return = strtol(argv[index + 1], &tmp, 10);
  // Make sure that, if tmp is a null character, that the argument wasn't
  // simply a string with no content.
  if ((*tmp != 0) || (argv[index + 1][0] == 0)) {
    printf("Invalid number given to argument %s: %s\n", argv[index],
      argv[index + 1]);
    PrintUsage(argv[0]);
  }
  return to_return;
}

// Like ParseIntArg, except expects a floating-point double arg.
static double ParseDoubleArg(int argc, char **argv, int index) {
  char *tmp = NULL;
  double to_return = 0.0;
  if ((index + 1) >= argc) {
    printf("Argument %s needs a value.\n", argv[index]);
    PrintUsage(argv[0]);
  }
  to_return = strtod(argv[index + 1], &tmp);
  if ((*tmp != 0) || (argv[index + 1][0] == 0)) {
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
    if (strcmp(argv[i], "-o") == 0) {
      if ((i + 1) >= argc) {
        printf("Missing output file name.\n");
        PrintUsage(argv[0]);
      }
      i++;
      g.output_image = argv[i];
      continue;
    }
    if (strcmp(argv[i], "-s") == 0) {
      if ((i + 1) >= argc) {
        printf("Missing in-progress buffer file name.\n");
        PrintUsage(argv[0]);
      }
      i++;
      g.inprogress_file = argv[i];
      continue;
    }
    if (strcmp(argv[i], "-m") == 0) {
      g.iterations.max_escape_iterations = ParseIntArg(argc, argv, i);
      if (g.iterations.max_escape_iterations > 60000) {
        printf("Warning: Using a high number of iterations may cause the "
          "program respond slowly to Ctrl+C or time running out.\n");
      }
      i++;
      continue;
    }
    if (strcmp(argv[i], "-c") == 0) {
      g.iterations.min_escape_iterations = ParseIntArg(argc, argv, i);
      i++;
      continue;
    }
    if (strcmp(argv[i], "-w") == 0) {
      g.dimensions.w = ParseIntArg(argc, argv, i);
      if (!RecomputePixelDeltas()) PrintUsage(argv[0]);
      i++;
      continue;
    }
    if (strcmp(argv[i], "-h") == 0) {
      g.dimensions.h = ParseIntArg(argc, argv, i);
      if (!RecomputePixelDeltas()) PrintUsage(argv[0]);
      i++;
      continue;
    }
    if (strcmp(argv[i], "-g") == 0) {
      g.gamma_correction = ParseDoubleArg(argc, argv, i);
      i++;
      continue;
    }
    if (strcmp(argv[i], "-t") == 0) {
      g.seconds_to_run = ParseDoubleArg(argc, argv, i);
      i++;
      continue;
    }
    if (strcmp(argv[i], "--min-real") == 0) {
      g.dimensions.min_real = ParseDoubleArg(argc, argv, i);
      if (!RecomputePixelDeltas()) PrintUsage(argv[0]);
      i++;
      continue;
    }
    if (strcmp(argv[i], "--max-real") == 0) {
      g.dimensions.max_real = ParseDoubleArg(argc, argv, i);
      if (!RecomputePixelDeltas()) PrintUsage(argv[0]);
      i++;
      continue;
    }
    if (strcmp(argv[i], "--min-imag") == 0) {
      g.dimensions.min_imag = ParseDoubleArg(argc, argv, i);
      if (!RecomputePixelDeltas()) PrintUsage(argv[0]);
      i++;
      continue;
    }
    if (strcmp(argv[i], "--max-imag") == 0) {
      g.dimensions.max_imag = ParseDoubleArg(argc, argv, i);
      if (!RecomputePixelDeltas()) PrintUsage(argv[0]);
      i++;
      continue;
    }
    // Unrecognized argument, print the usage string.
    printf("Invalid argument: %s\n", argv[i]);
    PrintUsage(argv[0]);
  }
}

void SignalHandler(int signal_number) {
  g.quit_signal_received = 1;
  printf("Signal %d received, waiting for current pass to finish...\n",
    signal_number);
}

int main(int argc, char **argv) {
  memset(&g, 0, sizeof(g));
  g.output_image = DEFAULT_OUTPUT_NAME;
  g.iterations.max_escape_iterations = 100;
  g.iterations.min_escape_iterations = 20;
  g.block_size = DEFAULT_BLOCK_SIZE;
  g.block_count = DEFAULT_BLOCK_COUNT;
  g.seconds_to_run = 10.0;
  g.gamma_correction = 1.0;
  SetDefaultCanvas();
  g.cuda_device = 0;
  ParseArguments(argc, argv);
  if (signal(SIGINT, SignalHandler) == SIG_ERR) {
    printf("Failed setting signal handler.\n");
    CleanupGlobals();
    return 1;
  }
  printf("Creating %dx%d image, %d max iterations.\n",
    g.dimensions.w, g.dimensions.h, g.iterations.max_escape_iterations);
  printf("Calculating image...\n");
  SetupCUDA();
  LoadInProgressBuffer();
  RenderImage();
  SaveInProgressBuffer();
  printf("Saving image.\n");
  SaveImage();
  printf("Done! Output image saved: %s\n", g.output_image);
  CleanupGlobals();
  return 0;
}
