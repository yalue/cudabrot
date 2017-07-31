CUDAbrot: A "Buddhabrot" Renderer using CUDA
============================================

About
-----

This project contains a small CUDA program for rendering the Buddhabrot fractal
using a CUDA-capable GPU.

I'm aware that there are at least two other github projects named "cudabrot",
but both of them render the Mandelbrot set rather than the Buddhabrot. The
Buddhabrot set is a variant of the Mandelbrot set similar to an
[attractor](https://en.wikipedia.org/wiki/Attractor), and is generally more
processor-intensive to render. Therefore, rendering high-resolution Buddhabrot
images is an excellent application of GPU computing.

For more information on how the Buddhabrot set is rendered, see the
[Wikipedia article](https://en.wikipedia.org/wiki/Buddhabrot) for information
about the algorithm and the relationship to the Mandelbrot set.

Usage
-----

To compile and run this program, you need to be using a Linux system with CUDA
installed (the more recent the version, the better), and a CUDA-capable GPU.

Compile the program simply by running `make`. Run it by running `./cudabrot`.
A summary of command-line arguments can be obtained by running
`./cudabrot --help`. Running the program will produce a single grayscale image.
Typically, a colored Buddhabrot image is created by rendering several single-
channel images with different parameters, then combining the results by
assigning each single-channel image to a color in the output image.

Examples and detailed description of options
--------------------------------------------

All examples below were rendered using an NVIDIA GTX 970 with 4GB of GPU RAM.

 - `-d <device number>`: Example: `./cudabrot -d 0`. If you have more than one
   GPU, providing the `-d` flag along with a device number allows you to run
   computations on a GPU of your choosing. If the `-d` flag isn't specified,
   the program defaults to using GPU 0.

 - `-o <output file name>`: Example: `./cudabrot -o image.pgm`. This program is
   capable only of saving `.pgm`-format images, which are a simple grayscale
   bitmap format. Output images always use 16-bit grayscale. If left
   unspecified, the program will save the image to a file named `output.pgm` by
   default.

 - `-r <resolution>`: Example: `./cudabrot -r 10000`. The `-r` flag controls
   the resolution of the output image. Since the Buddhabrot is always drawn on
   a square canvas, this option specifies the number of pixels along a single
   side of the square. Increasing the resolution won't slow down the program,
   but it *will* increase the amount of GPU and CPU memory required. For
   example, rendering a 20000x20000 image (`-r 20000`) takes at least 3 GB of
   GPU memory, so higher resolutions may only be possible with more-capable
   GPUs.

 - `-t <time to run (in seconds)>`: Example: `./cudabrot -t 60`. This option
   specifies the amount of time, in seconds, to run the rendering on the GPU.
   The longer the time, the sharper the image will appear (especially at high
   resolutions or number of iterations). Passing a special value of -1 to `-t`
   will cause the program to run until it is interrupted by the user (using
   `kill` or CTRL+C on Linux, for example). Example: `./cudabrot -t -1`. If the
   program is run with `-t -1` and killed by the user, it will save the
   currently-rendered output image. This is the recommended way to run the
   program, if, for example, you want to render an image overnight. This option
   defaults to 10 seconds.

 - `-g <gamma correction>`: Example: `./cudabrot -g 2.0`. This option specifies
   the amount of gamma correction to be applied post-rendering. Gamma
   correction brightens darker areas of the image, which enhances the
   visibility of some details. In most cases, it may be easier to apply gamma
   correction post-rendering using a separate image editor (where changes can
   be previewed), but this option is available for convenience and scripting.
   This option defaults to 1.0 (no gamma correction).
   Example images:

    | `./cudabrot -r 200 -m 10000 -c 8000 -t 30 -g 1.0` | `./cudabrot -r 200 -m 10000 -c 8000 -t 30 -g 1.5` | `./cudabrot -r 200 -m 10000 -c 8000 -t 30 -g 2.2` |
    | :---: | :---: | :---: |
    | ![No gamma correction](examples/gamma_1_0.png) | ![1.5 gamma](examples/gamma_1_5.png) | ![2.2 gamma](examples/gamma_2_2.png) |

 - `-m <max escape iterations>`: Example: `./cudabrot -m 10000`. This option
   specifies the maximum iterations to follow each particle before determining
   whether it remains in the Mandelbrot set (meaning that its path is included
   in the Buddhabrot set). In short, increasing this value will include more
   fine details in the resulting image. This value defaults to 100, which is a
   fairly low value. See these examples:

    | `./cudabrot -r 200 -t 10 -c 20 -m 100` | `./cudabrot -r 200 -t 10 -c 20 -m 1000` | `./cudabrot -r 200 -t 10 -c 20 -m 20000` |
    | :---: | :---: | :---: |
    | ![Low max iterations](examples/max_100.png) | ![Mid max iterations](examples/max_1000.png) | ![High max iterations](examples/max_20000.png) |

 - `-c <min escape iterations`: Example: `./cudabrot -m 5000 -c 4000`. This
   option specifies the minimum cutoff for the number of iterations for which
   points must *remain* in the Mandelbrot set if they are to be included in
   the Buddhabrot. Increasing the minimum cutoff iterations will therefore
   reduce the "cloudiness" of the generated image, enhancing the visibility of
   the details produced using higher `-m` values. This value defaults to 20,
   which will produce a cloudy, nebulous image. See these examples:

    | `./cudabrot -r 200 -t 30 -g 1.8 -m 20000 -c 20` | `./cudabrot -r 200 -t 30 -g 1.8 -m 20000 -c 2000` | `./cudabrot -r 200 -t 30 -g 1.8 -m 20000 -c 10000` |
    | :---: | :---: | :---: |
    | ![Low cutoff](examples/cutoff_20.png) | ![Mid cutoff](examples/cutoff_2000.png) | ![High cutoff](examples/cutoff_10000.png) |

Coloring the Buddhabrot
-----------------------

The Buddhabrot rendering maps most easily to grayscale images, so coloring is
left to post-processing. The "traditional" way to color a Buddhabrot is to
generate several grayscale images using different minimum and maximum iteration
values (the `-r` and `-c` options in this program). The grayscale images can
then be combined into a single output image, with each grayscale image
contributing to a different color channel in the output.

A free program that can be used to combine grayscale images into a single color
image exists [in a separate repository](https://github.com/yalue/image_combiner).

Here's an example of how to create a color image, using the `image_combiner`
tool linked above:

```bash
./cudabrot -g 2.0 -r 1000 -m 100 -c 20 -t 20 -o low_iterations.pgm
./cudabrot -g 2.0 -r 1000 -m 2000 -c 600 -t 20 -o mid_iterations.pgm
./cudabrot -g 2.5 -r 1000 -m 10000 -c 9000 -t 40 -o high_iterations.pgm
./image_combiner \
    low_iterations.pgm blue \
    mid_iterations.pgm lime \
    high_iterations.pgm red \
    color_output.jpg
```

The above commands result in this colored image:
![color Buddhabrot](examples/color_output.jpg)
