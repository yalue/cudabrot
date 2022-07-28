#!/bin/bash

# This simple script shows how to generate three different Buddhabrot images,
# to combine into a single colored output image. You will need a few tools in
# order to use it:
#
# - imagemagick's "convert": Install using "sudo apt install imagemagick"
#
#    - Note on "convert": It has some limits that prevent it from using
#      extremely high-res images by default. To fix them, edit
#      /etc/ImageMagick-6/policy.xml and modify the "memory", "map", "width",
#      "height", "area", and "disk" settings to MUCH larger values. I set
#      "width" and "height" to "50KP", and all of the size-related settings to
#      "12GiB".
#
# - My image_combiner_hsl tool (maybe there's a way way to do the same thing
#   using "convert", but I haven't tried): github.com/yalue/image_combiner_hsl.
#   You can download a precompiled version for x86_64 Linux using:
#     wget https://github.com/yalue/image_combiner_hsl/releases/download/1.0/image_combiner_hsl
#     chmod +x image_combiner_hsl


# The "fine" image will be generated at 40k x 30k pixel resolution, then
# downsampled to the same size as the others. We'll let it run for 8 hours. It
# uses 60000 max iterations and 45000 min iterations to focus on details as
# much as possible.
./cudabrot -o fine_hires.pgm \
	-m 60000 -c 45000 \
	-w 20000 -h 15000 \
	--min-imag -1.5 --max-imag 1.5 \
	--min-real -2.0 --max-real 2.0 \
	-t $((60 * 60 * 12))
# ImageMagick's "normalize" does a much better job at getting nice looking
# brightness levels than guessing good gamma-correction values.
convert fine_hires.pgm -normalize -quality 100 fine_hires.jpg
# Now that we've converted to jpg, we can save space by removing the .pgm.
rm fine_hires.pgm

# The "medium" image will be generated at 20k x 15k resolution. It requires
# between 1000 and 8000 iterations to allow more smoothness and cloudiness.
# This renders faster than the "fine" version, especially at lower resolutions,
# so we'll make it run for 2 hours.
./cudabrot -o med_hires.pgm \
	-m 8000 -c 1000 \
	-w 20000 -h 15000 \
	--min-imag -1.5 --max-imag 1.5 \
	--min-real -2.0 --max-real 2.0 \
	-t $((60 * 60 * 4))
convert med_hires.pgm -normalize -quality 100 med_hires.jpg
rm med_hires.pgm

# The "coarse" image requires between 100 and 5 iterations, to get the larger
# shapes and clouds surrounding the set. Only renders for 1 hour.
./cudabrot -o coarse_hires.pgm \
	-m 500 -c 20 \
	-w 20000 -h 15000 \
	--min-imag -1.5 --max-imag 1.5 \
	--min-real -2.0 --max-real 2.0 \
	-t $((60 * 60 * 2))
convert coarse_hires.pgm -normalize -quality 100 coarse_hires.jpg
rm coarse_hires.pgm

# Either put the image_combiner_hsl binary in this directory, or replace this
# path (see the comment at the top). Feel free to experiment with different
# mappings to H, S, or L, or different hue adjustments.
./image_combiner_hsl \
	-H med_hires.jpg \
	-S coarse_hires.jpg \
	-L fine_hires.jpg \
	-adjust_hue 0.3 \
	-o combined.jpg

