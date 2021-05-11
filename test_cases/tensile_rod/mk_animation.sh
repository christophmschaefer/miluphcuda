#!/bin/bash

# make plots
for FILE in `ls -1 rod.????.h5`; do
    if [ -f "$FILE.png" ]; then
        echo "$FILE already plotted."
    else
        echo "plotting $FILE..."
        ./mk_plot.py $FILE
    fi
done

# make video
ffmpeg -r 10 -f image2 -pattern_type glob -i "*.png" -r 10 animation.mp4
#ffmpeg -r 20 -i rod.%04d.h5.png -r 20 animation.mp4

