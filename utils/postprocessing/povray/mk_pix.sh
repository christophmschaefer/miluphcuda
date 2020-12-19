#!/bin/bash
for i in $@; do 
    echo  $i
    test -f $i.png && continue
    ./h5gen_povray.py $i
    cat header.pov $i.pov > bla.pov 
    povray -d +W1280 +H1280 bla.pov > .povray_output 2>&1
    mv bla.png $i.png
done


