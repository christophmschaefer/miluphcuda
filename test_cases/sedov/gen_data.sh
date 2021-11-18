#!/bin/bash
g++ --std=c++11 `pkg-config --cflags cxxopts` sedov.cpp
rm -f sedov_kernel.output
./a.out -d 0.013 -r 0.5
rm -f sedov.0000
cp sedov_kernel.output sedov.0000
