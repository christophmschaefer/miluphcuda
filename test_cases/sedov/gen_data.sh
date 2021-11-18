#!/bin/bash
g++ --std=c++11 `pkg-config --cflags cxxopts` sedov.cpp
./a.out -d 0.013 -r 0.5
