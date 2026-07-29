# This is the miluphcuda Makefile.

CC = /usr/bin/g++

# directory layout
SRC_DIR   = src
INC_DIR   = include
BUILD_DIR = build
OBJ_DIR   = $(BUILD_DIR)/obj
TARGET    = $(BUILD_DIR)/miluphcuda

CFLAGS   = -c -std=c99 -O3 -DVERSION=\"$(GIT_VERSION)\" -fPIC
#CFLAGS   = -c -std=c99 -g
LDFLAGS  = -lm

GIT_VERSION := $(shell git describe --abbrev=4 --dirty --always --tags)

CUDA_DIR    = /usr/local/cuda-12.8

NVCC   = ${CUDA_DIR}/bin/nvcc

# for using a debugger use the first, otherwise the third:
#NVFLAGS  = -ccbin ${CC} -x cu -c -dc  -G -lineinfo  -Xcompiler "-rdynamic -g -pthread"  -DVERSION=\"$(GIT_VERSION)\"
#NVFLAGS  = -x cu -c -dc -O3  -Xcompiler "-O3 -pthread" --ptxas-options=-v
NVFLAGS  = -ccbin ${CC} -x cu -c -dc -O3  -Xcompiler "-O3 -pthread" -Wno-deprecated-gpu-targets -DVERSION=\"$(GIT_VERSION)\"  --ptxas-options=-v

CUDA_LINK_FLAGS = -dlink
CUDA_LINK_OBJ = $(OBJ_DIR)/cuLink.o


# important: compute capability, corresponding to GPU model (e.g., -arch=sm_52 for 5.2)
GPU_ARCH = -arch=sm_61
# (very) incomplete list:
# compute capability    GPU models
#                2.0    GeForce GTX 570, Quadro 4000
#                3.0    GeForce GTX 680, GeForce GTX 770
#                3.5    GeForce GTX Titan, Tesla K40
#                3.7    Tesla K80
#                5.0    GeForce GTX 750 Ti
#                5.2    GeForce GTX 970, GeForce GTX Titan X
#                6.1    GeForce GTX 1080, GeForce GTX 1080 Ti


CUDA_LIB      = ${CUDA_DIR}
INCLUDE_DIRS += -I$(INC_DIR) -I$(CUDA_LIB)/include -I/usr/include/hdf5/serial -I/usr/lib/openmpi/include/
# if you use HDF5 I/O use the first, otherwise the second:
LDFLAGS      += -ccbin ${CC} -L$(CUDA_LIB)/lib64 -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lcudart -lpthread -lconfig -lhdf5
#LDFLAGS      += -ccbin ${CC} -L$(CUDA_LIB)/lib64 -lcudart -lpthread -lconfig


# default target
all: $(TARGET)

# headers and object files
HEADER_NAMES =
CUDA_HEADER_NAMES = cuda_utils.h  checks.h io.h  miluph.h  parameter.h  timeintegration.h  tree.h  euler.h rk2adaptive.h pressure.h soundspeed.h device_tools.h boundary.h predictor_corrector.h predictor_corrector_euler.h memory_handling.h plasticity.h porosity.h aneos.h kernel.h linalg.h xsph.h density.h rhs.h internal_forces.h velocity.h damage.h little_helpers.h gravity.h viscosity.h artificial_stress.h stress.h extrema.h sinking.h coupled_heun_rk4_sph_nbody.h rk4_pointmass.h config_parameter.h fast_integration.h

OBJ_NAMES =
CUDA_OBJ_NAMES = io.o  miluph.o  boundary.o timeintegration.o tree.o memory_handling.o euler.o rk2adaptive.o pressure.o soundspeed.o device_tools.o predictor_corrector.o predictor_corrector_euler.o plasticity.o porosity.o aneos.o kernel.o linalg.o xsph.o density.o rhs.o internal_forces.o velocity.o damage.o little_helpers.o gravity.o viscosity.o artificial_stress.o stress.o extrema.o sinking.o coupled_heun_rk4_sph_nbody.o rk4_pointmass.o config_parameter.o fast_integration.o

HEADERS      = $(addprefix $(INC_DIR)/,$(HEADER_NAMES))
CUDA_HEADERS = $(addprefix $(INC_DIR)/,$(CUDA_HEADER_NAMES))
OBJ          = $(addprefix $(OBJ_DIR)/,$(OBJ_NAMES))
CUDA_OBJ     = $(addprefix $(OBJ_DIR)/,$(CUDA_OBJ_NAMES))


documentation:
	cd doc && make all > .log

$(TARGET): $(OBJ) $(CUDA_OBJ) | $(BUILD_DIR)
#	$(NVCC) $(GPU_ARCH) $(CUDA_LINK_FLAGS) -o $(CUDA_LINK_OBJ) $(CUDA_OBJ)
	$(NVCC) $(GPU_ARCH) $(CUDA_OBJ) $(LDFLAGS) -Wno-deprecated-gpu-targets -o $@
#	$(CC) $(OBJ) $(CUDA_OBJ) $(CUDA_LINK_OBJ) $(LDFLAGS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) $(INCLUDE_DIRS) -o $@ $<

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(GPU_ARCH) $(NVFLAGS) $(INCLUDE_DIRS) -o $@ $<

$(BUILD_DIR) $(OBJ_DIR):
	@mkdir -p $@

.PHONY: all clean documentation
clean:
	@rm -rf	$(BUILD_DIR)
	@echo make clean: done


# dependencies for object files
$(OBJ):  $(HEADERS) Makefile
$(CUDA_OBJ): $(HEADERS) $(CUDA_HEADERS) Makefile
