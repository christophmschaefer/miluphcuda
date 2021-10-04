# miluphcuda makefile 0.01
#
CC = /usr/bin/g++-7
CFLAGS   = -c -std=c99 -O3 -DVERSION=\"$(GIT_VERSION)\" -fPIC
#CFLAGS   = -c -std=c99 -g
LDFLAGS  = -lm



GIT_VERSION := $(shell git describe --abbrev=4 --dirty --always --tags)

#CUDA_DIR    = /usr/local/cuda-9.0
#CUDA_DIR    = /usr/local/cuda-11.2
CUDA_DIR    = /usr/local/cuda

NVCC   = ${CUDA_DIR}/bin/nvcc

#The first NVFLAGS option is for when one wishes to use a debugger. Otherwise use the third option,
#NVFLAGS  = -ccbin ${CC} -x cu -c -dc  -G -lineinfo  -Xcompiler "-rdynamic -g -pthread"  -DVERSION=\"$(GIT_VERSION)\"
#NVFLAGS  = -x cu -c -dc -O3  -Xcompiler "-O3 -pthread" --ptxas-options=-v
NVFLAGS  = -ccbin ${CC} -x cu -c -dc -O3  -Xcompiler "-O3 -pthread" -Wno-deprecated-gpu-targets -DVERSION=\"$(GIT_VERSION)\"  --ptxas-options=-v

CUDA_LINK_FLAGS = -dlink
CUDA_LINK_OBJ = cuLink.o


# please make sure, that GPU_ARCH corresponds to your hardware
# otherwise the code does not work!
# gtx 570 and Quadro 4000
#GPU_ARCH = -arch=sm_20
# gtx titan and Kepler K40
#GPU_ARCH = -arch=sm_35
# Kepler K80
#GPU_ARCH = -arch=sm_37
# gtx 970 and GTX TITAN X
#GPU_ARCH = -arch=sm_52
# gtx 750 Ti
#GPU_ARCH = -arch=sm_50
# gtx 680 and gtx 770
#GPU_ARCH = -arch=sm_30
# gtx 1080
GPU_ARCH = -arch=sm_61

CUDA_LIB      = ${CUDA_DIR}
INCLUDE_DIRS += -I$(CUDA_LIB)/include -I/usr/include/hdf5/serial -I/usr/lib/openmpi/include/
LDFLAGS      += -ccbin ${CC} -L$(CUDA_LIB)/lib64 -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lcudart -lpthread -lconfig -lhdf5

# Default target
all: miluphcuda


# ---------------------------------------------------------
#         Set headers and object files
# ---------------------------------------------------------

HEADERS =
CUDA_HEADERS =  cuda_utils.h  checks.h io.h  miluph.h  parameter.h  timeintegration.h  tree.h  euler.h rk2adaptive.h pressure.h DISPH_pressure.h soundspeed.h device_tools.h boundary.h predictor_corrector.h predictor_corrector_euler.h memory_handling.h plasticity.h porosity.h aneos.h kernel.h linalg.h xsph.h density.h rhs.h internal_forces.h velocity.h damage.h little_helpers.h gravity.h viscosity.h artificial_stress.h stress.h extrema.h sinking.h coupled_heun_rk4_sph_nbody.h rk4_pointmass.h config_parameter.h
OBJ =
CUDA_OBJ = io.o  miluph.o  boundary.o timeintegration.o tree.o memory_handling.o euler.o rk2adaptive.o pressure.o soundspeed.o device_tools.o predictor_corrector.o predictor_corrector_euler.o plasticity.o porosity.o aneos.o kernel.o linalg.o xsph.o density.o rhs.o internal_forces.o velocity.o damage.o little_helpers.o gravity.o viscosity.o artificial_stress.o stress.o extrema.o sinking.o coupled_heun_rk4_sph_nbody.o rk4_pointmass.o config_parameter.o

documentation:
	cd doc && make all > .log

miluphcuda: $(OBJ) $(CUDA_OBJ)
#	$(NVCC) $(GPU_ARCH) $(CUDA_LINK_FLAGS) -o $(CUDA_LINK_OBJ) $(CUDA_OBJ)
	$(NVCC) $(GPU_ARCH) $(CUDA_OBJ) $(LDFLAGS) -Wno-deprecated-gpu-targets -o $@
#	$(CC) $(OBJ) $(CUDA_OBJ) $(CUDA_LINK_OBJ) $(LDFLAGS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDE_DIRS) $<

%.o: %.cu
	$(NVCC) $(GPU_ARCH) $(NVFLAGS) $(INCLUDE_DIRS) $<

.PHONY: clean
clean:
	@rm -f	*.o miluphcuda
	@echo make clean: done

# ---------------------------------------------------------
#          Dependencies for object files
# ---------------------------------------------------------

$(OBJ):  $(HEADERS) Makefile
$(CUDA_OBJ): $(HEADERS) $(CUDA_HEADERS) Makefile
