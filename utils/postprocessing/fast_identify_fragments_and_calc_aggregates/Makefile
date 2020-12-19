# This is the Makefile for fast_identify_fragments and calc_aggregates

# set compiler
CC = gcc

OPT = -O3 -Wall

CFLAGS = -I/usr/include/hdf5/serial/
LDFLAGS = -lm -L/usr/lib/x86_64-linux-gnu/hdf5/serial/ -lhdf5


# default target, build both
all: fast_identify_fragments calc_aggregates


fast_identify_fragments: fast_identify_fragments.c Makefile
	@echo ""
	@echo "Compiling fast_identify_fragments ..."
	$(CC) $@.c $(OPT) $(CFLAGS) $(LDFLAGS) -o $@
	@echo "fast_identify_fragments built successfully."


calc_aggregates: calc_aggregates.c Makefile
	@echo ""
	@echo "Compiling calc_aggregates ..."
	$(CC) $@.c $(OPT) $(CFLAGS) $(LDFLAGS) -o $@
	@echo "calc_aggregates built successfully."


clean:
	@echo "Cleaning up ..."
	@rm -vf fast_identify_fragments calc_aggregates
