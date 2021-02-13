/**
 * @author      Christoph Schaefer cm.schaefer@gmail.com
 *
 * @section     LICENSE
 * Copyright (c) 2019 Christoph Schaefer
 *
 * This file is part of miluphcuda.
 *
 * miluphcuda is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * miluphcuda is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef _IO_H
#define _IO_H

#include <stdio.h>
#include <unistd.h>
#include <libconfig.h>
#include "miluph.h"

typedef struct File {
	char	name[256];
	FILE	*data;
} File;

extern File inputFile;

extern int currentDiskIO;

extern void loadConfigFromFile(char *configFile);
extern void write_tree_to_file(File file);
extern void write_particles_to_file(File file);
extern void read_particles_from_file(File inputFile);
extern void *write_timestep(void *argument);
extern void write_performance(float *time);
extern void clear_performance_file();
extern void write_fragments_file();
extern void init_values();

void set_integration_parameters();

void copyToHostAndWriteToFile(int timestep, int lastTimestep);



#endif /* IO_H_ */
