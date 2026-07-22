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


int init_allocate_memory(void);
int copy_particle_data_to_device(void);
int free_memory(void);
int copy_particles_variables_device_to_device(struct Particle *, struct Particle *);
int copy_pointmass_variables_device_to_device(struct Pointmass *, struct Pointmass *);
int copy_particles_derivatives_device_to_device(struct Particle *, struct Particle *);
int copy_pointmass_derivatives_device_to_device(struct Pointmass *, struct Pointmass *);
int copy_particles_immutables_device_to_device(struct Particle *, struct Particle *);
int copy_pointmass_immutables_device_to_device(struct Pointmass *, struct Pointmass *);
int copy_gravitational_accels_device_to_device(struct Particle *, struct Particle *);
int allocate_particles_memory(struct Particle *, int allocate_immutables);
int allocate_pointmass_memory(struct Pointmass *, int allocate_immutables);
int free_particles_memory(struct Particle *, int free_immutables);
int free_pointmass_memory(struct Pointmass *, int free_immutables);
