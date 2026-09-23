sph2volren.py - SPH particles to gridded volume data for VAPOR and ParaView
============================================================================

What it does
------------
sph2volren.py reads a miluphcuda HDF5 snapshot, deposits the SPH particles
onto a regular 3D grid with the SPH kernel, and writes a CF-compliant NetCDF
file (.nc). VAPOR and ParaView can volume-render that file directly.

Volume renderers need gridded data. Rendering the particles themselves only
gives points or splats, never a lit, continuous surface with a soft ejecta
plume. That gridding step is what this script provides.

Companion scripts:
  make_movie.py     grid many snapshots with one fixed box; render a VAPOR
                    session file to frames and an MP4
  paraview_look.py  sets up the gray/yellow volume look in ParaView


Requirements
------------
  Python 3 with numpy, numba, h5py, netCDF4
  scipy        only for --smooth
  ffmpeg       only for movies
  VAPOR Python API only for rendering with make_movie.py:
      conda install -c conda-forge -c ncar-vapor vapor

  pip install numpy numba h5py netCDF4 scipy


Input
-----
One miluphcuda HDF5 output file per call. Dataset names used by default:

  x               positions, shape (N, 3)
  m               masses
  sml             smoothing length = kernel support radius (W = 0 for r >= h)
  v               velocities, shape (N, 3)   (only for --vcut)
  material_type   material ids                (only for --ejecta-mat)
  time            snapshot time               (optional, see --time)

Other names can be set with --name-x, --name-m, --name-sml, --name-v,
--name-mat. To list the datasets in a file:

  h5ls impact.0499.h5

If a dataset given with --color-var does not exist, the script stops and
lists all available datasets.


Output
------
A NetCDF file with cell-centred coordinates x, y, z, a time coordinate, and
float32 fields of shape (time, z, y, x):

  rho_total     SPH density, sum_j m_j W(r - r_j, h_j)            [kg/m^3]
  rho_ejecta    the same for the ejecta subset only                [kg/m^3]
                (only with --ejecta-mat or --vcut)
  f_ejecta      rho_ejecta / rho_total                             [-]
  <name>        mass-weighted SPH mean of each --color-var dataset,
                sum m q W / sum m W                                [as input]

Mass conservation: each particle's kernel is renormalised on the discrete
grid, so the mass on the grid matches the particle mass inside the box, even
where h is comparable to the cell size. The script prints this ratio.


Options
-------
  h5                    input HDF5 snapshot (positional)
  -o, --out FILE        output NetCDF file (required)
  --n N                 cells along the longest box axis (default 384)
  --box XMIN XMAX YMIN YMAX ZMIN ZMAX
                        fixed grid extent in code units
                        (default: automatic box from particle percentiles)
  --clip P              percent of particles cut on each side of each axis
                        for the automatic box (default 0.2)
  --hmin-cells F        minimum kernel support in grid cells (default 1.5)
  --hscale F            multiply all smoothing lengths by F (default 1.0)
  --smooth S            Gaussian filter on the grid, sigma S in cells
                        (default 0 = off; needs scipy)
  --color-var NAME ...  per-particle scalar(s) to grid as mass-weighted mean
  --ejecta-mat ID ...   particles with these material_type ids are "ejecta"
  --vcut V              particles with |v| > V are "ejecta"
                        (--ejecta-mat and --vcut are mutually exclusive)
  --time T              override the snapshot time
  --name-x/-m/-sml/-v/-mat NAME
                        HDF5 dataset names

What the script prints, for example:

  impact.0499.h5: 26243453 particles, t = 49.9
    total_plastic_strain: min 0  median ...  99% ...  max 4.1
    grid 384 x 301 x 297, dx = ..., 0.55 GB RAM for the grids
    box: --box -1.2e4 1.1e4 -9e3 9e3 -9e3 9e3
    particles outside box: 0 (0.000% of mass)
    mass on grid / mass inside box = 1.0002
    wrote impact.0499_eps.nc  (rho_total max = ...)

The "box:" line can be copied directly into later calls. The 99% value of
the colour variable is a good upper end for the colormap range.


Examples
--------
1) Basic: density plus plastic strain for colouring

   python sph2volren.py impact.0499.h5 -o impact.0499_eps.nc \
       --n 384 --color-var total_plastic_strain

2) The full ejecta plume (the automatic box cuts the fastest ejecta)

   First see the full particle extent:

   python sph2volren.py impact.0499.h5 -o test.nc --n 128 --clip 0

   Then copy the printed "box:" line, shrink the axes where nothing
   interesting happens if single fragments blow up the box, and use more
   cells so dx does not grow:

   python sph2volren.py impact.0499.h5 -o impact.0499_eps.nc \
       --n 512 --box -1.5e4 3.0e4 -1.2e4 1.2e4 -1.2e4 1.2e4 \
       --color-var total_plastic_strain

3) Softer look

   Larger kernel, smooths sparse ejecta most:

   python sph2volren.py impact.0499.h5 -o impact.0499_soft.nc \
       --n 384 --color-var total_plastic_strain --hscale 1.5

   Additionally a light Gaussian filter on the grid:

   python sph2volren.py impact.0499.h5 -o impact.0499_soft.nc \
       --n 384 --color-var total_plastic_strain --hscale 1.5 --smooth 1.0

   Fewer lighting artefacts / moire rings on the body surface:

   ... --hmin-cells 2.5

   All three options also soften the body surface and the crater rim.

4) Ejecta selected by material id or by speed

   python sph2volren.py impact.0499.h5 -o impact.0499_ej.nc \
       --n 384 --ejecta-mat 2

   python sph2volren.py impact.0499.h5 -o impact.0499_ej.nc \
       --n 384 --vcut 5.0

   Then colour by f_ejecta. The script warns if the selection contains all
   particles or none (f_ejecta would then be constant).

5) Several colour variables in one file

   python sph2volren.py impact.0499.h5 -o impact.0499.nc --n 384 \
       --color-var total_plastic_strain damage_total

   (use the dataset names of your file; h5ls lists them)

6) Time series with one common grid (needed for movies)

   Take the box from the latest, most extended snapshot, then:

   for f in impact.*.h5; do
       python sph2volren.py "$f" -o "${f%.h5}.nc" --n 384 \
           --box -1.5e4 3.0e4 -1.2e4 1.2e4 -1.2e4 1.2e4 \
           --color-var total_plastic_strain --hscale 1.5
   done

   or with make_movie.py (skips snapshots whose .nc is already up to date;
   extra options are passed on to sph2volren.py):

   python make_movie.py --grid "impact.*.h5" \
       --box -1.5e4 3.0e4 -1.2e4 1.2e4 -1.2e4 1.2e4 \
       --n 384 --color-var total_plastic_strain --hscale 1.5

   Each snapshot needs its own time value. It is read from the HDF5
   dataset "time"; if that is missing, pass --time for each file, e.g.

   i=0; for f in impact.*.h5; do
       python sph2volren.py "$f" -o "${f%.h5}.nc" --box ... --time $i
       i=$((i+1))
   done


Rendering in VAPOR (3.10)
-------------------------
  Import tab   NetCDF-CF, select the .nc file(s). Several files at once
               become one dataset with one time step per file.
  Render tab   "+" -> Volume renderer
    Variables    Variable Name          = rho_total (sets opacity)
                 Color mapped variable  = total_plastic_strain
    Appearance   Rendering Method -> Raytracing Algorithm = Regular
                   (not OSPRay - otherwise the next section is hidden)
                 Ray Tracing -> tick "Color by other variable"
                 Sampling Rate Multiplier 2x-4x for final images
                 upper transfer function (rho_total):
                   opacity ~0 below ~0.01 rho0, low plateau (haze) up to
                   ~0.35 rho0, steep rise to 1 at ~0.5 rho0
                 Colormap Transfer Function (strain):
                   gray at 0 -> yellow at 1 (double-click control points)
                 Lighting -> Enabled
  Annotate tab untick Axis Annotations and Display Domain Bounds,
               Background Color = black, Time Annotation = No annotation
  Export tab   TIFF/PNG, Current frame or Time series range,
               Output Resolution -> Use Custom Output Size

  Movie: set up the look with all .nc files loaded, File -> Save Session
  (e.g. deimos.vs3), then

   python make_movie.py deimos.vs3 --out deimos.mp4 --res 1920 1080 --fps 25


Rendering in ParaView (6.x)
---------------------------
  Open the .nc file (NetCDF CF reader), Apply. Several impact.*.nc files
  are grouped as a file series (time steps).

  Display properties (gear icon = advanced properties):
    Representation                 Volume
    Coloring                       total_plastic_strain
    Use Separate Opacity Array     tick, Volume Opacity Array = rho_total
    Shade                          tick
    Volume Rendering Mode          GPU Based
    Scalar Opacity Unit Distance   ~2-6 grid cells (larger = more transparent)

  With a separate opacity array ParaView keeps two transfer functions: the
  colormap of total_plastic_strain and the opacity function of rho_total.
  The opacity function starts at its default, so the ramp has to be set
  again after ticking the option.

  Easiest: select the source, View -> Python Shell -> Run Script ->
  paraview_look.py. Set RHO0 (bulk density of the target) at the top of
  that script first; HAZE, UNIT_CELLS, STRAIN_GRAY and STRAIN_YELLOW adjust
  the look.

  Gray -> yellow colormap without purple (Python Shell):

    disp = GetDisplayProperties(GetActiveSource())
    lut = disp.LookupTable
    lut.AutomaticRescaleRangeMode = 'Never'
    lut.ColorSpace = 'RGB'
    lut.RGBPoints = [0.0, 0.75, 0.75, 0.75,   # gray
                     0.3, 0.86, 0.84, 0.66,   # pale cream
                     0.6, 0.96, 0.89, 0.42,   # light yellow
                     1.0, 1.00, 0.85, 0.10]   # golden yellow, >1 clamps
    Render()

  Use disp.LookupTable rather than GetColorTransferFunction(...): if the
  display uses a separate colour map, the latter changes a table that is
  not shown.

  Movie: open all .nc files as one series, File -> Save Animation (or
  pvbatch), then

    ffmpeg -framerate 25 -i frame.%04d.png -c:v libx264 -pix_fmt yuv420p out.mp4


Troubleshooting
---------------
  Plume looks cropped with a flat edge
      The automatic box cut the fastest ejecta. Use --clip 0 or an explicit
      --box, and check "particles outside box" in the output.

  Everything has the same colour
      The ejecta selection contains all particles (e.g. --ejecta-mat with
      the target's id), or the colormap range does not fit the data. Check
      the ids with
        python -c "import h5py,numpy as np; print(np.unique(h5py.File('impact.0499.h5')['material_type'][:], return_counts=True))"
      and set the colormap range to about 0 ... the printed 99% value.

  Streaks, rings or blocky surface
      Raise the sampling rate (VAPOR) or use GPU Based volume rendering with
      linear interpolation (ParaView); regrid with --hmin-cells 2.5,
      --hscale 1.5 or --smooth 1.

  Speckled or dotty plume
      Sparse ejecta with small h: --hscale 1.5 ... 2.

  Note "h < 1.5 dx -- grid is coarser than the SPH resolution"
      Most kernels are smaller than a cell; raise --n for more detail
      (memory permitting).

  ParaView warning "required texture size of 65536, falling back to 16384"
      Harmless. Two opacity control points are very close together; move
      them apart to silence it.

  Frames of a time series jump or change resolution
      Use the same --box (and --n) for all snapshots.

  Runtime and memory
      The deposition is a serial numba loop. Expect a few minutes for ~10^7
      particles at --n 384-512. The grids need about 8 bytes per cell and
      field in RAM while running (printed by the script).
