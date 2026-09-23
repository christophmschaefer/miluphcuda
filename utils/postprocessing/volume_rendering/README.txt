Volume rendering of miluphcuda SPH output with VAPOR and ParaView
=================================================================

This directory contains tools to turn miluphcuda HDF5 snapshots into
volume-rendered images and movies, e.g. an impact on a small body with the
body shown as a lit, matte surface and the ejecta plume as a soft haze,
coloured by the total plastic strain (gray = undeformed, yellow = strongly
deformed).

Volume renderers need data on a regular grid. Rendering the SPH particles
themselves (points, splats) never gives a lit, continuous surface. The first
step is therefore always to deposit the particles onto a grid with the SPH
kernel (sph2volren.py). The gridded data can then be rendered with VAPOR or
with ParaView.

Contents
--------
  1.  Files
  2.  Requirements
  3.  Step 1: gridding with sph2volren.py
  4.  Choosing box, resolution and smoothing
  5.  Examples for gridding
  6.  Rendering with VAPOR (GUI)
  7.  Frames with VAPOR on a cluster (vapor_frames.py)
  8.  Rendering with ParaView (GUI)
  9.  Frames with ParaView on a cluster (make_frames_paraview_rgba.py)
  10. Making the movie (ffmpeg)
  11. Checking the initial conditions
  12. Troubleshooting and known pitfalls


1. Files
--------
  sph2volren.py                 SPH particles (HDF5) -> regular grid.
                                Writes a CF NetCDF file (.nc, for VAPOR and
                                ParaView) and optionally a VTK image file
                                (.vti, with pre-computed RGBA colours, for
                                ParaView).
  vapor_frames.py               renders PNG frames offscreen with VAPOR's
                                Python API from a session saved in the VAPOR
                                GUI.
  paraview_rgba_look.py         ParaView GUI: applies the look (colours,
                                opacity, lighting, colour bar) to a .vti.
  make_frames_paraview_rgba.py  renders PNG frames offscreen with pvbatch
                                from .vti files.
  shape_map.py                  diagnostic: map of the outer surface radius
                                of an SPH body (checks initial conditions).


2. Requirements
---------------
  Gridding:
    Python 3 with numpy, numba, h5py, netCDF4; scipy only for --smooth
      pip install numpy numba h5py netCDF4 scipy
  VAPOR:
    VAPOR 3.10 (GUI); for vapor_frames.py the VAPOR Python API:
      conda create -n vapor -c conda-forge -c ncar-vapor vapor
  ParaView:
    ParaView 6.x (tested with 6.1 on macOS); pvbatch for frames
  Movies:
    ffmpeg
  shape_map.py:
    numpy, h5py, matplotlib


3. Step 1: gridding with sph2volren.py
--------------------------------------
Input: one miluphcuda HDF5 snapshot per call. Dataset names used by default:

  x               positions, shape (N, 3)
  m               masses
  sml             smoothing length = kernel support radius (W = 0 for r >= h)
  v               velocities, shape (N, 3)   (only for --vcut)
  material_type   material ids                (only for --ejecta-mat)
  time            snapshot time               (optional, see --time)

Other names can be set with --name-x, --name-m, --name-sml, --name-v and
--name-mat. "h5ls impact.0499.h5" lists the datasets of a file. If a dataset
given with --color-var does not exist, the script stops and lists all
available ones.

Method: miluphcuda cubic spline kernel with support radius sml. Every
particle's kernel is renormalised on the discrete grid, so the mass on the
grid equals the particle mass inside the box, even where h is comparable to
the cell size (the script prints this ratio). h is floored at
--hmin-cells * dx so that no particle falls between grid points.

Output 1, NetCDF (.nc, always written; for VAPOR and ParaView). Cell-centred
coordinates x, y, z, a time coordinate, and float32 fields (time, z, y, x):

  rho_total     SPH density, sum_j m_j W(r - r_j, h_j)            [kg/m^3]
  <name>        mass-weighted SPH mean of each --color-var dataset,
                sum m q W / sum m W, e.g. total_plastic_strain    [as input]
  rho_ejecta    density of the ejecta subset only                  [kg/m^3]
  f_ejecta      rho_ejecta / rho_total                             [-]
                (the last two only with --ejecta-mat or --vcut)

Output 2, VTK image data (.vti, only with --vti; for ParaView). The same
grid and float fields as the .nc, plus

  rgba          4 components, 16-bit unsigned integers:
                R, G, B = colour from the strain (gray -> cream -> light
                          yellow -> yellow, fully yellow at --rgba-yellow-at)
                A       = rho_total / --rgba-rho-max, scaled to 0..65535
  field data    TimeValue, rgba_rho_max, rgba_yellow_at (read by the
                ParaView scripts)

The .vti is uncompressed, about 16 bytes per cell (e.g. ~130 MB for
256 x 213 x 153 cells). Why ParaView needs it: see section 8.

Options of sph2volren.py
  h5                      input HDF5 snapshot (positional)
  -o, --out FILE.nc       output NetCDF file (required)
  --n N                   cells along the longest box axis (default 384)
  --box XMIN XMAX YMIN YMAX ZMIN ZMAX
                          fixed grid extent in code units
                          (default: automatic box from particle percentiles)
  --clip P                percent of particles cut on each side of each axis
                          for the automatic box (default 0.2)
  --hmin-cells F          minimum kernel support in grid cells (default 1.5)
  --hscale F              multiply all smoothing lengths by F (default 1.0)
  --smooth S              Gaussian filter on the grid, sigma S in cells
                          (default 0 = off; needs scipy). Density and
                          density*quantity are filtered separately, so the
                          mean strain stays a consistent mass-weighted mean.
  --color-var NAME ...    per-particle scalar(s) to grid as mass-weighted mean
  --ejecta-mat ID ...     particles with these material_type ids are "ejecta"
  --vcut V                particles with |v| > V are "ejecta"
                          (--ejecta-mat and --vcut are mutually exclusive)
  --time T                override the snapshot time
  --vti FILE.vti          additionally write the RGBA .vti for ParaView
  --rgba-var NAME         colour variable for the .vti (default: first
                          --color-var)
  --rgba-yellow-at S      value that is fully yellow (default 1.0)
  --rgba-gray-below S     below this value the colour stays gray (default 0)
  --rgba-yellow R G B     yellow tone (default 1.0 0.85 0.10)
  --rgba-rho-max RHO      density mapped to A = 65535 (default: max of
                          rho_total). Use the SAME value for all snapshots
                          of a movie, otherwise the opacity jumps.
  --name-x/-m/-sml/-v/-mat NAME
                          HDF5 dataset names

Example of what the script prints:

  impact.0499.h5: 26243453 particles, t = 49.9
    total_plastic_strain: min 1.291e-05  median 0.002358  99% 0.4483  max 1118
    grid 512 x 425 x 305, dx = 0.432607, 1.06 GB RAM for the grids
    box: --box -95.2173 126.278 -90.932 92.7501 -65.8797 65.7022
    particles outside box: 21178 (0.081% of mass)
    Gaussian smoothing, sigma = 3.0 cells
    mass on grid / mass inside box = 1.0000
    wrote impact.0499_eps.nc  (rho_total max = 1568)

The "box:" line can be copied directly into later calls. The 99% value of
the colour variable is a good upper end for the colour scale.


4. Choosing box, resolution and smoothing
-----------------------------------------
Box
  --clip 0      includes every particle. A few far-flung fragments can make
                the box enormous (in one test ~100 km instead of ~200 m);
                the body then covers only a few cells and the script warns
                "100% of particles have h < 1.5 dx".
  --clip 0.1    a good compromise: drops the most distant stragglers but
                keeps the plume. Check "particles outside box" in the output.
  --box ...     full control. Needed for time series: all snapshots of a
                movie must use the SAME --box and --n.
  A plume that ends in a flat, straight edge was cut by the box on that
  side: enlarge the box in that direction.

Resolution
  More cells is not automatically better. Very fine grids resolve the
  particle arrangement itself (lattice or shell structure of the initial
  conditions), which the lighting shows as ripples and rings. For a smooth,
  matte body like in a typical overview picture, --n 256 has worked well;
  it is also cheap. Test with --n 128 first (minutes), then go up.

Smoothness
  --hscale 1.5 ... 2    larger SPH kernel; smooths the sparse ejecta most
  --smooth 1 ... 2      Gaussian filter on the grid (sigma in cells)
  --hmin-cells 2.5 ... 3
                        smoother gradients, fewer lighting artefacts
  All three also soften the body surface and the crater rim. --smooth
  only makes sense relative to dx: on a fine grid 2-3 cells is moderate,
  on a coarse grid 1 cell is already a lot.

Runtime and memory
  The deposition is a serial numba loop. Expect a few minutes for ~10^7
  particles at --n 256-512. --hscale 1.5 costs about 1.5^3 = 3.4 times more
  (more cells per kernel). The grids need about 8 bytes per cell and field
  in RAM while running (printed by the script).


5. Examples for gridding
------------------------
1) Density plus plastic strain, NetCDF only (VAPOR)

   python sph2volren.py impact.0499.h5 -o impact.0499_n256.nc \
       --n 256 --clip 0.1 --color-var total_plastic_strain

2) NetCDF plus RGBA .vti (VAPOR and ParaView from one run)

   python sph2volren.py impact.0499.h5 -o impact.0499_n256.nc \
       --vti impact.0499_n256.vti \
       --n 256 --clip 0.1 --color-var total_plastic_strain \
       --rgba-yellow-at 1.0 --rgba-rho-max 1700

3) Softer look

   python sph2volren.py impact.0499.h5 -o impact.0499_n256s.nc \
       --n 256 --clip 0.1 --color-var total_plastic_strain \
       --hscale 1.5 --smooth 1.5

4) Ejecta selected by material id or by speed (colour by f_ejecta)

   python sph2volren.py impact.0499.h5 -o impact.0499_ej.nc --n 256 --ejecta-mat 2
   python sph2volren.py impact.0499.h5 -o impact.0499_ej.nc --n 256 --vcut 5.0

   The script warns if the selection contains all particles or none.
   Material ids and their counts:
     python -c "import h5py,numpy as np; print(np.unique(h5py.File('impact.0499.h5')['material_type'][:], return_counts=True))"

5) Time series with one common grid (needed for movies)

   Take the box from the latest, most extended snapshot (--clip 0.1,
   "box:" line), enlarge it where the plume would be cut, then:

   for f in impact.*.h5; do
       python sph2volren.py "$f" -o "${f%.h5}_n256.nc" --vti "${f%.h5}_n256.vti" \
           --n 256 --box -100 150 -95 95 -70 70 \
           --color-var total_plastic_strain --rgba-rho-max 1700
   done

   Each snapshot needs its own time value. It is read from the HDF5
   dataset "time"; if that is missing, pass --time for each file:

   i=0; for f in impact.*.h5; do
       python sph2volren.py "$f" -o "${f%.h5}.nc" --box ... --time $i
       i=$((i+1))
   done

   The loop is embarrassingly parallel: on a cluster, run one snapshot per
   job (e.g. a job array); every job uses the same --box, --n and
   --rgba-rho-max.


6. Rendering with VAPOR (GUI)
-----------------------------
VAPOR reads the .nc file. Its volume renderer takes the lighting normals from
the density (the primary variable) and the colour from a second variable,
which is exactly what is needed here.

Import tab
  NetCDF-CF -> select the .nc file(s). Several files selected at once become
  one dataset with one time step per file.

Render tab -> "+" -> Volume renderer
  Variables tab
    Variable Name           rho_total             (controls the opacity)
    Color mapped variable   total_plastic_strain  (controls the colour)
  Appearance tab
    Rendering Method -> Raytracing Algorithm = Regular
      (NOT OSPRay; otherwise the "Ray Tracing" section below is hidden)
    Ray Tracing
      tick "Color by other variable"
      Sampling Rate Multiplier 2x-4x for final images (removes fine
      cross-hatch patterns caused by sampling)
    Transfer function of rho_total (opacity), in units of rho0 (bulk density):
      opacity ~0 below ~0.01 rho0
      low plateau (plume haze, ~0.08) from ~0.05 to ~0.35 rho0
      steep rise to 1 at ~0.5 rho0 (solid body)
    Colormap Transfer Function (strain):
      light gray at 0 -> yellow at 1 (double-click a control point to set its
      colour); a pale cream point in between keeps the transition in
      yellow tones
    Lighting (section at the end of the Appearance tab)
      Enabled    on
      Ambient    0.6
      Diffuse    0.5
      Specular   0.1
      Shininess  0.2
  Colorbar tab
    optional colour bar, e.g. title "strain", range 0 ... 1

Annotate tab
  untick "Axis Annotations Enabled"
  "3D Geometry": untick "Display Domain Bounds", Background Color = black
  Time Annotation = "No annotation" (or show it on purpose)

Export tab
  TIFF or PNG, "Current frame" or "Time series range";
  Output Resolution -> "Use Custom Output Size" for high-resolution images.

Session
  File -> Save Session (e.g. impact.vs3). A session stores everything:
  renderers, transfer functions, lighting, colour bar, camera, background,
  and the (absolute) paths of the data files.


7. Frames with VAPOR on a cluster (vapor_frames.py)
---------------------------------------------------
vapor_frames.py loads a session saved in the GUI and renders one PNG per time
step with VAPOR's Python API. All of the look comes from the session, so the
frames look like the GUI view.

  1. In the VAPOR GUI, set up the picture and save the session. Best: load
     ALL time steps (all impact.*_n256.nc at once) before saving. VAPOR only
     reads a file when it is needed, so this is quick.
  2. Copy the session to the cluster.
  3. Check the paths first (renders nothing):

       python vapor_frames.py impact.vs3 --out frames --dry-run \
           --remap /Users/me/sims/deimos /scratch/me/deimos

     It lists the data files found in the session and stops if any are
     missing on this machine.
  4. Render:

       python vapor_frames.py impact.vs3 --out frames --res 1920 1920 \
           --remap /Users/me/sims/deimos /scratch/me/deimos

Options of vapor_frames.py
  session             VAPOR session file (.vs3)
  --out DIR           frame directory (default frames); the rewritten
                      session <name>.cluster.vs3 is also written there
  --res W H           image size (default 1920 1920). Keep the aspect ratio
                      of the GUI window, otherwise the framing differs.
  --remap OLD NEW     replace the directory prefix OLD by NEW in the dataset
                      paths of the session (can be given several times)
  --files GLOB        replace the session's file list by all files matching
                      GLOB (quoted), e.g. "/scratch/me/deimos/impact.*_n256.nc";
                      for sessions saved with only one time step and exactly
                      one dataset
  --first N --last N  range of time steps (default: all)
  --skip-existing     continue an interrupted run
  --dry-run           only rewrite the session and check the paths

Rendering is offscreen; the API creates its own OpenGL context. If that
fails on a node without a GPU/display, run under a virtual display:

  xvfb-run -a python vapor_frames.py impact.vs3 ...

Note on --files: when a session saved with one time step gets a list of many
files, it has not been verified that VAPOR does not keep the animation range
stored in the session. Saving the session with all time steps loaded avoids
the question.


8. Rendering with ParaView (GUI)
--------------------------------
Why ParaView needs the .vti file
  The obvious ParaView setup (colour by total_plastic_strain, "Use Separate
  Opacity Array" = rho_total, Shade on) looks wrong: VTK then computes the
  lighting normals from the COLOUR array, i.e. from the plastic strain. The
  body shows the internal structure of the strain field (shear bands etc.)
  as crumpled fake relief, while the same data in VAPOR look smooth. VTK has
  a switch to change this (ComputeNormalFromOpacity), but ParaView 6.1 does
  not expose it, neither in the GUI nor from Python.
  With the 4-component RGBA array from sph2volren.py --vti and "Map Scalars"
  off, VTK takes the colour directly from R,G,B, the opacity from the opacity
  function applied to A, and the lighting normals from A, i.e. from the
  density -- as in VAPOR. This was verified with VTK 9.7 on a synthetic body
  with a striped strain field: separate opacity array -> stripes appear as
  relief; RGBA -> smooth body.
  Consequence: the colours are baked into the .vti (--rgba-yellow-at,
  --rgba-gray-below, --rgba-yellow); changing them means regridding. The
  opacity remains adjustable in ParaView.

Setup
  1. Open the .vti file, Apply.
  2. Select it in the Pipeline Browser.
  3. View -> Python Shell -> "Run Script" -> paraview_rgba_look.py

Settings at the top of paraview_rgba_look.py
  RHO0              bulk density [kg/m^3] (default 1500)
  HAZE              opacity of the thin plume (default 0.08)
  SOLID_LO, SOLID_HI
                    opacity ramp to solid between these fractions of RHO0
                    (default 0.35, 0.5). A wider ramp makes the body partly
                    transparent and therefore darker.
  UNIT_CELLS        Scalar Opacity Unit Distance in grid cells (default 2;
                    larger = more transparent)
  AMBIENT, DIFFUSE, SPECULAR, SPECULAR_POWER
                    lighting (default 0.6, 0.5, 0.1, 10 = the VAPOR values)
  COLORBAR          draw a "strain" colour bar (gray -> yellow)

What the script sets (for doing it by hand)
  Representation = Volume, Coloring = rgba, Map Scalars = off,
  Use Separate Opacity Array = off, Volume Rendering Mode = GPU Based,
  Shade = on, opacity function of "rgba" in units of the A channel
  (A = rho / rgba_rho_max * 65535), Scalar Opacity Unit Distance.
  Ambient, Diffuse, Specular and SpecularPower of a volume are NOT shown in
  the ParaView 6.1 GUI; they can only be set from Python (paste the lines
  between the markers into the Python Shell; no leading spaces):
----8<----
disp = GetDisplayProperties(GetActiveSource())
disp.Ambient = 0.6; disp.Diffuse = 0.5; disp.Specular = 0.1
Render()
----8<----

Saving a camera for the frame script (View -> Python Shell, one line):
----8<----
import json; v = GetActiveView(); json.dump(dict(position=list(v.CameraPosition), focal_point=list(v.CameraFocalPoint), view_up=list(v.CameraViewUp), view_angle=v.CameraViewAngle, parallel_scale=v.CameraParallelScale), open('/full/path/camera.json', 'w'))
----8<----


9. Frames with ParaView on a cluster (make_frames_paraview_rgba.py)
-------------------------------------------------------------------
  pvbatch make_frames_paraview_rgba.py "impact.*_n256.vti" --rho0 1500 \
      --colorbar --camera camera.json --res 1920 1080 --out frames

  (quote the pattern so the script expands it)

Options
  --out DIR              frame directory (default frames)
  --res W H              image size (default 1920 1080)
  --rho0 RHO             bulk density [kg/m^3] (default 1500)
  --haze A               plume opacity (default 0.08)
  --solid-lo F --solid-hi F
                         opacity ramp to solid (default 0.35 0.5)
  --unit-cells F         Scalar Opacity Unit Distance in cells (default 2)
  --ambient/--diffuse/--specular/--specular-power
                         lighting (default 0.6 0.5 0.1 10)
  --colorbar             draw a "strain" colour bar
  --background R G B     background colour (default 0 0 0)
  --camera FILE.json     camera saved from the GUI; default: fit the last
                         snapshot once, then keep the camera fixed
  --zoom F               zoom for the default camera (default 1)
  --orbit DEG            rotate the view by DEG degrees over the movie
  --first N --last N     range of time steps
  --skip-existing        continue an interrupted run

All values are fixed for all frames (no automatic rescaling), so colours and
brightness do not jump between frames -- provided all .vti files were
written with the same --box, --n and --rgba-rho-max.


10. Making the movie (ffmpeg)
-----------------------------
Both frame scripts write frames/frame_00000.png, frame_00001.png, ...

  ffmpeg -framerate 25 -i frames/frame_%05d.png -c:v libx264 \
         -pix_fmt yuv420p -crf 18 impact.mp4

  -crf 18 is visually lossless; larger values give smaller files.
  Odd image sizes: add -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2".


11. Checking the initial conditions
-----------------------------------
Rendering the t = 0 snapshot shows how the particles were placed; structures
seen there (smooth vs. irregular parts, rings) come from the setup, not from
the impact.

  Slice (ParaView): select the .nc/.vti, Filters -> Slice (normal e.g. X,
  origin at the centre), Representation = Surface, colour by rho_total,
  colour range narrowed around rho0 (e.g. 1400 ... 1600). Concentric shells
  or a lattice pattern at t = 0 are part of the initial particle setup.

  Surface map (shape_map.py): for every direction seen from the centre of
  mass the largest particle distance, as a longitude/latitude map. A sphere
  gives a flat, uniformly coloured map, an irregular body gives structure.

    python shape_map.py impact.0000.h5 --out shape_0000.png

  Radial histogram: a body built from spherical shells shows a comb of
  sharp, evenly spaced peaks; a lattice cut by a shape model gives a smooth
  curve.

  (copy the lines between the markers into radial_hist.py, python radial_hist.py)
----8<----
import h5py, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
x = h5py.File('impact.0000.h5')['x'][:]; d = x - x.mean(0); r = np.linalg.norm(d, axis=1)
for name, sel in (('+z half', d[:,2] > 0), ('-z half', d[:,2] < 0)):
    plt.hist(r[sel], bins=3000, histtype='step', label=name)
plt.xlabel('r'); plt.ylabel('particles'); plt.legend(); plt.savefig('radial_hist.png', dpi=120)
----8<----


12. Troubleshooting and known pitfalls
--------------------------------------
Gridding
  Plume cropped with a flat edge
      The box cut the plume. Enlarge --box on that side (or use a smaller
      --clip) and check "particles outside box".
  "100% of particles have h < 1.5 dx", very low rho_total max
      The box is far too large (typically --clip 0 with a few distant
      fragments). Use --clip 0.1 or an explicit --box, or raise --n.
  Ripples, rings, cross-hatch on the body
      Usually the particle arrangement or the grid made visible by the
      lighting. Use a coarser grid (--n 256), --hscale 1.5, --smooth 1-2 or
      --hmin-cells 2.5; in VAPOR also raise the Sampling Rate Multiplier.
  Speckled or dotty plume
      Sparse ejecta: --hscale 1.5 ... 2, slightly higher haze plateau.
  Everything has the same colour (--ejecta-mat)
      The selection contains all particles; check the material ids (section
      5, example 4).
  Frames of a time series jump or change resolution
      Different --box/--n (or --rgba-rho-max) between snapshots.

VAPOR
  "Color by other variable" not visible
      Raytracing Algorithm is set to OSPRay; set it to Regular.
  Nothing visible
      Check that the renderer is enabled in the Render tab, press "View All",
      and check that the opacity transfer function rises above zero within
      the data range.
  vapor_frames.py: "file(s) not found"
      The session stores absolute paths; use --remap OLD NEW (or --files).

ParaView
  Body looks crumpled/cracked, VAPOR shows it smooth
      "Use Separate Opacity Array" with Shade: the normals come from the
      strain. Use the .vti with the rgba array (section 8).
  Empty view; error "... vtkImageData, but a vtkStructuredGrid is required"
      The NetCDF reader switched its output type after the display was
      created. Delete the source, reopen the .nc and set Output Type = Image
      BEFORE Apply. (.vti files are always image data and not affected.)
  Nothing visible after opening a new/regridded file
      ParaView keeps transfer functions per array name for the whole session,
      so the new file inherits old opacity settings. Reload the file, Reset
      Camera and run paraview_rgba_look.py again, or restart ParaView.
  Purple colours in a gray/yellow colormap
      A preset is still active. For the .nc route set the colours on
      disp.LookupTable (the table the display really uses), not on
      GetColorTransferFunction(...). With the .vti the colours are baked in.
  Warning "required texture size of 65536, falling back to 16384"
      Harmless: two opacity control points are very close together.
  Ambient/Diffuse/Specular not in the GUI
      For volumes they exist only in Python (disp.Ambient etc.) in ParaView
      6.1.
  Mac freezes or crashes
      Volumetric Scattering Blending / Global Illumination Reach on large
      grids, or switching to OSPRay volume rendering, can take down the whole
      machine with Apple's OpenGL driver. Keep Volume Rendering Mode = GPU
      Based, ray tracing disabled, and use scattering/GI only with small grids
      (e.g. --n 128) or on Linux with an NVIDIA GPU. Save the state first.
  Python Shell: SyntaxError when pasting multi-line blocks
      The shell is line-based; blocks (for/try) need a blank line after them.
      Put longer code in a file and use "Run Script".
