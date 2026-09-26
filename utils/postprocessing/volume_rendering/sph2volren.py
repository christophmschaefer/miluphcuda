#!/usr/bin/env python
"""
sph2volren.py -- deposit SPH particles (miluphcuda HDF5 output) onto a regular
grid and write a CF-compliant NetCDF file (VAPOR, ParaView) and optionally a
VTK .vti file with pre-computed RGBA colours (ParaView, --vti). Volume
renderers need gridded data; rendering the particles themselves gives
points/splats, never a lit, continuous surface.

Fields written (float32, dims time,z,y,x, cell-centred coordinates x,y,z):
    rho_total   SPH density  sum_j m_j W(r - r_j, h_j)          [kg/m^3]
    rho_ejecta  same, restricted to the "ejecta" particle subset  [kg/m^3]
                (only with --ejecta-mat or --vcut)
    f_ejecta    rho_ejecta / rho_total (0 where rho_total ~ 0)     [-]
    <var>       mass-weighted SPH mean of any per-particle scalar given with
                --color-var (e.g. total_plastic_strain):
                sum m q W / sum m W                                   [as input]

Kernel: miluphcuda cubic spline, support radius = sml (W = 0 for r >= h).
Each particle's kernel is renormalised on the discrete grid, so mass is
conserved exactly even where h is comparable to the cell size, and h is
floored at --hmin-cells * dx so no particle falls between cells.

Usage examples:
    python sph2volren.py impact.0100.h5 -o impact.0100.nc --n 384 \
        --color-var total_plastic_strain
    python sph2volren.py impact.0100.h5 -o impact.0100.nc --n 512 --vcut 5.0
    for f in impact.*.h5; do python sph2volren.py $f -o ${f%.h5}.nc --n 384 \
        --box -2e4 2e4 -2e4 2e4 -2e4 2e4 --color-var total_plastic_strain; done

Grid box and resolution
    Without --box the extent is chosen from particle percentiles, dropping
    --clip percent (default 0.2) on each side of each axis. With many
    particles this can cut off the fastest ejecta (plume looks "cropped"
    with a flat edge). The script prints the box used (as a ready --box
    argument) and how many particles fall outside. For the full plume use
    --clip 0 (or a small value like 0.01) or an explicit --box, and raise --n
    so dx does not grow. Time series: use ONE fixed --box for all snapshots.

Smoothness
    --hscale 1.5 ... 2   larger SPH kernel (smooths sparse ejecta most)
    --smooth 1 ... 2     Gaussian filter on the grid, sigma in cells; density
                         and density*quantity are smoothed separately, so the
                         mean strain stays a consistent mass-weighted mean
    --adaptive-h 32      kernel >= distance to the 32nd neighbour: sparse ejecta
                       become a continuous haze instead of separate blobs
                       (needs scipy; --workers N for the neighbour search)
  --hmin-cells 2.5     smoother gradients -> fewer lighting artefacts and
                         less moire between particle lattice and grid
    All three also soften the body surface and crater rim.

-------------------------------------------------------------------------------
Rendering (details: README.txt)
-------------------------------------------------------------------------------
  VAPOR     open the .nc (NetCDF-CF). Volume renderer: Variable = rho_total,
            Color mapped variable = total_plastic_strain, "Color by other
            variable" on (Raytracing Algorithm = Regular), Lighting enabled.
            Frames on a cluster: vapor_frames.py with a saved session.
  ParaView  use the .vti written with --vti (RGBA: colour from the strain,
            alpha = density). Colour by "rgba", Map Scalars off, Shade on:
            paraview_rgba_look.py (GUI), make_frames_paraview_rgba.py (pvbatch).
            Do NOT use "Use Separate Opacity Array" with Shade: VTK then
            computes the lighting normals from the colour array (strain),
            which shows the strain field's internal structure as fake relief.
"""
import argparse
import warnings
# older h5py with NumPy >= 1.25 warns about np.product on every read; harmless
warnings.filterwarnings("ignore", message=".*product.*deprecated", category=DeprecationWarning)
import numpy as np
import h5py
import netCDF4
from numba import njit


# ----------------------------------------------------------------------------
# kernel + deposition
# ----------------------------------------------------------------------------
@njit(cache=True, inline="always")
def w_cubic(q):
    # un-normalised miluphcuda cubic spline, support q = r/h in [0,1)
    if q < 0.5:
        return 1.0 - 6.0 * q * q + 6.0 * q * q * q
    elif q < 1.0:
        t = 1.0 - q
        return 2.0 * t * t * t
    return 0.0


@njit(cache=True)
def deposit(pos, mass, h, qw, x0, dx, nx, ny, nz, grid_tot, grid_q):
    """Scatter particle mass onto grid (serial -> no write races).
    grid_q[c] accumulates sum_j m_j qw[j,c] W_j  (qw: (N, ncomp))."""
    ncomp = qw.shape[1]
    inv_dV = 1.0 / (dx * dx * dx)
    npart = pos.shape[0]
    for p in range(npart):
        hp = h[p]
        px = pos[p, 0]; py = pos[p, 1]; pz = pos[p, 2]
        # index range of cells whose centres can lie inside the kernel
        i0 = max(int(np.floor((px - hp - x0[0]) / dx - 0.5)), 0)
        i1 = min(int(np.ceil((px + hp - x0[0]) / dx - 0.5)), nx - 1)
        j0 = max(int(np.floor((py - hp - x0[1]) / dx - 0.5)), 0)
        j1 = min(int(np.ceil((py + hp - x0[1]) / dx - 0.5)), ny - 1)
        k0 = max(int(np.floor((pz - hp - x0[2]) / dx - 0.5)), 0)
        k1 = min(int(np.ceil((pz + hp - x0[2]) / dx - 0.5)), nz - 1)
        if i0 > i1 or j0 > j1 or k0 > k1:
            continue
        inv_h = 1.0 / hp
        # pass 1: discrete normalisation
        wsum = 0.0
        for k in range(k0, k1 + 1):
            cz = x0[2] + (k + 0.5) * dx - pz
            for j in range(j0, j1 + 1):
                cy = x0[1] + (j + 0.5) * dx - py
                for i in range(i0, i1 + 1):
                    cx = x0[0] + (i + 0.5) * dx - px
                    wsum += w_cubic(np.sqrt(cx * cx + cy * cy + cz * cz) * inv_h)
        if wsum <= 0.0:
            continue
        fac = mass[p] * inv_dV / wsum
        # pass 2: deposit
        for k in range(k0, k1 + 1):
            cz = x0[2] + (k + 0.5) * dx - pz
            for j in range(j0, j1 + 1):
                cy = x0[1] + (j + 0.5) * dx - py
                for i in range(i0, i1 + 1):
                    cx = x0[0] + (i + 0.5) * dx - px
                    w = w_cubic(np.sqrt(cx * cx + cy * cy + cz * cz) * inv_h)
                    if w > 0.0:
                        val = fac * w
                        grid_tot[k, j, i] += val
                        for c in range(ncomp):
                            grid_q[c, k, j, i] += val * qw[p, c]


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def read_miluph(fname, names, extra=()):
    with h5py.File(fname, "r") as f:
        q = {}
        for name in extra:
            if name not in f:
                raise SystemExit(f"no dataset '{name}' in {fname}; available: "
                                 + ", ".join(sorted(f.keys())))
            arr = np.asarray(f[name], dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            if arr.ndim != 1:
                raise SystemExit(f"'{name}' has shape {arr.shape}; need one scalar per particle")
            q[name] = arr
        pos = np.asarray(f[names["x"]], dtype=np.float64)
        mass = np.asarray(f[names["m"]], dtype=np.float64)
        h = np.asarray(f[names["sml"]], dtype=np.float64)
        v = np.asarray(f[names["v"]], dtype=np.float64) if names["v"] in f else None
        mat = np.asarray(f[names["mat"]], dtype=np.int64) if names["mat"] in f else None
        t = float(np.asarray(f["time"]).ravel()[0]) if "time" in f else 0.0
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise SystemExit(f"expected 3D positions (N,3), got {pos.shape}")
    return pos, mass, h, v, mat, t, q


# ----------------------------------------------------------------------------
# optional VTK image file with pre-computed RGBA (for ParaView)
# ----------------------------------------------------------------------------
GRAY = (0.75, 0.75, 0.75)
CREAM = (0.86, 0.84, 0.66)
LIGHT = (0.96, 0.89, 0.42)


def rgba_colors(q, s_yellow, s_gray, yellow):
    """gray -> cream -> light yellow -> yellow as a function of q (RGB in 0..1)"""
    s0 = max(s_gray, 0.0)
    xs = [0.0]
    cs = [GRAY]
    if s0 > 0.0:
        xs.append(s0); cs.append(GRAY)
    xs += [s0 + 0.3 * (s_yellow - s0), s0 + 0.6 * (s_yellow - s0), s_yellow]
    cs += [CREAM, LIGHT, tuple(yellow)]
    xs = np.array(xs)
    rgb = np.empty(q.shape + (3,), dtype=np.float32)
    for c in range(3):
        rgb[..., c] = np.interp(q, xs, [cc[c] for cc in cs])   # clamps above s_yellow
    return rgb


def write_vti(fname, origin, dx, fields, rgba, t, field_data):
    """VTK XML ImageData (raw appended, little endian). fields: name -> (z,y,x)
    float array, rgba: (z,y,x,4) uint16. Point order x fastest."""
    nz, ny, nx = rgba.shape[:3]
    arrays = [("rgba", "UInt16", 4, np.ascontiguousarray(rgba, dtype="<u2"))]
    for name, arr in fields.items():
        arrays.append((name, "Float32", 1, np.ascontiguousarray(arr, dtype="<f4")))
    fdata = [("TimeValue", "Float64", np.array([t], dtype="<f8"))]
    for name, val in field_data.items():
        fdata.append((name, "Float64", np.array([val], dtype="<f8")))
    offset = 0
    head = []
    head.append('<?xml version="1.0"?>\n<VTKFile type="ImageData" version="1.0" '
                'byte_order="LittleEndian" header_type="UInt64">\n')
    ext = "0 %d 0 %d 0 %d" % (nx - 1, ny - 1, nz - 1)
    head.append('<ImageData WholeExtent="%s" Origin="%.17g %.17g %.17g" '
                'Spacing="%.17g %.17g %.17g">\n' % ((ext,) + tuple(origin) + (dx, dx, dx)))
    blobs = []
    head.append("<FieldData>\n")
    for name, typ, arr in fdata:
        head.append('<DataArray type="%s" Name="%s" NumberOfTuples="1" format="appended" '
                    'offset="%d"/>\n' % (typ, name, offset))
        blobs.append(arr.tobytes()); offset += 8 + len(blobs[-1])
    head.append("</FieldData>\n")
    head.append('<Piece Extent="%s">\n<PointData Scalars="rgba">\n' % ext)
    for name, typ, ncomp, arr in arrays:
        head.append('<DataArray type="%s" Name="%s" NumberOfComponents="%d" '
                    'format="appended" offset="%d"/>\n' % (typ, name, ncomp, offset))
        blobs.append(arr.tobytes()); offset += 8 + len(blobs[-1])
    head.append("</PointData>\n<CellData/>\n</Piece>\n</ImageData>\n"
                '<AppendedData encoding="raw">\n_')
    with open(fname, "wb") as f:
        f.write("".join(head).encode())
        for b in blobs:
            f.write(np.uint64(len(b)).astype("<u8").tobytes())
            f.write(b)
        f.write(b"\n</AppendedData>\n</VTKFile>\n")



def write_cf(fname, x, y, z, t, fields, units):
    with netCDF4.Dataset(fname, "w", format="NETCDF4") as nc:
        nc.Conventions = "CF-1.8"
        nc.title = "SPH data deposited onto regular grid (sph2volren.py)"
        nc.createDimension("time", None)
        nc.createDimension("z", len(z))
        nc.createDimension("y", len(y))
        nc.createDimension("x", len(x))
        vt = nc.createVariable("time", "f8", ("time",))
        vt.units = "seconds since 2001-01-01 00:00:00"   # VAPOR shows "User time" relative to 2001-01-01
        vt.axis = "T"
        vt[0] = t
        for name, arr, ax in (("x", x, "X"), ("y", y, "Y"), ("z", z, "Z")):
            v = nc.createVariable(name, "f8", (name,))
            v.units = "m"
            v.axis = ax
            v[:] = arr
        for name, arr in fields.items():
            v = nc.createVariable(name, "f4", ("time", "z", "y", "x"),
                                  zlib=True, complevel=1)
            v.units = units[name]
            v[0] = arr.astype(np.float32)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--n", type=int, default=384,
                    help="cells along the longest box axis (default 384)")
    ap.add_argument("--box", type=float, nargs=6,
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                    help="grid extent in code units; default: percentile box")
    ap.add_argument("--clip", type=float, default=0.2,
                    help="percentile clipped on each side for the automatic box "
                         "(drops far-flung ejecta; default 0.2)")
    ap.add_argument("--hmin-cells", type=float, default=1.5,
                    help="floor for kernel support in units of dx (default 1.5)")
    ap.add_argument("--adaptive-h", type=int, metavar="K",
                    help="use at least the distance to the K-th nearest neighbour as "
                         "kernel support (e.g. 32): sparse ejecta become a continuous "
                         "haze instead of separate blobs. Needs scipy.")
    ap.add_argument("--workers", type=int, default=1,
                    help="threads for the neighbour search of --adaptive-h (default 1)")
    ap.add_argument("--hscale", type=float, default=1.0,
                    help="extra smoothing factor on sml (e.g. 1.5 for sparse ejecta)")
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--ejecta-mat", type=int, nargs="+",
                     help="material_type id(s) counted as ejecta/projectile")
    sel.add_argument("--vcut", type=float,
                     help="particles with |v| > vcut (code units) count as ejecta")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="extra Gaussian smoothing of the gridded fields, sigma in "
                         "grid cells (e.g. 1.0); needs scipy")
    ap.add_argument("--vti", metavar="FILE.vti",
                    help="additionally write a VTK image file with a pre-computed "
                         "RGBA array for ParaView (colour from --rgba-var, alpha "
                         "from rho_total); see README")
    ap.add_argument("--rgba-var", help="colour variable for --vti "
                    "(default: first --color-var)")
    ap.add_argument("--rgba-yellow-at", type=float, default=1.0,
                    help="value of --rgba-var that is fully yellow (default 1.0)")
    ap.add_argument("--rgba-gray-below", type=float, default=0.0,
                    help="below this value the colour stays gray (default 0)")
    ap.add_argument("--rgba-yellow", type=float, nargs=3, default=(1.0, 0.85, 0.10),
                    metavar=("R", "G", "B"))
    ap.add_argument("--rgba-rho-max", type=float,
                    help="density mapped to alpha = 65535 (default: max of "
                         "rho_total). Use the SAME value for all snapshots of a movie")
    ap.add_argument("--time", type=float,
                    help="override snapshot time (if the HDF5 file has no 'time')")
    ap.add_argument("--color-var", nargs="+", default=[],
                    help="per-particle scalar dataset(s) to grid as mass-weighted "
                         "mean, e.g. total_plastic_strain")
    ap.add_argument("--name-x", default="x")
    ap.add_argument("--name-v", default="v")
    ap.add_argument("--name-m", default="m")
    ap.add_argument("--name-sml", default="sml")
    ap.add_argument("--name-mat", default="material_type")
    a = ap.parse_args()

    names = dict(x=a.name_x, v=a.name_v, m=a.name_m, sml=a.name_sml, mat=a.name_mat)
    pos, mass, h, v, mat, t, q = read_miluph(a.h5, names, a.color_var)
    if a.time is not None:
        t = a.time
    print(f"{a.h5}: {len(mass)} particles, t = {t:g}")

    # ejecta selection
    if a.ejecta_mat is not None:
        if mat is None:
            raise SystemExit(f"no dataset '{a.name_mat}' in file")
        is_ej = np.isin(mat, a.ejecta_mat)
    elif a.vcut is not None:
        if v is None:
            raise SystemExit(f"no dataset '{a.name_v}' in file")
        is_ej = np.linalg.norm(v, axis=1) > a.vcut
    else:
        is_ej = None
    if is_ej is not None:
        print(f"  ejecta particles: {is_ej.sum()} of {len(mass)}")
        if is_ej.all() or not is_ej.any():
            print("  WARNING: ejecta selection is all-or-nothing -> f_ejecta is constant")
    for name, arr in q.items():
        print(f"  {name}: min {arr.min():.4g}  median {np.median(arr):.4g}  "
              f"99%% {np.percentile(arr, 99):.4g}  max {arr.max():.4g}".replace("%%", "%"))

    # grid
    if a.box is not None:
        lo = np.array(a.box[0::2]); hi = np.array(a.box[1::2])
    else:
        lo = np.percentile(pos, a.clip, axis=0)
        hi = np.percentile(pos, 100.0 - a.clip, axis=0)
        pad = 0.05 * (hi - lo).max()
        lo -= pad; hi += pad
    dx = (hi - lo).max() / a.n
    nx, ny, nz = (np.ceil((hi - lo) / dx).astype(int))
    print(f"  grid {nx} x {ny} x {nz}, dx = {dx:g}, "
          f"{nx*ny*nz*8*(1+len(a.color_var)+(is_ej is not None))/1e9:.2f} GB RAM for the grids")

    inside = (pos >= lo).all(1) & (pos <= hi).all(1)
    nout = (~inside).sum()
    print(f"  box: --box {lo[0]:.6g} {hi[0]:.6g} {lo[1]:.6g} {hi[1]:.6g} "
          f"{lo[2]:.6g} {hi[2]:.6g}")
    print(f"  particles outside box: {nout} "
          f"({100*mass[~inside].sum()/mass.sum():.3f}% of mass)")

    hs = h * a.hscale
    if a.adaptive_h:
        # adaptive kernel: at least the distance to the K-th neighbour, so that
        # sparse ejecta overlap into a continuous haze instead of separate blobs
        from scipy.spatial import cKDTree
        k = a.adaptive_h
        print(f"  adaptive h: building tree, distance to neighbour {k} "
              f"({a.workers} worker(s)) ...", flush=True)
        tree = cKDTree(pos)
        margin = hs.max() + a.hmin_cells * dx
        near = np.where((pos >= lo - margin).all(1) & (pos <= hi + margin).all(1))[0]
        hk = np.zeros_like(hs)
        chunk = 500000
        for c0 in range(0, len(near), chunk):
            idx = near[c0:c0 + chunk]
            d, _ = tree.query(pos[idx], k=k + 1, workers=a.workers)
            hk[idx] = d[:, -1]
        del tree
        grown = np.mean(hk[near] > hs[near])
        print(f"  adaptive h: kernel enlarged for {100*grown:.1f}% of the particles "
              f"(median d_{k} = {np.median(hk[near]):.4g})")
        hs = np.maximum(hs, hk)
    heff = np.maximum(hs, a.hmin_cells * dx)
    frac_floor = np.mean(hs < a.hmin_cells * dx)
    hq = np.percentile(hs, [5, 50, 95])
    print(f"  kernel support h*hscale: 5% {hq[0]:.4g}, median {hq[1]:.4g}, 95% {hq[2]:.4g}"
          f"  (= {hq[1]/dx:.2f} dx median); used: max(h*hscale, {a.hmin_cells} dx = "
          f"{a.hmin_cells*dx:.4g})")
    if frac_floor > 0.5:
        print(f"  note: {100*frac_floor:.0f}% of particles have h < {a.hmin_cells} dx: the "
              f"grid, not the SPH resolution, sets the finest detail (~2-3 dx = "
              f"{2*dx:.3g}-{3*dx:.3g}). Mass is still conserved. For more detail "
              f"raise --n (or use a smaller --box); median h is reached at "
              f"--n ~ {int(np.ceil(a.n * a.hmin_cells * dx / max(hq[1], 1e-300)))}")

    cols, qnames = [], []
    if is_ej is not None:
        cols.append(is_ej.astype(np.float64)); qnames.append("rho_ejecta")
    for name, arr in q.items():
        cols.append(arr); qnames.append(name)
    qw = np.stack(cols, axis=1) if cols else np.zeros((len(mass), 0))
    g_tot = np.zeros((nz, ny, nx), dtype=np.float64)
    g_q = np.zeros((len(cols), nz, ny, nx), dtype=np.float64)
    deposit(pos, mass, heff, qw, lo, dx, nx, ny, nz, g_tot, g_q)

    if a.smooth > 0:
        from scipy.ndimage import gaussian_filter
        # smooth density and density-weighted quantities separately, so the
        # ratios (f_ejecta, mean strain) stay consistent mass-weighted means
        g_tot = gaussian_filter(g_tot, a.smooth, mode="constant")
        for c in range(g_q.shape[0]):
            g_q[c] = gaussian_filter(g_q[c], a.smooth, mode="constant")
        print(f"  Gaussian smoothing, sigma = {a.smooth} cells")
    m_grid = g_tot.sum() * dx**3
    print(f"  mass on grid / mass inside box = {m_grid / mass[inside].sum():.4f}")

    thresh = 1e-6 * g_tot.max()
    safe = np.maximum(g_tot, thresh)
    fields = dict(rho_total=g_tot)
    units = dict(rho_total="kg m-3")
    for c, name in enumerate(qnames):
        if name == "rho_ejecta":
            fields["rho_ejecta"] = g_q[c]; units["rho_ejecta"] = "kg m-3"
            fields["f_ejecta"] = np.where(g_tot > thresh, g_q[c] / safe, 0.0)
            units["f_ejecta"] = "1"
        else:
            fields[name] = np.where(g_tot > thresh, g_q[c] / safe, 0.0)
            units[name] = "1"

    xc = lo[0] + (np.arange(nx) + 0.5) * dx
    yc = lo[1] + (np.arange(ny) + 0.5) * dx
    zc = lo[2] + (np.arange(nz) + 0.5) * dx
    write_cf(a.out, xc, yc, zc, t, fields, units)

    if a.vti:
        cvar = a.rgba_var or (a.color_var[0] if a.color_var else None)
        if cvar is None or cvar not in fields:
            raise SystemExit("--vti needs --rgba-var (or --color-var) with a gridded variable")
        rho_max = a.rgba_rho_max or float(g_tot.max())
        rgb = rgba_colors(fields[cvar], a.rgba_yellow_at, a.rgba_gray_below, a.rgba_yellow)
        rgba = np.empty(g_tot.shape + (4,), dtype=np.uint16)
        rgba[..., :3] = np.round(np.clip(rgb, 0, 1) * 65535)
        rgba[..., 3] = np.round(np.clip(g_tot / rho_max, 0, 1) * 65535)
        # pin the colour range to the full 0..65535 in two invisible corner
        # cells (alpha 0), so no renderer rescales the colours
        rgba[0, 0, 0] = (0, 0, 0, 0)
        rgba[-1, -1, -1] = (65535, 65535, 65535, 0)
        write_vti(a.vti, (xc[0], yc[0], zc[0]), dx, fields, rgba, t,
                  dict(rgba_rho_max=rho_max, rgba_yellow_at=a.rgba_yellow_at))
        print(f"  wrote {a.vti}  (rgba: colour from {cvar}, yellow at {a.rgba_yellow_at}; "
              f"alpha = rho_total / {rho_max:.6g})")
    print(f"  wrote {a.out}  (rho_total max = {g_tot.max():.4g})")


if __name__ == "__main__":
    main()
