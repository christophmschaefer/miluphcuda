#!/usr/bin/env python
"""
suggest_box.py -- suggest ONE common --box for sph2volren.py for a whole time
series. For a sample of snapshots (default: every 50th plus the last) it
computes the particle extent per axis, dropping --clip percent on each side
(same meaning as in sph2volren.py), and prints the union of these boxes,
padded by --pad (fraction of the largest extent).

    python suggest_box.py "impact.*.h5"
    python suggest_box.py "impact.*.h5" --every 25 --clip 0.1 --pad 0.05 --n 256

It also prints, for the last snapshot, how much mass would lie outside the
suggested box, and the cell size for a given --n.

Test a box of your own choice (mass outside it for every sampled snapshot):

    python suggest_box.py "impact.*.h5" --test-box -150 650 -450 150 -350 350
"""
import argparse
import glob
import sys

import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("pattern")
ap.add_argument("--every", type=int, default=50, help="use every N-th file (default 50)")
ap.add_argument("--clip", type=float, default=0.1, help="percent cut per side (default 0.1)")
ap.add_argument("--pad", type=float, default=0.05, help="padding, fraction of extent (default 0.05)")
ap.add_argument("--n", type=int, default=256, help="--n for the dx estimate (default 256)")
ap.add_argument("--test-box", type=float, nargs=6,
                metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                help="report the mass fraction outside this box per snapshot")
ap.add_argument("--name-x", default="x")
ap.add_argument("--name-m", default="m")
a = ap.parse_args()

files = sorted(glob.glob(a.pattern))
if not files:
    sys.exit("no files match %s" % a.pattern)
sample = files[::a.every]
if files[-1] not in sample:
    sample.append(files[-1])

if a.test_box:
    tlo = np.array(a.test_box[0::2]); thi = np.array(a.test_box[1::2])
    dx = (thi - tlo).max() / a.n
    dims = np.ceil((thi - tlo) / dx).astype(int)
    print("test box %s, with --n %d: dx = %.4g, grid %d x %d x %d"
          % (" ".join("%g" % v for v in a.test_box), a.n, dx, *dims))
    for f in sample:
        with h5py.File(f, "r") as h:
            x = np.asarray(h[a.name_x], dtype=np.float64)
            m = np.asarray(h[a.name_m], dtype=np.float64)
            t = float(np.asarray(h["time"]).ravel()[0]) if "time" in h else float("nan")
        out = ~((x >= tlo).all(1) & (x <= thi).all(1))
        print("  %s  t = %-8g outside: %6.3f%% of mass, %d particles"
              % (f, t, 100.0 * m[out].sum() / m.sum(), out.sum()))
    sys.exit(0)

lo_all = np.full(3, np.inf)
hi_all = np.full(3, -np.inf)
for f in sample:
    with h5py.File(f, "r") as h:
        x = np.asarray(h[a.name_x], dtype=np.float64)
        t = float(np.asarray(h["time"]).ravel()[0]) if "time" in h else float("nan")
    lo = np.percentile(x, a.clip, axis=0)
    hi = np.percentile(x, 100.0 - a.clip, axis=0)
    lo_all = np.minimum(lo_all, lo)
    hi_all = np.maximum(hi_all, hi)
    print("%s  t = %-10g x [%.4g, %.4g]  y [%.4g, %.4g]  z [%.4g, %.4g]"
          % (f, t, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))

pad = a.pad * (hi_all - lo_all).max()
lo_b, hi_b = lo_all - pad, hi_all + pad
dx = (hi_b - lo_b).max() / a.n
dims = np.ceil((hi_b - lo_b) / dx).astype(int)

with h5py.File(files[-1], "r") as h:
    x = np.asarray(h[a.name_x], dtype=np.float64)
    m = np.asarray(h[a.name_m], dtype=np.float64)
inside = (x >= lo_b).all(1) & (x <= hi_b).all(1)

print()
print("suggested for all %d files:" % len(files))
print("  --box %.6g %.6g %.6g %.6g %.6g %.6g"
      % (lo_b[0], hi_b[0], lo_b[1], hi_b[1], lo_b[2], hi_b[2]))
print("  with --n %d: dx = %.4g, grid %d x %d x %d" % (a.n, dx, dims[0], dims[1], dims[2]))
print("  last snapshot: %.3f%% of the mass outside this box"
      % (100.0 * m[~inside].sum() / m.sum()))
