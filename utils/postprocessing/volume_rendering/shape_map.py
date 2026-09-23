#!/usr/bin/env python
"""
shape_map.py -- map the outer surface of an SPH body: for every direction
(longitude, latitude seen from the centre of mass) the largest particle
distance. A sphere gives a flat map, an irregular shape gives structure.

    python shape_map.py impact.0000.h5 [--nlon 360] [--nlat 180] [--out shape.png]
"""
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("h5")
ap.add_argument("--nlon", type=int, default=360)
ap.add_argument("--nlat", type=int, default=180)
ap.add_argument("--out", default="shape_map.png")
ap.add_argument("--name-x", default="x")
ap.add_argument("--name-m", default="m")
a = ap.parse_args()

with h5py.File(a.h5, "r") as f:
    x = np.asarray(f[a.name_x], dtype=np.float64)
    m = np.asarray(f[a.name_m], dtype=np.float64)

com = (x * m[:, None]).sum(0) / m.sum()
d = x - com
r = np.linalg.norm(d, axis=1)
lon = np.degrees(np.arctan2(d[:, 1], d[:, 0]))              # -180..180
lat = np.degrees(np.arcsin(np.clip(d[:, 2] / np.maximum(r, 1e-300), -1, 1)))

i = np.clip(((lon + 180) / 360 * a.nlon).astype(int), 0, a.nlon - 1)
j = np.clip(((lat + 90) / 180 * a.nlat).astype(int), 0, a.nlat - 1)
rmax = np.zeros((a.nlat, a.nlon))
np.maximum.at(rmax, (j, i), r)
rmax[rmax == 0] = np.nan

print(f"centre of mass {com}")
print(f"surface radius: min {np.nanmin(rmax):.4g}  median {np.nanmedian(rmax):.4g}  "
      f"max {np.nanmax(rmax):.4g}")
for name, sel in (("x>0 hemisphere", np.abs(np.linspace(-180, 180, a.nlon, endpoint=False) + 180/a.nlon) < 90),
                  ("x<0 hemisphere", np.abs(np.linspace(-180, 180, a.nlon, endpoint=False) + 180/a.nlon) >= 90)):
    sub = rmax[:, sel]
    print(f"  {name}: std of surface radius {np.nanstd(sub):.4g} "
          f"(relative {np.nanstd(sub)/np.nanmedian(sub):.3%})")

fig, ax = plt.subplots(figsize=(10, 5.5))
im = ax.imshow(rmax, origin="lower", extent=(-180, 180, -90, 90), cmap="viridis",
               aspect="auto")
ax.set_xlabel("longitude from +x axis [deg]  (0 = +x, +-180 = -x)")
ax.set_ylabel("latitude [deg]  (+90 = +z)")
ax.set_title(f"outer surface radius from centre of mass: {a.h5}")
fig.colorbar(im, ax=ax, label="max particle distance")
fig.tight_layout()
fig.savefig(a.out, dpi=120)
print(f"wrote {a.out}")
