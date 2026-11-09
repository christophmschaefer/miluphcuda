#!/usr/bin/env python3
"""
Diagnostics for the von Mises radial return (VON_MISES_PLASTICITY) in the tensile rod test.

For each HDF5 output file, the von Mises equivalent stress sqrt(3 J2) of the particles that have yielded
(total_plastic_strain > 0) is compared to the yield stress Y, and the total plastic strain is summed.
J2 is computed from the DIM x DIM deviatoric stress as in src/plasticity.cu.

Note: the output holds the state after the integration step, i.e. before the next return, so sqrt(3 J2)/Y is
slightly above 1 in plastic flow. The limited stress that enters the forces is
    f = Y/sqrt(3 J2)      ->  exactly Y                 (current code)
    f = Y^2/(3 J2)        ->  Y^2/sqrt(3 J2) < Y        (Benz & Asphaug form, before)
and is printed as well, for both forms.

usage: check_von_mises.py --yield_stress 0.03 rod.0100.h5 [rod.0200.h5 ...]
"""

import argparse
import h5py
import numpy as np

parser = argparse.ArgumentParser(description="von Mises radial return diagnostics")
parser.add_argument("files", nargs="+", help="HDF5 output files")
parser.add_argument("--yield_stress", type=float, default=0.03, help="Y from material.cfg (default: 0.03)")
args = parser.parse_args()

Y = args.yield_stress
print("# file  time  n_plastic  mean(sqrt(3J2)/Y)  max(sqrt(3J2)/Y)  "
      "mean(limited/Y, sqrt form)  mean(limited/Y, squared form)  sum(m*ep)")
for fn in args.files:
    with h5py.File(fn, "r") as f:
        S = f["/deviatoric_stress"][:]
        ep = f["/total_plastic_strain"][:]
        m = f["/m"][:]
        t = float(np.ravel(f["/time"][:])[0])
    n = S.shape[0]
    dim = int(round(np.sqrt(S.shape[1])))
    S = S.reshape(n, dim, dim)
    J2 = 0.5 * np.einsum("nij,nij->n", S, S)
    seq = np.sqrt(3.0 * J2)
    plastic = ep > 0.0
    if not np.any(plastic):
        print(f"{fn}  {t:.6e}  0")
        continue
    r = seq[plastic] / Y
    lim_sqrt = np.where(r > 1.0, 1.0, r)
    lim_squared = np.where(r > 1.0, 1.0 / np.maximum(r, 1.0), r)
    print(f"{fn}  {t:.6e}  {plastic.sum()}  {r.mean():.6f}  {r.max():.6f}  "
          f"{lim_sqrt.mean():.6f}  {lim_squared.mean():.6f}  {np.sum(m * ep):.6e}")
