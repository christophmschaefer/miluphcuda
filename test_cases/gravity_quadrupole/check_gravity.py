#!/usr/bin/env python3
"""
Compares the tree self-gravity (/a_grav in <prefix>_<variant>_<theta>/setup.0001.h5) to a direct summation over the
positions in the same file, with the softening of miluphcuda (direct_selfgravity in gravity.cu):

    a_i = G sum_{j != i} m_j (x_j - x_i) / max(|x_j - x_i|, h_i)^3

and prints per variant and theta the relative error |a_tree - a_direct| / |a_direct| (median, 90%, 99%, max), the
error normalised with the rms acceleration (excludes the particles near the centre of the sphere, where |a| -> 0),
the net force |sum m a| / sum m |a| (zero for the direct sum), and the median duration of selfgravity per call
(DEBUG_RHS_RUNTIMES, includes calculateCentersOfMass, first call skipped).

Pass criterion: for every theta, the median and 99% error of quad_open are smaller than those of mono.
"""

import argparse
import os
import re
import sys

import h5py
import numpy as np


def direct_sum(x, m, h, G, chunk=128):
    a = np.zeros_like(x)
    n = len(m)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        d = x[None, :, :] - x[i0:i1, None, :]
        r = np.sqrt(np.sum(d*d, axis=2))
        r = np.maximum(r, h[i0:i1, None])
        f = G * m[None, :] / r**3
        f[np.arange(i1 - i0), np.arange(i0, i1)] = 0.0
        a[i0:i1] = np.einsum('ij,ijk->ik', f, d)
    return a


def selfgravity_time(outfile):
    t = []
    if not os.path.exists(outfile):
        return np.nan
    with open(outfile) as f:
        for line in f:
            mt = re.match(r'duration selfgravity:\s*([0-9.eE+-]+) ms', line)
            if mt:
                t.append(float(mt.group(1)))
    return np.median(t[1:]) if len(t) > 1 else (t[0] if t else np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prefix', default='run_3d', help='run directories are <prefix>_<variant>_<theta>')
    ap.add_argument('--thetas', nargs='+', default=['0.3', '0.5', '0.7', '0.9'])
    ap.add_argument('--variants', nargs='+', default=['mono', 'quad', 'quad_open'])
    ap.add_argument('--G', type=float, default=6.67408e-11, help='gravitational constant (default of miluphcuda)')
    ap.add_argument('--file', default='setup.0001.h5')
    args = ap.parse_args()

    ref = None
    res = {}
    ok = True
    print("%-10s %5s %10s %10s %10s %10s %10s %10s %12s" % ("variant", "theta", "median", "90%", "99%", "max",
          "99%/a_rms", "net force", "t_grav [ms]"))
    for v in args.variants:
        for th in args.thetas:
            rundir = "%s_%s_%s" % (args.prefix, v, th)
            fn = os.path.join(rundir, args.file)
            if not os.path.exists(fn):
                print("%-10s %5s  missing %s" % (v, th, fn))
                ok = False
                continue
            with h5py.File(fn, 'r') as f:
                x = f['x'][:]
                m = f['m'][:]
                h = f['sml'][:]
                ag = f['a_grav'][:]
            if ref is None or ref[0].shape != x.shape or np.max(np.abs(ref[0] - x)) > 1e-6 * np.min(h):
                sys.stderr.write("direct summation for %d particles ...\n" % len(m))
                ref = (x.copy(), direct_sum(x, m, h, args.G))
            ad = ref[1]
            absd = np.sqrt(np.sum(ad**2, axis=1))
            err = np.sqrt(np.sum((ag - ad)**2, axis=1))
            rel = err / absd
            arms = np.sqrt(np.mean(absd**2))
            net = np.linalg.norm(np.sum(m[:, None] * ag, axis=0)) / np.sum(m * np.sqrt(np.sum(ag**2, axis=1)))
            q = np.percentile(rel, [50, 90, 99])
            res[(v, th)] = q
            print("%-10s %5s %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %12.3f" % (v, th, q[0], q[1], q[2], rel.max(),
                  np.percentile(err, 99) / arms, net, selfgravity_time(os.path.join(rundir, "output.txt"))))
        print()

    for th in args.thetas:
        if ('mono', th) in res and ('quad_open', th) in res:
            if not (res[('quad_open', th)][0] < res[('mono', th)][0] and res[('quad_open', th)][2] < res[('mono', th)][2]):
                ok = False
                print("theta %s: quad_open is not more accurate than mono" % th)
    print("PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
