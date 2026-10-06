"""
render_dimorphos_vol.py -- volume/surface rendering of miluphcuda SPH output in
Blender >= 5.0 (Cycles). Input: frames from sph2blender.py.

Body   : slow particles -> signed distance field (Points to SDF Grid) ->
         smoothed -> mesh, coloured from the nearest particle (rock / damage /
         compaction). Looks like a solid, continuous surface with boulders.
Ejecta : particles faster than --vcut (relative to the bulk) -> density grid
         (Points to Volume + dilate + box filter ~ SPH number density) and a
         speed grid (nearest particle, smoothed). Rendered as a real volume:
         sunlit dust (forward scattering) + optional speed-coloured glow.

Everything is built with Blender's grid nodes, no OpenVDB-Python needed.

Usage
  blender -b -P render_dimorphos_vol.py -- --data blender_frames --out renders \
          --frames 1 --samples 64 --res 1280 720 --save-blend dart_vol.blend
  blender -b -P render_dimorphos_vol.py -- --data blender_frames --out renders \
          --frames 300 --timemap log --samples 128 --timestamp
  # still of one dump (exact dump time), camera as in the film:
  blender -b -P render_dimorphos_vol.py -- --data blender_frames --out stills \
          --dump 270 --samples 128 --timestamp
  ffmpeg -framerate 30 -i renders/frame_%04d.png -c:v libx264 -crf 16 -pix_fmt yuv420p dart.mp4
"""
import argparse
import json
import math
import os
import sys

import functools

import bpy
import numpy as np

VERSION = "vol-9"
print = functools.partial(print, flush=True)   # Blender -b may drop buffered stdout
from mathutils import Vector

if bpy.app.version < (5, 0, 0):
    raise SystemExit("render_dimorphos_vol.py needs Blender >= 5.0 (grid nodes); "
                     f"this is {bpy.app.version_string}")

# ----------------------------------------------------------------------------
# arguments
# ----------------------------------------------------------------------------
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser(prog="render_dimorphos_vol.py")
ap.add_argument("--data", required=True, help="directory from sph2blender.py")
ap.add_argument("--out", default="renders")
ap.add_argument("--frames", type=int, default=240, help="number of film frames")
ap.add_argument("--range", type=int, nargs=2, metavar=("F0", "F1"),
                help="render only film frames F0..F1 (for job arrays)")
ap.add_argument("--dump", nargs="+", metavar="N",
                help="render still(s) of these dumps instead of the film, e.g. --dump 270 "
                     "or --dump impact.0270.h5 (exact dump time, no interpolation); "
                     "camera as in the film at that time (same --frames/--timemap/--cam)")
ap.add_argument("--time", type=float, nargs="+", metavar="T",
                help="render still(s) at these simulation times [s] (interpolated)")
ap.add_argument("--timemap", choices=["linear", "log"], default="log")
ap.add_argument("--tau", type=float, default=0.05, help="time scale [s] of the log map")
ap.add_argument("--tmax", type=float, help="end time [s] (default: last dump)")
ap.add_argument("--res", type=int, nargs=2, default=[1920, 1080])
ap.add_argument("--samples", type=int, default=128)
ap.add_argument("--device", default="auto", help="auto|CPU|OPTIX|CUDA|METAL|HIP|ONEAPI")

g = ap.add_argument_group("geometry")
g.add_argument("--spacing", type=float, default=0.777,
               help="initial particle spacing [m]; sets the default radii/voxel sizes")
g.add_argument("--body-radius", type=float, help="SDF sphere radius [m] (default 1.0*spacing)")
g.add_argument("--body-voxel", type=float, help="SDF voxel size [m] (default 0.33*spacing)")
g.add_argument("--body-smooth", type=int, default=3, help="SDF mean-filter iterations")
g.add_argument("--body-width", type=int, default=2, help="SDF mean-filter half width [voxels]")
g.add_argument("--vcut", type=float, default=0.5,
               help="speed [m/s] rel. to bulk above which material is ejecta (volume)")
g.add_argument("--ej-radius", type=float, help="splat radius [m] (default 0.9*spacing)")
g.add_argument("--ej-voxel", type=float, help="ejecta voxel size [m] (default 0.6*spacing)")
g.add_argument("--ej-maxdist", type=float,
               help="ejecta farther than this [m] from --target are skipped "
                    "(default 5000 * ej-voxel, ~2.3 km)")
g.add_argument("--ej-rmax", type=float,
               help="max. adaptive splat radius [m] for thin ejecta (default 8*spacing)")
g.add_argument("--ej-k", type=int, default=8,
               help="neighbour number for the adaptive radius (SPH-like smoothing length)")
g.add_argument("--show-projectile", action="store_true",
               help="render the projectile particles (41 SPH particles, ~4-6 km/s backwards after impact)")
g.add_argument("--ej-vmax", type=float, default=0.0,
               help="skip ejecta faster than this [m/s] (single jetting particles), 0 = keep all")
g.add_argument("--ej-blur", type=int, default=1, help="extra smoothing of the ejecta density [voxels]")

s = ap.add_argument_group("look")
s.add_argument("--color", choices=["rock", "damage", "compaction"], default="rock")
s.add_argument("--ej-density", type=float, default=0.3, help="ejecta extinction [1/m] at full density")
s.add_argument("--ej-gamma", type=float, default=0.6,
               help="density exponent (<1 lifts thin ejecta, 1 = linear)")
s.add_argument("--ej-albedo", type=float, default=0.8)
s.add_argument("--ej-aniso", type=float, default=0.45, help="forward scattering (-1..1)")
s.add_argument("--glow", type=float, default=0.8, help="emission of ejecta (0 = only sunlit dust)")
s.add_argument("--vcmap", type=float, nargs=2, default=[0.4, 40.0], metavar=("VMIN", "VMAX"),
               help="speed range [m/s] of the glow colormap (log)")
s.add_argument("--bump", type=float, default=0.25, help="regolith micro bump strength")
s.add_argument("--target", type=float, nargs=3, default=[-4.0, -75.0, -8.0])
s.add_argument("--cam", type=float, nargs=3, default=[-35.0, 14.0, 420.0], metavar=("AZ", "EL", "DIST"))
s.add_argument("--cam-end", type=float, nargs=3, metavar=("AZ", "EL", "DIST"))
s.add_argument("--lens", type=float, default=50.0)
s.add_argument("--sun", type=float, nargs=3, default=[0.55, -0.75, 0.35], help="direction TO the sun")
s.add_argument("--sun-strength", type=float, default=5.0)
s.add_argument("--bloom", type=float, default=0.5)
s.add_argument("--stars", type=float, default=1.0)
s.add_argument("--didymos", action="store_true")
s.add_argument("--didymos-pos", type=float, nargs=3, default=[-1100.0, 1500.0, -350.0])
s.add_argument("--timestamp", action="store_true")

ap.add_argument("--reuse", action="store_true",
                help="keep scene/camera/materials of the opened .blend, only swap the data")
ap.add_argument("--save-blend")
ap.add_argument("--no-render", action="store_true")
ap.add_argument("--diag", action="store_true", help="print ejecta density diagnostics per frame")
A = ap.parse_args(argv)
a = A.spacing
A.body_radius = A.body_radius or 1.0 * a
A.body_voxel = A.body_voxel or 0.33 * a
A.ej_radius = A.ej_radius or 0.9 * a
A.ej_voxel = A.ej_voxel or 0.6 * a
A.ej_maxdist = A.ej_maxdist or 5000.0 * A.ej_voxel
A.ej_rmax = A.ej_rmax or 8.0 * a

BODY, EJECTA = "SPH_Body", "SPH_Ejecta"

# ----------------------------------------------------------------------------
# data: index, loading, Hermite interpolation
# ----------------------------------------------------------------------------
with open(os.path.join(A.data, "index.json")) as f:
    INDEX = json.load(f)
TIMES = np.array(INDEX["times"])
_cache = {}


def load(k):
    if k not in _cache:
        if len(_cache) > 2:
            _cache.pop(next(iter(_cache)))
        z = np.load(os.path.join(A.data, INDEX["files"][k]))
        _cache[k] = {n: z[n] for n in z.files}
    return _cache[k]


def state_at(t):
    k = int(np.clip(np.searchsorted(TIMES, t, side="right") - 1, 0, len(TIMES) - 1))
    a_ = load(k)
    if k == len(TIMES) - 1 or t <= TIMES[k] or INDEX["N"][k] != INDEX["N"][k + 1]:
        return dict(x=a_["x"].astype(np.float64), v=a_["v"].astype(np.float32), mat=a_["mat"],
                    dmg=a_["dmg"] / 255.0, comp=a_["comp"] / 255.0)
    b = load(k + 1)
    _, ia, ib = np.intersect1d(a_["ids"], b["ids"], assume_unique=True, return_indices=True)
    dt = TIMES[k + 1] - TIMES[k]
    s_ = (t - TIMES[k]) / dt
    h00, h10 = 2 * s_**3 - 3 * s_**2 + 1, s_**3 - 2 * s_**2 + s_
    h01, h11 = -2 * s_**3 + 3 * s_**2, s_**3 - s_**2
    x0, x1 = a_["x"][ia].astype(np.float64), b["x"][ib].astype(np.float64)
    v0, v1 = a_["v"][ia].astype(np.float64), b["v"][ib].astype(np.float64)
    x = h00 * x0 + h10 * dt * v0 + h01 * x1 + h11 * dt * v1
    lin = (1 - s_) * x0 + s_ * x1
    bad = np.linalg.norm(x - lin, axis=1) > 0.5 * np.linalg.norm(x1 - x0, axis=1) + 1.0
    x[bad] = lin[bad]
    v = (1 - s_) * v0 + s_ * v1
    lerp = lambda q: ((1 - s_) * a_[q][ia] + s_ * b[q][ib]) / 255.0
    return dict(x=x, v=v.astype(np.float32), mat=a_["mat"][ia], dmg=lerp("dmg"), comp=lerp("comp"))


def film_time(f):
    t0, t1 = TIMES[0], (A.tmax if A.tmax is not None else TIMES[-1])
    u = f / max(A.frames - 1, 1)
    if A.timemap == "linear":
        return t0 + u * (t1 - t0)
    return t0 + A.tau * (math.exp(u * math.log1p((t1 - t0) / A.tau)) - 1.0)


def film_u(t):
    """Inverse of film_time: position 0..1 of time t on the film's time axis."""
    t0, t1 = TIMES[0], (A.tmax if A.tmax is not None else TIMES[-1])
    if t1 <= t0:
        return 0.0
    if A.timemap == "linear":
        u = (t - t0) / (t1 - t0)
    else:
        u = math.log1p(max(t - t0, 0.0) / A.tau) / math.log1p((t1 - t0) / A.tau)
    return float(np.clip(u, 0.0, 1.0))


def find_dump(spec):
    """Index entry for a dump given as 270, 0270, frame_0270.npz or impact.0270.h5."""
    import re
    m = re.findall(r"\d+", os.path.basename(str(spec)).replace(".h5", "").replace(".npz", ""))
    if not m:
        raise SystemExit(f"--dump {spec}: no dump number found")
    num = int(m[-1])
    srcs = INDEX.get("source") or [None] * len(INDEX["files"])
    for k, (fn, src) in enumerate(zip(INDEX["files"], srcs)):
        for name in (fn, src):
            if name:
                d = re.findall(r"\d+", os.path.basename(name).replace(".h5", "").replace(".npz", ""))
                if d and int(d[-1]) == num:
                    return k
    have = ", ".join(INDEX["files"][:3] + (["..."] if len(INDEX["files"]) > 4 else [])
                     + INDEX["files"][-1:])
    raise SystemExit(f"--dump {spec}: dump {num} is not in {A.data}/index.json ({have})")


# ----------------------------------------------------------------------------
# colours (linear RGB)
# ----------------------------------------------------------------------------
INFERNO = np.array([[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66],
                    [245, 125, 21], [250, 193, 39], [252, 255, 164]]) / 255.0


def to_linear(srgb):
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def cmap(u, table=INFERNO):
    u = np.clip(u, 0, 1) * (len(table) - 1)
    i = np.minimum(u.astype(int), len(table) - 2)
    w = (u - i)[:, None]
    return to_linear((1 - w) * table[i] + w * table[i + 1])


ROCK = {0: (0.16, 0.135, 0.11), 1: (0.075, 0.068, 0.062), 2: (0.6, 0.55, 0.5)}


def diagnose(xe, spd_e, t):
    """Print how dense the ejecta are relative to the splat kernel."""
    n = len(xe)
    if n == 0:
        print(f"[diag] t={t:.3f}s: no ejecta above vcut={A.vcut} m/s")
        return
    from mathutils.kdtree import KDTree
    rng = np.random.default_rng(0)
    sub = rng.choice(n, min(n, 20000), replace=False)
    kd = KDTree(n)
    for i, p in enumerate(xe):
        kd.insert(p, i)
    kd.balance()
    nn = np.array([kd.find_n(xe[i], 2)[1][2] for i in sub])
    med = np.median(nn)
    fill = min(1.0, (4 / 3 * math.pi * A.ej_radius**3) / med**3)
    q = np.percentile(spd_e, [5, 50, 95])
    ext = np.ptp(xe, axis=0)
    print(f"[diag] t={t:.3f}s ejecta N={n}  |v| 5/50/95% = {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f} m/s  "
          f"extent {ext[0]:.0f}x{ext[1]:.0f}x{ext[2]:.0f} m")
    print(f"[diag] nearest-neighbour distance median {med:.2f} m (splat radius {A.ej_radius:.2f} m) "
          f"-> typical grid density ~{fill:.3f}; extinction ~{A.ej_density*fill**A.ej_gamma:.3f} /m")
    if med > 2.5 * A.ej_radius:
        print("[diag] ejecta are sparse compared to the splat radius: increase --ej-radius "
              f"(~{0.8*med:.1f}) and/or --ej-density, or use --ej-gamma 0.5")


def adaptive_kernel(xe):
    """SPH-like: radius grows with the local particle spacing (k-th neighbour),
    weight (r0/r)^3 keeps the column density (mass) right."""
    n = len(xe)
    r0 = A.ej_radius
    if n <= A.ej_k:
        return np.full(n, r0, np.float32), np.ones(n, np.float32)
    from mathutils.kdtree import KDTree
    kd = KDTree(n)
    for i, p in enumerate(xe):
        kd.insert(p, i)
    kd.balance()
    k = A.ej_k + 1
    dk = np.fromiter((kd.find_n(p, k)[-1][2] for p in xe), np.float64, n)
    # in the undisturbed lattice the k-th neighbour (k<=12, HCP) sits at ~spacing
    r = np.clip(r0 * dk / A.spacing, r0, A.ej_rmax)
    w = (r0 / r) ** 3
    return r.astype(np.float32), w.astype(np.float32)


def split_and_shade(st):
    v = st["v"].astype(np.float64)
    spd = np.linalg.norm(v - np.median(v, axis=0), axis=1)
    mat = st["mat"].astype(int)
    proj = mat == 2
    ej = (spd > A.vcut) & ~proj
    if A.show_projectile:
        ej |= proj
    body = ~(ej | proj)
    if A.ej_vmax > 0:
        fast = ej & (spd > A.ej_vmax)
        if fast.any():
            print(f"[info] {fast.sum()} ejecta particles faster than {A.ej_vmax:g} m/s not rendered (--ej-vmax)")
            ej &= ~fast
    # Cycles silently drops a volume whose index range gets too large (a few
    # particles kilometres away are enough), so far-flung ejecta are left out.
    far = ej & (np.linalg.norm(st["x"] - np.array(A.target), axis=1) > A.ej_maxdist)
    if far.any():
        print(f"[info] {far.sum()} ejecta particles beyond {A.ej_maxdist:.0f} m from the target "
              "are not rendered (--ej-maxdist)")
        ej &= ~far

    mb = mat[body]
    col = np.zeros((body.sum(), 3))
    for m, c in ROCK.items():
        col[mb == m] = c
    col[mb > 2] = ROCK[0]
    if A.color == "damage":
        w = ((mb == 1) * st["dmg"][body])[:, None]
        col = (1 - w) * col + w * cmap(0.3 + 0.6 * st["dmg"][body]) * 0.5
    elif A.color == "compaction":
        w = np.clip(st["comp"][body] * 1.5, 0, 1)[:, None]
        col = (1 - w) * col + w * np.array([0.02, 0.30, 0.40])
    rgba = np.ones((len(col), 4), np.float32)
    rgba[:, :3] = col

    lo, hi = math.log10(A.vcmap[0]), math.log10(A.vcmap[1])
    u = np.clip((np.log10(np.maximum(spd[ej], 1e-6)) - lo) / (hi - lo), 0, 1)
    if A.diag:
        diagnose(st["x"][ej], spd[ej], st.get("t", float("nan")))
    rad, wgt = adaptive_kernel(st["x"][ej])
    return (st["x"][body], rgba), (st["x"][ej], u.astype(np.float32), rad, wgt)


# ----------------------------------------------------------------------------
# node helpers
# ----------------------------------------------------------------------------
def N(tree, kind, loc=(0, 0), **props):
    n = tree.nodes.new(kind)
    n.location = loc
    for k, v in props.items():
        setattr(n, k, v)
    return n


def L(tree, a_, b):
    tree.links.new(a_, b)


def new_geo_group(name):
    ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    ng.is_modifier = True
    return ng, N(ng, "NodeGroupInput", (-1200, 0)), N(ng, "NodeGroupOutput", (1400, 0))


def new_object(scene, name):
    obj = bpy.data.objects.new(name, bpy.data.meshes.new(name))
    scene.collection.objects.link(obj)
    return obj


# ----------------------------------------------------------------------------
# body: points -> SDF -> smooth -> mesh, colour from nearest particle
# ----------------------------------------------------------------------------
def body_material():
    m = bpy.data.materials.new("Body")
    nt = m.node_tree
    nt.nodes.clear()
    out = N(nt, "ShaderNodeOutputMaterial", (700, 0))
    bsdf = N(nt, "ShaderNodeBsdfPrincipled", (400, 0))
    bsdf.inputs["Roughness"].default_value = 0.95
    bsdf.inputs["Specular IOR Level"].default_value = 0.1
    at = N(nt, "ShaderNodeAttribute", (0, 150), attribute_name="col")
    # regolith micro-relief and subtle albedo variation
    tc = N(nt, "ShaderNodeTexCoord", (-600, -200))
    nz = N(nt, "ShaderNodeTexNoise", (-350, -200))
    nz.inputs["Scale"].default_value = 0.9
    nz.inputs["Detail"].default_value = 12.0
    nz.inputs["Roughness"].default_value = 0.65
    L(nt, tc.outputs["Object"], nz.inputs["Vector"])
    bump = N(nt, "ShaderNodeBump", (150, -200))
    bump.inputs["Strength"].default_value = A.bump
    bump.inputs["Distance"].default_value = 0.3
    L(nt, nz.outputs["Fac"], bump.inputs["Height"])
    var = N(nt, "ShaderNodeMix", (200, 150), data_type="RGBA", blend_type="MULTIPLY")
    var.inputs["Factor"].default_value = 0.35
    L(nt, at.outputs["Color"], var.inputs["A"])
    L(nt, nz.outputs["Color"], var.inputs["B"])
    L(nt, var.outputs["Result"], bsdf.inputs["Base Color"])
    L(nt, bump.outputs["Normal"], bsdf.inputs["Normal"])
    L(nt, bsdf.outputs["BSDF"], out.inputs["Surface"])
    return m


def build_body(scene):
    obj = new_object(scene, BODY)
    ng, gi, go = new_geo_group("BodySurface")
    pts = N(ng, "GeometryNodeMeshToPoints", (-1000, 0))
    L(ng, gi.outputs[0], pts.inputs["Mesh"])
    sdf = N(ng, "GeometryNodePointsToSDFGrid", (-750, 0))
    sdf.inputs["Radius"].default_value = A.body_radius
    sdf.inputs["Voxel Size"].default_value = A.body_voxel
    L(ng, pts.outputs["Points"], sdf.inputs["Points"])
    mean = N(ng, "GeometryNodeSDFGridMean", (-500, 0))
    mean.inputs["Width"].default_value = A.body_width
    mean.inputs["Iterations"].default_value = A.body_smooth
    L(ng, sdf.outputs[0], mean.inputs["Grid"])
    # shrink back what the overlapping spheres added, so the surface sits on the particles
    off = N(ng, "GeometryNodeSDFGridOffset", (-300, 0))
    off.inputs["Distance"].default_value = -0.45 * A.body_radius
    L(ng, mean.outputs[0], off.inputs["Grid"])
    mesh = N(ng, "GeometryNodeGridToMesh", (-100, 0))
    mesh.inputs["Threshold"].default_value = 0.0
    mesh.inputs["Adaptivity"].default_value = 0.0
    L(ng, off.outputs[0], mesh.inputs["Grid"])
    # colour from nearest particle
    pos = N(ng, "GeometryNodeInputPosition", (-100, -300))
    near = N(ng, "GeometryNodeSampleNearest", (150, -250), domain="POINT")
    L(ng, pts.outputs["Points"], near.inputs["Geometry"])
    L(ng, pos.outputs[0], near.inputs["Sample Position"])
    natt = N(ng, "GeometryNodeInputNamedAttribute", (150, -450), data_type="FLOAT_COLOR")
    natt.inputs["Name"].default_value = "col"
    smp = N(ng, "GeometryNodeSampleIndex", (400, -300), data_type="FLOAT_COLOR", domain="POINT")
    L(ng, pts.outputs["Points"], smp.inputs["Geometry"])
    L(ng, natt.outputs["Attribute"], smp.inputs["Value"])
    L(ng, near.outputs["Index"], smp.inputs["Index"])
    store = N(ng, "GeometryNodeStoreNamedAttribute", (650, 0), data_type="FLOAT_COLOR", domain="POINT")
    store.inputs["Name"].default_value = "col"
    L(ng, mesh.outputs["Mesh"], store.inputs["Geometry"])
    L(ng, smp.outputs[0], store.inputs["Value"])
    smooth = N(ng, "GeometryNodeSetShadeSmooth", (900, 0))
    L(ng, store.outputs[0], smooth.inputs["Mesh"])
    setm = N(ng, "GeometryNodeSetMaterial", (1150, 0))
    setm.inputs["Material"].default_value = body_material()
    L(ng, smooth.outputs[0], setm.inputs["Geometry"])
    L(ng, setm.outputs[0], go.inputs[0])
    obj.modifiers.new("Surface", "NODES").node_group = ng
    return obj


# ----------------------------------------------------------------------------
# ejecta: points -> density grid (+ box filter) and speed grid -> volume
# ----------------------------------------------------------------------------
def ejecta_material():
    m = bpy.data.materials.new("Ejecta")
    nt = m.node_tree
    nt.nodes.clear()
    out = N(nt, "ShaderNodeOutputMaterial", (900, 0))
    vol = N(nt, "ShaderNodeVolumePrincipled", (600, 0))
    dn = N(nt, "ShaderNodeAttribute", (-400, 100), attribute_name="density")
    sp = N(nt, "ShaderNodeAttribute", (-400, -250), attribute_name="speed")
    clamp = N(nt, "ShaderNodeMath", (-150, 100), operation="MAXIMUM")
    clamp.inputs[1].default_value = 0.0
    L(nt, dn.outputs["Fac"], clamp.inputs[0])
    pw = N(nt, "ShaderNodeMath", (0, 100), operation="POWER")
    pw.inputs[1].default_value = A.ej_gamma
    L(nt, clamp.outputs[0], pw.inputs[0])
    dens = N(nt, "ShaderNodeMath", (200, 150), operation="MULTIPLY")
    dens.inputs[1].default_value = A.ej_density
    L(nt, pw.outputs[0], dens.inputs[0])
    L(nt, dens.outputs[0], vol.inputs["Density"])
    vol.inputs["Color"].default_value = (A.ej_albedo,) * 3 + (1,)
    vol.inputs["Anisotropy"].default_value = A.ej_aniso
    ramp = N(nt, "ShaderNodeValToRGB", (0, -250))
    els = ramp.color_ramp.elements
    tab = to_linear(INFERNO[4:])          # glow starts red-orange, not near-black
    while len(els) < len(tab):
        els.new(0.5)
    for e, p, c in zip(els, np.linspace(0, 1, len(tab)), tab):
        e.position, e.color = float(p), (*c, 1.0)
    L(nt, sp.outputs["Fac"], ramp.inputs["Fac"])
    L(nt, ramp.outputs["Color"], vol.inputs["Emission Color"])
    em = N(nt, "ShaderNodeMath", (200, -50), operation="MULTIPLY")
    em.inputs[1].default_value = A.glow
    L(nt, pw.outputs[0], em.inputs[0])
    L(nt, em.outputs[0], vol.inputs["Emission Strength"])
    L(nt, vol.outputs[0], out.inputs["Volume"])
    return m


def build_ejecta(scene):
    obj = new_object(scene, EJECTA)
    ng, gi, go = new_geo_group("EjectaVolume")
    pts = N(ng, "GeometryNodeMeshToPoints", (-1000, 0))
    L(ng, gi.outputs[0], pts.inputs["Mesh"])
    p2v = N(ng, "GeometryNodePointsToVolume", (-750, 0))
    p2v.inputs["Resolution Mode"].default_value = "Size"
    p2v.inputs["Voxel Size"].default_value = A.ej_voxel
    p2v.inputs["Density"].default_value = 1.0
    L(ng, pts.outputs["Points"], p2v.inputs["Points"])
    rad = N(ng, "GeometryNodeInputNamedAttribute", (-1000, -200), data_type="FLOAT")
    rad.inputs["Name"].default_value = "kernel_r"
    L(ng, rad.outputs["Attribute"], p2v.inputs["Radius"])
    get = N(ng, "GeometryNodeGetNamedGrid", (-500, 0), data_type="FLOAT")
    get.inputs["Name"].default_value = "density"
    get.inputs["Remove"].default_value = True
    L(ng, p2v.outputs[0], get.inputs["Volume"])
    dil = N(ng, "GeometryNodeGridDilateAndErode", (-300, -150), data_type="FLOAT")
    dil.inputs["Steps"].default_value = 2
    L(ng, get.outputs["Grid"], dil.inputs["Grid"])
    # speed grid on the same topology, nearest particle
    f2g = N(ng, "GeometryNodeFieldToGrid", (150, -400), data_type="FLOAT")
    f2g.grid_items.new("FLOAT", "speed")
    f2g.grid_items.new("FLOAT", "kdens")
    L(ng, dil.outputs[0], f2g.inputs["Topology"])
    pos = N(ng, "GeometryNodeInputPosition", (-300, -500))
    near = N(ng, "GeometryNodeSampleNearest", (-100, -500), domain="POINT")
    L(ng, pts.outputs["Points"], near.inputs["Geometry"])
    L(ng, pos.outputs[0], near.inputs["Sample Position"])
    natt = N(ng, "GeometryNodeInputNamedAttribute", (-100, -700), data_type="FLOAT")
    natt.inputs["Name"].default_value = "speed"
    smp = N(ng, "GeometryNodeSampleIndex", (100, -650), data_type="FLOAT", domain="POINT")
    L(ng, pts.outputs["Points"], smp.inputs["Geometry"])
    L(ng, natt.outputs["Attribute"], smp.inputs["Value"])
    L(ng, near.outputs["Index"], smp.inputs["Index"])
    L(ng, smp.outputs[0], f2g.inputs["speed"])
    # smooth kernel of the nearest particle: w_j * (1 - q^2)^3, q = |x - x_j| / r_j
    def sample(attr, dtype="FLOAT", y=-900):
        na = N(ng, "GeometryNodeInputNamedAttribute", (-100, y), data_type=dtype)
        na.inputs["Name"].default_value = attr
        si = N(ng, "GeometryNodeSampleIndex", (100, y), data_type=dtype, domain="POINT")
        L(ng, pts.outputs["Points"], si.inputs["Geometry"])
        L(ng, na.outputs["Attribute"], si.inputs["Value"])
        L(ng, near.outputs["Index"], si.inputs["Index"])
        return si.outputs[0]
    pj = N(ng, "GeometryNodeSampleIndex", (100, -1100), data_type="FLOAT_VECTOR", domain="POINT")
    L(ng, pts.outputs["Points"], pj.inputs["Geometry"])
    L(ng, pos.outputs[0], pj.inputs["Value"])
    L(ng, near.outputs["Index"], pj.inputs["Index"])
    dist = N(ng, "ShaderNodeVectorMath", (300, -1100), operation="DISTANCE")
    L(ng, pos.outputs[0], dist.inputs[0])
    L(ng, pj.outputs[0], dist.inputs[1])
    q = N(ng, "ShaderNodeMath", (450, -1100), operation="DIVIDE")
    L(ng, dist.outputs["Value"], q.inputs[0])
    L(ng, sample("kernel_r", y=-1300), q.inputs[1])
    q2 = N(ng, "ShaderNodeMath", (600, -1100), operation="MULTIPLY")
    L(ng, q.outputs[0], q2.inputs[0])
    L(ng, q.outputs[0], q2.inputs[1])
    om = N(ng, "ShaderNodeMath", (750, -1100), operation="SUBTRACT", use_clamp=True)
    om.inputs[0].default_value = 1.0
    L(ng, q2.outputs[0], om.inputs[1])
    cube = N(ng, "ShaderNodeMath", (900, -1100), operation="POWER")
    L(ng, om.outputs[0], cube.inputs[0])
    cube.inputs[1].default_value = 3.0
    kw = N(ng, "ShaderNodeMath", (1050, -1100), operation="MULTIPLY")
    L(ng, cube.outputs[0], kw.inputs[0])
    L(ng, sample("weight", y=-1500), kw.inputs[1])
    L(ng, kw.outputs[0], f2g.inputs["kdens"])
    wmean = N(ng, "GeometryNodeGridMean", (400, -700), data_type="FLOAT")
    wmean.inputs["Width"].default_value = max(A.ej_blur, 0)
    wmean.inputs["Iterations"].default_value = 2 if A.ej_blur > 0 else 0
    L(ng, f2g.outputs["kdens"], wmean.inputs["Grid"])
    smean = N(ng, "GeometryNodeGridMean", (400, -400), data_type="FLOAT")
    smean.inputs["Width"].default_value = 1
    smean.inputs["Iterations"].default_value = 1
    L(ng, f2g.outputs["speed"], smean.inputs["Grid"])
    st1 = N(ng, "GeometryNodeStoreNamedGrid", (500, 0), data_type="FLOAT")
    st1.inputs["Name"].default_value = "density"
    L(ng, get.outputs["Volume"], st1.inputs["Volume"])
    L(ng, wmean.outputs[0], st1.inputs["Grid"])
    st2 = N(ng, "GeometryNodeStoreNamedGrid", (750, 0), data_type="FLOAT")
    st2.inputs["Name"].default_value = "speed"
    L(ng, st1.outputs[0], st2.inputs["Volume"])
    L(ng, smean.outputs[0], st2.inputs["Grid"])
    setm = N(ng, "GeometryNodeSetMaterial", (1100, 0))
    setm.inputs["Material"].default_value = ejecta_material()
    L(ng, st2.outputs[0], setm.inputs["Geometry"])
    L(ng, setm.outputs[0], go.inputs[0])
    obj.modifiers.new("Volume", "NODES").node_group = ng
    return obj


# ----------------------------------------------------------------------------
# world, lights, camera, Didymos, compositor, device
# ----------------------------------------------------------------------------
def build_world(scene):
    world = bpy.data.worlds.new("Space")
    scene.world = world
    nt = world.node_tree
    nt.nodes.clear()
    out = N(nt, "ShaderNodeOutputWorld", (800, 0))
    bg = N(nt, "ShaderNodeBackground", (600, 0))
    L(nt, bg.outputs[0], out.inputs[0])
    if A.stars <= 0:
        bg.inputs["Color"].default_value = (0, 0, 0, 1)
        return
    tc = N(nt, "ShaderNodeTexCoord", (-600, 0))
    vor = N(nt, "ShaderNodeTexVoronoi", (-400, 0))
    vor.inputs["Scale"].default_value = 420.0
    L(nt, tc.outputs["Generated"], vor.inputs["Vector"])
    ramp = N(nt, "ShaderNodeValToRGB", (-150, 0))
    ramp.color_ramp.elements[0].color = (1, 1, 1, 1)
    ramp.color_ramp.elements[1].position = 0.035
    ramp.color_ramp.elements[1].color = (0, 0, 0, 1)
    L(nt, vor.outputs["Distance"], ramp.inputs["Fac"])
    nz = N(nt, "ShaderNodeTexNoise", (-400, -250))
    nz.inputs["Scale"].default_value = 60.0
    L(nt, tc.outputs["Generated"], nz.inputs["Vector"])
    mul = N(nt, "ShaderNodeMix", (150, 0), data_type="RGBA", blend_type="MULTIPLY")
    mul.inputs["Factor"].default_value = 1.0
    L(nt, ramp.outputs["Color"], mul.inputs["A"])
    L(nt, nz.outputs["Fac"], mul.inputs["B"])
    L(nt, mul.outputs["Result"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.6 * A.stars


def build_lights_camera(scene):
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = A.sun_strength
    sun.data.angle = math.radians(0.53)
    sun.rotation_euler = Vector(A.sun).normalized().to_track_quat("Z", "Y").to_euler()
    scene.collection.objects.link(sun)
    fill = bpy.data.objects.new("Fill", bpy.data.lights.new("Fill", "SUN"))
    fill.data.energy = 0.06 * A.sun_strength
    fill.rotation_euler = (-Vector(A.sun)).normalized().to_track_quat("Z", "Y").to_euler()
    scene.collection.objects.link(fill)
    tgt = bpy.data.objects.new("CamTarget", None)
    tgt.location = A.target
    scene.collection.objects.link(tgt)
    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    cam.data.lens = A.lens
    cam.data.clip_start = 0.5
    cam.data.clip_end = 2.0e5
    c = cam.constraints.new("TRACK_TO")
    c.target, c.track_axis, c.up_axis = tgt, "TRACK_NEGATIVE_Z", "UP_Y"
    scene.collection.objects.link(cam)
    scene.camera = cam


def build_didymos(scene):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=7, radius=380.0, location=A.didymos_pos)
    d = bpy.context.active_object
    d.name = "Didymos"
    d.scale = (1.0, 1.0, 0.82)
    tex = bpy.data.textures.new("DidymosRelief", "CLOUDS")
    tex.noise_scale, tex.noise_depth = 0.35, 4
    m = d.modifiers.new("Relief", "DISPLACE")
    m.texture, m.strength = tex, 18.0
    bpy.ops.object.shade_smooth()
    mat = bpy.data.materials.new("DidymosMat")
    b = mat.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (0.11, 0.10, 0.09, 1)
    b.inputs["Roughness"].default_value = 0.95
    d.data.materials.append(mat)


def setup_compositor(scene):
    if A.bloom <= 0:
        return
    try:
        ng = bpy.data.node_groups.new("Bloom", "CompositorNodeTree")
        ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        rl = N(ng, "CompositorNodeRLayers", (-300, 0))
        gl = N(ng, "CompositorNodeGlare", (0, 0))
        gl.inputs["Type"].default_value = "Bloom"
        gl.inputs["Quality"].default_value = "High"
        gl.inputs["Threshold"].default_value = 0.8
        gl.inputs["Strength"].default_value = A.bloom
        gl.inputs["Size"].default_value = 0.6
        go = N(ng, "NodeGroupOutput", (300, 0))
        L(ng, rl.outputs["Image"], gl.inputs["Image"])
        L(ng, gl.outputs["Image"], go.inputs[0])
        scene.compositing_node_group = ng
        scene.render.use_compositing = True
    except Exception as e:
        print("bloom disabled:", e)


def setup_device(scene):
    scene.cycles.device = "CPU"
    if A.device.upper() == "CPU":
        return "CPU"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    kinds = ["OPTIX", "CUDA", "METAL", "HIP", "ONEAPI"] if A.device == "auto" else [A.device.upper()]
    for kind in kinds:
        try:
            prefs.compute_device_type = kind
        except TypeError:
            continue
        prefs.get_devices()
        devs = [d for d in prefs.devices if d.type == kind]
        if devs:
            for d in prefs.devices:
                d.use = d.type == kind
            scene.cycles.device = "GPU"
            return f"{kind}: " + ", ".join(d.name for d in devs)
    return "CPU (no GPU found)"


def setup_render(scene):
    scene.render.engine = "CYCLES"
    print("Cycles device:", setup_device(scene))
    c = scene.cycles
    c.samples = A.samples
    c.use_denoising = True
    c.max_bounces = 6
    c.volume_bounces = 2
    c.transparent_max_bounces = 8
    scene.render.resolution_x, scene.render.resolution_y = A.res
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "16"
    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    if A.timestamp:
        r = scene.render
        for p in dir(r):
            if p.startswith("use_stamp_") and p != "use_stamp_note":
                try:
                    setattr(r, p, False)
                except (AttributeError, TypeError):
                    pass
        r.use_stamp = True
        r.use_stamp_note = True
        r.stamp_font_size = max(16, A.res[1] // 40)
        r.stamp_background = (0, 0, 0, 0)
    setup_compositor(scene)


def build_scene():
    scene = bpy.context.scene
    for o in list(scene.collection.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for c in list(scene.collection.children):
        scene.collection.children.unlink(c)
    build_body(scene)
    build_ejecta(scene)
    build_world(scene)
    build_lights_camera(scene)
    if A.didymos:
        build_didymos(scene)
    setup_render(scene)
    return scene


# ----------------------------------------------------------------------------
# per-frame update
# ----------------------------------------------------------------------------
def set_points(obj, x, *attrs):
    """attrs: (name, data_type, field, values) tuples"""
    mesh = obj.data
    n = len(x)
    if len(mesh.vertices) != n:
        mesh.clear_geometry()
        mesh.vertices.add(n)
    if n:
        mesh.vertices.foreach_set("co", x.astype(np.float32).ravel())
    for name, dtype, field, values in attrs:
        at = mesh.attributes.get(name)
        if at is None or at.data_type != dtype:
            if at is not None:
                mesh.attributes.remove(at)
            at = mesh.attributes.new(name, dtype, "POINT")
        if n:
            at.data.foreach_set(field, values.ravel())
    mesh.update()


def place_camera(scene, u):
    az0, el0, d0 = A.cam
    az1, el1, d1 = A.cam_end if A.cam_end else (az0 + 25.0, el0 + 4.0, d0 * 1.6)
    e = u * u * (3 - 2 * u)
    az, el, d = (math.radians(az0 + e * (az1 - az0)), math.radians(el0 + e * (el1 - el0)),
                 d0 + e * (d1 - d0))
    scene.camera.location = Vector(A.target) + d * Vector(
        (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))


def main():
    print(f"render_dimorphos_vol.py {VERSION}, Blender {bpy.app.version_string}, "
          f"{len(TIMES)} dump(s), t = {TIMES[0]:.3f} .. {TIMES[-1]:.3f} s")
    if A.reuse and BODY in bpy.data.objects:
        scene = bpy.context.scene
        print("Cycles device:", setup_device(scene))
    else:
        scene = build_scene()
    body, ejecta = bpy.data.objects[BODY], bpy.data.objects[EJECTA]
    os.makedirs(A.out, exist_ok=True)
    scene.frame_start, scene.frame_end = 0, A.frames - 1
    # jobs: (blender frame, simulation time, camera position 0..1, output file, skip if exists)
    jobs = []
    if A.dump or A.time:
        for spec in A.dump or []:
            k = find_dump(spec)
            t = float(TIMES[k])
            name = os.path.splitext(INDEX["files"][k])[0].replace("frame_", "dump_")
            jobs.append((0, t, film_u(t), f"{name}.png", False))
        for t in A.time or []:
            jobs.append((0, t, film_u(t), f"time_{t:010.4f}s.png", False))
    else:
        f0, f1 = A.range if A.range else (0, A.frames - 1)
        for f in range(f0, f1 + 1):
            jobs.append((f, film_time(f), f / max(A.frames - 1, 1), f"frame_{f:04d}.png", True))
    first = True
    for f, t, u, fname, skip in jobs:
        path = os.path.abspath(os.path.join(A.out, fname))
        if skip and os.path.exists(path) and not (A.save_blend or A.no_render or A.diag):
            continue
        st = state_at(t)
        st["t"] = t
        (xb, cb), (xe, ue, re, we) = split_and_shade(st)
        set_points(body, xb, ("col", "FLOAT_COLOR", "color", cb))
        set_points(ejecta, xe, ("speed", "FLOAT", "value", ue),
                   ("kernel_r", "FLOAT", "value", re), ("weight", "FLOAT", "value", we))
        scene.frame_set(f)
        if not A.reuse:
            place_camera(scene, u)
        if A.timestamp:
            scene.render.stamp_note_text = f"DART → Dimorphos    t = {t:7.3f} s"
        if first and A.save_blend:
            bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(A.save_blend))
            print("saved", A.save_blend)
        first = False
        if A.no_render:
            break
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        print(f"{fname}  t = {t:8.4f} s  body {len(xb)}  ejecta {len(xe)}  -> {path}", flush=True)


main()
