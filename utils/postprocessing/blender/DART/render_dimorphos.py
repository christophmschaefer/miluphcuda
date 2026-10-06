"""
render_dimorphos.py -- cinematic Cycles rendering of miluphcuda SPH output
(frames prepared by sph2blender.py). Tested with Blender 5.2.

Particles become real spheres (Geometry Nodes, Mesh to Points) shaded in
Cycles; colour/emission/radius are computed per particle in numpy each
frame and passed as attributes. Between SPH dumps positions are
interpolated with cubic Hermite splines (x and v at both ends), so you get
a smooth film from a few dozen dumps.

Usage (everything after "--" goes to this script):

  # quick look: 1 frame, saves the scene for tweaking in the GUI
  blender -b -P render_dimorphos.py -- --data blender_frames --out renders \
          --frames 240 --range 120 120 --samples 32 --save-blend dart.blend

  # full film on a GPU node (resumable, skips existing PNGs)
  blender -b -P render_dimorphos.py -- --data blender_frames --out renders \
          --frames 240 --timemap log --samples 128 --device auto

  # after tweaking camera/lights/materials in dart.blend in the GUI:
  blender -b dart.blend -P render_dimorphos.py -- --reuse --data blender_frames ...

  ffmpeg -framerate 30 -i renders/frame_%04d.png -c:v libx264 -crf 16 \
         -pix_fmt yuv420p dart.mp4
"""
import argparse
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

# ----------------------------------------------------------------------------
# arguments
# ----------------------------------------------------------------------------
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser(prog="render_dimorphos.py")
ap.add_argument("--data", required=True, help="directory from sph2blender.py")
ap.add_argument("--out", default="renders")
ap.add_argument("--frames", type=int, default=240, help="number of film frames")
ap.add_argument("--range", type=int, nargs=2, metavar=("F0", "F1"),
                help="render only film frames F0..F1 (for job arrays)")
ap.add_argument("--timemap", choices=["linear", "log"], default="log",
                help="log: slow motion early, fast later (crater growth)")
ap.add_argument("--tau", type=float, default=0.05, help="time scale [s] of the log map")
ap.add_argument("--tmax", type=float, help="end time [s] (default: last dump)")
ap.add_argument("--res", type=int, nargs=2, default=[1920, 1080])
ap.add_argument("--samples", type=int, default=64)
ap.add_argument("--device", default="auto", help="auto|CPU|OPTIX|CUDA|METAL|HIP|ONEAPI")
ap.add_argument("--color", choices=["rock", "damage", "compaction"], default="rock",
                help="colouring of the slow (non-ejecta) material")
ap.add_argument("--glow", type=float, default=6.0,
                help="emission strength of fast ejecta (0 = physical, sunlit only)")
ap.add_argument("--vglow", type=float, nargs=2, default=[0.3, 3.0], metavar=("LO", "HI"),
                help="speed range [m/s] over which material blends into the glow colormap")
ap.add_argument("--vcmap", type=float, nargs=2, default=[0.3, 300.0], metavar=("VMIN", "VMAX"),
                help="speed range [m/s] of the colormap (log)")
ap.add_argument("--radius", type=float, default=0.39, help="sphere radius [m] (~a/2)")
ap.add_argument("--target", type=float, nargs=3, default=[-4.0, -75.0, -8.0],
                help="camera look-at point [m]")
ap.add_argument("--cam", type=float, nargs=3, default=[-35.0, 14.0, 420.0],
                metavar=("AZ", "EL", "DIST"),
                help="camera azimuth/elevation [deg] around +z and distance [m] at start")
ap.add_argument("--cam-end", type=float, nargs=3, metavar=("AZ", "EL", "DIST"),
                help="camera at the end of the film (default: orbit 25 deg, 1.6x dist)")
ap.add_argument("--lens", type=float, default=50.0)
ap.add_argument("--sun", type=float, nargs=3, default=[0.55, -0.75, 0.35],
                help="direction TO the sun")
ap.add_argument("--sun-strength", type=float, default=7.0)
ap.add_argument("--bloom", type=float, default=0.6, help="compositor bloom strength, 0 = off")
ap.add_argument("--stars", type=float, default=1.0, help="star field brightness, 0 = off")
ap.add_argument("--didymos", action="store_true", help="add Didymos in the background")
ap.add_argument("--didymos-pos", type=float, nargs=3, default=[-1100.0, 1500.0, -350.0])
ap.add_argument("--timestamp", action="store_true", help="burn 't = ... s' into the frames")
ap.add_argument("--reuse", action="store_true",
                help="use scene/camera/materials of the opened .blend, only swap the data")
ap.add_argument("--save-blend", help="save the scene (with the first rendered frame loaded)")
ap.add_argument("--no-render", action="store_true")
A = ap.parse_args(argv)

PARTICLES = "SPH_Particles"

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
    """Particle state at time t (Hermite between bracketing dumps)."""
    k = int(np.clip(np.searchsorted(TIMES, t, side="right") - 1, 0, len(TIMES) - 1))
    a = load(k)
    if k == len(TIMES) - 1 or t <= TIMES[k] or INDEX["N"][k] != INDEX["N"][k + 1]:
        return dict(x=a["x"].astype(np.float64), v=a["v"].astype(np.float32), mat=a["mat"],
                    dmg=a["dmg"] / 255.0, comp=a["comp"] / 255.0)
    b = load(k + 1)
    _, ia, ib = np.intersect1d(a["ids"], b["ids"], assume_unique=True, return_indices=True)
    dt = TIMES[k + 1] - TIMES[k]
    s = (t - TIMES[k]) / dt
    h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
    h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
    x0, x1 = a["x"][ia].astype(np.float64), b["x"][ib].astype(np.float64)
    v0, v1 = a["v"][ia].astype(np.float64), b["v"][ib].astype(np.float64)
    x = h00 * x0 + h10 * dt * v0 + h01 * x1 + h11 * dt * v1
    # guard against Hermite overshoot for particles with inconsistent v (e.g. reflections)
    lin = (1 - s) * x0 + s * x1
    bad = np.linalg.norm(x - lin, axis=1) > 0.5 * np.linalg.norm(x1 - x0, axis=1) + 1.0
    x[bad] = lin[bad]
    v = (1 - s) * v0 + s * v1
    lerp = lambda q: ((1 - s) * a[q][ia] + s * b[q][ib]) / 255.0
    return dict(x=x, v=v.astype(np.float32), mat=a["mat"][ia], dmg=lerp("dmg"), comp=lerp("comp"))


def film_time(f):
    t0, t1 = TIMES[0], (A.tmax if A.tmax is not None else TIMES[-1])
    u = f / max(A.frames - 1, 1)
    if A.timemap == "linear":
        return t0 + u * (t1 - t0)
    return t0 + A.tau * (math.exp(u * math.log1p((t1 - t0) / A.tau)) - 1.0)


# ----------------------------------------------------------------------------
# colours (linear RGB)
# ----------------------------------------------------------------------------
INFERNO = np.array([[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66],
                    [245, 125, 21], [250, 193, 39], [252, 255, 164]]) / 255.0


def cmap(u, table=INFERNO):
    u = np.clip(u, 0, 1) * (len(table) - 1)
    i = np.minimum(u.astype(int), len(table) - 2)
    w = (u - i)[:, None]
    srgb = (1 - w) * table[i] + w * table[i + 1]
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


ROCK = {0: (0.115, 0.100, 0.085), 1: (0.070, 0.066, 0.062), 2: (1.0, 0.85, 0.55)}


def shade(st):
    n = len(st["x"])
    mat = st["mat"].astype(int)
    col = np.zeros((n, 3))
    for m, c in ROCK.items():
        col[mat == m] = c
    col[mat > 2] = ROCK[0]
    if A.color == "damage":
        hot = cmap(0.35 + 0.6 * st["dmg"])
        w = (mat == 1) * st["dmg"]
        col = (1 - w[:, None] * 0.85) * col + (w[:, None] * 0.85) * hot * 0.6
    elif A.color == "compaction":
        teal = np.array([0.02, 0.35, 0.45])
        w = np.clip(st["comp"] * 1.5, 0, 1)[:, None]
        col = (1 - w) * col + w * teal

    vbulk = np.median(st["v"].astype(np.float64), axis=0)
    spd = np.linalg.norm(st["v"].astype(np.float64) - vbulk, axis=1)
    lv = np.log10(np.maximum(spd, 1e-6))
    w = smoothstep(math.log10(A.vglow[0]), math.log10(A.vglow[1]), lv)
    lo, hi = math.log10(A.vcmap[0]), math.log10(A.vcmap[1])
    hotc = cmap(0.25 + 0.75 * (lv - lo) / (hi - lo))
    col = (1 - w[:, None]) * col + w[:, None] * hotc
    emit = w * A.glow
    emit[mat == 2] = max(A.glow, 3.0) * 3
    rad = np.full(n, A.radius) * (1 - 0.35 * w)
    rad[mat == 2] = 0.12
    rgba = np.ones((n, 4), np.float32)
    rgba[:, :3] = col
    return rgba, emit.astype(np.float32), rad.astype(np.float32)


# ----------------------------------------------------------------------------
# scene construction
# ----------------------------------------------------------------------------
def node_tree(idblock):
    """Material/world node tree; Blender 4.x needs use_nodes = True first."""
    if idblock.node_tree is None:
        idblock.use_nodes = True
    return idblock.node_tree


def nodes_new(tree, kind, loc=(0, 0)):
    n = tree.nodes.new(kind)
    n.location = loc
    return n


def build_particle_object(scene):
    mesh = bpy.data.meshes.new(PARTICLES)
    obj = bpy.data.objects.new(PARTICLES, mesh)
    scene.collection.objects.link(obj)

    # material: attributes col / emit drive a rough regolith BSDF + emission
    mat = bpy.data.materials.new("SPH_Material")
    nt = node_tree(mat)
    nt.nodes.clear()
    out = nodes_new(nt, "ShaderNodeOutputMaterial", (600, 0))
    bsdf = nodes_new(nt, "ShaderNodeBsdfPrincipled", (250, 0))
    acol = nodes_new(nt, "ShaderNodeAttribute", (-200, 100))
    acol.attribute_name = "col"
    aem = nodes_new(nt, "ShaderNodeAttribute", (-200, -150))
    aem.attribute_name = "emit"
    nt.links.new(acol.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(acol.outputs["Color"], bsdf.inputs["Emission Color"])
    nt.links.new(aem.outputs["Fac"], bsdf.inputs["Emission Strength"])
    bsdf.inputs["Roughness"].default_value = 0.92
    bsdf.inputs["Specular IOR Level"].default_value = 0.15
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    # geometry nodes: vertices -> spheres with per-point radius
    ng = bpy.data.node_groups.new("SPH_Spheres", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    if hasattr(ng, "is_modifier"):
        ng.is_modifier = True
    gi = nodes_new(ng, "NodeGroupInput", (-400, 0))
    go = nodes_new(ng, "NodeGroupOutput", (400, 0))
    m2p = nodes_new(ng, "GeometryNodeMeshToPoints", (-100, 0))
    rad = nodes_new(ng, "GeometryNodeInputNamedAttribute", (-350, -150))
    rad.data_type = "FLOAT"
    rad.inputs["Name"].default_value = "radius"
    smat = nodes_new(ng, "GeometryNodeSetMaterial", (150, 0))
    smat.inputs["Material"].default_value = mat
    ng.links.new(gi.outputs[0], m2p.inputs["Mesh"])
    ng.links.new(rad.outputs["Attribute"], m2p.inputs["Radius"])
    ng.links.new(m2p.outputs["Points"], smat.inputs["Geometry"])
    ng.links.new(smat.outputs["Geometry"], go.inputs[0])
    mod = obj.modifiers.new("Spheres", "NODES")
    mod.node_group = ng
    return obj


def build_world(scene):
    world = bpy.data.worlds.new("Space")
    scene.world = world
    nt = node_tree(world)
    nt.nodes.clear()
    out = nodes_new(nt, "ShaderNodeOutputWorld", (800, 0))
    bg = nodes_new(nt, "ShaderNodeBackground", (600, 0))
    nt.links.new(bg.outputs[0], out.inputs[0])
    if A.stars <= 0:
        bg.inputs["Color"].default_value = (0, 0, 0, 1)
        return
    tc = nodes_new(nt, "ShaderNodeTexCoord", (-600, 0))
    vor = nodes_new(nt, "ShaderNodeTexVoronoi", (-400, 0))
    vor.inputs["Scale"].default_value = 420.0
    vor.inputs["Randomness"].default_value = 1.0
    nt.links.new(tc.outputs["Generated"], vor.inputs["Vector"])
    ramp = nodes_new(nt, "ShaderNodeValToRGB", (-150, 0))
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (1, 1, 1, 1)
    ramp.color_ramp.elements[1].position = 0.035
    ramp.color_ramp.elements[1].color = (0, 0, 0, 1)
    nt.links.new(vor.outputs["Distance"], ramp.inputs["Fac"])
    noise = nodes_new(nt, "ShaderNodeTexNoise", (-400, -250))
    noise.inputs["Scale"].default_value = 60.0
    nt.links.new(tc.outputs["Generated"], noise.inputs["Vector"])
    mul = nodes_new(nt, "ShaderNodeMixRGB", (150, 0))
    mul.blend_type = "MULTIPLY"
    mul.inputs["Fac"].default_value = 1.0
    nt.links.new(ramp.outputs["Color"], mul.inputs["Color1"])
    nt.links.new(noise.outputs["Fac"], mul.inputs["Color2"])
    nt.links.new(mul.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.6 * A.stars


def build_lights_camera(scene):
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = A.sun_strength
    sun.data.angle = math.radians(0.53)
    sun.rotation_euler = Vector(A.sun).normalized().to_track_quat("Z", "Y").to_euler()
    scene.collection.objects.link(sun)
    # faint fill from the opposite side (light scattered from Didymos)
    fill = bpy.data.objects.new("Fill", bpy.data.lights.new("Fill", "SUN"))
    fill.data.energy = 0.08 * A.sun_strength
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
    c.target = tgt
    c.track_axis = "TRACK_NEGATIVE_Z"
    c.up_axis = "UP_Y"
    scene.collection.objects.link(cam)
    scene.camera = cam


def build_didymos(scene):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=7, radius=380.0, location=A.didymos_pos)
    d = bpy.context.active_object
    d.name = "Didymos"
    d.scale = (1.0, 1.0, 0.82)
    tex = bpy.data.textures.new("DidymosRelief", "CLOUDS")
    tex.noise_scale = 0.35
    tex.noise_depth = 4
    m = d.modifiers.new("Relief", "DISPLACE")
    m.texture = tex
    m.strength = 18.0
    bpy.ops.object.shade_smooth()
    mat = bpy.data.materials.new("DidymosMat")
    b = node_tree(mat).nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (0.11, 0.10, 0.09, 1)
    b.inputs["Roughness"].default_value = 0.95
    d.data.materials.append(mat)


def setup_compositor_4x(scene):
    try:
        scene.use_nodes = True
        nt = scene.node_tree
        nt.nodes.clear()
        rl = nodes_new(nt, "CompositorNodeRLayers", (-300, 0))
        gl = nodes_new(nt, "CompositorNodeGlare", (0, 0))
        gl.glare_type = "BLOOM" if "BLOOM" in [e.identifier for e in
                                               gl.bl_rna.properties["glare_type"].enum_items] else "FOG_GLOW"
        gl.quality = "HIGH"
        gl.threshold = 0.8
        gl.size = 8
        gl.mix = max(-1.0, min(1.0, A.bloom - 1.0))
        co = nodes_new(nt, "CompositorNodeComposite", (300, 0))
        nt.links.new(rl.outputs["Image"], gl.inputs["Image"])
        nt.links.new(gl.outputs["Image"], co.inputs["Image"])
        scene.render.use_compositing = True
    except Exception as e:
        print("bloom disabled:", e)


def setup_compositor(scene):
    if A.bloom <= 0:
        return
    if not hasattr(scene, "compositing_node_group"):
        return setup_compositor_4x(scene)
    try:
        ng = bpy.data.node_groups.new("Bloom", "CompositorNodeTree")
        ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        rl = nodes_new(ng, "CompositorNodeRLayers", (-300, 0))
        gl = nodes_new(ng, "CompositorNodeGlare", (0, 0))
        gl.inputs["Type"].default_value = "Bloom"
        gl.inputs["Quality"].default_value = "High"
        gl.inputs["Threshold"].default_value = 0.8
        gl.inputs["Strength"].default_value = A.bloom
        gl.inputs["Size"].default_value = 0.6
        go = nodes_new(ng, "NodeGroupOutput", (300, 0))
        ng.links.new(rl.outputs["Image"], gl.inputs["Image"])
        ng.links.new(gl.outputs["Image"], go.inputs[0])
        scene.compositing_node_group = ng
        scene.render.use_compositing = True
    except Exception as e:  # compositor API differs between Blender versions
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
    scene.cycles.samples = A.samples
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 4
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
            if p.startswith("use_stamp_") and p not in ("use_stamp_note",):
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
    build_particle_object(scene)
    build_world(scene)
    build_lights_camera(scene)
    if A.didymos:
        build_didymos(scene)
    setup_render(scene)
    return scene


# ----------------------------------------------------------------------------
# per-frame update
# ----------------------------------------------------------------------------
def set_attr(mesh, name, dtype, field, values):
    at = mesh.attributes.get(name)
    if at is None or at.data_type != dtype:
        if at is not None:
            mesh.attributes.remove(at)
        at = mesh.attributes.new(name, dtype, "POINT")
    at.data.foreach_set(field, values.ravel())


def update_particles(obj, st):
    mesh = obj.data
    n = len(st["x"])
    if len(mesh.vertices) != n:
        mesh.clear_geometry()
        mesh.vertices.add(n)
    mesh.vertices.foreach_set("co", st["x"].astype(np.float32).ravel())
    rgba, emit, rad = shade(st)
    set_attr(mesh, "col", "FLOAT_COLOR", "color", rgba)
    set_attr(mesh, "emit", "FLOAT", "value", emit)
    set_attr(mesh, "radius", "FLOAT", "value", rad)
    mesh.update()


def place_camera(scene, u):
    cam = scene.camera
    az0, el0, d0 = A.cam
    az1, el1, d1 = A.cam_end if A.cam_end else (az0 + 25.0, el0 + 4.0, d0 * 1.6)
    e = u * u * (3 - 2 * u)  # ease in/out
    az, el, d = (math.radians(az0 + e * (az1 - az0)), math.radians(el0 + e * (el1 - el0)),
                 d0 + e * (d1 - d0))
    t = Vector(A.target)
    cam.location = t + d * Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az),
                                   math.sin(el)))


# ----------------------------------------------------------------------------
def main():
    if A.reuse and PARTICLES in bpy.data.objects:
        scene = bpy.context.scene  # keep everything tweaked in the GUI
        print("Cycles device:", setup_device(scene))
    else:
        scene = build_scene()
    obj = bpy.data.objects[PARTICLES]
    os.makedirs(A.out, exist_ok=True)
    f0, f1 = A.range if A.range else (0, A.frames - 1)
    scene.frame_start, scene.frame_end = 0, A.frames - 1
    first = True
    for f in range(f0, f1 + 1):
        path = os.path.abspath(os.path.join(A.out, f"frame_{f:04d}.png"))
        if os.path.exists(path) and not A.save_blend:
            continue
        t = film_time(f)
        st = state_at(t)
        update_particles(obj, st)
        scene.frame_set(f)
        if not A.reuse:
            place_camera(scene, f / max(A.frames - 1, 1))
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
        print(f"frame {f:4d}  t = {t:8.4f} s  N = {len(st['x'])}  -> {path}", flush=True)


main()
