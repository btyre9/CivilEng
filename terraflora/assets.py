# SPDX-License-Identifier: GPL-3.0-or-later
"""
The plant library: low-poly trees and shrubs built in code (a few hundred triangles each),
kept in collections that are not linked to the scene, so they only appear as forest instances.

- Pines: tall southern-pine shapes (bare trunk, clumped crown) and a conical conifer.
- Leafy trees, summer: trunk with a rounded, lumpy crown.
- Leafy trees, winter: the same trees bare, as branching limbs.
- Shrubs: low lumpy bushes.
- Grass tufts: a handful of blades; green in summer, straw in winter.
- Cards: every tree and shrub also has a far-away version, two crossed upright pictures and one
  flat one (6 triangles instead of hundreds), painted from the 3D model when the library is made.

Each instance gets a random 'tf_tint' (set by the forest nodes); the materials use it to vary
the shade from tree to tree.
"""

import math
import random

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

COLLECTIONS = {
    "PINES": "TF Pines",
    "LEAFY_SUMMER": "TF Leafy Summer",
    "LEAFY_WINTER": "TF Leafy Winter",
    "SHRUBS": "TF Shrubs",
    "GRASS_SUMMER": "TF Grass Summer",
    "GRASS_WINTER": "TF Grass Winter",
}
# Far-away cards for each library collection that has them (same models, same order).
CARDS = {"PINES": "TF Pine Cards", "LEAFY_SUMMER": "TF Leafy Summer Cards",
         "LEAFY_WINTER": "TF Leafy Winter Cards", "SHRUBS": "TF Shrub Cards"}
ASSET_VERSION = 3
CARD_PX_PER_M = 12.0
CARD_MAX_PX = 320
LIGHT = Vector((-0.45, -0.6, 0.66)).normalized()


def _material(name, color_a, color_b, roughness=0.9):
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name=name)
    mat.diffuse_color = tuple((a + b) * 0.5 for a, b in zip(color_a, color_b))
    nt = mat.node_tree
    if nt is None:
        return mat
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if bsdf is None:
        bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
        out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None) or nt.nodes.new('ShaderNodeOutputMaterial')
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    bsdf.inputs["Roughness"].default_value = roughness
    attr = nt.nodes.new('ShaderNodeAttribute')
    attr.attribute_type = 'INSTANCER'
    attr.attribute_name = "tf_tint"
    mix = nt.nodes.new('ShaderNodeMix')
    mix.data_type = 'RGBA'
    mix.inputs["A"].default_value = color_a
    mix.inputs["B"].default_value = color_b
    nt.links.new(attr.outputs["Fac"], mix.inputs["Factor"])
    nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    attr.location = (bsdf.location.x - 500, bsdf.location.y)
    mix.location = (bsdf.location.x - 250, bsdf.location.y)
    return mat


def materials():
    return {
        "bark": _material("TF Bark", (0.10, 0.065, 0.04, 1), (0.17, 0.11, 0.07, 1)),
        "needles": _material("TF Pine Needles", (0.025, 0.08, 0.03, 1), (0.06, 0.14, 0.045, 1)),
        "leaves": _material("TF Leaves", (0.07, 0.18, 0.04, 1), (0.16, 0.30, 0.06, 1)),
        "twigs": _material("TF Winter Branches", (0.13, 0.10, 0.08, 1), (0.22, 0.18, 0.14, 1)),
        "shrub": _material("TF Shrub", (0.06, 0.14, 0.04, 1), (0.14, 0.22, 0.07, 1)),
        "grass": _material("TF Grass", (0.07, 0.15, 0.03, 1), (0.17, 0.25, 0.06, 1)),
        "straw": _material("TF Dry Grass", (0.24, 0.19, 0.10, 1), (0.36, 0.29, 0.15, 1)),
    }


def _tube(bm, p0, p1, r0, r1, sides, mat):
    axis = (p1 - p0)
    if axis.length < 1e-6:
        return
    axis.normalize()
    ref = Vector((1, 0, 0)) if abs(axis.x) < 0.9 else Vector((0, 1, 0))
    u = axis.cross(ref).normalized()
    v = axis.cross(u)
    ring0, ring1 = [], []
    for k in range(sides):
        a = 2.0 * math.pi * k / sides
        d = u * math.cos(a) + v * math.sin(a)
        ring0.append(bm.verts.new(p0 + d * r0))
        ring1.append(bm.verts.new(p1 + d * r1))
    for k in range(sides):
        f = bm.faces.new((ring0[k], ring0[(k + 1) % sides], ring1[(k + 1) % sides], ring1[k]))
        f.material_index = mat
    if r1 > 1e-4:
        f = bm.faces.new(list(reversed(ring1)))
        f.material_index = mat


def _blob(bm, center, radius, squash, rng, mat, jitter=0.18, subdivisions=1):
    res = bmesh.ops.create_icosphere(bm, subdivisions=subdivisions, radius=radius,
                                     matrix=Matrix.Translation(center))
    verts = res["verts"]
    for v in verts:
        off = v.co - center
        off.z *= squash
        off *= 1.0 + rng.uniform(-jitter, jitter)
        v.co = center + off
    for f in {f for v in verts for f in v.link_faces}:
        f.material_index = mat


def _cone(bm, base, radius, height, sides, mat):
    res = bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=sides, radius1=radius,
                                radius2=0.0, depth=height, matrix=Matrix.Translation(base + Vector((0, 0, height * 0.5))))
    for f in {f for v in res["verts"] for f in v.link_faces}:
        f.material_index = mat


def build_pine(rng, height, conical):
    bm = bmesh.new()
    if conical:
        _tube(bm, Vector((0, 0, 0)), Vector((0, 0, height * 0.35)), 0.28, 0.2, 6, 0)
        tiers = 4
        for t in range(tiers):
            f = t / tiers
            base = Vector((0, 0, height * (0.18 + 0.6 * f)))
            _cone(bm, base, height * (0.2 - 0.13 * f), height * (0.42 - 0.12 * f), 8, 1)
    else:
        # Southern pine: tall clean trunk, irregular crown in the top third.
        top = height * 0.78
        _tube(bm, Vector((0, 0, 0)), Vector((0, 0, top)), 0.32, 0.12, 6, 0)
        crown_r = height * 0.17
        for k in range(rng.randint(5, 7)):
            a = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(0.2, 0.9) * crown_r
            z = rng.uniform(0.72, 0.95) * height
            c = Vector((math.cos(a) * r, math.sin(a) * r, z))
            _blob(bm, c, crown_r * rng.uniform(0.55, 0.8), 0.6, rng, 1)
            _tube(bm, Vector((0, 0, z - crown_r * 0.6)), c, 0.1, 0.05, 4, 0)
    return bm


def build_leafy(rng, height, winter):
    bm = bmesh.new()
    fork = height * rng.uniform(0.35, 0.45)
    _tube(bm, Vector((0, 0, 0)), Vector((0, 0, fork)), 0.35, 0.25, 6, 0)
    crown_r = height * rng.uniform(0.32, 0.4)
    crown_c = Vector((0, 0, fork + crown_r * 0.9))
    if not winter:
        _blob(bm, crown_c, crown_r * 0.8, 0.85, rng, 1)
        for k in range(rng.randint(4, 6)):
            a = rng.uniform(0, 2 * math.pi)
            d = Vector((math.cos(a), math.sin(a), rng.uniform(-0.3, 0.6))) * crown_r * rng.uniform(0.45, 0.75)
            _blob(bm, crown_c + d, crown_r * rng.uniform(0.45, 0.65), 0.85, rng, 1)
    else:
        # Bare limbs: main branches up and out from the fork, each splitting once or twice.
        base = Vector((0, 0, fork))
        for k in range(rng.randint(4, 6)):
            a = rng.uniform(0, 2 * math.pi)
            up = rng.uniform(0.9, 1.6)
            d = Vector((math.cos(a), math.sin(a), up)).normalized()
            length = height * rng.uniform(0.3, 0.42)
            tip = base + d * length
            _tube(bm, base, tip, 0.16, 0.05, 5, 1)
            for s in range(rng.randint(2, 3)):
                f = rng.uniform(0.45, 0.85)
                start = base.lerp(tip, f)
                b = rng.uniform(0, 2 * math.pi)
                d2 = (d + Vector((math.cos(b), math.sin(b), 0.4)) * 0.8).normalized()
                _tube(bm, start, start + d2 * length * rng.uniform(0.35, 0.55), 0.06, 0.015, 4, 1)
    return bm


def build_shrub(rng, size):
    bm = bmesh.new()
    for k in range(rng.randint(1, 3)):
        a = rng.uniform(0, 2 * math.pi)
        r = size * rng.uniform(0.0, 0.5)
        c = Vector((math.cos(a) * r, math.sin(a) * r, size * 0.45))
        _blob(bm, c, size * rng.uniform(0.5, 0.75), 0.7, rng, 0, jitter=0.25)
    return bm


def build_grass(rng, height):
    """A tuft: thin tapered blades leaning out from one spot (2 triangles each)."""
    bm = bmesh.new()
    for k in range(rng.randint(7, 10)):
        a = rng.uniform(0, 2 * math.pi)
        lean = rng.uniform(0.15, 0.55)
        h = height * rng.uniform(0.6, 1.0)
        w = rng.uniform(0.025, 0.045)
        out = Vector((math.cos(a), math.sin(a), 0.0))
        side = Vector((-math.sin(a), math.cos(a), 0.0))
        base = out * rng.uniform(0.0, 0.08)
        mid = base + out * (lean * h * 0.4) + Vector((0, 0, h * 0.55))
        tip = base + out * (lean * h) + Vector((0, 0, h))
        v = [bm.verts.new(base - side * w), bm.verts.new(base + side * w),
             bm.verts.new(mid + side * w * 0.6), bm.verts.new(mid - side * w * 0.6), bm.verts.new(tip)]
        bm.faces.new((v[0], v[1], v[2], v[3]))
        bm.faces.new((v[3], v[2], v[4]))
    return bm


# ------------------------------------------------------------------------------
# Cards: pictures of the 3D models for trees far from the camera
# ------------------------------------------------------------------------------

def _mid_color(mat):
    """The material's middle colour (between its two tints)."""
    nt = mat.node_tree
    mix = next((n for n in nt.nodes if n.type == 'MIX'), None) if nt else None
    if mix is None:
        return np.array(mat.diffuse_color[:3])
    a = np.array(mix.inputs["A"].default_value[:3])
    b = np.array(mix.inputs["B"].default_value[:3])
    return (a + b) * 0.5


def _raster(tris, colors, u_axis, v_axis, d_axis, u0, v0, w, h, ppm):
    """Paints triangles (T, 3, 3) seen along d_axis into an RGBA image (h, w, 4), nearest wins."""
    img = np.zeros((h, w, 4), dtype=np.float32)
    zbuf = np.full((h, w), np.inf)
    U = (tris @ u_axis - u0) * ppm
    V = (tris @ v_axis - v0) * ppm
    D = tris @ d_axis
    for t in range(len(tris)):
        u, v, d = U[t], V[t], D[t]
        i0, i1 = max(0, int(np.floor(u.min()))), min(w, int(np.ceil(u.max())) + 1)
        j0, j1 = max(0, int(np.floor(v.min()))), min(h, int(np.ceil(v.max())) + 1)
        if i1 <= i0 or j1 <= j0:
            continue
        px, py = np.meshgrid(np.arange(i0, i1) + 0.5, np.arange(j0, j1) + 0.5)
        den = (v[1] - v[2]) * (u[0] - u[2]) + (u[2] - u[1]) * (v[0] - v[2])
        if abs(den) < 1e-12:
            continue
        a = ((v[1] - v[2]) * (px - u[2]) + (u[2] - u[1]) * (py - v[2])) / den
        b = ((v[2] - v[0]) * (px - u[2]) + (u[0] - u[2]) * (py - v[2])) / den
        c = 1.0 - a - b
        inside = (a >= -1e-6) & (b >= -1e-6) & (c >= -1e-6)
        depth = a * d[0] + b * d[1] + c * d[2]
        zs = zbuf[j0:j1, i0:i1]
        win = inside & (depth < zs)
        if not win.any():
            continue
        zs[win] = depth[win]
        sub = img[j0:j1, i0:i1]
        sub[win, :3] = colors[t]
        sub[win, 3] = 1.0
    # Spread colour a few pixels into the transparent surround, so filtering shows no dark fringe.
    rgb, alpha = img[:, :, :3].copy(), img[:, :, 3].copy()
    known = alpha > 0
    for _ in range(4):
        P = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), mode="edge")
        K = np.pad(known, 1, mode="constant")
        acc = np.zeros_like(rgb)
        cnt = np.zeros(known.shape)
        for dj, di in ((0, 1), (2, 1), (1, 0), (1, 2)):
            k = K[dj:dj + known.shape[0], di:di + known.shape[1]]
            acc += P[dj:dj + known.shape[0], di:di + known.shape[1]] * k[:, :, None]
            cnt += k
        grow = (~known) & (cnt > 0)
        rgb[grow] = acc[grow] / cnt[grow][:, None]
        known = known | grow
    img[:, :, :3] = rgb
    return img


def bake_card(ob):
    """
    A card version of a plant model: (mesh, image). The image holds the side view on the left
    and the top view on the right; the mesh is two crossed upright quads showing the side and
    one flat quad, at crown height, showing the top.
    """
    me = ob.data
    me.calc_loop_triangles()
    co = np.empty((len(me.vertices), 3))
    me.vertices.foreach_get("co", co.ravel())
    nt = len(me.loop_triangles)
    vi = np.empty(nt * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", vi)
    mi = np.empty(nt, dtype=np.int64)
    me.loop_triangles.foreach_get("material_index", mi)
    nrm = np.empty((nt, 3))
    me.loop_triangles.foreach_get("normal", nrm.ravel())
    tris = co[vi].reshape(nt, 3, 3)
    base = np.array([_mid_color(me.materials[k]) if me.materials[k] else (0.2, 0.2, 0.2)
                     for k in range(len(me.materials))])
    shade = 0.45 + 0.75 * np.abs(nrm @ np.array(LIGHT))   # either side of thin parts catches light
    colors = np.clip(base[mi] * shade[:, None], 0.0, 1.0)

    lo, hi = co.min(axis=0), co.max(axis=0)
    r = max(abs(lo[0]), abs(hi[0]), abs(lo[1]), abs(hi[1])) * 1.02 + 0.05
    height = hi[2] * 1.01 + 0.05
    ppm = min(CARD_PX_PER_M, CARD_MAX_PX / max(height, 2 * r))
    sw, sh = max(4, int(math.ceil(2 * r * ppm))), max(4, int(math.ceil(height * ppm)))
    side = _raster(tris, colors, np.array((1.0, 0, 0)), np.array((0, 0, 1.0)), np.array((0, 1.0, 0)),
                   -r, 0.0, sw, sh, ppm)
    top = _raster(tris, colors, np.array((1.0, 0, 0)), np.array((0, 1.0, 0)), np.array((0, 0, -1.0)),
                  -r, -r, sw, sw, ppm)
    H = max(sh, sw)
    atlas = np.zeros((H, sw * 2, 4), dtype=np.float32)
    atlas[:sh, :sw] = side
    atlas[:sw, sw:] = top
    # Material colours are linear; the image stores sRGB.
    c = np.clip(atlas[:, :, :3], 0.0, 1.0)
    atlas[:, :, :3] = np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)
    img = bpy.data.images.get(ob.name + " Card")
    if img is not None:
        bpy.data.images.remove(img)
    img = bpy.data.images.new(ob.name + " Card", sw * 2, H, alpha=True)
    img.pixels.foreach_set(atlas.ravel())
    img.update()
    try:
        img.pack()
    except RuntimeError:
        pass

    # Crown height for the flat card: where the top view's leaves are thickest.
    leafy = tris[mi > 0] if (mi > 0).any() else tris
    crown_z = float(np.percentile(leafy[:, :, 2], 70))

    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    us, vs = 0.5, sh / H
    for ax in ((1, 0), (0, 1)):
        d = Vector((ax[0], ax[1], 0.0)) * r
        vs_ = [bm.verts.new(-d), bm.verts.new(d), bm.verts.new(d + Vector((0, 0, height))),
               bm.verts.new(-d + Vector((0, 0, height)))]
        f = bm.faces.new(vs_)
        for loop, (a, b) in zip(f.loops, ((0, 0), (us, 0), (us, vs), (0, vs))):
            loop[uv].uv = (a, b)
    vt = sw / H
    q = [bm.verts.new((-r, -r, crown_z)), bm.verts.new((r, -r, crown_z)),
         bm.verts.new((r, r, crown_z)), bm.verts.new((-r, r, crown_z))]
    f = bm.faces.new(q)
    for loop, (a, b) in zip(f.loops, ((0.5, 0), (1.0, 0), (1.0, vt), (0.5, vt))):
        loop[uv].uv = (a, b)
    cme = bpy.data.meshes.get(ob.name + " Card")
    if cme is not None:
        bpy.data.meshes.remove(cme)
    cme = bpy.data.meshes.new(ob.name + " Card")
    bm.to_mesh(cme)
    bm.free()
    return cme, img


def card_material(name, img):
    """Picture on a card: cut out by its alpha, shaded a little lighter or darker per tree."""
    mat = bpy.data.materials.get(name)
    if mat is not None:
        bpy.data.materials.remove(mat)
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (0.12, 0.2, 0.08, 1.0)
    try:
        mat.surface_render_method = 'DITHERED'
    except (AttributeError, TypeError):
        pass
    try:
        mat.blend_method = 'CLIP'
    except (AttributeError, TypeError):
        pass
    nt = mat.node_tree
    if nt is None:
        return mat
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if bsdf is None:
        bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
        out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None) or nt.nodes.new('ShaderNodeOutputMaterial')
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    bsdf.inputs["Roughness"].default_value = 0.9
    tex = nt.nodes.new('ShaderNodeTexImage')
    tex.image = img
    tex.extension = 'CLIP'
    tex.interpolation = 'Closest'
    tex.location = (bsdf.location.x - 700, bsdf.location.y)
    attr = nt.nodes.new('ShaderNodeAttribute')
    attr.attribute_type = 'INSTANCER'
    attr.attribute_name = "tf_tint"
    attr.location = (bsdf.location.x - 700, bsdf.location.y - 300)
    bright = nt.nodes.new('ShaderNodeMapRange')
    bright.inputs["To Min"].default_value = 0.8
    bright.inputs["To Max"].default_value = 1.2
    bright.location = (bsdf.location.x - 450, bsdf.location.y - 300)
    nt.links.new(attr.outputs["Fac"], bright.inputs["Value"])
    mix = nt.nodes.new('ShaderNodeMix')
    mix.data_type = 'RGBA'
    mix.blend_type = 'MULTIPLY'
    mix.inputs["Factor"].default_value = 1.0
    mix.location = (bsdf.location.x - 250, bsdf.location.y)
    nt.links.new(tex.outputs["Color"], mix.inputs["A"])
    nt.links.new(bright.outputs["Result"], mix.inputs["B"])
    nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    # Hard cut-out: soft edges would look like fog at a distance.
    cut = nt.nodes.new('ShaderNodeMath')
    cut.operation = 'GREATER_THAN'
    cut.inputs[1].default_value = 0.5
    cut.location = (bsdf.location.x - 250, bsdf.location.y - 250)
    nt.links.new(tex.outputs["Alpha"], cut.inputs[0])
    nt.links.new(cut.outputs[0], bsdf.inputs["Alpha"])
    return mat


def _object(name, bm, mats, collection):
    me = bpy.data.meshes.new(name)
    for m in mats:
        me.materials.append(m)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = False
    ob = bpy.data.objects.new(name, me)
    collection.objects.link(ob)
    return ob


def ensure_assets():
    """Creates the plant collections once; returns {kind: collection} (cards as kind + '_CARDS')."""
    names = dict(COLLECTIONS)
    names.update({k + "_CARDS": n for k, n in CARDS.items()})
    cols = {k: bpy.data.collections.get(n) for k, n in names.items()}
    built = bpy.data.collections.get(COLLECTIONS["PINES"])
    if all(c is not None and len(c.objects) for c in cols.values()) \
            and built is not None and built.get("tf_assets") == ASSET_VERSION:
        return cols
    m = materials()
    rng = random.Random(7)
    for kind, name in COLLECTIONS.items():
        col = cols[kind] or bpy.data.collections.new(name)
        cols[kind] = col
        for ob in list(col.objects):
            bpy.data.objects.remove(ob)
        if kind == "PINES":
            for i, (h, conical) in enumerate(((22.0, False), (26.0, False), (18.0, False), (14.0, True))):
                _object("TF Pine %d" % (i + 1), build_pine(rng, h, conical), [m["bark"], m["needles"]], col)
        elif kind == "LEAFY_SUMMER":
            r2 = random.Random(11)
            for i, h in enumerate((13.0, 16.0, 11.0)):
                _object("TF Leafy %d" % (i + 1), build_leafy(r2, h, False), [m["bark"], m["leaves"]], col)
        elif kind == "LEAFY_WINTER":
            r2 = random.Random(11)
            for i, h in enumerate((13.0, 16.0, 11.0)):
                _object("TF Bare %d" % (i + 1), build_leafy(r2, h, True), [m["bark"], m["twigs"]], col)
        elif kind == "SHRUBS":
            for i, s in enumerate((1.6, 2.2, 1.2)):
                _object("TF Shrub %d" % (i + 1), build_shrub(rng, s), [m["shrub"]], col)
        else:
            r3 = random.Random(5)
            winter = kind == "GRASS_WINTER"
            for i, h in enumerate((0.55, 0.75, 0.45)):
                _object("TF %s %d" % ("Straw" if winter else "Grass", i + 1), build_grass(r3, h),
                        [m["straw" if winter else "grass"]], col)
    # Cards, painted from the models just made.
    for kind, name in CARDS.items():
        col = cols[kind + "_CARDS"] or bpy.data.collections.new(name)
        cols[kind + "_CARDS"] = col
        for ob in list(col.objects):
            bpy.data.objects.remove(ob)
        for src in sorted(cols[kind].objects, key=lambda o: o.name):
            cme, img = bake_card(src)
            cme.materials.append(card_material(src.name + " Card", img))
            card = bpy.data.objects.new(src.name.replace("TF ", "TF Card ", 1), cme)
            col.objects.link(card)
    cols["PINES"]["tf_assets"] = ASSET_VERSION
    return cols
