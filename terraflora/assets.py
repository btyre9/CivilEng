# SPDX-License-Identifier: GPL-3.0-or-later
"""
The plant library: low-poly trees and shrubs built in code (a few hundred triangles each),
kept in collections that are not linked to the scene, so they only appear as forest instances.

- Pines: tall southern-pine shapes (bare trunk, clumped crown) and a conical conifer.
- Leafy trees, summer: trunk with a rounded, lumpy crown.
- Leafy trees, winter: the same trees bare, as branching limbs.
- Shrubs: low lumpy bushes.

Each instance gets a random 'tf_tint' (set by the forest nodes); the materials use it to vary
the shade from tree to tree.
"""

import math
import random

import bmesh
import bpy
from mathutils import Matrix, Vector

COLLECTIONS = {
    "PINES": "TF Pines",
    "LEAFY_SUMMER": "TF Leafy Summer",
    "LEAFY_WINTER": "TF Leafy Winter",
    "SHRUBS": "TF Shrubs",
}
ASSET_VERSION = 1


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
    """Creates the plant collections once; returns {kind: collection}."""
    cols = {k: bpy.data.collections.get(n) for k, n in COLLECTIONS.items()}
    if all(c is not None and len(c.objects) for c in cols.values()):
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
        else:
            for i, s in enumerate((1.6, 2.2, 1.2)):
                _object("TF Shrub %d" % (i + 1), build_shrub(rng, s), [m["shrub"]], col)
    return cols
