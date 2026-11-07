# SPDX-License-Identifier: GPL-3.0-or-later
"""
Streetlight styles and utility lines for CivilFlow.

- Streetlights: the highway cobra head, plus classical park and downtown styles (acorn globe,
  Victorian twin lanterns, post-top lantern) and a modern LED blade. Roads use them through
  add_light(); "Path Lights" lines put them along any curve (park paths, plazas).
- Power lines along any curve: older wooden distribution poles with crossarms, modern compact
  distribution poles, wooden H-frame transmission, steel monopoles and steel lattice towers.
  Distribution poles can carry pole-mounted transformers; wires sag between poles.

A line is a mesh object that rebuilds itself from its curve (like the roads) and stands its
poles on CivilFlow terrain when there is some under them.
"""

import math
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

from . import terrain

# Material slots (the first twelve are the road's own; see ensure_civil_materials).
M_CONCRETE, M_STEEL, M_LAMP = 1, 6, 7
M_IRON, M_GLOBE, M_WOOD, M_PORCELAIN, M_WIRE, M_CAN, M_POLYMER = 12, 13, 14, 15, 16, 17, 18

UP = Vector((0.0, 0.0, 1.0))

LIGHT_STYLES = [
    ('COBRA', "Highway Cobra Head", "Tall steel pole with a mast arm over the road"),
    ('ACORN', "Classic Acorn Globe", "Black fluted cast-iron post with an acorn globe: parks, downtowns"),
    ('TWIN', "Victorian Twin Lantern", "Cast-iron post with a scrolled crossarm and two lanterns"),
    ('LANTERN', "Post-Top Lantern", "Short cast-iron post with a four-sided lantern"),
    ('MODERN', "Modern LED Blade", "Slim square pole with a flat LED blade head"),
]
# Natural height and spacing of each style (a new style choice starts from these).
LIGHT_DEFAULTS = {'COBRA': (9.5, 28.0), 'ACORN': (4.3, 18.0), 'TWIN': (4.8, 20.0),
                  'LANTERN': (3.8, 15.0), 'MODERN': (8.0, 26.0)}

POWER_STYLES = [
    ('WOOD', "Wood Pole, Crossarm (older)", "Classic wooden distribution pole: crossarm, pin insulators, transformers"),
    ('COMPACT', "Compact Pole (newer)", "Modern distribution pole with polymer post insulators and no crossarm"),
    ('H_FRAME', "Wood H-Frame Transmission (older)", "Two wooden poles and a crossbeam, porcelain insulator strings"),
    ('MONOPOLE', "Steel Monopole Transmission (newer)", "Tapered steel tube with arms for two circuits"),
    ('LATTICE', "Steel Lattice Tower", "Classic high-voltage steel lattice tower, two circuits"),
]
POWER_DEFAULTS = {'WOOD': (45.0, 0.022), 'COMPACT': (50.0, 0.02), 'H_FRAME': (180.0, 0.03),
                  'MONOPOLE': (230.0, 0.03), 'LATTICE': (320.0, 0.032)}
DISTRIBUTION = ('WOOD', 'COMPACT')


# ------------------------------------------------------------------------------
# Small mesh pieces
# ------------------------------------------------------------------------------

def _basis(axis):
    ref = Vector((1.0, 0.0, 0.0)) if abs(axis.x) < 0.9 else Vector((0.0, 1.0, 0.0))
    u = axis.cross(ref).normalized()
    return u, axis.cross(u).normalized()


def tube(bm, p0, p1, r0, r1=None, sides=8, mat=M_STEEL, caps=True, twist=0.0):
    """Tapered tube from p0 (radius r0) to p1 (radius r1)."""
    r1 = r0 if r1 is None else r1
    axis = p1 - p0
    if axis.length < 1e-6:
        return
    axis = axis.normalized()
    u, v = _basis(axis)
    ring0, ring1 = [], []
    for k in range(sides):
        a = 2.0 * math.pi * k / sides + twist
        d = u * math.cos(a) + v * math.sin(a)
        ring0.append(bm.verts.new(p0 + d * r0))
        ring1.append(bm.verts.new(p1 + d * r1))
    for k in range(sides):
        f = bm.faces.new((ring0[k], ring0[(k + 1) % sides], ring1[(k + 1) % sides], ring1[k]))
        f.material_index = mat
    if caps:
        if r0 > 1e-5:
            bm.faces.new(list(reversed(ring0))).material_index = mat
        if r1 > 1e-5:
            bm.faces.new(ring1).material_index = mat


def box(bm, center, ax, ay, hx, hy, hz, mat):
    """Box centred on `center`: half-sizes hx along ax, hy along ay, hz along their normal."""
    az = ax.cross(ay).normalized()
    v = [bm.verts.new(center + ax * (sx * hx) + ay * (sy * hy) + az * (sz * hz))
         for sz in (-1.0, 1.0) for sx, sy in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))]
    for idx in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
        bm.faces.new([v[i] for i in idx]).material_index = mat


def blob(bm, center, r, squash=1.0, mat=M_GLOBE, segs=12):
    res = bmesh.ops.create_uvsphere(bm, u_segments=segs, v_segments=max(6, segs // 2), radius=r,
                                    matrix=Matrix.Translation(center) @ Matrix.Diagonal((1.0, 1.0, squash, 1.0)))
    for f in {f for v in res["verts"] for f in v.link_faces}:
        f.material_index = mat


def wire(bm, a, b, sag, radius, segs=16, mat=M_WIRE):
    """A hanging wire from a to b, dipping `sag` metres at mid-span (parabola ~ catenary)."""
    prev = a
    for k in range(1, segs + 1):
        t = k / segs
        p = a.lerp(b, t) - UP * (4.0 * sag * t * (1.0 - t))
        tube(bm, prev, p, radius, sides=4, mat=mat, caps=False)
        prev = p


# ------------------------------------------------------------------------------
# Streetlights
# ------------------------------------------------------------------------------

def _cast_base(bm, base, h_skirt=0.65):
    """Fluted cast-iron base: a flared skirt and a ring."""
    tube(bm, base - UP * 0.05, base + UP * h_skirt, 0.25, 0.15, 12, M_IRON)
    tube(bm, base + UP * h_skirt, base + UP * (h_skirt + 0.07), 0.19, 0.19, 12, M_IRON)


def _lantern(bm, base, size=1.0):
    """Four-sided lantern on a short neck: glowing panes, a pyramid roof and a finial."""
    s = size
    tube(bm, base, base + UP * 0.12 * s, 0.05 * s, 0.11 * s, 4, M_IRON, twist=math.pi / 4)
    z0 = base + UP * 0.12 * s
    tube(bm, z0, z0 + UP * 0.42 * s, 0.11 * s, 0.2 * s, 4, M_GLOBE, twist=math.pi / 4)
    # Corner posts so it reads as a frame, not a glowing block.
    for k in range(4):
        a = math.pi / 4 + k * math.pi / 2
        d0 = Vector((math.cos(a), math.sin(a), 0.0))
        tube(bm, z0 + d0 * 0.12 * s, z0 + UP * 0.42 * s + d0 * 0.215 * s, 0.012 * s, sides=4, mat=M_IRON)
    roof = z0 + UP * 0.42 * s
    tube(bm, roof, roof + UP * 0.05 * s, 0.25 * s, 0.25 * s, 4, M_IRON, twist=math.pi / 4)
    tube(bm, roof + UP * 0.05 * s, roof + UP * 0.25 * s, 0.25 * s, 0.0, 4, M_IRON, twist=math.pi / 4)
    blob(bm, roof + UP * 0.29 * s, 0.035 * s, 1.0, M_IRON, 8)


def add_light(bm, base, toward, along, style, height, arm_len):
    """
    One streetlight standing at `base` (on the ground or pavement). `toward` is the horizontal
    direction to the road (where an arm reaches), `along` the road's direction.
    """
    toward = Vector((toward.x, toward.y, 0.0)).normalized()
    along = Vector((along.x, along.y, 0.0)).normalized()
    h = height
    if style == 'ACORN':
        _cast_base(bm, base)
        top = base + UP * (h - 0.8)
        tube(bm, base + UP * 0.72, top, 0.075, 0.062, 12, M_IRON)               # fluted shaft
        tube(bm, top, top + UP * 0.12, 0.11, 0.11, 12, M_IRON)                   # collar
        tube(bm, top + UP * 0.12, top + UP * 0.24, 0.08, 0.16, 12, M_IRON)       # cup
        blob(bm, top + UP * 0.52, 0.22, 1.35, M_GLOBE, 16)                       # acorn globe
        tube(bm, top + UP * 0.74, top + UP * 0.92, 0.15, 0.03, 12, M_IRON)       # cap
        blob(bm, top + UP * 0.95, 0.035, 1.0, M_IRON, 8)                         # finial
    elif style == 'TWIN':
        _cast_base(bm, base)
        arm_z = base + UP * (h - 0.75)
        tube(bm, base + UP * 0.72, arm_z, 0.08, 0.065, 12, M_IRON)
        tube(bm, arm_z, arm_z + UP * 0.35, 0.065, 0.04, 12, M_IRON)
        blob(bm, arm_z + UP * 0.4, 0.05, 1.0, M_IRON, 8)
        span = 0.6
        tube(bm, arm_z - along * span, arm_z + along * span, 0.03, sides=6, mat=M_IRON)
        for s in (-1.0, 1.0):
            end = arm_z + along * (s * span)
            # Scroll under the arm.
            prev = arm_z + along * (s * 0.12)
            for k in range(1, 7):
                a = math.pi * k / 6
                p = arm_z + along * (s * (0.12 + 0.2 * (1 - math.cos(a)))) - UP * (0.18 * math.sin(a))
                tube(bm, prev, p, 0.016, sides=4, mat=M_IRON, caps=False)
                prev = p
            _lantern(bm, end, 0.9)
    elif style == 'LANTERN':
        _cast_base(bm, base)
        top = base + UP * (h - 0.75)
        tube(bm, base + UP * 0.72, top, 0.075, 0.06, 12, M_IRON)
        tube(bm, top, top + UP * 0.08, 0.1, 0.1, 12, M_IRON)
        _lantern(bm, top + UP * 0.08, 1.15)
    elif style == 'MODERN':
        tube(bm, base - UP * 0.3, base + UP * 0.3, 0.2, 0.2, 4, M_CONCRETE, twist=math.pi / 4)
        top = base + UP * h
        tube(bm, base + UP * 0.3, top, 0.09, 0.06, 4, M_IRON, twist=math.pi / 4)
        reach = max(0.6, arm_len * 0.6)
        tip = top + toward * reach - UP * 0.05
        tube(bm, top - UP * 0.15, tip, 0.03, sides=4, mat=M_IRON)
        head = tip + toward * 0.3
        box(bm, head, toward, along, 0.42, 0.13, 0.03, M_IRON)
        box(bm, head - UP * 0.032, toward, along, 0.38, 0.1, 0.004, M_LAMP)
    else:  # COBRA: steel pole on a concrete plinth with a mast arm and an emissive head
        right = toward
        tang = along
        box(bm, base, right, tang, 0.22, 0.22, 0.35, M_CONCRETE)
        tube(bm, base, base + UP * h, 0.11, 0.11, 8, M_STEEL)
        tube(bm, base + UP * (h - 0.12), base + UP * (h - 0.12) + right * arm_len, 0.05, sides=6, mat=M_STEEL)
        head = base + right * (arm_len - 0.25) + UP * (h - 0.13)
        box(bm, head, right, tang, 0.38, 0.17, 0.07, M_STEEL)
        box(bm, head - UP * 0.09, right, tang, 0.32, 0.13, 0.02, M_LAMP)


# ------------------------------------------------------------------------------
# Power poles. Each returns its wire attachment points, in the same order on every pole.
# ------------------------------------------------------------------------------

def _insulator_pin(bm, p, mat=M_PORCELAIN):
    """Pin insulator standing on p; returns the wire point on top."""
    tube(bm, p, p + UP * 0.06, 0.02, 0.02, 6, M_STEEL)
    for k in range(2):
        z = p + UP * (0.06 + k * 0.07)
        tube(bm, z, z + UP * 0.05, 0.075 - 0.015 * k, 0.04, 8, mat)
    return p + UP * 0.2


def _insulator_string(bm, p, length, discs, r=0.13, mat=M_PORCELAIN):
    """Suspension string hanging from p; returns the wire point at its bottom."""
    tube(bm, p, p - UP * length, 0.025, sides=6, mat=M_STEEL)
    for k in range(discs):
        z = p - UP * (0.15 + (length - 0.3) * (k + 0.5) / discs)
        tube(bm, z + UP * 0.03, z - UP * 0.03, r * 0.35, r, 10, mat)
    return p - UP * length


def _transformer(bm, pole_c, across, along, z, phase_pt, pole_r):
    """A pole-mounted transformer can with its fuse cutout and drop wire."""
    side = -across
    c = pole_c + side * (pole_r + 0.34) + UP * z
    tube(bm, pole_c + UP * (z + 0.25), c + UP * 0.25, 0.04, sides=4, mat=M_STEEL)   # bracket
    tube(bm, pole_c + UP * (z - 0.25), c - UP * 0.25, 0.04, sides=4, mat=M_STEEL)
    tube(bm, c - UP * 0.45, c + UP * 0.42, 0.29, 0.29, 14, M_CAN)
    tube(bm, c + UP * 0.42, c + UP * 0.48, 0.31, 0.27, 14, M_CAN)                 # lid
    bush = c + UP * 0.48 + along * 0.12
    tube(bm, bush, bush + UP * 0.16, 0.035, 0.025, 6, M_PORCELAIN)                # bushing
    tube(bm, c - side * 0.28 + UP * 0.1, c - side * 0.28 + UP * 0.1 + along * 0.0 - side * 0.0 + UP * 0.12,
         0.03, sides=6, mat=M_PORCELAIN)
    cut = phase_pt - UP * 0.55
    tube(bm, cut, cut + UP * 0.35 + side * 0.08, 0.025, sides=6, mat=M_PORCELAIN)  # fuse cutout
    tube(bm, phase_pt, cut + UP * 0.35 + side * 0.08, 0.008, sides=4, mat=M_WIRE)
    tube(bm, cut, bush + UP * 0.16, 0.008, sides=4, mat=M_WIRE)                   # drop to the can


def pole_wood(bm, g, across, along, h, transformer):
    tube(bm, g - UP * 1.5, g + UP * h, 0.17, 0.11, 10, M_WOOD)
    arm_z = h - 0.55
    arm_c = g + UP * arm_z + along * 0.16
    box(bm, arm_c, across, along, 1.25, 0.05, 0.055, M_WOOD)
    for s in (-1.0, 1.0):
        tube(bm, g + UP * (arm_z - 0.75) + along * 0.12, arm_c + across * (s * 0.6) - UP * 0.05, 0.015,
             sides=4, mat=M_STEEL)
    pts = [_insulator_pin(bm, arm_c + across * x + UP * 0.055) for x in (-1.1, -0.4, 1.1)]
    # Neutral, and a low-voltage secondary below it.
    for z, x in ((h - 2.1, 0.22), (h - 3.4, 0.2)):
        p = g + UP * z + across * x
        tube(bm, g + UP * z, p, 0.025, sides=4, mat=M_STEEL)
        tube(bm, p - UP * 0.06, p + UP * 0.06, 0.05, 0.05, 8, M_PORCELAIN)
        pts.append(p)
    if transformer:
        _transformer(bm, g, across, along, h - 2.9, pts[0], 0.13)
    return pts


def pole_compact(bm, g, across, along, h, transformer):
    tube(bm, g - UP * 1.5, g + UP * h, 0.19, 0.12, 12, M_CAN)
    pts = []
    top = g + UP * h
    tube(bm, top, top + UP * 0.08, 0.06, 0.06, 6, M_STEEL)
    for k in range(5):  # vertical polymer post insulator on top
        z = top + UP * (0.08 + 0.07 * k)
        tube(bm, z, z + UP * 0.03, 0.07, 0.07, 8, M_POLYMER)
    tube(bm, top + UP * 0.08, top + UP * 0.45, 0.025, sides=6, mat=M_POLYMER)
    pts.append(top + UP * 0.47)
    for z, s in ((h - 0.8, 1.0), (h - 1.6, -1.0)):
        root = g + UP * z
        tip = root + across * (s * 0.75)
        tube(bm, root, root + across * (s * 0.18), 0.05, sides=6, mat=M_STEEL)
        tube(bm, root + across * (s * 0.18), tip, 0.03, sides=6, mat=M_POLYMER)
        for k in range(5):
            c = root + across * (s * (0.26 + 0.1 * k))
            tube(bm, c - across * (s * 0.015), c + across * (s * 0.015), 0.065, 0.065, 8, M_POLYMER)
        pts.append(tip + UP * 0.03)
    p = g + UP * (h - 2.9) + across * 0.22
    tube(bm, g + UP * (h - 2.9), p, 0.025, sides=4, mat=M_STEEL)
    pts.append(p)
    if transformer:
        _transformer(bm, g, across, along, h - 2.6, pts[2], 0.14)
    return pts


def pole_h_frame(bm, g, across, along, h, ground_at):
    half = 2.8
    legs = []
    for s in (-1.0, 1.0):
        foot = g + across * (s * half)
        gz = ground_at(foot)
        foot = Vector((foot.x, foot.y, gz if gz is not None else g.z))
        top = Vector((foot.x, foot.y, g.z + h))
        tube(bm, foot - UP * 2.0, top, 0.22, 0.15, 10, M_WOOD)
        legs.append((foot, top))
    beam_z = g.z + h - 1.0
    beam_c = Vector((g.x, g.y, beam_z))
    box(bm, beam_c, across, along, 4.3, 0.12, 0.16, M_WOOD)
    # X braces between the poles.
    lo, hi = beam_z - 5.5, beam_z - 0.6
    (fa, _), (fb, _) = legs
    a_lo, a_hi = Vector((fa.x, fa.y, lo)), Vector((fa.x, fa.y, hi))
    b_lo, b_hi = Vector((fb.x, fb.y, lo)), Vector((fb.x, fb.y, hi))
    tube(bm, a_lo, b_hi, 0.07, sides=6, mat=M_WOOD)
    tube(bm, b_lo, a_hi, 0.07, sides=6, mat=M_WOOD)
    pts = [_insulator_string(bm, beam_c + across * x - UP * 0.16, 1.5, 8) for x in (-3.9, 0.0, 3.9)]
    for foot, top in legs:  # shield wires on the pole tops
        pts.append(top + UP * 0.1)
    return pts


def pole_monopole(bm, g, across, along, h):
    tube(bm, g - UP * 0.4, g + UP * 0.4, 0.95, 0.95, 16, M_CONCRETE)
    tube(bm, g + UP * 0.4, g + UP * h, 0.6, 0.26, 16, M_STEEL)
    pts = []
    for z, reach in ((h - 2.5, 3.6), (h - 6.8, 4.0), (h - 11.0, 3.6)):
        r_pole = 0.6 - (0.34 * (z / h))
        for s in (-1.0, 1.0):
            root = g + UP * z + across * (s * r_pole * 0.8)
            tip = g + UP * (z + 0.5) + across * (s * reach)
            tube(bm, root - UP * 0.25, tip, 0.2, 0.08, 8, M_STEEL)
            pts.append(_insulator_string(bm, tip, 2.1, 9, r=0.09, mat=M_POLYMER))
    peak = g + UP * (h + 1.6)
    tube(bm, g + UP * h, peak, 0.2, 0.08, 8, M_STEEL)
    pts.append(peak)
    return pts


def pole_lattice(bm, g, across, along, h, ground_at):
    waist = h * 0.62
    base_half, waist_half, top_half = 4.6, 1.3, 0.9
    leg_r, brace_r = 0.11, 0.045

    def corner(z, half, sx, sy):
        return g + across * (sx * half) + along * (sy * half) + UP * z

    def half_at(z):
        return base_half + (waist_half - base_half) * (z / waist) if z <= waist else \
            waist_half + (top_half - waist_half) * ((z - waist) / (h - waist))

    levels = [0.0]
    z = 0.0
    while z < h - 0.5:
        z = min(h, z + max(2.5, 2.2 * half_at(z)))
        levels.append(z)
    cs = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    for a, b in zip(levels[:-1], levels[1:]):
        ha, hb = half_at(a), half_at(b)
        for k, (sx, sy) in enumerate(cs):
            tube(bm, corner(a, ha, sx, sy), corner(b, hb, sx, sy), leg_r, sides=4, mat=M_STEEL)
            nx, ny = cs[(k + 1) % 4]
            tube(bm, corner(b, hb, sx, sy), corner(b, hb, nx, ny), brace_r, sides=4, mat=M_STEEL)
            tube(bm, corner(a, ha, sx, sy), corner(b, hb, nx, ny), brace_r, sides=4, mat=M_STEEL)
            tube(bm, corner(a, ha, nx, ny), corner(b, hb, sx, sy), brace_r, sides=4, mat=M_STEEL)
    for sx, sy in cs:  # concrete footings
        f = corner(0.0, base_half, sx, sy)
        gz = ground_at(f)
        f = Vector((f.x, f.y, gz if gz is not None else g.z))
        tube(bm, f - UP * 1.0, f + UP * 0.4, 0.45, 0.45, 8, M_CONCRETE)
    pts = []
    for z, reach in ((h - 10.0, 6.0), (h - 5.0, 7.5), (h - 0.2, 6.0)):
        hz = half_at(z)
        for s in (-1.0, 1.0):
            tip = g + across * (s * reach) + UP * z
            for sy in (-1.0, 1.0):
                tube(bm, g + across * (s * hz) + along * (sy * hz) + UP * z, tip, brace_r * 1.4, sides=4, mat=M_STEEL)
                tube(bm, g + across * (s * hz) + along * (sy * hz) + UP * (z + 2.2), tip, brace_r, sides=4, mat=M_STEEL)
            pts.append(_insulator_string(bm, tip, 2.6, 12))
    # Earth peak with two small arms for the shield wires.
    top = g + UP * h
    peak = top + UP * 4.0
    for sx, sy in cs:
        tube(bm, corner(h, top_half, sx, sy), peak, brace_r * 1.5, sides=4, mat=M_STEEL)
    for s in (-1.0, 1.0):
        tip = peak - UP * 0.6 + across * (s * 3.0)
        tube(bm, peak - UP * 0.6, tip, brace_r * 1.4, sides=4, mat=M_STEEL)
        pts.append(tip)
    return pts


# ------------------------------------------------------------------------------
# Lines along curves
# ------------------------------------------------------------------------------

def curve_points(curve, depsgraph=None):
    """The curve as a world-space polyline, start to end."""
    dg = depsgraph or bpy.context.evaluated_depsgraph_get()
    ev = curve.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        cf = sys.modules[__package__]
        pts = cf.extract_ordered_polyline(me)
    finally:
        ev.to_mesh_clear()
    mw = curve.matrix_world
    return [mw @ p for p in pts]


def resample(pts, positions):
    """Points and horizontal directions at the given distances along a polyline."""
    P = np.array([tuple(p) for p in pts])
    seg = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)
    d = np.concatenate([[0.0], np.cumsum(seg)])
    out = []
    for s in positions:
        k = int(np.clip(np.searchsorted(d, s) - 1, 0, len(P) - 2))
        t = 0.0 if seg[k] < 1e-9 else (s - d[k]) / seg[k]
        p = Vector(P[k] + (P[k + 1] - P[k]) * t)
        dirn = Vector((P[k + 1][0] - P[k][0], P[k + 1][1] - P[k][1], 0.0))
        out.append((p, dirn.normalized() if dirn.length > 1e-9 else Vector((1.0, 0.0, 0.0))))
    return out, d[-1]


def _ground_fn(scene):
    def ground_at(p):
        hit = terrain.terrain_under(scene, p)
        return hit[1] if hit else None
    return ground_at


def build_line(obj, depsgraph=None):
    """Rebuilds a power line or path lights object from its curve."""
    p = obj.civil_line
    curve = p.curve
    if curve is None or curve.type != 'CURVE':
        return
    pts = curve_points(curve, depsgraph)
    if len(pts) < 2:
        return
    scene = bpy.context.scene
    ground_at = _ground_fn(scene)

    def on_ground(pt):
        gz = ground_at(pt) if p.follow_terrain else None
        return Vector((pt.x, pt.y, gz if gz is not None else pt.z))

    bm = bmesh.new()
    _, total = resample(pts, [0.0])
    if p.kind == 'POWER':
        n = max(1, int(round(total / max(p.span, 5.0))))
        places, _ = resample(pts, [total * k / n for k in range(n + 1)])
        poles = []
        for k, (pt, dirn) in enumerate(places):
            # Turn each pole to face along the line (bisecting at bends).
            d_prev = places[k - 1][1] if k > 0 else dirn
            d_next = places[k + 1][1] if k < n else dirn
            along = (d_prev + d_next)
            along = along.normalized() if along.length > 1e-6 else dirn
            across = along.cross(UP).normalized()
            g = on_ground(pt)
            h = p.pole_height
            tr = p.style in DISTRIBUTION and p.transformer_every > 0 and k % p.transformer_every == p.transformer_every // 2 \
                and 0 < k < n
            if p.style == 'WOOD':
                att = pole_wood(bm, g, across, along, h, tr)
            elif p.style == 'COMPACT':
                att = pole_compact(bm, g, across, along, h, tr)
            elif p.style == 'H_FRAME':
                att = pole_h_frame(bm, g, across, along, h, ground_at)
            elif p.style == 'MONOPOLE':
                att = pole_monopole(bm, g, across, along, h)
            else:
                att = pole_lattice(bm, g, across, along, h, ground_at)
            poles.append(att)
        r = 0.012 if p.style in DISTRIBUTION else 0.022
        for a, b in zip(poles[:-1], poles[1:]):
            for pa, pb in zip(a, b):
                span = Vector((pb.x - pa.x, pb.y - pa.y, 0.0)).length
                wire(bm, pa, pb, span * p.sag, r, segs=max(8, min(32, int(span / 6.0))))
        obj["cf_poles"] = len(poles)
    else:
        n = max(1, int(total / max(p.light_spacing, 2.0)))
        start = (total - (n - 1) * p.light_spacing) * 0.5 if n > 1 else total * 0.5
        places, _ = resample(pts, [start + k * p.light_spacing for k in range(n)])
        count = 0
        for k, (pt, dirn) in enumerate(places):
            side = dirn.cross(UP).normalized()  # to the right of the path
            sides = {'LEFT': (-1.0,), 'RIGHT': (1.0,), 'BOTH': (-1.0, 1.0), 'CENTER': (0.0,),
                     'ALTERNATE': ((-1.0 if k % 2 == 0 else 1.0),)}[p.light_side]
            for s in sides:
                base = on_ground(pt + side * (s * p.light_offset))
                toward = -side * s if s != 0.0 else side
                add_light(bm, base, toward, dirn, p.light_style, p.light_height, p.light_arm)
                count += 1
        obj["cf_poles"] = count

    me = obj.data
    bm.to_mesh(me)
    bm.free()
    if len(me.materials) < M_POLYMER + 1:
        sys.modules[__package__].ensure_civil_materials(obj)
    me.update()


# ------------------------------------------------------------------------------
# Properties, operators, panel
# ------------------------------------------------------------------------------

def _rebuild(self, context):
    obj = self.id_data
    if obj and obj.type == 'MESH' and self.is_line:
        build_line(obj, context.evaluated_depsgraph_get() if context else None)


def _style_changed(self, context):
    """A new pole style starts from its own usual height, span and sag."""
    span, sag = POWER_DEFAULTS[self.style]
    self["span"] = span
    self["sag"] = sag
    self["pole_height"] = {'WOOD': 12.0, 'COMPACT': 13.0, 'H_FRAME': 18.0, 'MONOPOLE': 26.0, 'LATTICE': 34.0}[self.style]
    _rebuild(self, context)


def _light_style_changed(self, context):
    h, sp = LIGHT_DEFAULTS[self.light_style]
    self["light_height"] = h
    self["light_spacing"] = sp
    _rebuild(self, context)


class CIVIL_PG_Line(bpy.types.PropertyGroup):
    is_line: bpy.props.BoolProperty(default=False)
    kind: bpy.props.EnumProperty(items=[('POWER', "Power Line", ""), ('LIGHTS', "Path Lights", "")], default='POWER')
    curve: bpy.props.PointerProperty(type=bpy.types.Object, name="Path", update=_rebuild,
                                     poll=lambda self, ob: ob.type == 'CURVE')
    follow_terrain: bpy.props.BoolProperty(name="Stand on Terrain", default=True, update=_rebuild,
                                           description="Stand poles on CivilFlow terrain under them")
    style: bpy.props.EnumProperty(name="Pole Style", items=POWER_STYLES, default='WOOD', update=_style_changed)
    span: bpy.props.FloatProperty(name="Pole Spacing", default=45.0, min=10.0, max=600.0, unit='LENGTH',
                                  update=_rebuild, description="Distance between poles")
    pole_height: bpy.props.FloatProperty(name="Pole Height", default=12.0, min=6.0, max=70.0, unit='LENGTH',
                                         update=_rebuild)
    sag: bpy.props.FloatProperty(name="Wire Sag", default=0.022, min=0.0, max=0.1, subtype='FACTOR',
                                 update=_rebuild, description="How far wires dip, as a share of the span")
    transformer_every: bpy.props.IntProperty(
        name="Transformer Every", default=4, min=0, max=20, update=_rebuild,
        description="Put a transformer on every Nth pole (0 = none). Only distribution poles carry them")
    light_style: bpy.props.EnumProperty(name="Light Style", items=LIGHT_STYLES, default='ACORN',
                                        update=_light_style_changed)
    light_spacing: bpy.props.FloatProperty(name="Spacing", default=18.0, min=3.0, max=100.0, unit='LENGTH',
                                           update=_rebuild)
    light_height: bpy.props.FloatProperty(name="Height", default=4.3, min=2.0, max=20.0, unit='LENGTH',
                                          update=_rebuild)
    light_arm: bpy.props.FloatProperty(name="Arm Length", default=2.0, min=0.5, max=8.0, unit='LENGTH',
                                       update=_rebuild)
    light_side: bpy.props.EnumProperty(
        name="Side", default='BOTH', update=_rebuild,
        items=[('BOTH', "Both Sides", ""), ('LEFT', "Left Side", ""), ('RIGHT', "Right Side", ""),
               ('ALTERNATE', "Alternating", ""), ('CENTER', "On the Line", "")])
    light_offset: bpy.props.FloatProperty(name="Offset", default=2.0, min=0.0, max=30.0, unit='LENGTH',
                                          update=_rebuild, description="Distance from the path to each light")


def is_line(ob):
    p = getattr(ob, "civil_line", None)
    return ob is not None and ob.type == 'MESH' and p is not None and p.is_line


def add_line(context, kind, curve=None, length=None):
    """New power line or path lights object along `curve` (or a new straight curve from the cursor)."""
    if curve is None:
        cf = sys.modules[__package__]
        c = context.scene.cursor.location.copy()
        L = length or (400.0 if kind == 'POWER' else 80.0)
        curve = cf.create_guide_curve(context, "Power_Line" if kind == 'POWER' else "Path_Lights",
                                      [tuple(c), tuple(c + Vector((L * 0.5, 0.0, 0.0))), tuple(c + Vector((L, 0.0, 0.0)))])
    name = "Power Line" if kind == 'POWER' else "Path Lights"
    me = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, me)
    for col in curve.users_collection or [context.collection]:
        col.objects.link(obj)
        break
    obj["cf_utility"] = kind
    sys.modules[__package__].ensure_civil_materials(obj)
    p = obj.civil_line
    p["kind"] = 0 if kind == 'POWER' else 1
    p.is_line = True
    p.curve = curve  # builds it
    return obj


class CIVIL_OT_AddLine(bpy.types.Operator):
    bl_idname = "civil.add_line"
    bl_label = "Add Power Line"
    bl_description = "Poles and wires (or a row of lights) along the selected curve, or a new one from the 3D cursor"
    bl_options = {'REGISTER', 'UNDO'}

    kind: bpy.props.EnumProperty(items=[('POWER', "Power Line", "Poles and wires along a curve"),
                                        ('LIGHTS', "Path Lights", "Streetlights along a path, for parks and plazas")])

    @classmethod
    def description(cls, context, props):
        if props.kind == 'LIGHTS':
            return "A row of streetlights along the selected curve (park paths, plazas), or a new one from the 3D cursor"
        return cls.bl_description

    def execute(self, context):
        ob = context.active_object
        curve = ob if ob is not None and ob.type == 'CURVE' and ob.select_get() else None
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        obj = add_line(context, self.kind, curve)
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        what = "poles" if self.kind == 'POWER' else "lights"
        self.report({'INFO'}, "%s: %d %s. Edit its curve to reshape it" % (obj.name, obj.get("cf_poles", 0), what))
        return {'FINISHED'}


class CIVIL_OT_EditLineCurve(bpy.types.Operator):
    bl_idname = "civil.edit_line_curve"
    bl_label = "Edit Path"
    bl_description = "Select the line's curve and edit its points"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ob = context.active_object
        if not is_line(ob) or ob.civil_line.curve is None:
            return {'CANCELLED'}
        c = ob.civil_line.curve
        for o in context.selected_objects:
            o.select_set(False)
        c.hide_set(False)
        c.select_set(True)
        context.view_layer.objects.active = c
        bpy.ops.object.mode_set(mode='EDIT')
        return {'FINISHED'}


class CIVIL_PT_Utilities(bpy.types.Panel):
    bl_label = "Power Lines & Path Lights"
    bl_idname = "CIVIL_PT_Utilities"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'CivilEng'
    bl_order = 3

    def draw(self, context):
        layout = self.layout
        row = layout.row(align=True)
        row.operator("civil.add_line", text="Power Line", icon='LIGHT_SUN').kind = 'POWER'
        row.operator("civil.add_line", text="Path Lights", icon='LIGHT_POINT').kind = 'LIGHTS'
        layout.label(text="Uses the selected curve, or starts one at the cursor", icon='INFO')
        ob = context.active_object
        if not is_line(ob):
            return
        p = ob.civil_line
        box = layout.box()
        box.label(text="%s · %d %s" % (ob.name, ob.get("cf_poles", 0), "poles" if p.kind == 'POWER' else "lights"),
                  icon='LIGHT_SUN' if p.kind == 'POWER' else 'LIGHT_POINT')
        box.prop(p, "curve")
        box.operator("civil.edit_line_curve", icon='CURVE_BEZCURVE')
        box.prop(p, "follow_terrain")
        if p.kind == 'POWER':
            box.prop(p, "style", text="")
            col = box.column(align=True)
            col.prop(p, "span")
            col.prop(p, "pole_height")
            col.prop(p, "sag")
            if p.style in DISTRIBUTION:
                box.prop(p, "transformer_every")
        else:
            box.prop(p, "light_style", text="")
            box.prop(p, "light_side")
            col = box.column(align=True)
            col.prop(p, "light_spacing")
            col.prop(p, "light_height")
            col.prop(p, "light_offset")
            if p.light_style in ('COBRA', 'MODERN'):
                col.prop(p, "light_arm")


_busy = False


@bpy.app.handlers.persistent
def _line_curves_changed(scene, depsgraph):
    """Lines follow their curves when you edit them."""
    global _busy
    if _busy:
        return
    changed = set()
    for u in depsgraph.updates:
        if isinstance(u.id, bpy.types.Curve):
            changed.add(u.id.original)
        elif isinstance(u.id, bpy.types.Object) and u.id.type == 'CURVE' and (u.is_updated_transform or u.is_updated_geometry):
            changed.add(u.id.original)
    if not changed:
        return
    _busy = True
    try:
        for ob in scene.objects:
            if is_line(ob) and ob.civil_line.curve is not None:
                c = ob.civil_line.curve
                if c.original in changed or c.data.original in changed:
                    build_line(ob, depsgraph)
    finally:
        _busy = False


classes = (CIVIL_PG_Line, CIVIL_OT_AddLine, CIVIL_OT_EditLineCurve, CIVIL_PT_Utilities)


def register_extra():
    bpy.types.Object.civil_line = bpy.props.PointerProperty(type=CIVIL_PG_Line)
    if _line_curves_changed not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_line_curves_changed)


def unregister_extra():
    if _line_curves_changed in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_line_curves_changed)
    del bpy.types.Object.civil_line
