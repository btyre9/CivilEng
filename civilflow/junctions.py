# SPDX-License-Identifier: GPL-3.0-or-later
"""
Intersections for CivilFlow roads.

Where two roads cross at about the same level (or one road ends at the side of another), an
intersection object joins them:

- an asphalt pad over the crossing with rounded curb returns at the corners and corner sidewalks,
  or a roundabout (circulating ring, central island, splitter islands);
- the roads' own surface, lines, sidewalks and lights are trimmed away inside it;
- crosswalks and stop lines on each approach;
- the control you pick: traffic signals on mast arms (they cycle as the animation plays),
  all-way or two-way stop signs, a roundabout with yield signs, or no control.

The intersection keeps itself on the roads: when either road is edited it finds the crossing
again and rebuilds, and the roads re-trim themselves.
"""

import json
import math
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector

from . import terrain

UP = Vector((0.0, 0.0, 1.0))
ASPHALT = 0.08            # asphalt top above the road's centreline samples
PAD_LIFT = 0.004          # the pad sits a hair above the road surfaces it overlaps
OVERLAP = 2.0             # the pad reaches this far past the trim, so no seam shows
LEVEL_TOL = 2.5           # roads within this height of each other can meet
M_CONCRETE, M_LINE_Y, M_LINE_W, M_STEEL = 1, 4, 5, 6
M_IRON, M_GRASS, M_RED = 12, 19, 20
M_SIGNAL0 = 21            # six per-intersection lamp materials follow: group 0 R Y G, group 1 R Y G

CONTROLS = [
    ('SIGNALS', "Traffic Signals", "Signal heads on mast arms over each approach; they cycle as the animation plays"),
    ('ALL_STOP', "All-Way Stop", "A stop sign on every approach"),
    ('TWO_STOP', "Two-Way Stop", "Stop signs on one road only; the other road has the right of way"),
    ('ROUNDABOUT', "Roundabout", "Circulating ring around a central island, with yield signs at each entry"),
    ('NONE', "No Control", "Just the joined pavement and crosswalks"),
]


# ------------------------------------------------------------------------------
# Roads as polylines
# ------------------------------------------------------------------------------

class RoadLine:
    """A road's centreline in world space, with its widths, for intersection maths."""

    def __init__(self, road, depsgraph=None):
        cf = sys.modules[__package__]
        p = road.civil_props
        self.road = road
        smp = cf.road_local_samples(road, depsgraph)
        M = road.matrix_world
        self.P = np.array([tuple(M @ s[0]) for s in smp]) if smp else np.zeros((0, 3))
        seg = np.linalg.norm(np.diff(self.P[:, :2], axis=0), axis=1) if len(self.P) > 1 else np.zeros(0)
        self.d = np.concatenate([[0.0], np.cumsum(seg)])
        self.total = float(self.d[-1]) if len(self.d) else 0.0
        lanes = p.lanes_left * p.lane_width + p.lanes_right * p.lane_width
        median = p.median_width if (p.lanes_left > 0 and p.lanes_right > 0) else 0.0
        self.hw = (lanes + median + 2.0 * p.shoulder_width) * 0.5
        self.sw = p.sidewalk_width if p.add_sidewalks else 0.0
        self.lane_w = p.lane_width
        self.shoulder = p.shoulder_width
        self.lanes_left, self.lanes_right = p.lanes_left, p.lanes_right
        self.left_hand = getattr(p, "traffic_left_hand", False)
        self.median = median

    def at(self, s):
        """World point at distance s along the road (clamped to its ends)."""
        s = min(max(s, 0.0), self.total)
        k = int(np.clip(np.searchsorted(self.d, s) - 1, 0, len(self.P) - 2))
        span = self.d[k + 1] - self.d[k]
        t = 0.0 if span < 1e-9 else (s - self.d[k]) / span
        return Vector(self.P[k] + (self.P[k + 1] - self.P[k]) * t)

    def mode_at(self, s):
        rec = terrain._CARVE.get(self.road.name)
        if rec is None or "mode" not in rec:
            return terrain.BRIDGE if self.road.civil_props.is_overpass else terrain.ROAD
        mode = rec["mode"]
        k = int(round(s / max(self.total, 1e-6) * (len(mode) - 1)))
        return int(mode[min(max(k, 0), len(mode) - 1)])

    def nearest(self, xy):
        """(distance along, distance away) of the centreline point nearest xy."""
        A = self.P[:-1, :2]
        B = self.P[1:, :2]
        AB = B - A
        L2 = np.maximum((AB ** 2).sum(axis=1), 1e-12)
        t = np.clip(((np.array(xy) - A) * AB).sum(axis=1) / L2, 0.0, 1.0)
        Q = A + AB * t[:, None]
        dist = np.hypot(Q[:, 0] - xy[0], Q[:, 1] - xy[1])
        k = int(np.argmin(dist))
        return float(self.d[k] + t[k] * np.sqrt(L2[k])), float(dist[k])


def crossings(a, b):
    """[(s along a, s along b, (x, y))] where two centrelines cross."""
    if len(a.P) < 2 or len(b.P) < 2:
        return []
    A0, A1 = a.P[:-1, None, :2], a.P[1:, None, :2]
    B0, B1 = b.P[None, :-1, :2], b.P[None, 1:, :2]
    r = A1 - A0
    s = B1 - B0
    den = r[..., 0] * s[..., 1] - r[..., 1] * s[..., 0]
    qp = B0 - A0
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (qp[..., 0] * s[..., 1] - qp[..., 1] * s[..., 0]) / den
        u = (qp[..., 0] * r[..., 1] - qp[..., 1] * r[..., 0]) / den
    hit = (np.abs(den) > 1e-12) & (t >= 0) & (t < 1) & (u >= 0) & (u < 1)
    out = []
    for i, j in zip(*np.nonzero(hit)):
        sa = a.d[i] + t[i, j] * (a.d[i + 1] - a.d[i])
        sb = b.d[j] + u[i, j] * (b.d[j + 1] - b.d[j])
        xy = A0[i, 0] + r[i, 0] * t[i, j]
        out.append((float(sa), float(sb), (float(xy[0]), float(xy[1]))))
    return out


def find_meetings(a, b):
    """Places two roads meet at about the same level: [(shape, s_a, s_b, end, (x, y))]."""
    found = []
    for sa, sb, xy in crossings(a, b):
        za, zb = a.at(sa).z, b.at(sb).z
        if abs(za - zb) > LEVEL_TOL:
            continue
        if a.mode_at(sa) != terrain.ROAD or b.mode_at(sb) != terrain.ROAD:
            continue
        found.append(('CROSS', sa, sb, None, xy))
    # One road ending at the side of the other (a T), either way round.
    for through, ending, flip in ((a, b, False), (b, a, True)):
        for end_s in (0.0, ending.total):
            e = ending.at(end_s)
            st, dist = through.nearest((e.x, e.y))
            if dist > through.hw + ending.sw + 8.0:
                continue
            if st < 3.0 or st > through.total - 3.0:
                continue  # end to end: that's a road continuing, not a junction
            if abs(through.at(st).z - e.z) > LEVEL_TOL or through.mode_at(st) != terrain.ROAD:
                continue
            c = through.at(st)
            if any(math.hypot(c.x - f[4][0], c.y - f[4][1]) < 15.0 for f in found):
                continue
            if flip:
                found.append(('T_REV', end_s, st, end_s, (c.x, c.y)))
            else:
                found.append(('T', st, end_s, end_s, (c.x, c.y)))
    return found


# ------------------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------------------

def _v2(v):
    return Vector((v.x, v.y))


def _rn(d):
    """Right of an outward direction (2D)."""
    return Vector((d.y, -d.x))


def _ln(d):
    return Vector((-d.y, d.x))


def _ang(v):
    return math.atan2(v.y, v.x)


def _quad(bm, pts, mat):
    vs = [bm.verts.new(p) for p in pts]
    f = bm.faces.new(vs)
    f.material_index = mat
    return f


def _box(bm, center, ax, ay, hx, hy, hz, mat):
    az = ax.cross(ay).normalized()
    v = [bm.verts.new(center + ax * (sx * hx) + ay * (sy * hy) + az * (sz * hz))
         for sz in (-1.0, 1.0) for sx, sy in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))]
    for idx in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
        bm.faces.new([v[i] for i in idx]).material_index = mat


def _disc(bm, center, normal, r, sides, mat, depth=0.0):
    u = normal.cross(UP if abs(normal.z) < 0.9 else Vector((1.0, 0.0, 0.0))).normalized()
    v = normal.cross(u).normalized()
    ring = [bm.verts.new(center + (u * math.cos(2 * math.pi * k / sides) + v * math.sin(2 * math.pi * k / sides)) * r)
            for k in range(sides)]
    bm.faces.new(ring).material_index = mat
    if depth > 0.0:
        back = [bm.verts.new(p.co - normal * depth) for p in ring]
        bm.faces.new(list(reversed(back))).material_index = mat
        for k in range(sides):
            bm.faces.new((ring[k], back[k], back[(k + 1) % sides], ring[(k + 1) % sides])).material_index = mat


def _inside(poly, X, Y):
    """Even-odd test of points against a polygon [(x, y)]."""
    P = np.asarray(poly)
    inside = np.zeros(np.shape(X), dtype=bool)
    x1, y1 = P[:, 0], P[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    for a, b, c, d in zip(x1, y1, x2, y2):
        cond = (b > Y) != (d > Y)
        with np.errstate(divide='ignore', invalid='ignore'):
            xi = a + (Y - b) * (c - a) / (d - b)
        inside ^= cond & (X < xi)
    return inside


# ------------------------------------------------------------------------------
# Building one intersection
# ------------------------------------------------------------------------------

class Arm:
    def __init__(self, line, s0, sign, group, center):
        self.line, self.s0, self.sign, self.group = line, s0, sign, group
        self.hw, self.sw = line.hw, line.sw
        far = self.pos(15.0)
        d = _v2(far) - _v2(center)
        self.d = d.normalized() if d.length > 1e-6 else Vector((1.0, 0.0))
        self.L = 0.0

    def pos(self, r):
        return self.line.at(self.s0 + self.sign * r)

    def z(self, r):
        return self.pos(r).z + ASPHALT + PAD_LIFT

    def frame(self, r):
        """(centre xy, outward unit dir, right unit) of the road's cross-section r metres out."""
        p0, p1 = self.pos(r - 0.5), self.pos(r + 0.5)
        d = _v2(p1) - _v2(p0)
        d = d.normalized() if d.length > 1e-6 else self.d
        return _v2(self.pos(r)), d, _rn(d)

    def incoming(self):
        """Sign (+1 = right of outward dir, -1 = left) of the side where traffic drives in."""
        return 1.0 if self.line.left_hand else -1.0

    def lanes_in(self):
        """Lanes bringing traffic into the junction on this arm."""
        road_left = (self.incoming() < 0) == (self.sign > 0)
        return self.line.lanes_left if road_left else self.line.lanes_right

    def lanes_out(self):
        road_left = (self.incoming() < 0) == (self.sign > 0)
        return self.line.lanes_right if road_left else self.line.lanes_left

    def lane_xs(self, count):
        """Distances out from the centreline of `count` lanes counted in from the road edge."""
        edge = self.hw - self.line.shoulder
        return [edge - self.line.lane_w * (k + 0.5) for k in range(count)]


def _arms(j, a, b, shape, sa, sb, center):
    arms = []
    if shape == 'CROSS':
        for line, s, g in ((a, sa, 0), (b, sb, 1)):
            for sign in (1.0, -1.0):
                room = (line.total - s) if sign > 0 else s
                if room > 6.0:
                    arms.append(Arm(line, s, sign, g, center))
    else:
        # a: the road that goes through; b ends at it (sb is its end).
        for sign in (1.0, -1.0):
            room = (a.total - sa) if sign > 0 else sa
            if room > 6.0:
                arms.append(Arm(a, sa, sign, 0, center))
        e = b.at(sb)
        gap = (_v2(e) - _v2(center)).length
        sign = 1.0 if sb < b.total * 0.5 else -1.0
        arms.append(Arm(b, sb - sign * gap, sign, 1, center))
    arms.sort(key=lambda m: _ang(m.d))
    return arms


def _fillet(C, ai, aj, R):
    """Curb return between arm i's left edge and arm j's right edge: (points, t_i, t_j) or None if straight."""
    phi = (_ang(aj.d) - _ang(ai.d)) % (2 * math.pi)
    if phi > math.radians(150) or phi < math.radians(20):
        return None
    pi_ = C + _ln(ai.d) * ai.hw
    pj = C + _rn(aj.d) * aj.hw
    den = ai.d.x * (-aj.d.y) - ai.d.y * (-aj.d.x)
    if abs(den) < 1e-9:
        return None
    q = pj - pi_
    ta = (q.x * (-aj.d.y) - q.y * (-aj.d.x)) / den
    K = pi_ + ai.d * ta
    R = min(R, 0.6 * max(1.0, (K - C).length * math.tan(phi / 2)))
    t = R / math.tan(phi / 2)
    Ti, Tj = K + ai.d * t, K + aj.d * t
    bis = (ai.d + aj.d).normalized()
    Cc = K + bis * (R / math.sin(phi / 2))
    a0, a1 = _ang(Ti - Cc), _ang(Tj - Cc)
    da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    n = max(3, int(abs(da) / (math.pi / 18)) + 1)
    pts = [Cc + Vector((math.cos(a0 + da * k / n), math.sin(a0 + da * k / n))) * R for k in range(n + 1)]
    return pts, (Ti - C).dot(ai.d), (Tj - C).dot(aj.d)


def _circle_corner(C, ai, aj, Rr, n_per_rad=10):
    """Roundabout: the ring's edge between where arm i's left edge and arm j's right edge meet it."""
    def meet(arm, side):
        h = arm.hw
        r = math.sqrt(max(Rr * Rr - h * h, 1.0))
        return C + arm.d * r + side * h
    qi = meet(ai, _ln(ai.d))
    qj = meet(aj, _rn(aj.d))
    a0, a1 = _ang(qi - C), _ang(qj - C)
    da = (a1 - a0) % (2 * math.pi)
    n = max(2, int(da * n_per_rad))
    return [C + Vector((math.cos(a0 + da * k / n), math.sin(a0 + da * k / n))) * Rr for k in range(n + 1)]


def lamp_materials(j, fps):
    """Red, yellow and green lamps for the two signal groups; they switch as the animation plays."""
    p = j.civil_junction
    G, Y, AR = p.green_time, p.yellow_time, p.all_red_time
    cyc = 2.0 * (G + Y + AR)
    mats = []
    for g in (0, 1):
        off = cyc * 0.5 * g
        t = "fmod(frame / %.4f + %.4f, %.4f)" % (fps, cyc - off, cyc)
        for name, color, dim, expr in (
                ("Red", (1.0, 0.05, 0.02), (0.12, 0.01, 0.01), "12.0 if %s >= %.4f else 0.0" % (t, G + Y)),
                ("Yellow", (1.0, 0.6, 0.02), (0.13, 0.08, 0.0), "12.0 if abs(%s - %.4f) < %.4f else 0.0" % (t, G + Y * 0.5, Y * 0.5)),
                ("Green", (0.1, 1.0, 0.45), (0.0, 0.09, 0.04), "12.0 if %s < %.4f else 0.0" % (t, G))):
            mname = "CF Signal %s %s %d" % (j.name, name, g + 1)
            mat = bpy.data.materials.get(mname) or bpy.data.materials.new(mname)
            mat.diffuse_color = color + (1.0,)
            nt = mat.node_tree
            bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None) if nt else None
            if bsdf is not None:
                bsdf.inputs["Base Color"].default_value = dim + (1.0,)
                bsdf.inputs["Emission Color"].default_value = color + (1.0,)
                inp = bsdf.inputs["Emission Strength"]
                try:
                    inp.driver_remove("default_value")
                except TypeError:
                    pass
                fc = inp.driver_add("default_value")
                fc.driver.type = 'SCRIPTED'
                fc.driver.expression = expr
            mats.append(mat)
    return mats


def _signal_head(bm, center, face, group):
    """Three-lamp signal head facing `face` (horizontal), with a backplate."""
    side = face.cross(UP).normalized()
    _box(bm, center - face * 0.12, side, UP, 0.3, 0.62, 0.02, M_IRON)       # backplate
    _box(bm, center, side, UP, 0.18, 0.5, 0.11, M_IRON)                     # housing
    base = M_SIGNAL0 + 3 * group
    for k, mat in ((1, base + 0), (0, base + 1), (-1, base + 2)):           # red top, green bottom
        c = center + UP * (k * 0.32) + face * 0.115
        _disc(bm, c, face, 0.12, 12, mat)
        _box(bm, c + UP * 0.12 + face * 0.08, side, face, 0.14, 0.08, 0.01, M_IRON)  # visor


def _stop_sign(bm, base, face, all_way):
    side = face.cross(UP).normalized()
    _box(bm, base + UP * 1.3, side, face, 0.035, 0.035, 1.3, M_STEEL)
    c = base + UP * 2.35 + face * 0.04
    _disc(bm, c - face * 0.005, face, 0.42, 8, M_LINE_W, depth=0.02)
    _disc(bm, c + face * 0.002, face, 0.38, 8, M_RED)
    cf = sys.modules[__package__]
    cf.add_text(bm, "STOP", c + face * 0.006, side, UP, face, 0.55, 0.22, M_LINE_W)
    if all_way:
        p = base + UP * 1.75 + face * 0.04
        _box(bm, p, side, UP, 0.3, 0.1, 0.01, M_LINE_W)
        cf.add_text(bm, "ALL WAY", p + face * 0.012, side, UP, face, 0.5, 0.12, M_RED)


def _yield_sign(bm, base, face):
    side = face.cross(UP).normalized()
    _box(bm, base + UP * 1.2, side, face, 0.035, 0.035, 1.2, M_STEEL)
    c = base + UP * 2.2 + face * 0.04
    for r, mat, off in ((0.55, M_RED, 0.0), (0.33, M_LINE_W, 0.004)):
        pts = [c + face * off + (side * math.sin(a) + UP * math.cos(a)) * r
               for a in (math.pi, math.pi + 2 * math.pi / 3, math.pi + 4 * math.pi / 3)]
        f = bm.faces.new([bm.verts.new(p) for p in pts])
        f.material_index = mat
        if f.normal.dot(face) < 0:
            f.normal_flip()


def _mast_signal(bm, pole, reach_dir, reach, face, heads, group):
    """Pole with a mast arm reaching `reach` metres along reach_dir; heads at those distances out."""
    top = 7.2
    _box(bm, pole + UP * 0.3, reach_dir, face, 0.45, 0.45, 0.3, M_CONCRETE)
    cf = sys.modules[__package__]
    cf.add_cylinder(bm, pole, reach_dir, face, UP, 0.16, top, 12, M_STEEL)
    arm_z = pole + UP * (top - 0.6)
    cf.add_cylinder_between(bm, arm_z, arm_z + reach_dir * reach, 0.09, 8, M_STEEL)
    cf.add_cylinder_between(bm, pole + UP * (top - 0.05), arm_z + reach_dir * (reach * 0.45), 0.03, 6, M_STEEL)
    for r in heads:
        _signal_head(bm, arm_z + reach_dir * r - UP * 0.75, face, group)
    # A pedestrian-height head on the pole too.
    _signal_head(bm, pole + UP * 3.6 + face * 0.25, face, group)


def _sidewalk(bm, path, outward_ref, width, zs, C):
    """Curb and sidewalk along a curb line `path` (2D points), on the side away from C."""
    n = len(path)
    if n < 2 or width <= 0.05:
        return
    inner, outer = [], []
    for k in range(n):
        a = path[max(0, k - 1)]
        b = path[min(n - 1, k + 1)]
        t = (b - a)
        t = t.normalized() if t.length > 1e-9 else Vector((1.0, 0.0))
        nrm = _ln(t)
        if nrm.dot(path[k] - C) < 0:
            nrm = -nrm
        z = zs[k]
        inner.append((Vector((path[k].x, path[k].y, z - 0.02)), Vector((path[k].x, path[k].y, z + 0.1))))
        o = path[k] + nrm * width
        outer.append((Vector((o.x, o.y, z - 0.3)), Vector((o.x, o.y, z + 0.1))))
    for k in range(n - 1):
        (i0b, i0t), (i1b, i1t) = inner[k], inner[k + 1]
        (o0b, o0t), (o1b, o1t) = outer[k], outer[k + 1]
        for quad in ((i0t, i1t, o1t, o0t), (i0b, i1b, i1t, i0t), (o0t, o1t, o1b, o0b)):
            f = _quad(bm, quad, M_CONCRETE)
            if quad[0] is i0t and f.normal.z < 0:
                f.normal_flip()


def build(j, depsgraph=None):
    """Rebuilds an intersection's mesh and its trim outline. Returns True if the trim changed."""
    p = j.civil_junction
    old_cut = list(j.get("cf_cut", []))
    if p.road_a is None or p.road_b is None:
        return False
    a, b = RoadLine(p.road_a, depsgraph), RoadLine(p.road_b, depsgraph)
    if len(a.P) < 2 or len(b.P) < 2:
        return False
    # Find the meeting again near where it was.
    at = Vector(tuple(j.get("cf_at", (0.0, 0.0))))
    best = None
    for shape, sa, sb, end, xy in find_meetings(a, b):
        if (shape == 'CROSS') != (p.shape == 'CROSS'):
            continue
        dist = math.hypot(xy[0] - at.x, xy[1] - at.y)
        if dist < 60.0 and (best is None or dist < best[0]):
            best = (dist, shape, sa, sb, xy)
    bm = bmesh.new()
    if best is None:
        j["cf_layout"] = ""
        j["cf_cut"] = []
        j["cf_lost"] = 1
        _finish(j, bm)
        return bool(old_cut)
    if "cf_lost" in j:
        del j["cf_lost"]
    _, shape, sa, sb, xy = best
    if shape == 'T_REV':  # the stored roads are the other way round for this T
        a, b, sa, sb = b, a, sb, sa
        shape = 'T'
    C = Vector(xy)
    j["cf_at"] = (C.x, C.y)
    arms = _arms(j, a, b, shape, sa, sb, C)
    if len(arms) < 3:
        j["cf_layout"] = ""
        j["cf_cut"] = []
        _finish(j, bm)
        return bool(old_cut)

    n = len(arms)
    rb = p.control == 'ROUNDABOUT'
    max_hw = max(m.hw for m in arms)
    Rr = p.roundabout_radius if p.roundabout_radius > 0 else min(32.0, max(13.0, max_hw + 9.0))
    cw = 5.0 if p.crosswalks else 1.5
    corners = []
    for i in range(n):
        ai, aj = arms[i], arms[(i + 1) % n]
        if rb:
            pts = _circle_corner(C, ai, aj, Rr)
            corners.append(pts)
            ti = math.sqrt(max(Rr ** 2 - ai.hw ** 2, 1.0))
            tj = math.sqrt(max(Rr ** 2 - aj.hw ** 2, 1.0))
        else:
            res = _fillet(C, ai, aj, p.corner_radius)
            if res is None:
                corners.append([])
                ti = tj = max_hw + 1.0
            else:
                pts, ti, tj = res
                corners.append(pts)
        ai.L = max(ai.L, ti + cw + (6.0 if rb else 0.0))
        aj.L = max(aj.L, tj + cw + (6.0 if rb else 0.0))
    for m in arms:
        m.L = max(m.L, max_hw + 2.0)
    # What the traffic simulation needs to know about this intersection.
    lane = max(m.line.lane_w for m in arms)
    Ri = max(4.0, Rr - 1.4 * lane - 1.5)
    j["cf_layout"] = json.dumps({
        "C": [C.x, C.y], "rb": rb, "Rr": Rr, "Rc": (Ri + 1.6 + Rr) * 0.5,
        "arms": [{"road": m.line.road.name, "sign": m.sign, "s0": m.s0, "L": m.L, "group": m.group,
                  "d": [m.d.x, m.d.y]} for m in arms]})

    # Pad outline: each arm's mouth (a little past the trim), then the corner to the next arm.
    ring, ring_z, mouth_edges = [], [], []
    for i, m in enumerate(arms):
        c, d, r = m.frame(m.L + OVERLAP)
        z = m.z(m.L + OVERLAP)
        ring += [c + r * m.hw, c - r * m.hw]
        ring_z += [z, z]
        mouth_edges.append(len(ring) - 2)
        pts = corners[i]
        mj = arms[(i + 1) % n]
        zi, zj = m.z(m.L), mj.z(mj.L)
        for k, q in enumerate(pts):
            ring.append(q)
            ring_z.append(zi + (zj - zi) * (k / max(1, len(pts) - 1)))
    zc = float(np.mean([m.z(m.L) for m in arms]))
    vc = bm.verts.new((C.x, C.y, zc))
    vr = [bm.verts.new((q.x, q.y, z)) for q, z in zip(ring, ring_z)]
    for k in range(len(vr)):
        f = bm.faces.new((vc, vr[k], vr[(k + 1) % len(vr)]))
        f.material_index = 0
        f.smooth = True
        if f.normal.z < 0:
            f.normal_flip()
    ring_ang = np.array([_ang(q - C) for q in ring])

    def pad_z(q):
        """Height of the pad's surface at q (2D), from the fan triangle it falls in."""
        a = _ang(q - C)
        k = int(np.argmin((ring_ang - a) % (2 * math.pi)))  # first ring point at or after a, going round
        k0 = (k - 1) % len(ring)
        p0, p1 = ring[k0], ring[k]
        z0, z1 = ring_z[k0], ring_z[k]
        # Barycentric weights in triangle (C, p0, p1).
        v0, v1, v2 = p0 - C, p1 - C, q - C
        den = v0.x * v1.y - v0.y * v1.x
        if abs(den) < 1e-9:
            return zc
        w0 = (v2.x * v1.y - v2.y * v1.x) / den
        w1 = (v0.x * v2.y - v0.y * v2.x) / den
        return zc + w0 * (z0 - zc) + w1 * (z1 - zc)
    # A skirt down the open sides so no gap shows against the ground.
    for k in range(len(vr)):
        if k in mouth_edges:
            continue
        a0, a1 = vr[k], vr[(k + 1) % len(vr)]
        _quad(bm, (a0.co, a1.co, a1.co - UP * 1.2, a0.co - UP * 1.2), M_CONCRETE)

    # Corner sidewalks, joining each road's sidewalks around the curb return.
    for i, m in enumerate(arms):
        mj = arms[(i + 1) % n]
        sw = min(m.sw, mj.sw)
        if sw <= 0.05:
            continue
        # The ends reach a little past the trim and sit a hair higher, so they lap over the
        # road's own sidewalks without a seam.
        ci, di, ri = m.frame(m.L + 1.2)
        cj, dj, rj = mj.frame(mj.L + 1.2)
        path = [ci - ri * m.hw] + corners[i] + [cj + rj * mj.hw]
        zi, zj = m.z(m.L + 1.2) + 0.003, mj.z(mj.L + 1.2) + 0.003
        zs = [zi + (zj - zi) * k / (len(path) - 1) for k in range(len(path))]
        _sidewalk(bm, path, None, sw, zs, C)

    # Markings: crosswalks and stop / yield lines on each approach.
    for m in arms:
        inc = m.incoming()
        for_lanes = m.hw - m.line.shoulder
        if p.crosswalks:
            r0, r1 = m.L - 4.6, m.L - 1.6
            x = -for_lanes + 0.3
            while x + 0.5 <= for_lanes - 0.3:
                if not (rb and abs(x + 0.25) < 1.6):
                    c0, d0, rr0 = m.frame(r0)
                    c1, d1, rr1 = m.frame(r1)
                    q = [c0 + rr0 * x, c0 + rr0 * (x + 0.5), c1 + rr1 * (x + 0.5), c1 + rr1 * x]
                    f = _quad(bm, [Vector((v.x, v.y, pad_z(v) + 0.02)) for v in q], M_LINE_W)
                    if f.normal.z < 0:
                        f.normal_flip()
                x += 1.1
        stops = _stops_here(p, m)
        if stops or p.control == 'SIGNALS' or rb:
            r_line = (math.sqrt(max(Rr ** 2 - m.hw ** 2, 1.0)) + 0.8) if rb else m.L - 0.9
            c0, d0, rr = m.frame(r_line)
            x0, x1 = max(0.12, for_lanes - m.lanes_in() * m.line.lane_w), for_lanes
            dash = rb
            x = x0
            while x < x1 - 0.05:
                xe = min(x1, x + (0.6 if dash else x1))
                a_ = c0 + rr * (inc * x)
                b_ = c0 + rr * (inc * xe)
                q = [a_ - d0 * 0.25, b_ - d0 * 0.25, b_ + d0 * 0.25, a_ + d0 * 0.25]
                f = _quad(bm, [Vector((v.x, v.y, pad_z(v) + 0.02)) for v in q], M_LINE_W)
                if f.normal.z < 0:
                    f.normal_flip()
                x = xe + (0.6 if dash else 0.0)

    # Control devices.
    if p.control in ('ALL_STOP', 'TWO_STOP'):
        for m in arms:
            if not _stops_here(p, m):
                continue
            c, d, r = m.frame(m.L - 0.5)
            side = r * m.incoming()
            base = c + side * (m.hw + max(0.6, min(m.sw, 2.0) * 0.6))
            _stop_sign(bm, Vector((base.x, base.y, m.z(m.L) + 0.1)), Vector((d.x, d.y, 0.0)), p.control == 'ALL_STOP')
    elif rb:
        _roundabout(bm, C, zc, Rr, arms, pad_z)
        for m in arms:
            r_y = math.sqrt(max(Rr ** 2 - m.hw ** 2, 1.0)) + 2.5
            c, d, r = m.frame(r_y)
            base = c + r * m.incoming() * (m.hw + 0.9)
            _yield_sign(bm, Vector((base.x, base.y, m.z(r_y) + 0.1)), Vector((d.x, d.y, 0.0)))
    elif p.control == 'SIGNALS':
        for m in arms:
            _signals_for(bm, m, arms)

    j["cf_cut"] = _cut_outline(C, arms, corners, rb, Rr)
    j["cf_z"] = zc
    j["cf_roads"] = [a.road.name, b.road.name]
    _finish(j, bm)
    # The ground under the intersection is shaped to it (corners beyond the road strips too).
    t_obj = a.road.civil_props.terrain_object if a.road.civil_props.follow_terrain else None
    pad = [tuple(q) for q in ring]
    if t_obj is not None and terrain.is_terrain(t_obj):
        terrain.record_area(j.name, t_obj, pad, [z - ASPHALT - PAD_LIFT for z in ring_z])
    else:
        terrain.forget_area(j.name)
    return [round(v, 2) for v in j["cf_cut"]] != [round(v, 2) for v in old_cut]


def _stops_here(p, m):
    if p.control == 'ALL_STOP':
        return True
    if p.control == 'TWO_STOP':
        return m.group == (1 if p.stop_on == 'B' else 0)
    return False


def _signals_for(bm, m, arms):
    """Mast-arm signal for traffic coming in on arm m: across the junction if there's a far side."""
    face = Vector((m.d.x, m.d.y, 0.0))  # heads look back down the approach
    lanes = max(1, m.lanes_in())
    far = min(arms, key=lambda k: k.d.dot(m.d))
    if far is not m and far.d.dot(m.d) < -0.75:
        # Far side: pole on the right of the departing traffic, arm over the lanes it drives into.
        c, d, r = far.frame(far.L - 1.5)
        out_side = r * (-far.incoming())
        xs = far.lane_xs(max(1, min(lanes, far.lanes_out())))
        z = far.z(far.L)
        w = far.hw
    else:
        c, d, r = m.frame(m.L - 1.2)
        out_side = r * m.incoming()
        xs = m.lane_xs(lanes)
        z = m.z(m.L)
        w = m.hw
    pole = c + out_side * (w + 1.2)
    heads = [(w + 1.2) - x for x in xs]
    reach = max(heads) + 0.6
    _mast_signal(bm, Vector((pole.x, pole.y, z)), Vector((-out_side.x, -out_side.y, 0.0)), reach, face, heads, m.group)


def _roundabout(bm, C, zc, Rr, arms, pad_z):
    """Central island (grass, curbed), truck apron and splitter islands, all following the pad."""
    lane = max(m.line.lane_w for m in arms)
    Ri = max(4.0, Rr - 1.4 * lane - 1.5)
    apron = 1.6
    sides = 48

    def ring_pts(r):
        return [Vector((C.x + r * math.cos(2 * math.pi * k / sides), C.y + r * math.sin(2 * math.pi * k / sides)))
                for k in range(sides)]

    def up(q, dz):
        return Vector((q.x, q.y, pad_z(q) + dz))

    o_apron, o_isle = ring_pts(Ri + apron), ring_pts(Ri)
    for k in range(sides):
        k2 = (k + 1) % sides
        a0, a1, b0, b1 = o_apron[k], o_apron[k2], o_isle[k], o_isle[k2]
        _quad(bm, [up(a0, 0.07), up(a1, 0.07), up(b1, 0.07), up(b0, 0.07)], M_CONCRETE)        # apron
        _quad(bm, [up(a0, -0.03), up(a1, -0.03), up(a1, 0.07), up(a0, 0.07)], M_CONCRETE)
        _quad(bm, [up(b0, 0.07), up(b1, 0.07), up(b1, 0.27), up(b0, 0.27)], M_CONCRETE)        # island curb
    # Grass on top: a fan, so it stays on a sloping pad.
    vc = bm.verts.new(up(C, 0.27))
    rim = [bm.verts.new(up(q, 0.27)) for q in o_isle]
    for k in range(sides):
        f = bm.faces.new((vc, rim[k], rim[(k + 1) % sides]))
        f.material_index = M_GRASS
        f.smooth = True
        if f.normal.z < 0:
            f.normal_flip()
    # Splitter islands: a raised wedge on each approach's centreline.
    for m in arms:
        r0 = math.sqrt(max(Rr ** 2 - m.hw ** 2, 1.0)) + 1.0
        r1 = m.L - 5.2
        if r1 - r0 < 2.0:
            continue
        pts_top, pts_bot = [], []
        for r, w in ((r0, 1.4), (r1, 0.25)):
            c, d, rr = m.frame(r)
            for s in (-1.0, 1.0):
                q = c + rr * (s * w)
                pts_top.append(up(q, 0.2))
                pts_bot.append(up(q, -0.02))
        order = (0, 1, 3, 2)
        f = bm.faces.new([bm.verts.new(pts_top[i]) for i in order])
        f.material_index = M_CONCRETE
        if f.normal.z < 0:
            f.normal_flip()
        for i in range(4):
            a_, b_ = order[i], order[(i + 1) % 4]
            _quad(bm, (pts_bot[a_], pts_bot[b_], pts_top[b_], pts_top[a_]), M_CONCRETE)


def _cut_outline(C, arms, corners, rb, Rr):
    """Outline the roads are trimmed to: the pad, its corner sidewalks and a little margin."""
    pts = []
    n = len(arms)
    for i, m in enumerate(arms):
        c, d, r = m.frame(m.L)
        w = m.hw + m.sw + 0.6
        pts += [c + r * w, c - r * w]
        mj = arms[(i + 1) % n]
        if rb:
            ro = Rr + max(m.sw, mj.sw) + 0.6
            wi, wj = m.hw + m.sw + 0.6, mj.hw + mj.sw + 0.6
            qi = C + m.d * math.sqrt(max(ro * ro - wi * wi, 1.0)) + _ln(m.d) * wi
            qj = C + mj.d * math.sqrt(max(ro * ro - wj * wj, 1.0)) + _rn(mj.d) * wj
            a0, a1 = _ang(qi - C), _ang(qj - C)
            da = (a1 - a0) % (2 * math.pi)
            k = max(2, int(da * 8))
            pts += [C + Vector((math.cos(a0 + da * t / k), math.sin(a0 + da * t / k))) * ro for t in range(k + 1)]
        elif corners[i]:
            wi, wj = m.hw + m.sw + 0.6, mj.hw + mj.sw + 0.6
            pi_ = C + _ln(m.d) * wi
            pj = C + _rn(mj.d) * wj
            den = m.d.x * (-mj.d.y) - m.d.y * (-mj.d.x)
            if abs(den) > 1e-9:
                q = pj - pi_
                ta = (q.x * (-mj.d.y) - q.y * (-mj.d.x)) / den
                pts.append(pi_ + m.d * ta)
    return [v for q in pts for v in (q.x, q.y)]


def _finish(j, bm):
    """Writes the mesh. Materials go on first: clearing them afterwards would reset every face's slot."""
    me = j.data
    cf = sys.modules[__package__]
    cf.ensure_civil_materials(j)
    scene = bpy.context.scene
    fps = scene.render.fps / max(scene.render.fps_base, 1e-6)
    for m in lamp_materials(j, fps):
        me.materials.append(m)
    bm.to_mesh(me)
    bm.free()
    me.update()


# ------------------------------------------------------------------------------
# Trimming roads, keeping things in step
# ------------------------------------------------------------------------------

def junctions_on(road):
    out = []
    for o in bpy.data.objects:
        p = getattr(o, "civil_junction", None)
        if p is not None and p.is_junction and o.users_scene and road in (p.road_a, p.road_b):
            out.append(o)
    return out


def trim_road(bm, road):
    """
    Removes the road's faces inside its intersections (called while the road is being built).
    Small separate pieces (lamps, signs, posts) go or stay whole, by where their middle is; the long
    road strips are cut face by face.
    """
    js = [j for j in junctions_on(road) if len(j.get("cf_cut", [])) >= 6]
    if not js:
        return
    M = road.matrix_world
    faces = list(bm.faces)
    if not faces:
        return
    cen = np.array([tuple(M @ f.calc_center_median()) for f in faces])
    kill = np.zeros(len(faces), dtype=bool)
    near = np.zeros(len(faces), dtype=bool)
    polys = []
    for j in js:
        poly = np.array(j["cf_cut"]).reshape(-1, 2)
        zc = float(j.get("cf_z", 0.0))
        polys.append((poly, zc))
        dz = cen[:, 2] - zc
        kill |= _inside(poly, cen[:, 0], cen[:, 1]) & (dz > -4.5) & (dz < 15.0)
        lo, hi = poly.min(axis=0) - 15.0, poly.max(axis=0) + 15.0
        near |= (cen[:, 0] > lo[0]) & (cen[:, 0] < hi[0]) & (cen[:, 1] > lo[1]) & (cen[:, 1] < hi[1])
    # Pieces near the junction: walk connected faces; small pieces are judged as a whole.
    index = {f: k for k, f in enumerate(faces)}
    seen = np.zeros(len(faces), dtype=bool)
    for start in np.nonzero(near)[0]:
        if seen[start]:
            continue
        stack, piece, big = [faces[start]], [], False
        seen[start] = True
        while stack:
            f = stack.pop()
            piece.append(index[f])
            if len(piece) > 600:
                big = True
            for v in f.verts:
                for g in v.link_faces:
                    gi = index[g]
                    if not seen[gi]:
                        seen[gi] = True
                        stack.append(g)
        if big:
            continue
        pc = cen[piece]
        if np.ptp(pc[:, 0]) > 15.0 or np.ptp(pc[:, 1]) > 15.0:
            continue  # a long strip: face by face
        mid = pc.mean(axis=0)
        inside = any(_inside(poly, np.array([mid[0]]), np.array([mid[1]]))[0] and -4.5 < mid[2] - zc < 15.0
                     for poly, zc in polys)
        kill[piece] = inside
    dead = [f for f, k in zip(faces, kill) if k]
    if dead:
        bmesh.ops.delete(bm, geom=dead, context='FACES')


_busy = False


def road_rebuilt(road, depsgraph=None):
    """After a road rebuilds: its intersections follow, and the roads re-trim if they moved."""
    global _busy
    if _busy:
        return
    js = junctions_on(road)
    if not js:
        return
    cf = sys.modules[__package__]
    _busy = True
    try:
        for j in js:
            if build(j, depsgraph):
                for r in (j.civil_junction.road_a, j.civil_junction.road_b):
                    if r is not None:
                        cf.rebuild_civil_mesh(r, depsgraph)
                t = j.civil_junction.road_a.civil_props.terrain_object if j.civil_junction.road_a else None
                if t is not None:
                    terrain.carve(t)
    finally:
        _busy = False


def refresh(j, context=None):
    """Rebuild an intersection and both its roads (after a setting changed)."""
    global _busy
    cf = sys.modules[__package__]
    dg = context.evaluated_depsgraph_get() if context else None
    _busy = True
    try:
        build(j, dg)
        for r in (j.civil_junction.road_a, j.civil_junction.road_b):
            if r is not None:
                cf.rebuild_civil_mesh(r, dg)
        t = j.civil_junction.road_a.civil_props.terrain_object if j.civil_junction.road_a else None
        if t is not None:
            terrain.carve(t)
    finally:
        _busy = False


def find_all(context):
    """Creates intersections wherever roads meet at about the same level. Returns the new ones."""
    roads = [o for o in context.scene.objects
             if o.type == 'MESH' and o.civil_props.is_civil_object and o.civil_props.target_curve is not None]
    lines = {r.name: RoadLine(r) for r in roads}
    existing = [o for o in context.scene.objects if getattr(o, "civil_junction", None) and o.civil_junction.is_junction]
    made = []
    for i in range(len(roads)):
        for k in range(i + 1, len(roads)):
            a, b = lines[roads[i].name], lines[roads[k].name]
            for shape, sa, sb, end, xy in find_meetings(a, b):
                if any(math.hypot(xy[0] - o["cf_at"][0], xy[1] - o["cf_at"][1]) < 12.0 for o in existing + made):
                    continue
                ra, rb_ = (a.road, b.road) if shape != 'T_REV' else (b.road, a.road)
                made.append(new_junction(context, ra, rb_, 'CROSS' if shape == 'CROSS' else 'T', xy))
    for j in made:
        refresh(j, context)
    return made


def new_junction(context, road_a, road_b, shape, xy):
    me = bpy.data.meshes.new("Intersection")
    j = bpy.data.objects.new("Intersection", me)
    (road_a.users_collection[0] if road_a.users_collection else context.collection).objects.link(j)
    j["cf_at"] = (float(xy[0]), float(xy[1]))
    j["cf_utility"] = "JUNCTION"
    p = j.civil_junction
    p.road_a = road_a
    p.road_b = road_b
    p.shape = 'CROSS' if shape == 'CROSS' else 'T'
    p["control"] = 0 if shape == 'CROSS' else 2   # signals at crossings, two-way stop at a T (no rebuild yet)
    p.is_junction = True
    return j


# ------------------------------------------------------------------------------
# Properties, operators, panel
# ------------------------------------------------------------------------------

def _changed(self, context):
    j = self.id_data
    if self.is_junction and j.type == 'MESH':
        refresh(j, context)


class CIVIL_PG_Junction(bpy.types.PropertyGroup):
    is_junction: bpy.props.BoolProperty(default=False)
    shape: bpy.props.EnumProperty(items=[('CROSS', "Crossing", ""), ('T', "T-Junction", "")], default='CROSS')
    road_a: bpy.props.PointerProperty(type=bpy.types.Object, name="Road A")
    road_b: bpy.props.PointerProperty(type=bpy.types.Object, name="Road B")
    control: bpy.props.EnumProperty(name="Type", items=CONTROLS, default='SIGNALS', update=_changed)
    stop_on: bpy.props.EnumProperty(
        name="Stop On", default='B', update=_changed,
        items=[('A', "Road A", "Traffic on road A stops; road B has the right of way"),
               ('B', "Road B", "Traffic on road B stops; road A has the right of way")])
    crosswalks: bpy.props.BoolProperty(name="Crosswalks", default=True, update=_changed)
    corner_radius: bpy.props.FloatProperty(name="Corner Radius", default=7.0, min=1.0, max=40.0, unit='LENGTH',
                                           update=_changed, description="Radius of the curb at each corner")
    roundabout_radius: bpy.props.FloatProperty(name="Roundabout Radius", default=0.0, min=0.0, max=60.0,
                                               unit='LENGTH', update=_changed,
                                               description="Outer radius of the ring (0 = sized to the roads)")
    green_time: bpy.props.FloatProperty(name="Green", default=25.0, min=3.0, max=180.0, update=_changed,
                                        description="Seconds of green for each road in turn")
    yellow_time: bpy.props.FloatProperty(name="Yellow", default=3.5, min=1.0, max=10.0, update=_changed)
    all_red_time: bpy.props.FloatProperty(name="All Red", default=1.5, min=0.0, max=10.0, update=_changed,
                                          description="Seconds when every light is red between phases")


def is_junction(ob):
    p = getattr(ob, "civil_junction", None)
    return ob is not None and p is not None and p.is_junction


class CIVIL_OT_FindJunctions(bpy.types.Operator):
    bl_idname = "civil.find_junctions"
    bl_label = "Find Intersections"
    bl_description = ("Join roads wherever they cross at about the same level, or where one ends at the side "
                      "of another")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        made = find_all(context)
        if not made:
            self.report({'INFO'}, "No new intersections: roads must cross (or meet) within %.1f m of the same height"
                        % LEVEL_TOL)
            return {'FINISHED'}
        for o in context.selected_objects:
            o.select_set(False)
        for j in made:
            j.select_set(True)
        context.view_layer.objects.active = made[0]
        lost = sum(1 for j in made if j.get("cf_lost"))
        self.report({'INFO'}, "%d intersection%s added%s" % (len(made), "" if len(made) == 1 else "s",
                                                            " (%d could not be built)" % lost if lost else ""))
        return {'FINISHED'}


class CIVIL_OT_RemoveJunction(bpy.types.Operator):
    bl_idname = "civil.remove_junction"
    bl_label = "Remove Intersection"
    bl_description = "Delete this intersection; the roads go back to crossing without it"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        j = context.active_object
        if not is_junction(j):
            return {'CANCELLED'}
        roads = [j.civil_junction.road_a, j.civil_junction.road_b]
        terrain.forget_area(j.name)
        me = j.data
        bpy.data.objects.remove(j)
        if me.users == 0:
            bpy.data.meshes.remove(me)
        cf = sys.modules[__package__]
        for r in roads:
            if r is not None:
                cf.rebuild_civil_mesh(r)
        return {'FINISHED'}


class CIVIL_PT_Junctions(bpy.types.Panel):
    bl_label = "Intersections"
    bl_idname = "CIVIL_PT_Junctions"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'CivilEng'
    bl_order = 2

    def draw(self, context):
        layout = self.layout
        layout.operator("civil.find_junctions", icon='VIEWZOOM')
        j = context.active_object
        if not is_junction(j):
            layout.label(text="Select an intersection to change its type", icon='INFO')
            return
        p = j.civil_junction
        box = layout.box()
        a = p.road_a.name if p.road_a else "?"
        b = p.road_b.name if p.road_b else "?"
        box.label(text="%s: %s × %s" % ("Crossing" if p.shape == 'CROSS' else "T-Junction", a, b), icon='CON_TRACKTO')
        if j.get("cf_lost"):
            box.label(text="These roads no longer meet here", icon='ERROR')
        box.prop(p, "control", text="")
        if p.control == 'TWO_STOP':
            box.prop(p, "stop_on")
        if p.control == 'SIGNALS':
            col = box.column(align=True)
            col.label(text="Signal timing (seconds):")
            row = col.row(align=True)
            row.prop(p, "green_time")
            row.prop(p, "yellow_time")
            row.prop(p, "all_red_time")
        if p.control == 'ROUNDABOUT':
            box.prop(p, "roundabout_radius")
        else:
            box.prop(p, "corner_radius")
        box.prop(p, "crosswalks")
        box.operator("civil.remove_junction", icon='X')


classes = (CIVIL_PG_Junction, CIVIL_OT_FindJunctions, CIVIL_OT_RemoveJunction, CIVIL_PT_Junctions)


def register_extra():
    bpy.types.Object.civil_junction = bpy.props.PointerProperty(type=CIVIL_PG_Junction)


def unregister_extra():
    del bpy.types.Object.civil_junction
