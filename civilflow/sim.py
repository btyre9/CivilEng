# SPDX-License-Identifier: GPL-3.0-or-later
"""
Simulated traffic for road networks with intersections.

Roads joined by intersections get traffic that is worked out ahead of time and stored as keyframes
on each vehicle, so it plays back like any animation (no scripts needed):

- Lanes are cut where they enter and leave each intersection; inside it, every allowed movement
  (straight on, right, left, or round a roundabout) is its own short path.
- Each vehicle chooses where to go at each intersection (mostly straight on), keeps a safe gap
  to the vehicle ahead (Intelligent Driver Model), slows for turns, and obeys the intersection:
  signals (the same timing as the lamps you see), stop signs (full stop, then first come, first
  served), two-way stops (the side road waits for a gap), left turns that yield to oncoming
  traffic, and roundabout entries that yield to traffic already going round.
- Vehicles leaving the network at a road end come back in at another road end when there's room.

The traffic runs for a while before the first frame, so the scene starts with traffic already
flowing and queues already formed.
"""

import json
import math
import random

import bpy
import numpy as np
from mathutils import Vector

from . import junctions, traffic

A_MAX = 1.5        # comfortable acceleration (m/s²)
B_COMF = 2.2       # comfortable braking (m/s²)
S_MIN = 2.2        # gap kept when stopped (m)
HEADWAY = 1.3      # seconds of gap kept when moving
LOOKAHEAD = 140.0  # how far ahead a driver looks for the vehicle in front (m)
GAP_TIME = 6.5     # seconds of gap a yielding driver waits for (it takes ~6 s to turn across from a stop)
STOP_HOLD = 1.2    # seconds a vehicle stays stopped at a stop sign
WARMUP = 25.0      # seconds of traffic before the first frame
HIDDEN_Z = -10000.0
TURN_WEIGHTS = {'THROUGH': 0.6, 'KERB': 0.22, 'FAR': 0.18}


# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------

class Seg:
    """A stretch of lane, or a movement through an intersection."""

    def __init__(self, sid, kind, pts, tilts, v0):
        self.id, self.kind = sid, kind
        self.P = np.array([tuple(p) for p in pts], dtype=np.float64)
        self.tilt = np.array(tilts, dtype=np.float64)
        d = np.linalg.norm(np.diff(self.P, axis=0), axis=1) if len(self.P) > 1 else np.zeros(0)
        self.cum = np.concatenate([[0.0], np.cumsum(d)])
        self.length = float(self.cum[-1])
        self.v0 = v0
        self.succ = []          # following segment ids
        self.pred = []
        self.end_j = None       # (junction index, arm index): this lane stretch ends at that stop line
        self.start_j = None
        self.j = None           # movement: junction index
        self.arm_in = self.arm_out = None
        self.entry = None       # movement: the lane stretch it comes from
        self.turn = None        # movement: THROUGH / KERB / FAR
        self.ring = None        # roundabout movement: ring angle at each point (nan off the ring)
        self.rank = 0

    def at(self, x):
        """(point, unit direction, tilt) at distance x along."""
        x = min(max(x, 0.0), self.length)
        k = int(np.clip(np.searchsorted(self.cum, x) - 1, 0, len(self.P) - 2))
        span = self.cum[k + 1] - self.cum[k]
        t = 0.0 if span < 1e-9 else (x - self.cum[k]) / span
        p = self.P[k] + (self.P[k + 1] - self.P[k]) * t
        d = self.P[k + 1] - self.P[k]
        n = np.linalg.norm(d)
        return p, (d / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])), self.tilt[k] + (self.tilt[k + 1] - self.tilt[k]) * t

    def ring_angle(self, x):
        if self.ring is None:
            return math.nan
        k = int(np.clip(np.searchsorted(self.cum, x) - 1, 0, len(self.P) - 2))
        return self.ring[k]


def _cut(pts, f0, f1):
    """Part of a [(Vector, tilt)] polyline between fractional indices f0 < f1."""
    n = len(pts)
    f0, f1 = max(0.0, f0), min(n - 1.0, f1)
    if f1 - f0 < 1e-6:
        return []

    def at(f):
        k = min(int(f), n - 2)
        t = f - k
        return (pts[k][0].lerp(pts[k + 1][0], t), pts[k][1] + (pts[k + 1][1] - pts[k][1]) * t)
    out = [at(f0)]
    out += [pts[k] for k in range(int(math.floor(f0)) + 1, int(math.ceil(f1)))]
    out.append(at(f1))
    return out


def _bezier(p0, h0, p1, h1, k=None, n=None):
    dist = (p1 - p0).length
    k = 0.45 * dist if k is None else k
    n = n or max(6, int(dist / 1.2))
    c0, c1 = p0 + h0 * k, p1 - h1 * k
    out = []
    for i in range(n + 1):
        t = i / n
        a = (1 - t) ** 3
        b = 3 * (1 - t) ** 2 * t
        c = 3 * (1 - t) * t ** 2
        d = t ** 3
        out.append(p0 * a + c0 * b + c1 * c + p1 * d)
    return out


def _heading(pts, at_end):
    a, b = (pts[-2][0], pts[-1][0]) if at_end else (pts[0][0], pts[1][0])
    v = b - a
    v.z = 0.0
    return v.normalized() if v.length > 1e-6 else Vector((1.0, 0.0, 0.0))


def _turn_speed(path, v_in):
    """Comfortable speed round a path: from its tightest bend."""
    if len(path) < 3:
        return v_in
    P = np.array([tuple(p) for p in path])[:, :2]
    d = np.diff(P, axis=0)
    ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    seg = np.linalg.norm(d, axis=1)
    if len(ang) < 2:
        return v_in
    curv = np.abs(np.diff(ang)) / np.maximum(0.5 * (seg[1:] + seg[:-1]), 1e-3)
    window = max(1, len(curv) // 4)
    kmax = float(np.max(np.convolve(curv, np.ones(window) / window, mode='same'))) if len(curv) else 0.0
    if kmax < 1e-4:
        return v_in
    return float(np.clip(math.sqrt(2.2 / kmax), 4.0, v_in))


class Network:
    """Lane stretches, intersection movements, and the intersections' rules."""

    def __init__(self, scene, road, left_hand, depsgraph=None):
        self.left_hand = left_hand
        self.segs = []
        roads = [o for o in scene.objects if traffic.is_traffic_road(o)]
        lanes = []
        for r in roads:
            lanes.extend(traffic.road_lanes(r, left_hand, depsgraph=depsgraph))
        links = traffic.link_lanes(lanes)
        self.juncs = []
        for o in scene.objects:
            if junctions.is_junction(o) and o.get("cf_layout"):
                lay = json.loads(o["cf_layout"])
                p = o.civil_junction
                lay.update(obj=o, control=p.control, stop_on=p.stop_on, G=p.green_time, Y=p.yellow_time,
                           AR=p.all_red_time)
                self.juncs.append(lay)

        # The roads that share traffic with `road`: through road-end links and intersections.
        adj = {}
        for i, k in links.items():
            a, b = lanes[i]["road"].name, lanes[k]["road"].name
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        for J in self.juncs:
            names = {arm["road"] for arm in J["arms"]}
            for a in names:
                adj.setdefault(a, set()).update(names - {a})
        comp, stack = {road.name}, [road.name]
        while stack:
            for nb in adj.get(stack.pop(), ()):
                if nb not in comp:
                    comp.add(nb)
                    stack.append(nb)
        self.roads = [r for r in roads if r.name in comp]
        self.juncs = [J for J in self.juncs if any(arm["road"] in comp for arm in J["arms"])]
        keep = [i for i, l in enumerate(lanes) if l["road"].name in comp]
        lines = {r.name: junctions.RoadLine(r, depsgraph) for r in self.roads}

        # Cut each lane where it enters and leaves intersections.
        first_piece, last_piece = {}, {}
        for li in keep:
            lane = lanes[li]
            rname = lane["road"].name
            line = lines[rname]
            props = lane["road"].civil_props
            direction = next(d for key, x, d, rank in traffic.lane_layout(props, left_hand) if key == lane["key"])
            pts = lane["pts"]
            n = len(pts)
            if n < 2 or len(line.d) != n:
                continue
            idx = np.arange(n, dtype=np.float64)

            def frac(s):
                f = float(np.interp(s, line.d, idx))
                return f if direction > 0 else (n - 1) - f
            events = []
            for ji, J in enumerate(self.juncs):
                for ai, arm in enumerate(J["arms"]):
                    if arm["road"] != rname:
                        continue
                    s_line = arm["s0"] + arm["sign"] * (arm["L"] - 0.9)
                    if direction == -arm["sign"]:
                        events.append((frac(s_line), 'end', ji, ai))
                    else:
                        events.append((frac(s_line), 'start', ji, ai))
            events.sort(key=lambda e: e[0])
            inside = bool(events) and events[0][1] == 'start'
            cur0, cur_tag = 0.0, None
            pieces = []
            for f, kind, ji, ai in events:
                if kind == 'end' and not inside:
                    pieces.append((cur0, f, cur_tag, (ji, ai)))
                    inside = True
                elif kind == 'start':
                    cur0, cur_tag = f, (ji, ai)
                    inside = False
            if not inside:
                pieces.append((cur0, n - 1.0, cur_tag, None))
            made = []
            for f0, f1, st, en in pieces:
                part = _cut(pts, f0, f1)
                if len(part) < 2:
                    continue
                s = self._add('LANE', [q for q, _ in part], [t for _, t in part], lane["speed"])
                s.start_j, s.end_j = st, en
                s.rank, s.n_dir, s.road = lane["rank"], lane["n_dir"], rname
                s.lane_pts = part
                made.append(s)
            if made:
                if made[0].start_j is None:
                    first_piece[li] = made[0]
                if made[-1].end_j is None:
                    last_piece[li] = made[-1]
        for i, k in links.items():
            if i in last_piece and k in first_piece:
                self._link(last_piece[i], first_piece[k])

        # Movements through each intersection.
        for ji, J in enumerate(self.juncs):
            entries = [s for s in self.segs if s.kind == 'LANE' and s.end_j and s.end_j[0] == ji]
            exits = [s for s in self.segs if s.kind == 'LANE' and s.start_j and s.start_j[0] == ji]
            J["moves"] = []
            C = Vector((J["C"][0], J["C"][1], 0.0))
            for e in entries:
                ai = e.end_j[1]
                h0 = _heading(e.lane_pts, True)
                for x in exits:
                    ak = x.start_j[1]
                    if ak == ai:
                        continue
                    h1 = _heading(x.lane_pts, False)
                    ang = math.atan2(h0.x * h1.y - h0.y * h1.x, h0.dot(h1))
                    if abs(ang) > math.radians(160):
                        continue
                    if abs(ang) < math.radians(35):
                        turn = 'THROUGH'
                    else:
                        turn = 'KERB' if ((ang < 0) != left_hand) else 'FAR'
                    if not J["rb"]:
                        if turn == 'THROUGH' and min(e.rank, x.n_dir - 1) != x.rank:
                            continue
                        if turn == 'KERB' and (e.rank != 0 or x.rank != 0):
                            continue
                        if turn == 'FAR' and (e.rank != e.n_dir - 1 or x.rank != x.n_dir - 1):
                            continue
                    elif e.rank != 0 or x.rank != 0:
                        continue
                    p0, p1 = e.lane_pts[-1][0], x.lane_pts[0][0]
                    if J["rb"]:
                        path, ring = self._ring_path(J, C, p0, h0, p1, h1)
                    else:
                        path, ring = _bezier(p0, h0, p1, h1), None
                    v = _turn_speed(path, min(e.v0, x.v0))
                    m = self._add('MOVE', path, [0.0] * len(path), v)
                    m.j, m.arm_in, m.arm_out, m.entry, m.turn = ji, ai, ak, e.id, turn
                    if ring is not None:
                        m.ring = np.array(ring)
                    self._link(e, m)
                    self._link(m, x)
                    J["moves"].append(m.id)
            # Which movements cross each other's paths.
            J["conflict"] = {}
            for a in J["moves"]:
                for b in J["moves"]:
                    if a >= b:
                        continue
                    sa, sb = self.segs[a], self.segs[b]
                    if sa.entry == sb.entry:
                        continue
                    A_, B_ = sa.P[:, :2], sb.P[:, :2]
                    dmin = np.min(np.hypot(A_[:, None, 0] - B_[None, :, 0], A_[:, None, 1] - B_[None, :, 1]))
                    if dmin < 2.4:
                        J["conflict"].setdefault(a, set()).add(b)
                        J["conflict"].setdefault(b, set()).add(a)
        self.entries = [s.id for s in self.segs if s.kind == 'LANE' and not s.pred and s.start_j is None]

    def _add(self, kind, pts, tilts, v0):
        s = Seg(len(self.segs), kind, pts, tilts, v0)
        self.segs.append(s)
        return s

    def _link(self, a, b):
        a.succ.append(b.id)
        b.pred.append(a.id)

    def _ring_path(self, J, C, p0, h0, p1, h1):
        """Entry curve, arc round the ring (anticlockwise for right-hand traffic), exit curve."""
        Rc = J["Rc"]
        circ = -1.0 if self.left_hand else 1.0
        a_in = math.atan2(p0.y - C.y, p0.x - C.x) + circ * 0.42
        a_out = math.atan2(p1.y - C.y, p1.x - C.x) - circ * 0.42
        sweep = ((a_out - a_in) * circ) % (2 * math.pi)
        if sweep < 0.2:
            sweep += 2 * math.pi

        def on_ring(a):
            q = Vector((C.x + Rc * math.cos(a), C.y + Rc * math.sin(a), 0.0))
            t = Vector((-math.sin(a), math.cos(a), 0.0)) * circ
            return q, t
        z0, z1 = p0.z, p1.z
        q_in, t_in = on_ring(a_in)
        q_out, t_out = on_ring(a_in + circ * sweep)
        q_in.z, q_out.z = z0, z1
        first = _bezier(p0, h0, q_in, t_in, k=0.4 * (q_in - p0).length)
        n_arc = max(4, int(sweep * Rc / 1.5))
        arc, ring = [], []
        for i in range(1, n_arc):
            a = a_in + circ * sweep * i / n_arc
            q, _ = on_ring(a)
            q.z = z0 + (z1 - z0) * i / n_arc
            arc.append(q)
            ring.append(a)
        last = _bezier(q_out, t_out, p1, h1, k=0.4 * (p1 - q_out).length)
        path = first + arc + last[1:]
        # Ring angle wherever the path is on or near the ring (the merge and the peel-off too), so
        # vehicles there see each other; nan further out.
        rings = []
        unwrap_ref = a_in
        for q in path:
            r = math.hypot(q.x - C.x, q.y - C.y)
            if r < Rc + 3.5:
                a = math.atan2(q.y - C.y, q.x - C.x)
                a = unwrap_ref + ((a - unwrap_ref + math.pi) % (2 * math.pi) - math.pi)
                unwrap_ref = a
                rings.append(a)
            else:
                rings.append(math.nan)
        return path, rings


# ------------------------------------------------------------------------------
# Driving
# ------------------------------------------------------------------------------

class Vehicle:
    __slots__ = ("kind", "length", "seg", "x", "v", "vf", "plan", "go", "stop_t", "arrived", "hidden", "rng")

    def __init__(self, kind, seg, x, v, vf, rng):
        self.kind, self.length = kind, traffic.VEHICLE_TYPES[kind][1]
        self.seg, self.x, self.v, self.vf = seg, x, v, vf
        self.plan, self.go, self.stop_t, self.arrived, self.hidden, self.rng = [], False, 0.0, None, False, rng


def _signal(J, group, t):
    """'G', 'Y' or 'R' for a signal group at absolute time t (same timing as the lamps)."""
    G, Y, AR = J["G"], J["Y"], J["AR"]
    cyc = 2.0 * (G + Y + AR)
    off = cyc * 0.5 * group
    u = math.fmod(t + cyc - off, cyc)
    if u < 0:
        u += cyc
    return 'G' if u < G else ('Y' if u < G + Y else 'R')


class Sim:
    def __init__(self, net, vehicles, fps):
        self.net, self.veh, self.dt, self.t = net, vehicles, 1.0 / fps, 0.0

    # -- planning -------------------------------------------------------------
    def extend(self, v):
        segs = self.net.segs
        last = v.plan[-1] if v.plan else v.seg
        while len(v.plan) < 4:
            succ = segs[last].succ
            if not succ:
                break
            if len(succ) == 1:
                nxt = succ[0]
            else:
                w = [TURN_WEIGHTS.get(segs[s].turn, 0.3) for s in succ]
                nxt = v.rng.choices(succ, weights=w)[0]
            v.plan.append(nxt)
            last = nxt

    # -- looking ahead --------------------------------------------------------
    def leader(self, v, on_seg):
        """(gap, speed of the vehicle ahead) along v's planned path."""
        segs = self.net.segs
        dist = -v.x
        best = (math.inf, 0.0)
        for sid in [v.seg] + v.plan:
            for x_w, w in on_seg.get(sid, ()):
                if w is v:
                    continue
                along = dist + x_w
                if sid == v.seg and x_w <= v.x:
                    continue
                if along > 0.0:
                    gap = along - (v.length + w.length) * 0.5
                    if gap < best[0]:
                        best = (gap, w.v)
                    break  # the list is sorted, the first one ahead is the nearest
            if best[0] < math.inf:
                break
            dist += segs[sid].length
            if dist > LOOKAHEAD:
                break
        # Round a roundabout, vehicles on other movements share the ring.
        s = segs[v.seg]
        if s.ring is not None:
            a = s.ring_angle(v.x)
            if not math.isnan(a):
                J = self.net.juncs[s.j]
                circ = -1.0 if self.net.left_hand else 1.0
                for mid in J["moves"]:
                    if mid == v.seg:
                        continue
                    for x_w, w in on_seg.get(mid, ()):
                        b = segs[mid].ring_angle(x_w)
                        if math.isnan(b):
                            continue
                        ahead = ((b - a) * circ) % (2 * math.pi)
                        gap = ahead * J["Rc"] - (v.length + w.length) * 0.5
                        if ahead < math.pi and gap < best[0]:
                            best = (gap, w.v)
        return best

    # -- the rules at intersections ---------------------------------------------
    def box_busy(self, J, mid, on_seg):
        segs = self.net.segs
        conf = J["conflict"].get(mid, ())
        for other in conf:
            if on_seg.get(other):
                return True
        # Vehicles already let go and about to enter count as inside: two never go into crossing
        # paths at the same moment.
        for sid in {segs[m].entry for m in J["moves"]}:
            if sid == segs[mid].entry:
                continue
            for x_w, w in on_seg.get(sid, ()):
                to_line = segs[sid].length - x_w
                if w.go and w.plan and w.plan[0] in conf and (to_line < 8.0 or to_line / max(w.v, 0.5) < 3.5):
                    return True
        entry = segs[mid].entry
        for m2 in J["moves"]:
            if segs[m2].entry == entry:
                for x_w, w in on_seg.get(m2, ()):
                    if x_w < w.length + 5.0:
                        return True
        return False

    def oncoming(self, J, mid, on_seg, who):
        """A vehicle with priority (`who(entry segment, its planned movement)`) will reach a crossing path soon."""
        segs = self.net.segs
        conf = J["conflict"].get(mid, set())
        for sid in {segs[m].entry for m in J["moves"]}:
            if sid == segs[mid].entry:
                continue
            for x_w, w in on_seg.get(sid, ()):
                if not w.plan or w.plan[0] not in conf or not who(segs[sid], w):
                    continue
                t_arrive = (segs[sid].length - x_w) / max(w.v, 0.5)
                if t_arrive < GAP_TIME and (w.v > 1.0 or w.go):
                    return True
        return False

    def permitted(self, v, on_seg):
        segs = self.net.segs
        s = segs[v.seg]
        mid = v.plan[0]
        J = self.net.juncs[s.end_j[0]]
        arm = J["arms"][s.end_j[1]]
        ctrl = J["control"]
        to_line = s.length - v.x - v.length * 0.5
        far_turn = segs[mid].turn == 'FAR'

        def stopped_long_enough():
            if to_line < 2.5 and v.v < 0.4:
                v.stop_t += self.dt
                if v.arrived is None:
                    v.arrived = self.t
            return v.stop_t >= STOP_HOLD

        if ctrl == 'SIGNALS':
            light = _signal(J, arm["group"], self.t)
            if light == 'R':
                return False
            if light == 'Y':
                # Only a moving vehicle too close to stop carries on; one waiting at the line stays.
                return v.v > 3.0 and to_line < v.v * v.v / (2.0 * 3.0) + 1.0
            if self.box_busy(J, mid, on_seg):
                return False
            if far_turn:
                return not self.oncoming(J, mid, on_seg,
                                         lambda e, w: J["arms"][e.end_j[1]]["group"] == arm["group"]
                                         and segs[w.plan[0]].turn != 'FAR')
            return True
        if ctrl == 'ROUNDABOUT':
            a_in = segs[mid].ring[np.nanargmin(np.where(np.isnan(segs[mid].ring), np.inf, 0.0))]
            circ = -1.0 if self.net.left_hand else 1.0
            for m2 in J["moves"]:
                for x_w, w in on_seg.get(m2, ()):
                    b = segs[m2].ring_angle(x_w)
                    if math.isnan(b):
                        if m2 != mid and segs[m2].entry == segs[mid].entry and x_w < w.length + 4.0:
                            return False
                        continue
                    upstream = ((a_in - b) * circ) % (2 * math.pi) * J["Rc"]
                    if upstream < 30.0:
                        return False
            return True
        minor = ctrl == 'ALL_STOP' or (ctrl == 'TWO_STOP' and arm["group"] == (1 if J["stop_on"] == 'B' else 0))
        if minor:
            if not stopped_long_enough():
                return False
            if self.box_busy(J, mid, on_seg):
                return False
            if ctrl == 'ALL_STOP':
                # First come, first served among the vehicles waiting at the other stop lines.
                for sid in {segs[m].entry for m in J["moves"]}:
                    if sid == v.seg:
                        continue
                    lst = on_seg.get(sid)
                    if lst:
                        x_w, w = lst[0]
                        if w.arrived is not None and w.arrived < v.arrived and w.plan and \
                                w.plan[0] in J["conflict"].get(mid, ()):
                            return False
                return True
            major = 1 - (1 if J["stop_on"] == 'B' else 0)
            return not self.oncoming(J, mid, on_seg, lambda e, w: J["arms"][e.end_j[1]]["group"] == major)
        # Main road at a two-way stop, or no control at all.
        if self.box_busy(J, mid, on_seg):
            return False
        if far_turn:
            return not self.oncoming(J, mid, on_seg, lambda e, w: segs[w.plan[0]].turn != 'FAR')
        return True

    # -- one step ------------------------------------------------------------------
    def step(self):
        segs = self.net.segs
        dt = self.dt
        on_seg = {}
        for v in self.veh:
            if not v.hidden:
                on_seg.setdefault(v.seg, []).append((v.x, v))
        for lst in on_seg.values():
            lst.sort(key=lambda e: e[0])
        acc = []
        for v in self.veh:
            if v.hidden:
                acc.append(0.0)
                continue
            self.extend(v)
            s = segs[v.seg]
            v0 = max(1.0, s.v0 * v.vf)
            gap, v_lead = self.leader(v, on_seg)
            # Ease off ahead of a slower movement (a turn) and for the stop line if not allowed on.
            to_end = s.length - v.x
            if v.plan:
                nxt = segs[v.plan[0]]
                v0 = min(v0, math.sqrt(max(0.0, (nxt.v0 * v.vf) ** 2 + 2.0 * B_COMF * max(0.0, to_end - 3.0))))
            if s.kind == 'LANE' and s.end_j is not None and v.plan:
                first = all(x_w <= v.x for x_w, w in on_seg.get(v.seg, ()) if w is not v)
                if first and to_end < 90.0:
                    # Looked at again every moment until too close to stop: lights change, and
                    # other traffic may have got there first.
                    if not v.go or to_end > v.v * v.v / (2 * 3.0) + 1.5:
                        v.go = self.permitted(v, on_seg)
                    if not v.go:
                        # Drawn up to the line: the model keeps S_MIN to whatever it stops for.
                        stop_gap = to_end - v.length * 0.5 - 0.6 + S_MIN
                        if stop_gap < gap:
                            gap, v_lead = max(stop_gap, 0.01), 0.0
            # Intelligent Driver Model.
            dv = v.v - v_lead
            s_star = S_MIN + max(0.0, v.v * HEADWAY + v.v * dv / (2.0 * math.sqrt(A_MAX * B_COMF)))
            free = 1.0 - (v.v / v0) ** 4
            a = A_MAX * (free - (s_star / max(gap, 0.01)) ** 2) if gap < math.inf else A_MAX * free
            acc.append(max(-9.0, min(A_MAX, a)))
        for v, a in zip(self.veh, acc):
            if v.hidden:
                self.respawn(v, on_seg)
                continue
            v.v = max(0.0, v.v + a * dt)
            v.x += v.v * dt
            while v.x > segs[v.seg].length:
                over = v.x - segs[v.seg].length
                if not v.plan:
                    self.extend(v)
                if not v.plan:
                    v.hidden = True
                    break
                v.seg = v.plan.pop(0)
                v.x = over
                if segs[v.seg].kind == 'LANE':
                    v.go, v.stop_t, v.arrived = False, 0.0, None
        self.t += dt

    def respawn(self, v, on_seg):
        segs = self.net.segs
        entries = list(self.net.entries)
        v.rng.shuffle(entries)
        for sid in entries:
            busy = any(x_w < 25.0 for x_w, w in on_seg.get(sid, ()))
            if not busy:
                v.seg, v.x, v.v, v.plan, v.hidden = sid, v.length * 0.5, segs[sid].v0 * v.vf * 0.8, [], False
                v.go, v.stop_t, v.arrived = False, 0.0, None
                on_seg.setdefault(sid, []).insert(0, (v.x, v))
                return

    def pose(self, v):
        if v.hidden:
            return (0.0, 0.0, HIDDEN_Z), (0.0, 0.0, 0.0)
        p, d, tilt = self.net.segs[v.seg].at(v.x)
        yaw = math.atan2(d[1], d[0])
        pitch = -math.asin(max(-1.0, min(1.0, d[2])))
        return (p[0], p[1], p[2]), (tilt, pitch, yaw)


# ------------------------------------------------------------------------------
# Building and baking
# ------------------------------------------------------------------------------

def build(context, road, net):
    """Simulated traffic for `road`'s network (see Network). Returns (vehicles, roads)."""
    from bpy_extras import anim_utils
    p = road.civil_props
    scene = context.scene
    for r in net.roads:
        traffic.remove_traffic(r)
    lane_segs = [s for s in net.segs if s.kind == 'LANE' and s.length > 1.0]
    if not lane_segs:
        return 0, net.roads

    col = bpy.data.collections.new(f"Traffic_{road.name}")
    scene.collection.children.link(col)
    col["cf_simulated"] = True
    for r in net.roads:
        r.civil_props.traffic_collection = col

    rng = random.Random(p.traffic_seed)
    vehicles = []
    for s in lane_segs:
        wanted = int(round(s.length * p.traffic_density / 100.0))
        x = rng.uniform(4.0, 12.0)
        for _ in range(wanted):
            kind = traffic._pick_kind(rng, p.traffic_truck_share)
            L = traffic.VEHICLE_TYPES[kind][1]
            x += L * 0.5
            if x > s.length - L * 0.5 - 2.0:
                break
            vehicles.append(Vehicle(kind, s.id, x, s.v0 * 0.7, rng.uniform(0.9, 1.08), random.Random(rng.random())))
            x += L * 0.5 + traffic.MIN_GAP + rng.uniform(0.0, 100.0 / max(p.traffic_density, 0.1))
    fps = scene.render.fps / max(scene.render.fps_base, 1e-6)
    sim = Sim(net, vehicles, fps)
    f0, f1 = scene.frame_start, max(scene.frame_end, scene.frame_start + int(p.traffic_duration * fps))
    scene.frame_end = f1
    sim.t = f0 / fps - WARMUP
    for _ in range(int(WARMUP * fps)):
        sim.step()
    frames = f1 - f0 + 1
    loc = np.zeros((len(vehicles), frames, 3))
    rot = np.zeros((len(vehicles), frames, 3))
    for k in range(frames):
        for i, v in enumerate(vehicles):
            loc[i, k], rot[i, k] = sim.pose(v)
        sim.step()

    # Write the motion as keyframes.
    times = np.arange(f0, f1 + 1, dtype=np.float64)
    for i, v in enumerate(vehicles):
        obj = bpy.data.objects.new(v.kind.title().replace('_', ' '), traffic.vehicle_mesh(v.kind))
        col.objects.link(obj)
        name, rgba = traffic._pick_color(rng)
        obj.color = rgba
        obj.material_slots[traffic.PAINT].link = 'OBJECT'
        obj.material_slots[traffic.PAINT].material = traffic.paint_material(name, rgba)
        obj["cf_kind"] = v.kind
        obj.location = loc[i, 0]
        obj.rotation_euler = rot[i, 0]
        obj.keyframe_insert("location", frame=f0)
        obj.keyframe_insert("rotation_euler", frame=f0)
        ad = obj.animation_data
        cb = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
        rot[i, :, 2] = np.unwrap(rot[i, :, 2])
        for fc in cb.fcurves:
            vals = (loc if fc.data_path == "location" else rot)[i, :, fc.array_index]
            kp = fc.keyframe_points
            kp.clear()
            kp.add(frames)
            co = np.empty(2 * frames)
            co[0::2] = times
            co[1::2] = vals
            kp.foreach_set("co", co)
            kp.foreach_set("interpolation", [1] * frames)  # linear: one key per frame
            fc.update()
    col["cf_frames"] = (f0, f1)
    return len(vehicles), net.roads
