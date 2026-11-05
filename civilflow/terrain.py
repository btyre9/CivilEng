# SPDX-License-Identifier: GPL-3.0-or-later
"""
Real-world terrain for CivilFlow.

- Downloads elevation (USGS 3DEP) and aerial photos (USDA NAIP, via The National Map) for any
  spot in the US, or imports a DEM GeoTIFF from disk, and builds a textured ground mesh at true
  scale (1 Blender unit = 1 m). All terrains in a scene share one origin, so tiles line up.
- Answers "how high is the ground here?" for roads (Ground), drapes roads over the land with a
  grade limit (drape), sorts each stretch of road into on-ground / bridge / tunnel (classify),
  and cuts the terrain down where a road runs through it (carve).

Map data from USGS and USDA is in the public domain.
"""

import math
import os
import ssl
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor

import bpy
import numpy as np
from mathutils import Quaternion, Vector
from mathutils.bvhtree import BVHTree

from . import geo

DEM_URL = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
NAIP_URL = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer/exportImage"
MAX_DEM_PX = 8000
MAX_IMG_PX = 4000       # the photo server's largest image; bigger photos come in tiles
MAX_PHOTO_MP = 256      # total photo budget (~1 GB in Blender); past it the photo gets coarser
PHOTO_WORKERS = 4
MAX_VERTS = 4_000_000

SLOPE = 1.75      # fill banks, horizontal : vertical
SLOPE_CUT = 1.0   # cut banks are steeper; the terrain is carved to match
MAX_CUT = 40.0    # tallest cut slope carved into the terrain (m); deeper than that wants a tunnel
ROAD, BRIDGE, TUNNEL = 0, 1, 2


# ------------------------------------------------------------------------------
# Scene origin: every terrain in a scene is placed relative to the first one
# ------------------------------------------------------------------------------

def scene_geo(scene):
    g = scene.get("cf_geo")
    if not g:
        return None
    return {"zone": int(g["zone"]), "north": bool(g["north"]), "e0": float(g["e0"]),
            "n0": float(g["n0"]), "z0": float(g["z0"])}


def scene_zone(scene, lat, lon):
    g = scene_geo(scene)
    return (g["zone"], g["north"]) if g else (geo.utm_zone(lon), lat >= 0.0)


def is_terrain(obj):
    return bool(obj and obj.type == 'MESH' and obj.get("cf_terrain"))


def cursor_geo(scene):
    """(lat, lon, elevation or None) of the 3D cursor, when the scene has a terrain origin."""
    g = scene_geo(scene)
    if not g:
        return None
    c = scene.cursor.location
    lat, lon = geo.utm_to_latlon(g["e0"] + c.x, g["n0"] + c.y, g["zone"], g["north"])
    elev = None
    hit = terrain_under(scene, c)
    if hit:
        elev = hit[1] + g["z0"]
    return float(lat), float(lon), elev


# ------------------------------------------------------------------------------
# Ground height queries
# ------------------------------------------------------------------------------

class Ground:
    """Heights of a terrain object at world (x, y) points; nan where the terrain doesn't reach."""

    def __init__(self, obj):
        self.name = obj.name
        self.mw = np.array(obj.matrix_world, dtype=np.float64)
        self.mi = np.linalg.inv(self.mw)
        self.grid = None
        self.bvh = None
        me = obj.data
        if is_terrain(obj) and "cf_ground" in me.attributes:
            x0, y0, dx, dy, nx, ny = obj["cf_grid"]
            nx, ny = int(nx), int(ny)
            a = np.empty(nx * ny, dtype=np.float32)
            me.attributes["cf_ground"].data.foreach_get("value", a)
            self.H = a.reshape(ny, nx).astype(np.float64)
            self.grid = (float(x0), float(y0), float(dx), float(dy), nx, ny)
        else:
            mw = obj.matrix_world
            verts = [mw @ v.co for v in me.vertices]
            polys = [tuple(p.vertices) for p in me.polygons]
            self.top = max((v.z for v in verts), default=0.0) + 10.0
            self.bvh = BVHTree.FromPolygons(verts, polys) if polys else None

    def world_heights(self, x, y):
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        if self.grid is None:
            out = np.full(x.shape, np.nan)
            if self.bvh is not None:
                down = Vector((0.0, 0.0, -1.0))
                for k, (px, py) in enumerate(zip(x.ravel(), y.ravel())):
                    hit = self.bvh.ray_cast(Vector((px, py, self.top)), down)
                    if hit[0] is not None:
                        out.flat[k] = hit[0].z
            return out
        mi, mw = self.mi, self.mw
        lx = mi[0, 0] * x + mi[0, 1] * y + mi[0, 3]
        ly = mi[1, 0] * x + mi[1, 1] * y + mi[1, 3]
        x0, y0, dx, dy, nx, ny = self.grid
        gx = (lx - x0) / dx
        gy = (ly - y0) / dy
        inside = (gx >= 0.0) & (gx <= nx - 1) & (gy >= 0.0) & (gy <= ny - 1)
        gx = np.clip(gx, 0.0, nx - 1.000001)
        gy = np.clip(gy, 0.0, ny - 1.000001)
        i = gx.astype(np.int64)
        j = gy.astype(np.int64)
        fx, fy = gx - i, gy - j
        H = self.H
        h = (H[j, i] * (1 - fx) * (1 - fy) + H[j, i + 1] * fx * (1 - fy)
             + H[j + 1, i] * (1 - fx) * fy + H[j + 1, i + 1] * fx * fy)
        wz = mw[2, 0] * lx + mw[2, 1] * ly + mw[2, 2] * h + mw[2, 3]
        return np.where(inside, wz, np.nan)


_GROUND = {}


def ground_for(obj):
    if obj is None or obj.type != 'MESH':
        return None
    key = (obj.data.as_pointer(), obj.get("cf_version"), len(obj.data.vertices),
           tuple(round(v, 6) for row in obj.matrix_world for v in row))
    hit = _GROUND.get(obj.name)
    if hit and hit[0] == key:
        return hit[1]
    g = Ground(obj)
    _GROUND[obj.name] = (key, g)
    return g


def clear_cache():
    _GROUND.clear()


def terrain_under(scene, world_pt):
    """(terrain object, world ground z) under a world point, or None."""
    for o in scene.objects:
        if is_terrain(o):
            h = ground_for(o).world_heights([world_pt[0]], [world_pt[1]])[0]
            if not np.isnan(h):
                return o, float(h)
    return None


def local_heights(obj, ground, pts):
    """Ground z, in obj's local space, straight below each of obj's local points (N, 3)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    M = np.array(obj.matrix_world, dtype=np.float64)
    P = pts @ M[:3, :3].T + M[:3, 3]
    wz = ground.world_heights(P[:, 0], P[:, 1])
    W = np.stack([P[:, 0], P[:, 1], wz], axis=1)
    Mi = np.linalg.inv(M)
    return (W @ Mi[:3, :3].T + Mi[:3, 3])[:, 2]


def fill_along(a, fallback=0.0):
    """Fills nan gaps in a 1-D profile from the nearest known values."""
    a = np.asarray(a, dtype=np.float64)
    ok = ~np.isnan(a)
    if ok.all():
        return a
    if not ok.any():
        return np.full(a.shape, fallback)
    idx = np.arange(len(a))
    return np.interp(idx, idx[ok], a[ok])


# ------------------------------------------------------------------------------
# Roads on the land
# ------------------------------------------------------------------------------

def _box(z, k):
    if k < 1 or len(z) < 3:
        return z
    zp = np.pad(z, k, mode="edge")
    c = np.concatenate([[0.0], np.cumsum(zp)])
    return (c[2 * k + 1:] - c[:-2 * k - 1]) / (2 * k + 1)


def drape(road, samples, ground, pin_start_z=None):
    """
    Lays the road's local samples over the ground: the ground under the centreline, smoothed,
    raised by the height above ground and held to the maximum grade. pin_start_z (world) keeps
    the start on the end of the road this one continues from. Banking is kept.
    """
    p = road.civil_props
    n = len(samples)
    if n < 2:
        return samples
    M = road.matrix_world
    Mi = M.inverted_safe()
    R = M.to_3x3()
    Ri = Mi.to_3x3()
    W = [M @ s[0] for s in samples]
    xs = np.array([w.x for w in W])
    ys = np.array([w.y for w in W])
    g = ground.world_heights(xs, ys)
    if np.isnan(g).all():
        return samples
    g = fill_along(g)
    d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    step = max(1e-3, d[-1] / (n - 1))

    z = g
    if p.terrain_smoothing > 0.0:
        k = int(round(p.terrain_smoothing / (2.0 * step)))
        z = _box(_box(z, k), k)
    z = z + p.terrain_offset

    w = np.ones(n)
    # Overpasses you placed lift the road until there's the clearance you asked for underneath.
    deck = p.deck_thickness + 0.1
    for st in p.structures:
        if st.kind != 'BRIDGE':
            continue
        sel = np.abs(d - st.center) <= st.length * 0.5
        need = g + st.clearance + deck
        lift = sel & (need >= z - 0.01)
        z = np.where(lift, need, z)
        w = np.where(lift, 1000.0, w)
    if pin_start_z is not None:
        z = z.copy()
        z[0] = pin_start_z
        w[0] = 1e6  # the start stays on the end of the road this one continues from
    z = fit_grade(z, (p.max_grade / 100.0) * np.diff(d), w)
    # Round off the kinks between slopes (vertical curves); averaging keeps the grade limit.
    k = int(round(min(60.0, max(10.0, p.terrain_smoothing * 0.4)) / (2.0 * step)))
    z = _box(z, k)
    if pin_start_z is not None:
        L = max(30.0, p.terrain_smoothing)
        t = np.clip(1.0 - d / L, 0.0, 1.0)
        z = z + (pin_start_z - z[0]) * (t * t * (3.0 - 2.0 * t))

    up = Vector((0.0, 0.0, 1.0))
    P = [Vector((xs[i], ys[i], z[i])) for i in range(n)]
    out = []
    for i in range(n):
        t = P[min(n - 1, i + 1)] - P[max(0, i - 1)]
        t = t.normalized() if t.length > 1e-9 else (R @ samples[i][1]).normalized()
        r_old = (R @ samples[i][3]).normalized()
        a = math.asin(max(-1.0, min(1.0, r_old.z)))
        rf = t.cross(up)
        rf = rf.normalized() if rf.length > 1e-6 else r_old
        nf = rf.cross(t).normalized()
        right = (rf * math.cos(a) + nf * math.sin(a)).normalized()
        normal = right.cross(t).normalized()
        out.append((Mi @ P[i], (Ri @ t).normalized(), (Ri @ normal).normalized(), (Ri @ right).normalized()))
    return out


def fit_grade(target, max_rise, weights=None):
    """Best of fitting from either end (the block merging below is greedy, so they can differ)."""
    t = np.asarray(target, dtype=np.float64)
    c = np.asarray(max_rise, dtype=np.float64)
    w = np.ones(len(t)) if weights is None else np.asarray(weights, dtype=np.float64)
    a = _fit_grade(t, c, w)
    b = _fit_grade(t[::-1], c[::-1], w[::-1])[::-1]
    return a if np.sum(w * (a - t) ** 2) <= np.sum(w * (b - t) ** 2) else b


def _fit_grade(target, max_rise, weights=None):
    """
    The profile closest to `target` (weighted least squares) whose rise between neighbouring
    samples never exceeds max_rise[i] either way. Pool-adjacent-violators: samples join into
    blocks that climb or fall at the limit, each block settling where its cut and fill balance.
    """
    t = np.asarray(target, dtype=np.float64)
    n = len(t)
    if n < 2:
        return t.copy()
    c = np.asarray(max_rise, dtype=np.float64)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    sign = np.zeros(max(n - 1, 1))
    # Each block: [start, end, W, S, off_last] with S = sum w * (t - offset from the block start)
    blocks = []
    for i in range(n):
        blocks.append([i, i, w[i], w[i] * t[i], 0.0])
        while len(blocks) > 1:
            b1, b2 = blocks[-2], blocks[-1]
            k = b1[1]
            gap = b2[3] / b2[2] - (b1[3] / b1[2] + b1[4])
            if gap > c[k] + 1e-9:
                s = 1.0
            elif gap < -c[k] - 1e-9:
                s = -1.0
            else:
                break
            shift = b1[4] + s * c[k]
            b1[3] += b2[3] - b2[2] * shift
            b1[2] += b2[2]
            b1[4] = shift + b2[4]
            b1[1] = b2[1]
            sign[k] = s
            blocks.pop()
    z = np.empty(n)
    for start, end, W, S, _off in blocks:
        z[start] = S / W
        for k in range(start, end):
            z[k + 1] = z[k] + sign[k] * c[k]
    return z


def _runs(mode):
    """[(kind, first, last)] for each stretch of equal values."""
    runs = []
    a = 0
    for i in range(1, len(mode) + 1):
        if i == len(mode) or mode[i] != mode[a]:
            runs.append((int(mode[a]), a, i - 1))
            a = i
    return runs


def _run_len(d, a, b):
    return d[min(b + 1, len(d) - 1)] - d[a]


OVERRIDE_MODE = {'BRIDGE': BRIDGE, 'BANK': ROAD, 'TUNNEL': TUNNEL}


def classify(d, surface_z, ground_z, props, manual_tunnel=None, overrides=()):
    """
    Per-sample ROAD / BRIDGE / TUNNEL, with short stretches tidied away. overrides are
    (kind, from, to) distances along the road that you set by hand; they win.
    """
    n = len(d)
    mode = np.full(n, ROAD, dtype=np.int64)
    h = surface_z - ground_z
    if props.is_overpass:
        mode[:] = BRIDGE
    elif props.auto_bridges:
        mode[h > props.bridge_min_height] = BRIDGE
        # Close short dips back to ground between two bridges, then drop stub bridges.
        runs = _runs(mode)
        for k, (kind, a, b) in enumerate(runs):
            if kind == ROAD and 0 < k < len(runs) - 1 and _run_len(d, a, b) < 30.0:
                mode[a:b + 1] = BRIDGE
        for kind, a, b in _runs(mode):
            if kind == BRIDGE and _run_len(d, a, b) < 12.0:
                mode[a:b + 1] = ROAD
    if props.auto_tunnels:
        deep = (-h) > props.tunnel_min_depth
        mode[deep] = TUNNEL
        runs = _runs(mode)
        for k, (kind, a, b) in enumerate(runs):
            if kind != TUNNEL and 0 < k < len(runs) - 1 and _run_len(d, a, b) < 40.0 \
                    and runs[k - 1][0] == TUNNEL and runs[k + 1][0] == TUNNEL:
                mode[a:b + 1] = TUNNEL
        for kind, a, b in _runs(mode):
            if kind == TUNNEL and _run_len(d, a, b) < 30.0 and not (manual_tunnel and manual_tunnel[0] <= a <= manual_tunnel[1]):
                mode[a:b + 1] = ROAD if not props.is_overpass else BRIDGE
    if manual_tunnel and manual_tunnel[1] > manual_tunnel[0]:
        mode[manual_tunnel[0]:manual_tunnel[1] + 1] = TUNNEL
    for kind, d0, d1 in overrides:
        mode[(d >= d0) & (d <= d1)] = OVERRIDE_MODE[kind]
    return mode


def runs_with_overlap(mode):
    """Runs extended to the next run's first sample, so pieces built per run meet up."""
    n = len(mode)
    return [(kind, a, min(b + 1, n - 1)) for kind, a, b in _runs(mode)]


# ------------------------------------------------------------------------------
# Cutting the terrain down where roads pass through it
# ------------------------------------------------------------------------------

_CARVE = {}


def record_carve(road, terrain_obj, samples, mode, left_edges, right_edges, verge, deck_t,
                 covered=None, tube_pad=0.0, holes=()):
    """
    Remembers, in world space, the trench this road needs in the terrain. Tunnel samples under
    enough ground (covered) leave the hill alone; shallow ones sit in an open cut. holes are the
    portal openings, as (x, y, ax, ay, along0, along1, half_width) in world space.
    """
    M = road.matrix_world
    n = len(samples)
    x = np.empty(n); y = np.empty(n); z = np.empty(n)
    rx = np.empty(n); ry = np.empty(n)
    wl = np.empty(n); wr = np.empty(n); sl = np.empty(n)
    active = np.ones(n, dtype=bool)
    for i, (pos, _t, _nrm, right) in enumerate(samples):
        wc = M @ pos
        wle = M @ (pos + right * left_edges[i])
        wre = M @ (pos + right * right_edges[i])
        r = (M.to_3x3() @ right)
        rh = Vector((r.x, r.y))
        rh = rh.normalized() if rh.length > 1e-6 else Vector((0.0, -1.0))
        x[i], y[i], rx[i], ry[i] = wc.x, wc.y, rh.x, rh.y
        low = min(wc.z, wle.z, wre.z)
        if mode[i] == BRIDGE:
            z[i] = low - deck_t - 0.6
            wl[i], wr[i], sl[i] = -left_edges[i] + 1.0, right_edges[i] + 1.0, 1.0
        elif mode[i] == TUNNEL:
            z[i] = low - 0.4
            wl[i], wr[i], sl[i] = -left_edges[i] + tube_pad, right_edges[i] + tube_pad, SLOPE_CUT
            active[i] = covered is None or not covered[i]
        else:
            z[i] = low - 0.4  # stays under the earth banks, which tuck 0.25 m into the ground
            wl[i], wr[i], sl[i] = -left_edges[i] + verge, right_edges[i] + verge, SLOPE_CUT
    step = 0.0
    if n > 1:
        step = float(np.hypot(np.diff(x), np.diff(y)).sum() / (n - 1))
    _CARVE[road.name] = dict(terrain=terrain_obj.name, x=x, y=y, z=z, rx=rx, ry=ry,
                             wl=wl, wr=wr, sl=sl, active=active, step=step, mode=np.array(mode),
                             holes=list(holes))


def forget_carve(road_name):
    return _CARVE.pop(road_name, None)


def carved_by(road_name):
    rec = _CARVE.get(road_name)
    return rec["terrain"] if rec else None


def _road_carves(road, terrain_obj):
    p = road.civil_props
    return (road.type == 'MESH' and p.is_civil_object and p.shape_terrain and p.terrain_object == terrain_obj)


def _hole_mask(g, holes):
    """Faces (ny-1, nx-1) whose centre lies inside any portal opening."""
    x0, y0, dx, dy, nx, ny = g.grid
    mi = g.mi
    cx = x0 + dx * (np.arange(nx - 1) + 0.5)
    cy = y0 + dy * (np.arange(ny - 1) + 0.5)
    mask = np.zeros((ny - 1, nx - 1), dtype=bool)
    for hx, hy, ax, ay, a0, a1, lat in holes:
        lx = mi[0, 0] * hx + mi[0, 1] * hy + mi[0, 3]
        ly = mi[1, 0] * hx + mi[1, 1] * hy + mi[1, 3]
        axl = mi[0, 0] * ax + mi[0, 1] * ay
        ayl = mi[1, 0] * ax + mi[1, 1] * ay
        ln = math.hypot(axl, ayl) or 1.0
        axl, ayl = axl / ln, ayl / ln
        reach = max(abs(a0), abs(a1)) + lat + abs(dx)
        i0 = max(0, int((lx - reach - x0) / dx))
        i1 = min(nx - 1, int((lx + reach - x0) / dx) + 2)
        j0 = max(0, int((ly - reach - y0) / dy))
        j1 = min(ny - 1, int((ly + reach - y0) / dy) + 2)
        if i1 <= i0 or j1 <= j0:
            continue
        DX = cx[i0:i1][None, :] - lx
        DY = cy[j0:j1][:, None] - ly
        along = DX * axl + DY * ayl
        side = DY * axl - DX * ayl
        mask[j0:j1, i0:i1] |= (along >= a0) & (along <= a1) & (np.abs(side) <= lat)
    return mask


def hole_extent(terrain_obj, hole):
    """(min along, max along, max |side|) covered by the faces a portal opening removes, or None."""
    g = ground_for(terrain_obj)
    if g is None or g.grid is None:
        return None
    mask = _hole_mask(g, [hole])
    if not mask.any():
        return None
    x0, y0, dx, dy, nx, ny = g.grid
    jj, ii = np.nonzero(mask)
    hx, hy, ax, ay = hole[:4]
    mw = g.mw
    ext_a, ext_s = [], []
    for ci, cj in ((0, 0), (1, 0), (0, 1), (1, 1)):
        lx = x0 + dx * (ii + ci)
        ly = y0 + dy * (jj + cj)
        wx = mw[0, 0] * lx + mw[0, 1] * ly + mw[0, 3]
        wy = mw[1, 0] * lx + mw[1, 1] * ly + mw[1, 3]
        ext_a.append((wx - hx) * ax + (wy - hy) * ay)
        ext_s.append(np.abs((wy - hy) * ax - (wx - hx) * ay))
    a = np.concatenate(ext_a)
    return float(a.min()), float(a.max()), float(np.concatenate(ext_s).max())


HOLE_MOD = "CivilFlow Portals"


def _hole_node_group():
    ng = bpy.data.node_groups.get("CF_TerrainPortals")
    if ng is not None:
        return ng
    ng = bpy.data.node_groups.new("CF_TerrainPortals", 'GeometryNodeTree')
    ng.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    gin = ng.nodes.new('NodeGroupInput')
    gout = ng.nodes.new('NodeGroupOutput')
    attr = ng.nodes.new('GeometryNodeInputNamedAttribute')
    attr.data_type = 'BOOLEAN'
    attr.inputs["Name"].default_value = "cf_hole"
    delete = ng.nodes.new('GeometryNodeDeleteGeometry')
    delete.domain = 'FACE'
    gin.location, attr.location, delete.location, gout.location = (-400, 0), (-400, -150), (-150, 0), (100, 0)
    ng.links.new(gin.outputs[0], delete.inputs["Geometry"])
    ng.links.new(attr.outputs["Attribute"], delete.inputs["Selection"])
    ng.links.new(delete.outputs["Geometry"], gout.inputs[0])
    return ng


def _write_holes(terrain_obj, mask):
    me = terrain_obj.data
    attr = me.attributes.get("cf_hole")
    flat = mask.ravel()
    if attr is None:
        if not flat.any():
            return
        attr = me.attributes.new("cf_hole", 'BOOLEAN', 'FACE')
    old = np.zeros(len(flat), dtype=bool)
    attr.data.foreach_get("value", old)
    if not np.array_equal(old, flat):
        attr.data.foreach_set("value", flat)
    if flat.any() and terrain_obj.modifiers.get(HOLE_MOD) is None:
        mod = terrain_obj.modifiers.new(HOLE_MOD, 'NODES')
        mod.node_group = _hole_node_group()


def carve(terrain_obj, rebuild_fn=None):
    """Resets the terrain to the original ground, then cuts every road's trench into it."""
    if not is_terrain(terrain_obj):
        return
    g = ground_for(terrain_obj)
    if g is None or g.grid is None:
        return
    # Roads that have not been built this session yet (e.g. after opening the file).
    if rebuild_fn:
        for o in bpy.data.objects:
            if o.name not in _CARVE and _road_carves(o, terrain_obj):
                rebuild_fn(o)

    x0, y0, dx, dy, nx, ny = g.grid
    Z = g.H.copy()
    holes = []
    xs = x0 + dx * np.arange(nx)
    ys = y0 + dy * np.arange(ny)
    mi = g.mi
    for name, rec in list(_CARVE.items()):
        road = bpy.data.objects.get(name)
        if rec["terrain"] != terrain_obj.name or road is None or not _road_carves(road, terrain_obj):
            continue
        holes.extend(rec.get("holes", ()))
        lx = mi[0, 0] * rec["x"] + mi[0, 1] * rec["y"] + mi[0, 3]
        ly = mi[1, 0] * rec["x"] + mi[1, 1] * rec["y"] + mi[1, 3]
        lz = mi[2, 0] * rec["x"] + mi[2, 1] * rec["y"] + mi[2, 2] * rec["z"] + mi[2, 3]
        rx = mi[0, 0] * rec["rx"] + mi[0, 1] * rec["ry"]
        ry = mi[1, 0] * rec["rx"] + mi[1, 1] * rec["ry"]
        rl = np.hypot(rx, ry)
        rl[rl < 1e-9] = 1.0
        rx, ry = rx / rl, ry / rl
        n = len(lx)
        stride = max(1, int(abs(dx) * 0.7 / max(rec["step"], 1e-3)))
        half_slab = 0.65 * stride * max(rec["step"], 1e-3) + 0.15 * abs(dx)
        for k in range(0, n, stride):
            if not rec["active"][k]:
                continue
            # One grid cell of extra flat floor: a triangle spanning floor and slope would otherwise
            # bulge above the road edge (the earth bank covers this strip).
            wl, wr, sl = rec["wl"][k] + abs(dx), rec["wr"][k] + abs(dx), rec["sl"][k]
            reach = max(wl, wr) + MAX_CUT * sl
            i0 = max(0, int((lx[k] - reach - x0) / dx))
            i1 = min(nx, int((lx[k] + reach - x0) / dx) + 2)
            j0 = max(0, int((ly[k] - reach - y0) / dy))
            j1 = min(ny, int((ly[k] + reach - y0) / dy) + 2)
            if i1 <= i0 or j1 <= j0:
                continue
            DX = xs[i0:i1][None, :] - lx[k]
            DY = ys[j0:j1][:, None] - ly[k]
            s = DX * rx[k] + DY * ry[k]
            along = DY * rx[k] - DX * ry[k]
            over = np.where(s >= 0.0, s - wr, -s - wl)
            env = lz[k] + np.maximum(over, 0.0) / sl
            env = np.where((np.abs(along) <= half_slab) & (over <= MAX_CUT * sl), env, np.inf)
            sub = Z[j0:j1, i0:i1]
            np.minimum(sub, env, out=sub)

    _write_holes(terrain_obj, _hole_mask(g, holes))
    me = terrain_obj.data
    co = np.empty((nx * ny, 3), dtype=np.float32)
    me.vertices.foreach_get("co", co.ravel())
    z_new = Z.ravel().astype(np.float32)
    if np.array_equal(co[:, 2], z_new):
        return
    co[:, 2] = z_new
    me.vertices.foreach_set("co", co.ravel())
    me.update()


# ------------------------------------------------------------------------------
# Building the terrain mesh
# ------------------------------------------------------------------------------

def _fill_nans(H):
    if not np.isnan(H).any():
        return H
    if np.isnan(H).all():
        raise ValueError("There is no elevation data for this area.")
    H = H.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for _ in range(64):
            m = np.isnan(H)
            if not m.any():
                break
            P = np.pad(H, 1, mode="edge")
            nb = np.stack([P[:-2, 1:-1], P[2:, 1:-1], P[1:-1, :-2], P[1:-1, 2:]])
            avg = np.nanmean(nb, axis=0)
            H[m] = avg[m]
    H[np.isnan(H)] = np.nanmin(H)
    return H


def terrain_material(name, image):
    mat = bpy.data.materials.new(name=name + "_Ground")
    earth = (0.32, 0.30, 0.22, 1.0)
    if mat.node_tree:
        nt = mat.node_tree
        bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if bsdf is None:
            bsdf = nt.nodes.new(type='ShaderNodeBsdfPrincipled')
            out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None) \
                or nt.nodes.new(type='ShaderNodeOutputMaterial')
            nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 1.0
        if image is not None:
            tex = nt.nodes.new(type='ShaderNodeTexImage')
            tex.image = image
            tex.extension = 'EXTEND'
            tex.interpolation = 'Linear'
            tex.location = (bsdf.location.x - 420, bsdf.location.y)
            nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
            nt.nodes.active = tex
        elif "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = earth
    mat.diffuse_color = (0.45, 0.42, 0.33, 1.0) if image is not None else earth
    return mat


def build_terrain(context, name, E, N, H, zone, north, photos=(), info=""):
    """
    Ground mesh from a height grid. E (nx,) and N (ny,) are ascending scene-zone UTM metres,
    H (ny, nx) elevations in metres (nan for unknown). photos is a list of
    (image, uv_fn, core) where uv_fn(E, N) -> (u, v) maps that image and core = (e0, n0, e1, n1)
    is the area it covers; each face takes the photo whose core holds its centre.
    """
    scene = context.scene
    E = np.asarray(E, dtype=np.float64)
    N = np.asarray(N, dtype=np.float64)
    H = _fill_nans(np.asarray(H, dtype=np.float64))
    nx, ny = len(E), len(N)
    if nx < 2 or ny < 2:
        raise ValueError("The area is too small for the mesh spacing.")
    g = scene_geo(scene)
    if g is None:
        scene["cf_geo"] = {"zone": int(zone), "north": int(bool(north)), "e0": float(round(E.mean())),
                           "n0": float(round(N.mean())), "z0": float(math.floor(np.min(H)))}
        g = scene_geo(scene)

    x = E - g["e0"]
    y = N - g["n0"]
    Z = (H - g["z0"]).astype(np.float32)
    X, Y = np.meshgrid(x, y)
    co = np.empty((nx * ny, 3), dtype=np.float32)
    co[:, 0] = X.ravel()
    co[:, 1] = Y.ravel()
    co[:, 2] = Z.ravel()

    me = bpy.data.meshes.new(name)
    me.vertices.add(nx * ny)
    me.vertices.foreach_set("co", co.ravel())
    ii, jj = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1))
    v0 = (jj * nx + ii).ravel()
    quads = np.stack([v0, v0 + 1, v0 + 1 + nx, v0 + nx], axis=1).astype(np.int32)
    nf = len(quads)
    me.loops.add(nf * 4)
    me.loops.foreach_set("vertex_index", quads.ravel())
    me.polygons.add(nf)
    me.polygons.foreach_set("loop_start", np.arange(0, nf * 4, 4, dtype=np.int32))
    me.update(calc_edges=True)
    me.shade_smooth()
    attr = me.attributes.new("cf_ground", 'FLOAT', 'POINT')
    attr.data.foreach_set("value", Z.ravel())

    photos = list(photos)
    if photos:
        # Corner positions per loop, and which photo each face takes.
        VE = X.ravel() + g["e0"]
        VN = Y.ravel() + g["n0"]
        LE = VE[quads]
        LN = VN[quads]
        fe, fn = LE.mean(axis=1), LN.mean(axis=1)
        which = np.zeros(nf, dtype=np.int32)
        if len(photos) > 1:
            for k, (_img, _fn, core) in enumerate(photos):
                e0_, n0_, e1_, n1_ = core
                which[(fe >= e0_) & (fe < e1_) & (fn >= n0_) & (fn < n1_)] = k
        uv = np.zeros((nf, 4, 2), dtype=np.float32)
        for k, (_img, uv_fn, _core) in enumerate(photos):
            sel = which == k
            if not sel.any():
                continue
            u, v = uv_fn(LE[sel], LN[sel])
            uv[sel, :, 0] = u
            uv[sel, :, 1] = v
        uvl = me.uv_layers.new(name="UVMap")
        uvl.data.foreach_set("uv", uv.ravel())
        for k, (img, _fn, _core) in enumerate(photos):
            me.materials.append(terrain_material(name if len(photos) == 1 else "%s_%d" % (name, k + 1), img))
        me.polygons.foreach_set("material_index", which)
    else:
        me.materials.append(terrain_material(name, None))

    obj = bpy.data.objects.new(name, me)
    context.collection.objects.link(obj)
    obj["cf_terrain"] = 1
    obj["cf_grid"] = [float(x[0]), float(y[0]), float(x[1] - x[0]), float(y[1] - y[0]), float(nx), float(ny)]
    obj["cf_version"] = int(time.time() * 1000) % (2 ** 31)
    obj["cf_info"] = info
    obj["cf_elev_range"] = [float(np.min(H)), float(np.max(H))]
    return obj


def frame_terrain(context, obj):
    """Top view over the new terrain, far enough clipping to see it all, photo shown in Solid mode."""
    x0, y0, dx, dy, nx, ny = obj["cf_grid"]
    w, h = dx * (nx - 1), dy * (ny - 1)
    cx, cy = x0 + w * 0.5, y0 + h * 0.5
    g = ground_for(obj)
    cz = g.world_heights([cx], [cy])[0]
    cz = 0.0 if np.isnan(cz) else float(cz)
    context.scene.cursor.location = (cx, cy, cz)
    size = max(w, h)
    wm = context.window_manager
    for window in (wm.windows if wm else []):
        for area in window.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            sp = area.spaces.active
            sp.clip_end = max(sp.clip_end, size * 6.0)
            sp.clip_start = max(sp.clip_start, 0.05)
            if sp.shading.type == 'SOLID':
                sp.shading.color_type = 'TEXTURE'
            r3d = sp.region_3d
            r3d.view_perspective = 'ORTHO'
            r3d.view_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
            r3d.view_location = (cx, cy, cz)
            r3d.view_distance = size * 0.8


# ------------------------------------------------------------------------------
# Getting the data
# ------------------------------------------------------------------------------

def _download_dir():
    try:
        return bpy.utils.extension_path_user(__package__, path="downloads", create=True)
    except (ValueError, AttributeError, TypeError):
        path = os.path.join(tempfile.gettempdir(), "civilflow_terrain")
        os.makedirs(path, exist_ok=True)
        return path


def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch(url, params, path, timeout=180):
    """Downloads one exportImage request to path (kept as a cache for the same area)."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params),
                                 headers={"User-Agent": "CivilFlow-Blender-Extension"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
                data = r.read()
                ctype = (r.headers.get("Content-Type") or "").lower()
            break
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == 3:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 3:
                raise
        time.sleep(3.0 * (attempt + 1))  # busy server: wait and try again
    if "json" in ctype or "html" in ctype or data[:1] in (b"{", b"<"):
        msg = data[:300].decode("utf-8", "replace")
        raise RuntimeError("The USGS server didn't return an image: " + msg)
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)
    return path


def download_terrain(context, lat, lon, width_m, height_m, spacing, imagery=True, image_res=0.6):
    zone, north = scene_zone(context.scene, lat, lon)
    epsg = geo.utm_epsg(zone, north)
    ec, nc = geo.latlon_to_utm(lat, lon, zone, north)
    ec, nc = float(ec), float(nc)
    nxp = max(2, int(round(width_m / spacing)))
    nyp = max(2, int(round(height_m / spacing)))
    if nxp > MAX_DEM_PX or nyp > MAX_DEM_PX:
        raise ValueError("Too many points: use a mesh spacing of at least %.1f m for this area."
                         % (max(width_m, height_m) / MAX_DEM_PX))
    if nxp * nyp > MAX_VERTS:
        raise ValueError("That would be %.1f million vertices. Use a bigger mesh spacing or a smaller area."
                         % (nxp * nyp / 1e6))
    xmin, xmax = ec - width_m * 0.5, ec + width_m * 0.5
    ymin, ymax = nc - height_m * 0.5, nc + height_m * 0.5
    bbox = "%.3f,%.3f,%.3f,%.3f" % (xmin, ymin, xmax, ymax)
    folder = _download_dir()
    tag = "%.5f_%.5f_%dx%d" % (lat, lon, round(width_m), round(height_m))

    dem_path = fetch(DEM_URL, dict(bbox=bbox, bboxSR=epsg, imageSR=epsg, size="%d,%d" % (nxp, nyp),
                                   format="tiff", pixelType="F32", interpolation="RSP_BilinearInterpolation",
                                   noDataInterpretation="esriNoDataMatchAny", f="image"),
                     os.path.join(folder, "dem_%s_%gm.tif" % (tag, spacing)))
    t = geo.GeoTiff(dem_path)
    try:
        rows = t.read()
    finally:
        t.close()
    if np.isnan(rows).all() or np.nanmax(rows) < -1000:
        raise ValueError("USGS has no elevation data here. 3DEP covers the United States and its territories.")
    H = rows[::-1, :]  # north-up rows -> ascending northing
    E = xmin + (np.arange(nxp) + 0.5) * (width_m / nxp)
    N = ymin + (np.arange(nyp) + 0.5) * (height_m / nyp)

    photos, img_note = [], ""
    if imagery:
        photos, res, count = download_photo(xmin, ymin, width_m, height_m, image_res, spacing, epsg, folder, tag)
        img_note = " · NAIP %.2f m%s" % (res, " in %d tiles" % count if count > 1 else "")

    info = "%.5f, %.5f · USGS 3DEP %g m%s" % (lat, lon, spacing, img_note)
    name = "Terrain_%.4f_%.4f" % (lat, lon)
    return build_terrain(context, name, E, N, H, zone, north, photos, info)


def plan_photo(width_m, height_m, image_res, spacing):
    """(metres per pixel, tiles across, tiles down, tile width, tile height, overlap) for an area."""
    res = max(image_res, math.sqrt(width_m * height_m / (MAX_PHOTO_MP * 1e6)))
    # Tiles overlap by a couple of mesh cells, so faces on a tile edge never run off their photo.
    margin = 2.0 * spacing + 4.0 * res
    core = MAX_IMG_PX * res - 2.0 * margin
    if core < 50.0:
        res = (50.0 + 2.0 * margin) / MAX_IMG_PX
        margin = 2.0 * spacing + 4.0 * res
        core = MAX_IMG_PX * res - 2.0 * margin
    nx = max(1, math.ceil(width_m / core))
    ny = max(1, math.ceil(height_m / core))
    return res, nx, ny, width_m / nx, height_m / ny, margin


def download_photo(xmin, ymin, width_m, height_m, image_res, spacing, epsg, folder, tag):
    """NAIP photo of the area as overlapping tiles fetched side by side; returns (photos, res, count)."""
    res, nx, ny, tx, ty, margin = plan_photo(width_m, height_m, image_res, spacing)
    jobs = []
    for j in range(ny):
        for i in range(nx):
            c0, r0 = xmin + i * tx, ymin + j * ty
            box = (c0 - margin, r0 - margin, c0 + tx + margin, r0 + ty + margin)
            ipx = min(MAX_IMG_PX, int(round((box[2] - box[0]) / res)))
            ipy = min(MAX_IMG_PX, int(round((box[3] - box[1]) / res)))
            path = os.path.join(folder, "naip_%s_%.2fm_%d_%d.jpg" % (tag, res, i, j))
            params = dict(bbox="%.3f,%.3f,%.3f,%.3f" % box, bboxSR=epsg, imageSR=epsg,
                          size="%d,%d" % (ipx, ipy), format="jpg", f="image")
            core = (c0, r0, c0 + tx, r0 + ty)
            # The outermost tiles take everything beyond the area too.
            core = (core[0] if i else -1e12, core[1] if j else -1e12,
                    core[2] if i < nx - 1 else 1e12, core[3] if j < ny - 1 else 1e12)
            jobs.append((params, path, box, core))
    with ThreadPoolExecutor(max_workers=PHOTO_WORKERS) as pool:
        list(pool.map(lambda job: fetch(NAIP_URL, job[0], job[1]), jobs))

    photos = []
    for params, path, box, core in jobs:
        img = bpy.data.images.load(path, check_existing=True)
        img.alpha_mode = 'NONE'
        try:
            img.pack()
        except RuntimeError:
            pass

        def uv_fn(EE, NN, b=box):
            return (EE - b[0]) / (b[2] - b[0]), (NN - b[1]) / (b[3] - b[1])
        photos.append((img, uv_fn, core))
    return photos, res, len(jobs)


def _bilinear(win, col, row):
    h, w = win.shape
    inside = (col >= -0.5) & (col <= w - 0.5) & (row >= -0.5) & (row <= h - 0.5)
    c = np.clip(col, 0.0, w - 1.000001)
    r = np.clip(row, 0.0, h - 1.000001)
    i = c.astype(np.int64)
    j = r.astype(np.int64)
    fx, fy = c - i, r - j
    i1 = np.minimum(i + 1, w - 1)
    j1 = np.minimum(j + 1, h - 1)
    v = (win[j, i] * (1 - fx) * (1 - fy) + win[j, i1] * fx * (1 - fy)
         + win[j1, i] * (1 - fx) * fy + win[j1, i1] * fx * fy)
    return np.where(inside, v, np.nan)


def import_terrain_file(context, dem_path, image_path="", spacing=4.0, crop_m=0.0):
    """Terrain from a DEM GeoTIFF (UTM, lat/long or web-mercator), optionally with an aerial image."""
    t = geo.GeoTiff(dem_path)
    try:
        if t.x0 is None:
            raise ValueError("This GeoTIFF has no map position (georeferencing).")
        lat, lon = t.center_latlon()
        zone, north = scene_zone(context.scene, lat, lon)
        cx, cy = t.corners()
        clat, clon = t.to_latlon(cx, cy)
        ce, cn = geo.latlon_to_utm(np.asarray(clat), np.asarray(clon), zone, north)
        emin, emax = max(ce[0], ce[3]), min(ce[1], ce[2])
        nmin, nmax = max(cn[2], cn[3]), min(cn[0], cn[1])
        if crop_m > 0.0:
            ec, nc = (emin + emax) * 0.5, (nmin + nmax) * 0.5
            emin, emax = max(emin, ec - crop_m * 0.5), min(emax, ec + crop_m * 0.5)
            nmin, nmax = max(nmin, nc - crop_m * 0.5), min(nmax, nc + crop_m * 0.5)
        E = np.arange(emin + spacing * 0.5, emax, spacing)
        N = np.arange(nmin + spacing * 0.5, nmax, spacing)
        if len(E) * len(N) > MAX_VERTS:
            raise ValueError("That would be %.1f million vertices. Use a bigger mesh spacing or crop the area."
                             % (len(E) * len(N) / 1e6))
        EE, NN = np.meshgrid(E, N)
        fx, fy = t.to_file_xy(EE, NN, zone, north)
        col = (fx - t.x0) / t.sx - 0.5
        row = (t.y0 - fy) / t.sy - 0.5
        c0 = int(max(0, math.floor(np.nanmin(col)) - 1))
        c1 = int(min(t.width - 1, math.ceil(np.nanmax(col)) + 1))
        r0 = int(max(0, math.floor(np.nanmin(row)) - 1))
        r1 = int(min(t.height - 1, math.ceil(np.nanmax(row)) + 1))
        win = t.read(c0, r0, c1 - c0 + 1, r1 - r0 + 1)
        H = _bilinear(win, col - c0, row - r0)
        res = abs(t.sx) * (111320.0 * math.cos(math.radians(lat)) if t.geographic else 1.0)
    finally:
        t.close()

    image, uv_fn = None, None
    if image_path:
        image = bpy.data.images.load(image_path, check_existing=True)
        image.alpha_mode = 'NONE'
        it = None
        ref = None
        if image_path.lower().endswith((".tif", ".tiff")):
            it = geo.GeoTiff(image_path)
            if it.x0 is not None:
                ref = (it.x0, it.y0, it.sx, it.sy)
        if ref is None:
            ref = geo.read_world_file(image_path)
            if it is not None:
                it.close()
            it = geo.GeoTiff(dem_path)  # a world file is in the DEM's map projection
        if ref is None:
            image = None
        else:
            ix0, iy0, isx, isy = ref
            iw, ih = image.size

            def uv_fn(EE, NN, it=it):
                x, y = it.to_file_xy(EE, NN, zone, north)
                return (x - ix0) / (isx * iw), 1.0 - (iy0 - y) / (isy * ih)
    name = "Terrain_" + os.path.splitext(os.path.basename(dem_path))[0][:40]
    info = "%.5f, %.5f · %s (%.2g m data, %g m mesh)" % (lat, lon, os.path.basename(dem_path), res, spacing)
    photos = [(image, uv_fn, None)] if image is not None else []
    obj = build_terrain(context, name, E, N, H, zone, north, photos, info)
    return obj, (image_path and image is None)


# ------------------------------------------------------------------------------
# UI
# ------------------------------------------------------------------------------

class CIVIL_PG_TerrainSettings(bpy.types.PropertyGroup):
    location: bpy.props.StringProperty(
        name="Location", default="34.8660, -118.8870",
        description="Latitude, longitude of the middle of the area. In Google Maps, right-click a "
                    "spot and click the numbers at the top to copy them",
    )
    width_km: bpy.props.FloatProperty(name="Width", default=2.0, min=0.2, max=30.0, unit='NONE',
                                      description="East-west size of the area, in kilometres")
    height_km: bpy.props.FloatProperty(name="Height", default=2.0, min=0.2, max=30.0,
                                       description="North-south size of the area, in kilometres")
    spacing: bpy.props.FloatProperty(
        name="Mesh Spacing", default=4.0, min=1.0, max=100.0, unit='LENGTH',
        description="Distance between terrain vertices. USGS has 1 m data for most of the US; "
                    "smaller spacing is sharper but heavier",
    )
    imagery: bpy.props.BoolProperty(name="Aerial Photo", default=True,
                                    description="Drape a USDA NAIP aerial photo over the terrain")
    image_res: bpy.props.FloatProperty(
        name="Photo Detail", default=0.3, min=0.3, max=10.0, unit='LENGTH',
        description="Metres per photo pixel; 0.3 m is NAIP's sharpest. Big areas download in tiles, "
                    "and very big ones (over ~4.8 km square) get a little coarser to stay within memory",
    )
    dem_file: bpy.props.StringProperty(name="Elevation File", subtype='FILE_PATH',
                                       description="A DEM GeoTIFF, e.g. a USGS 3DEP 1 m or 1/3 arc-second tile")
    image_file: bpy.props.StringProperty(name="Photo File", subtype='FILE_PATH',
                                         description="Optional aerial image: a GeoTIFF, or a JPG/PNG with a world file")
    crop_km: bpy.props.FloatProperty(name="Crop To", default=3.0, min=0.0, max=50.0,
                                     description="Use only this many kilometres around the file's centre (0 = all)")


def _busy(context, on):
    try:
        if on:
            context.window.cursor_modal_set('WAIT')
        else:
            context.window.cursor_modal_restore()
    except AttributeError:
        pass


def _finish(context, obj):
    for o in context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj
    frame_terrain(context, obj)


class CIVIL_OT_TerrainDownload(bpy.types.Operator):
    bl_idname = "civil.terrain_download"
    bl_label = "Download Terrain"
    bl_description = ("Download USGS elevation and a NAIP aerial photo for the area around the location "
                      "and build it as a ground mesh. US only; free public-domain data")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.civil_terrain
        if not getattr(bpy.app, "online_access", True):
            self.report({'ERROR'}, "Online access is off. Turn on Preferences > System > Network > "
                                   "Allow Online Access, or import files instead")
            return {'CANCELLED'}
        ll = geo.parse_latlon(s.location)
        if ll is None:
            self.report({'ERROR'}, "Location should be latitude, longitude, e.g. 34.8660, -118.8870")
            return {'CANCELLED'}
        _busy(context, True)
        try:
            obj = download_terrain(context, ll[0], ll[1], s.width_km * 1000.0, s.height_km * 1000.0,
                                   s.spacing, s.imagery, s.image_res)
        except Exception as e:  # network, server and data errors all end up here
            self.report({'ERROR'}, "Terrain download failed: %s" % e)
            return {'CANCELLED'}
        finally:
            _busy(context, False)
        _finish(context, obj)
        self.report({'INFO'}, "Terrain ready. Draw a curve over it (top view) and pick a road preset.")
        return {'FINISHED'}


class CIVIL_OT_TerrainImportFile(bpy.types.Operator):
    bl_idname = "civil.terrain_import_file"
    bl_label = "Import Terrain Files"
    bl_description = "Build a ground mesh from a DEM GeoTIFF on disk, with an optional aerial photo"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.civil_terrain
        dem = bpy.path.abspath(s.dem_file)
        img = bpy.path.abspath(s.image_file) if s.image_file else ""
        if not os.path.isfile(dem):
            self.report({'ERROR'}, "Pick an elevation GeoTIFF first")
            return {'CANCELLED'}
        _busy(context, True)
        try:
            obj, photo_unplaced = import_terrain_file(context, dem, img, s.spacing, s.crop_km * 1000.0)
        except Exception as e:
            self.report({'ERROR'}, "Terrain import failed: %s" % e)
            return {'CANCELLED'}
        finally:
            _busy(context, False)
        _finish(context, obj)
        if photo_unplaced:
            self.report({'WARNING'}, "Terrain imported, but the photo has no map position (GeoTIFF tags or world file)")
        else:
            self.report({'INFO'}, "Terrain imported")
        return {'FINISHED'}


class CIVIL_PT_Terrain(bpy.types.Panel):
    bl_label = "Real-World Terrain"
    bl_idname = "CIVIL_PT_Terrain"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'CivilEng'
    bl_order = 0

    def draw(self, context):
        layout = self.layout
        s = context.scene.civil_terrain

        box = layout.box()
        box.label(text="Download (USGS, US only)", icon='URL')
        col = box.column(align=True)
        col.prop(s, "location", text="")
        row = col.row(align=True)
        row.prop(s, "width_km", text="Width km")
        row.prop(s, "height_km", text="Height km")
        col.prop(s, "spacing")
        row = col.row(align=True)
        row.prop(s, "imagery")
        sub = row.row(align=True)
        sub.enabled = s.imagery
        sub.prop(s, "image_res", text="Detail")
        verts = (s.width_km * 1000.0 / s.spacing) * (s.height_km * 1000.0 / s.spacing)
        box.label(text="About %.2f million vertices" % (verts / 1e6),
                  icon='ERROR' if verts > MAX_VERTS else 'INFO')
        if s.imagery:
            res, tnx, tny, *_ = plan_photo(s.width_km * 1000.0, s.height_km * 1000.0, s.image_res, s.spacing)
            n_t = tnx * tny
            box.label(text="Photo %.2f m per pixel%s" % (res, ", %d tiles" % n_t if n_t > 1 else ""), icon='IMAGE_DATA')
        box.operator("civil.terrain_download", icon='IMPORT')

        fbox = layout.box()
        fbox.label(text="From Files", icon='FILE_FOLDER')
        col = fbox.column(align=True)
        col.prop(s, "dem_file", text="")
        col.prop(s, "image_file", text="")
        col.prop(s, "crop_km", text="Crop To km")
        fbox.operator("civil.terrain_import_file", icon='IMPORT')

        obj = context.active_object
        if is_terrain(obj):
            ibox = layout.box()
            ibox.label(text=obj.name, icon='WORLD')
            ibox.label(text=str(obj.get("cf_info", "")))
            rng = obj.get("cf_elev_range")
            if rng:
                ibox.label(text="Elevation %.0f to %.0f m" % (rng[0], rng[1]))
        cg = cursor_geo(context.scene)
        if cg:
            lat, lon, elev = cg
            txt = "Cursor %.5f, %.5f" % (lat, lon)
            if elev is not None:
                txt += "  ·  %.1f m" % elev
            layout.label(text=txt, icon='PIVOT_CURSOR')


classes = (
    CIVIL_PG_TerrainSettings,
    CIVIL_OT_TerrainDownload,
    CIVIL_OT_TerrainImportFile,
    CIVIL_PT_Terrain,
)
