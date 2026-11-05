# SPDX-License-Identifier: GPL-3.0-or-later
"""
Lakes and ponds on CivilFlow terrain, from the USGS National Hydrography Dataset (NHD).

USGS elevation is 'hydro-flattened': every lake is already one flat level in the DEM, and that
level is the water surface. For each lake outline (lakes, ponds, reservoirs) this:

- lowers the lakebed under the outline, shelving down from the shore, so the water has depth;
- builds a flat water surface from the outline, in one "<terrain> Water" object;
- stores the water level per terrain vertex (attribute cf_water), so roads keep their clearance
  over water and automatic bridges span it.

The elevation as downloaded is kept in cf_dem, so water can be removed or fetched again.
NHD data is in the public domain.
"""

import json
import os

import bpy
import numpy as np
from mathutils import Vector
from mathutils.geometry import tessellate_polygon

from . import geo, terrain

NHD_URL = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/12/query"
FTYPES = {390: "lake or pond", 436: "reservoir", 493: "estuary"}
WATER_ATTR = "cf_water"     # water surface height per vertex (DRY where there's none)
DEM_ATTR = "cf_dem"         # ground as downloaded, before lakebeds were dug
DRY = -1.0e6
SHELF = 0.6                 # depth right at the shore (m)
SHORE_GRADE = 0.15          # extra depth per metre out from the shore
MAX_DEPTH = 3.0
CLEARANCE = 5.0             # roads cross water at least this high above it (m, to the deck's underside)
MATERIAL = "CivilFlow Water"


# ------------------------------------------------------------------------------
# Getting the outlines
# ------------------------------------------------------------------------------

def terrain_extent(scene, obj):
    """(lat0, lon0, lat1, lon1) of a terrain, and its scene geo (zone, north, e0, n0, z0)."""
    g = terrain.scene_geo(scene)
    if g is None:
        raise ValueError("This terrain has no map position (download or import it with CivilFlow)")
    x0, y0, dx, dy, nx, ny = obj["cf_grid"]
    mw = obj.matrix_world
    xs = (x0, x0 + dx * (nx - 1))
    ys = (y0, y0 + dy * (ny - 1))
    lats, lons = [], []
    for x in xs:
        for y in ys:
            w = mw @ Vector((x, y, 0.0))
            lat, lon = geo.utm_to_latlon(w.x + g["e0"], w.y + g["n0"], g["zone"], bool(g["north"]))
            lats.append(float(lat))
            lons.append(float(lon))
    return (min(lats), min(lons), max(lats), max(lons)), g


def fetch_outlines(scene, obj):
    """
    Lake outlines over a terrain, in its local coordinates: a list of
    (name, kind, [ring (N, 2), ...]) where the first ring is the shore and the rest are islands.
    """
    (lat0, lon0, lat1, lon1), g = terrain_extent(scene, obj)
    pad = 0.002
    epsg = geo.utm_epsg(g["zone"], bool(g["north"]))
    x0, y0, dx, dy, nx, ny = obj["cf_grid"]
    folder = terrain._download_dir()
    tag = "%.5f_%.5f_%.5f_%.5f" % (lat0, lon0, lat1, lon1)
    mi = obj.matrix_world.inverted()
    out = []
    offset = 0
    while True:
        params = dict(geometry="%.6f,%.6f,%.6f,%.6f" % (lon0 - pad, lat0 - pad, lon1 + pad, lat1 + pad),
                      geometryType="esriGeometryEnvelope", inSR=4326, outSR=epsg,
                      spatialRel="esriSpatialRelIntersects",
                      where="FTYPE IN (%s)" % ",".join(str(k) for k in FTYPES),
                      outFields="GNIS_NAME,FTYPE", returnGeometry="true", geometryPrecision=2,
                      maxAllowableOffset="%.2f" % (abs(dx) * 0.25), resultOffset=offset,
                      resultRecordCount=1000, f="json")
        path = os.path.join(folder, "nhd_%s_%d.json" % (tag, offset))
        try:
            terrain.fetch(NHD_URL, params, path, timeout=90, expect="json")
        except Exception as e:
            raise RuntimeError("the lake outlines didn't arrive (%s). Try again in a minute"
                               % terrain._describe(e)) from e
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if "error" in data:
            os.remove(path)
            raise RuntimeError("the USGS lake service said: %s" % data["error"].get("message", data["error"]))
        feats = data.get("features", [])
        for f in feats:
            rings = []
            for ring in (f.get("geometry") or {}).get("rings", []):
                a = np.array(ring, dtype=np.float64)
                if len(a) < 4:
                    continue
                # Map metres -> scene -> terrain local.
                wx = a[:, 0] - g["e0"]
                wy = a[:, 1] - g["n0"]
                lx = mi[0][0] * wx + mi[0][1] * wy + mi[0][3]
                ly = mi[1][0] * wx + mi[1][1] * wy + mi[1][3]
                rings.append(np.stack([lx, ly], axis=1)[:-1])
            if rings:
                attrs = f.get("attributes", {})
                out.append((attrs.get("GNIS_NAME") or "", FTYPES.get(attrs.get("FTYPE"), "water"), rings))
        if not data.get("exceededTransferLimit") or not feats:
            break
        offset += len(feats)
    return out


# ------------------------------------------------------------------------------
# Shaping the terrain
# ------------------------------------------------------------------------------

def _area(ring):
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _fill_mask(rings, xs, ys):
    """Grid vertices inside the outline (even-odd: islands stay dry)."""
    mask = np.zeros((len(ys), len(xs)), dtype=bool)
    segs = []
    for r in rings:
        segs.append(np.concatenate([r, np.roll(r, -1, axis=0)], axis=1))
    S = np.concatenate(segs)
    ymin, ymax = S[:, [1, 3]].min(), S[:, [1, 3]].max()
    dy = ys[1] - ys[0]
    j0 = max(0, int(np.floor((ymin - ys[0]) / dy)))
    j1 = min(len(ys), int(np.ceil((ymax - ys[0]) / dy)) + 1)
    x1, y1, x2, y2 = S[:, 0], S[:, 1], S[:, 2], S[:, 3]
    for j in range(j0, j1):
        y = ys[j]
        hit = ((y1 <= y) & (y < y2)) | ((y2 <= y) & (y < y1))
        if not hit.any():
            continue
        xi = np.sort(x1[hit] + (y - y1[hit]) * (x2[hit] - x1[hit]) / (y2[hit] - y1[hit]))
        for a, b in zip(xi[0::2], xi[1::2]):
            i0 = np.searchsorted(xs, a)
            i1 = np.searchsorted(xs, b)
            mask[j, i0:i1] = True
    return mask


def _shore_rings(mask, steps):
    """How many cells in from the shore each wet vertex is (1 = next to dry land)."""
    ring = np.zeros(mask.shape, dtype=np.int32)
    cur = mask.copy()
    for k in range(1, steps + 1):
        P = np.pad(cur, 1, mode="edge")  # the terrain edge is not a shore
        inner = cur & P[:-2, 1:-1] & P[2:, 1:-1] & P[1:-1, :-2] & P[1:-1, 2:]
        ring[cur & ~inner] = k
        cur = inner
        if not cur.any():
            break
    ring[cur] = steps + 1
    return ring


def _clip_ring(ring, x0, y0, x1, y1):
    """Sutherland-Hodgman clip of a ring to a rectangle."""
    pts = [tuple(p) for p in ring]
    for axis, lim, keep_ge in ((0, x0, True), (0, x1, False), (1, y0, True), (1, y1, False)):
        if not pts:
            break
        res = []
        for k in range(len(pts)):
            p, q = pts[k - 1], pts[k]
            pin = p[axis] >= lim if keep_ge else p[axis] <= lim
            qin = q[axis] >= lim if keep_ge else q[axis] <= lim
            if qin:
                if not pin:
                    res.append(_cross(p, q, axis, lim))
                res.append(q)
            elif pin:
                res.append(_cross(p, q, axis, lim))
        pts = res
    return np.array(pts) if len(pts) >= 3 else None


def _cross(p, q, axis, lim):
    t = (lim - p[axis]) / (q[axis] - p[axis])
    return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))


def water_material():
    mat = bpy.data.materials.get(MATERIAL)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(MATERIAL)
    mat.diffuse_color = (0.10, 0.22, 0.28, 1.0)
    mat.roughness = 0.05
    nt = mat.node_tree
    if nt is None:
        return mat
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if bsdf is None:
        bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
        out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None) or nt.nodes.new('ShaderNodeOutputMaterial')
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    bsdf.inputs["Base Color"].default_value = (0.012, 0.045, 0.05, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.04
    if "IOR" in bsdf.inputs:
        bsdf.inputs["IOR"].default_value = 1.333
    # Small ripples.
    coord = nt.nodes.new('ShaderNodeTexCoord')
    noise = nt.nodes.new('ShaderNodeTexNoise')
    noise.inputs["Scale"].default_value = 0.35
    noise.inputs["Detail"].default_value = 6.0
    bump = nt.nodes.new('ShaderNodeBump')
    bump.inputs["Strength"].default_value = 0.08
    coord.location = (bsdf.location.x - 750, bsdf.location.y - 350)
    noise.location = (bsdf.location.x - 550, bsdf.location.y - 350)
    bump.location = (bsdf.location.x - 300, bsdf.location.y - 350)
    nt.links.new(coord.outputs["Object"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def water_object(obj):
    for o in bpy.data.objects:
        if o.get("cf_water_of") == obj.name and o.type == 'MESH':
            return o
    return None


def apply_water(context, obj, outlines):
    """Digs lakebeds under the outlines and builds the water surface. Returns the lakes made."""
    me = obj.data
    x0, y0, dx, dy, nx, ny = obj["cf_grid"]
    nx, ny = int(nx), int(ny)
    if DEM_ATTR not in me.attributes:
        a = np.empty(nx * ny, dtype=np.float32)
        me.attributes["cf_ground"].data.foreach_get("value", a)
        me.attributes.new(DEM_ATTR, 'FLOAT', 'POINT').data.foreach_set("value", a)
    dem = np.empty(nx * ny, dtype=np.float32)
    me.attributes[DEM_ATTR].data.foreach_get("value", dem)
    H = dem.reshape(ny, nx).astype(np.float64)
    bed = H.copy()
    W = np.full((ny, nx), DRY)
    xs = x0 + dx * np.arange(nx)
    ys = y0 + dy * np.arange(ny)
    cell = min(abs(dx), abs(dy))
    steps = int(np.ceil((MAX_DEPTH - SHELF) / (SHORE_GRADE * cell))) + 1
    lakes = []
    for name, kind, rings in outlines:
        mask = _fill_mask(rings, xs, ys)
        if mask.sum() < 3:
            continue  # smaller than the mesh can show
        level = float(np.median(H[mask]))
        depth = np.minimum(MAX_DEPTH, SHELF + (_shore_rings(mask, steps) - 1) * cell * SHORE_GRADE)
        bed = np.where(mask, np.minimum(bed, level - depth), bed)
        W = np.where(mask, np.maximum(W, level), W)
        lakes.append((name, kind, rings, level))

    me.attributes["cf_ground"].data.foreach_set("value", bed.ravel().astype(np.float32))
    wa = me.attributes.get(WATER_ATTR) or me.attributes.new(WATER_ATTR, 'FLOAT', 'POINT')
    wa.data.foreach_set("value", W.ravel().astype(np.float32))
    co = np.empty((nx * ny, 3), dtype=np.float32)
    me.vertices.foreach_get("co", co.ravel())
    co[:, 2] = bed.ravel()
    me.vertices.foreach_set("co", co.ravel())
    me.update()
    obj["cf_version"] = int(obj.get("cf_version", 0)) + 1
    terrain.clear_cache()

    # The water surfaces, clipped to the terrain.
    ex0, ex1 = min(xs[0], xs[-1]), max(xs[0], xs[-1])
    ey0, ey1 = min(ys[0], ys[-1]), max(ys[0], ys[-1])
    verts, faces = [], []
    for name, kind, rings, level in lakes:
        clipped = [r for r in (_clip_ring(r, ex0, ey0, ex1, ey1) for r in rings) if r is not None]
        clipped = [r for r in clipped if abs(_area(r)) > 1.0]
        if not clipped:
            continue
        z = level + 0.02
        loops = [[Vector((p[0], p[1], z)) for p in r] for r in clipped]
        base = len(verts)
        for loop in loops:
            verts.extend(loop)
        for tri in tessellate_polygon(loops):
            faces.append(tuple(base + k for k in tri))
    wobj = water_object(obj)
    name = obj.name + " Water"
    wme = bpy.data.meshes.new(name)
    wme.from_pydata([tuple(v) for v in verts], [], faces)
    for p in wme.polygons:
        if p.normal.z < 0:
            p.flip()
    wme.update()
    wme.materials.append(water_material())
    if wobj is None:
        wobj = bpy.data.objects.new(name, wme)
        for col in obj.users_collection:
            col.objects.link(wobj)
        wobj["cf_water_of"] = obj.name
        wobj.parent = obj
    else:
        old = wobj.data
        wobj.data = wme
        if old.users == 0:
            bpy.data.meshes.remove(old)
    named = sorted({n for n, *_ in lakes if n})
    obj["cf_water_info"] = "%d %s%s" % (len(lakes), "lake or pond" if len(lakes) == 1 else "lakes and ponds",
                                        (": " + ", ".join(named[:3]) + ("…" if len(named) > 3 else "")) if named else "")
    return lakes


def remove_water(obj):
    """Puts the terrain back as downloaded and deletes its water."""
    me = obj.data
    if DEM_ATTR in me.attributes:
        n = len(me.vertices)
        a = np.empty(n, dtype=np.float32)
        me.attributes[DEM_ATTR].data.foreach_get("value", a)
        me.attributes["cf_ground"].data.foreach_set("value", a)
        co = np.empty((n, 3), dtype=np.float32)
        me.vertices.foreach_get("co", co.ravel())
        co[:, 2] = a
        me.vertices.foreach_set("co", co.ravel())
        me.attributes.remove(me.attributes[DEM_ATTR])
    if WATER_ATTR in me.attributes:
        me.attributes.remove(me.attributes[WATER_ATTR])
    me.update()
    w = water_object(obj)
    if w is not None:
        wme = w.data
        bpy.data.objects.remove(w)
        if wme.users == 0:
            bpy.data.meshes.remove(wme)
    if "cf_water_info" in obj:
        del obj["cf_water_info"]
    obj["cf_version"] = int(obj.get("cf_version", 0)) + 1
    terrain.clear_cache()


def add_water(context, obj):
    """Downloads the lake outlines for a terrain and applies them. Returns the lakes made."""
    outlines = fetch_outlines(context.scene, obj)
    return apply_water(context, obj, outlines)


def _rebuild_roads(context, obj):
    """Roads on this terrain follow the new ground (and carve it again)."""
    try:
        bpy.ops.civil.refresh_all()
    except RuntimeError:
        pass


class CIVIL_OT_TerrainWater(bpy.types.Operator):
    bl_idname = "civil.terrain_water"
    bl_label = "Add Lakes and Ponds"
    bl_description = ("Download lake, pond and reservoir outlines (USGS National Hydrography Dataset) for "
                      "this terrain: dig their beds, fill them with water, and bridge roads over them")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return terrain.is_terrain(context.active_object)

    def execute(self, context):
        obj = context.active_object
        if not getattr(bpy.app, "online_access", True):
            self.report({'ERROR'}, "Online access is off: Preferences > System > Network > Allow Online Access")
            return {'CANCELLED'}
        terrain._busy(context, True)
        try:
            lakes = add_water(context, obj)
        except Exception as e:
            self.report({'ERROR'}, "Lakes failed: %s" % e)
            return {'CANCELLED'}
        finally:
            terrain._busy(context, False)
        _rebuild_roads(context, obj)
        if not lakes:
            self.report({'INFO'}, "No mapped lakes or ponds on this terrain")
        else:
            self.report({'INFO'}, "Water added: %s" % obj.get("cf_water_info", ""))
        return {'FINISHED'}


class CIVIL_OT_TerrainWaterRemove(bpy.types.Operator):
    bl_idname = "civil.terrain_water_remove"
    bl_label = "Remove Water"
    bl_description = "Fill the lakebeds back in and delete the water surfaces"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return terrain.is_terrain(context.active_object)

    def execute(self, context):
        remove_water(context.active_object)
        _rebuild_roads(context, context.active_object)
        return {'FINISHED'}


classes = (CIVIL_OT_TerrainWater, CIVIL_OT_TerrainWaterRemove)
