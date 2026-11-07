"""
Headless smoke test for CivilFlow.

Run from the repo root:
    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
        --python tools/smoke_test.py -- [--render OUT_DIR]

Loads civilflow/ straight from this checkout (no install needed), builds every preset on its
default curve, checks the meshes, traffic and roads on a synthetic terrain (no network needed),
and optionally renders a Workbench preview of each.
Exits non-zero if any check fails.
"""
import sys
import os
import math
import importlib.util

import bpy
import mathutils
from mathutils import Vector

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
render_dir = None
if "--render" in argv:
    render_dir = os.path.abspath(argv[argv.index("--render") + 1])
    os.makedirs(render_dir, exist_ok=True)


def load_addon():
    spec = importlib.util.spec_from_file_location(
        "civilflow", os.path.join(REPO, "civilflow", "__init__.py"),
        submodule_search_locations=[os.path.join(REPO, "civilflow")])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["civilflow"] = mod
    spec.loader.exec_module(mod)
    mod.register()
    return mod


def clear_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for c in list(bpy.data.curves):
        bpy.data.curves.remove(c)
    for m in list(bpy.data.meshes):
        bpy.data.meshes.remove(m)


def world_bbox(obj):
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


def render_preview(obj, path):
    scene = bpy.context.scene
    try:
        scene.render.engine = 'BLENDER_WORKBENCH'
    except TypeError:
        pass
    scene.display.shading.light = 'STUDIO'
    scene.display.shading.color_type = 'MATERIAL'
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = 'PNG'

    lo, hi = world_bbox(obj)
    center = (lo + hi) * 0.5
    radius = max((hi - lo).length * 0.5, 5.0)

    cam_data = bpy.data.cameras.new("SmokeCam")
    cam_data.lens = 35
    cam_data.clip_end = 5000
    cam = bpy.data.objects.new("SmokeCam", cam_data)
    scene.collection.objects.link(cam)
    direction = Vector((0.75, -1.0, 0.65)).normalized()
    cam.location = center + direction * radius * 1.9
    cam.rotation_euler = (center - cam.location).to_track_quat('-Z', 'Y').to_euler()
    scene.camera = cam

    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam, do_unlink=True)


def write_geotiff(path, arr, x0, y0, px, epsg, predictor):
    """Minimal single-strip, Deflate-compressed GeoTIFF writer (float32 or int16), north-up rows."""
    import struct
    import zlib
    import numpy as np
    h, w = arr.shape
    is_float = arr.dtype == np.float32
    bps = 4 if is_float else 2
    rows = []
    for r in arr:
        r = np.ascontiguousarray(r.astype("<f4" if is_float else "<i2"))
        if predictor == 3:  # floating-point predictor: byte planes, most significant first, differenced
            b = r.view(np.uint8).reshape(w, bps)[:, ::-1].T.ravel().copy()
            b[1:] = (b[1:].astype(np.int16) - b[:-1].astype(np.int16)).astype(np.uint8)
            rows.append(b.tobytes())
        elif predictor == 2:
            d = r.copy()
            d[1:] = (r[1:].astype(np.int32) - r[:-1].astype(np.int32)).astype("<i2")
            rows.append(d.tobytes())
        else:
            rows.append(r.tobytes())
    data = zlib.compress(b"".join(rows))
    extra = {33550: struct.pack("<3d", px, px, 0.0),
             33922: struct.pack("<6d", 0, 0, 0, x0, y0, 0),
             34735: struct.pack("<16H", 1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, epsg)}
    tags = [(256, 3, 1, w), (257, 3, 1, h), (258, 3, 1, bps * 8), (259, 3, 1, 8), (262, 3, 1, 1),
            (273, 4, 1, None), (277, 3, 1, 1), (278, 3, 1, h), (279, 4, 1, len(data)),
            (317, 3, 1, predictor), (339, 3, 1, 3 if is_float else 2),
            (33550, 12, 3, "x"), (33922, 12, 6, "x"), (34735, 3, 16, "x")]
    out = bytearray(b"II*\x00" + struct.pack("<I", 0))
    data_off = len(out)
    out += data
    if len(out) % 2:
        out += b"\x00"
    extra_off = {}
    for tag, blob in extra.items():
        extra_off[tag] = len(out)
        out += blob
    ifd_off = len(out)
    out[4:8] = struct.pack("<I", ifd_off)
    out += struct.pack("<H", len(tags))
    for tag, typ, cnt, val in tags:
        if tag == 273:
            val = data_off
        if val == "x":
            out += struct.pack("<HHII", tag, typ, cnt, extra_off[tag])
        elif typ == 3:
            out += struct.pack("<HHIHH", tag, typ, cnt, val, 0)
        else:
            out += struct.pack("<HHII", tag, typ, cnt, val)
    out += struct.pack("<I", 0)
    with open(path, "wb") as fh:
        fh.write(out)


def terrain_tests(mod, failures):
    """Synthetic land with a valley and a hill: bridges, tunnels, banks, cuts and portals."""
    import tempfile
    import numpy as np
    from mathutils.bvhtree import BVHTree
    T = sys.modules["civilflow.terrain"]
    G = sys.modules["civilflow.geo"]

    def report(name, ok, msg):
        print(f"[{'OK ' if ok else 'BAD'}] {name:13s} {msg}")
        if not ok:
            failures.append(name)

    # Map maths
    e, n = G.latlon_to_utm(34.866, -118.887, 11)
    lat, lon = G.utm_to_latlon(e, n, 11)
    parsed = [G.parse_latlon(t) for t in ("34.866, -118.887", "34.866 -118.887", "34°51'57.6\"N 118°53'13.2\"W")]
    ok = (abs(float(lat) - 34.866) < 1e-8 and abs(float(lon) + 118.887) < 1e-8
          and all(p is not None and abs(p[0] - 34.866) < 1e-4 and abs(p[1] + 118.887) < 1e-4 for p in parsed))
    report("GEO", ok, f"UTM round trip {abs(float(lat) - 34.866):.1e} deg; lat/long text formats parsed={ok}")

    clear_scene()
    for key in ("cf_geo",):
        if key in bpy.context.scene:
            del bpy.context.scene[key]
    x = np.arange(-500.0, 500.0 + 1e-6, 4.0)
    y = np.arange(-300.0, 300.0 + 1e-6, 4.0)
    X, Y = np.meshgrid(x, y)
    def land(X, Y):
        return 100.0 + 45.0 * np.exp(-((X - 200.0) / 70.0) ** 2) - 32.0 * np.exp(-((X + 150.0) / 60.0) ** 2) + 0.02 * Y
    ter = T.build_terrain(bpy.context, "Terrain_Test", 400000.0 + x, 3800000.0 + y, land(X, Y), 11, True)
    g = T.ground_for(ter)
    z0 = bpy.context.scene["cf_geo"]["z0"]
    probe = g.world_heights([-150.0, 200.0, 13.0], [0.0, 10.0, -77.0])
    want = land(np.array([-150.0, 200.0, 13.0]), np.array([0.0, 10.0, -77.0])) - z0
    err = float(np.max(np.abs(probe - want)))
    report("TERRAIN BUILD", len(ter.data.vertices) == len(x) * len(y) and err < 0.6,
           f"{len(ter.data.vertices)} verts, ground query error {err:.2f} m (4 m grid)")

    # A highway built from the 3D cursor picks the terrain up and follows it.
    bpy.context.scene.cursor.location = (-480.0, 0.0, 0.0)
    for o in bpy.context.selected_objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = None
    bpy.ops.civil.create_road_preset(preset_type='HIGHWAY', length=960.0)
    road = bpy.context.active_object
    p = road.civil_props
    rec = T._CARVE.get(road.name)
    smp = mod.road_local_samples(road)
    zs = np.array([s[0].z for s in smp])
    xy = np.array([(s[0].x, s[0].y) for s in smp])
    grade = float(np.max(np.abs(np.diff(zs)) / np.linalg.norm(np.diff(xy, axis=0), axis=1))) * 100.0
    kinds = {k for k, _a, _b in T._runs(rec["mode"])} if rec else set()
    ok = (p.terrain_object == ter and p.follow_terrain and grade <= p.max_grade + 0.05
          and T.BRIDGE in kinds and T.TUNNEL in kinds)
    report("TERRAIN ROAD", ok, f"on terrain={p.terrain_object == ter}, max grade {grade:.2f}% (limit {p.max_grade:g}), "
           f"stretches {sorted(['ROAD', 'BRIDGE', 'TUNNEL'][k] for k in kinds)}")

    # Piers stand on the valley floor.
    me = road.data
    vz = np.array([(v.co.x, v.co.z) for v in me.vertices])
    near = vz[np.abs(vz[:, 0] + 150.0) < 25.0]
    floor = float(np.min(g.world_heights(np.linspace(-175, -125, 11), np.zeros(11))))
    pier_low = float(near[:, 1].min()) if len(near) else 1e9
    report("TERRAIN PIERS", abs(pier_low - floor) < 1.5,
           f"lowest pier point {pier_low:.1f} m, valley floor {floor:.1f} m")

    # Nothing of the terrain is left over the road outside the tunnel, portals included.
    dg = bpy.context.evaluated_depsgraph_get()
    bvh = BVHTree.FromObject(ter, dg)
    hits = 0
    half = (p.lanes_left + p.lanes_right) * p.lane_width * 0.5 + p.median_width * 0.5 + p.shoulder_width
    for i, (pos, _t, nr, r) in enumerate(smp):
        if rec["mode"][i] == T.TUNNEL:
            continue
        for off in np.linspace(-half, half, 7):
            q = road.matrix_world @ (pos + r * off + nr * 0.1)
            if bvh.ray_cast(q, Vector((0.0, 0.0, 1.0)), 80.0)[0] is not None:
                hits += 1
    holes = int(np.count_nonzero([d.value for d in ter.data.attributes["cf_hole"].data])) \
        if "cf_hole" in ter.data.attributes else 0
    report("TERRAIN CLEAR", hits == 0 and holes > 0 and ter.modifiers.get(T.HOLE_MOD) is not None,
           f"road points under terrain: {hits}; portal openings: {holes} faces")

    # Your own structures: an overpass on a flat stretch, an earth bank instead of the valley bridge.
    bpy.context.view_layer.objects.active = road
    bpy.context.scene.cursor.location = (-400.0, 0.0, 0.0)
    bpy.ops.civil.structure_add(kind='BRIDGE')
    st = p.structures[0]
    bpy.context.scene.cursor.location = (-150.0, 0.0, 0.0)
    bpy.ops.civil.structure_add(kind='BANK')
    rec = T._CARVE[road.name]
    smp_o = mod.road_local_samples(road)
    xs_o = np.array([sp[0].x for sp in smp_o])
    zs_o = np.array([sp[0].z for sp in smp_o])
    i_br = int(np.argmin(np.abs(xs_o + 400.0)))
    i_bank = int(np.argmin(np.abs(xs_o + 150.0)))
    g_br = float(g.world_heights([xs_o[i_br]], [0.0])[0])
    clear = zs_o[i_br] - p.deck_thickness - g_br
    vz = np.array([(v.co.x, v.co.z) for v in road.data.vertices])
    under = vz[(np.abs(vz[:, 0] + 400.0) < st.clear_span * 0.5) & (vz[:, 1] < zs_o[i_br] - p.deck_thickness - 1.0)]
    valley_bridge = any(rec["mode"][i] == T.BRIDGE for i in range(len(xs_o)) if abs(xs_o[i] + 150.0) < 25.0)
    ok = (rec["mode"][i_br] == T.BRIDGE and clear >= st.clearance - 0.5 and len(under) == 0
          and rec["mode"][i_bank] == T.ROAD and not valley_bridge)
    report("TERRAIN YOURS", ok, f"overpass placed: bridge={rec['mode'][i_br] == T.BRIDGE}, clearance {clear:.1f} m "
           f"(asked {st.clearance:g}), pillar parts in its clear span {len(under)}; "
           f"earth bank replaced valley bridge={not valley_bridge}")
    while len(p.structures):
        bpy.ops.civil.structure_remove()

    # A road continued from it starts exactly on its end and keeps following the land.
    bpy.ops.civil.road_continue(preset_type='HIGHWAY', length=150.0, bend=-20.0)
    road2 = bpy.context.active_object
    s2 = mod.road_local_samples(road2)
    gap = ((road2.matrix_world @ s2[0][0]) - (road.matrix_world @ smp[-1][0])).length
    report("TERRAIN JOIN", road2.civil_props.terrain_object == ter and gap < 0.05,
           f"continued road on terrain={road2.civil_props.terrain_object == ter}, join gap {gap:.3f} m")

    # Taking the roads off the terrain puts the land back as it was.
    road.civil_props.terrain_object = None
    road2.civil_props.terrain_object = None
    co = np.empty(len(ter.data.vertices) * 3, dtype=np.float32)
    ter.data.vertices.foreach_get("co", co)
    orig = np.empty(len(ter.data.vertices), dtype=np.float32)
    ter.data.attributes["cf_ground"].data.foreach_get("value", orig)
    left = int(np.count_nonzero([d.value for d in ter.data.attributes["cf_hole"].data]))
    diff = float(np.max(np.abs(co[2::3] - orig)))
    report("TERRAIN RESET", diff < 1e-4 and left == 0, f"terrain back to original (max change {diff:.5f} m, openings {left})")

    # An elevated road whose ends ramp down to meet the ground.
    curve = mod.create_guide_curve(bpy.context, "RampTest", [(-400.0, -280.0, 0.0), (-400.0, 280.0, 0.0)])
    for o in bpy.context.selected_objects:
        o.select_set(False)
    curve.select_set(True)
    bpy.context.view_layer.objects.active = curve
    bpy.ops.civil.create_road_preset(preset_type='OVERPASS')
    ramp = bpy.context.active_object
    rp = ramp.civil_props
    rp.ramp_start = True
    rp.ramp_end = True
    smp_r = mod.road_local_samples(ramp)
    P = np.array([tuple(ramp.matrix_world @ sp[0]) for sp in smp_r])
    above = P[:, 2] - g.world_heights(P[:, 0], P[:, 1])
    grade_r = float(np.max(np.abs(np.diff(P[:, 2])) / np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))) * 100.0
    mode_r = T._CARVE[ramp.name]["mode"]
    ok = (abs(above[0] - 0.3) < 0.3 and abs(above[-1] - 0.3) < 0.3 and above.max() > rp.terrain_offset - 0.5
          and grade_r <= rp.max_grade + 0.05 and mode_r[0] == T.ROAD and mode_r[-1] == T.ROAD
          and T.BRIDGE in set(mode_r.tolist()))
    report("TERRAIN RAMPS", ok, f"elevated road: start {above[0]:.2f} m and end {above[-1]:.2f} m above ground, "
           f"peak {above.max():.1f} m (asked {rp.terrain_offset:g}), max grade {grade_r:.2f}%")
    bpy.data.objects.remove(ramp)

    # A lake: bed dug under the outline, water on top, and a street across it bridges the water.
    W = sys.modules["civilflow.water"]
    lx = np.arange(2700.0, 3300.0 + 1e-6, 4.0)
    ly = np.arange(-200.0, 200.0 + 1e-6, 4.0)
    LX, LY = np.meshgrid(lx, ly)
    inside = np.hypot(LX - 3000.0, LY) < 90.0
    dem = np.where(inside, 148.0, 150.0 + 0.004 * np.hypot(LX - 3000.0, LY))  # hydro-flattened lake
    lake_t = T.build_terrain(bpy.context, "Terrain_Lake", 400000.0 + lx, 3800000.0 + ly, dem, 11, True)
    a = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    shore = np.stack([3000.0 + 92.0 * np.cos(a), 92.0 * np.sin(a)], axis=1)
    isle = np.stack([3030.0 + 12.0 * np.cos(a), 12.0 * np.sin(a)], axis=1)
    lakes = W.apply_water(bpy.context, lake_t, [("Test Lake", "lake or pond", [shore, isle])])
    gl = T.ground_for(lake_t)
    level = 148.0 - z0
    wl = gl.water_levels([3000.0, 3030.0, 3200.0], [0.0, 0.0, 0.0])
    bed = gl.world_heights([2960.0], [0.0])[0]
    wobj = W.water_object(lake_t)
    ok = (len(lakes) == 1 and abs(lakes[0][3] - level) < 1e-3 and abs(wl[0] - level) < 1e-3
          and np.isnan(wl[1]) and np.isnan(wl[2]) and abs(bed - (level - W.MAX_DEPTH)) < 0.01
          and wobj is not None and len(wobj.data.polygons) > 50
          and all(abs(v.co.z - level - 0.02) < 1e-3 for v in wobj.data.vertices))
    report("WATER", ok, f"level {wl[0]:.2f} (DEM {level:.2f}), island dry={np.isnan(wl[1])}, bed {level - bed:.1f} m deep, "
           f"water mesh {len(wobj.data.polygons) if wobj else 0} faces")

    bpy.context.scene.cursor.location = (2760.0, 0.0, 0.0)
    for o in bpy.context.selected_objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = None
    bpy.ops.civil.create_road_preset(preset_type='STREET', length=480.0)
    st = bpy.context.active_object
    smp_l = mod.road_local_samples(st)
    PL = np.array([tuple(st.matrix_world @ sp[0]) for sp in smp_l])
    wet = ~np.isnan(gl.water_levels(PL[:, 0], PL[:, 1]))
    mode_l = T._CARVE[st.name]["mode"]
    deck_under = PL[:, 2] - st.civil_props.deck_thickness
    ok = (st.civil_props.terrain_object == lake_t and wet.any() and bool(np.all(mode_l[wet] == T.BRIDGE))
          and float(deck_under[wet].min() - level) >= T.WATER_CLEARANCE - 0.05
          and mode_l[0] == T.ROAD and mode_l[-1] == T.ROAD)
    report("WATER BRIDGE", ok, f"{int(wet.sum())} samples over water, all bridge={bool(np.all(mode_l[wet] == T.BRIDGE))}, "
           f"deck {deck_under[wet].min() - level:.1f} m above the water, ends on the ground={mode_l[0] == T.ROAD}")

    W.remove_water(lake_t)
    bpy.ops.civil.refresh_all()
    co_l = np.empty(len(lake_t.data.vertices) * 3, dtype=np.float32)
    lake_t.data.vertices.foreach_get("co", co_l)
    mode_after = T._CARVE[st.name]["mode"]
    ok = (W.water_object(lake_t) is None and "cf_water" not in lake_t.data.attributes
          and abs(T.ground_for(lake_t).world_heights([2960.0], [0.0])[0] - level) < 1e-3
          and not np.any(mode_after == T.BRIDGE))
    report("WATER REMOVE", ok, f"lakebed filled back to {T.ground_for(lake_t).world_heights([2960.0], [0.0])[0]:.2f}, "
           f"water gone={W.water_object(lake_t) is None}, street back on the ground={not np.any(mode_after == T.BRIDGE)}")
    bpy.data.objects.remove(st)
    bpy.data.objects.remove(lake_t)

    # Photo tiles: each face takes the tile whose core holds it and stays inside that tile's image.
    res, tnx, tny, tx, ty, margin = T.plan_photo(3000.0, 2000.0, 0.3, 4.0)
    ex = np.arange(-1500.0, 1500.0 + 1e-6, 8.0) + 400000.0
    ny_ = np.arange(-1000.0, 1000.0 + 1e-6, 8.0) + 3800000.0
    photos = []
    for j in range(tny):
        for i in range(tnx):
            c0, r0 = ex[0] + i * tx, ny_[0] + j * ty
            box = (c0 - margin, r0 - margin, c0 + tx + margin, r0 + ty + margin)
            core = (c0 if i else -1e12, r0 if j else -1e12,
                    c0 + tx if i < tnx - 1 else 1e12, r0 + ty if j < tny - 1 else 1e12)
            img = bpy.data.images.new("tile_%d_%d" % (i, j), 8, 8)
            photos.append((img, lambda E_, N_, b=box: ((E_ - b[0]) / (b[2] - b[0]), (N_ - b[1]) / (b[3] - b[1])), core))
    EE, NN = np.meshgrid(ex, ny_)
    tob = T.build_terrain(bpy.context, "Terrain_Tiles", ex, ny_, np.zeros_like(EE) + 150.0, 11, True, photos)
    uvs = np.empty(len(tob.data.loops) * 2, dtype=np.float32)
    tob.data.uv_layers[0].data.foreach_get("uv", uvs)
    used = np.empty(len(tob.data.polygons), dtype=np.int32)
    tob.data.polygons.foreach_get("material_index", used)
    ok = (len(tob.data.materials) == tnx * tny and len(set(used.tolist())) == tnx * tny
          and uvs.min() >= 0.0 and uvs.max() <= 1.0)
    report("TERRAIN PHOTO", ok, f"3 x 2 km at {res:g} m: {tnx} x {tny} tiles, all used={len(set(used.tolist())) == tnx * tny}, "
           f"UVs {uvs.min():.3f}..{uvs.max():.3f}")
    bpy.data.objects.remove(tob)

    # GeoTIFF files: float32 + floating-point predictor, int16 + horizontal predictor, both Deflate.
    tmp = tempfile.mkdtemp()
    for name, dtype, pred in (("dem_f32.tif", np.float32, 3), ("dem_i16.tif", np.int16, 2)):
        cols, rws, px = 60, 50, 10.0
        ex = 401000.0 + px * (np.arange(cols) + 0.5)
        ny = 3800300.0 - px * (np.arange(rws) + 0.5)
        EE, NN = np.meshgrid(ex, ny)
        vals = 200.0 + 0.05 * (EE - 401000.0) - 0.03 * (NN - 3800000.0)
        path = os.path.join(tmp, name)
        write_geotiff(path, vals.astype(dtype), 401000.0, 3800300.0, px, 32611, pred)
        obj, _ = T.import_terrain_file(bpy.context, path, "", spacing=10.0, crop_m=0.0)
        gx, gy = 401300.0 - 400000.0, 3800050.0 - 3800000.0
        h = float(T.ground_for(obj).world_heights([gx], [gy])[0]) + z0
        want = 200.0 + 0.05 * 300.0 - 0.03 * 50.0
        tol = 0.6 if dtype == np.int16 else 0.05
        report("TERRAIN FILE", abs(h - want) < tol,
               f"{name}: {len(obj.data.vertices)} verts, height at a test point {h:.2f} m (want {want:.2f})")


def junction_tests(mod, failures):
    """Intersections: finding them, trimming the roads, devices, signal cycle, following edits, terrain."""
    import numpy as np
    J = sys.modules["civilflow.junctions"]
    T = sys.modules["civilflow.terrain"]

    def report(name, ok, msg):
        print(f"[{'OK ' if ok else 'BAD'}] {name:13s} {msg}")
        if not ok:
            failures.append(name)

    def road(name, pts, preset='STREET', terrain=False):
        c = mod.create_guide_curve(bpy.context, name, pts)
        for o in bpy.context.selected_objects:
            o.select_set(False)
        c.select_set(True)
        bpy.context.view_layer.objects.active = c
        bpy.ops.civil.create_road_preset(preset_type=preset)
        r = bpy.context.active_object
        if not terrain:
            r.civil_props.follow_terrain = False
        return r, c

    def faces_near(ob, xy, rad, mats=None):
        M = ob.matrix_world
        n = 0
        for pg in ob.data.polygons:
            c = M @ pg.center
            if math.hypot(c.x - xy[0], c.y - xy[1]) < rad and (mats is None or pg.material_index in mats):
                n += 1
        return n

    clear_scene()
    for k in ("cf_geo",):
        if k in bpy.context.scene:
            del bpy.context.scene[k]
    a, ca = road("JA", [(-150.0, 0.0, 0.0), (0.0, 0.0, 0.0), (150.0, 0.0, 0.0)])
    b, cb = road("JB", [(0.0, -150.0, 0.0), (0.0, 0.0, 0.0), (0.0, 150.0, 0.0)])
    t, ct = road("JT", [(-90.0, 100.0, 0.0), (-90.0, 50.0, 0.0), (-90.0, 0.0, 0.0)])
    o, co_ = road("JO", [(60.0, -150.0, 0.0), (60.0, 0.0, 0.0), (60.0, 150.0, 0.0)], preset='OVERPASS')
    before = faces_near(a, (0.0, 0.0), 3.0)
    bpy.ops.civil.find_junctions()
    js = [ob for ob in bpy.context.scene.objects if J.is_junction(ob)]
    shapes = sorted(ob.civil_junction.shape for ob in js)
    cross = next((ob for ob in js if ob.civil_junction.shape == 'CROSS'), None)
    tee = next((ob for ob in js if ob.civil_junction.shape == 'T'), None)
    report("JUNCTION FIND", shapes == ['CROSS', 'T'] and tee is not None and tee.civil_junction.road_b == t,
           f"found {shapes} (the overpass crossing above is left alone); T ends road {tee.civil_junction.road_b.name if tee else '-'}")

    left = faces_near(a, (0.0, 0.0), 6.0) + faces_near(b, (0.0, 0.0), 6.0)
    pad = faces_near(cross, (0.0, 0.0), 6.0, mats={0})
    walks = faces_near(cross, (0.0, 0.0), 30.0, mats={5})
    lamps = sum(1 for pg in cross.data.polygons if pg.material_index >= J.M_SIGNAL0)
    report("JUNCTION TRIM", before > 0 and left == 0 and pad > 0 and walks >= 16 and lamps >= 24,
           f"road faces at the centre {before} -> {left}; pad faces {pad}, crosswalk/stop bars {walks}, lamp faces {lamps}")

    # Signals: each road's green, then yellow, then red; never both green.
    fps = bpy.context.scene.render.fps
    p = cross.civil_junction
    def lit(frame):
        bpy.context.scene.frame_set(frame)
        dg = bpy.context.evaluated_depsgraph_get()
        out = []
        for m in cross.data.materials[J.M_SIGNAL0:J.M_SIGNAL0 + 6]:
            me = m.evaluated_get(dg)
            bsdf = next(n for n in me.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
            out.append(bsdf.inputs["Emission Strength"].default_value > 1.0)
        return out  # R1 Y1 G1 R2 Y2 G2
    cyc = 2 * (p.green_time + p.yellow_time + p.all_red_time)
    states = [lit(int(fps * s) + 1) for s in np.arange(0.5, cyc, 1.0)]
    one_each = all(sum(st[0:3]) == 1 and sum(st[3:6]) == 1 for st in states)
    never_both = not any(st[2] and st[5] for st in states)
    greens = (sum(st[2] for st in states), sum(st[5] for st in states))
    bpy.context.scene.frame_set(1)
    report("SIGNAL CYCLE", one_each and never_both and min(greens) >= p.green_time - 1,
           f"one lamp per head always={one_each}, never both green={never_both}, seconds of green {greens}")

    p.control = 'ALL_STOP'
    octs = sum(1 for pg in cross.data.polygons if pg.material_index == J.M_RED and len(pg.vertices) == 8)
    p.control = 'ROUNDABOUT'
    grass = sum(1 for pg in cross.data.polygons if pg.material_index == J.M_GRASS)
    tris = sum(1 for pg in cross.data.polygons if pg.material_index == J.M_RED and len(pg.vertices) == 3)
    still = faces_near(a, (0.0, 0.0), 14.0) + faces_near(b, (0.0, 0.0), 14.0)
    report("JUNCTION TYPES", octs == 4 and grass >= 1 and tris == 4 and still == 0,
           f"all-way stop signs {octs}, roundabout island {grass > 0}, yield signs {tris}, road faces inside the ring {still}")
    p.control = 'SIGNALS'

    # Editing a road: the intersection follows it.
    cb.location.x += 6.0
    bpy.context.view_layer.update()
    at = cross["cf_at"]
    moved = faces_near(b, (6.0, 0.0), 5.0)
    report("JUNCTION EDIT", abs(at[0] - 6.0) < 0.5 and abs(at[1]) < 0.5 and moved == 0,
           f"road moved 6 m: intersection now at ({at[0]:.1f}, {at[1]:.1f}), road faces at its centre {moved}")

    bpy.context.view_layer.objects.active = cross
    bpy.ops.civil.remove_junction()
    back = faces_near(a, (6.0, 0.0), 3.0)
    report("JUNCTION REMOVE", back > 0, f"road surface back at the crossing: {back} faces")

    # On terrain: the ground comes down under the pad's corners.
    clear_scene()
    for k in ("cf_geo",):
        if k in bpy.context.scene:
            del bpy.context.scene[k]
    x = np.arange(-200.0, 200.0 + 1e-6, 2.0)
    y = np.arange(-200.0, 200.0 + 1e-6, 2.0)
    X, Y = np.meshgrid(x, y)
    ter = T.build_terrain(bpy.context, "Terrain_J", 600000.0 + x, 3600000.0 + y, 50.0 + 0.04 * X + 0.03 * Y, 11, True)
    bpy.context.scene.cursor.location = (-150.0, 0.0, 0.0)
    a2, _ = road("TA", [(-150.0, 0.0, 0.0), (0.0, 0.0, 0.0), (150.0, 0.0, 0.0)], terrain=True)
    b2, _ = road("TB", [(0.0, -150.0, 0.0), (0.0, 0.0, 0.0), (0.0, 150.0, 0.0)], terrain=True)
    a2.civil_props.terrain_object = ter
    b2.civil_props.terrain_object = ter
    bpy.ops.civil.find_junctions()
    j2 = next((ob for ob in bpy.context.scene.objects if J.is_junction(ob)), None)
    ok = j2 is not None and len(j2.get("cf_cut", [])) > 0
    if ok:
        bpy.ops.civil.refresh_all()
        # Ray down through the pad's asphalt top at many points: the terrain must be below it.
        from mathutils.bvhtree import BVHTree
        dg = bpy.context.evaluated_depsgraph_get()
        tb = BVHTree.FromObject(ter.evaluated_get(dg), dg)
        pad_tris = [pg for pg in j2.data.polygons if pg.material_index == 0 and pg.normal.z > 0.9]
        worst = -9.0
        for pg in pad_tris:
            c = pg.center
            hit = tb.ray_cast(c + mathutils.Vector((0.0, 0.0, 30.0)), mathutils.Vector((0.0, 0.0, -1.0)))
            if hit[0] is not None:
                worst = max(worst, hit[0].z - c.z)
        ok = len(pad_tris) > 0 and worst < 0.0
        msg = f"{len(pad_tris)} pad faces; terrain at most {worst:.2f} m relative to the asphalt on top (must be below)"
    else:
        msg = "no intersection found on terrain"
    report("JUNCTION GROUND", ok, msg)


def traffic_sim_tests(mod, failures):
    """Traffic through intersections: no overlaps, nobody on red, full stops at stop signs, turns."""
    import numpy as np
    from bpy_extras import anim_utils
    J = sys.modules["civilflow.junctions"]
    S = sys.modules["civilflow.sim"]

    def report(name, ok, msg):
        print(f"[{'OK ' if ok else 'BAD'}] {name:13s} {msg}")
        if not ok:
            failures.append(name)

    def road(name, pts):
        c = mod.create_guide_curve(bpy.context, name, pts)
        for o in bpy.context.selected_objects:
            o.select_set(False)
        c.select_set(True)
        bpy.context.view_layer.objects.active = c
        bpy.ops.civil.create_road_preset(preset_type='STREET')
        r = bpy.context.active_object
        r.civil_props.follow_terrain = False
        return r

    def tracks(col):
        vs = [o for o in col.objects if o.get("cf_kind")]
        out = []
        for o in vs:
            ad = o.animation_data
            cb = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
            xyz = {}
            for fc in cb.fcurves:
                if fc.data_path == "location":
                    co = np.empty(len(fc.keyframe_points) * 2)
                    fc.keyframe_points.foreach_get("co", co)
                    xyz[fc.array_index] = co[1::2]
            out.append(np.stack([xyz[0], xyz[1], xyz[2]], axis=1))
        return vs, np.array(out)

    for control in ('SIGNALS', 'ALL_STOP', 'ROUNDABOUT'):
        clear_scene()
        bpy.context.scene.frame_start, bpy.context.scene.frame_end = 1, 250
        a = road("SA", [(-200.0, 0.0, 0.0), (0.0, 0.0, 0.0), (200.0, 0.0, 0.0)])
        b = road("SB", [(0.0, -200.0, 0.0), (0.0, 0.0, 0.0), (0.0, 200.0, 0.0)])
        bpy.ops.civil.find_junctions()
        j = next(o for o in bpy.context.scene.objects if J.is_junction(o))
        j.civil_junction.control = control
        a.civil_props.traffic_duration = 45.0
        bpy.context.view_layer.objects.active = a
        bpy.ops.civil.traffic_generate()
        col = a.civil_props.traffic_collection
        vs, T = tracks(col)
        fps = bpy.context.scene.render.fps
        vis = T[:, :, 2] > -100.0
        worst = 99.0
        for k in range(0, T.shape[1], 2):
            idx = np.nonzero(vis[:, k])[0]
            if len(idx) < 2:
                continue
            P = T[idx, k, :2]
            D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
            np.fill_diagonal(D, 99.0)
            worst = min(worst, float(D.min()))
        spd = np.linalg.norm(np.diff(T[:, :, :2], axis=1), axis=2) * fps
        ok_v = vis[:, 1:] & vis[:, :-1] & (spd < 60.0)
        moving = float(np.mean(spd[ok_v])) if ok_v.any() else 0.0
        # Each pass through the intersection: where the vehicle came from and went.
        r_zone = 9.0 if control != 'ROUNDABOUT' else 20.0
        zone = (np.hypot(T[:, :, 0], T[:, :, 1]) < r_zone) & vis
        p = j.civil_junction
        bad, entries, turns, rolled = 0, 0, 0, 0
        for vi in range(len(vs)):
            z = zone[vi]
            k = 1
            while k < len(z):
                if z[k] and not z[k - 1]:
                    k_in = k
                    while k < len(z) and z[k]:
                        k += 1
                    if k >= len(z):
                        break  # still inside at the end
                    entries += 1
                    a_in, a_out = T[vi, k_in - 1, :2], T[vi, k, :2]
                    if (abs(a_in[0]) > abs(a_in[1])) != (abs(a_out[0]) > abs(a_out[1])):
                        turns += 1
                    group = 0 if abs(a_in[0]) > abs(a_in[1]) else 1
                    if control == 'SIGNALS':
                        light = S._signal({"G": p.green_time, "Y": p.yellow_time, "AR": p.all_red_time},
                                          group, (bpy.context.scene.frame_start + k_in) / fps)
                        bad += light == 'R'
                    if control == 'ALL_STOP' and k_in > 6 * fps:
                        window = spd[vi, k_in - 6 * fps:k_in][ok_v[vi, k_in - 6 * fps:k_in]]
                        rolled += not (len(window) and window.min() < 0.5)
                k += 1
        ok = (len(vs) > 10 and worst > 2.5 and moving > 1.5 and entries > 5 and turns > 0
              and bad == 0 and rolled == 0)
        report("TRAFFIC " + control[:8], ok,
               f"{len(vs)} vehicles over {T.shape[1] / fps:.0f} s: closest {worst:.1f} m, mean speed {moving:.1f} m/s, "
               f"{entries} through the middle ({turns} turning)"
               + (f", {bad} on red" if control == 'SIGNALS' else "")
               + (f", {rolled} without stopping" if control == 'ALL_STOP' else ""))


def lines_tests(mod, failures):
    """Streetlight styles and sides, one-sided sign arms, power lines and path lights."""
    import numpy as np
    U = sys.modules["civilflow.utilities"]
    T = sys.modules["civilflow.terrain"]

    def report(name, ok, msg):
        print(f"[{'OK ' if ok else 'BAD'}] {name:13s} {msg}")
        if not ok:
            failures.append(name)

    def centers(ob, mat):
        me = ob.data
        return np.array([tuple(ob.matrix_world @ p.center) for p in me.polygons if p.material_index == mat])

    # Each streetlight style on a street, lights on the right side only.
    clear_scene()
    for k in ("cf_geo",):
        if k in bpy.context.scene:
            del bpy.context.scene[k]
    bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
    bpy.ops.civil.create_road_preset(preset_type='STREET', length=120.0)
    road = bpy.context.active_object
    p = road.civil_props
    p.follow_terrain = False
    smp = mod.road_local_samples(road)
    mid = smp[len(smp) // 2]
    right_w = road.matrix_world.to_3x3() @ mid[3]
    origin = road.matrix_world @ mid[0]
    msgs, ok = [], True
    for style, *_r in U.LIGHT_STYLES:
        p.streetlight_style = style
        for side in ('RIGHT', 'LEFT'):
            p.streetlight_placement = side
            glow = 13 if style in ('ACORN', 'TWIN', 'LANTERN') else 7
            c = centers(road, glow)
            if not len(c):
                ok = False
                msgs.append(f"{style}: no lamps")
                continue
            sd = (c - np.array(origin)) @ np.array(right_w)
            want = 1.0 if side == 'RIGHT' else -1.0
            if not np.all(sd * want > 2.0):
                ok = False
                msgs.append(f"{style} {side}: lamps on the wrong side")
        h_want = U.LIGHT_DEFAULTS[style][0]
        if abs(p.streetlight_height - h_want) > 1e-6:
            ok = False
            msgs.append(f"{style}: height {p.streetlight_height} not {h_want}")
    report("LIGHT STYLES", ok, "; ".join(msgs) or f"{len(U.LIGHT_STYLES)} styles, each on the right only and the left only")

    # One-sided overhead sign arm.
    clear_scene()
    bpy.ops.civil.create_road_preset(preset_type='HIGHWAY', length=300.0)
    hw = bpy.context.active_object
    hp = hw.civil_props
    hp.follow_terrain = False
    hp.add_gantry_signs = True
    hp.gantry_style = 'CANTILEVER'
    smp = mod.road_local_samples(hw)
    mid = smp[len(smp) // 2]
    right_w = hw.matrix_world.to_3x3() @ mid[3]
    origin = np.array(hw.matrix_world @ mid[0])
    res = {}
    for side in ('BOTH', 'RIGHT', 'LEFT'):
        hp.gantry_side = side
        c = centers(hw, 8)
        sd = (c - origin) @ np.array(right_w) if len(c) else np.zeros(0)
        res[side] = (int((sd > 0).sum()), int((sd < 0).sum()))
    ok = (res['BOTH'][0] > 0 and res['BOTH'][1] > 0 and res['RIGHT'][0] > 0 and res['RIGHT'][1] == 0
          and res['LEFT'][1] > 0 and res['LEFT'][0] == 0)
    report("SIGN ARM SIDE", ok, f"sign faces right/left: both {res['BOTH']}, right only {res['RIGHT']}, left only {res['LEFT']}")

    # Power lines: every style along a curve over terrain, poles standing on the ground.
    clear_scene()
    for k in ("cf_geo",):
        if k in bpy.context.scene:
            del bpy.context.scene[k]
    x = np.arange(-400.0, 400.0 + 1e-6, 4.0)
    y = np.arange(-200.0, 200.0 + 1e-6, 4.0)
    X, Y = np.meshgrid(x, y)
    ter = T.build_terrain(bpy.context, "Terrain_Lines", 500000.0 + x, 3700000.0 + y,
                          100.0 + 20.0 * np.sin(X / 90.0) + 0.03 * Y, 11, True)
    g = T.ground_for(ter)
    msgs, ok = [], True
    for k, (style, *_r) in enumerate(U.POWER_STYLES):
        yy = -150.0 + 75.0 * k
        curve = mod.create_guide_curve(bpy.context, style, [(-380.0, yy, 0.0), (0.0, yy + 20.0, 0.0), (380.0, yy, 0.0)])
        for o in bpy.context.selected_objects:
            o.select_set(False)
        curve.select_set(True)
        bpy.context.view_layer.objects.active = curve
        bpy.ops.civil.add_line(kind='POWER')
        line = bpy.context.active_object
        lp = line.civil_line
        lp.style = style
        n_poles = line.get("cf_poles", 0)
        co = np.array([tuple(v.co) for v in line.data.vertices])
        mats = {pg.material_index for pg in line.data.polygons}
        span_ok = abs(lp.span - U.POWER_DEFAULTS[style][0]) < 1e-6
        # Every pole's foot sits in the ground: the lowest points near each pole are below it.
        low = co[:, 2] - g.world_heights(co[:, 0], co[:, 1])
        wires = U.M_WIRE in mats
        cans = U.M_CAN in mats if style == 'WOOD' else True
        if not (n_poles >= 2 and span_ok and wires and cans and np.nanmin(low) < -0.3 and np.nanmax(low) > 0.95 * lp.pole_height):
            ok = False
            msgs.append(f"{style}: poles {n_poles}, wires {wires}, transformers {cans}, "
                        f"heights {np.nanmin(low):.1f}..{np.nanmax(low):.1f} m")
        else:
            msgs.append(f"{style} {n_poles}")
    report("POWER LINES", ok, "poles per style: " + ", ".join(msgs))

    # Moving the curve moves the line.
    before = np.array([tuple(v.co) for v in line.data.vertices]).mean(axis=0)
    curve.location.y += 10.0
    bpy.context.view_layer.update()
    after = np.array([tuple(v.co) for v in line.data.vertices]).mean(axis=0)
    report("LINE FOLLOWS", abs((after - before)[1] - 10.0) < 0.5, f"curve moved 10 m, line moved {(after - before)[1]:.2f} m")

    # Path lights on one side of a path.
    curve = mod.create_guide_curve(bpy.context, "Park", [(-60.0, 0.0, 0.0), (0.0, 30.0, 0.0), (60.0, 0.0, 0.0)])
    pl = U.add_line(bpy.context, 'LIGHTS', curve)
    pl.civil_line.light_side = 'LEFT'
    c = centers(pl, 13)
    pts = sys.modules["civilflow.utilities"].curve_points(curve)
    P = np.array([tuple(q) for q in pts])
    # Left of a path heading +x that bends up through y=30: the lamps sit north of the curve.
    nearest = np.array([P[np.argmin(np.hypot(P[:, 0] - q[0], P[:, 1] - q[1]))] for q in c])
    ok = len(c) > 0 and pl.get("cf_poles", 0) >= 5 and bool(np.all(c[:, 1] > nearest[:, 1] - 0.05))
    report("PATH LIGHTS", ok, f"{pl.get('cf_poles', 0)} acorn lights, all on the left={ok}")


def main():
    mod = load_addon()
    rna = bpy.ops.civil.create_road_preset.get_rna_type()
    presets = [i.identifier for i in rna.properties["preset_type"].enum_items]
    failures = []

    for preset in presets:
        clear_scene()
        res = bpy.ops.civil.create_road_preset(preset_type=preset)
        obj = bpy.context.active_object
        me = obj.data if obj else None
        nv = len(me.vertices) if me else 0
        nf = len(me.polygons) if me else 0
        ok = res == {'FINISHED'} and nf > 0
        lo, hi = world_bbox(obj) if obj else (Vector(), Vector())
        loose = sum(1 for v in me.vertices) - len({vi for p in me.polygons for vi in p.vertices}) if me else 0
        print(f"[{'OK ' if ok else 'BAD'}] {preset:13s} verts={nv:7d} faces={nf:7d} "
              f"loose_verts={loose} bbox=({lo.x:.1f},{lo.y:.1f},{lo.z:.1f})..({hi.x:.1f},{hi.y:.1f},{hi.z:.1f})")
        if not ok:
            failures.append(preset)
        if loose:
            failures.append(f"{preset}: {loose} loose verts")
        if render_dir and obj:
            render_preview(obj, os.path.join(render_dir, f"{preset.lower()}.png"))

    # Interchange ramp between two roads
    clear_scene()
    bpy.ops.civil.create_road_preset(preset_type='STREET')
    a = bpy.context.active_object
    a.civil_props.target_curve.location = (0, 0, 0)
    bpy.ops.civil.create_road_preset(preset_type='STREET')
    b = bpy.context.active_object
    b.civil_props.target_curve.location = (130, 40, 8)
    b.civil_props.target_curve.rotation_euler = (0, 0, math.radians(90))
    bpy.context.view_layer.update()
    bpy.ops.civil.refresh_all()
    for o in bpy.context.selected_objects:
        o.select_set(False)
    a.select_set(True)
    b.select_set(True)
    bpy.context.view_layer.objects.active = b
    res = bpy.ops.civil.create_interchange_ramp()
    ramp = bpy.context.active_object
    ok = res == {'FINISHED'} and ramp and ramp.name.startswith("Civil_Interchange_Ramp")
    print(f"[{'OK ' if ok else 'BAD'}] RAMP          faces={len(ramp.data.polygons) if ok else 0}")
    if ok:
        spline = ramp.civil_props.target_curve.data.splines[0]
        ends = [tuple(round(c, 1) for c in bp.co) for bp in spline.bezier_points]
        print(f"      ramp endpoints: {ends}")
    else:
        failures.append("RAMP")
    if render_dir and ok:
        render_preview(ramp, os.path.join(render_dir, "ramp.png"))

    # Segment builder: 100 m straight, then extend with a 90 degree left curve of 60 m
    clear_scene()
    bpy.context.scene.cursor.location = (0, 0, 0)
    bpy.ops.civil.create_road_preset(preset_type='STREET', length=100.0)
    road = bpy.context.active_object
    res = bpy.ops.civil.segment_add(kind='LEFT', length=60.0, bend=90.0)
    curve = road.civil_props.target_curve
    bpy.context.view_layer.update()
    R = 60.0 / (math.pi / 2)
    end = curve.matrix_world @ curve.data.splines[0].bezier_points[-1].co
    want = Vector((100.0 + R, R, 0.0))
    dg = bpy.context.evaluated_depsgraph_get()
    me = curve.evaluated_get(dg).to_mesh()
    radii = [((curve.matrix_world @ v.co) - Vector((100.0, R, 0.0))).length
             for v in me.vertices if (curve.matrix_world @ v.co).x > 100.5]
    curve.evaluated_get(dg).to_mesh_clear()
    worst = max(abs(r - R) for r in radii)
    lo, hi = world_bbox(road)
    ok = (res == {'FINISHED'} and (end - want).length < 0.05 and worst < 0.05
          and len(road.civil_props.segments) == 2 and len(road.data.polygons) > 0)
    print(f"[{'OK ' if ok else 'BAD'}] SEGMENTS      end={tuple(round(c, 2) for c in end)} want={tuple(round(c, 2) for c in want)} "
          f"arc radius error={worst:.3f} m, road bbox x {lo.x:.1f}..{hi.x:.1f} y {lo.y:.1f}..{hi.y:.1f}")
    if not ok:
        failures.append("SEGMENTS")
    # Bending the first segment with its slider reshapes the road
    road.civil_props.segments[0].bend = -30.0
    bpy.context.view_layer.update()
    end2 = curve.matrix_world @ curve.data.splines[0].bezier_points[-1].co
    ok = (end2 - end).length > 10.0
    print(f"[{'OK ' if ok else 'BAD'}] BEND SLIDER   end moved to {tuple(round(c, 1) for c in end2)}")
    if not ok:
        failures.append("BEND SLIDER")
    if render_dir:
        render_preview(road, os.path.join(render_dir, "segments.png"))

    # Traffic: vehicles drive their lanes in the right direction, at the right speed, banked with the road
    clear_scene()
    bpy.ops.civil.create_road_preset(preset_type='OVERPASS', length=200.0, bend=60.0)
    road = bpy.context.active_object
    road.civil_props.traffic_density = 3.0
    res = bpy.ops.civil.traffic_generate()
    col = road.civil_props.traffic_collection
    cars = [o for o in col.objects if o.get("cf_kind")]
    scene = bpy.context.scene
    fps = scene.render.fps / scene.render.fps_base
    simple = all(o.animation_data.drivers[0].driver.is_simple_expression for o in cars)

    from civilflow import road_local_samples
    samples = road_local_samples(road, bpy.context.evaluated_depsgraph_get())
    mw = road.matrix_world

    def nearest(p):
        return min(samples, key=lambda s: ((mw @ s[0]) - p).length)

    scene.frame_set(1)
    pos1 = {o.name: o.matrix_world.translation.copy() for o in cars}
    worst_up = 1.0
    worst_fwd = 1.0
    wrong_way = 0
    for o in cars:
        s = nearest(pos1[o.name])
        up = o.matrix_world.to_3x3().col[2].normalized()
        fwd = o.matrix_world.to_3x3().col[0].normalized()
        worst_up = min(worst_up, up.dot((mw.to_3x3() @ s[2]).normalized()))
        along = fwd.dot((mw.to_3x3() @ s[1]).normalized())
        worst_fwd = min(worst_fwd, abs(along))
        first_lane = o.constraints[0].target["cf_route"].split("\n")[0].split("\t")[1]
        lane_dir = 1.0 if first_lane.startswith("R") else -1.0
        if along * lane_dir < 0:
            wrong_way += 1
    scene.frame_set(1 + int(fps * 2))
    speed_err = 0.0
    moved = 0
    for o in cars:
        d = (o.matrix_world.translation - pos1[o.name]).length
        expect = o["cf_speed"] * 2.0
        if d < expect * 1.5:  # skip vehicles that wrapped from the end back to the start
            speed_err = max(speed_err, abs(d - expect) / expect)
            moved += 1
    ok = (res == {'FINISHED'} and len(cars) >= 8 and simple and wrong_way == 0 and worst_up > 0.995
          and worst_fwd > 0.98 and moved > len(cars) // 2 and speed_err < 0.05)
    kinds = sorted({o["cf_kind"] for o in cars})
    print(f"[{'OK ' if ok else 'BAD'}] TRAFFIC       {len(cars)} vehicles {kinds}, simple drivers={simple}, "
          f"wrong-way={wrong_way}, up-vs-road={worst_up:.4f}, heading={worst_fwd:.4f}, "
          f"speed error={speed_err * 100:.1f}% over {moved} vehicles")
    if not ok:
        failures.append("TRAFFIC")

    # Editing the road moves its traffic with it
    scene.frame_set(1)
    before = cars[0].matrix_world.translation.copy()
    road.civil_props.segments[0].bend = -60.0
    bpy.context.view_layer.update()
    scene.frame_set(1)
    after = cars[0].matrix_world.translation.copy()
    samples = road_local_samples(road, bpy.context.evaluated_depsgraph_get())
    off_road = min(((mw @ s[0]) - after).length for s in samples)
    ok = (after - before).length > 1.0 and off_road < road.civil_props.lanes_left * 3.6 + 6
    print(f"[{'OK ' if ok else 'BAD'}] TRAFFIC EDIT  vehicle moved {(after - before).length:.1f} m with the road, "
          f"{off_road:.1f} m from centreline")
    if not ok:
        failures.append("TRAFFIC EDIT")
    if render_dir:
        road.civil_props.segments[0].bend = 60.0
        scene.frame_set(1)
        render_preview(road, os.path.join(render_dir, "traffic.png"))
    bpy.ops.civil.traffic_remove()
    left = [o for o in bpy.data.objects if o.get("cf_kind") or o.get("cf_route")]
    ok = not left and road.civil_props.traffic_collection is None
    print(f"[{'OK ' if ok else 'BAD'}] TRAFFIC REMOVE {len(left)} leftovers")
    if not ok:
        failures.append("TRAFFIC REMOVE")

    def max_step(cars, frames):
        """Largest distance any vehicle moves between consecutive frames, relative to its speed."""
        worst = 0.0
        prev = None
        for fr in frames:
            scene.frame_set(fr)
            cur = {o.name: o.matrix_world.translation.copy() for o in cars}
            if prev:
                for o in cars:
                    worst = max(worst, (cur[o.name] - prev[o.name]).length / (o["cf_speed"] / fps))
            prev = cur
        return worst

    # Linked roads: a highway continued by an overpass, then a street; traffic flows across all three
    clear_scene()
    bpy.context.scene.cursor.location = (0, 0, 0)
    bpy.ops.civil.create_road_preset(preset_type='HIGHWAY', length=120.0)
    hw = bpy.context.active_object
    bpy.ops.civil.road_continue(preset_type='OVERPASS', length=150.0, bend=45.0, rise=6.0)
    ov = bpy.context.active_object
    bpy.ops.civil.road_continue(preset_type='STREET', length=80.0, match_lanes=False)
    st = bpy.context.active_object
    bpy.context.view_layer.objects.active = hw
    hw.select_set(True)
    res = bpy.ops.civil.traffic_generate()
    col = hw.civil_props.traffic_collection
    routes = [o for o in col.all_objects if o.get("cf_route")]
    spans = [len({ln.split("\t")[0] for ln in r["cf_route"].split("\n")}) for r in routes]
    shared = all(r.civil_props.traffic_collection == col for r in (hw, ov, st))
    cars = [o for o in col.objects if o.get("cf_kind")]
    ov_start = mod.road_end_frame(ov, at_end=False)[0]
    hw_end = mod.road_end_frame(hw, at_end=True)[0]
    # Within a route, the only jump allowed is the wrap from its end back to its start.
    jumps = 0
    for o in cars:
        L = o.constraints[0].target["cf_length"]
        prev = None
        for fr in range(1, 1 + int(fps * 4)):
            scene.frame_set(fr)
            cur = o.matrix_world.translation.copy()
            if prev is not None and (cur - prev).length > 3 * o["cf_speed"] / fps:
                jumps += 1
            prev = cur
    wraps_allowed = sum(1 for o in cars if o["cf_speed"] * 4 > 0)  # at most one wrap each in 4 s
    ok = (res == {'FINISHED'} and shared and (hw_end - ov_start).length < 0.01 and max(spans) == 3
          and len(cars) > 10 and jumps <= wraps_allowed)
    print(f"[{'OK ' if ok else 'BAD'}] LINKED ROADS  {len(routes)} routes, longest crosses {max(spans)} roads, "
          f"{len(cars)} vehicles, shared={shared}, join gap={(hw_end - ov_start).length:.3f} m, jumps={jumps}")
    if not ok:
        failures.append("LINKED ROADS")
    # Editing the middle road re-lays routes that cross it
    route = max(routes, key=lambda r: len(r["cf_route"]))
    ov.civil_props.segments[0].bend = -30.0
    bpy.context.view_layer.update()
    sp = route.data.splines[0]
    near = min((Vector(pt.co[:3]) - mod.road_end_frame(ov, at_end=True)[0]).length for pt in sp.points)
    gap = (mod.road_end_frame(ov, True)[0] - mod.road_end_frame(st, False)[0]).length
    ok = near < ov.civil_props.lanes_right * ov.civil_props.lane_width + 6 and gap < 0.01
    print(f"[{'OK ' if ok else 'BAD'}] LINKED EDIT   street stayed on the reshaped overpass (gap {gap:.3f} m); "
          f"route passes {near:.1f} m from the join")
    if not ok:
        failures.append("LINKED EDIT")
    if render_dir:
        ov.civil_props.segments[0].bend = 45.0
        scene.frame_set(30)
        render_preview(ov, os.path.join(render_dir, "linked.png"))

    # A road closed into a loop: traffic circulates with no jumps at all
    clear_scene()
    bpy.ops.civil.create_road_preset(preset_type='STREET', length=60.0, bend=90.0)
    loop = bpy.context.active_object
    for _ in range(3):
        bpy.ops.civil.segment_add(kind='LEFT', length=60.0, bend=90.0)
    loop.civil_props.traffic_density = 4.0
    bpy.ops.civil.traffic_generate()
    col = loop.civil_props.traffic_collection
    routes = [o for o in col.all_objects if o.get("cf_route")]
    cars = [o for o in col.objects if o.get("cf_kind")]
    worst = max_step(cars, range(1, 1 + int(fps * 12)))
    cyclic = all(r["cf_cyclic"] for r in routes)
    ok = cyclic and len(routes) == 2 and worst < 1.2 and len(cars) >= 4
    print(f"[{'OK ' if ok else 'BAD'}] LOOP          {len(routes)} closed routes={cyclic}, {len(cars)} vehicles, "
          f"worst frame step {worst:.2f}x expected over 12 s")
    if not ok:
        failures.append("LOOP")

    # Join Roads snaps the active road's start onto the other road's end
    clear_scene()
    bpy.ops.civil.create_road_preset(preset_type='STREET', length=50.0, bend=30.0)
    a = bpy.context.active_object
    bpy.context.scene.cursor.location = (30, -80, 5)
    bpy.ops.civil.create_road_preset(preset_type='STREET', length=50.0)
    b = bpy.context.active_object
    b.civil_props.target_curve.rotation_euler = (0, 0, 2.0)
    bpy.context.view_layer.update()
    a.select_set(True)
    b.select_set(True)
    bpy.context.view_layer.objects.active = b
    res = bpy.ops.civil.road_snap()
    bpy.context.view_layer.update()
    (pa, ha), (pb, hb) = mod.road_end_frame(a, True), mod.road_end_frame(b, False)
    ok = res == {'FINISHED'} and (pa - pb).length < 0.01 and abs(math.remainder(ha - hb, math.tau)) < 0.01
    print(f"[{'OK ' if ok else 'BAD'}] JOIN ROADS    gap {(pa - pb).length:.4f} m, "
          f"heading diff {math.degrees(abs(math.remainder(ha - hb, math.tau))):.3f} deg")
    if not ok:
        failures.append("JOIN ROADS")
    bpy.context.scene.cursor.location = (0, 0, 0)

    # Banked road: lamp heads, sign cabinets and walls stand level/vertical; both gantry styles carry text
    clear_scene()
    bpy.ops.civil.create_road_preset(preset_type='HIGHWAY', length=160.0, bend=70.0)
    road = bpy.context.active_object
    p = road.civil_props
    p.is_civil_object = False
    p.superelevation_factor = 5.0
    p.max_bank_angle = 12.0
    p.add_toll_plaza = False
    p.streetlight_placement = 'BOTH'
    p.add_sound_walls = True
    p.add_gantry_signs = True
    p.gantry_spacing = 50.0
    p.is_civil_object = True
    for style in ('FULL_SPAN', 'CANTILEVER'):
        p.gantry_style = style
        p.gantry_sign_type = 'ALTERNATE'
        me = road.data
        samples = mod.road_local_samples(road)
        bank = max(math.degrees(math.asin(min(1.0, abs(r.z)))) for _, _, _, r in samples)

        def tilt_of(mat):
            worst = 0.0
            for f in me.polygons:
                if f.material_index == mat:
                    nz = abs(f.normal.z)
                    worst = max(worst, min(math.degrees(math.acos(min(1.0, nz))), math.degrees(math.asin(min(1.0, nz)))))
            return worst
        lamp_tilt = tilt_of(7)     # luminaires
        cabinet_tilt = tilt_of(11)  # LED sign cabinets
        # Sound wall sides must be vertical (their base follows the banked road, so skip flat-ish faces).
        wall_tilt = max((math.degrees(math.asin(min(1.0, abs(f.normal.z)))) for f in me.polygons
                         if f.material_index == 3 and abs(f.normal.z) < 0.5), default=0.0)
        led_text = sum(1 for f in me.polygons if f.material_index == 10)
        guide_faces = sum(1 for f in me.polygons if f.material_index == 8)
        ok = (bank > 5.0 and lamp_tilt < 0.5 and cabinet_tilt < 0.5 and wall_tilt < 0.5
              and led_text > 200 and guide_faces >= 1)
        print(f"[{'OK ' if ok else 'BAD'}] UPRIGHT {style:10s} road banks {bank:.1f} deg; lamps tilt {lamp_tilt:.2f}, "
              f"LED cabinets {cabinet_tilt:.2f}, walls {wall_tilt:.2f} deg; LED text faces {led_text}, guide faces {guide_faces}")
        if not ok:
            failures.append(f"UPRIGHT {style}")

    # Spline tilt lands where the control point really is along the curve
    clear_scene()
    curve = mod.create_guide_curve(bpy.context, "TiltTest", [(0, 0, 0), (10, 0, 0), (100, 0, 0)])
    curve.data.splines[0].bezier_points[1].tilt = math.radians(20)
    bpy.context.view_layer.update()
    samples = mod.sample_bezier_curve(curve, 1.0, use_spline_tilt=True)
    tilts = [abs(math.asin(max(-1.0, min(1.0, r.z)))) for (_, _, _, r) in samples]
    peak_x = samples[tilts.index(max(tilts))][0].x
    ok = abs(peak_x - 10.0) < 3.0 and abs(math.degrees(max(tilts)) - 20) < 1.0
    print(f"[{'OK ' if ok else 'BAD'}] TILT          peak at x={peak_x:.1f} (control point at x=10), "
          f"angle={math.degrees(max(tilts)):.1f} deg")
    if not ok:
        failures.append("TILT")

    terrain_tests(mod, failures)
    lines_tests(mod, failures)
    junction_tests(mod, failures)
    traffic_sim_tests(mod, failures)

    mod.unregister()
    if failures:
        print("FAILURES:", failures)
        sys.exit(1)
    print("ALL PRESETS OK")


main()
