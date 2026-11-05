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

    mod.unregister()
    if failures:
        print("FAILURES:", failures)
        sys.exit(1)
    print("ALL PRESETS OK")


main()
