"""
Headless smoke test for CivilFlow.

Run from the repo root:
    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
        --python tools/smoke_test.py -- [--render OUT_DIR]

Loads civilflow/ straight from this checkout (no install needed), builds every preset on its
default curve, checks the meshes, and optionally renders a Workbench preview of each.
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
        lane_dir = 1.0 if o.constraints[0].target.get("cf_lane", "").startswith("R") else -1.0
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
    left = [o for o in bpy.data.objects if o.get("cf_kind") or o.get("cf_lane")]
    ok = not left and road.civil_props.traffic_collection is None
    print(f"[{'OK ' if ok else 'BAD'}] TRAFFIC REMOVE {len(left)} leftovers")
    if not ok:
        failures.append("TRAFFIC REMOVE")

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

    mod.unregister()
    if failures:
        print("FAILURES:", failures)
        sys.exit(1)
    print("ALL PRESETS OK")


main()
