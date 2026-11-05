"""
Headless test for TerraFlora (no network needed).

    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
        --python tools/terraflora_test.py

Loads civilflow/ and terraflora/ from this checkout, builds a synthetic terrain (forest on the
west half, grass on the east, a pond), grows the forest and checks painting, road clearing,
seasons, camera culling, the land-cover classifier and the panel. Exits non-zero on failure.
"""
import importlib.util
import math
import os
import sys

import bpy
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
failures = []


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, name, "__init__.py"),
                                                  submodule_search_locations=[os.path.join(REPO, name)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    mod.register()
    return mod


def report(name, ok, msg):
    print(f"[{'OK ' if ok else 'BAD'}] {name:13s} {msg}")
    if not ok:
        failures.append(name)


def plants(forest):
    """{kind: (N, 3) positions} of the forest's evaluated instances."""
    dg = bpy.context.evaluated_depsgraph_get()
    out = {}
    for inst in dg.object_instances:
        if inst.is_instance and inst.parent and inst.parent.original == forest:
            kind = inst.object.original.name.rsplit(" ", 1)[0]
            out.setdefault(kind, []).append(tuple(inst.matrix_world.translation))
    return {k: np.array(v) for k, v in out.items()}


def count_in(p, x0, x1, y0, y1):
    n = 0
    for arr in p.values():
        n += int(((arr[:, 0] > x0) & (arr[:, 0] < x1) & (arr[:, 1] > y0) & (arr[:, 1] < y1)).sum())
    return n


def main():
    cf = load("civilflow")
    load("terraflora")
    T = sys.modules["civilflow.terrain"]
    F = sys.modules["terraflora.forest"]
    L = sys.modules["terraflora.landcover"]
    TF = sys.modules["terraflora"]

    # Classifier on the band values measured over real Georgia forest, field and lake.
    def px(ndvi, n, vis, g_minus_r):
        r = n * (1 - ndvi) / (1 + ndvi)
        g = r + g_minus_r
        b = 3 * vis - r - g
        return np.array([r]), np.array([g]), np.array([b]), np.array([n])
    tree_f, grass_f, water_f = (v[0] for v in L.classify_pixels(*px(0.34, 0.726, 0.356, 0.034)))
    tree_g, grass_g, water_g = (v[0] for v in L.classify_pixels(*px(0.161, 0.790, 0.516, -0.023)))
    tree_w, grass_w, water_w = (v[0] for v in L.classify_pixels(*px(-0.277, 0.236, 0.336, 0.044)))
    tree_r, grass_r, water_r = (v[0] for v in L.classify_pixels(*px(0.034, 0.675, 0.607, 0.001)))
    ok = tree_f > 0.8 and grass_g > 0.6 and tree_g < 0.2 and water_w > 0.8 and max(tree_r, grass_r, water_r) < 0.3
    report("CLASSIFY", ok, f"forest->tree {tree_f:.2f}, field->grass {grass_g:.2f} (tree {tree_g:.2f}), "
           f"lake->water {water_w:.2f}, roof->nothing (max {max(tree_r, grass_r, water_r):.2f})")

    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    if "cf_geo" in bpy.context.scene:
        del bpy.context.scene["cf_geo"]
    x = np.arange(-300.0, 300.0 + 1e-6, 4.0)
    y = np.arange(-200.0, 200.0 + 1e-6, 4.0)
    X, Y = np.meshgrid(x, y)
    ter = T.build_terrain(bpy.context, "Terrain_TF", 500000 + x, 3700000 + y, 100 + 0.01 * X, 16, True)
    pond = ((X - 120) ** 2 + Y ** 2 < 40 ** 2).astype(float)
    layers = {"tf_pines": np.where(X < 0, 0.6, 0.0), "tf_leafy": np.where(X < 0, 0.4, 0.0),
              "tf_shrubs": np.where(X < 0, 0.3, 0.02), "tf_grass": np.where(X >= 0, 0.8, 0.0), "tf_water": pond}
    for n, a in layers.items():
        ter.data.attributes.new(n, 'FLOAT', 'POINT').data.foreach_set("value", a.astype(np.float32).ravel())
    L.write_cover_colors(ter, layers)

    bpy.context.view_layer.objects.active = ter
    bpy.ops.terraflora.grow_forest()
    forest = F.find_forest(ter)
    p = plants(forest)
    west = count_in(p, -300, -5, -200, 200)
    east_trees = sum(int((arr[:, 0] > 5).sum()) for k, arr in p.items() if k not in ("TF Shrub", "TF Grass"))
    in_pond = sum(int((np.hypot(arr[:, 0] - 120, arr[:, 1]) < 33).sum()) for arr in p.values())
    ok = forest is not None and west > 500 and east_trees == 0 and in_pond == 0 and {"TF Pine", "TF Leafy", "TF Shrub"} <= set(p)
    report("GROW", ok, f"{sum(len(a) for a in p.values())} plants {{{', '.join(f'{k}: {len(a)}' for k, a in p.items())}}}; "
           f"trees on the grass side {east_trees}, in the pond {in_pond}")
    g = p.get("TF Grass", np.zeros((0, 3)))
    g_west = int((g[:, 0] < -5).sum())
    # tf_grass 0.8 on the east half (300 x 400 m, less the pond) at 4000 per hectare, viewport share
    expect = 0.8 * 4000 * (300 * 400 - math.pi * 40 ** 2) / 10000 * forest.terraflora.viewport_amount
    report("GRASS", len(g) > 0.6 * expect and g_west == 0,
           f"{len(g)} tufts on the grass side (about {expect:.0f} expected), {g_west} in the forest")

    # Paint: remove trees from a square in the forest, add pines to a patch of grass.
    pos = np.array([tuple(v.co) for v in ter.data.vertices])
    def paint(name, x0, x1, y0, y1):
        vg = ter.vertex_groups.get(name) or ter.vertex_groups.new(name=name)
        sel = np.nonzero((pos[:, 0] >= x0) & (pos[:, 0] <= x1) & (pos[:, 1] >= y0) & (pos[:, 1] <= y1))[0]
        vg.add(sel.tolist(), 1.0, 'REPLACE')
        ter.data.update()
    before = count_in(plants(forest), -200, -100, -50, 50)
    paint("TF Remove Trees", -210, -90, -60, 60)
    after = count_in(plants(forest), -195, -105, -45, 45)
    paint("TF Add Pines", 150, 260, 100, 180)
    added = count_in({"p": plants(forest).get("TF Pine", np.zeros((0, 3)))}, 155, 255, 105, 175)
    report("PAINT", before > 30 and after == 0 and added > 10,
           f"plants in painted-out square {before} -> {after}; pines painted onto grass {added}")
    bpy.ops.terraflora.clear_paint()
    restored = count_in(plants(forest), -195, -105, -45, 45)
    report("PAINT RESET", restored > 30 and not any(n.startswith("TF Add") or n == "TF Remove Trees"
                                                    for n in ter.vertex_groups.keys()),
           f"after Undo All My Painting: {restored} plants back in the square")

    # A CivilFlow road through the forest clears its own lane.
    bpy.context.scene.cursor.location = (-290.0, -120.0, 0.0)
    for o in bpy.context.selected_objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = None
    bpy.ops.civil.create_road_preset(preset_type='STREET', length=260.0)
    road = bpy.context.active_object
    F.sync_clearing(bpy.context.scene)
    bpy.context.view_layer.update()
    p = plants(forest)
    smp = cf.road_local_samples(road)
    cl = np.array([tuple(road.matrix_world @ s[0])[:2] for s in smp])
    half = (road.civil_props.lanes_left + road.civil_props.lanes_right) * road.civil_props.lane_width * 0.5 \
        + road.civil_props.shoulder_width
    allp = np.concatenate([a for a in p.values()])
    dd = np.hypot(allp[:, None, 0] - cl[None, :, 0], allp[:, None, 1] - cl[None, :, 1])
    d, nearest = dd.min(axis=1), dd.argmin(axis=1)
    along = (nearest > 0) & (nearest < len(cl) - 1)  # beside the road, not past its ends
    on_road = int(((d < half + forest.terraflora.road_clearance - 0.5) & along).sum())
    report("ROAD CLEAR", road.name in F.clearing_collection().objects and on_road == 0,
           f"road in clearing={road.name in F.clearing_collection().objects}; plants on the road strip {on_road}")

    # Seasons: leafy trees go bare in winter, pines stay.
    forest.terraflora.season = 'WINTER'
    p = plants(forest)
    report("SEASON", "TF Bare" in p and "TF Leafy" not in p and "TF Pine" in p,
           f"winter kinds {sorted(p)}")
    forest.terraflora.season = 'SUMMER'

    # Camera culling: a camera in the middle looking east keeps nothing behind it.
    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    bpy.context.scene.collection.objects.link(cam)
    cam.location = (0.0, 0.0, 60.0)
    cam.rotation_euler = (math.radians(90), 0.0, math.radians(-90))
    bpy.context.scene.camera = cam
    forest.terraflora.view_distance = 5000.0
    forest.terraflora.camera_culling = True
    p = plants(forest)
    behind = count_in(p, -300, -60, -200, 200)
    report("CULLING", behind == 0, f"plants behind the camera with culling on: {behind}")
    forest.terraflora.camera_culling = False

    # Level of detail: cards beyond Card Distance, full models within it, grass only nearby.
    fp = forest.terraflora
    fp.card_distance, fp.grass_distance = 150.0, 100.0
    p = plants(forest)
    cam_xyz = np.array(cam.location)
    def dists(kind):
        a = p.get(kind, np.zeros((0, 3)))
        return np.linalg.norm(a - cam_xyz, axis=1)
    full = np.concatenate([dists(k) for k in ("TF Pine", "TF Leafy", "TF Shrub")])
    card = np.concatenate([dists(k) for k in ("TF Card Pine", "TF Card Leafy", "TF Card Shrub")])
    grass = dists("TF Grass")
    img = bpy.data.images.get("TF Pine 1 Card")
    alpha = np.array(img.pixels[:]).reshape(-1, 4)[:, 3] if img else np.zeros(1)
    ok = (len(card) > 100 and len(full) > 50 and card.min() > 149.0 and full.max() < 151.0
          and len(grass) > 0 and grass.max() < 101.0 and 0.05 < alpha.mean() < 0.8)
    report("CARDS", ok, f"{len(full)} full plants (max {full.max() if len(full) else 0:.0f} m), {len(card)} cards "
           f"(min {card.min() if len(card) else 0:.0f} m), grass to {grass.max() if len(grass) else 0:.0f} m; "
           f"pine card {alpha.mean() * 100:.0f}% covered")
    fp.use_cards = False
    p = plants(forest)
    report("NO CARDS", not any(k.startswith("TF Card") for k in p), f"kinds with cards off: {sorted(p)}")
    fp.use_cards = True

    # Panel draws for the terrain and the forest.
    problems = []

    class Lay:
        def __getattr__(self, n):
            return self

        def __call__(self, *a, **k):
            return self

        def prop(self, data, name, **k):
            if not hasattr(data, name):
                problems.append(name)
            return self

        def operator(self, idname, **k):
            m, o = idname.split(".")
            if not hasattr(getattr(bpy.ops, m), o):
                problems.append(idname)
            return self

    for active in (ter, forest, None):
        ctx = type("C", (), {"active_object": active, "scene": bpy.context.scene, "mode": 'OBJECT'})()
        try:
            TF.TF_PT_Main.draw(type("P", (), {"layout": Lay()})(), ctx)
        except Exception as e:
            problems.append(repr(e))
    report("PANEL", not problems, "draws for terrain, forest and nothing" if not problems else str(problems))

    if failures:
        print("FAILURES:", failures)
        sys.exit(1)
    print("TERRAFLORA OK")


main()
