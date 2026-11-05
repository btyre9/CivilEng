"""
CivilFlow traffic: low-poly cars and trucks driving along a road's lanes.

Each lane gets a hidden path curve (parented to the road) that runs in that lane's direction of
travel and carries the road's banking as tilt. Each vehicle follows its lane with a Follow Path
constraint whose offset is driven by a simple driver expression (no Python auto-run needed):

    offset = fract(phase + frame * speed / (fps * lane_length))

Speed and phase are custom properties on each vehicle, so they can be tweaked per vehicle.
Lane paths are rewritten whenever the road rebuilds, so traffic follows road edits live.
"""

import math
import random

import bpy
import bmesh
from mathutils import Vector

# ==============================================================================
# Materials
# ==============================================================================

PAINT, GLASS, TIRE, HEADLIGHT, TAILLIGHT, TRIM, CARGO = range(7)

PAINT_COLORS = [
    ("White", (0.80, 0.80, 0.80, 1.0), 5),
    ("Black", (0.02, 0.02, 0.02, 1.0), 4),
    ("Silver", (0.45, 0.46, 0.48, 1.0), 4),
    ("Graphite", (0.18, 0.18, 0.19, 1.0), 3),
    ("Navy", (0.02, 0.06, 0.20, 1.0), 2),
    ("Red", (0.45, 0.02, 0.02, 1.0), 2),
    ("Green", (0.03, 0.15, 0.06, 1.0), 1),
    ("Beige", (0.45, 0.38, 0.26, 1.0), 1),
    ("Orange", (0.60, 0.35, 0.02, 1.0), 1),
]


def paint_material(name, color):
    """Car paint in one colour; set per vehicle (object-linked slot) so Solid view shows it too."""
    return _material(f"CF_Paint_{name}", color, roughness=0.3, metallic=0.4)


def _material(name, color, roughness=0.5, metallic=0.0, emission=None, strength=0.0, object_color=False):
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name=name)
    nt = mat.node_tree
    if nt is not None:
        bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if bsdf is None:
            bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
            out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None)
            if out is None:
                out = nt.nodes.new('ShaderNodeOutputMaterial')
            nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
        if emission and "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = emission
            bsdf.inputs["Emission Strength"].default_value = strength
        if object_color:
            # Each vehicle's paint comes from its Object > Viewport Display > Color.
            info = nt.nodes.new('ShaderNodeObjectInfo')
            info.location = (bsdf.location.x - 250, bsdf.location.y)
            nt.links.new(info.outputs["Color"], bsdf.inputs["Base Color"])
    mat.diffuse_color = color
    return mat


def vehicle_materials():
    return [
        # Placeholder for the paint slot; each vehicle overrides it with its own paint colour.
        _material("CF_Vehicle_Paint", (0.5, 0.5, 0.5, 1.0), roughness=0.3, metallic=0.4, object_color=True),
        _material("CF_Vehicle_Glass", (0.03, 0.05, 0.07, 1.0), roughness=0.05, metallic=0.2),
        _material("CF_Vehicle_Tire", (0.02, 0.02, 0.02, 1.0), roughness=0.9),
        _material("CF_Vehicle_Headlight", (1.0, 0.97, 0.9, 1.0), roughness=0.2,
                  emission=(1.0, 0.95, 0.85, 1.0), strength=6.0),
        _material("CF_Vehicle_Taillight", (0.6, 0.0, 0.0, 1.0), roughness=0.3,
                  emission=(1.0, 0.02, 0.01, 1.0), strength=3.0),
        _material("CF_Vehicle_Trim", (0.06, 0.06, 0.06, 1.0), roughness=0.6),
        _material("CF_Vehicle_Cargo", (0.85, 0.85, 0.83, 1.0), roughness=0.6),
    ]


# ==============================================================================
# Low-poly vehicle meshes (front faces +X, origin on the ground at the vehicle centre)
# ==============================================================================

def _prism(bm, bottom, top, z0, z1, mat):
    """Box whose bottom (x0, x1, half_y) and top (x0, x1, half_y) rectangles can differ (tapers)."""
    bx0, bx1, by = bottom
    tx0, tx1, ty = top
    v = [bm.verts.new(c) for c in (
        (bx0, -by, z0), (bx1, -by, z0), (bx1, by, z0), (bx0, by, z0),
        (tx0, -ty, z1), (tx1, -ty, z1), (tx1, ty, z1), (tx0, ty, z1),
    )]
    for f in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
        bm.faces.new([v[i] for i in f]).material_index = mat


def _box(bm, x0, x1, hy, z0, z1, mat, y_center=0.0):
    if y_center == 0.0:
        _prism(bm, (x0, x1, hy), (x0, x1, hy), z0, z1, mat)
        return
    geom_start = len(bm.verts)
    _prism(bm, (x0, x1, hy), (x0, x1, hy), z0, z1, mat)
    bm.verts.ensure_lookup_table()
    for v in bm.verts[geom_start:]:
        v.co.y += y_center


def _wheel(bm, x, y, r, w, segs=10):
    rings = []
    for side in (-w * 0.5, w * 0.5):
        ring = []
        for k in range(segs):
            a = 2.0 * math.pi * k / segs
            ring.append(bm.verts.new((x + r * math.cos(a), y + side, r + r * math.sin(a))))
        rings.append(ring)
    a, b = rings
    for k in range(segs):
        kn = (k + 1) % segs
        bm.faces.new((a[k], a[kn], b[kn], b[k])).material_index = TIRE
    bm.faces.new(a).material_index = TIRE
    bm.faces.new(list(reversed(b))).material_index = TIRE


def _lights(bm, front_x, rear_x, hy, z0, z1, rz0=None, rz1=None):
    rz0 = z0 if rz0 is None else rz0
    rz1 = z1 if rz1 is None else rz1
    for s in (-1.0, 1.0):
        yc = s * (hy - 0.18)
        _box(bm, front_x - 0.02, front_x + 0.03, 0.14, z0, z1, HEADLIGHT, y_center=yc)
        _box(bm, rear_x - 0.03, rear_x + 0.02, 0.14, rz0, rz1, TAILLIGHT, y_center=yc)


def _axles(bm, xs, track, r, w):
    for x in xs:
        for s in (-1.0, 1.0):
            _wheel(bm, x, s * track, r, w)


def build_car(bm):
    _box(bm, -2.25, 2.25, 0.88, 0.30, 0.82, PAINT)
    _prism(bm, (-1.35, 1.05, 0.84), (-1.0, 0.35, 0.72), 0.82, 1.36, GLASS)
    _prism(bm, (-1.0, 0.35, 0.72), (-0.95, 0.30, 0.68), 1.36, 1.43, PAINT)
    _box(bm, 2.20, 2.32, 0.86, 0.26, 0.44, TRIM)
    _box(bm, -2.32, -2.20, 0.86, 0.26, 0.44, TRIM)
    _lights(bm, 2.25, -2.25, 0.88, 0.58, 0.70)
    _axles(bm, (1.38, -1.38), 0.78, 0.32, 0.22)


def build_suv(bm):
    _box(bm, -2.45, 2.45, 0.95, 0.36, 1.02, PAINT)
    _prism(bm, (-2.40, 1.25, 0.92), (-2.35, 0.55, 0.82), 1.02, 1.78, GLASS)
    _prism(bm, (-2.35, 0.55, 0.82), (-2.30, 0.50, 0.78), 1.78, 1.88, PAINT)
    _box(bm, 2.40, 2.52, 0.92, 0.30, 0.52, TRIM)
    _box(bm, -2.52, -2.40, 0.92, 0.30, 0.52, TRIM)
    _lights(bm, 2.45, -2.45, 0.95, 0.72, 0.86, 0.80, 0.98)
    _axles(bm, (1.55, -1.55), 0.84, 0.37, 0.26)


def build_box_truck(bm):
    _box(bm, -4.30, 4.30, 0.90, 0.45, 0.88, TRIM)
    _box(bm, 2.20, 4.30, 1.15, 0.55, 1.90, PAINT)
    _prism(bm, (2.40, 4.25, 1.10), (2.60, 3.90, 1.05), 1.90, 2.75, GLASS)
    _box(bm, 2.60, 3.90, 1.05, 2.75, 2.85, PAINT)
    _box(bm, -4.30, 2.05, 1.25, 0.90, 3.60, CARGO)
    _box(bm, 4.28, 4.40, 1.12, 0.45, 0.75, TRIM)
    _lights(bm, 4.30, -4.30, 1.15, 0.85, 1.05, 1.00, 1.20)
    _axles(bm, (3.30, -2.40, -3.40), 1.00, 0.50, 0.30)


def build_semi(bm):
    _box(bm, -8.25, 4.40, 0.90, 0.90, 1.15, TRIM)
    _box(bm, -8.25, 4.40, 1.28, 1.15, 4.00, CARGO)
    _box(bm, 4.20, 8.25, 0.90, 0.50, 1.00, TRIM)
    _box(bm, 5.40, 8.25, 1.20, 0.60, 2.30, PAINT)
    _prism(bm, (5.60, 8.20, 1.15), (5.80, 7.80, 1.10), 2.30, 3.10, GLASS)
    _prism(bm, (5.40, 7.80, 1.15), (5.60, 7.00, 1.00), 3.10, 3.85, PAINT)
    _box(bm, 8.22, 8.36, 1.18, 0.50, 0.85, TRIM)
    _lights(bm, 8.25, -8.25, 1.20, 1.00, 1.20, 1.25, 1.45)
    _axles(bm, (7.40, 5.30, 4.40, -5.90, -7.10), 1.00, 0.52, 0.30)


VEHICLE_TYPES = {
    # key: (builder, length in metres, is_truck)
    'CAR': (build_car, 4.6, False),
    'SUV': (build_suv, 5.0, False),
    'BOX_TRUCK': (build_box_truck, 8.8, True),
    'SEMI': (build_semi, 16.6, True),
}


def vehicle_mesh(kind):
    """Shared mesh data for one vehicle type, created on first use."""
    name = f"CF_{kind.title().replace('_', '')}_Mesh"
    me = bpy.data.meshes.get(name)
    if me is not None and len(me.polygons):
        return me
    if me is None:
        me = bpy.data.meshes.new(name)
    # Materials first: clearing a mesh's material list resets every face to slot 0.
    me.materials.clear()
    for m in vehicle_materials():
        me.materials.append(m)
    bm = bmesh.new()
    VEHICLE_TYPES[kind][0](bm)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    return me


# ==============================================================================
# Lanes
# ==============================================================================

LANE_Z = 0.08  # asphalt top above the guide curve


def lane_layout(props):
    """[(key, x offset, direction, rank)]; direction +1 travels along the curve, rank 0 is the kerb lane."""
    med = props.median_width if (props.lanes_left > 0 and props.lanes_right > 0) else 0.0
    half_w = ((props.lanes_left + props.lanes_right) * props.lane_width + med + props.shoulder_width * 2.0) * 0.5
    right_edge = half_w - props.shoulder_width
    left_edge = -half_w + props.shoulder_width
    # Right-hand traffic travels along the curve on its right side; left-hand traffic on its left.
    fwd = -1.0 if props.traffic_left_hand else 1.0
    lanes = []
    for k in range(props.lanes_right):
        lanes.append((f"R{k}", right_edge - (k + 0.5) * props.lane_width, fwd, k))
    for k in range(props.lanes_left):
        lanes.append((f"L{k}", left_edge + (k + 0.5) * props.lane_width, -fwd, k))
    return lanes


def lane_points(samples, x, direction):
    """Lane centreline points in travel order, with the road's bank as curve tilt."""
    up = Vector((0.0, 0.0, 1.0))
    pts = []
    for pos, tan, nrm, right in samples:
        p = pos + right * x + nrm * LANE_Z
        t = tan * direction
        r = right * direction
        flat_r = t.cross(up)
        tilt = 0.0
        if flat_r.length > 1e-6:
            flat_r.normalize()
            tilt = math.atan2(flat_r.cross(r).dot(t), flat_r.dot(r))
        pts.append((p, tilt))
    if direction < 0:
        pts.reverse()
    return pts


def write_lane_curve(lane_obj, pts):
    cu = lane_obj.data
    cu.splines.clear()
    sp = cu.splines.new('POLY')
    sp.points.add(len(pts) - 1)
    length = 0.0
    for i, (p, tilt) in enumerate(pts):
        sp.points[i].co = (p.x, p.y, p.z, 1.0)
        sp.points[i].tilt = tilt
        if i:
            length += (p - pts[i - 1][0]).length
    lane_obj["cf_length"] = max(length, 0.01)
    cu.update_tag()
    return length


def update_lane_curves(road, local_samples):
    """Called after every road rebuild: keeps lane paths on the road as it changes."""
    col = road.civil_props.traffic_collection
    if col is None or len(local_samples) < 2:
        return
    layout = {key: (x, d) for key, x, d, _ in lane_layout(road.civil_props)}
    for o in col.all_objects:
        key = o.get("cf_lane")
        if o.type == 'CURVE' and key in layout:
            x, d = layout[key]
            write_lane_curve(o, lane_points(local_samples, x, d))


# ==============================================================================
# Vehicles
# ==============================================================================

DRIVER_EXPR = "p + frame * s * b / (f * L) - floor(p + frame * s * b / (f * L))"


def _add_var(drv, name, id_type, id_ref, path):
    var = drv.variables.new()
    var.name = name
    var.type = 'SINGLE_PROP'
    var.targets[0].id_type = id_type
    var.targets[0].id = id_ref
    var.targets[0].data_path = path


def add_vehicle(scene, collection, kind, lane_obj, phase, speed, color):
    obj = bpy.data.objects.new(kind.title().replace('_', ' '), vehicle_mesh(kind))
    collection.objects.link(obj)
    name, rgba = color
    obj.color = rgba
    obj.material_slots[PAINT].link = 'OBJECT'
    obj.material_slots[PAINT].material = paint_material(name, rgba)
    obj["cf_phase"] = phase
    obj["cf_speed"] = speed  # metres per second
    obj["cf_kind"] = kind

    con = obj.constraints.new('FOLLOW_PATH')
    con.target = lane_obj
    con.use_curve_follow = True
    con.use_fixed_location = True
    con.forward_axis = 'FORWARD_X'
    con.up_axis = 'UP_Z'

    drv = con.driver_add("offset_factor").driver
    drv.type = 'SCRIPTED'
    _add_var(drv, "p", 'OBJECT', obj, '["cf_phase"]')
    _add_var(drv, "s", 'OBJECT', obj, '["cf_speed"]')
    _add_var(drv, "L", 'OBJECT', lane_obj, '["cf_length"]')
    _add_var(drv, "f", 'SCENE', scene, "render.fps")
    _add_var(drv, "b", 'SCENE', scene, "render.fps_base")
    drv.expression = DRIVER_EXPR
    return obj


def _pick_kind(rng, truck_share):
    if rng.random() < truck_share:
        return 'SEMI' if rng.random() < 0.5 else 'BOX_TRUCK'
    return 'SUV' if rng.random() < 0.35 else 'CAR'


def _pick_color(rng):
    """(name, rgba) from the paint palette, weighted towards common car colours."""
    total = sum(w for _, _, w in PAINT_COLORS)
    r = rng.random() * total
    for name, c, w in PAINT_COLORS:
        r -= w
        if r <= 0:
            return name, c
    return PAINT_COLORS[0][0], PAINT_COLORS[0][1]


def remove_traffic(road):
    p = road.civil_props
    col = p.traffic_collection
    p.traffic_collection = None
    if col is None:
        return
    subs = list(col.children_recursive)
    for c in [col] + subs:
        for o in list(c.objects):
            data = o.data
            bpy.data.objects.remove(o, do_unlink=True)
            if isinstance(data, bpy.types.Curve) and data.users == 0:
                bpy.data.curves.remove(data)
    for c in reversed(subs):
        bpy.data.collections.remove(c)
    bpy.data.collections.remove(col)


def build_traffic(context, road, local_samples):
    """(Re)creates lane paths and vehicles for `road`. Returns the number of vehicles."""
    p = road.civil_props
    remove_traffic(road)
    lanes = lane_layout(p)
    if not lanes or len(local_samples) < 2:
        return 0

    scene = context.scene
    col = bpy.data.collections.new(f"Traffic_{road.name}")
    scene.collection.children.link(col)
    lane_col = bpy.data.collections.new(f"Lanes_{road.name}")
    col.children.link(lane_col)
    lane_col.hide_render = True
    p.traffic_collection = col

    rng = random.Random(p.traffic_seed)
    base_speed = p.traffic_speed / 3.6
    min_gap = 6.0
    count = 0
    lanes_per_dir = {}
    for _, _, d, _ in lanes:
        lanes_per_dir[d] = lanes_per_dir.get(d, 0) + 1

    for key, x, d, rank in lanes:
        cu = bpy.data.curves.new(f"Lane_{road.name}_{key}", type='CURVE')
        cu.dimensions = '3D'
        cu.use_path = True
        cu.twist_mode = 'Z_UP'
        lane_obj = bpy.data.objects.new(f"Lane_{road.name}_{key}", cu)
        lane_obj["cf_lane"] = key
        lane_col.objects.link(lane_obj)
        lane_obj.parent = road
        lane_obj.hide_render = True
        lane_obj.display_type = 'WIRE'
        length = write_lane_curve(lane_obj, lane_points(local_samples, x, d))

        # Kerb lanes run slower and carry most of the trucks; inner lanes are faster.
        n_dir = lanes_per_dir[d]
        speed = base_speed * (0.88 + 0.16 * (rank / max(1, n_dir - 1))) * rng.uniform(0.97, 1.03)
        if n_dir == 1:
            truck_share = p.traffic_truck_share
        elif rank == 0:
            truck_share = min(1.0, p.traffic_truck_share * 1.8)
        else:
            truck_share = p.traffic_truck_share * 0.4

        # Choose the vehicles first, then share the leftover road between the gaps, so no two
        # vehicles in a lane ever overlap (they all move at the lane's speed).
        wanted = max(1, round(length * p.traffic_density / 100.0))
        kinds = [_pick_kind(rng, truck_share) for _ in range(wanted)]
        while kinds and sum(VEHICLE_TYPES[k][1] + min_gap for k in kinds) > length:
            kinds.pop()
        if not kinds:
            continue
        spare = length - sum(VEHICLE_TYPES[k][1] + min_gap for k in kinds)
        weights = [rng.uniform(0.3, 1.7) for _ in kinds]
        wsum = sum(weights)
        cursor = rng.uniform(0.0, length)
        for k, w in zip(kinds, weights):
            veh_len = VEHICLE_TYPES[k][1]
            centre = cursor + veh_len * 0.5
            add_vehicle(scene, col, k, lane_obj, (centre / length) % 1.0, speed, _pick_color(rng))
            cursor += veh_len + min_gap + spare * w / wsum
            count += 1

    return count


# ==============================================================================
# Operators
# ==============================================================================

def _active_road(context):
    obj = context.active_object
    if obj and obj.type == 'MESH' and obj.civil_props.is_civil_object:
        return obj
    return None


class CIVIL_OT_TrafficGenerate(bpy.types.Operator):
    bl_idname = "civil.traffic_generate"
    bl_label = "Generate Traffic"
    bl_description = ("Fill the road's lanes with low-poly cars and trucks that drive along it "
                      "when you play the animation. Replaces this road's existing traffic")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        road = _active_road(context)
        return road is not None and (road.civil_props.lanes_left + road.civil_props.lanes_right) > 0

    def execute(self, context):
        from . import road_local_samples
        road = _active_road(context)
        samples = road_local_samples(road, context.evaluated_depsgraph_get())
        if len(samples) < 2:
            self.report({'WARNING'}, "The road has no usable guide curve.")
            return {'CANCELLED'}
        n = build_traffic(context, road, samples)
        context.view_layer.objects.active = road
        self.report({'INFO'}, f"Added {n} vehicles to {road.name}. Press Space to play.")
        return {'FINISHED'}


class CIVIL_OT_TrafficRemove(bpy.types.Operator):
    bl_idname = "civil.traffic_remove"
    bl_label = "Remove Traffic"
    bl_description = "Delete this road's vehicles and lane paths"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        road = _active_road(context)
        return road is not None and road.civil_props.traffic_collection is not None

    def execute(self, context):
        remove_traffic(_active_road(context))
        return {'FINISHED'}


classes = (
    CIVIL_OT_TrafficGenerate,
    CIVIL_OT_TrafficRemove,
)
