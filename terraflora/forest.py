# SPDX-License-Identifier: GPL-3.0-or-later
"""
The forest: one object whose Geometry Nodes modifier scatters the plant library over a terrain.

Per layer (pines, leafy trees, shrubs) the density at each spot is

    clamp(auto * (1 - painted removal) + painted addition) * (1 - water)

where 'auto' is the land cover read from the photo (tf_pines / tf_leafy / tf_shrubs) and the
painted layers are vertex groups you paint in Weight Paint. Plants are spaced with Poisson-disk
sampling, kept clear of roads (anything in the TerraFlora Clearing collection, measured on the
map so bridges and banks clear too), optionally culled to what the camera sees, and placed as
instances (shared meshes), so even 100,000 trees stay light.
"""

import bpy

from . import assets

NODE_GROUP = "TF_Forest"
NODE_VERSION = 2
CLEARING = "TerraFlora Clearing"
MOD_NAME = "TerraFlora"

PAINT_LAYERS = {
    'REMOVE': ("TF Remove Trees", "Remove Trees", "Clear trees and shrubs where you paint", 'X'),
    'PINES': ("TF Add Pines", "Add Pines", "Plant pines where you paint", 'OUTLINER_OB_FORCE_FIELD'),
    'LEAFY': ("TF Add Leafy Trees", "Add Leafy Trees", "Plant leafy trees where you paint", 'OUTLINER_OB_POINTCLOUD'),
    'SHRUBS': ("TF Add Shrubs", "Add Shrubs", "Plant shrubs where you paint", 'SEQ_HISTOGRAM'),
}

INPUTS = [
    # name, socket type, default, min, max, description
    ("Terrain", 'NodeSocketObject', None, None, None, "Ground to grow on"),
    ("Clearing", 'NodeSocketCollection', None, None, None, "Roads and anything else to keep clear"),
    ("Pines", 'NodeSocketCollection', None, None, None, ""),
    ("Leafy Trees", 'NodeSocketCollection', None, None, None, ""),
    ("Shrubs", 'NodeSocketCollection', None, None, None, ""),
    ("Pines per Hectare", 'NodeSocketFloat', 300.0, 0.0, 3000.0, "Pines in full forest, per 10,000 m²"),
    ("Leafy Trees per Hectare", 'NodeSocketFloat', 260.0, 0.0, 3000.0, "Leafy trees in full forest, per 10,000 m²"),
    ("Shrubs per Hectare", 'NodeSocketFloat', 250.0, 0.0, 5000.0, "Shrubs where they grow fully, per 10,000 m²"),
    ("Road Clearance", 'NodeSocketFloat', 3.0, 0.0, 100.0, "Distance kept clear around roads"),
    ("Seed", 'NodeSocketInt', 0, 0, 100000, "Change for a different random layout"),
    ("Viewport Amount", 'NodeSocketFloat', 0.35, 0.0, 1.0, "Share of the plants shown in the viewport (renders show all)"),
    ("Camera", 'NodeSocketObject', None, None, None, "Camera for culling"),
    ("Camera Culling", 'NodeSocketBool', False, None, None, "Only grow what the camera can see"),
    ("View Distance", 'NodeSocketFloat', 1500.0, 10.0, 50000.0, "Furthest distance grown when culling"),
    ("View Angle", 'NodeSocketFloat', 110.0, 10.0, 180.0, "Width of the cone kept around the camera's view, in degrees"),
]


class _B:
    """Tiny helper for building node trees in code."""

    def __init__(self, ng):
        self.ng = ng

    def node(self, idname, x, y, **props):
        n = self.ng.nodes.new(idname)
        n.location = (x, y)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def put(self, sock, value):
        if isinstance(value, bpy.types.NodeSocket):
            self.ng.links.new(value, sock)
        elif value is not None:
            sock.default_value = value

    def math(self, op, a, b=None, x=0, y=0, clamp=False):
        n = self.node('ShaderNodeMath', x, y, operation=op, use_clamp=clamp)
        self.put(n.inputs[0], a)
        if b is not None:
            self.put(n.inputs[1], b)
        return n.outputs[0]

    def vmath(self, op, a, b=None, x=0, y=0, out=0):
        n = self.node('ShaderNodeVectorMath', x, y, operation=op)
        self.put(n.inputs[0], a)
        if b is not None:
            self.put(n.inputs[1], b)
        return n.outputs[out]

    def attr(self, name, x, y):
        n = self.node('GeometryNodeInputNamedAttribute', x, y, data_type='FLOAT')
        n.inputs["Name"].default_value = name
        return n.outputs["Attribute"]

    def compare(self, op, a, b, x, y, data_type='FLOAT'):
        n = self.node('FunctionNodeCompare', x, y, data_type=data_type, operation=op)
        self.put(n.inputs["A"], a)
        self.put(n.inputs["B"], b)
        return n.outputs["Result"]

    def boolean(self, op, a, b=None, x=0, y=0):
        n = self.node('FunctionNodeBooleanMath', x, y, operation=op)
        self.put(n.inputs[0], a)
        if b is not None:
            self.put(n.inputs[1], b)
        return n.outputs[0]

    def random(self, data_type, lo, hi, seed, x, y):
        n = self.node('FunctionNodeRandomValue', x, y, data_type=data_type)
        self.put(n.inputs["Min"], lo)
        self.put(n.inputs["Max"], hi)
        self.put(n.inputs["Seed"], seed)
        return n.outputs["Value"]


def build_node_group():
    ng = bpy.data.node_groups.get(NODE_GROUP)
    if ng is not None and ng.get("tf_version") == NODE_VERSION:
        return ng
    if ng is None:
        ng = bpy.data.node_groups.new(NODE_GROUP, 'GeometryNodeTree')
    ng.nodes.clear()
    ng.interface.clear()
    ng["tf_version"] = NODE_VERSION
    ng.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    for name, stype, default, lo, hi, desc in INPUTS:
        s = ng.interface.new_socket(name=name, in_out='INPUT', socket_type=stype)
        s.description = desc
        if default is not None:
            s.default_value = default
        if lo is not None:
            s.min_value, s.max_value = lo, hi
    ng.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

    b = _B(ng)
    gin = b.node('NodeGroupInput', -2200, 0)
    gout = b.node('NodeGroupOutput', 1600, 0)
    I = gin.outputs

    terrain = b.node('GeometryNodeObjectInfo', -1900, 400, transform_space='RELATIVE')
    b.put(terrain.inputs["Object"], I["Terrain"])
    ground = terrain.outputs["Geometry"]

    # Fewer plants while working in the viewport.
    isvp = b.node('GeometryNodeIsViewport', -1900, 150)
    vp = b.node('GeometryNodeSwitch', -1700, 150, input_type='FLOAT')
    b.put(vp.inputs["Switch"], isvp.outputs[0])
    b.put(vp.inputs["False"], 1.0)
    b.put(vp.inputs["True"], I["Viewport Amount"])
    vp_amount = vp.outputs[0]

    # Roads and other clearings, flattened onto the map so distance is measured across the ground.
    cinfo = b.node('GeometryNodeCollectionInfo', -1900, -250, transform_space='RELATIVE')
    b.put(cinfo.inputs["Collection"], I["Clearing"])
    real = b.node('GeometryNodeRealizeInstances', -1700, -250)
    b.put(real.inputs["Geometry"], cinfo.outputs[0])
    pos = b.node('GeometryNodeInputPosition', -1700, -400)
    flat = b.vmath('MULTIPLY', pos.outputs[0], (1.0, 1.0, 0.0), -1500, -400)
    setp = b.node('GeometryNodeSetPosition', -1300, -250)
    b.put(setp.inputs["Geometry"], real.outputs[0])
    b.put(setp.inputs["Position"], flat)
    clear_geo = setp.outputs[0]
    size = b.node('GeometryNodeAttributeDomainSize', -1500, -550, component='MESH')
    b.put(size.inputs["Geometry"], real.outputs[0])
    has_clear = b.compare('GREATER_THAN', size.outputs["Face Count"], 0, -1300, -550, data_type='INT')

    # Camera view cone.
    cam = b.node('GeometryNodeObjectInfo', -1900, -800, transform_space='RELATIVE')
    b.put(cam.inputs["Object"], I["Camera"])
    rot = b.node('FunctionNodeRotateVector', -1700, -800)
    b.put(rot.inputs["Vector"], (0.0, 0.0, -1.0))
    b.put(rot.inputs["Rotation"], cam.outputs["Rotation"])
    half_angle = b.math('MULTIPLY', b.math('RADIANS', I["View Angle"], x=-1700, y=-950), 0.5, -1500, -950)
    cos_half = b.math('COSINE', half_angle, x=-1300, y=-950)

    rem = b.attr("TF Remove Trees", -1500, 900)
    water = b.attr("tf_water", -1500, 750)
    dry = b.math('SUBTRACT', 1.0, water, -1300, 750)
    keep_auto = b.math('SUBTRACT', 1.0, rem, -1300, 900)

    layers = (
        ("tf_pines", "TF Add Pines", "Pines per Hectare", "Pines", 4.5, 0),
        ("tf_leafy", "TF Add Leafy Trees", "Leafy Trees per Hectare", "Leafy Trees", 4.5, 100),
        ("tf_shrubs", "TF Add Shrubs", "Shrubs per Hectare", "Shrubs", 2.0, 200),
    )
    join = b.node('GeometryNodeJoinGeometry', 1400, 0)
    for k, (auto_name, add_name, per_ha, coll, spacing, seed_off) in enumerate(layers):
        y = 1300 - k * 900
        x = -1100
        auto = b.attr(auto_name, x, y)
        add = b.attr(add_name, x, y - 150)
        d = b.math('MULTIPLY', auto, keep_auto, x + 200, y)
        d = b.math('ADD', d, add, x + 400, y, clamp=True)
        d = b.math('MULTIPLY', d, dry, x + 600, y)
        dmax = b.math('MULTIPLY', b.math('DIVIDE', I[per_ha], 10000.0, x + 200, y - 300), vp_amount, x + 400, y - 300)
        seed = b.math('ADD', I["Seed"], seed_off, x + 200, y - 450)

        dist = b.node('GeometryNodeDistributePointsOnFaces', x + 800, y, distribute_method='POISSON')
        b.put(dist.inputs["Mesh"], ground)
        b.put(dist.inputs["Distance Min"], spacing)
        b.put(dist.inputs["Density Max"], dmax)
        b.put(dist.inputs["Density Factor"], d)
        b.put(dist.inputs["Seed"], seed)
        pts = dist.outputs["Points"]

        # Keep off roads.
        ppos = b.node('GeometryNodeInputPosition', x + 800, y - 300)
        pflat = b.vmath('MULTIPLY', ppos.outputs[0], (1.0, 1.0, 0.0), x + 1000, y - 300)
        prox = b.node('GeometryNodeProximity', x + 1200, y - 250, target_element='FACES')
        b.put(prox.inputs["Geometry"], clear_geo)
        b.put(prox.inputs["Sample Position"], pflat)
        near = b.compare('LESS_THAN', prox.outputs["Distance"], I["Road Clearance"], x + 1400, y - 250)
        near = b.boolean('AND', near, has_clear, x + 1600, y - 250)
        dele = b.node('GeometryNodeDeleteGeometry', x + 1800, y, domain='POINT')
        b.put(dele.inputs["Geometry"], pts)
        b.put(dele.inputs["Selection"], near)
        pts = dele.outputs[0]

        # Camera culling.
        rel = b.vmath('SUBTRACT', ppos.outputs[0], cam.outputs["Location"], x + 1000, y - 550)
        length = b.vmath('LENGTH', rel, x=x + 1200, y=y - 500, out=1)
        dirn = b.vmath('NORMALIZE', rel, x=x + 1200, y=y - 650)
        dot = b.vmath('DOT_PRODUCT', dirn, rot.outputs[0], x + 1400, y - 650, out=1)
        in_cone = b.compare('GREATER_THAN', dot, cos_half, x + 1600, y - 650)
        in_range = b.compare('LESS_THAN', length, I["View Distance"], x + 1600, y - 500)
        visible = b.boolean('AND', in_cone, in_range, x + 1800, y - 550)
        hide = b.boolean('NIMPLY', I["Camera Culling"], visible, x + 2000, y - 550)
        cull = b.node('GeometryNodeDeleteGeometry', x + 2000, y, domain='POINT')
        b.put(cull.inputs["Geometry"], pts)
        b.put(cull.inputs["Selection"], hide)
        pts = cull.outputs[0]

        # Plants: a random one of the layer's models, turned and sized at random.
        cinst = b.node('GeometryNodeCollectionInfo', x + 2000, y + 300, transform_space='ORIGINAL')
        b.put(cinst.inputs["Collection"], I[coll])
        cinst.inputs["Separate Children"].default_value = True
        cinst.inputs["Reset Children"].default_value = True
        idx = b.random('INT', 0, 1000, seed_off + 1, x + 2000, y - 200)
        ang = b.random('FLOAT', 0.0, 6.2832, seed_off + 2, x + 2000, y - 350)
        sc = b.random('FLOAT', 0.75, 1.3, seed_off + 3, x + 2000, y - 500)
        e = b.node('ShaderNodeCombineXYZ', x + 2200, y - 350)
        b.put(e.inputs["Z"], ang)
        r = b.node('FunctionNodeEulerToRotation', x + 2400, y - 350)
        b.put(r.inputs[0], e.outputs[0])
        s3 = b.node('ShaderNodeCombineXYZ', x + 2200, y - 500)
        for c in ("X", "Y", "Z"):
            b.put(s3.inputs[c], sc)
        iop = b.node('GeometryNodeInstanceOnPoints', x + 2600, y)
        b.put(iop.inputs["Points"], pts)
        b.put(iop.inputs["Instance"], cinst.outputs[0])
        iop.inputs["Pick Instance"].default_value = True
        b.put(iop.inputs["Instance Index"], idx)
        b.put(iop.inputs["Rotation"], r.outputs[0])
        b.put(iop.inputs["Scale"], s3.outputs[0])
        tint = b.random('FLOAT', 0.0, 1.0, seed_off + 4, x + 2600, y - 300)
        store = b.node('GeometryNodeStoreNamedAttribute', x + 2800, y, data_type='FLOAT', domain='INSTANCE')
        b.put(store.inputs["Geometry"], iop.outputs[0])
        store.inputs["Name"].default_value = "tf_tint"
        b.put(store.inputs["Value"], tint)
        ng.links.new(store.outputs[0], join.inputs[0])
    ng.links.new(join.outputs[0], gout.inputs[0])
    return ng


def clearing_collection():
    """Collection of everything trees keep away from: every CivilFlow road, kept up to date."""
    col = bpy.data.collections.get(CLEARING)
    if col is None:
        col = bpy.data.collections.new(CLEARING)
    return col


def sync_clearing(scene):
    col = bpy.data.collections.get(CLEARING)
    if col is None:
        return
    for ob in scene.objects:
        props = getattr(ob, "civil_props", None)
        if ob.type == 'MESH' and props is not None and props.is_civil_object and ob.name not in col.objects:
            col.objects.link(ob)


def socket_id(ng, name):
    for item in ng.interface.items_tree:
        if getattr(item, "item_type", None) == 'SOCKET' and item.in_out == 'INPUT' and item.name == name:
            return item.identifier
    raise KeyError(name)


def find_forest(terrain):
    for ob in bpy.data.objects:
        p = getattr(ob, "terraflora", None)
        if p is not None and p.is_forest and p.terrain == terrain:
            return ob
    return None


def sync_modifier(forest, scene=None):
    """Copies the forest's settings onto its Geometry Nodes inputs."""
    p = forest.terraflora
    mod = forest.modifiers.get(MOD_NAME)
    if mod is None or mod.node_group is None:
        return
    ng = mod.node_group
    cols = assets.ensure_assets()
    camera = p.camera or (scene.camera if scene else None)
    values = {
        "Terrain": p.terrain,
        "Clearing": clearing_collection(),
        "Pines": cols["PINES"],
        "Leafy Trees": cols["LEAFY_WINTER"] if p.season == 'WINTER' else cols["LEAFY_SUMMER"],
        "Shrubs": cols["SHRUBS"],
        "Pines per Hectare": p.pines_per_ha,
        "Leafy Trees per Hectare": p.leafy_per_ha,
        "Shrubs per Hectare": p.shrubs_per_ha,
        "Road Clearance": p.road_clearance,
        "Seed": p.seed,
        "Viewport Amount": p.viewport_amount,
        "Camera": camera,
        "Camera Culling": p.camera_culling and camera is not None,
        "View Distance": p.view_distance,
        "View Angle": p.view_angle,
    }
    for name, value in values.items():
        set_input(mod, socket_id(ng, name), value)
    forest.update_tag()


def _input_slot(mod, key):
    """Blender 5.x keeps each modifier input as an object with a .value under properties.inputs."""
    inputs = getattr(getattr(mod, "properties", None), "inputs", None)
    if inputs is not None and hasattr(inputs, key):
        return getattr(inputs, key)
    return None


def set_input(mod, key, value):
    slot = _input_slot(mod, key)
    if slot is not None:
        if slot.value != value:
            slot.value = value
    elif mod.get(key) != value:
        mod[key] = value


def get_input(mod, key):
    slot = _input_slot(mod, key)
    return slot.value if slot is not None else mod.get(key)


def grow_forest(context, terrain):
    """Creates (or refreshes) the forest object for a terrain."""
    assets.ensure_assets()
    ng = build_node_group()
    clearing_collection()
    sync_clearing(context.scene)
    forest = find_forest(terrain)
    if forest is None:
        me = bpy.data.meshes.new("TerraFlora Forest")
        forest = bpy.data.objects.new("TerraFlora Forest", me)
        context.scene.collection.objects.link(forest)
        forest.terraflora.is_forest = True
        forest.terraflora.terrain = terrain
    mod = forest.modifiers.get(MOD_NAME) or forest.modifiers.new(MOD_NAME, 'NODES')
    mod.node_group = ng
    sync_modifier(forest, context.scene)
    return forest
