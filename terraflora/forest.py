# SPDX-License-Identifier: GPL-3.0-or-later
"""
The forest: one object whose Geometry Nodes modifier scatters the plant library over a terrain.

Per layer (pines, leafy trees, shrubs) the density at each spot is

    clamp(auto * (1 - painted removal) + painted addition) * (1 - water)

where 'auto' is the land cover read from the photo (tf_pines / tf_leafy / tf_shrubs / tf_grass)
and the painted layers are vertex groups you paint in Weight Paint. Plants are spaced with
Poisson-disk sampling, kept clear of roads (anything in the TerraFlora Clearing collection,
measured on the map so bridges and banks clear too), optionally culled to what the camera sees,
and placed as instances (shared meshes), so even 100,000 trees stay light.

Level of detail: trees and shrubs farther than Card Distance from the camera are drawn as cards
(pictures of the same model, 6 triangles); grass tufts only grow within Grass Distance of it.
"""

import bpy

from . import assets

NODE_GROUP = "TF_Forest"
NODE_VERSION = 3
CLEARING = "TerraFlora Clearing"
MOD_NAME = "TerraFlora"

PAINT_LAYERS = {
    'REMOVE': ("TF Remove Trees", "Remove Trees", "Clear trees and shrubs where you paint", 'X'),
    'PINES': ("TF Add Pines", "Add Pines", "Plant pines where you paint", 'OUTLINER_OB_FORCE_FIELD'),
    'LEAFY': ("TF Add Leafy Trees", "Add Leafy Trees", "Plant leafy trees where you paint", 'OUTLINER_OB_POINTCLOUD'),
    'SHRUBS': ("TF Add Shrubs", "Add Shrubs", "Plant shrubs where you paint", 'SEQ_HISTOGRAM'),
    'GRASS': ("TF Add Grass", "Add Grass", "Grow grass tufts where you paint", 'STROKE'),
    'NO_GRASS': ("TF Remove Grass", "Remove Grass", "Clear grass tufts where you paint", 'PANEL_CLOSE'),
}
GRASS_CLEARANCE = 0.6   # grass grows right up to the road shoulder

INPUTS = [
    # name, socket type, default, min, max, description
    ("Terrain", 'NodeSocketObject', None, None, None, "Ground to grow on"),
    ("Clearing", 'NodeSocketCollection', None, None, None, "Roads and anything else to keep clear"),
    ("Pines", 'NodeSocketCollection', None, None, None, ""),
    ("Leafy Trees", 'NodeSocketCollection', None, None, None, ""),
    ("Shrubs", 'NodeSocketCollection', None, None, None, ""),
    ("Grass", 'NodeSocketCollection', None, None, None, ""),
    ("Pine Cards", 'NodeSocketCollection', None, None, None, ""),
    ("Leafy Cards", 'NodeSocketCollection', None, None, None, ""),
    ("Shrub Cards", 'NodeSocketCollection', None, None, None, ""),
    ("Pines per Hectare", 'NodeSocketFloat', 300.0, 0.0, 3000.0, "Pines in full forest, per 10,000 m²"),
    ("Leafy Trees per Hectare", 'NodeSocketFloat', 260.0, 0.0, 3000.0, "Leafy trees in full forest, per 10,000 m²"),
    ("Shrubs per Hectare", 'NodeSocketFloat', 250.0, 0.0, 5000.0, "Shrubs where they grow fully, per 10,000 m²"),
    ("Grass per Hectare", 'NodeSocketFloat', 4000.0, 0.0, 12000.0, "Grass tufts in full grass, per 10,000 m²"),
    ("Road Clearance", 'NodeSocketFloat', 3.0, 0.0, 100.0, "Distance kept clear around roads"),
    ("Seed", 'NodeSocketInt', 0, 0, 100000, "Change for a different random layout"),
    ("Viewport Amount", 'NodeSocketFloat', 0.35, 0.0, 1.0, "Share of the plants shown in the viewport (renders show all)"),
    ("Camera", 'NodeSocketObject', None, None, None, "Camera for culling"),
    ("Camera Culling", 'NodeSocketBool', False, None, None, "Only grow what the camera can see"),
    ("View Distance", 'NodeSocketFloat', 1500.0, 10.0, 50000.0, "Furthest distance grown when culling"),
    ("View Angle", 'NodeSocketFloat', 110.0, 10.0, 180.0, "Width of the cone kept around the camera's view, in degrees"),
    ("Has Camera", 'NodeSocketBool', False, None, None, "A camera is set, so distances can be measured"),
    ("Use Cards", 'NodeSocketBool', True, None, None, "Draw distant trees as cards"),
    ("Card Distance", 'NodeSocketFloat', 300.0, 0.0, 50000.0, "Trees farther than this from the camera are cards"),
    ("Viewport Cards", 'NodeSocketBool', False, None, None, "Every tree is a card in the viewport"),
    ("Grass Distance", 'NodeSocketFloat', 200.0, 1.0, 50000.0, "Grass grows only this close to the camera"),
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

    water = b.attr("tf_water", -1500, 750)
    dry = b.math('SUBTRACT', 1.0, water, -1300, 750)
    rem_trees = b.math('SUBTRACT', 1.0, b.attr("TF Remove Trees", -1500, 900), -1300, 900)
    rem_grass = b.math('SUBTRACT', 1.0, b.attr("TF Remove Grass", -1500, 1050), -1300, 1050)
    grass_on = b.compare('GREATER_THAN', I["Grass per Hectare"], 0.0, -1300, 1200)
    # Cards: by distance from the camera, or everywhere in the viewport if asked.
    cards_by_dist = b.boolean('AND', I["Use Cards"], I["Has Camera"], -1300, 1350)
    cards_vp = b.boolean('AND', I["Viewport Cards"], isvp.outputs[0], -1300, 1500)

    layers = (
        # land cover, painted add, kept share, per ha, models, cards, spacing, seed, clearance
        ("tf_pines", "TF Add Pines", rem_trees, "Pines per Hectare", "Pines", "Pine Cards", 4.5, 0, None),
        ("tf_leafy", "TF Add Leafy Trees", rem_trees, "Leafy Trees per Hectare", "Leafy Trees", "Leafy Cards", 4.5, 100, None),
        ("tf_shrubs", "TF Add Shrubs", rem_trees, "Shrubs per Hectare", "Shrubs", "Shrub Cards", 2.0, 200, None),
        ("tf_grass", "TF Add Grass", rem_grass, "Grass per Hectare", "Grass", None, 0.5, 300, GRASS_CLEARANCE),
    )
    join = b.node('GeometryNodeJoinGeometry', 3400, 0)
    for k, (auto_name, add_name, keep, per_ha, coll, cards, spacing, seed_off, clearance) in enumerate(layers):
        y = 2600 - k * 1300
        x = -1100
        auto = b.attr(auto_name, x, y)
        add = b.attr(add_name, x, y - 150)
        d = b.math('MULTIPLY', auto, keep, x + 200, y)
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
        if coll == "Grass":
            # Grass is skipped entirely when its density is 0 (no work at all).
            sw = b.node('GeometryNodeSwitch', x + 1000, y + 150, input_type='GEOMETRY')
            b.put(sw.inputs["Switch"], grass_on)
            b.put(sw.inputs["True"], pts)
            pts = sw.outputs[0]

        # Keep off roads.
        ppos = b.node('GeometryNodeInputPosition', x + 800, y - 300)
        pflat = b.vmath('MULTIPLY', ppos.outputs[0], (1.0, 1.0, 0.0), x + 1000, y - 300)
        prox = b.node('GeometryNodeProximity', x + 1200, y - 250, target_element='FACES')
        b.put(prox.inputs["Geometry"], clear_geo)
        b.put(prox.inputs["Sample Position"], pflat)
        near = b.compare('LESS_THAN', prox.outputs["Distance"],
                         clearance if clearance is not None else I["Road Clearance"], x + 1400, y - 250)
        near = b.boolean('AND', near, has_clear, x + 1600, y - 250)
        dele = b.node('GeometryNodeDeleteGeometry', x + 1800, y, domain='POINT')
        b.put(dele.inputs["Geometry"], pts)
        b.put(dele.inputs["Selection"], near)
        pts = dele.outputs[0]

        # Camera culling (and, for grass, only near the camera).
        rel = b.vmath('SUBTRACT', ppos.outputs[0], cam.outputs["Location"], x + 1000, y - 550)
        length = b.vmath('LENGTH', rel, x=x + 1200, y=y - 500, out=1)
        dirn = b.vmath('NORMALIZE', rel, x=x + 1200, y=y - 650)
        dot = b.vmath('DOT_PRODUCT', dirn, rot.outputs[0], x + 1400, y - 650, out=1)
        in_cone = b.compare('GREATER_THAN', dot, cos_half, x + 1600, y - 650)
        in_range = b.compare('LESS_THAN', length, I["View Distance"], x + 1600, y - 500)
        visible = b.boolean('AND', in_cone, in_range, x + 1800, y - 550)
        hide = b.boolean('NIMPLY', I["Camera Culling"], visible, x + 2000, y - 550)
        if coll == "Grass":
            too_far = b.compare('GREATER_THAN', length, I["Grass Distance"], x + 1800, y - 800)
            too_far = b.boolean('AND', too_far, I["Has Camera"], x + 2000, y - 800)
            hide = b.boolean('OR', hide, too_far, x + 2200, y - 650)
        cull = b.node('GeometryNodeDeleteGeometry', x + 2000, y, domain='POINT')
        b.put(cull.inputs["Geometry"], pts)
        b.put(cull.inputs["Selection"], hide)
        pts = cull.outputs[0]

        # Each plant's random model, turn, size and shade, kept on the point so the near model and
        # the far card of the same plant match.
        for name, dtype, lo, hi, off in (("tf_pick", 'INT', 0, 1000, 1), ("tf_turn", 'FLOAT', 0.0, 6.2832, 2),
                                         ("tf_size", 'FLOAT', 0.75, 1.3, 3), ("tf_tint", 'FLOAT', 0.0, 1.0, 4)):
            st = b.node('GeometryNodeStoreNamedAttribute', x + 2200 + off * 150, y, data_type=dtype, domain='POINT')
            b.put(st.inputs["Geometry"], pts)
            st.inputs["Name"].default_value = name
            b.put(st.inputs["Value"], b.random(dtype, lo, hi, seed_off + off, x + 2200 + off * 150, y - 250))
            pts = st.outputs[0]

        def plant(points, collection, xx, yy):
            cinst = b.node('GeometryNodeCollectionInfo', xx, yy + 300, transform_space='ORIGINAL')
            b.put(cinst.inputs["Collection"], I[collection])
            cinst.inputs["Separate Children"].default_value = True
            cinst.inputs["Reset Children"].default_value = True
            pick = b.node('GeometryNodeInputNamedAttribute', xx, yy - 200, data_type='INT')
            pick.inputs["Name"].default_value = "tf_pick"
            e = b.node('ShaderNodeCombineXYZ', xx + 200, yy - 350)
            b.put(e.inputs["Z"], b.attr("tf_turn", xx, yy - 350))
            r = b.node('FunctionNodeEulerToRotation', xx + 400, yy - 350)
            b.put(r.inputs[0], e.outputs[0])
            sc = b.attr("tf_size", xx, yy - 500)
            s3 = b.node('ShaderNodeCombineXYZ', xx + 200, yy - 500)
            for c in ("X", "Y", "Z"):
                b.put(s3.inputs[c], sc)
            iop = b.node('GeometryNodeInstanceOnPoints', xx + 600, yy)
            b.put(iop.inputs["Points"], points)
            b.put(iop.inputs["Instance"], cinst.outputs[0])
            iop.inputs["Pick Instance"].default_value = True
            b.put(iop.inputs["Instance Index"], pick.outputs["Attribute"])
            b.put(iop.inputs["Rotation"], r.outputs[0])
            b.put(iop.inputs["Scale"], s3.outputs[0])
            ng.links.new(iop.outputs[0], join.inputs[0])

        if cards is None:
            plant(pts, coll, x + 3000, y)
            continue
        far = b.compare('GREATER_THAN', length, I["Card Distance"], x + 2800, y - 700)
        far = b.boolean('AND', far, cards_by_dist, x + 3000, y - 700)
        far = b.boolean('OR', far, cards_vp, x + 3200, y - 700)
        sep = b.node('GeometryNodeSeparateGeometry', x + 3000, y, domain='POINT')
        b.put(sep.inputs["Geometry"], pts)
        b.put(sep.inputs["Selection"], far)
        plant(sep.outputs["Inverted"], coll, x + 3400, y + 200)
        plant(sep.outputs["Selection"], cards, x + 3400, y - 500)
    ng.links.new(join.outputs[0], gout.inputs[0])
    return ng


def clearing_collection():
    """Collection of everything plants keep away from: CivilFlow roads and lakes, kept up to date."""
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
        road = props is not None and props.is_civil_object
        lake = bool(ob.get("cf_water_of"))  # CivilFlow lakes and ponds
        if ob.type == 'MESH' and (road or lake) and ob.name not in col.objects:
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
        "Grass": cols["GRASS_WINTER"] if p.season == 'WINTER' else cols["GRASS_SUMMER"],
        "Pine Cards": cols["PINES_CARDS"],
        "Leafy Cards": cols["LEAFY_WINTER_CARDS"] if p.season == 'WINTER' else cols["LEAFY_SUMMER_CARDS"],
        "Shrub Cards": cols["SHRUBS_CARDS"],
        "Pines per Hectare": p.pines_per_ha,
        "Leafy Trees per Hectare": p.leafy_per_ha,
        "Shrubs per Hectare": p.shrubs_per_ha,
        "Grass per Hectare": p.grass_per_ha if p.grass else 0.0,
        "Road Clearance": p.road_clearance,
        "Seed": p.seed,
        "Viewport Amount": p.viewport_amount,
        "Camera": camera,
        "Camera Culling": p.camera_culling and camera is not None,
        "View Distance": p.view_distance,
        "View Angle": p.view_angle,
        "Has Camera": camera is not None,
        "Use Cards": p.use_cards,
        "Card Distance": p.card_distance,
        "Viewport Cards": p.viewport_cards,
        "Grass Distance": p.grass_distance,
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
