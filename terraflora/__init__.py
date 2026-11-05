# SPDX-License-Identifier: GPL-3.0-or-later
"""
TerraFlora: forests, shrubs and clearings grown from real aerial photos.

Works on terrains made by CivilFlow (Real-World Terrain): reads where the trees, grass and water
are from the photo's colour and infrared bands plus USGS land cover, then grows low-poly pines,
leafy trees and shrubs there with Geometry Nodes instancing. Paint to remove or add plants;
CivilFlow roads clear their own space automatically.
"""

bl_info = {
    "name": "TerraFlora: Forests from Aerial Photos",
    "author": "AI Assistant",
    "version": (0, 1, 0),
    "blender": (5, 1, 0),
    "location": "View3D > Sidebar > TerraFlora",
    "description": "Grows trees and shrubs where the aerial photo shows them, with paintable clearings.",
    "category": "Object",
}

import bpy

from . import assets, forest, landcover


def _sync(self, context):
    ob = self.id_data
    if self.is_forest and ob.modifiers.get(forest.MOD_NAME):
        forest.sync_modifier(ob, context.scene if context else None)


class TF_PG_Forest(bpy.types.PropertyGroup):
    is_forest: bpy.props.BoolProperty(default=False)
    terrain: bpy.props.PointerProperty(type=bpy.types.Object, name="Terrain", update=_sync)
    season: bpy.props.EnumProperty(
        name="Season", default='SUMMER', update=_sync,
        items=[('SUMMER', "Summer", "Leafy trees in leaf"), ('WINTER', "Winter", "Leafy trees bare; pines stay green")])
    pines_per_ha: bpy.props.FloatProperty(name="Pines", default=300.0, min=0.0, max=3000.0, update=_sync,
                                          description="Pines per hectare (10,000 m²) where the forest is full")
    leafy_per_ha: bpy.props.FloatProperty(name="Leafy Trees", default=260.0, min=0.0, max=3000.0, update=_sync,
                                          description="Leafy trees per hectare where the forest is full")
    shrubs_per_ha: bpy.props.FloatProperty(name="Shrubs", default=250.0, min=0.0, max=5000.0, update=_sync,
                                           description="Shrubs per hectare where they grow fully")
    road_clearance: bpy.props.FloatProperty(name="Road Clearance", default=3.0, min=0.0, max=100.0, unit='LENGTH',
                                            update=_sync, description="Distance kept clear around CivilFlow roads")
    seed: bpy.props.IntProperty(name="Seed", default=0, min=0, update=_sync,
                                description="Change for a different random layout")
    viewport_amount: bpy.props.FloatProperty(name="Viewport Amount", default=0.35, min=0.0, max=1.0, subtype='FACTOR',
                                             update=_sync,
                                             description="Share of the plants shown while you work; renders show all")
    camera_culling: bpy.props.BoolProperty(name="Only What the Camera Sees", default=False, update=_sync,
                                           description="Grow only plants inside the camera's view, for big areas")
    camera: bpy.props.PointerProperty(type=bpy.types.Object, name="Camera", update=_sync,
                                      poll=lambda self, ob: ob.type == 'CAMERA',
                                      description="Camera to cull for (the scene camera when empty)")
    view_distance: bpy.props.FloatProperty(name="View Distance", default=1500.0, min=10.0, max=50000.0,
                                           unit='LENGTH', update=_sync)
    view_angle: bpy.props.FloatProperty(name="View Angle", default=110.0, min=10.0, max=180.0, update=_sync,
                                        description="Width of the cone kept around the camera's view, in degrees")


class TF_PG_Scene(bpy.types.PropertyGroup):
    terrain: bpy.props.PointerProperty(
        type=bpy.types.Object, name="Terrain",
        poll=lambda self, ob: ob.type == 'MESH' and bool(ob.get("cf_terrain")),
        description="CivilFlow terrain to grow on")


def scene_terrain(context):
    s = context.scene.terraflora
    if s.terrain:
        return s.terrain
    ob = context.active_object
    if ob is not None and ob.get("cf_terrain"):
        return ob
    if ob is not None and getattr(ob, "terraflora", None) and ob.terraflora.is_forest:
        return ob.terraflora.terrain
    return next((o for o in context.scene.objects if o.get("cf_terrain")), None)


class TF_OT_ReadLandCover(bpy.types.Operator):
    bl_idname = "terraflora.read_land_cover"
    bl_label = "Read Land Cover from Photo"
    bl_description = ("Work out where trees, grass and water are, from the aerial photo's colour and "
                      "infrared and USGS land cover (needs internet)")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        terrain = scene_terrain(context)
        if terrain is None:
            self.report({'ERROR'}, "Download a terrain with CivilFlow (Real-World Terrain) first")
            return {'CANCELLED'}
        if not getattr(bpy.app, "online_access", True):
            self.report({'ERROR'}, "Online access is off: Preferences > System > Network > Allow Online Access")
            return {'CANCELLED'}
        try:
            context.window.cursor_modal_set('WAIT')
        except AttributeError:
            pass
        try:
            landcover.read_land_cover(context, terrain)
        except Exception as e:
            self.report({'ERROR'}, "Land cover failed: %s" % e)
            return {'CANCELLED'}
        finally:
            try:
                context.window.cursor_modal_restore()
            except AttributeError:
                pass
        context.scene.terraflora.terrain = terrain
        if terrain.get("tf_note"):
            self.report({'WARNING'}, terrain["tf_note"])
        else:
            self.report({'INFO'}, terrain["tf_summary"])
        return {'FINISHED'}


class TF_OT_GrowForest(bpy.types.Operator):
    bl_idname = "terraflora.grow_forest"
    bl_label = "Grow Forest"
    bl_description = "Grow trees and shrubs over the terrain from its land cover and your painting"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        terrain = scene_terrain(context)
        if terrain is None:
            self.report({'ERROR'}, "Pick a terrain first")
            return {'CANCELLED'}
        if "tf_pines" not in terrain.data.attributes:
            self.report({'WARNING'}, "No land cover read yet: only painted plants will grow")
        f = forest.grow_forest(context, terrain)
        context.scene.terraflora.terrain = terrain
        self.report({'INFO'}, "Forest ready: %s" % f.name)
        return {'FINISHED'}


class TF_OT_ShowCover(bpy.types.Operator):
    bl_idname = "terraflora.show_cover"
    bl_label = "Show Cover Map"
    bl_description = ("Colour the terrain by what grows there (dark green pines, light green leafy trees, "
                      "yellow grass, blue water, grey nothing). Click again for the photo")
    bl_options = {'REGISTER'}

    def execute(self, context):
        terrain = scene_terrain(context)
        if terrain is None or landcover.COVER_ATTR not in terrain.data.color_attributes:
            self.report({'WARNING'}, "Read the land cover first")
            return {'CANCELLED'}
        ca = terrain.data.color_attributes
        ca.active_color = ca[landcover.COVER_ATTR]
        showing = None
        for area in context.screen.areas if context.screen else []:
            if area.type == 'VIEW_3D':
                sh = area.spaces.active.shading
                if showing is None:
                    showing = sh.color_type == 'VERTEX'
                sh.type = 'SOLID' if sh.type == 'WIREFRAME' else sh.type
                sh.color_type = 'TEXTURE' if showing else 'VERTEX'
        return {'FINISHED'}


class TF_OT_Paint(bpy.types.Operator):
    bl_idname = "terraflora.paint"
    bl_label = "Paint Plants"
    bl_description = "Paint on the terrain to change what grows there"
    bl_options = {'REGISTER', 'UNDO'}

    layer: bpy.props.EnumProperty(items=[(k, v[1], v[2], v[3], i) for i, (k, v) in enumerate(forest.PAINT_LAYERS.items())])

    def execute(self, context):
        terrain = scene_terrain(context)
        if terrain is None:
            self.report({'ERROR'}, "Pick a terrain first")
            return {'CANCELLED'}
        name = forest.PAINT_LAYERS[self.layer][0]
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        for ob in context.selected_objects:
            ob.select_set(False)
        terrain.hide_set(False)
        terrain.select_set(True)
        context.view_layer.objects.active = terrain
        vg = terrain.vertex_groups.get(name) or terrain.vertex_groups.new(name=name)
        terrain.vertex_groups.active_index = vg.index
        bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
        try:
            context.tool_settings.weight_paint.unified_paint_settings.weight = 1.0
        except AttributeError:
            pass
        self.report({'INFO'}, "%s: paint on the terrain. Hold Ctrl to erase. Done Painting when finished"
                    % forest.PAINT_LAYERS[self.layer][1])
        return {'FINISHED'}


class TF_OT_PaintDone(bpy.types.Operator):
    bl_idname = "terraflora.paint_done"
    bl_label = "Done Painting"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        return {'FINISHED'}


class TF_OT_ClearPaint(bpy.types.Operator):
    bl_idname = "terraflora.clear_paint"
    bl_label = "Undo All My Painting"
    bl_description = "Remove everything you painted, so the photo decides again everywhere"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        terrain = scene_terrain(context)
        if terrain is None:
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        for name, *_ in forest.PAINT_LAYERS.values():
            vg = terrain.vertex_groups.get(name)
            if vg:
                terrain.vertex_groups.remove(vg)
        terrain.data.update()
        return {'FINISHED'}


class TF_PT_Main(bpy.types.Panel):
    bl_label = "TerraFlora"
    bl_idname = "TF_PT_Main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TerraFlora'

    def draw(self, context):
        layout = self.layout
        s = context.scene.terraflora
        terrain = scene_terrain(context)

        box = layout.box()
        box.label(text="1. Land Cover", icon='WORLD')
        box.prop(s, "terrain")
        if terrain is None:
            box.label(text="Download a terrain in CivilEng > Real-World Terrain", icon='INFO')
            return
        box.operator("terraflora.read_land_cover", icon='IMAGE_DATA')
        if terrain.get("tf_summary"):
            box.label(text=str(terrain["tf_summary"]))
            box.operator("terraflora.show_cover", icon='COLOR')

        box = layout.box()
        box.label(text="2. Forest", icon='OUTLINER_OB_FORCE_FIELD')
        f = forest.find_forest(terrain)
        box.operator("terraflora.grow_forest", text="Grow Forest" if f is None else "Refresh Forest", icon='PLAY')
        if f is not None:
            p = f.terraflora
            box.prop(p, "season", expand=True)
            col = box.column(align=True)
            col.label(text="Per hectare in full cover:")
            col.prop(p, "pines_per_ha")
            col.prop(p, "leafy_per_ha")
            col.prop(p, "shrubs_per_ha")
            col = box.column(align=True)
            col.prop(p, "road_clearance")
            col.prop(p, "seed")
            col.prop(p, "viewport_amount")
            box.prop(p, "camera_culling")
            if p.camera_culling:
                col = box.column(align=True)
                col.prop(p, "camera")
                col.prop(p, "view_distance")
                col.prop(p, "view_angle")

        box = layout.box()
        box.label(text="3. Paint", icon='BRUSH_DATA')
        if context.mode == 'PAINT_WEIGHT':
            vg = terrain.vertex_groups.active
            box.label(text="Painting: %s" % (vg.name.replace("TF ", "") if vg else "?"), icon='BRUSH_DATA')
            box.label(text="Hold Ctrl to erase · F to resize the brush")
            box.operator("terraflora.paint_done", icon='CHECKMARK')
        grid = box.grid_flow(columns=2, align=True)
        for key, (name, label, desc, icon) in forest.PAINT_LAYERS.items():
            op = grid.operator("terraflora.paint", text=label, icon=icon)
            op.layer = key
        box.operator("terraflora.clear_paint", icon='LOOP_BACK')


@bpy.app.handlers.persistent
def _keep_roads_clear(scene, depsgraph):
    """New CivilFlow roads join the clearing collection, so trees make room for them."""
    try:
        forest.sync_clearing(scene)
    except (ReferenceError, RuntimeError):
        pass


classes = (
    TF_PG_Forest,
    TF_PG_Scene,
    TF_OT_ReadLandCover,
    TF_OT_GrowForest,
    TF_OT_ShowCover,
    TF_OT_Paint,
    TF_OT_PaintDone,
    TF_OT_ClearPaint,
    TF_PT_Main,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.terraflora = bpy.props.PointerProperty(type=TF_PG_Forest)
    bpy.types.Scene.terraflora = bpy.props.PointerProperty(type=TF_PG_Scene)
    if _keep_roads_clear not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_keep_roads_clear)


def unregister():
    if _keep_roads_clear in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_keep_roads_clear)
    del bpy.types.Scene.terraflora
    del bpy.types.Object.terraflora
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
