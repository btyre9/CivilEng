bl_info = {
    "name": "CivilFlow: Road, Highway & Overpass Generator",
    "author": "AI Assistant",
    "version": (1, 13, 1),
    "blender": (5, 1, 0),
    "location": "View3D > Sidebar (N) > CivilEng",
    "description": "Procedural roads built from straights and curves, highways, banking, guardrails, gantry signs, tunnels, embankments, cable-stayed towers, toll plazas, animated low-poly traffic, and real USGS terrain with automatic bridges, tunnels and earthworks.",
    "category": "Add Mesh",
}

import bpy
import bmesh
from mathutils import Vector, Matrix
import math

import numpy as np

from . import traffic
from . import terrain
from . import water
from . import utilities
from . import junctions
from . import sim

# ==============================================================================
# 1. PROCEDURAL MATERIALS HELPER (BLENDER 5.x API)
# ==============================================================================

def get_or_create_material(name, base_color, roughness=0.8, metallic=0.0, emission_color=None, emission_strength=0.0):
    """Creates or retrieves a PBR material using Blender 5.x node trees (no deprecated use_nodes)."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        if mat.node_tree:
            nodes = mat.node_tree.nodes
            bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
            if bsdf is None:
                bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
                output = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
                if output is None:
                    output = nodes.new(type='ShaderNodeOutputMaterial')
                mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

            if "Base Color" in bsdf.inputs:
                bsdf.inputs["Base Color"].default_value = base_color
            if "Roughness" in bsdf.inputs:
                bsdf.inputs["Roughness"].default_value = roughness
            if "Metallic" in bsdf.inputs:
                bsdf.inputs["Metallic"].default_value = metallic
            if emission_color and "Emission Color" in bsdf.inputs:
                bsdf.inputs["Emission Color"].default_value = emission_color
            if emission_strength > 0.0 and "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = emission_strength
        mat.diffuse_color = base_color
    return mat


def ensure_civil_materials(obj):
    """Ensures standard civil engineering materials are assigned to the object."""
    mats = [
        get_or_create_material("Civil_Asphalt", (0.08, 0.08, 0.09, 1.0), roughness=0.85),        # Slot 0: Asphalt
        get_or_create_material("Civil_Concrete", (0.55, 0.54, 0.52, 1.0), roughness=0.9),        # Slot 1: Concrete/Deck/Toll/Pylons
        get_or_create_material("Civil_Barrier", (0.65, 0.65, 0.63, 1.0), roughness=0.75),        # Slot 2: Jersey Barrier
        get_or_create_material("Civil_SoundWall", (0.45, 0.42, 0.38, 1.0), roughness=0.85),      # Slot 3: Sound Wall
        get_or_create_material("Civil_Line_Yellow", (0.85, 0.60, 0.02, 1.0), roughness=0.5),     # Slot 4: Yellow Line
        get_or_create_material("Civil_Line_White", (0.90, 0.90, 0.90, 1.0), roughness=0.5),      # Slot 5: White Line / Sign Border
        get_or_create_material("Civil_GalvanizedSteel", (0.62, 0.65, 0.68, 1.0), roughness=0.35, metallic=0.85), # Slot 6: Guardrail / Cables / Gantry
        get_or_create_material("Civil_Lamp_Emissive", (1.0, 0.92, 0.78, 1.0), roughness=0.2,
                               emission_color=(1.0, 0.90, 0.72, 1.0), emission_strength=18.0),   # Slot 7: Streetlight / LED
        get_or_create_material("Civil_Sign_Green", (0.01, 0.24, 0.12, 1.0), roughness=0.35, metallic=0.1), # Slot 8: Sign Face
        get_or_create_material("Civil_Embankment_Dirt", (0.32, 0.25, 0.18, 1.0), roughness=0.95),  # Slot 9: Earth Embankment Slopes
        get_or_create_material("Civil_LED_Amber", (1.0, 0.55, 0.05, 1.0), roughness=0.3,
                               emission_color=(1.0, 0.5, 0.02, 1.0), emission_strength=12.0),  # Slot 10: LED message text
        get_or_create_material("Civil_Sign_Cabinet", (0.025, 0.025, 0.028, 1.0), roughness=0.5),  # Slot 11: LED sign cabinet
        get_or_create_material("Civil_CastIron", (0.018, 0.018, 0.02, 1.0), roughness=0.4, metallic=0.6),  # Slot 12: painted cast iron
        get_or_create_material("Civil_Globe_Warm", (0.95, 0.88, 0.75, 1.0), roughness=0.3,
                               emission_color=(1.0, 0.8, 0.52, 1.0), emission_strength=10.0),  # Slot 13: warm glass globe
        get_or_create_material("Civil_Wood_Pole", (0.13, 0.085, 0.055, 1.0), roughness=0.9),     # Slot 14: creosoted wood
        get_or_create_material("Civil_Porcelain", (0.28, 0.11, 0.05, 1.0), roughness=0.2),        # Slot 15: glazed porcelain
        get_or_create_material("Civil_Wire", (0.03, 0.03, 0.03, 1.0), roughness=0.4, metallic=0.8),  # Slot 16: power wire
        get_or_create_material("Civil_Transformer", (0.42, 0.45, 0.46, 1.0), roughness=0.5, metallic=0.3),  # Slot 17
        get_or_create_material("Civil_Polymer", (0.55, 0.57, 0.58, 1.0), roughness=0.6),          # Slot 18: polymer insulators
        get_or_create_material("Civil_Grass", (0.09, 0.18, 0.05, 1.0), roughness=0.9),            # Slot 19: island grass
        get_or_create_material("Civil_Sign_Red", (0.5, 0.01, 0.01, 1.0), roughness=0.4),          # Slot 20: stop / yield signs
    ]
    obj.data.materials.clear()
    for m in mats:
        obj.data.materials.append(m)


# ==============================================================================
# 2. CURVE SAMPLING, TILT & SUPERELEVATION ENGINE
# ==============================================================================

def extract_ordered_polyline(mesh):
    """Walks the evaluated mesh edge chain so vertices are strictly ordered start-to-end."""
    num_verts = len(mesh.vertices)
    if num_verts < 2 or len(mesh.edges) == 0:
        return [v.co.copy() for v in mesh.vertices]

    adj = {i: [] for i in range(num_verts)}
    for e in mesh.edges:
        v0, v1 = e.vertices[0], e.vertices[1]
        adj[v0].append(v1)
        adj[v1].append(v0)

    start_v = 0
    for v_idx, neighbors in adj.items():
        if len(neighbors) == 1:
            start_v = v_idx
            break

    ordered = []
    visited = set()
    curr = start_v
    while curr is not None and curr not in visited:
        visited.add(curr)
        ordered.append(mesh.vertices[curr].co.copy())
        next_v = None
        for nb in adj[curr]:
            if nb not in visited:
                next_v = nb
                break
        curr = next_v

    return ordered if len(ordered) >= 2 else [v.co.copy() for v in mesh.vertices]


def extract_spline_control_tilts(curve_obj):
    """Extracts world-space control point coordinates and their tilt values (radians)."""
    mat_w = curve_obj.matrix_world
    if not curve_obj.data.splines:
        return []
    spline = curve_obj.data.splines[0]
    pts = []
    if spline.type == 'BEZIER':
        for bp in spline.bezier_points:
            pts.append((mat_w @ bp.co, bp.tilt))
    else:
        for pt in spline.points:
            pts.append((mat_w @ Vector((pt.co.x, pt.co.y, pt.co.z)), pt.tilt))
    return pts


def control_point_arc_fractions(curve_obj, ctrl_pts, verts, cum_lengths):
    """
    Where each control point sits along the evaluated curve, as a 0..1 fraction of its true
    arc length. A non-cyclic Bezier spline evaluates to resolution_u points per segment, so
    control point k is evaluated vertex k * resolution_u. Other spline types fall back to
    spacing the control points by the straight-line distance between them.
    """
    total = max(1e-6, cum_lengths[-1])
    spline = curve_obj.data.splines[0]
    n = len(ctrl_pts)
    if spline.type == 'BEZIER' and not spline.use_cyclic_u:
        res = spline.resolution_u
        if len(verts) == (n - 1) * res + 1:
            # The polyline walk may start from either end of the spline.
            reverse = (verts[0] - ctrl_pts[0][0]).length > (verts[0] - ctrl_pts[-1][0]).length
            if reverse:
                return [cum_lengths[(n - 1 - k) * res] / total for k in range(n)]
            return [cum_lengths[k * res] / total for k in range(n)]

    c_dists = [0.0]
    for i in range(1, n):
        c_dists.append(c_dists[-1] + (ctrl_pts[i][0] - ctrl_pts[i - 1][0]).length)
    c_total = max(1e-6, c_dists[-1])
    return [d / c_total for d in c_dists]


def sample_bezier_curve(curve_obj, step_distance, use_spline_tilt=True, auto_bank=False, auto_bank_factor=1.0, max_bank_deg=12.0, depsgraph=None):
    """
    Samples a Blender Curve object in world space and applies spline point tilt (Ctrl+T)
    and optional curvature-based superelevation banking.
    Returns list of (world_pos, world_tangent, world_normal, world_right).
    """
    if depsgraph is None:
        depsgraph = bpy.context.evaluated_depsgraph_get()

    eval_obj = curve_obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()

    if len(mesh.vertices) < 2:
        eval_obj.to_mesh_clear()
        return []

    local_verts = extract_ordered_polyline(mesh)
    eval_obj.to_mesh_clear()

    mat_world = curve_obj.matrix_world
    verts = [mat_world @ v for v in local_verts]

    cum_lengths = [0.0]
    for i in range(1, len(verts)):
        dist = (verts[i] - verts[i - 1]).length
        cum_lengths.append(cum_lengths[-1] + dist)

    total_length = cum_lengths[-1]
    if total_length < 0.001:
        return []

    ctrl_pts = extract_spline_control_tilts(curve_obj) if use_spline_tilt else []
    ctrl_arc_tilts = []
    if len(ctrl_pts) >= 2:
        ctrl_fracs = control_point_arc_fractions(curve_obj, ctrl_pts, verts, cum_lengths)
        for frac, (_, tilt_val) in zip(ctrl_fracs, ctrl_pts):
            ctrl_arc_tilts.append((frac, tilt_val))
        ctrl_arc_tilts.sort(key=lambda ft: ft[0])

    def eval_ctrl_tilt(u_norm):
        if not ctrl_arc_tilts:
            return 0.0
        if u_norm <= ctrl_arc_tilts[0][0]:
            return ctrl_arc_tilts[0][1]
        if u_norm >= ctrl_arc_tilts[-1][0]:
            return ctrl_arc_tilts[-1][1]
        for k in range(len(ctrl_arc_tilts) - 1):
            u0, t0 = ctrl_arc_tilts[k]
            u1, t1 = ctrl_arc_tilts[k + 1]
            if u0 <= u_norm <= u1:
                fac = (u_norm - u0) / max(1e-6, (u1 - u0))
                fac = fac * fac * (3.0 - 2.0 * fac)
                return t0 + (t1 - t0) * fac
        return 0.0

    num_samples = max(2, int(total_length / max(0.2, step_distance)) + 1)
    sample_dists = [i * (total_length / (num_samples - 1)) for i in range(num_samples)]

    raw_positions = []
    raw_tangents = []
    idx = 0

    for d in sample_dists:
        while idx < len(cum_lengths) - 2 and cum_lengths[idx + 1] < d:
            idx += 1

        seg_len = cum_lengths[idx + 1] - cum_lengths[idx]
        t = (d - cum_lengths[idx]) / seg_len if seg_len > 1e-6 else 0.0
        pos = verts[idx].lerp(verts[idx + 1], t)

        tangent = (verts[idx + 1] - verts[idx]).normalized()
        if tangent.length < 1e-6:
            tangent = Vector((0.0, 1.0, 0.0))

        raw_positions.append(pos)
        raw_tangents.append(tangent)

    smooth_tangents = []
    for i in range(len(raw_tangents)):
        if 0 < i < len(raw_tangents) - 1:
            t_s = (raw_tangents[i - 1] + raw_tangents[i] * 2.0 + raw_tangents[i + 1]).normalized()
            smooth_tangents.append(t_s)
        else:
            smooth_tangents.append(raw_tangents[i])

    max_bank_rad = math.radians(max_bank_deg)
    auto_bank_angles = [0.0] * len(raw_positions)
    if auto_bank and len(raw_positions) >= 3:
        raw_auto = [0.0] * len(raw_positions)
        for i in range(1, len(raw_positions) - 1):
            t_prev = Vector((smooth_tangents[i - 1].x, smooth_tangents[i - 1].y, 0.0)).normalized()
            t_next = Vector((smooth_tangents[i + 1].x, smooth_tangents[i + 1].y, 0.0)).normalized()
            ds = (raw_positions[i + 1] - raw_positions[i - 1]).length
            if ds > 1e-4 and t_prev.length > 1e-4 and t_next.length > 1e-4:
                cross_z = t_prev.x * t_next.y - t_prev.y * t_next.x
                curvature = cross_z / ds
                bank_angle = max(-max_bank_rad, min(max_bank_rad, -curvature * 12.0 * auto_bank_factor))
                raw_auto[i] = bank_angle
        raw_auto[0] = raw_auto[1]
        raw_auto[-1] = raw_auto[-2]

        for i in range(len(raw_auto)):
            lo = max(0, i - 2)
            hi = min(len(raw_auto), i + 3)
            auto_bank_angles[i] = sum(raw_auto[lo:hi]) / (hi - lo)

    up = Vector((0.0, 0.0, 1.0))
    samples = []
    for i in range(len(raw_positions)):
        pos = raw_positions[i]
        tangent = smooth_tangents[i]
        u_norm = sample_dists[i] / max(1e-6, total_length)

        right = tangent.cross(up).normalized()
        if right.length < 1e-6:
            right = Vector((1.0, 0.0, 0.0))
        normal = right.cross(tangent).normalized()

        total_tilt = (eval_ctrl_tilt(u_norm) if use_spline_tilt else 0.0) + auto_bank_angles[i]
        if abs(total_tilt) > 1e-5:
            rot_mat = Matrix.Rotation(total_tilt, 3, tangent)
            right = (rot_mat @ right).normalized()
            normal = (rot_mat @ normal).normalized()

        samples.append((pos, tangent, normal, right))

    return samples


# ==============================================================================
# 3. MESH EXTRUSION & CIVIL ENGINEERING COMPONENT GENERATORS
# ==============================================================================

def add_extruded_profile(bm, samples, profile_2d, mat_index, cap_ends=True, dashed=False, dash_interval=3, x_offsets=None):
    """Extrudes a closed 2D profile [(x, z), ...] along sampled 3D curve frames."""
    num_pts = len(profile_2d)
    if num_pts < 3 or len(samples) < 2:
        return

    rings = []
    for s_idx, (pos, tangent, normal, right) in enumerate(samples):
        dx = x_offsets[s_idx] if (x_offsets and s_idx < len(x_offsets)) else 0.0
        ring_verts = []
        for x, z in profile_2d:
            world_pt = pos + (right * (x + dx)) + (normal * z)
            ring_verts.append(bm.verts.new(world_pt))
        rings.append(ring_verts)

    for i in range(len(rings) - 1):
        if dashed and (i % dash_interval != 0):
            continue

        r1 = rings[i]
        r2 = rings[i + 1]
        for j in range(num_pts):
            j_next = (j + 1) % num_pts
            try:
                f = bm.faces.new((r1[j], r1[j_next], r2[j_next], r2[j]))
                f.material_index = mat_index
            except ValueError:
                pass

        if dashed:
            try:
                f_start = bm.faces.new(reversed(r1))
                f_start.material_index = mat_index
                f_end = bm.faces.new(r2)
                f_end.material_index = mat_index
            except ValueError:
                pass

    if cap_ends and not dashed:
        try:
            f_start = bm.faces.new(reversed(rings[0]))
            f_start.material_index = mat_index
            f_end = bm.faces.new(rings[-1])
            f_end.material_index = mat_index
        except ValueError:
            pass


def add_variable_extrusion(bm, samples, profile_fn, mat_index, cap_ends=True):
    """Extrudes a 2D profile where each sample index generates its own vertex layout."""
    if len(samples) < 2:
        return

    rings = []
    num_pts = 0
    for i, (pos, tangent, normal, right) in enumerate(samples):
        prof = profile_fn(i)
        if not prof or len(prof) < 3:
            continue
        num_pts = len(prof)
        ring_verts = [bm.verts.new(pos + right * x + normal * z) for (x, z) in prof]
        rings.append(ring_verts)

    if len(rings) < 2 or num_pts < 3:
        return

    for i in range(len(rings) - 1):
        r1, r2 = rings[i], rings[i + 1]
        for j in range(num_pts):
            jn = (j + 1) % num_pts
            try:
                f = bm.faces.new((r1[j], r1[jn], r2[jn], r2[j]))
                f.material_index = mat_index
            except ValueError:
                pass

    if cap_ends:
        try:
            f0 = bm.faces.new(reversed(rings[0]))
            f0.material_index = mat_index
            f1 = bm.faces.new(rings[-1])
            f1.material_index = mat_index
        except ValueError:
            pass


def split_profile_edges(points, n, edge_indices):
    """Splits the listed edges (point i -> i+1) of a closed 2D profile into n equal parts."""
    out = []
    m = len(points)
    for i in range(m):
        a = points[i]
        b = points[(i + 1) % m]
        out.append(a)
        if i in edge_indices:
            for k in range(1, n):
                f = k / n
                out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
    return out


def add_box_prism(bm, p_base, right, tangent, normal, half_w, half_d, height, mat_index):
    """Helper to build an oriented 3D rectangular box/prism."""
    b_corners = [
        p_base - right * half_w - tangent * half_d,
        p_base + right * half_w - tangent * half_d,
        p_base + right * half_w + tangent * half_d,
        p_base - right * half_w + tangent * half_d,
    ]
    t_corners = [c + normal * height for c in b_corners]

    v_bot = [bm.verts.new(c) for c in b_corners]
    v_top = [bm.verts.new(c) for c in t_corners]

    for j in range(4):
        jn = (j + 1) % 4
        f = bm.faces.new((v_bot[j], v_bot[jn], v_top[jn], v_top[j]))
        f.material_index = mat_index

    f_b = bm.faces.new(reversed(v_bot))
    f_b.material_index = mat_index
    f_t = bm.faces.new(v_top)
    f_t.material_index = mat_index


def add_slab(bm, corners, up, thickness, mat_index):
    """Thin prism from an arbitrary 4-corner footprint (used for painted chevrons)."""
    v_bot = [bm.verts.new(c) for c in corners]
    v_top = [bm.verts.new(c + up * thickness) for c in corners]
    n = len(corners)
    for j in range(n):
        jn = (j + 1) % n
        f = bm.faces.new((v_bot[j], v_bot[jn], v_top[jn], v_top[j]))
        f.material_index = mat_index
    f_b = bm.faces.new(reversed(v_bot))
    f_b.material_index = mat_index
    f_t = bm.faces.new(v_top)
    f_t.material_index = mat_index


def add_cylinder(bm, center_pos, right, tangent, normal, radius, height, num_segments=12, mat_index=6):
    """Generates a cylinder along `normal`, with its cross-section in the right/tangent plane."""
    ring_bot = []
    ring_top = []
    top_center = center_pos + normal * height

    for k in range(num_segments):
        ang = (2.0 * math.pi * k) / num_segments
        dir_vec = right * math.cos(ang) + tangent * math.sin(ang)
        ring_bot.append(bm.verts.new(center_pos + dir_vec * radius))
        ring_top.append(bm.verts.new(top_center + dir_vec * radius))

    for k in range(num_segments):
        kn = (k + 1) % num_segments
        f = bm.faces.new((ring_bot[k], ring_bot[kn], ring_top[kn], ring_top[k]))
        f.material_index = mat_index

    f_b = bm.faces.new(reversed(ring_bot))
    f_b.material_index = mat_index
    f_t = bm.faces.new(ring_top)
    f_t.material_index = mat_index


def add_cylinder_between(bm, p0, p1, radius, num_segments=8, mat_index=6):
    """Cylinder running from p0 to p1 with a cross-section perpendicular to its own axis."""
    axis = p1 - p0
    length = axis.length
    if length < 1e-5:
        return
    axis_n = axis / length
    ref = Vector((0.0, 0.0, 1.0)) if abs(axis_n.z) < 0.9 else Vector((1.0, 0.0, 0.0))
    u = axis_n.cross(ref).normalized()
    v = axis_n.cross(u).normalized()
    add_cylinder(bm, p0, u, v, axis_n, radius, length, num_segments, mat_index)


def get_w_beam_profile(center_x, side, rail_height, base_z=0.08):
    """Closed W-beam guardrail cross-section. Corrugations bulge toward the roadway (-side)."""
    z_top = base_z + rail_height
    z_bot = z_top - 0.31
    zs = [z_bot, z_bot + 0.078, z_bot + 0.155, z_bot + 0.233, z_top]
    bulge = [0.0, 0.085, 0.03, 0.085, 0.0]
    thick = 0.025
    front = [(center_x - side * (thick + b), z) for b, z in zip(bulge, zs)]
    back = [(center_x - side * b, z) for b, z in zip(reversed(bulge), reversed(zs))]
    return front + back


def add_exit_gore_striping(bm, samples, base_x, side, gore_widths, asphalt_h, line_w=0.2, chevron_spacing=8.0):
    """
    Paints the exit gore: a solid line holding the mainline edge, a second solid line diverging
    toward the exit, and diagonal chevron hatching in the wedge between them.
    """
    n = len(samples)
    if n < 2:
        return
    z0 = asphalt_h
    z1 = asphalt_h + 0.012
    hw = line_w * 0.5
    prof = [(base_x - hw, z0), (base_x + hw, z0), (base_x + hw, z1), (base_x - hw, z1)]

    split = [side * w * 0.4 for w in gore_widths]
    add_extruded_profile(bm, samples, prof, mat_index=5)
    add_extruded_profile(bm, samples, prof, mat_index=5, x_offsets=split)

    accum = 0.0
    next_d = chevron_spacing * 0.5
    for i in range(n):
        if i > 0:
            accum += (samples[i][0] - samples[i - 1][0]).length
        if accum < next_d:
            continue
        next_d = accum + chevron_spacing
        gap = abs(split[i]) - line_w
        if gap < 0.4:
            continue
        pos, tang, norm, right = samples[i]
        xa = base_x + side * hw
        xb = base_x + split[i] - side * hw
        stripe = 0.45
        corners = [(xa, 0.0), (xa, stripe), (xb, stripe + gap), (xb, gap)]
        pts = [pos + right * x + tang * t + norm * z0 for x, t in corners]
        add_slab(bm, pts, norm, 0.012, mat_index=5)


def add_embankment_slopes(bm, samples, left_edges, right_edges, ground_z, verge_w=0.0, mat_index=9,
                          terrain_rings=None):
    """
    Constructs sloped earth berms (embankments) from the outer road edges down to ground_z.
    A flat verge of `verge_w` runs from the asphalt edge out past the sidewalks, barriers,
    walls and posts before the slope starts, so the slope never cuts through them.
    terrain_rings, when given, is {side: [(edge, top, toe), ...]} worked out against real ground:
    fills slope down to it and cuts slope up to it.
    """
    if len(samples) < 2:
        return

    verge_z = -0.01  # just under the sidewalk underside, so the two never z-fight
    for side, edge_list in ((-1.0, left_edges), (1.0, right_edges)):
        if terrain_rings is not None:
            rings = [[bm.verts.new(p) for p in ring[:3]] for ring in terrain_rings[side]]
            walls = [ring[3] for ring in terrain_rings[side]]
            _slope_faces(bm, rings, side, mat_index, walls)
            continue
        rings = []
        for i, (pos, tangent, normal, right) in enumerate(samples):
            ex = edge_list[i]
            world_edge = pos + right * ex + normal * verge_z
            world_top = world_edge + right * (side * verge_w)
            dz = world_top.z - ground_z
            berm_w = abs(dz) * 1.75 + 1.5  # 1.75:1 slope ratio
            # Run the slope horizontally outward, even when the road is banked.
            out = Vector((right.x, right.y, 0.0))
            out = out.normalized() if out.length > 1e-6 else Vector((1.0, 0.0, 0.0))
            world_toe = world_top + out * (side * berm_w)
            world_toe.z = ground_z
            rings.append([bm.verts.new(p) for p in (world_edge, world_top, world_toe)])
        _slope_faces(bm, rings, side, mat_index)


def _slope_faces(bm, rings, side, mat_index, walls=None):
    for i in range(len(rings) - 1):
        a, b = rings[i], rings[i + 1]
        for k in range(2):
            quad = (a[k], b[k], b[k + 1], a[k + 1]) if side > 0 else (a[k], a[k + 1], b[k + 1], b[k])
            try:
                f = bm.faces.new(quad)
                wall = k == 1 and walls is not None and (walls[i] or walls[i + 1])
                f.material_index = 1 if wall else mat_index
            except ValueError:
                pass


def terrain_slope_rings(obj, ground, samples, edges, side, verge_w, max_reach=140.0, fill_reach=30.0):
    """
    (edge, top, toe, is_wall) per sample for one side of the road: a flat verge, then a bank that
    runs down (fill, terrain.SLOPE:1) or up (cut, terrain.SLOPE_CUT:1) until it meets the real
    ground. A fill that would spread further than fill_reach (a hillside steeper than the bank)
    becomes a concrete retaining wall straight down to the ground instead.
    """
    n = len(samples)
    edges_p, tops, outs = [], [], []
    for i, (pos, _t, normal, right) in enumerate(samples):
        e = pos + right * edges[i] + normal * -0.01
        top = e + right * (side * verge_w)
        out = Vector((right.x, right.y, 0.0))
        out = out.normalized() if out.length > 1e-6 else Vector((1.0, 0.0, 0.0))
        edges_p.append(e)
        tops.append(top)
        outs.append(out * side)
    T = np.array([tuple(t) for t in tops])
    O = np.array([tuple(o) for o in outs])
    D = np.arange(0.0, max_reach + 1e-6, 1.0)
    P = T[:, None, :] + O[:, None, :] * D[None, :, None]
    G = terrain.local_heights(obj, ground, P.reshape(-1, 3)).reshape(n, len(D))
    g0 = G[:, 0]
    fill = ~(T[:, 2] < g0)  # unknown ground counts as fill
    sgn = np.where(fill, -1.0, 1.0)
    S = np.where(fill, terrain.SLOPE, terrain.SLOPE_CUT)
    line = T[:, 2:3] + sgn[:, None] * D[None, :] / S[:, None]
    f = (line - G) * (-sgn[:, None])  # <= 0 once the bank has met the ground
    rings = []
    for i in range(n):
        crossed = np.nonzero(np.nan_to_num(f[i], nan=1.0) <= 0.0)[0]
        if len(crossed) and crossed[0] > 0:
            k = crossed[0]
            f0, f1 = f[i, k - 1], f[i, k]
            dist = D[k - 1] + (D[k] - D[k - 1]) * (f0 / (f0 - f1) if f0 != f1 else 1.0)
        elif len(crossed):
            dist = 0.3
        else:
            gz = g0[i] if not np.isnan(g0[i]) else T[i, 2] - 3.0
            dist = min(max_reach, abs(T[i, 2] - gz) * S[i])
        if fill[i] and dist > fill_reach and not np.isnan(g0[i]):
            toe = tops[i] + outs[i] * 0.05
            toe.z = g0[i] - 0.5
            rings.append((edges_p[i], tops[i], toe, True))
            continue
        toe = tops[i] + outs[i] * dist
        # A fill's toe tucks under the ground; a cut ends on it (the terrain is carved below the cut).
        toe.z = T[i, 2] + sgn[i] * dist / S[i] - (0.25 if fill[i] else 0.0)
        rings.append((edges_p[i], tops[i], toe, False))
    return rings


def add_tunnel_section(bm, samples, left_edges, right_edges, clearance_h, wall_t, arched, asphalt_h):
    """
    Concrete tunnel shell (arched vault or rectangular box), thickened portal collars at both
    ends, and a dashed emissive LED strip along the crown.
    """
    n = len(samples)
    if n < 2:
        return
    arch_segs = 12
    margin = 0.6

    def shell_profile(i, extra=0.0):
        lx = left_edges[i] - margin
        rx = right_edges[i] + margin
        cx = (lx + rx) * 0.5
        hw = (rx - lx) * 0.5
        t = wall_t + extra
        if arched:
            spring = max(asphalt_h + 1.5, clearance_h * 0.5)
            rise = clearance_h - spring
            inner = [(cx - hw, 0.0)]
            for k in range(arch_segs + 1):
                a = math.pi * (1.0 - k / arch_segs)
                inner.append((cx + hw * math.cos(a), spring + rise * math.sin(a)))
            inner.append((cx + hw, 0.0))
            outer = [(cx + hw + t, 0.0)]
            for k in range(arch_segs + 1):
                a = math.pi * (k / arch_segs)
                outer.append((cx + (hw + t) * math.cos(a), spring + (rise + t) * math.sin(a)))
            outer.append((cx - hw - t, 0.0))
        else:
            inner = [(cx - hw, 0.0), (cx - hw, clearance_h), (cx + hw, clearance_h), (cx + hw, 0.0)]
            outer = [(cx + hw + t, 0.0), (cx + hw + t, clearance_h + t), (cx - hw - t, clearance_h + t), (cx - hw - t, 0.0)]
        return inner + outer

    add_variable_extrusion(bm, samples, shell_profile, mat_index=1, cap_ends=True)

    # Portal collars protruding outward from each end
    collar_len = 0.8
    for end_i, outward in ((0, -1.0), (n - 1, 1.0)):
        pos, tang, norm, right = samples[end_i]
        frames = [(pos, tang, norm, right), (pos + tang * (outward * collar_len), tang, norm, right)]
        add_variable_extrusion(bm, frames, lambda _k, ei=end_i: shell_profile(ei, extra=0.7), mat_index=1, cap_ends=True)

    # Ceiling LED light strip (dashed fixtures) following the tunnel centreline
    cx_list = [(left_edges[i] + right_edges[i]) * 0.5 for i in range(n)]
    led = [(-0.15, clearance_h - 0.1), (0.15, clearance_h - 0.1), (0.15, clearance_h - 0.01), (-0.15, clearance_h - 0.01)]
    add_extruded_profile(bm, samples, led, mat_index=7, dashed=True, dash_interval=2, x_offsets=cx_list)


def flat_frame(tangent):
    """(tangent, right, up) for things that stand upright: the road's heading with banking removed."""
    up = Vector((0.0, 0.0, 1.0))
    t = Vector((tangent.x, tangent.y, 0.0))
    t = t.normalized() if t.length > 1e-6 else Vector((0.0, 1.0, 0.0))
    return t, t.cross(up).normalized(), up


def add_oriented_box(bm, center, ax, ay, az, hx, hy, hz, mat, front_mat=None):
    """Box centred on `center` with half-sizes along unit axes ax, ay, az; its +az face gets front_mat."""
    v = [bm.verts.new(center + ax * (sx * hx) + ay * (sy * hy) + az * (sz * hz))
         for sz in (-1.0, 1.0) for sx, sy in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))]
    faces = (((0, 3, 2, 1), mat), ((4, 5, 6, 7), mat if front_mat is None else front_mat),
             ((0, 1, 5, 4), mat), ((1, 2, 6, 5), mat), ((2, 3, 7, 6), mat), ((3, 0, 4, 7), mat))
    for idx, m in faces:
        bm.faces.new([v[i] for i in idx]).material_index = m


_TEXT_CACHE = {}


def text_shape(text):
    """Flat lettering for `text` (lines split by newlines), centred on the origin, 1 unit font size.
    Returns (verts [(x, y)], faces [indices], width, height). Cached per string."""
    if text in _TEXT_CACHE:
        return _TEXT_CACHE[text]
    cu = bpy.data.curves.new("CF_SignText_tmp", 'FONT')
    cu.body = text
    cu.align_x = 'CENTER'
    cu.align_y = 'CENTER'
    ob = bpy.data.objects.new("CF_SignText_tmp", cu)
    shape = ([], [], 0.0, 0.0)
    try:
        me = bpy.data.meshes.new_from_object(ob)
        if len(me.vertices):
            xs = [v.co.x for v in me.vertices]
            ys = [v.co.y for v in me.vertices]
            cx, cy = (min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5
            shape = ([(v.co.x - cx, v.co.y - cy) for v in me.vertices],
                     [tuple(f.vertices) for f in me.polygons],
                     max(xs) - min(xs), max(ys) - min(ys))
        bpy.data.meshes.remove(me)
    finally:
        bpy.data.objects.remove(ob)
        bpy.data.curves.remove(cu)
    _TEXT_CACHE[text] = shape
    return shape


def add_text(bm, text, center, ax, ay, az, max_w, max_h, mat):
    """Lettering on a sign face: fitted inside max_w x max_h, facing az. Faces are tagged so the
    final normal recalculation leaves them facing the reader."""
    verts, faces, w, h = text_shape(text)
    if not verts or w <= 0 or h <= 0:
        return
    scale = min(max_w / w, max_h / h)
    bv = [bm.verts.new(center + ax * (x * scale) + ay * (y * scale)) for x, y in verts]
    for f in faces:
        try:
            face = bm.faces.new([bv[i] for i in f])
        except ValueError:
            continue
        face.material_index = mat
        face.normal_update()
        if face.normal.dot(az) < 0:
            face.normal_flip()
        face.tag = True


def sign_message(spec, index):
    """Pick the index-th message from 'msg one; msg two', with '|' as a line break."""
    msgs = [m.strip() for m in spec.split(";") if m.strip()]
    if not msgs:
        return ""
    return "\n".join(line.strip() for line in msgs[index % len(msgs)].split("|"))


def add_sign_panel(bm, center, look_dir, up, width, height, kind, text):
    """
    One sign, centred on `center`, read by drivers travelling along look_dir.
    GUIDE: green face, white border and white lettering, galvanised back with bracing.
    LED: deep dark message cabinet with amber LED lettering and amber hazard beacons.
    """
    az = -look_dir                      # the face points back at oncoming drivers
    ax = look_dir.cross(up).normalized()  # drivers' right
    ay = up
    hw, hh = width * 0.5, height * 0.5
    if kind == 'LED':
        depth = 0.32
        add_oriented_box(bm, center, ax, ay, az, hw, hh, depth, 11, front_mat=11)
        face = center + az * (depth + 0.012)
        add_text(bm, text, face, ax, ay, az, width * 0.84, height * 0.66, 10)
        for sx in (-1.0, 1.0):
            add_oriented_box(bm, face + ax * (sx * (hw - 0.22)) + ay * (hh - 0.2), ax, ay, az,
                             0.11, 0.11, 0.02, 10)
        return
    plate = 0.04
    add_oriented_box(bm, center, ax, ay, az, hw, hh, plate, 6, front_mat=8)
    face = center + az * (plate + 0.004)
    bw = 0.06
    for sy in (-1.0, 1.0):
        add_oriented_box(bm, face + ay * (sy * (hh - bw)), ax, ay, az, hw - 0.03, bw * 0.5, 0.004, 5)
    for sx in (-1.0, 1.0):
        add_oriented_box(bm, face + ax * (sx * (hw - bw)), ax, ay, az, bw * 0.5, hh - 0.03, 0.004, 5)
    add_text(bm, text, face + az * 0.006, ax, ay, az, width * 0.84, height * 0.7, 5)
    # Horizontal bracing on the back, so the rear reads as the back of a real sign.
    for sy in (-0.5, 0.5):
        add_oriented_box(bm, center - az * (plate + 0.06) + ay * (sy * hh), ax, ay, az,
                         hw * 0.96, 0.05, 0.05, 6)


def add_truss(bm, origin, r_f, t_f, up, x0, x1, z_bot, z_top):
    """Box truss of four chords between lateral positions x0..x1 (z values are world heights)."""
    def at(x, z, dt=0.0):
        q = origin + r_f * x + t_f * dt
        q.z = z
        return q
    cx, hw = (x0 + x1) * 0.5, (x1 - x0) * 0.5
    for z in (z_bot, z_top):
        for dt in (-0.25, 0.25):
            add_box_prism(bm, at(cx, z - 0.09, dt), r_f, t_f, up, hw, 0.07, 0.18, mat_index=6)
    n_bay = max(1, int(round((x1 - x0) / 2.5)))
    for k in range(n_bay + 1):
        x = x0 + (x1 - x0) * k / n_bay
        for dt in (-0.25, 0.25):
            add_box_prism(bm, at(x, z_bot, dt), r_f, t_f, up, 0.04, 0.04, z_top - z_bot, mat_index=6)
        if k < n_bay:
            xn = x0 + (x1 - x0) * (k + 1) / n_bay
            for dt in (-0.25, 0.25):
                a, b = (at(x, z_bot, dt), at(xn, z_top, dt)) if k % 2 == 0 else (at(x, z_top, dt), at(xn, z_bot, dt))
                add_cylinder_between(bm, a, b, 0.035, num_segments=5, mat_index=6)


def add_gantry_post(bm, origin, r_f, t_f, up, x, surface_z, top_z):
    base = origin + r_f * x
    base.z = surface_z
    add_box_prism(bm, base - up * 0.45, r_f, t_f, up, 0.55, 0.55, 0.65, mat_index=1)  # footing
    add_box_prism(bm, base, r_f, t_f, up, 0.18, 0.18, top_z - surface_z, mat_index=6)


def add_gantry(bm, sample, style, carriageways, left_x, right_x, clearance, sign_h, asphalt_h, extra_signs=()):
    """
    Overhead sign structure at one point on the road. Everything stands upright however the road
    is banked. carriageways: [(sign centre x, sign width, travel direction ±1, sign kind, text, road side ±1)].
    FULL_SPAN: one truss across the whole road on two posts, each direction's sign on the side
    facing its own traffic. CANTILEVER: per direction, one post outside the road with an arm
    reaching over its lanes carrying a single sign.
    """
    pos, tang, norm, right_b = sample
    t_f, r_f, up = flat_frame(tang)

    def surface_z(x):
        return (pos + right_b * x + norm * asphalt_h).z

    road_top = max(surface_z(left_x), surface_z(right_x), surface_z(0.0))
    z_bot = road_top + clearance
    z_top = z_bot + sign_h
    post_top = z_top + 0.25

    def place_sign(cx, width, direction, kind, text, _side=None):
        look = t_f * direction
        height = sign_h * (0.82 if kind == 'LED' else 1.0)
        depth = 0.32 if kind == 'LED' else 0.04
        c = pos + r_f * cx - look * (0.25 + 0.07 + depth + 0.08)
        c.z = (z_bot + z_top) * 0.5
        add_sign_panel(bm, c, look, up, width, height, kind, text)

    if style == 'CANTILEVER':
        for cx, width, direction, kind, text, side in carriageways:
            post_x = (right_x if side > 0 else left_x) + side * 0.8
            far_x = cx - side * (width * 0.5 + 0.35)
            add_gantry_post(bm, pos, r_f, t_f, up, post_x, surface_z(post_x), post_top)
            add_truss(bm, pos, r_f, t_f, up, min(post_x, far_x), max(post_x, far_x), z_bot + 0.1, z_top - 0.1)
            place_sign(cx, width, direction, kind, text)
        return

    post_l, post_r = left_x - 0.8, right_x + 0.8
    for x in (post_l, post_r):
        add_gantry_post(bm, pos, r_f, t_f, up, x, surface_z(x), post_top)
    add_truss(bm, pos, r_f, t_f, up, post_l, post_r, z_bot + 0.1, z_top - 0.1)
    for sign in list(carriageways) + list(extra_signs):
        place_sign(*sign)


def add_streetlight_at(bm, base_p, right, tangent, normal, arm_dir, pole_h, arm_len, style='COBRA'):
    """Streetlight in the given style (see utilities.LIGHT_STYLES); the cobra head is a steel pole on a
    concrete plinth with a mast arm and an emissive luminaire head. Pass an upright frame (see
    flat_frame) so the pole stands vertical on banked roads."""
    if style != 'COBRA':
        utilities.add_light(bm, base_p, right * arm_dir, tangent, style, pole_h, arm_len)
        return
    add_box_prism(bm, base_p - normal * 0.35, right, tangent, normal, 0.22, 0.22, 0.7, mat_index=1)
    add_cylinder(bm, base_p, right, tangent, normal, radius=0.11, height=pole_h, num_segments=8, mat_index=6)

    arm_base = base_p + normal * (pole_h - 0.12) + right * (arm_dir * arm_len * 0.5)
    add_box_prism(bm, arm_base, right, tangent, normal, arm_len * 0.5, 0.05, 0.1, mat_index=6)

    head_c = base_p + right * (arm_dir * (arm_len - 0.25)) + normal * (pole_h - 0.2)
    add_box_prism(bm, head_c, right, tangent, normal, 0.38, 0.17, 0.14, mat_index=6)
    add_box_prism(bm, head_c - normal * 0.03, right, tangent, normal, 0.32, 0.13, 0.04, mat_index=7)


def add_pier_bent(bm, pos, right, normal, center_x, soffit_w, deck_t, ground_z, style,
                  col_width, col_depth, mat_index=1):
    """
    One bridge support under the deck: a cap beam that spans the underside of the girder
    (banked with the deck), carried on vertical columns that stand on spread footings.

    style 'HAMMERHEAD' is one wide column under a cantilevered cap; 'MULTI' is a row of round
    columns (one every ~8 m of deck) under a full-width cap beam.
    """
    up = Vector((0.0, 0.0, 1.0))
    tang = normal.cross(right).normalized()
    soffit = pos + right * center_x - normal * deck_t
    cap_h = 1.3 if style == 'MULTI' else 1.8
    cap_w = max(1.5, soffit_w - 0.4)
    cap_d = col_depth + 0.5

    # The column tops are found under the cap, so check there is room for a pier at all.
    flat_r = Vector((right.x, right.y, 0.0))
    flat_r = flat_r.normalized() if flat_r.length > 1e-6 else Vector((1.0, 0.0, 0.0))
    flat_t = up.cross(flat_r).normalized()
    cap_base = soffit - normal * cap_h

    if style == 'MULTI':
        n_cols = max(2, math.ceil(cap_w / 8.0))
        span = cap_w - 2.0 * max(col_depth, 1.2)
        xs = [(-span * 0.5 + span * k / (n_cols - 1)) for k in range(n_cols)]
    else:
        xs = [0.0]

    tops = [cap_base + right * x for x in xs]
    # ground_z is a height, or a function giving the ground height under a point (None off the map)
    if callable(ground_z):
        grounds = [ground_z(t) for t in tops]
        known = [gz for gz in grounds if gz is not None]
        if not known:
            return
        grounds = [gz if gz is not None else min(known) for gz in grounds]
    else:
        grounds = [ground_z] * len(tops)
    if min(t.z - gz for t, gz in zip(tops, grounds)) < 0.8:
        return  # deck is too close to the ground for a pier here

    add_box_prism(bm, cap_base, right, tang, normal, cap_w * 0.5, cap_d * 0.5, cap_h, mat_index)

    for x, top, gz in zip(xs, tops, grounds):
        height = top.z - gz + 0.05  # sink slightly into the cap so there is no gap on banks
        base = Vector((top.x, top.y, gz))
        if style == 'MULTI':
            add_cylinder(bm, base, flat_r, flat_t, up, col_depth * 0.5, height, num_segments=16, mat_index=mat_index)
            foot = col_depth * 0.5 + 0.6
            add_box_prism(bm, base - up * 0.5, flat_r, flat_t, up, foot, foot, 0.6, mat_index)
        else:
            add_box_prism(bm, base, flat_r, flat_t, up, col_width * 0.5, col_depth * 0.5, height, mat_index)
            add_box_prism(bm, base - up * 0.5, flat_r, flat_t, up, col_width * 0.5 + 0.8, col_depth * 0.5 + 0.8,
                          0.6, mat_index)


def add_cable_stayed_bridge(bm, samples, half_w, tower_spacing, tower_height, deck_t, ground_z):
    """
    Constructs concrete A-frame bridge towers (legs outside the deck, meeting above the
    roadway) and fans of steel stay cables anchored along both deck edges.
    """
    num_s = len(samples)
    if num_s < 10:
        return

    total_len = sum((samples[i][0] - samples[i - 1][0]).length for i in range(1, num_s))
    step = max(1e-3, total_len / (num_s - 1))

    # Place towers at regular intervals along the curve
    tower_indices = []
    accum = 0.0
    next_t_dist = tower_spacing * 0.5
    for i in range(1, num_s):
        accum += (samples[i][0] - samples[i - 1][0]).length
        if accum >= next_t_dist:
            tower_indices.append(i)
            next_t_dist += tower_spacing

    if not tower_indices:
        tower_indices = [num_s // 3, (num_s * 2) // 3]

    leg_x = half_w + 1.4
    knee_h = 6.5  # legs stay vertical up to this height above the deck to clear traffic

    for t_idx in tower_indices:
        p_pos, tang, norm, right = samples[t_idx]
        t_up, r_up, up = flat_frame(tang)  # towers stand upright even where the deck is banked
        apex = p_pos + up * tower_height

        for side in (-1.0, 1.0):
            base = p_pos + right * (side * leg_x)
            knee = base + up * knee_h
            ground_pt = base.copy()
            gz = ground_z(base) if callable(ground_z) else ground_z
            ground_pt.z = gz if gz is not None else base.z
            if knee.z - ground_pt.z > 0.5:
                add_cylinder_between(bm, ground_pt, knee, radius=0.95, num_segments=8, mat_index=1)
            add_cylinder_between(bm, knee, apex + r_up * (side * 0.7), radius=0.8, num_segments=8, mat_index=1)

        # Lower cross-beam carrying the deck, and the apex cable anchorage block
        add_box_prism(bm, p_pos - norm * (deck_t + 1.2), right, tang, norm,
                      half_w=leg_x + 0.6, half_d=1.0, height=1.2, mat_index=1)
        add_box_prism(bm, apex - up * 4.0, r_up, t_up, up,
                      half_w=1.5, half_d=1.1, height=5.0, mat_index=1)

        # High-tension steel stay cables fanning from the tower head to both deck edges
        cable_pairs = 6
        reach_span = min(tower_spacing * 0.45, 35.0)
        for side in (-1.0, 1.0):
            edge_x = side * (half_w - 0.4)
            for c_i in range(1, cable_pairs + 1):
                frac = c_i / (cable_pairs + 1.0)
                offset_d = (frac - 0.5) * 2.0 * reach_span
                target_idx = max(0, min(num_s - 1, t_idx + int(round(offset_d / step))))
                deck_pos, _, d_norm, d_right = samples[target_idx]
                deck_anchor = deck_pos + d_right * edge_x + d_norm * 0.1

                # Farther cables attach higher on the tower head
                head_drop = 3.5 * (1.0 - 2.0 * abs(frac - 0.5))
                apex_pt = apex - up * (0.5 + head_drop) + r_up * (side * 0.9)
                add_cylinder_between(bm, apex_pt, deck_anchor, radius=0.07, num_segments=6, mat_index=6)


def add_toll_plaza_at(bm, plaza_center, right, tangent, normal, num_lanes, booth_w, island_l, asphalt_h):
    """
    Constructs a highway toll plaza with concrete island splitters, impact attenuator nose blocks,
    modular toll-booth cabins, and an overhead concrete toll canopy roof.
    """
    lane_spacing = 3.8
    total_plaza_w = num_lanes * lane_spacing
    start_x = -total_plaza_w * 0.5
    # Islands sit on the road surface (which may be banked); booths, columns and roof stand upright.
    surf_right = right
    tangent, right, normal = flat_frame(tangent)

    def surface(x):
        return plaza_center + surf_right * x

    # 1. Concrete Island Splitters & Toll Booth Cabins
    for l_i in range(num_lanes + 1):
        island_cx = start_x + l_i * lane_spacing
        if l_i == 0 or l_i == num_lanes:
            continue  # Outer bounds

        island_base = surface(island_cx) + normal * asphalt_h
        # Concrete splitter island curb (sunk a little so it meets the banked surface)
        add_box_prism(bm, island_base - normal * 0.15, right, tangent, normal, half_w=0.65, half_d=island_l * 0.5, height=0.5, mat_index=1)
        # Sloped Impact Attenuator Nose Blocks at entry/exit
        for nose_dir in (-1.0, 1.0):
            nose_base = island_base + tangent * (nose_dir * island_l * 0.45)
            add_box_prism(bm, nose_base, right, tangent, normal, half_w=0.55, half_d=1.2, height=0.25, mat_index=2)

        # Modular Toll Booth Cabin on center of island
        booth_base = island_base + normal * 0.35
        add_box_prism(bm, booth_base, right, tangent, normal, half_w=booth_w * 0.5, half_d=1.8, height=2.8, mat_index=1)

    # 2. Overhead Concrete Toll Canopy Roof spanning across all lanes, carried on a pair of
    #    columns standing on each island either side of its booth.
    # The roof is level, 5.2 m above the highest point of the road under it.
    canopy_world_z = max(surface(start_x).z, surface(-start_x).z, plaza_center.z) + 5.2
    col_top = asphalt_h + 0.35
    col_d = min(island_l * 0.35, 1.8 + 1.2)
    for l_i in range(1, num_lanes):
        island_base = surface(start_x + l_i * lane_spacing) + normal * col_top
        for d in (-col_d, col_d):
            add_box_prism(bm, island_base + tangent * d, right, tangent, normal,
                          half_w=0.28, half_d=0.28, height=canopy_world_z - island_base.z, mat_index=1)
    canopy_base = plaza_center.copy()
    canopy_base.z = canopy_world_z
    add_box_prism(bm, canopy_base, right, tangent, normal, half_w=total_plaza_w * 0.52, half_d=island_l * 0.65, height=0.75, mat_index=1)


# ==============================================================================
# 4. MASTER PROCEDURAL MESH REBUILDER
# ==============================================================================

def road_local_samples(obj, depsgraph=None):
    """The road's guide curve sampled into frames in the road object's local space."""
    props = obj.civil_props
    curve_obj = props.target_curve
    if not curve_obj or curve_obj.type != 'CURVE':
        return []
    samples = sample_bezier_curve(
        curve_obj,
        props.resolution_step,
        use_spline_tilt=props.use_spline_tilt,
        auto_bank=props.auto_superelevation,
        auto_bank_factor=props.superelevation_factor,
        max_bank_deg=props.max_bank_angle,
        depsgraph=depsgraph,
    )
    if len(samples) < 2:
        return []

    inv_mat = obj.matrix_world.inverted_safe()
    inv_rot = inv_mat.to_3x3()
    local_samples = []
    for pos, tang, norm, right in samples:
        l_pos = inv_mat @ pos
        l_tang = (inv_rot @ tang).normalized()
        l_norm = (inv_rot @ norm).normalized()
        l_right = (inv_rot @ right).normalized()
        local_samples.append((l_pos, l_tang, l_norm, l_right))

    # Following the land: height from the terrain, smoothed and held to the grade limit.
    t_obj = props.terrain_object
    if props.follow_terrain and t_obj and t_obj.type == 'MESH' and t_obj != obj and obj.name not in _draping:
        ground = terrain.ground_for(t_obj)
        pin = None
        prev = props.attached_to
        if prev and prev != obj and prev.civil_props.is_civil_object and prev.civil_props.target_curve:
            _draping.add(obj.name)
            try:
                prev_samples = road_local_samples(prev, depsgraph)
            finally:
                _draping.discard(obj.name)
            if prev_samples:
                pin = (prev.matrix_world @ prev_samples[-1][0]).z
        local_samples = terrain.drape(obj, local_samples, ground, pin)
    return local_samples


_draping = set()


def add_abutment(bm, sample, lx, rx, ground_at, toward):
    """Concrete end wall where a bridge deck meets the earth bank, from the ground up to the deck."""
    pos, tang, norm, right = sample
    t_up, r_up, up = flat_frame(tang)
    edge_l, edge_r = pos + right * lx, pos + right * rx
    center = pos + right * ((lx + rx) * 0.5)
    known = [g for g in (ground_at(edge_l), ground_at(center), ground_at(edge_r)) if g is not None]
    if not known:
        return
    bottom = min(known) - 0.6
    top = min(edge_l.z, edge_r.z) - 0.12
    if top - bottom < 0.6:
        return
    base = Vector((center.x, center.y, bottom)) + t_up * (toward * 0.9)
    add_box_prism(bm, base, r_up, t_up, up, (rx - lx) * 0.5 + 0.8, 0.9, top - bottom, mat_index=1)


def _rebuild_without_carving(obj):
    rebuild_civil_mesh(obj, carve=False)


def rebuild_civil_mesh(obj, depsgraph=None, carve=True):
    """Main procedural mesh builder triggered whenever parameters or curves change."""
    if not obj or obj.type != 'MESH' or not obj.civil_props.is_civil_object:
        return
    if obj.mode == 'EDIT':
        return

    props = obj.civil_props
    local_samples = road_local_samples(obj, depsgraph)
    if len(local_samples) < 2:
        return
    inv_mat = obj.matrix_world.inverted_safe()

    ensure_civil_materials(obj)
    bm = bmesh.new()

    num_s = len(local_samples)
    total_lane_w = props.lanes_left * props.lane_width + props.lanes_right * props.lane_width
    median_w = props.median_width if (props.lanes_left > 0 and props.lanes_right > 0) else 0.0
    total_w = total_lane_w + median_w + (props.shoulder_width * 2.0)
    half_w = total_w * 0.5
    deck_t = props.deck_thickness
    asphalt_h = 0.08
    local_ground_z = (inv_mat @ Vector((0.0, 0.0, props.ground_level_z))).z

    # Calculate per-sample Exit Gore Flare offsets (Left & Right)
    left_gore_ext = [0.0] * num_s
    right_gore_ext = [0.0] * num_s
    g_start = min(props.exit_gore_start, props.exit_gore_end - 0.05)
    g_end = max(props.exit_gore_end, g_start + 0.05)

    if props.add_exit_gore and total_w > 0.1:
        for i in range(num_s):
            u = i / max(1, num_s - 1)
            if g_start <= u <= g_end:
                t_g = (u - g_start) / max(1e-5, (g_end - g_start))
                flare = props.exit_gore_width * (t_g * t_g * (3.0 - 2.0 * t_g))
                if props.exit_gore_side in ('LEFT', 'BOTH'):
                    left_gore_ext[i] = flare
                if props.exit_gore_side in ('RIGHT', 'BOTH'):
                    right_gore_ext[i] = flare

    left_road_edges = [-half_w - left_gore_ext[i] for i in range(num_s)]
    right_road_edges = [half_w + right_gore_ext[i] for i in range(num_s)]

    # Flat strip beside the asphalt that clears sidewalks, barriers, walls and posts.
    sw = props.sidewalk_width if props.add_sidewalks else 0.0
    verge = sw + 0.5
    if props.add_side_barriers:
        verge = max(verge, sw + 0.4)
    if props.add_streetlights and props.streetlight_placement != 'MEDIAN':
        verge = max(verge, sw + 0.8)
    if props.add_sound_walls:
        verge = max(verge, sw + 0.45 + props.sound_wall_thickness * 0.5)
    if props.add_gantry_signs:
        verge = max(verge, sw + 1.3)

    # Real ground: each stretch of road is on the ground, a bridge, or a tunnel.
    terrain_obj = props.terrain_object
    if not (terrain_obj and terrain_obj.type == 'MESH' and terrain_obj != obj):
        terrain_obj = None
    ground = terrain.ground_for(terrain_obj) if terrain_obj else None
    modes = None
    ground_at = local_ground_z
    if ground is not None:
        cen = np.array([tuple(sp[0]) for sp in local_samples])
        lft = np.array([tuple(sp[0] + sp[3] * left_road_edges[i]) for i, sp in enumerate(local_samples)])
        rgt = np.array([tuple(sp[0] + sp[3] * right_road_edges[i]) for i, sp in enumerate(local_samples)])
        gc = terrain.local_heights(obj, ground, cen)
        if not np.isnan(gc).all():
            gc = terrain.fill_along(gc, local_ground_z)
            d_along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(cen[:, :2], axis=0), axis=1))])
            manual = None
            if props.add_tunnel and total_w > 0.1:
                ts_u = min(props.tunnel_start, props.tunnel_end - 0.05)
                te_u = max(props.tunnel_end, ts_u + 0.05)
                manual = (max(0, int(ts_u * (num_s - 1))), min(num_s - 1, int(te_u * (num_s - 1))))
            low_edge = np.minimum(lft[:, 2], rgt[:, 2])
            overrides = [(st.kind, st.center - st.length * 0.5, st.center + st.length * 0.5)
                         for st in props.structures]
            wet = ~np.isnan(terrain.local_water(obj, ground, cen))
            modes = terrain.classify(d_along, np.minimum(cen[:, 2], low_edge), gc, props, manual, overrides, wet)
            # Tunnel stretches with too little ground over them sit in an open cut instead.
            crown = cen[:, 2] + props.tunnel_height + props.tunnel_wall_thickness + 1.5
            covered = (modes == terrain.TUNNEL) & (gc >= crown)

            def ground_at(p):
                h = terrain.local_heights(obj, ground, [tuple(p)])[0]
                return None if np.isnan(h) else float(h)

    # 1. Asphalt Road Surface & Roadbed/Girder Profile
    # Wide faces are split into ~2 m strips: a single full-width quad twists on banked or
    # graded curves, and its triangulation then lets the deck below poke through the asphalt.
    max_road_w = max(right_road_edges[i] - left_road_edges[i] for i in range(num_s))
    w_splits = max(1, math.ceil(max_road_w / 2.0))

    if total_w > 0.01:
        def asphalt_prof_fn(i):
            lx, rx = left_road_edges[i], right_road_edges[i]
            return split_profile_edges([(lx, 0.0), (rx, 0.0), (rx, asphalt_h), (lx, asphalt_h)], w_splits, (0, 2))

        add_variable_extrusion(bm, local_samples, asphalt_prof_fn, mat_index=0, cap_ends=True)

        def girder_prof_fn(i):
            lx, rx = left_road_edges[i], right_road_edges[i]
            w_curr = rx - lx
            g_inset = max(0.4, w_curr * 0.18)
            return split_profile_edges([
                (lx - 0.25, 0.0),
                (rx + 0.25, 0.0),
                (rx + 0.25, -0.25),
                (rx - g_inset, -deck_t),
                (lx + g_inset, -deck_t),
                (lx - 0.25, -0.25),
            ], w_splits, (0, 3))

        def base_prof_fn(i):
            lx, rx = left_road_edges[i], right_road_edges[i]
            return split_profile_edges([
                (lx - 0.15, -deck_t * 0.4),
                (rx + 0.15, -deck_t * 0.4),
                (rx + 0.05, 0.0),
                (lx - 0.05, 0.0),
            ], w_splits, (0, 2))

        if modes is not None:
            # Girders under the bridge stretches, a roadbed everywhere else.
            for kind, a, b in terrain.runs_with_overlap(modes):
                if b <= a:
                    continue
                fn = girder_prof_fn if kind == terrain.BRIDGE else base_prof_fn
                add_variable_extrusion(bm, local_samples[a:b + 1], lambda k, a=a, fn=fn: fn(a + k),
                                       mat_index=1, cap_ends=True)
        elif props.is_overpass:
            add_variable_extrusion(bm, local_samples, girder_prof_fn, mat_index=1, cap_ends=True)
        else:
            add_variable_extrusion(bm, local_samples, base_prof_fn, mat_index=1, cap_ends=True)

    # 2. Terrain Cut-and-Fill Embankment Slopes (Berms)
    if props.add_embankments and modes is not None:
        # Banks down to (fill) or up to (cut) the real ground, wherever the road is on the ground.
        for kind, a, b in terrain.runs_with_overlap(modes):
            if kind != terrain.ROAD or b <= a:
                continue
            sub = local_samples[a:b + 1]
            le, re_ = left_road_edges[a:b + 1], right_road_edges[a:b + 1]
            rings = {-1.0: terrain_slope_rings(obj, ground, sub, le, -1.0, verge),
                     1.0: terrain_slope_rings(obj, ground, sub, re_, 1.0, verge)}
            add_embankment_slopes(bm, sub, le, re_, ground_z=local_ground_z, verge_w=verge,
                                  mat_index=9, terrain_rings=rings)
    elif props.add_embankments and not props.is_overpass:
        add_embankment_slopes(bm, local_samples, left_road_edges, right_road_edges,
                              ground_z=local_ground_z, verge_w=verge, mat_index=9)

    # 3. Exit Gore Channelizing V-Lines & Chevron Markings
    if props.add_exit_gore and total_w > 0.1:
        gore_indices = [i for i in range(num_s) if (left_gore_ext[i] > 0.01 or right_gore_ext[i] > 0.01)]
        if len(gore_indices) >= 3:
            g_sub_samples = [local_samples[i] for i in gore_indices]
            if props.exit_gore_side in ('RIGHT', 'BOTH'):
                g_w_right = [right_gore_ext[i] for i in gore_indices]
                add_exit_gore_striping(bm, g_sub_samples, half_w - props.shoulder_width, 1.0, g_w_right, asphalt_h)
            if props.exit_gore_side in ('LEFT', 'BOTH'):
                g_w_left = [left_gore_ext[i] for i in gore_indices]
                add_exit_gore_striping(bm, g_sub_samples, -half_w + props.shoulder_width, -1.0, g_w_left, asphalt_h)

    # 4. Sidewalks / Curbs
    if props.add_sidewalks:
        sw_w = props.sidewalk_width
        sw_h = 0.18
        for side, x_list in ((-1.0, [-left_gore_ext[i] for i in range(num_s)]), (1.0, right_gore_ext)):
            x_inner = side * half_w
            x_outer = side * (half_w + sw_w)
            prof = [
                (min(x_inner, x_outer), 0.0),
                (max(x_inner, x_outer), 0.0),
                (max(x_inner, x_outer), sw_h),
                (min(x_inner, x_outer), sw_h),
            ]
            add_extruded_profile(bm, local_samples, prof, mat_index=1, x_offsets=x_list)

    def get_jersey_profile(center_x, height, base_w=0.6, top_w=0.2):
        hw_b = base_w * 0.5
        hw_m = base_w * 0.32
        hw_t = top_w * 0.5
        return [
            (center_x - hw_b, asphalt_h),
            (center_x + hw_b, asphalt_h),
            (center_x + hw_b, asphalt_h + 0.08),
            (center_x + hw_m, asphalt_h + 0.35),
            (center_x + hw_t, asphalt_h + height),
            (center_x - hw_t, asphalt_h + height),
            (center_x - hw_m, asphalt_h + 0.35),
            (center_x - hw_b, asphalt_h + 0.08),
        ]

    med_center_x = (-half_w + props.shoulder_width + props.lanes_left * props.lane_width + median_w * 0.5)

    # 5. Median Concrete Jersey Barrier
    if props.add_median_barrier:
        j_prof = get_jersey_profile(med_center_x, props.barrier_height)
        add_extruded_profile(bm, local_samples, j_prof, mat_index=2)

    # 6. Side Concrete Barriers
    outer_edge_offset = half_w + (props.sidewalk_width if props.add_sidewalks else 0.0)
    if props.add_side_barriers:
        for side, x_list in ((-1.0, [-left_gore_ext[i] for i in range(num_s)]), (1.0, right_gore_ext)):
            j_prof = get_jersey_profile(side * outer_edge_offset, props.barrier_height, base_w=0.5, top_w=0.18)
            add_extruded_profile(bm, local_samples, j_prof, mat_index=2, x_offsets=x_list)

    # 7. Metallic W-Beam Guardrails & Support Posts
    if props.add_guardrails:
        gr_offset = outer_edge_offset - 0.15
        gr_h = props.guardrail_height
        for side, x_list in ((-1.0, [-left_gore_ext[i] for i in range(num_s)]), (1.0, right_gore_ext)):
            w_prof = get_w_beam_profile(side * gr_offset, side, rail_height=gr_h, base_z=asphalt_h)
            add_extruded_profile(bm, local_samples, w_prof, mat_index=6, x_offsets=x_list)

        post_accum = 0.0
        next_post_d = 0.0
        for i in range(num_s):
            if i > 0:
                post_accum += (local_samples[i][0] - local_samples[i - 1][0]).length
            if post_accum >= next_post_d:
                next_post_d = post_accum + props.guardrail_post_spacing
                p_curr, tang, norm, right = local_samples[i]
                t_up, r_up, up = flat_frame(tang)
                for side, g_ext in ((-1.0, left_gore_ext[i]), (1.0, right_gore_ext[i])):
                    post_base = p_curr + right * (side * (gr_offset + g_ext + 0.05)) + norm * asphalt_h
                    add_box_prism(bm, post_base - up * 0.15, r_up, t_up, up,
                                  half_w=0.06, half_d=0.08, height=gr_h + 0.18, mat_index=6)

    # 8. Highway Concrete Sound / Retaining Walls
    if props.add_sound_walls:
        wall_offset = outer_edge_offset + 0.35
        wh = props.sound_wall_height
        wt = props.sound_wall_thickness
        for side, x_list in ((-1.0, [-left_gore_ext[i] for i in range(num_s)]), (1.0, right_gore_ext)):
            # Walls stand vertical: each cross-section is a rectangle on the road edge, raised straight up.
            rings = []
            for i, (pos, tang, norm, right) in enumerate(local_samples):
                _, _, up = flat_frame(tang)
                cx = side * wall_offset + x_list[i]
                a = pos + right * (cx - wt * 0.5) - up * 0.3
                b = pos + right * (cx + wt * 0.5) - up * 0.3
                top = max(a.z, b.z) + wh + 0.3
                a_top, b_top = a.copy(), b.copy()
                a_top.z = b_top.z = top
                rings.append([bm.verts.new(q) for q in (a, b, b_top, a_top)])
            for i in range(len(rings) - 1):
                r1, r2 = rings[i], rings[i + 1]
                for j in range(4):
                    jn = (j + 1) % 4
                    bm.faces.new((r1[j], r1[jn], r2[jn], r2[j])).material_index = 3
            for ring in (rings[0], list(reversed(rings[-1]))):
                bm.faces.new(ring).material_index = 3

    # 9. Lane Markings
    if props.add_lane_markings and (props.lanes_left > 0 or props.lanes_right > 0):
        line_w = 0.15
        line_z = asphalt_h + 0.01

        def add_marking_strip(x_pos, mat_idx, is_dashed=False, x_offs=None):
            p = [
                (x_pos - line_w * 0.5, asphalt_h),
                (x_pos + line_w * 0.5, asphalt_h),
                (x_pos + line_w * 0.5, line_z),
                (x_pos - line_w * 0.5, line_z),
            ]
            add_extruded_profile(bm, local_samples, p, mat_index=mat_idx, cap_ends=True, dashed=is_dashed, dash_interval=3, x_offsets=x_offs)

        left_edge = -half_w + props.shoulder_width
        right_edge = half_w - props.shoulder_width

        add_marking_strip(left_edge, 5, is_dashed=False, x_offs=[-left_gore_ext[i] for i in range(num_s)])
        add_marking_strip(right_edge, 5, is_dashed=False, x_offs=right_gore_ext)

        for l_idx in range(1, props.lanes_left):
            lx = left_edge + l_idx * props.lane_width
            add_marking_strip(lx, 5, is_dashed=True)

        for r_idx in range(1, props.lanes_right):
            rx = right_edge - r_idx * props.lane_width
            add_marking_strip(rx, 5, is_dashed=True)

        if props.lanes_left > 0 and props.lanes_right > 0:
            c_left = left_edge + props.lanes_left * props.lane_width
            c_right = right_edge - props.lanes_right * props.lane_width
            add_marking_strip(c_left + 0.15, 4, is_dashed=False)
            add_marking_strip(c_right - 0.15, 4, is_dashed=False)

    # Determine Tunnel Sample Range
    t_start_idx, t_end_idx = -1, -1
    if modes is not None:
        if total_w > 0.1:
            for kind, a, b in terrain.runs_with_overlap(modes):
                if kind == terrain.TUNNEL and b > a:
                    add_tunnel_section(
                        bm, local_samples[a:b + 1], left_road_edges[a:b + 1], right_road_edges[a:b + 1],
                        clearance_h=props.tunnel_height,
                        wall_t=props.tunnel_wall_thickness,
                        arched=(props.tunnel_profile == 'ARCHED'),
                        asphalt_h=asphalt_h
                    )
    elif props.add_tunnel and total_w > 0.1:
        ts_u = min(props.tunnel_start, props.tunnel_end - 0.05)
        te_u = max(props.tunnel_end, ts_u + 0.05)
        t_start_idx = max(0, int(ts_u * (num_s - 1)))
        t_end_idx = min(num_s - 1, int(te_u * (num_s - 1)))
        if t_end_idx > t_start_idx:
            t_samples = local_samples[t_start_idx:t_end_idx + 1]
            t_left = [left_road_edges[k] for k in range(t_start_idx, t_end_idx + 1)]
            t_right = [right_road_edges[k] for k in range(t_start_idx, t_end_idx + 1)]
            add_tunnel_section(
                bm, t_samples, t_left, t_right,
                clearance_h=props.tunnel_height,
                wall_t=props.tunnel_wall_thickness,
                arched=(props.tunnel_profile == 'ARCHED'),
                asphalt_h=asphalt_h
            )

    # Portals: where a tunnel goes under the hill, a concrete headwall around the opening, and a
    # matching hole in the terrain so the hillside doesn't hang over the tunnel mouth.
    portal_holes = []
    tube_pad = 0.6 + props.tunnel_wall_thickness + 0.7
    if modes is not None and ground.grid is not None:
        cell = abs(ground.grid[2])
        for k in range(num_s - 1):
            if covered[k] == covered[k + 1]:
                continue
            j, o = (k + 1, k) if covered[k + 1] else (k, k + 1)
            into = 1.0 if j > o else -1.0
            pos, tang, norm, right = local_samples[j]
            t_up, r_up, up = flat_frame(tang)
            axis = t_up * into
            lx, rx = left_road_edges[j], right_road_edges[j]
            center = pos + right * ((lx + rx) * 0.5)
            open_hw = (rx - lx) * 0.5 + tube_pad
            M = obj.matrix_world
            wc = M @ center
            wa = (M.to_3x3() @ axis)
            wa = Vector((wa.x, wa.y)).normalized()
            hole = (wc.x, wc.y, wa.x, wa.y, -1.25 * cell - 0.5, 2.0 * cell + 0.5, open_hw + 0.3)
            ext = terrain.hole_extent(terrain_obj, hole)
            if ext is None:
                continue
            portal_holes.append(hole)
            a0, a1, lat = ext[0] - 0.3, ext[1] + 0.3, ext[2]
            half_w = max(lat, open_hw) + 3.0
            # Up to the hillside over the whole headwall.
            probe = [center + axis * a + r_up * x for a in np.linspace(a0, a1, 4)
                     for x in np.linspace(-half_w, half_w, 5)]
            hs = terrain.local_heights(obj, ground, [tuple(q) for q in probe])
            top = (np.nanmax(hs) + 0.5) if not np.isnan(hs).all() else center.z + 12.0
            top = min(top, center.z + 45.0)
            bottom = center.z - 1.0
            crown_o = props.tunnel_height + props.tunnel_wall_thickness + 0.7
            mid = center + axis * ((a0 + a1) * 0.5)
            half_d = (a1 - a0) * 0.5
            base = Vector((mid.x, mid.y, bottom))
            side_hw = (half_w - open_hw) * 0.5
            for sgn in (-1.0, 1.0):
                add_box_prism(bm, base + r_up * (sgn * (open_hw + side_hw)), r_up, axis, up,
                              side_hw, half_d, top - bottom, mat_index=1)
            if top > center.z + crown_o:
                add_box_prism(bm, Vector((mid.x, mid.y, center.z + crown_o)), r_up, axis, up,
                              open_hw, half_d, top - center.z - crown_o, mat_index=1)

    def is_inside_tunnel(idx):
        if modes is not None:
            return modes[idx] == terrain.TUNNEL
        return props.add_tunnel and (t_start_idx <= idx <= t_end_idx)

    # 10. Cable-Stayed Bridge Towers & Cables
    if props.add_cable_bridge and props.is_overpass:
        add_cable_stayed_bridge(
            bm, local_samples, half_w,
            tower_spacing=props.cable_tower_spacing,
            tower_height=props.cable_tower_height,
            deck_t=deck_t,
            ground_z=ground_at,
        )

    # 11. Highway Toll Booth Plazas
    if props.add_toll_plaza and num_s >= 5:
        toll_idx = int(props.toll_plaza_position * (num_s - 1))
        toll_idx = max(2, min(num_s - 3, toll_idx))
        p_pos, tang, norm, right = local_samples[toll_idx]
        if not is_inside_tunnel(toll_idx):
            add_toll_plaza_at(
                bm, p_pos, right, tang, norm,
                num_lanes=(props.lanes_left + props.lanes_right + 2),
                booth_w=props.toll_booth_width,
                island_l=props.toll_island_length,
                asphalt_h=asphalt_h
            )

    # 12. Overhead Highway Gantry Signs
    if props.add_gantry_signs and total_w > 0.5:
        gantry_accum = 0.0
        next_gantry_d = props.gantry_spacing * 0.45
        g_index = 0
        sw = props.sidewalk_width if props.add_sidewalks else 0.0
        # Right lanes travel along the curve with right-hand traffic, against it with left-hand.
        fwd = -1.0 if props.traffic_left_hand else 1.0

        for i in range(num_s):
            if i > 0:
                gantry_accum += (local_samples[i][0] - local_samples[i - 1][0]).length
            if gantry_accum < next_gantry_d:
                continue
            next_gantry_d = gantry_accum + props.gantry_spacing
            if is_inside_tunnel(i):
                continue

            sign_type = props.gantry_sign_type
            if sign_type == 'ALTERNATE':
                sign_type = 'GUIDE' if g_index % 2 == 0 else 'LED'
            text_spec = props.led_sign_text if sign_type == 'LED' else props.guide_sign_text

            carriageways = []
            for n_lanes, cx, direction, side in (
                (props.lanes_right, half_w - props.shoulder_width - props.lanes_right * props.lane_width * 0.5, fwd, 1.0),
                (props.lanes_left, -half_w + props.shoulder_width + props.lanes_left * props.lane_width * 0.5, -fwd, -1.0),
            ):
                if n_lanes <= 0:
                    continue
                if props.gantry_style == 'CANTILEVER' and props.gantry_side != 'BOTH' \
                        and side != (1.0 if props.gantry_side == 'RIGHT' else -1.0):
                    continue  # one arm, on the side you picked
                span = n_lanes * props.lane_width
                if props.gantry_style == 'CANTILEVER':
                    # One sign over the outer lanes, on a short arm from the roadside post.
                    width = min(span * 0.9, max(3.4, props.lane_width * 1.15), 6.0)
                    cx = cx + side * (span * 0.5 - width * 0.5 - 0.2)
                elif sign_type == 'LED':
                    width = min(max(3.2, span * 0.72), 10.0)
                else:
                    width = min(max(2.8, span * 0.78), 12.0)
                text = sign_message(text_spec, g_index + len(carriageways))
                carriageways.append((cx, width, direction, sign_type, text, side))

            extra = []
            if props.gantry_style == 'FULL_SPAN' and right_gore_ext[i] > 1.8:
                extra.append((half_w + right_gore_ext[i] * 0.55, min(3.8, right_gore_ext[i] * 0.85), fwd,
                              'GUIDE', "EXIT", 1.0))

            add_gantry(bm, local_samples[i], props.gantry_style, carriageways,
                       left_x=left_road_edges[i] - sw, right_x=right_road_edges[i] + sw,
                       clearance=props.gantry_clearance, sign_h=props.gantry_sign_height,
                       asphalt_h=asphalt_h, extra_signs=extra)
            g_index += 1

    # 13. Highway Streetlights
    if props.add_streetlights:
        light_accum = 0.0
        next_light_d = props.streetlight_spacing * 0.25
        pole_idx = 0
        pole_offset = outer_edge_offset + 0.25

        for i in range(num_s):
            if i > 0:
                light_accum += (local_samples[i][0] - local_samples[i - 1][0]).length
            if light_accum >= next_light_d:
                next_light_d = light_accum + props.streetlight_spacing
                if is_inside_tunnel(i):
                    continue

                p_curr, tang, norm, right = local_samples[i]
                t_up, r_up, up = flat_frame(tang)  # poles stand vertical on banked roads

                if props.streetlight_placement == 'BOTH':
                    for side, g_ext in ((-1.0, left_gore_ext[i]), (1.0, right_gore_ext[i])):
                        base_p = p_curr + right * (side * (pole_offset + g_ext)) + norm * asphalt_h
                        add_streetlight_at(bm, base_p, r_up, t_up, up, arm_dir=-side,
                                           pole_h=props.streetlight_height, arm_len=props.streetlight_arm_length,
                                           style=props.streetlight_style)
                elif props.streetlight_placement in ('LEFT', 'RIGHT'):
                    side = -1.0 if props.streetlight_placement == 'LEFT' else 1.0
                    g_ext = left_gore_ext[i] if side < 0 else right_gore_ext[i]
                    base_p = p_curr + right * (side * (pole_offset + g_ext)) + norm * asphalt_h
                    add_streetlight_at(bm, base_p, r_up, t_up, up, arm_dir=-side,
                                       pole_h=props.streetlight_height, arm_len=props.streetlight_arm_length,
                                       style=props.streetlight_style)
                elif props.streetlight_placement == 'ALTERNATING':
                    side = -1.0 if (pole_idx % 2 == 0) else 1.0
                    g_ext = left_gore_ext[i] if side < 0 else right_gore_ext[i]
                    base_p = p_curr + right * (side * (pole_offset + g_ext)) + norm * asphalt_h
                    add_streetlight_at(bm, base_p, r_up, t_up, up, arm_dir=-side,
                                       pole_h=props.streetlight_height, arm_len=props.streetlight_arm_length,
                                       style=props.streetlight_style)
                elif props.streetlight_placement == 'MEDIAN':
                    base_p = p_curr + right * med_center_x + norm * asphalt_h
                    add_streetlight_at(bm, base_p, r_up, t_up, up, arm_dir=-1.0,
                                       pole_h=props.streetlight_height, arm_len=props.streetlight_arm_length,
                                       style=props.streetlight_style)
                    if props.streetlight_style in ('COBRA', 'MODERN'):  # twin arms; post-top lights stand alone
                        add_streetlight_at(bm, base_p, r_up, t_up, up, arm_dir=1.0,
                                           pole_h=props.streetlight_height, arm_len=props.streetlight_arm_length,
                                           style=props.streetlight_style)
                pole_idx += 1

    # 14. Overpass Support Pillars
    # A support stands at each end of the deck, with the rest spread evenly between them.
    # Cable-stayed bridges hang between their towers, so they only get the two end supports.
    def place_pier(d, cum):
        j = 0
        while j < num_s - 2 and cum[j + 1] < d:
            j += 1
        seg = max(1e-6, cum[j + 1] - cum[j])
        f = min(1.0, max(0.0, (d - cum[j]) / seg))
        p0, t0, n0, r0 = local_samples[j]
        p1, t1, n1, r1 = local_samples[j + 1]
        lx = left_road_edges[j] + (left_road_edges[j + 1] - left_road_edges[j]) * f
        rx = right_road_edges[j] + (right_road_edges[j + 1] - right_road_edges[j]) * f
        w_curr = rx - lx
        g_inset = max(0.4, w_curr * 0.18)  # matches the girder profile above
        soffit_w = w_curr - 2.0 * g_inset
        style = props.pier_style
        if style == 'AUTO':
            style = 'MULTI' if soffit_w > 9.0 else 'HAMMERHEAD'
        add_pier_bent(bm, p0.lerp(p1, f), r0.lerp(r1, f).normalized(), n0.lerp(n1, f).normalized(),
                      (lx + rx) * 0.5, soffit_w, deck_t, ground_at,
                      style, min(props.pillar_width, max(1.0, soffit_w * 0.6)), props.pillar_depth)

    if modes is not None and props.add_pillars:
        # Each bridge stretch: piers spread along it, abutments where it meets the earth bank.
        cum = [0.0]
        for i in range(1, num_s):
            cum.append(cum[-1] + (local_samples[i][0] - local_samples[i - 1][0]).length)
        runs = terrain.runs_with_overlap(modes)
        # Clear span is the free width between pillar faces (footings included).
        pier_half = props.pillar_depth * 0.5 + 0.9
        clear_spans = [(float(np.interp(st.center, d_along, cum)), st.clear_span * 0.5 + pier_half)
                       for st in props.structures if st.kind == 'BRIDGE' and st.clear_span > 0.0]
        for k, (kind, a, b) in enumerate(runs):
            if kind != terrain.BRIDGE or b <= a:
                continue
            prev_kind = runs[k - 1][0] if k > 0 else None
            next_kind = runs[k + 1][0] if k + 1 < len(runs) else None
            d0, d1 = cum[a], cum[b]
            inset = min(1.5, (d1 - d0) * 0.05)
            s0 = d0 + (0.0 if prev_kind is not None else inset)
            s1 = d1 - (0.0 if next_kind is not None else inset)
            if props.add_cable_bridge and props.is_overpass:
                ds = [s0, s1]
            else:
                n_sp = max(1, round((s1 - s0) / props.pillar_spacing))
                ds = [s0 + (s1 - s0) * q / n_sp for q in range(n_sp + 1)]
            if prev_kind is not None:
                ds = ds[1:]  # the abutment or tunnel portal carries this end
            if next_kind is not None:
                ds = ds[:-1]
            for d in ds:
                if any(abs(d - c) < half for c, half in clear_spans):
                    continue  # keep the tracks / road under an overpass clear
                place_pier(d, cum)
            if prev_kind == terrain.ROAD:
                add_abutment(bm, local_samples[a], left_road_edges[a], right_road_edges[a], ground_at, -1.0)
            if next_kind == terrain.ROAD:
                add_abutment(bm, local_samples[b], left_road_edges[b], right_road_edges[b], ground_at, 1.0)

    elif props.is_overpass and props.add_pillars:
        cum = [0.0]
        for i in range(1, num_s):
            cum.append(cum[-1] + (local_samples[i][0] - local_samples[i - 1][0]).length)
        deck_len = cum[-1]
        end_inset = min(1.5, deck_len * 0.05)
        usable = max(0.0, deck_len - 2.0 * end_inset)
        if props.add_cable_bridge:
            pier_ds = [end_inset, deck_len - end_inset]
        else:
            n_spans = max(1, round(usable / props.pillar_spacing))
            pier_ds = [end_inset + usable * k / n_spans for k in range(n_spans + 1)]
        for d in pier_ds:
            place_pier(d, cum)

    # Intersections: the road gives way to their pads.
    junctions.trim_road(bm, obj)

    # Dashed extrusions leave unused ring vertices behind; drop them.
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context='VERTS')

    # Sign lettering is open, flat geometry already facing the reader; leave it out of the recalculation.
    bmesh.ops.recalc_face_normals(bm, faces=[f for f in bm.faces if not f.tag])
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()

    # Roads continuing from this one stay on its end.
    rebuild_attached_roads(obj, depsgraph)

    # Keep the traffic routes that run over this road on it as it changes.
    traffic.update_routes(obj, local_samples, depsgraph)

    # Cut the terrain down where the road passes through it.
    old_terrain = terrain.carved_by(obj.name)
    if modes is not None:
        terrain.record_carve(obj, terrain_obj, local_samples, modes, left_road_edges, right_road_edges,
                             verge, deck_t, covered=covered, tube_pad=tube_pad + 1.0, holes=portal_holes)
        if carve:
            terrain.carve(terrain_obj, rebuild_fn=_rebuild_without_carving)
    else:
        terrain.forget_carve(obj.name)
    if carve and old_terrain and (terrain_obj is None or old_terrain != terrain_obj.name):
        terrain.carve(bpy.data.objects.get(old_terrain), rebuild_fn=_rebuild_without_carving)

    # Intersections on this road follow it.
    junctions.road_rebuilt(obj, depsgraph)


# ==============================================================================
# 5. PROPERTY GROUP DEFINITION
# ==============================================================================

def _streetlight_style_changed(self, context):
    """A new light style starts from its own usual height and spacing."""
    h, spacing = utilities.LIGHT_DEFAULTS[self.streetlight_style]
    self["streetlight_height"] = h
    self["streetlight_spacing"] = spacing
    update_civil_prop(self, context)


def update_civil_prop(self, context):
    obj = self.id_data
    if obj and obj.type == 'MESH':
        dg = context.evaluated_depsgraph_get() if context else None
        rebuild_civil_mesh(obj, depsgraph=dg)


# ------------------------------------------------------------------------------
# Segment builder: a road described as a chain of straights and arcs
# ------------------------------------------------------------------------------

def segment_chain_nodes(segments):
    """
    Turns a chain of (length, bend, rise) segments into Bezier nodes in curve-local space.
    The chain starts at the origin heading +X. Positive bend turns left. Arcs are split into
    pieces of at most 90 degrees so the Bezier circle approximation stays accurate.
    Returns a list of (co, heading, handle_in_len, handle_out_len, grade_in, grade_out).
    """
    pos = Vector((0.0, 0.0, 0.0))
    heading = 0.0
    nodes = [[pos.copy(), heading, 0.0, 0.0, None, None]]
    for seg in segments:
        length = max(0.1, seg.length)
        theta = math.radians(seg.bend)
        grade = seg.rise / length
        n_sub = max(1, math.ceil(abs(seg.bend) / 90.0 - 1e-6))
        sub_len = length / n_sub
        sub_theta = theta / n_sub
        for _ in range(n_sub):
            if abs(sub_theta) < 1e-4:
                step = Vector((math.cos(heading), math.sin(heading), 0.0)) * sub_len
                handle = sub_len / 3.0
                new_heading = heading
            else:
                radius = sub_len / abs(sub_theta)
                chord = 2.0 * radius * math.sin(abs(sub_theta) * 0.5)
                mid = heading + sub_theta * 0.5
                step = Vector((math.cos(mid), math.sin(mid), 0.0)) * chord
                handle = (4.0 / 3.0) * math.tan(abs(sub_theta) / 4.0) * radius
                new_heading = heading + sub_theta
            nodes[-1][3] = handle
            nodes[-1][5] = grade
            pos = pos + step + Vector((0.0, 0.0, grade * sub_len))
            nodes.append([pos.copy(), new_heading, handle, 0.0, grade, None])
            heading = new_heading
    return nodes


def write_segments_to_curve(curve_obj, segments):
    """Replaces the guide curve's shape with the segment chain (in the curve object's local space)."""
    if not segments:
        return
    nodes = segment_chain_nodes(segments)
    cu = curve_obj.data
    cu.splines.clear()
    cu.dimensions = '3D'
    cu.resolution_u = 32
    spline = cu.splines.new('BEZIER')
    spline.bezier_points.add(len(nodes) - 1)
    for i, (co, heading, h_in, h_out, g_in, g_out) in enumerate(nodes):
        grades = [g for g in (g_in, g_out) if g is not None]
        g = sum(grades) / len(grades) if grades else 0.0
        d = Vector((math.cos(heading), math.sin(heading), g))
        bp = spline.bezier_points[i]
        bp.handle_left_type = 'FREE'
        bp.handle_right_type = 'FREE'
        bp.co = co
        bp.handle_left = co - d * (h_in if h_in > 0 else h_out)
        bp.handle_right = co + d * (h_out if h_out > 0 else h_in)
    cu.update_tag()


def road_end_frame(road, at_end=True, depsgraph=None):
    """(world position, horizontal heading in radians) of the road centreline's end or start."""
    samples = sample_bezier_curve(road.civil_props.target_curve, 1.0, use_spline_tilt=False, depsgraph=depsgraph)
    if len(samples) < 2:
        return None
    pos, tan, _, _ = samples[-1] if at_end else samples[0]
    return pos, math.atan2(tan.y, tan.x)


def snap_start_to_end(road, target, depsgraph=None):
    """Moves and turns road's guide curve so the road starts where target ends. True if it moved."""
    end = road_end_frame(target, True, depsgraph)
    start = road_end_frame(road, False, depsgraph)
    if end is None or start is None:
        return False
    (end_pos, end_h), (start_pos, start_h) = end, start
    turn = math.remainder(end_h - start_h, math.tau)
    if (end_pos - start_pos).length < 1e-4 and abs(turn) < 1e-5:
        return False
    xform = Matrix.Translation(end_pos) @ Matrix.Rotation(turn, 4, 'Z') @ Matrix.Translation(-start_pos)
    curve = road.civil_props.target_curve
    curve.matrix_world = xform @ curve.matrix_world
    return True


_following = set()


def follow_attached(road, depsgraph=None):
    """Puts `road` back on the end of the road it continues from, then rebuilds it."""
    p = road.civil_props
    if p.attached_to is None or p.target_curve is None or road.name in _following:
        return
    if p.attached_to.civil_props.target_curve in (None, p.target_curve):
        return
    _following.add(road.name)
    try:
        snap_start_to_end(road, p.attached_to, depsgraph)
        rebuild_civil_mesh(road, depsgraph=depsgraph)
    finally:
        _following.discard(road.name)


def rebuild_attached_roads(road, depsgraph=None):
    """Roads that continue from `road` follow its end (and theirs follow them, and so on)."""
    if road.name in _following and len(_following) > 50:
        return
    for o in bpy.data.objects:
        if o.type == 'MESH' and o.civil_props.is_civil_object and o.civil_props.attached_to == road:
            follow_attached(o, depsgraph)


def regenerate_from_segments(obj, context):
    p = obj.civil_props
    if not (p.is_civil_object and p.use_segments and p.target_curve and len(p.segments)):
        return
    write_segments_to_curve(p.target_curve, p.segments)
    dg = context.evaluated_depsgraph_get() if context else None
    rebuild_civil_mesh(obj, depsgraph=dg)


def update_segment(self, context):
    obj = self.id_data
    if obj and obj.type == 'MESH':
        regenerate_from_segments(obj, context)


class CIVIL_PG_Segment(bpy.types.PropertyGroup):
    length: bpy.props.FloatProperty(
        name="Length", default=60.0, min=1.0, max=2000.0, unit='LENGTH',
        description="Length of this piece of road, measured along its centreline",
        update=update_segment
    )
    bend: bpy.props.FloatProperty(
        name="Bend", default=0.0, min=-180.0, max=180.0, step=100, precision=1,
        description="How far the road turns over this piece, in degrees. 0 is a straight; "
                    "positive turns left, negative turns right",
        update=update_segment
    )
    rise: bpy.props.FloatProperty(
        name="Rise", default=0.0, min=-100.0, max=100.0, unit='LENGTH',
        description="Height gained (or lost, if negative) over this piece",
        update=update_segment
    )


STRUCTURE_KINDS = [
    ('BRIDGE', "Overpass", "Bridge this stretch, high enough to clear what's underneath (tracks, a road, a creek)",
     'MOD_ARRAY', 0),
    ('BANK', "Earth Bank", "Keep this stretch on an earth bank or in a cutting: never a bridge or a tunnel",
     'MOD_SMOOTH', 1),
    ('TUNNEL', "Tunnel", "Put this stretch in a tunnel", 'VIEW_PERSPECTIVE', 2),
]
STRUCTURE_LENGTH = {'BRIDGE': 40.0, 'BANK': 60.0, 'TUNNEL': 100.0}


class CIVIL_PG_Structure(bpy.types.PropertyGroup):
    """A stretch of road where you decide what gets built, instead of the terrain."""
    kind: bpy.props.EnumProperty(name="Type", items=STRUCTURE_KINDS, default='BRIDGE', update=update_civil_prop)
    center: bpy.props.FloatProperty(
        name="At", default=0.0, min=0.0, max=100000.0, unit='LENGTH',
        description="Distance along the road, from its start, to the middle of this stretch",
        update=update_civil_prop
    )
    length: bpy.props.FloatProperty(
        name="Length", default=40.0, min=4.0, max=5000.0, unit='LENGTH',
        description="How long this stretch is", update=update_civil_prop
    )
    clearance: bpy.props.FloatProperty(
        name="Clearance", default=7.0, min=0.0, max=60.0, unit='LENGTH',
        description="Free height under the bridge. About 7 m clears a railway, 5 m a road. "
                    "The road is lifted (within its grade limit) to make room",
        update=update_civil_prop
    )
    clear_span: bpy.props.FloatProperty(
        name="Clear Span", default=20.0, min=0.0, max=500.0, unit='LENGTH',
        description="Width in the middle of the bridge kept free of pillars, over the tracks or road below",
        update=update_civil_prop
    )


class CIVIL_PG_ObjectProps(bpy.types.PropertyGroup):
    is_civil_object: bpy.props.BoolProperty(name="Is Civil Object", default=False)

    target_curve: bpy.props.PointerProperty(
        name="Guide Curve",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'CURVE',
        update=update_civil_prop
    )
    resolution_step: bpy.props.FloatProperty(
        name="Mesh Step", default=2.0, min=0.5, max=20.0, unit='LENGTH',
        description="Distance between cross-sections along the curve. Smaller is smoother but heavier",
        update=update_civil_prop
    )

    # The road this one carries on from: its start stays on that road's end.
    attached_to: bpy.props.PointerProperty(
        name="Continues From",
        type=bpy.types.Object,
        description="This road's start stays on the end of this road, following it when it changes. "
                    "Clear it to move the road freely",
        poll=lambda self, o: (o.type == 'MESH' and o.civil_props.is_civil_object and o != self.id_data),
        update=lambda self, context: follow_attached(self.id_data, context.evaluated_depsgraph_get())
    )

    # Segment builder
    use_segments: bpy.props.BoolProperty(
        name="Shape From Segments", default=False,
        description="Drive the guide curve from the segment list below. Changing a segment "
                    "replaces the curve's shape, including any hand edits and Ctrl+T tilt",
        update=lambda self, context: regenerate_from_segments(self.id_data, context)
    )
    segments: bpy.props.CollectionProperty(type=CIVIL_PG_Segment)
    active_segment: bpy.props.IntProperty(name="Active Segment", default=0, min=0)

    # Curve Banking & Superelevation
    use_spline_tilt: bpy.props.BoolProperty(
        name="Use Curve Point Tilt (Ctrl+T)", default=True,
        description="Banks the road using the spline control point tilt angles set in Curve Edit Mode",
        update=update_civil_prop
    )
    auto_superelevation: bpy.props.BoolProperty(
        name="Auto-Bank Turns (Superelevation)", default=False,
        description="Automatically banks curves inward based on horizontal turn radius",
        update=update_civil_prop
    )
    superelevation_factor: bpy.props.FloatProperty(
        name="Auto-Bank Strength", default=1.0, min=0.0, max=5.0,
        update=update_civil_prop
    )
    max_bank_angle: bpy.props.FloatProperty(
        name="Max Auto-Bank (Deg)", default=10.0, min=1.0, max=35.0,
        update=update_civil_prop
    )

    # Lanes & Roadway
    lanes_left: bpy.props.IntProperty(
        name="Left Lanes", default=2, min=0, max=8, update=update_civil_prop
    )
    lanes_right: bpy.props.IntProperty(
        name="Right Lanes", default=2, min=0, max=8, update=update_civil_prop
    )
    lane_width: bpy.props.FloatProperty(
        name="Lane Width", default=3.6, min=2.0, max=6.0, unit='LENGTH', update=update_civil_prop
    )
    shoulder_width: bpy.props.FloatProperty(
        name="Shoulder Width", default=1.8, min=0.0, max=5.0, unit='LENGTH', update=update_civil_prop
    )
    median_width: bpy.props.FloatProperty(
        name="Median Width", default=1.2, min=0.0, max=15.0, unit='LENGTH', update=update_civil_prop
    )
    add_lane_markings: bpy.props.BoolProperty(
        name="Lane Markings", default=True, update=update_civil_prop
    )

    # Terrain Cut-and-Fill Embankments
    add_embankments: bpy.props.BoolProperty(
        name="Terrain Embankment Slopes (Berms)", default=False,
        description="Grades sloped earth berms from the road edge down to ground elevation Z",
        update=update_civil_prop
    )
    ground_level_z: bpy.props.FloatProperty(
        name="Ground Level Z", default=0.0, min=-100.0, max=100.0, unit='LENGTH',
        description="World Z of the ground plane, used by embankments, overpass piers and bridge towers",
        update=update_civil_prop
    )

    # Real-world terrain
    terrain_object: bpy.props.PointerProperty(
        name="Terrain", type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH' and not obj.civil_props.is_civil_object,
        description="Ground mesh the road is built on (an imported terrain or any mesh). "
                    "Piers, banks and tunnels then fit the real ground",
        update=update_civil_prop
    )
    follow_terrain: bpy.props.BoolProperty(
        name="Follow Terrain", default=True,
        description="Set the road's height from the ground, smoothed and held to the maximum grade. "
                    "Off: the guide curve's own height is used",
        update=update_civil_prop
    )
    terrain_offset: bpy.props.FloatProperty(
        name="Height Above Ground", default=0.3, min=-60.0, max=200.0, unit='LENGTH',
        description="How far the road sits above the smoothed ground",
        update=update_civil_prop
    )
    max_grade: bpy.props.FloatProperty(
        name="Max Grade %", default=6.0, min=0.5, max=30.0, precision=1,
        description="Steepest climb allowed, in percent (6% is typical for highways)",
        update=update_civil_prop
    )
    terrain_smoothing: bpy.props.FloatProperty(
        name="Smoothing", default=80.0, min=0.0, max=1000.0, unit='LENGTH',
        description="Length over which bumps in the ground are evened out",
        update=update_civil_prop
    )
    ramp_start: bpy.props.BoolProperty(
        name="Start at Ground", default=False,
        description="Ramp the start of the road down to the ground (within the max grade), "
                    "e.g. where an elevated highway comes down to meet the land",
        update=update_civil_prop
    )
    ramp_end: bpy.props.BoolProperty(
        name="End at Ground", default=False,
        description="Ramp the end of the road down to the ground (within the max grade)",
        update=update_civil_prop
    )
    auto_bridges: bpy.props.BoolProperty(
        name="Auto Bridges", default=True,
        description="Build a bridge with piers wherever the road is high above the ground",
        update=update_civil_prop
    )
    bridge_min_height: bpy.props.FloatProperty(
        name="Above", default=7.0, min=1.5, max=100.0, unit='LENGTH',
        description="Road height above the ground at which an earth bank becomes a bridge",
        update=update_civil_prop
    )
    auto_tunnels: bpy.props.BoolProperty(
        name="Auto Tunnels", default=True,
        description="Bore a tunnel wherever the road is deep below the ground",
        update=update_civil_prop
    )
    tunnel_min_depth: bpy.props.FloatProperty(
        name="Below", default=18.0, min=3.0, max=200.0, unit='LENGTH',
        description="Depth below the ground at which a cutting becomes a tunnel",
        update=update_civil_prop
    )
    structures: bpy.props.CollectionProperty(type=CIVIL_PG_Structure)
    active_structure: bpy.props.IntProperty(name="Active Structure", default=0, min=0)
    shape_terrain: bpy.props.BoolProperty(
        name="Cut Terrain", default=True,
        description="Lower the terrain mesh where the road passes through it, so hills don't cover the road",
        update=update_civil_prop
    )

    # Exit Lane Gore Markings & Wedge
    add_exit_gore: bpy.props.BoolProperty(
        name="Exit Lane Gore & Chevron Taper", default=False,
        description="Widens the roadway with an auxiliary exit taper and painted chevron gore markings",
        update=update_civil_prop
    )
    exit_gore_side: bpy.props.EnumProperty(
        name="Gore Side",
        items=[
            ('RIGHT', "Right Side Exit", "Exit gore taper on the right side"),
            ('LEFT', "Left Side Exit", "Exit gore taper on the left side"),
            ('BOTH', "Both Sides", "Symmetrical exit tapers on both sides"),
        ],
        default='RIGHT',
        update=update_civil_prop
    )
    exit_gore_start: bpy.props.FloatProperty(
        name="Start Factor", default=0.35, min=0.0, max=0.95, subtype='FACTOR',
        update=update_civil_prop
    )
    exit_gore_end: bpy.props.FloatProperty(
        name="End Factor", default=0.85, min=0.05, max=1.0, subtype='FACTOR',
        update=update_civil_prop
    )
    exit_gore_width: bpy.props.FloatProperty(
        name="Exit Flare Width", default=5.2, min=2.0, max=14.0, unit='LENGTH',
        update=update_civil_prop
    )

    # Barriers, Guardrails & Walls
    add_median_barrier: bpy.props.BoolProperty(
        name="Median Jersey Barrier", default=True, update=update_civil_prop
    )
    add_side_barriers: bpy.props.BoolProperty(
        name="Side Jersey Barriers", default=True, update=update_civil_prop
    )
    barrier_height: bpy.props.FloatProperty(
        name="Barrier Height", default=0.85, min=0.4, max=2.5, unit='LENGTH', update=update_civil_prop
    )
    add_guardrails: bpy.props.BoolProperty(
        name="W-Beam Metal Guardrails", default=False, update=update_civil_prop
    )
    guardrail_height: bpy.props.FloatProperty(
        name="Guardrail Height", default=0.78, min=0.5, max=1.5, unit='LENGTH', update=update_civil_prop
    )
    guardrail_post_spacing: bpy.props.FloatProperty(
        name="Post Spacing", default=4.0, min=1.5, max=12.0, unit='LENGTH', update=update_civil_prop
    )
    add_sidewalks: bpy.props.BoolProperty(
        name="Sidewalks / Curbs", default=False, update=update_civil_prop
    )
    sidewalk_width: bpy.props.FloatProperty(
        name="Sidewalk Width", default=2.0, min=0.5, max=6.0, unit='LENGTH', update=update_civil_prop
    )
    add_sound_walls: bpy.props.BoolProperty(
        name="Concrete Sound Walls", default=False, update=update_civil_prop
    )
    sound_wall_height: bpy.props.FloatProperty(
        name="Wall Height", default=4.0, min=1.5, max=12.0, unit='LENGTH', update=update_civil_prop
    )
    sound_wall_thickness: bpy.props.FloatProperty(
        name="Wall Thickness", default=0.35, min=0.15, max=1.5, unit='LENGTH', update=update_civil_prop
    )

    # Overhead Gantry Signs
    add_gantry_signs: bpy.props.BoolProperty(
        name="Overhead Highway Gantry Signs", default=False,
        description="Generates full-span steel truss gantries with green highway directional signboards",
        update=update_civil_prop
    )
    gantry_spacing: bpy.props.FloatProperty(
        name="Gantry Spacing", default=55.0, min=15.0, max=200.0, unit='LENGTH',
        update=update_civil_prop
    )
    gantry_clearance: bpy.props.FloatProperty(
        name="Vertical Clearance", default=6.2, min=4.5, max=12.0, unit='LENGTH',
        update=update_civil_prop
    )
    gantry_style: bpy.props.EnumProperty(
        name="Gantry Style",
        items=[
            ('FULL_SPAN', "Full Span", "One truss across the whole road on two posts"),
            ('CANTILEVER', "Single Sign (Cantilever)",
             "For each direction, one post beside the road with an arm reaching over its lanes and one sign"),
        ],
        default='FULL_SPAN',
        update=update_civil_prop
    )
    gantry_side: bpy.props.EnumProperty(
        name="Side",
        items=[
            ('BOTH', "Both Sides", "An arm from each side of the road, each over its own lanes"),
            ('RIGHT', "Right Side Only", "One post on the right edge with an arm and sign over the right lanes"),
            ('LEFT', "Left Side Only", "One post on the left edge with an arm and sign over the left lanes"),
        ],
        default='BOTH',
        update=update_civil_prop
    )
    gantry_sign_type: bpy.props.EnumProperty(
        name="Signs",
        items=[
            ('GUIDE', "Guide Signs", "Green direction signs"),
            ('LED', "LED Message Signs", "Electronic signs with amber messages about traffic and hazards"),
            ('ALTERNATE', "Alternate", "Guide signs and LED message signs on alternating gantries"),
        ],
        default='GUIDE',
        update=update_civil_prop
    )
    guide_sign_text: bpy.props.StringProperty(
        name="Guide Text",
        default="EXIT 24|Downtown;Airport|Next Right;Interstate 5|North",
        description="Lettering on guide signs. Use | for a new line and ; between signs; each sign "
                    "along the road takes the next one",
        update=update_civil_prop
    )
    led_sign_text: bpy.props.StringProperty(
        name="LED Messages",
        default="ACCIDENT AHEAD|LEFT LANE CLOSED;TRAFFIC AHEAD|EXPECT DELAYS;ROAD WORK|NEXT 2 MILES;"
                "SLOW DOWN|FOG AHEAD",
        description="Messages on LED signs. Use | for a new line and ; between messages; each sign "
                    "along the road shows the next one",
        update=update_civil_prop
    )
    gantry_sign_height: bpy.props.FloatProperty(
        name="Signboard Height", default=2.4, min=1.2, max=5.0, unit='LENGTH',
        update=update_civil_prop
    )

    # Streetlights
    add_streetlights: bpy.props.BoolProperty(
        name="Highway Streetlights", default=False, update=update_civil_prop
    )
    streetlight_placement: bpy.props.EnumProperty(
        name="Placement",
        items=[
            ('BOTH', "Both Sides", "Dual streetlights on left and right edges"),
            ('LEFT', "Left Side Only", "Streetlights along the left edge only"),
            ('RIGHT', "Right Side Only", "Streetlights along the right edge only"),
            ('ALTERNATING', "Alternating Staggered", "Staggered left/right streetlights"),
            ('MEDIAN', "Central Median Double-Mast", "Center twin-arm mast poles along the median"),
        ],
        default='BOTH',
        update=update_civil_prop
    )
    streetlight_style: bpy.props.EnumProperty(
        name="Light Style", items=utilities.LIGHT_STYLES, default='COBRA',
        update=lambda self, context: _streetlight_style_changed(self, context),
        description="Highway cobra heads, classical park and downtown lights, or modern LED blades"
    )
    streetlight_spacing: bpy.props.FloatProperty(
        name="Light Spacing", default=28.0, min=8.0, max=80.0, unit='LENGTH', update=update_civil_prop
    )
    streetlight_height: bpy.props.FloatProperty(
        name="Pole Height", default=9.5, min=4.0, max=20.0, unit='LENGTH', update=update_civil_prop
    )
    streetlight_arm_length: bpy.props.FloatProperty(
        name="Mast Arm Length", default=3.2, min=1.0, max=8.0, unit='LENGTH', update=update_civil_prop
    )

    # Cable-Stayed Bridge Towers
    add_cable_bridge: bpy.props.BoolProperty(
        name="Cable-Stayed Bridge Towers & Cables", default=False,
        description="Generates A-frame concrete bridge pylons and high-tension steel stay cables",
        update=update_civil_prop
    )
    cable_tower_spacing: bpy.props.FloatProperty(
        name="Tower Interval", default=45.0, min=20.0, max=120.0, unit='LENGTH',
        update=update_civil_prop
    )
    cable_tower_height: bpy.props.FloatProperty(
        name="Tower Height", default=22.0, min=10.0, max=50.0, unit='LENGTH',
        update=update_civil_prop
    )

    # Highway Toll Booth Plaza
    add_toll_plaza: bpy.props.BoolProperty(
        name="Highway Toll Booth Plaza", default=False,
        description="Generates a multi-lane toll collection plaza with concrete islands and canopy",
        update=update_civil_prop
    )
    toll_plaza_position: bpy.props.FloatProperty(
        name="Plaza Position", default=0.5, min=0.1, max=0.9, subtype='FACTOR',
        update=update_civil_prop
    )
    toll_island_length: bpy.props.FloatProperty(
        name="Island Length", default=16.0, min=8.0, max=35.0, unit='LENGTH',
        update=update_civil_prop
    )
    toll_booth_width: bpy.props.FloatProperty(
        name="Cabin Width", default=2.2, min=1.2, max=4.0, unit='LENGTH',
        update=update_civil_prop
    )

    # Concrete Tunnel Portals & Enclosure
    add_tunnel: bpy.props.BoolProperty(
        name="Concrete Tunnel Enclosure & Portals", default=False,
        description="Generates a reinforced concrete tunnel shell with entry/exit portal collars and ceiling lights",
        update=update_civil_prop
    )
    tunnel_profile: bpy.props.EnumProperty(
        name="Tunnel Style",
        items=[
            ('ARCHED', "Arched Vault Tunnel", "Bored mountain/urban arched concrete vault with portals"),
            ('BOX', "Cut-and-Cover Box Tunnel", "Rectangular reinforced concrete underpass box tunnel"),
        ],
        default='ARCHED',
        update=update_civil_prop
    )
    tunnel_start: bpy.props.FloatProperty(
        name="Tunnel Start", default=0.20, min=0.0, max=0.95, subtype='FACTOR',
        update=update_civil_prop
    )
    tunnel_end: bpy.props.FloatProperty(
        name="Tunnel End", default=0.80, min=0.05, max=1.0, subtype='FACTOR',
        update=update_civil_prop
    )
    tunnel_height: bpy.props.FloatProperty(
        name="Interior Clearance", default=6.8, min=4.5, max=14.0, unit='LENGTH',
        update=update_civil_prop
    )
    tunnel_wall_thickness: bpy.props.FloatProperty(
        name="Concrete Thickness", default=0.85, min=0.35, max=2.5, unit='LENGTH',
        update=update_civil_prop
    )

    # Overpass & Viaducts
    is_overpass: bpy.props.BoolProperty(
        name="Overpass Box-Girder Deck", default=False, update=update_civil_prop
    )
    deck_thickness: bpy.props.FloatProperty(
        name="Deck Depth", default=1.6, min=0.3, max=5.0, unit='LENGTH', update=update_civil_prop
    )
    add_pillars: bpy.props.BoolProperty(
        name="Generate Support Pillars", default=True, update=update_civil_prop
    )
    pillar_spacing: bpy.props.FloatProperty(
        name="Pillar Spacing", default=18.0, min=5.0, max=60.0, unit='LENGTH', update=update_civil_prop
    )
    pillar_width: bpy.props.FloatProperty(
        name="Pillar Width", default=2.8, min=0.8, max=15.0, unit='LENGTH', update=update_civil_prop
    )
    pillar_depth: bpy.props.FloatProperty(
        name="Pillar Depth", default=1.8, min=0.8, max=10.0, unit='LENGTH',
        description="Column thickness along the road (the diameter of round columns)",
        update=update_civil_prop
    )
    # Traffic (applied with the Generate Traffic button)
    traffic_density: bpy.props.FloatProperty(
        name="Density", default=1.5, min=0.1, max=8.0,
        description="Vehicles per lane for every 100 m of road"
    )
    traffic_speed: bpy.props.FloatProperty(
        name="Speed (km/h)", default=100.0, min=5.0, max=200.0,
        description="Typical speed. Kerb lanes run a little slower, inner lanes a little faster"
    )
    traffic_truck_share: bpy.props.FloatProperty(
        name="Trucks", default=0.2, min=0.0, max=1.0, subtype='FACTOR',
        description="Share of trucks among the vehicles; they keep mostly to the kerb lane"
    )
    traffic_seed: bpy.props.IntProperty(
        name="Seed", default=1, min=0,
        description="Change to get a different mix and spacing of vehicles"
    )
    traffic_left_hand: bpy.props.BoolProperty(
        name="Drive on Left", default=False,
        description="Left-hand traffic (UK, Japan, Australia...). Otherwise traffic keeps right"
    )
    traffic_duration: bpy.props.FloatProperty(
        name="Simulate (seconds)", default=60.0, min=5.0, max=1800.0,
        description="Where roads meet at intersections, traffic is worked out ahead of time for this long "
                    "(the scene's frame range grows to fit) and stored as keyframes"
    )
    traffic_collection: bpy.props.PointerProperty(type=bpy.types.Collection)

    pier_style: bpy.props.EnumProperty(
        name="Pier Style",
        items=[
            ('AUTO', "Auto", "Hammerhead piers under narrow decks, multi-column bents under wide ones"),
            ('HAMMERHEAD', "Hammerhead", "One wide column under a cantilevered cap beam"),
            ('MULTI', "Multi-Column Bent", "A row of round columns under a full-width cap beam"),
        ],
        default='AUTO',
        update=update_civil_prop
    )


# ==============================================================================
# 6. OPERATORS (PRESETS, BUILDERS & INTERCHANGE RAMPS)
# ==============================================================================

def curve_start_point(curve_obj):
    """Local position of the curve's first point."""
    sp = curve_obj.data.splines[0] if curve_obj.data.splines else None
    if sp is None:
        return Vector()
    if sp.type == 'BEZIER':
        return sp.bezier_points[0].co.copy()
    return sp.points[0].co.xyz


def create_guide_curve(context, name, points_coords):
    """Creates a 3D Bezier curve with auto-handles from a list of (x,y,z) tuples."""
    curve_data = bpy.data.curves.new(name=f"{name}_Curve", type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 24
    curve_data.twist_mode = 'MINIMUM'

    spline = curve_data.splines.new('BEZIER')
    spline.bezier_points.add(len(points_coords) - 1)

    for i, co in enumerate(points_coords):
        bp = spline.bezier_points[i]
        bp.co = Vector(co)
        bp.handle_left_type = 'AUTO'
        bp.handle_right_type = 'AUTO'

    curve_obj = bpy.data.objects.new(f"{name}_Path", curve_data)
    context.collection.objects.link(curve_obj)
    return curve_obj


class CIVIL_OT_CreateRoadPreset(bpy.types.Operator):
    bl_idname = "civil.create_road_preset"
    bl_label = "Create Civil Infrastructure"
    bl_description = "Generate procedural roads, cable-stayed bridges, toll plazas, tunnels, or embankments"
    bl_options = {'REGISTER', 'UNDO'}

    preset_type: bpy.props.EnumProperty(
        name="Preset",
        items=[
            ('HIGHWAY', "4-Lane Divided Highway + Berms & Toll", "Highway with earth embankments, toll plaza, and streetlights"),
            ('CABLE_BRIDGE', "Cable-Stayed Suspension Bridge", "Elevated viaduct with A-frame concrete pylons and steel stay cables"),
            ('TUNNEL', "Highway Tunnel & Portals", "4-lane highway passing through an arched concrete tunnel with LED strips"),
            ('OVERPASS', "Elevated Highway Overpass", "Elevated viaduct with concrete girder and support piers"),
            ('STREET', "2-Lane City Street", "Local road with sidewalks and streetlights"),
            ('BARRIER_WALL', "Standalone Jersey Barrier / Wall", "Dedicated concrete barrier or sound wall along curve"),
        ],
        default='HIGHWAY'
    )
    length: bpy.props.FloatProperty(
        name="Length", default=100.0, min=5.0, max=2000.0, unit='LENGTH',
        description="Length of the first piece of road (ignored when building on a selected curve)"
    )
    bend: bpy.props.FloatProperty(
        name="Bend", default=0.0, min=-180.0, max=180.0, step=100, precision=1,
        description="How far the first piece turns, in degrees. 0 is straight; positive turns left"
    )
    elevation: bpy.props.FloatProperty(
        name="Elevation", default=0.0, min=-100.0, max=200.0, unit='LENGTH',
        description="Height of the road above the 3D cursor. Left unset, each preset picks its own"
    )

    PRESET_ELEVATION = {'HIGHWAY': 3.5, 'CABLE_BRIDGE': 12.0, 'OVERPASS': 10.0}
    # Height above real ground when the road starts on a terrain.
    TERRAIN_OFFSET = {'CABLE_BRIDGE': 14.0, 'OVERPASS': 10.0}

    def execute(self, context):
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        active_obj = context.active_object
        use_segments = not (active_obj and active_obj.type == 'CURVE')
        elevation_set = self.properties.is_property_set("elevation")
        if not use_segments:
            curve_obj = active_obj
            start = curve_obj.matrix_world @ curve_start_point(curve_obj)
        else:
            start = context.scene.cursor.location.copy()
        on_terrain = terrain.terrain_under(context.scene, start)
        if use_segments:
            if not elevation_set:
                self.elevation = self.PRESET_ELEVATION.get(self.preset_type, 0.0)
            # The road starts at the 3D cursor heading +X; the segment chain gives its shape.
            curve_obj = create_guide_curve(context, self.preset_type.title(), [(0, 0, 0), (1, 0, 0)])
            curve_obj.location = context.scene.cursor.location + Vector((0.0, 0.0, self.elevation))
            if on_terrain:
                curve_obj.location.z = on_terrain[1] + 0.3

        mesh_data = bpy.data.meshes.new(f"Civil_{self.preset_type.title()}_Mesh")
        road_obj = bpy.data.objects.new(f"Civil_{self.preset_type.title()}", mesh_data)
        context.collection.objects.link(road_obj)

        # is_civil_object is switched on last so each assignment below doesn't trigger a full rebuild.
        p = road_obj.civil_props
        p.target_curve = curve_obj
        if use_segments:
            seg = p.segments.add()
            seg.length = self.length
            seg.bend = self.bend
            p.use_segments = True

        p.traffic_speed = 50.0 if self.preset_type == 'STREET' else 100.0
        if self.preset_type == 'STREET':
            p.lanes_left = 1
            p.lanes_right = 1
            p.shoulder_width = 0.5
            p.median_width = 0.0
            p.add_median_barrier = False
            p.add_side_barriers = False
            p.add_guardrails = False
            p.add_sidewalks = True
            p.add_streetlights = True
            p.streetlight_placement = 'ALTERNATING'
            p.is_overpass = False

        elif self.preset_type == 'HIGHWAY':
            p.lanes_left = 2
            p.lanes_right = 2
            p.shoulder_width = 2.0
            p.median_width = 1.2
            p.add_median_barrier = True
            p.add_side_barriers = False
            p.add_guardrails = True
            p.add_embankments = True
            p.ground_level_z = 0.0
            p.add_toll_plaza = True
            p.add_streetlights = True
            p.streetlight_placement = 'MEDIAN'
            p.auto_superelevation = True
            p.is_overpass = False

        elif self.preset_type == 'CABLE_BRIDGE':
            p.lanes_left = 2
            p.lanes_right = 2
            p.shoulder_width = 2.0
            p.median_width = 1.2
            p.add_median_barrier = True
            p.add_side_barriers = True
            p.add_cable_bridge = True
            p.cable_tower_spacing = 50.0
            p.cable_tower_height = 25.0
            p.is_overpass = True
            p.add_pillars = True  # end supports only; the towers carry the rest
            p.auto_superelevation = True

        elif self.preset_type == 'OVERPASS':
            p.lanes_left = 2
            p.lanes_right = 2
            p.shoulder_width = 1.8
            p.median_width = 1.0
            p.add_median_barrier = True
            p.add_side_barriers = True
            p.add_guardrails = False
            p.add_streetlights = True
            p.streetlight_placement = 'BOTH'
            p.add_gantry_signs = True
            p.auto_superelevation = True
            p.is_overpass = True
            p.add_pillars = True
            p.pillar_spacing = 18.0

        elif self.preset_type == 'TUNNEL':
            p.lanes_left = 2
            p.lanes_right = 2
            p.shoulder_width = 1.6
            p.median_width = 1.0
            p.add_median_barrier = True
            p.add_side_barriers = True
            p.add_guardrails = False
            p.add_streetlights = True
            p.add_tunnel = True
            p.tunnel_profile = 'ARCHED'
            p.tunnel_start = 0.22
            p.tunnel_end = 0.78

        elif self.preset_type == 'BARRIER_WALL':
            p.lanes_left = 0
            p.lanes_right = 0
            p.shoulder_width = 0.0
            p.median_width = 0.6
            p.add_median_barrier = True
            p.add_side_barriers = False
            p.add_lane_markings = False
            p.is_overpass = False

        if on_terrain:
            # Built over real land: follow it, and let the ground decide on banks, bridges and tunnels.
            p.terrain_object = on_terrain[0]
            p.follow_terrain = True
            p.terrain_offset = (self.elevation if elevation_set
                                else self.TERRAIN_OFFSET.get(self.preset_type, 0.3))
            p.add_embankments = self.preset_type not in ('BARRIER_WALL',)
            p.add_toll_plaza = False

        p.is_civil_object = True
        if use_segments:
            write_segments_to_curve(curve_obj, p.segments)
        dg = context.evaluated_depsgraph_get()
        rebuild_civil_mesh(road_obj, depsgraph=dg)

        for o in context.selected_objects:
            o.select_set(False)
        road_obj.select_set(True)
        context.view_layer.objects.active = road_obj
        return {'FINISHED'}


LANE_SETUP_PROPS = ("lanes_left", "lanes_right", "lane_width", "shoulder_width", "median_width",
                    "traffic_speed", "traffic_left_hand")
TERRAIN_PROPS = ("terrain_object", "follow_terrain", "max_grade", "terrain_smoothing", "auto_bridges",
                 "bridge_min_height", "auto_tunnels", "tunnel_min_depth", "shape_terrain")


class CIVIL_OT_RoadContinue(bpy.types.Operator):
    bl_idname = "civil.road_continue"
    bl_label = "New Road From End"
    bl_description = ("Start a new road where the active road ends, heading the same way. "
                      "Its traffic will flow on from this road")
    bl_options = {'REGISTER', 'UNDO'}

    preset_type: bpy.props.EnumProperty(
        name="Road Type",
        items=[
            ('HIGHWAY', "Highway", "Divided highway with embankments"),
            ('OVERPASS', "Overpass", "Elevated deck on piers"),
            ('CABLE_BRIDGE', "Cable-Stayed Bridge", "Deck hung from A-frame towers"),
            ('TUNNEL', "Tunnel", "Road through an arched tunnel"),
            ('STREET', "City Street", "Two-lane street with sidewalks"),
        ],
        default='HIGHWAY'
    )
    length: bpy.props.FloatProperty(name="Length", default=100.0, min=5.0, max=2000.0, unit='LENGTH')
    bend: bpy.props.FloatProperty(name="Bend", default=0.0, min=-180.0, max=180.0, step=100, precision=1)
    rise: bpy.props.FloatProperty(name="Rise", default=0.0, min=-100.0, max=100.0, unit='LENGTH',
                                  description="Height gained (or lost) over the new road")
    match_lanes: bpy.props.BoolProperty(
        name="Match Lanes", default=True,
        description="Give the new road the same lanes, widths and traffic settings as the one it continues"
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return bool(obj and obj.type == 'MESH' and obj.civil_props.is_civil_object and obj.civil_props.target_curve)

    def execute(self, context):
        prev = context.active_object
        frame = road_end_frame(prev, at_end=True)
        if frame is None:
            self.report({'WARNING'}, "The active road has no usable guide curve.")
            return {'CANCELLED'}
        end_pos, heading = frame

        for o in context.selected_objects:
            o.select_set(False)
        context.view_layer.objects.active = None
        bpy.ops.civil.create_road_preset(preset_type=self.preset_type, length=self.length,
                                         bend=self.bend, elevation=0.0)
        road = context.active_object
        p = road.civil_props
        p.is_civil_object = False  # hold rebuilds while the new road is set up
        try:
            if self.match_lanes:
                for name in LANE_SETUP_PROPS:
                    setattr(p, name, getattr(prev.civil_props, name))
            if len(p.segments):
                p.segments[0].rise = self.rise
            p.add_toll_plaza = False  # one plaza per highway is plenty
            pp = prev.civil_props
            if pp.terrain_object:
                for name in TERRAIN_PROPS:
                    setattr(p, name, getattr(pp, name))
                p.add_embankments = self.preset_type != 'BARRIER_WALL'
                lifted = CIVIL_OT_CreateRoadPreset.TERRAIN_OFFSET
                if self.preset_type in lifted:
                    p.terrain_offset = lifted[self.preset_type]  # ramps up from the previous road's end
                else:
                    p.terrain_offset = pp.terrain_offset if not pp.is_overpass else 0.3
            curve = p.target_curve
            curve.location = end_pos
            curve.rotation_euler = (0.0, 0.0, heading)
            p.attached_to = prev
        finally:
            p.is_civil_object = True
        if len(p.segments):
            write_segments_to_curve(p.target_curve, p.segments)
        rebuild_civil_mesh(road, depsgraph=context.evaluated_depsgraph_get())
        return {'FINISHED'}


class CIVIL_OT_RoadSnap(bpy.types.Operator):
    bl_idname = "civil.road_snap"
    bl_label = "Join Roads"
    bl_description = ("Select two roads: moves and turns the active road so it starts exactly where "
                      "the other one ends, and keeps it there when the other road changes. "
                      "Their lanes and traffic link up")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        sel = [o for o in context.selected_objects if o.type == 'MESH' and o.civil_props.is_civil_object]
        act = context.active_object
        return len(sel) == 2 and act in sel

    def execute(self, context):
        act = context.active_object
        other = next(o for o in context.selected_objects if o != act and o.type == 'MESH')
        if act.civil_props.target_curve == other.civil_props.target_curve:
            self.report({'WARNING'}, "Both roads use the same guide curve.")
            return {'CANCELLED'}
        if road_end_frame(other) is None or road_end_frame(act, False) is None:
            self.report({'WARNING'}, "Both roads need a usable guide curve.")
            return {'CANCELLED'}
        # Attaching snaps the road onto the other's end and keeps it there from now on.
        act.civil_props.attached_to = other
        return {'FINISHED'}


def active_segment_road(context):
    obj = context.active_object
    if obj and obj.type == 'MESH' and obj.civil_props.is_civil_object and obj.civil_props.use_segments:
        return obj
    return None


class CIVIL_OT_SegmentAdd(bpy.types.Operator):
    bl_idname = "civil.segment_add"
    bl_label = "Extend Road"
    bl_description = "Add a new piece to the end of the road"
    bl_options = {'REGISTER', 'UNDO'}

    kind: bpy.props.EnumProperty(
        name="Kind",
        items=[
            ('STRAIGHT', "Straight", "Add a straight"),
            ('LEFT', "Curve Left", "Add a curve turning left"),
            ('RIGHT', "Curve Right", "Add a curve turning right"),
        ],
        default='STRAIGHT'
    )
    length: bpy.props.FloatProperty(name="Length", default=60.0, min=1.0, max=2000.0, unit='LENGTH')
    bend: bpy.props.FloatProperty(name="Bend", default=45.0, min=0.0, max=180.0, step=100, precision=1,
                                  description="Turn angle for a curve, in degrees")
    rise: bpy.props.FloatProperty(name="Rise", default=0.0, min=-100.0, max=100.0, unit='LENGTH')

    @classmethod
    def poll(cls, context):
        return active_segment_road(context) is not None

    def execute(self, context):
        obj = active_segment_road(context)
        p = obj.civil_props
        p.is_civil_object = False  # hold rebuilds while the new segment is filled in
        try:
            seg = p.segments.add()
            seg.length = self.length
            seg.bend = 0.0 if self.kind == 'STRAIGHT' else (self.bend if self.kind == 'LEFT' else -self.bend)
            seg.rise = self.rise
            p.active_segment = len(p.segments) - 1
        finally:
            p.is_civil_object = True
        regenerate_from_segments(obj, context)
        return {'FINISHED'}


class CIVIL_OT_SegmentRemove(bpy.types.Operator):
    bl_idname = "civil.segment_remove"
    bl_label = "Remove Segment"
    bl_description = "Remove the selected piece of road"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = active_segment_road(context)
        return obj is not None and len(obj.civil_props.segments) > 1

    def execute(self, context):
        obj = active_segment_road(context)
        p = obj.civil_props
        idx = min(p.active_segment, len(p.segments) - 1)
        p.segments.remove(idx)
        p.active_segment = max(0, min(idx, len(p.segments) - 1))
        regenerate_from_segments(obj, context)
        return {'FINISHED'}


class CIVIL_UL_Segments(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        if abs(item.bend) < 0.05:
            seg_icon, kind = 'REMOVE', "Straight"
        else:
            seg_icon = 'LOOP_BACK' if item.bend > 0 else 'LOOP_FORWARDS'
            kind = f"{'Left' if item.bend > 0 else 'Right'} {abs(item.bend):.0f}°"
        row.label(text=f"{index + 1}. {kind}", icon=seg_icon)
        row.label(text=f"{item.length:.0f} m")


def road_distance_at(road, world_pt):
    """(distance along the road, horizontal distance off it) of the road point nearest world_pt."""
    samples = road_local_samples(road)
    if len(samples) < 2:
        return None
    M = road.matrix_world
    pts = np.array([tuple(M @ sp[0]) for sp in samples])
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1))])
    off = np.hypot(pts[:, 0] - world_pt[0], pts[:, 1] - world_pt[1])
    i = int(np.argmin(off))
    return float(d[i]), float(off[i])


def active_terrain_road(context):
    obj = context.active_object
    if obj and obj.type == 'MESH' and obj.civil_props.is_civil_object and obj.civil_props.target_curve:
        return obj
    return None


class CIVIL_OT_StructureAdd(bpy.types.Operator):
    bl_idname = "civil.structure_add"
    bl_label = "Add Structure at Cursor"
    bl_description = ("Decide what gets built where the 3D cursor is on this road (Shift+right-click "
                      "the spot first): an overpass, an earth bank or a tunnel")
    bl_options = {'REGISTER', 'UNDO'}

    kind: bpy.props.EnumProperty(name="Type", items=STRUCTURE_KINDS, default='BRIDGE')

    @classmethod
    def poll(cls, context):
        return active_terrain_road(context) is not None

    def execute(self, context):
        road = active_terrain_road(context)
        hit = road_distance_at(road, context.scene.cursor.location)
        if hit is None:
            return {'CANCELLED'}
        at, off = hit
        if off > 60.0:
            self.report({'WARNING'}, "Put the 3D cursor on the road first (Shift+right-click the spot)")
            return {'CANCELLED'}
        p = road.civil_props
        p.is_civil_object = False  # one rebuild once it's all set
        try:
            st = p.structures.add()
            st.kind = self.kind
            st.center = at
            st.length = STRUCTURE_LENGTH[self.kind]
            p.active_structure = len(p.structures) - 1
        finally:
            p.is_civil_object = True
        rebuild_civil_mesh(road, depsgraph=context.evaluated_depsgraph_get())
        self.report({'INFO'}, "%s added %.0f m along the road" % (st.bl_rna.properties["kind"].enum_items[self.kind].name, at))
        return {'FINISHED'}


class CIVIL_OT_StructureRemove(bpy.types.Operator):
    bl_idname = "civil.structure_remove"
    bl_label = "Remove Structure"
    bl_description = "Remove the selected structure; the terrain decides again there"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        road = active_terrain_road(context)
        return road is not None and len(road.civil_props.structures) > 0

    def execute(self, context):
        road = active_terrain_road(context)
        p = road.civil_props
        idx = min(p.active_structure, len(p.structures) - 1)
        p.structures.remove(idx)
        p.active_structure = max(0, min(idx, len(p.structures) - 1))
        rebuild_civil_mesh(road, depsgraph=context.evaluated_depsgraph_get())
        return {'FINISHED'}


class CIVIL_UL_Structures(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        kind = next(k for k in STRUCTURE_KINDS if k[0] == item.kind)
        row = layout.row(align=True)
        row.label(text=kind[1], icon=kind[3])
        row.label(text=f"at {item.center:.0f} m, {item.length:.0f} m long")


class CIVIL_OT_CreateInterchangeRamp(bpy.types.Operator):
    bl_idname = "civil.create_interchange_ramp"
    bl_label = "Connect Roads with Ramp"
    bl_description = ("Select two Civil Roads (or two Curves) to join their nearest ends with a smooth banked ramp. "
                      "The ramp leaves the right-hand lane of the other road and enters the active one")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sel = context.selected_objects
        if len(sel) < 2:
            self.report({'WARNING'}, "Select two Civil Roads or two Guide Curves first.")
            return {'CANCELLED'}

        def get_curve(o):
            if o.type == 'CURVE':
                return o
            if o.type == 'MESH' and o.civil_props.is_civil_object and o.civil_props.target_curve:
                return o.civil_props.target_curve
            return None

        c1 = get_curve(sel[0])
        c2 = get_curve(sel[1])

        if len(sel) > 2:
            self.report({'WARNING'}, "Select exactly two Civil Roads or Guide Curves.")
            return {'CANCELLED'}
        if not c1 or not c2 or c1 == c2:
            self.report({'WARNING'}, "Could not find two distinct guide curves from selection.")
            return {'CANCELLED'}

        # The active object is the destination; the other selected one is where the ramp leaves.
        active = context.active_object
        src_obj, dst_obj = sel[0], sel[1]
        if active is not None and active in (sel[0], sel[1]):
            dst_obj = active
            src_obj = sel[1] if active is sel[0] else sel[0]
        c1, c2 = get_curve(src_obj), get_curve(dst_obj)

        dg = context.evaluated_depsgraph_get()
        s1 = sample_bezier_curve(c1, 2.0, depsgraph=dg)
        s2 = sample_bezier_curve(c2, 2.0, depsgraph=dg)
        if not s1 or not s2:
            return {'CANCELLED'}

        # Join the closest pair of road ends: leave road 1 through one of its ends and
        # enter road 2 through one of its ends, travelling with the roads' direction.
        best = None
        for i1, out_sign in ((0, -1.0), (len(s1) - 1, 1.0)):
            for i2, in_sign in ((0, 1.0), (len(s2) - 1, -1.0)):
                d = (s2[i2][0] - s1[i1][0]).length
                if best is None or d < best[0]:
                    best = (d, i1, out_sign, i2, in_sign)
        _, i1, out_sign, i2, in_sign = best

        def right_lane_offset(o, sample, travel_sign):
            """World offset from the centreline to the outermost lane on the right of travel."""
            if o.type != 'MESH' or not o.civil_props.is_civil_object:
                return Vector((0.0, 0.0, 0.0))
            p = o.civil_props
            med = p.median_width if (p.lanes_left > 0 and p.lanes_right > 0) else 0.0
            hw = ((p.lanes_left + p.lanes_right) * p.lane_width + med + p.shoulder_width * 2.0) * 0.5
            # Travelling along the curve, the right-hand lanes are the curve's right lanes;
            # travelling against it, they are the curve's left lanes.
            lanes = p.lanes_right if travel_sign > 0 else p.lanes_left
            if lanes <= 0:
                return Vector((0.0, 0.0, 0.0))
            x = hw - p.shoulder_width - p.lane_width * 0.5
            return sample[3] * (x * travel_sign)

        start_pos = s1[i1][0] + right_lane_offset(src_obj, s1[i1], out_sign)
        start_tan = s1[i1][1] * out_sign
        end_pos = s2[i2][0] + right_lane_offset(dst_obj, s2[i2], in_sign)
        end_tan = s2[i2][1] * in_sign

        dist = (end_pos - start_pos).length
        if dist < 1.0:
            self.report({'WARNING'}, "The selected roads' ends already touch; nothing to connect.")
            return {'CANCELLED'}
        handle_len = max(8.0, dist * 0.35)

        curve_data = bpy.data.curves.new(name="Interchange_Ramp_Curve", type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = 32
        spline = curve_data.splines.new('BEZIER')
        spline.bezier_points.add(1)

        bp0 = spline.bezier_points[0]
        bp0.co = start_pos
        bp0.handle_left_type = 'FREE'
        bp0.handle_right_type = 'FREE'
        bp0.handle_left = start_pos - start_tan * handle_len
        bp0.handle_right = start_pos + start_tan * handle_len

        bp1 = spline.bezier_points[1]
        bp1.co = end_pos
        bp1.handle_left_type = 'FREE'
        bp1.handle_right_type = 'FREE'
        bp1.handle_left = end_pos - end_tan * handle_len
        bp1.handle_right = end_pos + end_tan * handle_len

        ramp_curve = bpy.data.objects.new("Ramp_Guide_Curve", curve_data)
        context.collection.objects.link(ramp_curve)

        mesh_data = bpy.data.meshes.new("Civil_Interchange_Ramp_Mesh")
        ramp_obj = bpy.data.objects.new("Civil_Interchange_Ramp", mesh_data)
        context.collection.objects.link(ramp_obj)

        p = ramp_obj.civil_props
        p.target_curve = ramp_curve
        p.lanes_left = 0
        p.lanes_right = 1
        p.lane_width = 4.2
        p.shoulder_width = 1.2
        p.add_median_barrier = False
        p.add_side_barriers = False
        p.add_guardrails = True
        p.auto_superelevation = True
        p.superelevation_factor = 1.4
        p.traffic_speed = 60.0
        p.traffic_truck_share = 0.1
        p.is_overpass = True
        p.add_pillars = True
        p.pillar_spacing = 15.0
        p.is_civil_object = True

        dg = context.evaluated_depsgraph_get()
        rebuild_civil_mesh(ramp_obj, depsgraph=dg)

        for o in context.selected_objects:
            o.select_set(False)
        ramp_obj.select_set(True)
        context.view_layer.objects.active = ramp_obj
        return {'FINISHED'}


class CIVIL_OT_RefreshAll(bpy.types.Operator):
    bl_idname = "civil.refresh_all"
    bl_label = "Rebuild All Civil Roads"
    bl_description = "Re-evaluates all civil engineering meshes after editing guide curves"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        dg = context.evaluated_depsgraph_get()
        terrain.clear_cache()
        count = 0
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and obj.civil_props.is_civil_object:
                rebuild_civil_mesh(obj, depsgraph=dg)
                count += 1
        self.report({'INFO'}, f"Updated {count} Civil Infrastructure objects.")
        return {'FINISHED'}


class CIVIL_OT_SelectGuideCurve(bpy.types.Operator):
    bl_idname = "civil.select_guide_curve"
    bl_label = "Edit Guide Curve (Ctrl+T to Bank)"
    bl_description = "Selects the underlying Bezier curve and switches to Edit Mode"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if obj and obj.civil_props.is_civil_object and obj.civil_props.target_curve:
            curve = obj.civil_props.target_curve
            if context.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            for o in context.selected_objects:
                o.select_set(False)
            curve.select_set(True)
            context.view_layer.objects.active = curve
            bpy.ops.object.mode_set(mode='EDIT')
        return {'FINISHED'}


# ==============================================================================
# 7. UI PANEL IN 3D VIEWPORT SIDEBAR (N-PANEL)
# ==============================================================================

class CIVIL_PT_MainPanel(bpy.types.Panel):
    bl_label = "CivilFlow Infrastructure"
    bl_idname = "CIVIL_PT_MainPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'CivilEng'
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        box = layout.box()
        box.label(text="Generate Infrastructure", icon='MOD_BUILD')
        col = box.column(align=True)
        op = col.operator("civil.create_road_preset", text="4-Lane Highway + Berms & Toll", icon='AUTO')
        op.preset_type = 'HIGHWAY'
        op = col.operator("civil.create_road_preset", text="Cable-Stayed Bridge Pylons", icon='MOD_PARTICLES')
        op.preset_type = 'CABLE_BRIDGE'
        op = col.operator("civil.create_road_preset", text="Elevated Overpass", icon='MOD_ARRAY')
        op.preset_type = 'OVERPASS'
        op = col.operator("civil.create_road_preset", text="Highway Tunnel & Portals", icon='VIEW_PERSPECTIVE')
        op.preset_type = 'TUNNEL'
        op = col.operator("civil.create_road_preset", text="2-Lane City Street", icon='MESH_PLANE')
        op.preset_type = 'STREET'
        op = col.operator("civil.create_road_preset", text="Jersey Barrier Wall", icon='MESH_CUBE')
        op.preset_type = 'BARRIER_WALL'

        layout.separator()
        row = layout.row(align=True)
        row.operator("civil.create_interchange_ramp", text="Bridge Selected Roads (Ramp)", icon='IPO_BEZIER')
        layout.operator("civil.road_snap", text="Join Active Road to Other's End", icon='SNAP_ON')
        layout.operator("civil.refresh_all", text="Rebuild All From Curves", icon='FILE_REFRESH')

        if obj and obj.type == 'MESH' and obj.civil_props.is_civil_object:
            props = obj.civil_props
            layout.separator()

            p_box = layout.box()
            p_box.label(text=f"Active: {obj.name}", icon='PREFERENCES')
            p_box.prop(props, "target_curve")
            p_box.operator("civil.select_guide_curve", icon='CURVE_BEZCURVE')
            p_box.prop(props, "resolution_step")

            # Road shape: chain of straights and curves
            r_box = layout.box()
            r_box.label(text="Road Shape", icon='CURVE_PATH')
            if len(props.segments):
                r_box.prop(props, "use_segments")
            if props.use_segments and len(props.segments):
                row = r_box.row()
                row.template_list("CIVIL_UL_Segments", "", props, "segments", props, "active_segment", rows=3)
                col = row.column(align=True)
                col.operator("civil.segment_remove", icon='REMOVE', text="")
                idx = min(props.active_segment, len(props.segments) - 1)
                seg = props.segments[idx]
                col = r_box.column(align=True)
                col.prop(seg, "length")
                col.prop(seg, "bend", slider=True)
                col.prop(seg, "rise")
                r_box.label(text="Extend Road:")
                row = r_box.row(align=True)
                op = row.operator("civil.segment_add", text="Left", icon='LOOP_BACK')
                op.kind = 'LEFT'
                op = row.operator("civil.segment_add", text="Straight", icon='REMOVE')
                op.kind = 'STRAIGHT'
                op = row.operator("civil.segment_add", text="Right", icon='LOOP_FORWARDS')
                op.kind = 'RIGHT'
            elif not len(props.segments):
                r_box.label(text="Shaped by hand on its guide curve", icon='INFO')
            r_box.operator_menu_enum("civil.road_continue", "preset_type", text="New Road From End", icon='FORWARD')
            r_box.prop(props, "attached_to", icon='LINKED')

            # Real ground
            g_box = layout.box()
            g_box.label(text="Terrain", icon='WORLD')
            g_box.prop(props, "terrain_object")
            if props.terrain_object:
                g_box.prop(props, "follow_terrain")
                if props.follow_terrain:
                    col = g_box.column(align=True)
                    col.prop(props, "terrain_offset")
                    col.prop(props, "max_grade")
                    col.prop(props, "terrain_smoothing")
                    row = g_box.row(align=True)
                    row.prop(props, "ramp_start", toggle=True, icon='TRIA_DOWN_BAR')
                    row.prop(props, "ramp_end", toggle=True, icon='TRIA_DOWN_BAR')
                row = g_box.row(align=True)
                row.prop(props, "auto_bridges")
                sub = row.row(align=True)
                sub.enabled = props.auto_bridges and not props.is_overpass
                sub.prop(props, "bridge_min_height")
                row = g_box.row(align=True)
                row.prop(props, "auto_tunnels")
                sub = row.row(align=True)
                sub.enabled = props.auto_tunnels
                sub.prop(props, "tunnel_min_depth")
                row = g_box.row(align=True)
                row.prop(props, "add_embankments", text="Earth Banks")
                row.prop(props, "shape_terrain")

                s_box = g_box.box()
                s_box.label(text="Your Structures (at the 3D cursor)", icon='PIVOT_CURSOR')
                row = s_box.row(align=True)
                for kind_id, label, _d, kind_icon, _n in STRUCTURE_KINDS:
                    op = row.operator("civil.structure_add", text=label, icon=kind_icon)
                    op.kind = kind_id
                if len(props.structures):
                    row = s_box.row()
                    row.template_list("CIVIL_UL_Structures", "", props, "structures", props,
                                      "active_structure", rows=2)
                    row.operator("civil.structure_remove", icon='REMOVE', text="")
                    st = props.structures[min(props.active_structure, len(props.structures) - 1)]
                    col = s_box.column(align=True)
                    col.prop(st, "kind")
                    col.prop(st, "center")
                    col.prop(st, "length")
                    if st.kind == 'BRIDGE':
                        col.prop(st, "clearance")
                        col.prop(st, "clear_span")
                else:
                    s_box.label(text="Shift+right-click a spot on the road, then pick one")

            # Banking & Superelevation
            t_box = layout.box()
            t_box.label(text="Banking & Superelevation", icon='ORIENTATION_GIMBAL')
            t_box.prop(props, "use_spline_tilt")
            t_box.prop(props, "auto_superelevation")
            if props.auto_superelevation:
                row = t_box.row(align=True)
                row.prop(props, "superelevation_factor")
                row.prop(props, "max_bank_angle")

            # Roadway, Lanes & Embankments
            l_box = layout.box()
            l_box.label(text="Roadway, Lanes & Earthwork", icon='DRIVER_DISTANCE')
            row = l_box.row(align=True)
            row.prop(props, "lanes_left")
            row.prop(props, "lanes_right")
            l_box.prop(props, "lane_width")
            l_box.prop(props, "shoulder_width")
            l_box.prop(props, "median_width")
            l_box.prop(props, "add_lane_markings")

            l_box.separator()
            l_box.prop(props, "add_embankments")
            l_box.prop(props, "ground_level_z")

            l_box.separator()
            l_box.prop(props, "add_exit_gore")
            if props.add_exit_gore:
                l_box.prop(props, "exit_gore_side")
                row = l_box.row(align=True)
                row.prop(props, "exit_gore_start")
                row.prop(props, "exit_gore_end")
                l_box.prop(props, "exit_gore_width")

            # Barriers, Guardrails & Walls
            b_box = layout.box()
            b_box.label(text="Barriers, Guardrails & Walls", icon='MOD_SOLIDIFY')
            b_box.prop(props, "add_median_barrier")
            b_box.prop(props, "add_side_barriers")
            if props.add_median_barrier or props.add_side_barriers:
                b_box.prop(props, "barrier_height")

            b_box.prop(props, "add_guardrails")
            if props.add_guardrails:
                row = b_box.row(align=True)
                row.prop(props, "guardrail_height")
                row.prop(props, "guardrail_post_spacing")

            b_box.prop(props, "add_sidewalks")
            if props.add_sidewalks:
                b_box.prop(props, "sidewalk_width")

            b_box.prop(props, "add_sound_walls")
            if props.add_sound_walls:
                row = b_box.row(align=True)
                row.prop(props, "sound_wall_height")
                row.prop(props, "sound_wall_thickness")

            # Bridges, Toll Plazas & Tunnels
            s_box = layout.box()
            s_box.label(text="Bridges, Tolls & Tunnels", icon='MOD_BUILD')
            s_box.prop(props, "add_cable_bridge")
            if props.add_cable_bridge:
                row = s_box.row(align=True)
                row.prop(props, "cable_tower_spacing")
                row.prop(props, "cable_tower_height")

            s_box.separator()
            s_box.prop(props, "add_toll_plaza")
            if props.add_toll_plaza:
                s_box.prop(props, "toll_plaza_position")
                row = s_box.row(align=True)
                row.prop(props, "toll_island_length")
                row.prop(props, "toll_booth_width")

            s_box.separator()
            s_box.prop(props, "add_tunnel")
            if props.add_tunnel:
                s_box.prop(props, "tunnel_profile")
                row = s_box.row(align=True)
                row.prop(props, "tunnel_start")
                row.prop(props, "tunnel_end")
                row = s_box.row(align=True)
                row.prop(props, "tunnel_height")
                row.prop(props, "tunnel_wall_thickness")

            # Gantry Signs & Streetlights
            g_box = layout.box()
            g_box.label(text="Gantry Signs & Streetlights", icon='LIGHT_SPOT')
            g_box.prop(props, "add_gantry_signs")
            if props.add_gantry_signs:
                g_box.prop(props, "gantry_style")
                if props.gantry_style == 'CANTILEVER':
                    g_box.prop(props, "gantry_side")
                g_box.prop(props, "gantry_sign_type")
                if props.gantry_sign_type in ('GUIDE', 'ALTERNATE'):
                    g_box.prop(props, "guide_sign_text", text="Guide")
                if props.gantry_sign_type in ('LED', 'ALTERNATE'):
                    g_box.prop(props, "led_sign_text", text="LED")
                g_box.prop(props, "gantry_spacing")
                row = g_box.row(align=True)
                row.prop(props, "gantry_clearance")
                row.prop(props, "gantry_sign_height")

            g_box.separator()
            g_box.prop(props, "add_streetlights")
            if props.add_streetlights:
                g_box.prop(props, "streetlight_style")
                g_box.prop(props, "streetlight_placement")
                g_box.prop(props, "streetlight_spacing")
                row = g_box.row(align=True)
                row.prop(props, "streetlight_height")
                row.prop(props, "streetlight_arm_length")

            # Traffic
            tr_box = layout.box()
            tr_box.label(text="Traffic", icon='AUTO')
            col = tr_box.column(align=True)
            col.prop(props, "traffic_density")
            col.prop(props, "traffic_speed")
            col.prop(props, "traffic_truck_share", slider=True)
            row = tr_box.row(align=True)
            row.prop(props, "traffic_seed")
            row.prop(props, "traffic_left_hand", toggle=True)
            on_junction = any(junctions.is_junction(o) and obj in (o.civil_junction.road_a, o.civil_junction.road_b)
                              for o in context.scene.objects)
            if on_junction:
                tr_box.prop(props, "traffic_duration")
            row = tr_box.row(align=True)
            has_traffic = props.traffic_collection is not None
            row.operator("civil.traffic_generate", icon='PLAY',
                         text="Regenerate Traffic" if has_traffic else "Generate Traffic")
            if has_traffic:
                row.operator("civil.traffic_remove", icon='TRASH', text="")
                n = sum(1 for o in props.traffic_collection.objects if o.get("cf_kind"))
                n_roads = len(traffic.roads_sharing_traffic(props.traffic_collection))
                linked = f" on {n_roads} linked roads" if n_roads > 1 else ""
                tr_box.label(text=f"{n} vehicles{linked}. Press Space to play.", icon='INFO')
                if props.traffic_collection.get("cf_simulated"):
                    tr_box.label(text="Stops, signals and turns are worked out ahead:", icon='TIME')
                    tr_box.label(text="regenerate after changing roads or intersections")
            else:
                tr_box.label(text="Roads whose ends meet share their traffic", icon='LINKED')

            # Overpass Piers
            o_box = layout.box()
            o_box.label(text="Overpass Viaduct Piers", icon='MOD_ARCHIMEDES')
            o_box.prop(props, "is_overpass")
            o_box.prop(props, "deck_thickness")
            if props.is_overpass:
                o_box.prop(props, "add_pillars",
                           text="End Supports" if props.add_cable_bridge else "Generate Support Pillars")
                if props.add_pillars:
                    o_box.prop(props, "pier_style")
                    if not props.add_cable_bridge:
                        o_box.prop(props, "pillar_spacing")
                    row = o_box.row(align=True)
                    row.prop(props, "pillar_width")
                    row.prop(props, "pillar_depth")


# ==============================================================================
# 8. REGISTRATION & DEPSGRAPH HANDLER
# ==============================================================================

_is_rebuilding = False

@bpy.app.handlers.persistent
def civil_depsgraph_update_handler(scene, depsgraph):
    """Updates road geometry when a Guide Curve is edited."""
    global _is_rebuilding
    if _is_rebuilding:
        return

    updated_curves = set()
    for update in depsgraph.updates:
        if isinstance(update.id, bpy.types.Curve):
            updated_curves.add(update.id.original)
        elif isinstance(update.id, bpy.types.Object) and update.id.type == 'CURVE':
            # Curve objects are also listed when only something that depends on them changed.
            if update.is_updated_transform or update.is_updated_geometry:
                updated_curves.add(update.id.original)

    if not updated_curves:
        return

    _is_rebuilding = True
    try:
        for obj in scene.objects:
            if obj.type == 'MESH' and obj.civil_props.is_civil_object:
                t_curve = obj.civil_props.target_curve
                if t_curve and (t_curve.original in updated_curves or t_curve.data.original in updated_curves):
                    rebuild_civil_mesh(obj, depsgraph=depsgraph)
    finally:
        _is_rebuilding = False

classes = (
    CIVIL_PG_Segment,
    CIVIL_PG_Structure,
    CIVIL_PG_ObjectProps,
    CIVIL_OT_CreateRoadPreset,
    CIVIL_OT_SegmentAdd,
    CIVIL_OT_SegmentRemove,
    CIVIL_OT_RoadContinue,
    CIVIL_OT_RoadSnap,
    CIVIL_UL_Segments,
    CIVIL_OT_StructureAdd,
    CIVIL_OT_StructureRemove,
    CIVIL_UL_Structures,
    CIVIL_OT_CreateInterchangeRamp,
    CIVIL_OT_RefreshAll,
    CIVIL_OT_SelectGuideCurve,
    CIVIL_PT_MainPanel,
)

def register():
    for cls in classes + traffic.classes + terrain.classes + water.classes + utilities.classes + junctions.classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.civil_props = bpy.props.PointerProperty(type=CIVIL_PG_ObjectProps)
    bpy.types.Scene.civil_terrain = bpy.props.PointerProperty(type=terrain.CIVIL_PG_TerrainSettings)
    utilities.register_extra()
    junctions.register_extra()
    if civil_depsgraph_update_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(civil_depsgraph_update_handler)

def unregister():
    if civil_depsgraph_update_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(civil_depsgraph_update_handler)
    del bpy.types.Scene.civil_terrain
    del bpy.types.Object.civil_props
    utilities.unregister_extra()
    junctions.unregister_extra()
    for cls in reversed(classes + traffic.classes + terrain.classes + water.classes + utilities.classes + junctions.classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
