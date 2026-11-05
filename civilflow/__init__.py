bl_info = {
    "name": "CivilFlow: Road, Highway & Overpass Generator",
    "author": "AI Assistant",
    "version": (1, 6, 0),
    "blender": (5, 1, 0),
    "location": "View3D > Sidebar (N) > CivilEng",
    "description": "Procedural roads built from straights and curves, highways, banking, guardrails, gantry signs, tunnels, embankments, cable-stayed towers, and toll plazas.",
    "category": "Add Mesh",
}

import bpy
import bmesh
from mathutils import Vector, Matrix
import math

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


def add_embankment_slopes(bm, samples, left_edges, right_edges, ground_z, verge_w=0.0, mat_index=9):
    """
    Constructs sloped earth berms (embankments) from the outer road edges down to ground_z.
    A flat verge of `verge_w` runs from the asphalt edge out past the sidewalks, barriers,
    walls and posts before the slope starts, so the slope never cuts through them.
    """
    if len(samples) < 2:
        return

    verge_z = -0.01  # just under the sidewalk underside, so the two never z-fight
    for side, edge_list in ((-1.0, left_edges), (1.0, right_edges)):
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

        for i in range(len(rings) - 1):
            a, b = rings[i], rings[i + 1]
            for k in range(2):
                quad = (a[k], b[k], b[k + 1], a[k + 1]) if side > 0 else (a[k], a[k + 1], b[k + 1], b[k])
                try:
                    f = bm.faces.new(quad)
                    f.material_index = mat_index
                except ValueError:
                    pass


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


def add_gantry_sign_at(bm, p, right, tangent, normal, left_x, right_x, clearance_h, sign_h, panels):
    """Full-span steel truss gantry on two posts carrying green signboards with white borders."""
    post_l = left_x - 0.7
    post_r = right_x + 0.7
    top_h = clearance_h + sign_h + 0.5

    for x in (post_l, post_r):
        base = p + right * x
        add_box_prism(bm, base - normal * 0.2, right, tangent, normal, 0.5, 0.5, 0.6, mat_index=1)   # Footing
        add_box_prism(bm, base, right, tangent, normal, 0.18, 0.18, top_h, mat_index=6)             # Post

    span_c = (post_l + post_r) * 0.5
    span_hw = (post_r - post_l) * 0.5
    truss_off = tangent * 0.28
    chord_z = (clearance_h + 0.15, clearance_h + sign_h + 0.15)
    for z in chord_z:
        add_box_prism(bm, p + right * span_c + normal * z + truss_off, right, tangent, normal,
                      span_hw, 0.12, 0.2, mat_index=6)

    n_web = max(2, int((post_r - post_l) / 2.5))
    web_h = chord_z[1] - chord_z[0] - 0.2
    for k in range(1, n_web):
        x = post_l + (post_r - post_l) * k / n_web
        add_box_prism(bm, p + right * x + normal * (chord_z[0] + 0.2) + truss_off, right, tangent, normal,
                      0.05, 0.05, web_h, mat_index=6)

    for cx, w in panels:
        border_base = p + right * cx + normal * (clearance_h - 0.08) + tangent * 0.1
        add_box_prism(bm, border_base, right, tangent, normal, w * 0.5 + 0.08, 0.06, sign_h + 0.16, mat_index=5)
        face_base = p + right * cx + normal * clearance_h
        add_box_prism(bm, face_base, right, tangent, normal, w * 0.5, 0.06, sign_h, mat_index=8)


def add_streetlight_at(bm, base_p, right, tangent, normal, arm_dir, pole_h, arm_len):
    """Steel pole on a concrete plinth with a mast arm and an emissive luminaire head."""
    add_box_prism(bm, base_p - normal * 0.1, right, tangent, normal, 0.22, 0.22, 0.45, mat_index=1)
    add_cylinder(bm, base_p, right, tangent, normal, radius=0.11, height=pole_h, num_segments=8, mat_index=6)

    arm_base = base_p + normal * (pole_h - 0.12) + right * (arm_dir * arm_len * 0.5)
    add_box_prism(bm, arm_base, right, tangent, normal, arm_len * 0.5, 0.05, 0.1, mat_index=6)

    head_c = base_p + right * (arm_dir * (arm_len - 0.25)) + normal * (pole_h - 0.2)
    add_box_prism(bm, head_c, right, tangent, normal, 0.38, 0.17, 0.14, mat_index=6)
    add_box_prism(bm, head_c - normal * 0.03, right, tangent, normal, 0.32, 0.13, 0.04, mat_index=7)


def add_pillar_at(bm, pier_top, right, tangent, width, depth, ground_z, mat_index=1):
    """Vertical concrete pier from ground_z up to the deck soffit, topped with a wider pier cap."""
    up = Vector((0.0, 0.0, 1.0))
    r = Vector((right.x, right.y, 0.0))
    if r.length < 1e-6:
        r = Vector((1.0, 0.0, 0.0))
    r.normalize()
    t = up.cross(r).normalized()

    height = pier_top.z - ground_z
    if height < 0.2:
        return
    base = Vector((pier_top.x, pier_top.y, ground_z))
    cap_h = min(1.0, height * 0.25)
    add_box_prism(bm, base, r, t, up, width * 0.5, depth * 0.5, height - cap_h, mat_index)
    add_box_prism(bm, base + up * (height - cap_h), r, t, up, width * 0.5 + 0.9, depth * 0.5 + 0.15, cap_h, mat_index)


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
        apex = p_pos + norm * tower_height

        for side in (-1.0, 1.0):
            base = p_pos + right * (side * leg_x)
            knee = base + norm * knee_h
            ground_pt = base.copy()
            ground_pt.z = ground_z
            if knee.z - ground_pt.z > 0.5:
                add_cylinder_between(bm, ground_pt, knee, radius=0.95, num_segments=8, mat_index=1)
            add_cylinder_between(bm, knee, apex + right * (side * 0.7), radius=0.8, num_segments=8, mat_index=1)

        # Lower cross-beam carrying the deck, and the apex cable anchorage block
        add_box_prism(bm, p_pos - norm * (deck_t + 1.2), right, tang, norm,
                      half_w=leg_x + 0.6, half_d=1.0, height=1.2, mat_index=1)
        add_box_prism(bm, apex - norm * 4.0, right, tang, norm,
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
                apex_pt = apex - norm * (0.5 + head_drop) + right * (side * 0.9)
                add_cylinder_between(bm, apex_pt, deck_anchor, radius=0.07, num_segments=6, mat_index=6)


def add_toll_plaza_at(bm, plaza_center, right, tangent, normal, num_lanes, booth_w, island_l, asphalt_h):
    """
    Constructs a highway toll plaza with concrete island splitters, impact attenuator nose blocks,
    modular toll-booth cabins, and an overhead concrete toll canopy roof.
    """
    lane_spacing = 3.8
    total_plaza_w = num_lanes * lane_spacing
    start_x = -total_plaza_w * 0.5

    # 1. Concrete Island Splitters & Toll Booth Cabins
    for l_i in range(num_lanes + 1):
        island_cx = start_x + l_i * lane_spacing
        if l_i == 0 or l_i == num_lanes:
            continue  # Outer bounds

        island_base = plaza_center + right * island_cx + normal * asphalt_h
        # Concrete splitter island curb
        add_box_prism(bm, island_base, right, tangent, normal, half_w=0.65, half_d=island_l * 0.5, height=0.35, mat_index=1)
        # Sloped Impact Attenuator Nose Blocks at entry/exit
        for nose_dir in (-1.0, 1.0):
            nose_base = island_base + tangent * (nose_dir * island_l * 0.45)
            add_box_prism(bm, nose_base, right, tangent, normal, half_w=0.55, half_d=1.2, height=0.25, mat_index=2)

        # Modular Toll Booth Cabin on center of island
        booth_base = island_base + normal * 0.35
        add_box_prism(bm, booth_base, right, tangent, normal, half_w=booth_w * 0.5, half_d=1.8, height=2.8, mat_index=1)

    # 2. Overhead Concrete Toll Canopy Roof spanning across all lanes, carried on a pair of
    #    columns standing on each island either side of its booth.
    canopy_z = 5.2
    col_top = asphalt_h + 0.35
    col_d = min(island_l * 0.35, 1.8 + 1.2)
    for l_i in range(1, num_lanes):
        island_base = plaza_center + right * (start_x + l_i * lane_spacing) + normal * col_top
        for d in (-col_d, col_d):
            add_box_prism(bm, island_base + tangent * d, right, tangent, normal,
                          half_w=0.28, half_d=0.28, height=canopy_z - col_top, mat_index=1)
    canopy_base = plaza_center + normal * canopy_z
    add_box_prism(bm, canopy_base, right, tangent, normal, half_w=total_plaza_w * 0.52, half_d=island_l * 0.65, height=0.75, mat_index=1)


# ==============================================================================
# 4. MASTER PROCEDURAL MESH REBUILDER
# ==============================================================================

def rebuild_civil_mesh(obj, depsgraph=None):
    """Main procedural mesh builder triggered whenever parameters or curves change."""
    if not obj or obj.type != 'MESH' or not obj.civil_props.is_civil_object:
        return
    if obj.mode == 'EDIT':
        return

    props = obj.civil_props
    curve_obj = props.target_curve
    if not curve_obj or curve_obj.type != 'CURVE':
        return

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
        return

    inv_mat = obj.matrix_world.inverted_safe()
    inv_rot = inv_mat.to_3x3()
    local_samples = []
    for pos, tang, norm, right in samples:
        l_pos = inv_mat @ pos
        l_tang = (inv_rot @ tang).normalized()
        l_norm = (inv_rot @ norm).normalized()
        l_right = (inv_rot @ right).normalized()
        local_samples.append((l_pos, l_tang, l_norm, l_right))

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

        if props.is_overpass:
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
            add_variable_extrusion(bm, local_samples, girder_prof_fn, mat_index=1, cap_ends=True)
        else:
            def base_prof_fn(i):
                lx, rx = left_road_edges[i], right_road_edges[i]
                return split_profile_edges([
                    (lx - 0.15, -deck_t * 0.4),
                    (rx + 0.15, -deck_t * 0.4),
                    (rx + 0.05, 0.0),
                    (lx - 0.05, 0.0),
                ], w_splits, (0, 2))
            add_variable_extrusion(bm, local_samples, base_prof_fn, mat_index=1, cap_ends=True)

    # 2. Terrain Cut-and-Fill Embankment Slopes (Berms)
    if props.add_embankments and not props.is_overpass:
        # Clear everything built beside the asphalt before the slope starts.
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
                for side, g_ext in ((-1.0, left_gore_ext[i]), (1.0, right_gore_ext[i])):
                    post_base = p_curr + right * (side * (gr_offset + g_ext + 0.05))
                    add_box_prism(bm, post_base, right, tang, norm, half_w=0.06, half_d=0.08, height=gr_h + 0.03, mat_index=6)

    # 8. Highway Concrete Sound / Retaining Walls
    if props.add_sound_walls:
        wall_offset = outer_edge_offset + 0.35
        wh = props.sound_wall_height
        wt = props.sound_wall_thickness
        for side, x_list in ((-1.0, [-left_gore_ext[i] for i in range(num_s)]), (1.0, right_gore_ext)):
            cx = side * wall_offset
            w_prof = [
                (cx - wt * 0.5, 0.0),
                (cx + wt * 0.5, 0.0),
                (cx + wt * 0.5, wh),
                (cx - wt * 0.5, wh),
            ]
            add_extruded_profile(bm, local_samples, w_prof, mat_index=3, x_offsets=x_list)

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
    if props.add_tunnel and total_w > 0.1:
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

    def is_inside_tunnel(idx):
        return props.add_tunnel and (t_start_idx <= idx <= t_end_idx)

    # 10. Cable-Stayed Bridge Towers & Cables
    if props.add_cable_bridge and props.is_overpass:
        add_cable_stayed_bridge(
            bm, local_samples, half_w,
            tower_spacing=props.cable_tower_spacing,
            tower_height=props.cable_tower_height,
            deck_t=deck_t,
            ground_z=local_ground_z,
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

        for i in range(num_s):
            if i > 0:
                gantry_accum += (local_samples[i][0] - local_samples[i - 1][0]).length
            if gantry_accum >= next_gantry_d:
                next_gantry_d = gantry_accum + props.gantry_spacing
                if is_inside_tunnel(i):
                    continue

                p_curr, tang, norm, right = local_samples[i]
                lx = left_road_edges[i] - (props.sidewalk_width if props.add_sidewalks else 0.0)
                rx = right_road_edges[i] + (props.sidewalk_width if props.add_sidewalks else 0.0)

                panels = []
                if props.lanes_right > 0:
                    r_w = max(2.8, props.lanes_right * props.lane_width * 0.78)
                    r_cx = half_w - props.shoulder_width - (props.lanes_right * props.lane_width * 0.5)
                    panels.append((r_cx, r_w))
                if props.lanes_left > 0:
                    l_w = max(2.8, props.lanes_left * props.lane_width * 0.78)
                    l_cx = -half_w + props.shoulder_width + (props.lanes_left * props.lane_width * 0.5)
                    panels.append((l_cx, l_w))
                if right_gore_ext[i] > 1.8:
                    panels.append((half_w + right_gore_ext[i] * 0.55, min(3.8, right_gore_ext[i] * 0.85)))

                if not panels:
                    panels.append((0.0, max(3.0, total_w * 0.6)))

                add_gantry_sign_at(
                    bm, p_curr, right, tang, norm,
                    left_x=lx, right_x=rx,
                    clearance_h=props.gantry_clearance,
                    sign_h=props.gantry_sign_height,
                    panels=panels
                )

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

                if props.streetlight_placement == 'BOTH':
                    for side, g_ext in ((-1.0, left_gore_ext[i]), (1.0, right_gore_ext[i])):
                        base_p = p_curr + right * (side * (pole_offset + g_ext))
                        add_streetlight_at(bm, base_p, right, tang, norm, arm_dir=-side,
                                           pole_h=props.streetlight_height, arm_len=props.streetlight_arm_length)
                elif props.streetlight_placement == 'ALTERNATING':
                    side = -1.0 if (pole_idx % 2 == 0) else 1.0
                    g_ext = left_gore_ext[i] if side < 0 else right_gore_ext[i]
                    base_p = p_curr + right * (side * (pole_offset + g_ext))
                    add_streetlight_at(bm, base_p, right, tang, norm, arm_dir=-side,
                                       pole_h=props.streetlight_height, arm_len=props.streetlight_arm_length)
                elif props.streetlight_placement == 'MEDIAN':
                    base_p = p_curr + right * med_center_x + norm * asphalt_h
                    add_streetlight_at(bm, base_p, right, tang, norm, arm_dir=-1.0,
                                       pole_h=props.streetlight_height, arm_len=props.streetlight_arm_length)
                    add_streetlight_at(bm, base_p, right, tang, norm, arm_dir=1.0,
                                       pole_h=props.streetlight_height, arm_len=props.streetlight_arm_length)
                pole_idx += 1

    # 14. Overpass Support Pillars
    if props.is_overpass and props.add_pillars and not props.add_cable_bridge:
        dist_accum = 0.0
        next_pillar_dist = props.pillar_spacing * 0.5

        for i in range(1, num_s):
            p_prev = local_samples[i - 1][0]
            p_curr, tang, norm, right = local_samples[i]
            seg_d = (p_curr - p_prev).length
            dist_accum += seg_d

            if dist_accum >= next_pillar_dist:
                next_pillar_dist += props.pillar_spacing
                pier_top = p_curr - Vector((0.0, 0.0, deck_t * 0.95))
                p_width = min(max(1.0, total_w * 0.45), props.pillar_width)
                add_pillar_at(bm, pier_top, right, tang, p_width, props.pillar_depth, local_ground_z, mat_index=1)

    # Dashed extrusions leave unused ring vertices behind; drop them.
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context='VERTS')

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


# ==============================================================================
# 5. PROPERTY GROUP DEFINITION
# ==============================================================================

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
            ('ALTERNATING', "Alternating Staggered", "Staggered left/right streetlights"),
            ('MEDIAN', "Central Median Double-Mast", "Center twin-arm mast poles along the median"),
        ],
        default='BOTH',
        update=update_civil_prop
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
        name="Pillar Depth", default=1.8, min=0.8, max=10.0, unit='LENGTH', update=update_civil_prop
    )


# ==============================================================================
# 6. OPERATORS (PRESETS, BUILDERS & INTERCHANGE RAMPS)
# ==============================================================================

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

    def execute(self, context):
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        active_obj = context.active_object
        use_segments = not (active_obj and active_obj.type == 'CURVE')
        if not use_segments:
            curve_obj = active_obj
        else:
            if not self.properties.is_property_set("elevation"):
                self.elevation = self.PRESET_ELEVATION.get(self.preset_type, 0.0)
            # The road starts at the 3D cursor heading +X; the segment chain gives its shape.
            curve_obj = create_guide_curve(context, self.preset_type.title(), [(0, 0, 0), (1, 0, 0)])
            curve_obj.location = context.scene.cursor.location + Vector((0.0, 0.0, self.elevation))

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
            p.add_pillars = False
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
                g_box.prop(props, "gantry_spacing")
                row = g_box.row(align=True)
                row.prop(props, "gantry_clearance")
                row.prop(props, "gantry_sign_height")

            g_box.separator()
            g_box.prop(props, "add_streetlights")
            if props.add_streetlights:
                g_box.prop(props, "streetlight_placement")
                g_box.prop(props, "streetlight_spacing")
                row = g_box.row(align=True)
                row.prop(props, "streetlight_height")
                row.prop(props, "streetlight_arm_length")

            # Overpass Piers
            o_box = layout.box()
            o_box.label(text="Overpass Viaduct Piers", icon='MOD_ARCHIMEDES')
            o_box.prop(props, "is_overpass")
            o_box.prop(props, "deck_thickness")
            if props.is_overpass and not props.add_cable_bridge:
                o_box.prop(props, "add_pillars")
                if props.add_pillars:
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
    CIVIL_PG_ObjectProps,
    CIVIL_OT_CreateRoadPreset,
    CIVIL_OT_SegmentAdd,
    CIVIL_OT_SegmentRemove,
    CIVIL_UL_Segments,
    CIVIL_OT_CreateInterchangeRamp,
    CIVIL_OT_RefreshAll,
    CIVIL_OT_SelectGuideCurve,
    CIVIL_PT_MainPanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.civil_props = bpy.props.PointerProperty(type=CIVIL_PG_ObjectProps)
    if civil_depsgraph_update_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(civil_depsgraph_update_handler)

def unregister():
    if civil_depsgraph_update_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(civil_depsgraph_update_handler)
    del bpy.types.Object.civil_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
