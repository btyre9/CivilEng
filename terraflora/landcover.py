# SPDX-License-Identifier: GPL-3.0-or-later
"""
Reading land cover from aerial photos.

For a CivilFlow terrain it downloads the four NAIP bands (red, green, blue and near-infrared)
at about 1 m, and the USGS National Land Cover Database (NLCD 2021, 30 m). Per pixel:

- water: negative vegetation index (NDVI) and little infrared;
- tree canopy: strong NDVI and dark in visible light (crowns and their shadows);
- grass and fields: some NDVI but bright and smooth;
- everything else (roofs, pavement, bare ground) grows nothing.

The pixels are averaged over each terrain vertex's cell, and NLCD's forest type (evergreen,
deciduous, mixed) splits the canopy into pines and leafy trees. The results are stored on
the terrain as point attributes that the forest's Geometry Nodes read.
"""

import json
import math
import os

import bpy
import numpy as np

from . import geo, net

NAIP_URL = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer/exportImage"
NLCD_WMS = "https://www.mrlc.gov/geoserver/mrlc_display/wms"
NLCD_LAYER = "NLCD_2021_Land_Cover_L48"
MAX_PX = 4000

# NLCD legend colours -> class codes
NLCD_COLORS = {
    11: (70, 107, 159), 12: (209, 222, 248), 21: (222, 197, 197), 22: (217, 146, 130),
    23: (235, 0, 0), 24: (171, 0, 0), 31: (179, 172, 159), 41: (104, 171, 95),
    42: (28, 95, 44), 43: (181, 201, 142), 52: (204, 184, 121), 71: (223, 223, 194),
    81: (220, 217, 57), 82: (171, 108, 40), 90: (184, 217, 235), 95: (108, 159, 184),
}
# Share of the canopy that is pine (evergreen) for each class; yard and field trees default below.
PINE_SHARE = {42: 0.95, 43: 0.5, 41: 0.05, 90: 0.25, 52: 0.45}
DEFAULT_PINE_SHARE = 0.35
SHRUBLAND = {52: 1.0, 71: 0.25, 31: 0.15}

ATTRS = ("tf_pines", "tf_leafy", "tf_shrubs", "tf_grass", "tf_water")
COVER_ATTR = "TF Cover"


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def classify_pixels(R, G, B, N):
    """Soft per-pixel (tree, grass, water) in 0..1 from 0..1 band values."""
    ndvi = (N - R) / (N + R + 1e-6)
    vis = (R + G + B) / 3.0
    water = _sig((-0.05 - ndvi) / 0.04) * _sig((0.45 - N) / 0.06)
    tree = _sig((ndvi - 0.27) / 0.03) * _sig((0.42 - vis) / 0.025) * (1.0 - water)
    grass = _sig((ndvi - 0.08) / 0.03) * (1.0 - tree) * (1.0 - water)
    return tree, grass, water


def _blur(a, k):
    """Box blur of a 2-D grid over (2k+1) cells, twice (close to a Gaussian)."""
    if k < 1:
        return a
    for _ in range(2):
        for axis in (0, 1):
            p = np.pad(a, [(k, k) if ax == axis else (0, 0) for ax in range(2)], mode="edge")
            c = np.cumsum(p, axis=axis)
            c = np.concatenate([np.zeros_like(np.take(c, [0], axis=axis)), c], axis=axis)
            n = a.shape[axis]
            a = (np.take(c, np.arange(2 * k + 1, 2 * k + 1 + n), axis=axis)
                 - np.take(c, np.arange(0, n), axis=axis)) / (2 * k + 1)
    return a


def _open(mask):
    """Morphological opening (erode then dilate) over 3 x 3 cells: drops specks, keeps ponds."""
    def shift_all(m, fn):
        p = np.pad(m, 1, mode="edge")
        out = m.copy()
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                out = fn(out, p[dy:dy + m.shape[0], dx:dx + m.shape[1]])
        return out
    return shift_all(shift_all(mask, np.minimum), np.maximum)


def terrain_grid(scene, terrain):
    """Scene-zone UTM coordinates of the terrain's vertex columns and rows, and the zone."""
    g = scene.get("cf_geo")
    if not g or "cf_grid" not in terrain:
        raise ValueError("Pick a terrain made by CivilFlow (Real-World Terrain); it knows where it is on the map")
    x0, y0, dx, dy, nx, ny = terrain["cf_grid"]
    nx, ny = int(nx), int(ny)
    E = g["e0"] + x0 + dx * np.arange(nx)
    N = g["n0"] + y0 + dy * np.arange(ny)
    return E, N, int(g["zone"]), bool(g["north"])


def _box_means(img, x0r, y0r, res, E, N, half):
    """Mean of img (rows from the north) over each vertex's cell, for the E x N vertex grid."""
    h, w = img.shape
    I = np.zeros((h + 1, w + 1))
    I[1:, 1:] = np.cumsum(np.cumsum(img, axis=0), axis=1)
    c0 = np.clip(np.floor((E - half - x0r) / res), 0, w - 1).astype(np.int64)
    c1 = np.clip(np.ceil((E + half - x0r) / res), 1, w).astype(np.int64)
    r0 = np.clip(np.floor((y0r - (N + half)) / res), 0, h - 1).astype(np.int64)
    r1 = np.clip(np.ceil((y0r - (N - half)) / res), 1, h).astype(np.int64)
    c1 = np.maximum(c1, c0 + 1)
    r1 = np.maximum(r1, r0 + 1)
    R0, C0 = np.meshgrid(r0, c0, indexing="ij")
    R1, C1 = np.meshgrid(r1, c1, indexing="ij")
    s = I[R1, C1] - I[R0, C1] - I[R1, C0] + I[R0, C0]
    return s / ((R1 - R0) * (C1 - C0))


def fetch_bands(E, N, zone, north, spacing, tag):
    """NAIP red, green, blue, infrared over the grid (rows from the north) and its georeference."""
    xmin, xmax = E[0] - spacing * 0.5, E[-1] + spacing * 0.5
    ymin, ymax = N[0] - spacing * 0.5, N[-1] + spacing * 0.5
    res = max(1.0, (xmax - xmin) / MAX_PX, (ymax - ymin) / MAX_PX)
    w = int(math.ceil((xmax - xmin) / res))
    h = int(math.ceil((ymax - ymin) / res))
    xmax, ymin = xmin + w * res, ymax - h * res
    epsg = geo.utm_epsg(zone, north)
    path = os.path.join(net.download_dir(), "naip4_%s_%.2fm.tif" % (tag, res))
    net.fetch(NAIP_URL, dict(bbox="%.3f,%.3f,%.3f,%.3f" % (xmin, ymin, xmax, ymax), bboxSR=epsg, imageSR=epsg,
                             size="%d,%d" % (w, h), format="tiff", pixelType="U8", bandIds="0,1,2,3",
                             renderingRule=json.dumps({"rasterFunction": "None"}), f="image"), path)
    t = geo.GeoTiff(path)
    try:
        a = t.read_bands()
    finally:
        t.close()
    if a.ndim != 3 or a.shape[2] < 4:
        raise RuntimeError("the photo server didn't send the infrared band")
    return a[..., :4].astype(np.float32) / 255.0, xmin, ymax, res


def fetch_nlcd(E, N, zone, north, tag):
    """NLCD class code for each vertex (E x N grid), or None when it can't be had."""
    EE, NN = np.meshgrid(E, N)
    lat, lon = geo.utm_to_latlon(EE, NN, zone, north)
    lat0, lat1 = float(lat.min()) - 0.0005, float(lat.max()) + 0.0005
    lon0, lon1 = float(lon.min()) - 0.0005, float(lon.max()) + 0.0005
    wpx = int(min(2048, max(64, (E[-1] - E[0]) / 10.0)))
    hpx = int(min(2048, max(64, (N[-1] - N[0]) / 10.0)))
    path = os.path.join(net.download_dir(), "nlcd_%s_%dx%d.png" % (tag, wpx, hpx))
    net.fetch(NLCD_WMS, dict(service="WMS", version="1.1.1", request="GetMap", layers=NLCD_LAYER, styles="",
                             srs="EPSG:4326", bbox="%.6f,%.6f,%.6f,%.6f" % (lon0, lat0, lon1, lat1),
                             width=wpx, height=hpx, format="image/png"), path)
    img = bpy.data.images.load(path, check_existing=False)
    try:
        img.colorspace_settings.name = 'Non-Color'
        px = np.empty(wpx * hpx * 4, dtype=np.float32)
        img.pixels.foreach_get(px)
    finally:
        bpy.data.images.remove(img)
    rgb = (px.reshape(hpx, wpx, 4)[::-1, :, :3] * 255.0 + 0.5).astype(np.int32)  # rows from the north
    codes = np.array(list(NLCD_COLORS.keys()))
    pal = np.array(list(NLCD_COLORS.values()))
    col = np.clip(((lon - lon0) / (lon1 - lon0) * wpx).astype(np.int64), 0, wpx - 1)
    row = np.clip(((lat1 - lat) / (lat1 - lat0) * hpx).astype(np.int64), 0, hpx - 1)
    c = rgb[row, col]
    d = ((c[..., None, :] - pal[None, None, :, :]) ** 2).sum(axis=-1)
    best = d.argmin(axis=-1)
    out = codes[best]
    out[d.min(axis=-1) > 300] = 0  # no-data / unknown colour
    return out


def read_land_cover(context, terrain, report=None):
    """Classifies the terrain's land and stores tf_* attributes and the TF Cover preview colours."""
    scene = context.scene
    E, N, zone, north = terrain_grid(scene, terrain)
    spacing = float(abs(terrain["cf_grid"][2]))
    tag = "%d_%d_%d_%d" % (round(E[0]), round(N[0]), len(E), len(N))
    try:
        bands, x0r, y0r, res = fetch_bands(E, N, zone, north, spacing, tag)
    except Exception as e:
        raise RuntimeError("the aerial photo bands didn't arrive (%s). Try again in a minute" % net.describe(e)) from e
    tree_p, grass_p, water_p = classify_pixels(bands[..., 0], bands[..., 1], bands[..., 2], bands[..., 3])
    half = spacing * 0.5
    tree = _box_means(tree_p, x0r, y0r, res, E, N, half)
    grass = _box_means(grass_p, x0r, y0r, res, E, N, half)
    water = _box_means(water_p, x0r, y0r, res, E, N, half)
    water = _open(np.clip((water - 0.35) / 0.3, 0.0, 1.0))  # firm shorelines, no rooftop specks

    note = ""
    try:
        nlcd = fetch_nlcd(E, N, zone, north, tag)
    except Exception as e:
        nlcd = None
        note = "NLCD forest types unavailable (%s); pines and leafy trees mixed evenly" % net.describe(e)
    if nlcd is not None:
        share = np.full(nlcd.shape, DEFAULT_PINE_SHARE)
        shrubland = np.zeros(nlcd.shape)
        for code, v in PINE_SHARE.items():
            share[nlcd == code] = v
        for code, v in SHRUBLAND.items():
            shrubland[nlcd == code] = v
    else:
        share = np.full(tree.shape, 0.5)
        shrubland = np.zeros(tree.shape)
    # NLCD comes in 30 m squares: blend forest types over ~50 m so stands don't look blocky.
    k = max(1, int(round(25.0 / spacing)))
    share = _blur(share, k)
    shrubland = _blur(shrubland, k)

    # Shadows and gaps make photos under-read canopy; lift partial cover so woods close up.
    fill = np.clip(tree, 0.0, 1.0) ** 0.6
    pines = fill * share
    leafy = fill * (1.0 - share)
    shrubs = np.clip(0.3 * tree + 0.6 * shrubland * (1.0 - tree), 0.0, 1.0) * (1.0 - water)
    layers = {"tf_pines": pines, "tf_leafy": leafy, "tf_shrubs": shrubs, "tf_grass": grass, "tf_water": water}

    me = terrain.data
    for name, arr in layers.items():
        attr = me.attributes.get(name) or me.attributes.new(name, 'FLOAT', 'POINT')
        attr.data.foreach_set("value", arr.astype(np.float32).ravel())
    write_cover_colors(terrain, layers)
    me.update()

    total = tree.size
    summary = {
        "trees": float(tree.mean()), "pines": float(pines.mean()), "leafy": float(leafy.mean()),
        "grass": float(grass.mean()), "water": float(water.mean()),
    }
    terrain["tf_summary"] = ("Trees %d%% (pines %d%%, leafy %d%%) · grass %d%% · water %d%%"
                             % (round(summary["trees"] * 100), round(summary["pines"] * 100),
                                round(summary["leafy"] * 100), round(summary["grass"] * 100),
                                round(summary["water"] * 100)))
    terrain["tf_note"] = note
    return summary, note, total


def write_cover_colors(terrain, layers):
    """A colour per vertex showing what grows there (shown by Show Cover Map)."""
    w = layers
    rest = np.clip(1.0 - (w["tf_pines"] + w["tf_leafy"] + w["tf_grass"] + w["tf_water"]), 0.0, 1.0)
    c = (w["tf_pines"][..., None] * np.array([0.05, 0.30, 0.12])
         + w["tf_leafy"][..., None] * np.array([0.35, 0.65, 0.20])
         + w["tf_grass"][..., None] * np.array([0.80, 0.80, 0.35])
         + w["tf_water"][..., None] * np.array([0.10, 0.30, 0.85])
         + rest[..., None] * np.array([0.60, 0.58, 0.55]))
    rgba = np.concatenate([np.clip(c, 0, 1), np.ones(c.shape[:-1] + (1,))], axis=-1).astype(np.float32)
    me = terrain.data
    attr = me.color_attributes.get(COVER_ATTR) or me.color_attributes.new(COVER_ATTR, 'BYTE_COLOR', 'POINT')
    attr.data.foreach_set("color_srgb" if hasattr(attr.data[0], "color_srgb") else "color", rgba.ravel())
