# SPDX-License-Identifier: GPL-3.0-or-later
"""
Map maths for terrain import: UTM <-> latitude/longitude (WGS84, Krueger series, mm accurate)
and a small GeoTIFF reader (strips or tiles; none, LZW, Deflate or PackBits compression;
horizontal and floating-point predictors). Works on plain floats and on numpy arrays.
"""

import math
import mmap
import re
import struct
import zlib

import numpy as np

_A = 6378137.0
_F = 1.0 / 298.257223563
_K0 = 0.9996
_N = _F / (2.0 - _F)
_AA = _A / (1.0 + _N) * (1.0 + _N ** 2 / 4.0 + _N ** 4 / 64.0)
_ALPHA = (_N / 2 - 2 * _N ** 2 / 3 + 5 * _N ** 3 / 16,
          13 * _N ** 2 / 48 - 3 * _N ** 3 / 5,
          61 * _N ** 3 / 240)
_BETA = (_N / 2 - 2 * _N ** 2 / 3 + 37 * _N ** 3 / 96,
         _N ** 2 / 48 + _N ** 3 / 15,
         17 * _N ** 3 / 480)
_DELTA = (2 * _N - 2 * _N ** 2 / 3 - 2 * _N ** 3,
          7 * _N ** 2 / 3 - 8 * _N ** 3 / 5,
          56 * _N ** 3 / 15)
_E2N = 2.0 * math.sqrt(_N) / (1.0 + _N)


def utm_zone(lon):
    return int(math.floor((lon + 180.0) / 6.0)) % 60 + 1


def utm_epsg(zone, north=True):
    """WGS84 UTM EPSG code. NAD83 UTM (269xx) is within about a metre of it in the US."""
    return (32600 if north else 32700) + zone


def _central_meridian(zone):
    return math.radians(zone * 6.0 - 183.0)


def latlon_to_utm(lat, lon, zone, north=True):
    phi = np.radians(lat)
    lam = np.radians(lon) - _central_meridian(zone)
    sp = np.sin(phi)
    t = np.sinh(np.arctanh(sp) - _E2N * np.arctanh(_E2N * sp))
    xi = np.arctan2(t, np.cos(lam))
    eta = np.arctanh(np.sin(lam) / np.sqrt(1.0 + t * t))
    e = eta.copy() if isinstance(eta, np.ndarray) else eta
    n = xi.copy() if isinstance(xi, np.ndarray) else xi
    for j, a in enumerate(_ALPHA, start=1):
        e = e + a * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
        n = n + a * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
    easting = 500000.0 + _K0 * _AA * e
    northing = _K0 * _AA * n + (0.0 if north else 10000000.0)
    return easting, northing


def utm_to_latlon(easting, northing, zone, north=True):
    xi = (np.asarray(northing, dtype=float) - (0.0 if north else 10000000.0)) / (_K0 * _AA)
    eta = (np.asarray(easting, dtype=float) - 500000.0) / (_K0 * _AA)
    xp, ep = xi, eta
    for j, b in enumerate(_BETA, start=1):
        xp = xp - b * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
        ep = ep - b * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
    chi = np.arcsin(np.sin(xp) / np.cosh(ep))
    phi = chi
    for j, d in enumerate(_DELTA, start=1):
        phi = phi + d * np.sin(2 * j * chi)
    lam = _central_meridian(zone) + np.arctan2(np.sinh(ep), np.cos(xp))
    return np.degrees(phi), np.degrees(lam)


def parse_latlon(text):
    """'34.86, -118.89', '34.86 -118.89' or '34°51'36"N 118°53'24"W' -> (lat, lon) or None."""
    import re
    s = text.strip()
    dms = re.findall(r"(\d+(?:\.\d+)?)\s*°\s*(?:(\d+(?:\.\d+)?)\s*['′]\s*)?(?:(\d+(?:\.\d+)?)\s*[\"″]\s*)?([NSEW])", s, re.I)
    if len(dms) == 2:
        vals = []
        for d, m, sec, hemi in dms:
            v = float(d) + float(m or 0) / 60.0 + float(sec or 0) / 3600.0
            vals.append((-v if hemi.upper() in "SW" else v, hemi.upper()))
        lat = next((v for v, h in vals if h in "NS"), None)
        lon = next((v for v, h in vals if h in "EW"), None)
        if lat is not None and lon is not None:
            return lat, lon
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if len(nums) >= 2:
        lat, lon = float(nums[0]), float(nums[1])
        if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
            return lat, lon
    return None


# ------------------------------------------------------------------------------
# GeoTIFF
# ------------------------------------------------------------------------------

_TYPES = {1: ("B", 1), 2: ("s", 1), 3: ("H", 2), 4: ("I", 4), 5: ("II", 8), 6: ("b", 1),
          7: ("B", 1), 8: ("h", 2), 9: ("i", 4), 10: ("ii", 8), 11: ("f", 4), 12: ("d", 8),
          16: ("Q", 8), 17: ("q", 8)}


class GeoTiff:
    """Header and georeferencing of a TIFF; read() decodes a window of band 0 as float32."""

    def __init__(self, path):
        self.path = path
        self._fh = open(path, "rb")
        self.data = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        d = self.data
        bo = d[:2]
        if bo == b"II":
            self.e = "<"
        elif bo == b"MM":
            self.e = ">"
        else:
            raise ValueError("not a TIFF file")
        magic = struct.unpack(self.e + "H", d[2:4])[0]
        if magic == 42:
            self.big = False
            ifd = struct.unpack(self.e + "I", d[4:8])[0]
        elif magic == 43:
            self.big = True
            ifd = struct.unpack(self.e + "Q", d[8:16])[0]
        else:
            raise ValueError("not a TIFF file")
        self.tags = self._read_ifd(ifd)
        t = self.tags
        self.width = t[256][0]
        self.height = t[257][0]
        self.bits = t.get(258, [8])[0]
        self.compression = t.get(259, [1])[0]
        self.spp = t.get(277, [1])[0]
        self.planar = t.get(284, [1])[0]
        self.predictor = t.get(317, [1])[0]
        self.sample_format = t.get(339, [1])[0]
        self.tiled = 322 in t
        if self.tiled:
            self.tw, self.th = t[322][0], t[323][0]
            self.offsets, self.counts = t[324], t[325]
        else:
            self.tw = self.width
            self.th = t.get(278, [self.height])[0]
            self.offsets, self.counts = t[273], t[279]
        nodata = t.get(42113)
        self.nodata = None
        if nodata:
            try:
                self.nodata = float(nodata.strip("\x00 ").split()[0])
            except (ValueError, IndexError):
                pass
        self._geo()

    def _read_ifd(self, off):
        d, e = self.data, self.e
        if self.big:
            n = struct.unpack(e + "Q", d[off:off + 8])[0]
            p, entry, inline = off + 8, 20, 8
        else:
            n = struct.unpack(e + "H", d[off:off + 2])[0]
            p, entry, inline = off + 2, 12, 4
        tags = {}
        for k in range(n):
            q = p + k * entry
            tag, typ = struct.unpack(e + "HH", d[q:q + 4])
            cnt = struct.unpack(e + ("Q" if self.big else "I"), d[q + 4:q + 4 + (8 if self.big else 4)])[0]
            if typ not in _TYPES:
                continue
            fmt, size = _TYPES[typ]
            total = size * cnt
            vpos = q + (12 if self.big else 8)
            if total > inline:
                vpos = struct.unpack(e + ("Q" if self.big else "I"), d[vpos:vpos + inline])[0]
            raw = d[vpos:vpos + total]
            if typ == 2:
                tags[tag] = raw.decode("latin-1")
            elif typ in (5, 10):
                vals = struct.unpack(e + fmt[0] * (2 * cnt), raw)
                tags[tag] = [vals[i] / vals[i + 1] if vals[i + 1] else 0.0 for i in range(0, len(vals), 2)]
            else:
                tags[tag] = list(struct.unpack(e + fmt * cnt, raw))
        return tags

    def _geo(self):
        t = self.tags
        self.epsg = None
        self.geographic = False
        keys = t.get(34735)
        if keys:
            model = proj = geog = None
            for i in range(4, 4 + 4 * keys[3], 4):
                key, loc, _cnt, val = keys[i:i + 4]
                if loc != 0:
                    continue
                if key == 1024:  # GTModelTypeGeoKey: 1 projected, 2 geographic
                    model = val
                elif key == 3072 and val not in (0, 32767):  # ProjectedCSTypeGeoKey
                    proj = val
                elif key == 2048 and val not in (0, 32767):  # GeographicTypeGeoKey
                    geog = val
            self.epsg = proj or geog
            self.geographic = (model == 2) or (proj is None and geog is not None)
        self.zone = self.north = None
        self.mercator = self.epsg in (3857, 102100, 900913)
        e = self.epsg or 0
        if 32601 <= e <= 32660 or 32701 <= e <= 32760:
            self.zone, self.north = e % 100, e < 32700
        elif 26901 <= e <= 26923:
            self.zone, self.north = e - 26900, True
        elif 6330 <= e <= 6348:  # NAD83(2011) / UTM zones 1N-19N
            self.zone, self.north = e - 6329, True
        elif not self.geographic and not self.mercator:
            citation = t.get(34737, "")  # GeoAsciiParams, e.g. "NAD83(2011) / UTM zone 11N|"
            m = re.search(r"UTM zone (\d+)\s*([NS])?", citation if isinstance(citation, str) else "", re.I)
            if m:
                self.zone, self.north = int(m.group(1)), (m.group(2) or "N").upper() == "N"
        # Pixel (0, 0)'s top-left corner and pixel size: x = x0 + col * sx, y = y0 - row * sy
        if 34264 in t:  # ModelTransformationTag
            m = t[34264]
            self.x0, self.sx, self.y0, self.sy = m[3], m[0], m[7], -m[5]
        elif 33550 in t and 33922 in t:
            sx, sy = t[33550][0], t[33550][1]
            i, j, _k, x, y = t[33922][:5]
            self.x0, self.y0, self.sx, self.sy = x - i * sx, y + j * sy, sx, sy
        else:
            self.x0 = self.y0 = None
            self.sx = self.sy = None
        # Pixel-is-point rasters (RasterTypeGeoKey 1025 == 2) are referenced to pixel centres.
        if keys and self.x0 is not None:
            for i in range(4, 4 + 4 * keys[3], 4):
                if keys[i] == 1025 and keys[i + 3] == 2:
                    self.x0 -= self.sx * 0.5
                    self.y0 += self.sy * 0.5

    def close(self):
        try:
            self.data.close()
            self._fh.close()
        except (OSError, ValueError, BufferError):
            pass

    def to_file_xy(self, easting, northing, zone, north=True):
        """Scene UTM coordinates -> this raster's own map coordinates."""
        if self.zone == zone and self.north == north:
            return easting, northing
        lat, lon = utm_to_latlon(easting, northing, zone, north)
        if self.geographic:
            return lon, lat
        if self.mercator:
            x = np.radians(lon) * _A
            y = np.log(np.tan(np.pi / 4.0 + np.radians(lat) / 2.0)) * _A
            return x, y
        if self.zone is None:
            raise ValueError("can't tell which map projection this GeoTIFF uses (expected UTM or lat/long)")
        return latlon_to_utm(lat, lon, self.zone, self.north)

    def to_latlon(self, x, y):
        """This raster's map coordinates -> (lat, lon) in degrees."""
        if self.geographic:
            return y, x
        if self.mercator:
            return np.degrees(2.0 * np.arctan(np.exp(np.asarray(y) / _A)) - np.pi / 2.0), np.degrees(np.asarray(x) / _A)
        if self.zone is None:
            raise ValueError("can't tell which map projection this GeoTIFF uses (expected UTM or lat/long)")
        return utm_to_latlon(x, y, self.zone, self.north)

    def corners(self):
        """Map coordinates of the four outer corners."""
        x1 = self.x0 + self.sx * self.width
        y1 = self.y0 - self.sy * self.height
        return np.array([self.x0, x1, x1, self.x0]), np.array([self.y0, self.y0, y1, y1])

    def center_latlon(self):
        lat, lon = self.to_latlon(self.x0 + self.sx * self.width * 0.5, self.y0 - self.sy * self.height * 0.5)
        return float(lat), float(lon)

    def _dtype(self):
        kind = {1: "u", 2: "i", 3: "f"}.get(self.sample_format, "u")
        return np.dtype(self.e + kind + str(self.bits // 8))

    def _decode(self, idx):
        raw = self.data[self.offsets[idx]:self.offsets[idx] + self.counts[idx]]
        c = self.compression
        if c == 1:
            return raw
        if c in (8, 32946):
            return zlib.decompress(raw)
        if c == 5:
            return _lzw_decode(raw)
        if c == 32773:
            return _packbits_decode(raw)
        raise ValueError("unsupported TIFF compression %d (use an uncompressed, LZW or Deflate GeoTIFF)" % c)

    def _block(self, idx, rows):
        """Decode one strip/tile to (rows, tw, spp) array of the stored type."""
        dt = self._dtype()
        spp = self.spp if self.planar == 1 else 1
        buf = self._decode(idx)
        need = rows * self.tw * spp * dt.itemsize
        if len(buf) < need:
            buf = buf + bytes(need - len(buf))
        if self.predictor == 3:
            b = np.frombuffer(buf[:need], dtype=np.uint8).reshape(rows, self.tw * spp * dt.itemsize).copy()
            b = np.cumsum(b, axis=1, dtype=np.uint8)
            # Bytes are stored most significant first per row, one plane per byte position.
            b = b.reshape(rows, dt.itemsize, self.tw * spp)
            if self.e == "<":
                b = b[:, ::-1, :]
            b = np.ascontiguousarray(b.transpose(0, 2, 1))
            arr = b.view(np.dtype(self.e + dt.kind + str(dt.itemsize))).reshape(rows, self.tw, spp)
            return arr
        arr = np.frombuffer(buf[:need], dtype=dt).reshape(rows, self.tw, spp)
        if self.predictor == 2:
            arr = np.cumsum(arr, axis=1, dtype=dt)
        return arr

    def read_bands(self, x0=0, y0=0, w=None, h=None):
        """All bands of the pixel window as an (h, w, bands) array of the stored type (interleaved files)."""
        w = self.width - x0 if w is None else w
        h = self.height - y0 if h is None else h
        if self.planar != 1:
            raise ValueError("separate-plane multi-band TIFFs aren't supported")
        out = np.zeros((h, w, self.spp), dtype=self._dtype())
        if self.tiled:
            across = (self.width + self.tw - 1) // self.tw
            for ty in range(y0 // self.th, (y0 + h - 1) // self.th + 1):
                for tx in range(x0 // self.tw, (x0 + w - 1) // self.tw + 1):
                    blk = self._block(ty * across + tx, self.th)
                    bx, by = tx * self.tw, ty * self.th
                    sx0, sy0 = max(x0, bx), max(y0, by)
                    sx1, sy1 = min(x0 + w, bx + self.tw, self.width), min(y0 + h, by + self.th, self.height)
                    if sx1 > sx0 and sy1 > sy0:
                        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = blk[sy0 - by:sy1 - by, sx0 - bx:sx1 - bx]
        else:
            for s in range(y0 // self.th, (y0 + h - 1) // self.th + 1):
                rows = min(self.th, self.height - s * self.th)
                blk = self._block(s, rows)
                by = s * self.th
                sy0, sy1 = max(y0, by), min(y0 + h, by + rows)
                out[sy0 - y0:sy1 - y0] = blk[sy0 - by:sy1 - by, x0:x0 + w]
        return out

    def read(self, x0=0, y0=0, w=None, h=None):
        """Band 0 of the pixel window (columns x0.., rows y0..) as float32 with nodata -> nan."""
        w = self.width - x0 if w is None else w
        h = self.height - y0 if h is None else h
        out = np.full((h, w), np.nan, dtype=np.float32)
        if self.tiled:
            across = (self.width + self.tw - 1) // self.tw
            for ty in range(y0 // self.th, (y0 + h - 1) // self.th + 1):
                for tx in range(x0 // self.tw, (x0 + w - 1) // self.tw + 1):
                    blk = self._block(ty * across + tx, self.th)[:, :, 0]
                    bx, by = tx * self.tw, ty * self.th
                    sx0, sy0 = max(x0, bx), max(y0, by)
                    sx1, sy1 = min(x0 + w, bx + self.tw, self.width), min(y0 + h, by + self.th, self.height)
                    if sx1 > sx0 and sy1 > sy0:
                        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = blk[sy0 - by:sy1 - by, sx0 - bx:sx1 - bx]
        else:
            for s in range(y0 // self.th, (y0 + h - 1) // self.th + 1):
                rows = min(self.th, self.height - s * self.th)
                blk = self._block(s, rows)[:, :, 0]
                by = s * self.th
                sy0, sy1 = max(y0, by), min(y0 + h, by + rows)
                out[sy0 - y0:sy1 - y0, :] = blk[sy0 - by:sy1 - by, x0:x0 + w]
        if self.nodata is not None:
            out[out == self.nodata] = np.nan
        out[out < -1e30] = np.nan
        return out


def _lzw_decode(data):
    out = bytearray()
    table = [bytes([i]) for i in range(256)] + [b"", b""]
    width, bitpos, nbits = 9, 0, len(data) * 8
    prev = None
    get = int.from_bytes
    while bitpos + width <= nbits:
        byte = bitpos >> 3
        chunk = get(data[byte:byte + 3].ljust(3, b"\0"), "big")
        code = (chunk >> (24 - (bitpos & 7) - width)) & ((1 << width) - 1)
        bitpos += width
        if code == 257:
            break
        if code == 256:
            table = table[:258]
            width = 9
            prev = None
            continue
        if prev is None:
            entry = table[code]
            out += entry
            prev = entry
            continue
        if code < len(table):
            entry = table[code]
            table.append(prev + entry[:1])
        else:
            entry = prev + prev[:1]
            table.append(entry)
        out += entry
        prev = entry
        n = len(table)
        if n >= 511 and width == 9:
            width = 10
        elif n >= 1023 and width == 10:
            width = 11
        elif n >= 2047 and width == 11:
            width = 12
    return bytes(out)


def _packbits_decode(data):
    out = bytearray()
    i = 0
    while i < len(data):
        n = data[i]
        i += 1
        if n < 128:
            out += data[i:i + n + 1]
            i += n + 1
        elif n > 128:
            out += data[i:i + 1] * (257 - n)
            i += 1
    return bytes(out)


def read_world_file(image_path):
    """(x0, y0, sx, sy) of the top-left pixel corner from a .tfw/.jgw/.pgw/.j2w world file, or None."""
    import os
    base, ext = os.path.splitext(image_path)
    ext = ext.lower().lstrip(".")
    cands = [base + "." + ext[0] + ext[-1] + "w", base + "." + ext + "w", base + ".wld"]
    for c in cands:
        for name in (c, c.upper()):
            if os.path.exists(name):
                with open(name) as fh:
                    v = [float(x) for x in fh.read().split()[:6]]
                sx, sy = v[0], -v[3]
                return v[4] - sx * 0.5, v[5] + sy * 0.5, sx, sy
    return None
