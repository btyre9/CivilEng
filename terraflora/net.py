# SPDX-License-Identifier: GPL-3.0-or-later
"""Downloads with retries (busy public servers time out now and then) and a cache folder."""

import http.client
import os
import ssl
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

import bpy


def download_dir():
    try:
        return bpy.utils.extension_path_user(__package__, path="downloads", create=True)
    except (ValueError, AttributeError, TypeError):
        path = os.path.join(tempfile.gettempdir(), "terraflora_downloads")
        os.makedirs(path, exist_ok=True)
        return path


def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def describe(err):
    if isinstance(err, urllib.error.HTTPError):
        return "the server answered %d %s" % (err.code, err.reason)
    if isinstance(err, urllib.error.URLError):
        return "no connection: %s" % err.reason
    if "timed out" in str(err).lower():
        return "the server took too long"
    return "%s: %s" % (type(err).__name__, err)


def fetch(url, params, path, timeout=180, tries=5):
    """Downloads url?params to path (kept as a cache), retrying busy-server errors."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params),
                                 headers={"User-Agent": "TerraFlora-Blender-Extension"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
                data = r.read()
                ctype = (r.headers.get("Content-Type") or "").lower()
            break
        except urllib.error.HTTPError as e:
            if (e.code < 500 and e.code != 429) or attempt == tries - 1:
                raise
        except (urllib.error.URLError, OSError, http.client.HTTPException):
            if attempt == tries - 1:
                raise
        time.sleep(min(20.0, 3.0 * 2 ** attempt))
    if "json" in ctype or "xml" in ctype or "html" in ctype or data[:1] in (b"{", b"<"):
        raise RuntimeError("the server sent an error instead of an image: " + data[:200].decode("utf-8", "replace"))
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)
    return path
