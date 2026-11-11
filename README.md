# CivilFlow

Blender extension for procedural roads, highways, interchanges, bridges, tunnels, real USGS terrain, lakes and animated traffic.

- **Source:** `civilflow/` (the extension package, with `blender_manifest.toml`)
- **Requires:** Blender 5.1+
- **Install:** download the zip from [Releases](../../releases), then in Blender use Edit → Preferences → Get Extensions → ▾ → Install from Disk.

## Build

```bash
blender --command extension build --source-dir civilflow --output-dir .
```

## Test

```bash
blender --background --factory-startup --python tools/smoke_test.py
```
