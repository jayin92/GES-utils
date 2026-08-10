# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A collection of standalone Python scripts (`scripts/`) that turn Google Earth Studio (GES) renders into NeRF / 3DGS training datasets. There is no package, no build step, no test suite, and no dependency manifest — every script is run directly with `python scripts/<name>.py` and parses its own `argparse` flags (except `visualize_cam_poses.py`, which uses `tyro`).

Dependencies are installed ad hoc: `numpy`, `scipy`, `opencv-python`, `pymap3d`, `geopy`, `viser`, `tyro`, `imageio`, `tqdm`, `trimesh` (optional, PLY loading). External binaries: `colmap` and `ffmpeg` on `PATH`.

## The dataset pipeline

The scripts are stages of one workflow; understanding the whole chain is necessary before editing any single stage.

1. **`gen_esp.py`** → writes a `.esp` GES project file describing a grid of camera positions. Import into Google Earth Studio.
2. GES renders the animation to an image sequence + a `tracking.json` (3D Tracking export). **Render twice** with the watermark in different corners.
3. **`merge_image.py`** (image folders) or **`prep_ges_datasets.py`** (videos) → composites the two renders to erase the watermark.
4. **`ges2colmap.py`** → `tracking.json` → `transforms{,_train,_test}.json` → COLMAP sparse model with *known* poses → triangulated point cloud (needed by 3DGS). Formerly two scripts, `ges2nerf.py` and `transform_to_colmap.py`; both were deleted when they merged.
5. **`visualize_cam_poses.py`** → viser web viewer to sanity-check poses before training.

`kml2esp.py` is a third-party alternative entry point (path-following animation from a KML LineString, © Pat Wilson) and is not part of the grid workflow.

## Coordinate conventions — the core invariant

**GES `.esp` keyframes store normalized "relative" values, not real units.** The same fixed ranges are hardcoded in `gen_esp.py`, `kml2esp.py`, and inverted in `ges2colmap.py`:

| field | min | max |
|---|---|---|
| longitude | -180 | 180 |
| latitude | -90 | 90 |
| altitude | -500 | 65117481 |
| tilt (rotationY) | 0 | 180 |
| pan (rotationX) | 0 | 360 |

`load_tracking()` in `ges2colmap.py` reverses this to recover the world origin: `lat0 = 180*lat - 90`, `lon0 = 360*lon - 180`, `alt0 = (max-min)*alt + min`. Change a range in one file and you must change it in all of them.

**World frame** is ENU centred on the *first* `trackPoints` entry of `tracking.json`. Poses are built as `c2w = rot_ecef2enu(lat0, lon0) @ Rotation.from_euler('XYZ', rx, ry, rz)`, translation from `geodetic2enu(...)` in **metres**.

**The output is already in COLMAP / OpenCV convention** (+X right, +Y down, +Z forward) — commit `8c46e7f` removed the OpenGL conversion on purpose. `build_poses()` in `ges2colmap.py` keeps the axis-flip lines (`c2w[0:3, 1:3] *= -1`, etc.) commented out as a record of that decision. Do not re-enable them and do not delete them as dead code; re-enabling silently corrupts every downstream pose.

## Watermark removal

GES stamps its watermark in the bottom-left. The fix is to render the same animation twice with the watermark relocated, then paste the clean bottom-left region from render B into render A. The two scripts differ in how that region is defined:

- `prep_ges_datasets.py` — video input, region **hardcoded** as `[1877:2159, 0:1500]` (only valid for 3840×2160).
- `merge_image.py` — image-folder input, region computed fractionally (bottom 25%, left 50%) after resizing to `--size` (default 2048×2048), so it adapts to resolution.

Prefer `merge_image.py` for new work.

## Filename contract

`ges2colmap.py` writes `file_path` as `images/frame_{i:04}.jpg`, where `i` is the index into `cameraFrames`. **Frame index must match render order** — any renaming or re-sorting between stages breaks the pose↔image correspondence.

`merge_image.py` already emits `frame_{i:04d}.jpg`. `rename_images.py` normalizes GES's raw `<name>_0001.jpeg` exports to the same form: `python scripts/rename_images.py <folder> [--dry-run] [--max-index N]`. It aborts without renaming anything if two sources would collide on one target or if a rename would overwrite an existing file.

`ges2colmap.py` resolves images relative to `<output_dir>/images` (override with `--images`), falling back across `.jpg`/`.jpeg`/`.png`. Missing images are a hard error listing the files; `--allow-missing-images` opts into triangulating without them. It never prompts, so it is safe to run unattended.

**Two COLMAP details that are easy to regress:**
- **Image IDs must come from `database.db`**, not from the frame index. `read_database_images()` reads them after `feature_extractor` runs, which is why extraction happens *before* the model is written. The old `image_id = i + 1` coincidentally matched for a full `transforms.json` but mis-ID'd every image in a holdout subset.
- **Intrinsics are rescaled to the real image size.** `resolve_intrinsics()` reads the actual dimensions off the first image because `merge_image.py` resizes while `tracking.json` reports the GES render size. Taking `w`/`h` straight from `tracking.json` describes the wrong camera.

## Common commands

```bash
# 1. Generate a GES project: 10x10 grid, 4 pan directions per point
python scripts/gen_esp.py --lat 40.7147 --lon -74.0158 --alt 500 --tilt 45 \
    --offset 0.256 --sample 10 --center --output tokyo.esp
# --center treats lat/lon as the square's centre; omit for bottom-left corner.
# Frame count = sample^2 * 4.

# 3. Watermark removal (image folders)
python scripts/merge_image.py --left renders_a/ --right renders_b/ \
    --output dataset/ --size 2048x2048
# ...or from two videos
python scripts/prep_ges_datasets.py --left a.mp4 --right b.mp4 --output dataset/images

# 4. tracking.json -> transforms*.json -> COLMAP model + points (needs colmap on PATH)
python scripts/ges2colmap.py --tracking dataset/ --output_dir dataset/ --holdout 50
# --tracking takes tracking.json or its containing directory.
# holdout N: every Nth frame goes to test, the rest to train. COLMAP always gets
# all frames — the split only affects the NeRF outputs.
# --transforms-only stops before COLMAP; --dry-run prints the colmap commands.
# --matcher {exhaustive,sequential,vocab_tree,spatial}; exhaustive is O(N^2) and
# becomes impractical past ~1000 frames.

# 5. Inspect poses (tyro CLI — flags use hyphens, opens a viser server)
python scripts/visualize_cam_poses.py --transforms-path dataset/transforms.json \
    --images-path dataset/images --downsample-factor 2
```

## Repo conventions

`*.esp`, `*.kml`, and `tokyo_tower*` are gitignored — sample `.esp` files in `scripts/` are local artefacts, not tracked. Datasets live outside the repo.
