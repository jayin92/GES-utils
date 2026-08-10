# GES-utils

Turn Google Earth Studio (GES) renders into NeRF / 3DGS training datasets.

A collection of standalone scripts — no package, no build step. Run each one directly.

## Install

There is no dependency manifest. Install what you need:

```bash
pip install numpy scipy opencv-python pymap3d geopy viser tyro imageio tqdm trimesh
```

`colmap` and `ffmpeg` must be on your `PATH`. (`trimesh` is optional — it only enables PLY
loading in the pose viewer.)

## Pipeline

### 1. Generate a GES project file

```bash
python scripts/gen_esp.py --lat 40.7147 --lon -74.0158 --alt 500 --tilt 45 \
    --offset 0.256 --sample 10 --center --output nyc.esp
```

Writes an `.esp` describing a `--sample` × `--sample` grid of camera positions, each shot from
4 pan directions, for `sample² × 4` frames total. `--offset` is the square's side length in km.
`--center` treats `--lat/--lon` as the square's centre; omit it to treat them as the
bottom-left corner.

### 2. Render it twice in Google Earth Studio

Import the `.esp`, then export **both**:

- the image sequence, and
- the **3D Tracking** data (`tracking.json`) — this carries the camera poses.

Render the animation **twice, with the watermark in a different corner each time**. GES stamps
its watermark into the bottom-left of every frame, and the only way to remove it is to
composite two renders. Keep both image folders.

### 3. Erase the watermark

```bash
python scripts/merge_image.py --left renders_a/ --right renders_b/ \
    --output dataset/ --size 2048x2048
```

Pastes the clean bottom-left region of render B over render A and writes `frame_0000.jpg`,
`frame_0001.jpg`, … to `dataset/merged_frames/`. Move or rename that to `dataset/images/`.

> **Keep the aspect ratio.** `--size` resizes with no regard for it, so squaring a 16:9 render
> squashes the image. `ges2colmap.py` detects this and rescales the intrinsics to compensate,
> but the lost detail is not recoverable — prefer a `--size` matching your render's aspect ratio.

For video input instead of image folders, use `prep_ges_datasets.py` (its merge region is
hardcoded for 3840×2160).

### 4. Convert poses and build the COLMAP model

```bash
python scripts/ges2colmap.py --tracking dataset/ --output_dir dataset/ --holdout 50
```

One command covering everything downstream of the render:

```
tracking.json
  -> transforms.json / transforms_train.json / transforms_test.json   (NeRF)
  -> sparse/0/{cameras,images,points3D}.txt + triangulated points     (COLMAP / 3DGS)
```

`--holdout N` sends every Nth frame to test and the rest to train; it only affects the NeRF
split, since triangulation always uses every frame. Useful flags:

| flag | effect |
|---|---|
| `--transforms-only` | Stop after `transforms*.json`; skip COLMAP entirely |
| `--dry-run` | Print the COLMAP commands without running them |
| `--matcher` | `exhaustive` (default), `sequential`, `vocab_tree`, `spatial` |
| `--no-gpu` | Disable GPU for SIFT extraction and matching |
| `--images` | Image directory (default `<output_dir>/images`) |
| `--allow-missing-images` | Triangulate from present images instead of failing |

Matching is `exhaustive` by default, which is O(N²) in image count. That is fine for
`--sample 10` (400 frames) but impractical for `--sample 30` (3600 frames, ~6.5M pairs) —
switch `--matcher` for large grids.

### 5. Check the poses before training

```bash
python scripts/visualize_cam_poses.py --transforms-path dataset/transforms.json \
    --images-path dataset/images --downsample-factor 2
```

Opens a [viser](https://github.com/nerfstudio-project/viser) server showing each camera frustum
with its image. Cameras should form a coherent grid; if they fan out or invert, the poses are
wrong and training will fail silently.

## Scripts

| script | purpose |
|---|---|
| `gen_esp.py` | Generate a `.esp` GES project for a grid of camera positions |
| `merge_image.py` | Composite two image folders to erase the watermark (resolution-aware) |
| `prep_ges_datasets.py` | Same, from two videos (merge region hardcoded for 3840×2160) |
| `rename_images.py` | Normalize GES's `<name>_0001.jpeg` exports to `frame_0001.jpg` |
| `ges2colmap.py` | `tracking.json` → `transforms*.json` → COLMAP model → triangulated points |
| `visualize_cam_poses.py` | viser viewer for sanity-checking poses |
| `kml2esp.py` | Third-party: path-following animation from a KML LineString (© Pat Wilson) — not part of the grid workflow |

## Coordinate conventions

Poses are written in **COLMAP / OpenCV convention** (+X right, +Y down, +Z forward). They are
already in that convention — do not add OpenGL axis flips such as `c2w[0:3, 1:3] *= -1`.
Commit `8c46e7f` removed that conversion deliberately; re-adding it silently corrupts every
downstream pose. The dead lines are kept commented in `ges2colmap.py` as a record of that
decision.

The world frame is ENU, centred on the first `trackPoints` entry of `tracking.json`, with
translations in metres.

GES `.esp` keyframes store values normalized against fixed ranges rather than real units
(longitude ±180, latitude ±90, altitude −500…65117481, tilt 0…180, pan 0…360). These ranges are
hardcoded in `gen_esp.py` and `kml2esp.py` and inverted in `ges2colmap.py` — change one and you
must change all of them.

Frame index is the contract between poses and images: `ges2colmap.py` writes `file_path` as
`images/frame_{i:04}.jpg` where `i` indexes `cameraFrames`. Any renaming or re-sorting between
stages breaks the correspondence.

## Roadmap

1. Directly inherit NeRFStudio to create a GES Dataloader
2. Web UI for generating ESP file more straightforward
