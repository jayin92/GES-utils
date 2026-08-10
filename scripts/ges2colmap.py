#!/usr/bin/env python3
"""
Convert a Google Earth Studio 3D Tracking export into a COLMAP sparse model
with known poses, and triangulate a point cloud from it.

Replaces the old two-step ges2nerf.py -> transform_to_colmap.py pipeline.

    tracking.json
      -> transforms.json / transforms_train.json / transforms_test.json  (NeRF)
      -> sparse/0/{cameras,images,points3D}.txt + triangulated points     (COLMAP / 3DGS)

COORDINATE CONVENTION -- READ BEFORE EDITING
--------------------------------------------
The poses written here are ALREADY in COLMAP / OpenCV convention
(+X right, +Y down, +Z forward). Commit 8c46e7f deliberately removed the
OpenGL axis conversion that used to live in ges2nerf.py. Do not add flips such
as `c2w[0:3, 1:3] *= -1` back in -- they silently corrupt every downstream pose.
The dead lines are kept commented in build_poses() as documentation of that
decision, not as code waiting to be re-enabled.
"""

import argparse
import json
import math
import shlex
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
from pymap3d.enu import geodetic2enu
from scipy.spatial.transform import Rotation
from tqdm import tqdm

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

MATCHERS = {
    "exhaustive": "exhaustive_matcher",
    "sequential": "sequential_matcher",
    "vocab_tree": "vocab_tree_matcher",
    "spatial": "spatial_matcher",
}


# --------------------------------------------------------------------------
# tracking.json -> poses
# --------------------------------------------------------------------------


def load_tracking(tracking_path):
    """
    Read tracking.json and recover the ENU world origin.

    GES stores keyframe values normalized against fixed ranges rather than in
    real units, so the origin has to be un-normalized back out of the first
    trackPoints entry. The same ranges are hardcoded in gen_esp.py and
    kml2esp.py -- change one and you must change all of them.
    """
    with open(tracking_path, "rb") as f:
        raw = json.load(f)

    position = raw["trackPoints"][0]["coordinate"]["position"]["attributes"]
    lon = position[0]["value"]["relative"]
    lat = position[1]["value"]["relative"]
    alt = position[2]["value"]["relative"]
    alt_min = position[2]["value"]["minValueRange"]
    alt_max = position[2]["value"]["maxValueRange"]

    origin = (
        180 * lat - 90,  # lat0
        360 * lon - 180,  # lon0
        (alt_max - alt_min) * alt + alt_min,  # alt0
    )

    return (
        raw,
        origin,
        raw["width"],
        raw["height"],
        raw["cameraFrames"][0]["fovVertical"],
    )


def rot_ecef2enu(lat, lon):
    lamb = np.deg2rad(lon)
    phi = np.deg2rad(lat)
    sL = np.sin(lamb)
    sP = np.sin(phi)
    cL = np.cos(lamb)
    cP = np.cos(phi)
    rot = np.array(
        [
            [-sL, cL, 0],
            [-sP * cL, -sP * sL, cP],
            [cP * cL, cP * sL, sP],
        ]
    )
    return rot


def build_poses(raw_tracking_data, origin):
    """Build one 4x4 camera-to-world matrix per cameraFrames entry."""
    lat0, lon0, alt0 = origin
    rot = rot_ecef2enu(lat0, lon0)

    poses = []
    for frame in tqdm(raw_tracking_data["cameraFrames"], desc="Building poses"):
        x, y, z = geodetic2enu(
            frame["coordinate"]["latitude"],
            frame["coordinate"]["longitude"],
            frame["coordinate"]["altitude"],
            lat0,
            lon0,
            alt0,
        )
        rx, ry, rz = (
            frame["rotation"]["x"],
            frame["rotation"]["y"],
            frame["rotation"]["z"],
        )
        R = Rotation.from_euler("XYZ", [rx, ry, rz], degrees=True).as_matrix()
        c2w = np.block(
            [
                [rot @ R, np.array([x, y, z]).reshape(-1, 1)],
                [np.zeros((1, 3)), 1],
            ]
        )

        # Deliberately NOT applied -- see the module docstring. Output is
        # already COLMAP/OpenCV convention.
        # c2w[0:3, 1:3] *= -1
        # c2w[2, :] *= -1

        poses.append(c2w)

    return poses


# --------------------------------------------------------------------------
# intrinsics
# --------------------------------------------------------------------------


def first_image(images_dir):
    if not images_dir.is_dir():
        return None
    for path in sorted(images_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            return path
    return None


def resolve_intrinsics(render_w, render_h, fov_v, images_dir):
    """
    Derive intrinsics from the GES vertical FOV, scaled to the size the images
    on disk actually are.

    merge_image.py resizes frames to --size (2048x2048 by default) while
    tracking.json reports the GES render dimensions (1920x1080 by default).
    Taking w/h from tracking.json therefore describes a 16:9 camera over square
    images, so every intrinsic is wrong and triangulation degrades. GES has
    square pixels, so the vertical FOV gives one focal length that is then
    scaled independently per axis.
    """
    f_render = render_h / (2 * math.tan(math.radians(fov_v) / 2))

    sample = first_image(images_dir)
    if sample is None:
        print(
            f"No images found in {images_dir}; using tracking.json dimensions "
            f"{render_w}x{render_h} for intrinsics."
        )
        w, h = render_w, render_h
    else:
        import cv2  # imported lazily so --transforms-only works without opencv

        image = cv2.imread(str(sample))
        if image is None:
            raise SystemExit(f"Error: could not read image {sample}")
        h, w = image.shape[:2]

        if (w, h) != (render_w, render_h):
            print(
                f"Image size {w}x{h} differs from the tracking.json render size "
                f"{render_w}x{render_h}; rescaling intrinsics to match the images."
            )
            render_aspect = render_w / render_h
            image_aspect = w / h
            if abs(render_aspect - image_aspect) > 1e-6:
                print(
                    f"  WARNING: aspect ratio changed ({render_aspect:.4f} -> "
                    f"{image_aspect:.4f}). The render was squashed, not just scaled "
                    f"-- usually a merge_image.py --size mistake. Intrinsics below "
                    f"describe the squashed images, but detail is already lost."
                )

    return {
        "w": w,
        "h": h,
        "k1": 0,
        "k2": 0,
        "p1": 0,
        "p2": 0,
        "fl_x": f_render * w / render_w,
        "fl_y": f_render * h / render_h,
        "cx": w / 2,
        "cy": h / 2,
    }


# --------------------------------------------------------------------------
# transforms*.json
# --------------------------------------------------------------------------


def write_transforms(intrinsics, poses, output_dir, holdout):
    """Write transforms.json plus the train/test split used by NeRF training."""
    frames = [
        {
            "file_path": f"images/frame_{i:04}.jpg",
            "transform_matrix": c2w.tolist(),
            "frame_id": i,
        }
        for i, c2w in enumerate(poses)
    ]

    splits = {
        "transforms.json": frames,
        "transforms_train.json": [f for i, f in enumerate(frames) if i % holdout != 0],
        "transforms_test.json": [f for i, f in enumerate(frames) if i % holdout == 0],
    }

    for name, split_frames in splits.items():
        out = dict(intrinsics)
        out["frames"] = split_frames
        with open(output_dir / name, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4)

    print(
        f"Wrote {len(frames)} frames "
        f"({len(splits['transforms_train.json'])} train, "
        f"{len(splits['transforms_test.json'])} test) to {output_dir}"
    )

    return frames


# --------------------------------------------------------------------------
# images on disk
# --------------------------------------------------------------------------


def resolve_image_names(frames, images_dir, allow_missing):
    """
    Map each frame to the filename that actually exists in images_dir.

    GES exports .jpeg while transforms.json names .jpg, so fall back across
    extensions before giving up. Unlike the old transform_to_colmap.py this
    never prompts -- it fails with the list of missing files, or skips them
    when explicitly allowed.
    """
    resolved = {}
    missing = []

    for i, frame in enumerate(frames):
        name = Path(frame["file_path"]).name
        if (images_dir / name).exists():
            resolved[i] = name
            continue

        stem = Path(name).stem
        for suffix in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            if (images_dir / (stem + suffix)).exists():
                resolved[i] = stem + suffix
                break
        else:
            missing.append(name)

    if missing:
        print(
            f"\n{len(missing)} image(s) referenced by transforms.json are missing "
            f"from {images_dir}:"
        )
        for name in missing[:10]:
            print(f"  - {name}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        if not allow_missing:
            raise SystemExit(
                "Aborting. Pass --allow-missing-images to triangulate "
                "from the images that are present."
            )
        print("Continuing without them (--allow-missing-images).")

    return resolved


# --------------------------------------------------------------------------
# COLMAP model
# --------------------------------------------------------------------------


def read_database_images(db_path):
    """
    Return {image_name: image_id} and the camera_id, straight from COLMAP's
    database.

    The old script assumed image_id == frame_index + 1. That happens to hold
    for a full transforms.json (zero-padded names sort in frame order) but is
    wrong for every image once frames are skipped, e.g. transforms_train.json.
    Reading the real IDs is correct either way.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT image_id, name, camera_id FROM images").fetchall()
    finally:
        conn.close()

    if not rows:
        raise SystemExit(
            f"Error: no images in {db_path}; feature extraction found nothing."
        )

    name_to_id = {name: image_id for image_id, name, _ in rows}
    camera_ids = {camera_id for _, _, camera_id in rows}
    if len(camera_ids) != 1:
        raise SystemExit(
            f"Error: expected a single camera in {db_path}, found {len(camera_ids)}."
        )

    return name_to_id, camera_ids.pop()


def c2w_to_colmap(c2w):
    """Camera-to-world 4x4 -> COLMAP world-to-camera (qw, qx, qy, qz, t)."""
    R = c2w[:3, :3]
    t = c2w[:3, 3]

    # Rigid inverse; exact and cheaper than np.linalg.inv on the full 4x4.
    R_w2c = R.T
    t_w2c = -R.T @ t

    qx, qy, qz, qw = Rotation.from_matrix(R_w2c).as_quat()  # scipy returns x,y,z,w
    return qw, qx, qy, qz, t_w2c


def write_colmap_model(
    sparse_dir, intrinsics, frames, image_names, name_to_id, camera_id
):
    sparse_dir.mkdir(parents=True, exist_ok=True)

    with open(sparse_dir / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(
            f"{camera_id} PINHOLE {intrinsics['w']} {intrinsics['h']} "
            f"{intrinsics['fl_x']} {intrinsics['fl_y']} "
            f"{intrinsics['cx']} {intrinsics['cy']}\n"
        )

    written = 0
    unknown = []
    lines = []
    for i, frame in enumerate(frames):
        name = image_names.get(i)
        if name is None:
            continue
        image_id = name_to_id.get(name)
        if image_id is None:
            unknown.append(name)
            continue

        qw, qx, qy, qz, t = c2w_to_colmap(np.array(frame["transform_matrix"]))
        lines.append(
            f"{image_id} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {camera_id} {name}\n"
        )
        lines.append("\n")  # POINTS2D, empty -- point_triangulator fills these in
        written += 1

    if unknown:
        raise SystemExit(
            f"Error: {len(unknown)} image(s) are in transforms.json but not in the COLMAP "
            f"database, e.g. {unknown[:5]}. Delete database.db and re-run."
        )

    with open(sparse_dir / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {written}\n")
        f.writelines(lines)

    with open(sparse_dir / "points3D.txt", "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write(
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        )
        f.write("# Number of points: 0\n")

    print(f"Wrote COLMAP model for {written} images to {sparse_dir}")


# --------------------------------------------------------------------------
# COLMAP invocation
# --------------------------------------------------------------------------


def run_colmap_stage(argv, dry_run=False):
    """Run one COLMAP stage. Uses an argv list so paths with spaces survive."""
    print("\n$ " + " ".join(shlex.quote(str(a)) for a in argv))
    if dry_run:
        return
    result = subprocess.run([str(a) for a in argv])
    if result.returncode != 0:
        raise SystemExit(f"{argv[1]} failed with exit code {result.returncode}")


def extract_features(colmap, database, images_dir, intrinsics, use_gpu, dry_run):
    params = f"{intrinsics['fl_x']},{intrinsics['fl_y']},{intrinsics['cx']},{intrinsics['cy']}"
    run_colmap_stage(
        [
            colmap,
            "feature_extractor",
            "--database_path",
            database,
            "--image_path",
            images_dir,
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            "PINHOLE",
            "--ImageReader.camera_params",
            params,
            "--SiftExtraction.use_gpu",
            "1" if use_gpu else "0",
        ],
        dry_run,
    )


def match_features(colmap, database, matcher, vocab_tree_path, use_gpu, dry_run):
    argv = [
        colmap,
        MATCHERS[matcher],
        "--database_path",
        database,
        "--SiftMatching.use_gpu",
        "1" if use_gpu else "0",
    ]
    if matcher == "vocab_tree":
        argv += ["--VocabTreeMatching.vocab_tree_path", vocab_tree_path]
    run_colmap_stage(argv, dry_run)


def triangulate(colmap, database, images_dir, sparse_dir, dry_run):
    run_colmap_stage(
        [
            colmap,
            "point_triangulator",
            "--database_path",
            database,
            "--image_path",
            images_dir,
            "--input_path",
            sparse_dir,
            "--output_path",
            sparse_dir,
        ],
        dry_run,
    )
    run_colmap_stage(
        [
            colmap,
            "model_converter",
            "--input_path",
            sparse_dir,
            "--output_path",
            sparse_dir,
            "--output_type",
            "TXT",
        ],
        dry_run,
    )


# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Convert a GES 3D Tracking export to transforms*.json and a "
        "triangulated COLMAP model."
    )
    parser.add_argument(
        "--tracking",
        required=True,
        help="tracking.json, or the directory containing it",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Dataset root; receives transforms*.json, sparse/0 and database.db",
    )
    parser.add_argument(
        "--images", default=None, help="Image directory (default: <output_dir>/images)"
    )
    parser.add_argument(
        "--holdout",
        type=int,
        default=50,
        help="Every Nth frame goes to test, the rest to train (default: 50)",
    )
    parser.add_argument(
        "--transforms-only",
        action="store_true",
        help="Stop after writing transforms*.json; do not run COLMAP",
    )
    parser.add_argument(
        "--matcher",
        choices=sorted(MATCHERS),
        default="exhaustive",
        help="COLMAP matcher (default: exhaustive, O(N^2) in image count)",
    )
    parser.add_argument(
        "--vocab-tree-path",
        default=None,
        help="Vocabulary tree file, required by --matcher vocab_tree",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU for SIFT extraction and matching",
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Triangulate from present images instead of failing on missing ones",
    )
    parser.add_argument(
        "--colmap-executable",
        default="colmap",
        help="Path to the COLMAP binary (default: colmap)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print COLMAP commands without running them",
    )

    args = parser.parse_args()

    if args.matcher == "vocab_tree" and not args.vocab_tree_path:
        parser.error("--matcher vocab_tree requires --vocab-tree-path")

    tracking_path = Path(args.tracking)
    if tracking_path.is_dir():
        tracking_path = tracking_path / "tracking.json"
    if not tracking_path.is_file():
        raise SystemExit(f"Error: {tracking_path} not found")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images) if args.images else output_dir / "images"

    raw, origin, render_w, render_h, fov_v = load_tracking(tracking_path)
    print(f"World origin: lat0={origin[0]}, lon0={origin[1]}, alt0={origin[2]}")

    intrinsics = resolve_intrinsics(render_w, render_h, fov_v, images_dir)
    poses = build_poses(raw, origin)
    frames = write_transforms(intrinsics, poses, output_dir, args.holdout)

    if args.transforms_only:
        print("\n--transforms-only: skipping COLMAP.")
        return 0

    # COLMAP is fed every frame. The holdout split only affects NeRF training;
    # triangulation should use all available views.
    image_names = resolve_image_names(frames, images_dir, args.allow_missing_images)

    colmap = args.colmap_executable
    database = output_dir / "database.db"
    sparse_dir = output_dir / "sparse" / "0"
    use_gpu = not args.no_gpu

    # Extraction must run before the model is written: it creates database.db,
    # and the model's image IDs have to be the ones COLMAP assigned there.
    extract_features(colmap, database, images_dir, intrinsics, use_gpu, args.dry_run)

    if args.dry_run:
        # database.db does not exist, so the real IDs cannot be read. Keep going
        # so the remaining commands are still printed for inspection.
        print(
            f"\n  [dry-run] would read image IDs from {database} "
            f"and write the model to {sparse_dir}"
        )
    else:
        name_to_id, camera_id = read_database_images(database)
        write_colmap_model(
            sparse_dir, intrinsics, frames, image_names, name_to_id, camera_id
        )

    match_features(
        colmap, database, args.matcher, args.vocab_tree_path, use_gpu, args.dry_run
    )
    triangulate(colmap, database, images_dir, sparse_dir, args.dry_run)

    print(f'\nDone. Point cloud: {sparse_dir / "points3D.txt"}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
