# The conversion scripts for Google Earth datasets from SGAM


import argparse
import json
import os
import math
import numpy as np

from pathlib import Path

from pymap3d.ecef import geodetic2ecef
from pymap3d.enu import geodetic2enu
from scipy.spatial.transform import Rotation


def parse_args():
    parser = argparse.ArgumentParser(description="convert GES to transforms_train/test.json")

    parser.add_argument("--recon_dir", type=str, default="landmark/dataset/your_dataset/sparse/0")
    parser.add_argument("--output_dir", type=str, default="landmark/dataset/your_dataset")
    parser.add_argument("--holdout", type=int, default=50)

    args = parser.parse_args()
    return args

def rot_ecef2enu(lat, lon):
    lamb = np.deg2rad(lon)
    phi = np.deg2rad(lat)
    sL = np.sin(lamb)
    sP = np.sin(phi)
    cL = np.cos(lamb)
    cP = np.cos(phi)
    rot = np.array([
        [     -sL,       cL,  0],
        [-sP * cL, -sP * sL, cP],
        [ cP * cL,  cP * sL, sP],
    ])
    return rot

def compute_w2c_ecef(rx, ry, rz):
    R = Rotation.from_euler('XYZ', [rx, ry, rz], degrees=True).as_matrix()
    return R

def ges_to_json(recon_dir, output_dir, holdout):

    with open(os.path.join(recon_dir, "tracking.json"), "rb") as f:
        raw_tracking_data = json.load(f)

    w       = raw_tracking_data["width"]
    h       = raw_tracking_data["height"]
    lon     = raw_tracking_data["trackPoints"][0]["coordinate"]["position"]["attributes"][0]["value"]["relative"]
    lat     = raw_tracking_data["trackPoints"][0]["coordinate"]["position"]["attributes"][1]["value"]["relative"]
    alt     = raw_tracking_data["trackPoints"][0]["coordinate"]["position"]["attributes"][2]["value"]["relative"]
    alt_min = raw_tracking_data["trackPoints"][0]["coordinate"]["position"]["attributes"][2]["value"]["minValueRange"]
    alt_max = raw_tracking_data["trackPoints"][0]["coordinate"]["position"]["attributes"][2]["value"]["maxValueRange"]
    fov_v = raw_tracking_data["cameraFrames"][0]["fovVertical"]

    lat0 = 180 * lat - 90
    lon0 = 360 * lon - 180
    alt0 = (alt_max - alt_min) * alt + alt_min
    print(f"lat0: {lat0}, lon0: {lon0}, alt0: {alt0}")

    rot = rot_ecef2enu(lat0, lon0)

    theta_v_rad = math.radians(fov_v)
    intrinsic = {
        "w": w,
        "h": h,
        "k1": 0,
        "k2": 0,
        "p1": 0,
        "p2": 0,
    }

    intrinsic["fl_x"] = h / (2 * math.tan(theta_v_rad / 2))
    intrinsic["fl_y"] = h / (2 * math.tan(theta_v_rad / 2))
    intrinsic["cx"] = w / 2
    intrinsic["cy"] = h / 2

    frames = []
    positions = []
    for i, frame in enumerate(raw_tracking_data["cameraFrames"]):
        x, y, z = geodetic2enu(
            frame['coordinate']['latitude'],
            frame['coordinate']['longitude'],
            frame['coordinate']['altitude'], 
            lat0, lon0, alt0
        )
        print(x, y, z)
        rx, ry, rz = frame['rotation']['x'], frame['rotation']['y'], frame['rotation']['z']
        R = compute_w2c_ecef(rx, ry, rz)
        c2w = np.block([
            [rot @ R, np.array([x, y, z]).reshape(-1, 1) / 1000],
            [np.zeros((1, 3)), 1]
        ])
        
        # c2w[0:3, 1:3] *= -1
        # c2w = c2w[[1,0,2,3],:]
        # c2w[2,:] *= -1 # flip whole world upside down
        # c2w = c2w[np.array([1, 0, 2, 3]), :]
        # c2w[2, :] *= -1
        # c2w[1:3, 0:3] *= -1
        print(c2w)
        frame = {
            "file_path": f"images/frame_{i:04}.jpg",
            "transform_matrix": c2w.tolist(),
            "frame_id": i
        }

        frames.append(frame)

    # import ipdb
    # ipdb.set_trace()
    out = dict(intrinsic)
    out_train = dict(intrinsic)
    out_test = dict(intrinsic)

    frames_train = [f for i, f in enumerate(frames) if i % holdout != 0]
    frames_test = [f for i, f in enumerate(frames) if i % holdout == 0]

    out["frames"] = frames
    out_train["frames"] = frames_train
    out_test["frames"] = frames_test


    print("Train frames:", len(frames_train))
    print("Test frames:", len(frames_test))

    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    with open(output_dir / "transforms_train.json", "w", encoding="utf-8") as f:
        json.dump(out_train, f, indent=4)

    with open(output_dir / "transforms_test.json", "w", encoding="utf-8") as f:
        json.dump(out_test, f, indent=4)

    return len(frames)


if __name__ == "__main__":
    init_args = parse_args()
    Recondir = Path(init_args.recon_dir)
    Outputdir = Path(init_args.output_dir)
    Holdout = init_args.holdout
    ges_to_json(Recondir, Outputdir, Holdout)
