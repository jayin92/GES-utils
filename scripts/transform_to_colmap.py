#!/usr/bin/env python3
"""
Generate 3D point cloud using COLMAP with known camera poses from transforms.json
"""

import os
import json
import numpy as np
import argparse
from pathlib import Path

def transforms_to_colmap(transforms_json_path, output_path, colmap_executable="colmap"):
    """
    Convert transforms.json to COLMAP format and generate point cloud
    """
    
    # Load transforms.json
    with open(transforms_json_path, 'r') as f:
        transforms = json.load(f)
    
    # Create output directories
    sparse_dir = Path(output_path) / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = Path(output_path) / "images"
    
    # Extract camera parameters
    if 'fl_x' in transforms and 'fl_y' in transforms:
        fx = transforms['fl_x']
        fy = transforms['fl_y']
    elif 'camera_angle_x' in transforms:
        # Convert from camera angle to focal length
        w = transforms.get('w', 800)  # default width
        fx = w / (2 * np.tan(transforms['camera_angle_x'] / 2))
        fy = fx  # assume square pixels
    else:
        raise ValueError("Cannot determine focal length from transforms.json")
    
    cx = transforms.get('cx', transforms.get('w', 800) / 2)
    cy = transforms.get('cy', transforms.get('h', 600) / 2)
    w = transforms.get('w', 800)
    h = transforms.get('h', 600)
    
    # Write cameras.txt
    cameras_file = sparse_dir / "cameras.txt"
    with open(cameras_file, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")
    
    # Write images.txt
    images_file = sparse_dir / "images.txt"
    with open(images_file, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(transforms['frames'])}\n")
        
        for i, frame in enumerate(transforms['frames']):
            # Convert transform matrix to quaternion and translation
            c2w = np.array(frame['transform_matrix'])  # camera-to-world from NeRF
            
            # NeRF uses OpenGL convention, COLMAP uses computer vision convention
            # Apply coordinate system conversion
            # NeRF: +X right, +Y up, +Z backward (into screen)
            # COLMAP: +X right, +Y down, +Z forward (out of screen)
            
            # Flip Y and Z axes for coordinate system conversion
            # c2w[1:3] *= -1
            
            # Convert from camera-to-world (c2w) to world-to-camera (w2c)
            # w2c = inverse(c2w)
            w2c = np.linalg.inv(c2w)
            
            # Extract rotation and translation from w2c matrix
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            
            # Convert rotation matrix to quaternion
            qw, qx, qy, qz = rotation_matrix_to_quaternion(R)
            
            # Get image filename
            image_name = frame['file_path']
            if not image_name.endswith(('.jpg', '.jpeg', '.png')):
                image_name += '.jpg'  # assume jpg if no extension
            
            # Write image entry
            image_id = i + 1
            f.write(f"{image_id} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {image_name}\n")
            f.write("\n")  # Empty line for POINTS2D
    
    # Create empty points3D.txt (will be populated by triangulation)
    points_file = sparse_dir / "points3D.txt"
    with open(points_file, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0\n")
    
    print(f"COLMAP files written to {sparse_dir}")
    
    # Now run COLMAP to generate the point cloud
    generate_point_cloud(output_path, colmap_executable)

def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion (w, x, y, z)"""
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return qw, qx, qy, qz

def generate_point_cloud(source_path, colmap_executable="colmap", use_gpu=True):
    """
    Generate point cloud using COLMAP with known poses
    """
    colmap_command = f'"{colmap_executable}"' if colmap_executable != "colmap" else "colmap"
    use_gpu_flag = 1 if use_gpu else 0
    
    print("Step 1: Feature extraction...")
    # Feature extraction
    feat_extraction_cmd = f"""{colmap_command} feature_extractor \
        --database_path {source_path}/database.db \
        --image_path {source_path}/images \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model PINHOLE \
        --SiftExtraction.use_gpu {use_gpu_flag}"""
    
    exit_code = os.system(feat_extraction_cmd)
    if exit_code != 0:
        print(f"Feature extraction failed with code {exit_code}")
        return False
    
    print("Step 2: Feature matching...")
    # Feature matching
    feat_matching_cmd = f"""{colmap_command} exhaustive_matcher \
        --database_path {source_path}/database.db \
        --SiftMatching.use_gpu {use_gpu_flag}"""
    
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        print(f"Feature matching failed with code {exit_code}")
        return False
    
    print("Step 3: Point triangulation...")
    # Point triangulation (this generates the 3D points)
    triangulation_cmd = f"""{colmap_command} point_triangulator \
        --database_path {source_path}/database.db \
        --image_path {source_path}/images \
        --input_path {source_path}/sparse/0 \
        --output_path {source_path}/sparse/0"""
    
    exit_code = os.system(triangulation_cmd)
    if exit_code != 0:
        print(f"Point triangulation failed with code {exit_code}")
        return False
    
    print("Step 4: Bundle adjustment (optional refinement)...")
    # Optional: Bundle adjustment to refine the points
    ba_cmd = f"""{colmap_command} bundle_adjuster \
        --input_path {source_path}/sparse/0 \
        --output_path {source_path}/sparse/0"""
    
    exit_code = os.system(ba_cmd)
    if exit_code != 0:
        print(f"Bundle adjustment failed with code {exit_code}, but points should still be generated")
    
    print(f"Point cloud generation complete! Check {source_path}/sparse/0/points3D.txt")
    return True

def main():
    parser = argparse.ArgumentParser("Generate COLMAP point cloud from known poses")
    parser.add_argument("--transforms_json", "-t", required=True, 
                       help="Path to transforms.json file")
    parser.add_argument("--output_path", "-o", required=True,
                       help="Output directory for COLMAP reconstruction")
    parser.add_argument("--colmap_executable", default="colmap",
                       help="Path to COLMAP executable")
    parser.add_argument("--no_gpu", action='store_true',
                       help="Disable GPU acceleration")
    
    args = parser.parse_args()
    
    # Convert transforms.json to COLMAP format and generate point cloud
    transforms_to_colmap(
        args.transforms_json, 
        args.output_path, 
        args.colmap_executable
    )

if __name__ == "__main__":
    main()