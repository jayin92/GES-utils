import json
import random
import time
from pathlib import Path
from typing import List, Optional

import imageio.v3 as iio
import numpy as np
import tyro
from tqdm.auto import tqdm

import viser
import viser.transforms as vtf


def load_transforms_json(transforms_path: Path) -> dict:
    """Load transforms.json file"""
    with open(transforms_path, 'r') as f:
        return json.load(f)


def transforms_to_camera_poses(transforms_data: dict) -> tuple:
    """
    Convert transforms.json data to camera poses
    Returns: (camera_poses, camera_intrinsics, image_names)
    """
    frames = transforms_data['frames']
    
    # Extract camera intrinsics
    if 'fl_x' in transforms_data and 'fl_y' in transforms_data:
        fx = transforms_data['fl_x']
        fy = transforms_data['fl_y']
    elif 'camera_angle_x' in transforms_data:
        # Convert from camera angle to focal length
        w = transforms_data.get('w', 800)
        fx = w / (2 * np.tan(transforms_data['camera_angle_x'] / 2))
        fy = fx  # assume square pixels if fy not provided
    else:
        # Default values if not specified
        fx = fy = 800.0
    
    cx = transforms_data.get('cx', transforms_data.get('w', 800) / 2)
    cy = transforms_data.get('cy', transforms_data.get('h', 600) / 2)
    w = transforms_data.get('w', 800)
    h = transforms_data.get('h', 600)
    
    camera_intrinsics = {
        'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
        'width': w, 'height': h
    }
    
    camera_poses = []
    image_names = []
    
    for frame in frames:
        # Get camera-to-world transformation matrix
        c2w = np.array(frame['transform_matrix'])
        
        # Convert to SE3 transformation (camera-to-world)
        T_world_camera = vtf.SE3.from_matrix(c2w)
        
        camera_poses.append(T_world_camera)
        
        # Get image filename
        image_name = frame['file_path']
        if not image_name.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
            # Try common extensions
            for ext in ['.jpg', '.png', '.jpeg']:
                if (Path(image_name).parent / (Path(image_name).stem + ext)).exists():
                    image_name = str(Path(image_name).parent / (Path(image_name).stem + ext))
                    break
        image_names.append(image_name)
    
    return camera_poses, camera_intrinsics, image_names


def main(
    transforms_path: Path = Path("transforms.json"),
    images_path: Optional[Path] = None,
    downsample_factor: int = 2,
    reorient_scene: bool = False,
    point_cloud_path: Optional[Path] = None,
) -> None:
    """
    Visualize camera poses from transforms.json format
    
    Args:
        transforms_path: Path to transforms.json file
        images_path: Path to images directory (if None, uses directory of transforms.json)
        downsample_factor: Factor to downsample images for display
        reorient_scene: Whether to reorient scene based on average camera direction
        point_cloud_path: Optional path to point cloud file (.ply or .pcd)
    """
    server = viser.ViserServer()
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Load transforms.json
    if not transforms_path.exists():
        print(f"Error: {transforms_path} not found!")
        return
    
    transforms_data = load_transforms_json(transforms_path)
    camera_poses, camera_intrinsics, image_names = transforms_to_camera_poses(transforms_data)
    
    # Set images path
    if images_path is None:
        images_path = transforms_path.parent
    
    print(f"Loaded {len(camera_poses)} camera poses")
    print(f"Camera intrinsics: fx={camera_intrinsics['fx']:.1f}, fy={camera_intrinsics['fy']:.1f}")
    print(f"Image resolution: {camera_intrinsics['width']}x{camera_intrinsics['height']}")

    # Optional: Load point cloud if provided
    point_cloud_handle = None
    if point_cloud_path and point_cloud_path.exists():
        try:
            if point_cloud_path.suffix.lower() == '.ply':
                # Simple PLY reader (you might want to use a more robust library)
                import trimesh
                mesh = trimesh.load(point_cloud_path)
                if hasattr(mesh, 'vertices'):
                    points = np.array(mesh.vertices)
                    colors = np.array(mesh.visual.vertex_colors[:, :3]) / 255.0 if hasattr(mesh.visual, 'vertex_colors') else None
                    if colors is None:
                        colors = np.ones_like(points) * 0.5  # Gray color
                    
                    point_cloud_handle = server.scene.add_point_cloud(
                        name="/point_cloud",
                        points=points,
                        colors=colors,
                        point_size=0.01,
                    )
                    print(f"Loaded point cloud with {len(points)} points")
        except ImportError:
            print("Install trimesh to load PLY files: pip install trimesh")
        except Exception as e:
            print(f"Failed to load point cloud: {e}")

    # GUI controls
    gui_reset_up = server.gui.add_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
    )

    # Reorient scene based on average camera direction
    if reorient_scene and len(camera_poses) > 0:
        # Calculate average up direction from camera poses
        up_vectors = []
        for pose in camera_poses:
            # Extract up vector (negative Y in camera frame)
            R = pose.rotation().as_matrix()
            up_vector = -R[:, 1]  # -Y axis in camera frame
            up_vectors.append(up_vector)
        
        average_up = np.mean(up_vectors, axis=0)
        average_up /= np.linalg.norm(average_up)
        server.scene.set_up_direction((average_up[0], average_up[1], average_up[2]))

    @gui_reset_up.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        # Set up direction based on current camera orientation
        R = vtf.SO3(client.camera.wxyz).as_matrix()
        up_direction = -R[:, 1]  # -Y axis
        client.camera.up_direction = up_direction

    gui_frames = server.gui.add_slider(
        "Max frames",
        min=1,
        max=len(camera_poses),
        step=1,
        initial_value=min(len(camera_poses), 50),
    )
    
    gui_scale = server.gui.add_slider(
        "Camera scale",
        min=0.01,
        max=1.0,
        step=0.01,
        initial_value=0.15,
    )

    if point_cloud_handle:
        gui_point_size = server.gui.add_slider(
            "Point size", min=0.001, max=0.1, step=0.001, initial_value=0.01
        )
        
        @gui_point_size.on_update
        def _(_) -> None:
            point_cloud_handle.point_size = gui_point_size.value

    frames: List[viser.FrameHandle] = []

    def visualize_frames() -> None:
        # Remove existing frames
        for frame in frames:
            frame.remove()
        frames.clear()

        # Select frames to display
        frame_indices = list(range(len(camera_poses)))
        random.shuffle(frame_indices)
        frame_indices = sorted(frame_indices[:gui_frames.value])

        for i in tqdm(frame_indices, desc="Loading camera frames"):
            pose = camera_poses[i]
            image_name = image_names[i]
            
            # Check if image exists
            image_path = images_path / image_name
            if not image_path.exists():
                # Try without leading path separators
                image_path = images_path / Path(image_name).name
                if not image_path.exists():
                    print(f"Warning: Image not found: {image_name}")
                    continue

            # Add camera frame
            frame = server.scene.add_frame(
                f"/camera_frame_{i}",
                wxyz=pose.rotation().wxyz,
                position=pose.translation(),
                axes_length=gui_scale.value,
                axes_radius=gui_scale.value * 0.03,
            )
            frames.append(frame)

            # Load and display image
            try:
                image = iio.imread(image_path)
                if downsample_factor > 1:
                    image = image[::downsample_factor, ::downsample_factor]
                
                # Calculate FOV from intrinsics
                H, W = camera_intrinsics['height'], camera_intrinsics['width']
                fy = camera_intrinsics['fy']
                fov = 2 * np.arctan2(H / 2, fy)
                aspect = W / H
                
                # Add camera frustum with image
                frustum = server.scene.add_camera_frustum(
                    f"/camera_frame_{i}/frustum",
                    fov=fov,
                    aspect=aspect,
                    scale=gui_scale.value,
                    image=image,
                )

                @frustum.on_click
                def _(_, frame=frame) -> None:
                    # Set viewer camera to clicked camera pose
                    for client in server.get_clients().values():
                        client.camera.wxyz = frame.wxyz
                        client.camera.position = frame.position

            except Exception as e:
                print(f"Failed to load image {image_path}: {e}")

    need_update = True

    @gui_frames.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    @gui_scale.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    # Add coordinate system axes
    server.scene.add_frame(
        "/world_frame",
        axes_length=0.5,
        axes_radius=0.02,
    )

    # Main loop
    while True:
        if need_update:
            need_update = False
            visualize_frames()

        time.sleep(1e-3)


if __name__ == "__main__":
    tyro.cli(main)