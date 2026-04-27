import cv2
import numpy as np
import scipy.spatial.transform as st
import matplotlib.pyplot as plt
from datetime import datetime

IPHONE_DEPTH_INTRINSIC = np.array([[4.665952e+02, 0.000000e+00, 3.189309e+02], [0.000000e+00, 4.665952e+02, 2.408455e+02], [0.000000e+00, 0.000000e+00, 1.000000e+00]])

def get_image_transform_with_border(in_res, out_res, mode='rgb', bgr_to_rgb: bool=False):
    """ adds a border to make the input image square, and then resizes it to the output resolution """
    iw, ih = in_res
    interp_method = cv2.INTER_AREA
    if mode == 'depth' or mode == 'pointmap':
        interp_method = cv2.INTER_NEAREST  # avoid blending invalid depth

    # Determine the size of the square
    size = max(iw, ih)
    top = (size - ih) // 2
    bottom = size - ih - top
    left = (size - iw) // 2
    right = size - iw - left

    def transform(img: np.ndarray):
        if mode == 'rgb':
            assert img.shape == (ih, iw, 3)
            padded = cv2.copyMakeBorder(
                img, top, bottom, left, right,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )
            resized = cv2.resize(padded, out_res, interpolation=interp_method)
            if bgr_to_rgb:
                resized = resized[:, :, ::-1]
            return resized
        
        elif mode == 'depth' or mode == 'pointmap':
            assert img.dtype in [np.float16, np.float32]
            img = img.astype(np.float32)

            padded = cv2.copyMakeBorder(
                img, top, bottom, left, right,
                borderType=cv2.BORDER_CONSTANT,
                value=0 if mode == 'depth' else [0, 0, 0]
            )
            resized = cv2.resize(padded, out_res, interpolation=interp_method)
            resized = resized.astype(np.float16)
            return resized
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    
    return transform

def depth2pointcloud(depth, rgb, confidence, intrinsic, depth_scale, width, height):
    import open3d as o3d
    depth[confidence != 2] = 0
    depth_o3d = o3d.geometry.Image(
        np.ascontiguousarray(depth * depth_scale).astype(np.float32)
    )
    rgb_o3d = o3d.geometry.Image(
        np.ascontiguousarray(rgb).astype(np.uint8)
    )

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, convert_rgb_to_intensity=False
    )

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=intrinsic[0, 0],
        fy=intrinsic[1, 1],
        cx=intrinsic[0, 2],
        cy=intrinsic[1, 2],
    )
    temp = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, camera_intrinsics)
    temp.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = temp.points
    pcd.colors = temp.colors
    # plot
    o3d.visualization.draw_geometries([pcd])
    return pcd

def depth2xyzmap(depth:np.ndarray, K, uvs:np.ndarray=None):
    H,W = depth.shape[:2]
    if uvs is None:
        vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:,0]
        vs = uvs[:,1]
    zs = depth[vs,us]
    xs = (us-K[0,2])*zs/K[0,0]
    ys = (vs-K[1,2])*zs/K[1,1]
    pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
    xyz_map = np.zeros((H,W,3), dtype=np.float16)
    xyz_map[vs,us] = pts
    return xyz_map

def proj_3d_to_2d(pt3d, K=IPHONE_DEPTH_INTRINSIC):
    """
    Projects a 3D point (x, y, z) to image pixel coordinates (u, v) using camera intrinsic K.
    pt3d: (3,) or (N, 3) numpy array; 3D point(s)
    K: (3,3) camera intrinsics matrix
    Returns: (u, v) or (N, 2) array of image coordinates
    """
    pt3d = np.asarray(pt3d)
    if pt3d.ndim == 1:
        x, y, z = pt3d
        u = K[0,0] * x / (z + 1e-6) + K[0,2]
        v = K[1,1] * y / (z + 1e-6) + K[1,2]
        return u, v
    elif pt3d.ndim == 2:
        x = pt3d[:,0]
        y = pt3d[:,1]
        z = pt3d[:,2]
        u = K[0,0] * x / (z + 1e-6) + K[0,2]
        v = K[1,1] * y / (z + 1e-6) + K[1,2]
        return u, v
    else:
        raise ValueError('pt3d must have shape (3,) or (N,3)')
    
def overlay_lookat_on_images(
    batch,
    image_key='camera_head_main_rgb',
    lookat_key='camera_head_lookatpoint',
    # action_lookat_key='camera_head_lookatpoint',   # optional; if you want to overlay policy prediction too
    marker_obs='o', marker_action='x',
    color_obs='lime',  color_action='red'
):
    """
    Visualizes batch images with overlaid look at point.
    batch: Output of your __getitem__, as dict with 'obs', etc.
    image_key: key for the RGB image in batch['obs'].
    lookat_key: key for the lookat point in batch['obs'].
    action_lookat_key: if present, key for predicted lookat point in batch['action'].
    proj_3d_to_2d: function(xyz_array) -> (u,v) array, projects 3D points to image plane.
    """
    obs = batch['obs']
    images = obs[image_key][0].cpu().numpy()    # (T, C, H, W) or (B, T, C, H, W)
    obs_lookat = obs[lookat_key][0].cpu().numpy()  # (T, 3)
    actions = batch['action'][0].cpu().numpy()     # (action-horizon, 23)
    img = np.transpose(images[-1], (1, 2, 0))   # (H, W, 3)
    plt.imshow(img)
    point = obs_lookat[-1]
    u, v = proj_3d_to_2d(point)
    plt.scatter([u], [v], marker=marker_obs, color=color_obs, s=60, label='obs lookat')

    action_point = actions[..., 9:12]
    au, av = proj_3d_to_2d(action_point)
    plt.scatter([au], [av], marker=marker_action, color=color_action, s=60, label='action lookat')

    plt.axis('off')
    plt.legend()
    print(f"saving visualizations")
    plt.savefig(f'./runs/lookatpoint/{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')


def apply_cone_frustum_mask(
    image: np.ndarray,
    gripper_pose: np.ndarray,
    K: np.ndarray,
    world_to_cam: np.ndarray,
    cone_length_top: float = 0.13,
    cone_radius_top: float = 0.05,
    cone_length: float = 0.5,
    cone_radius: float = 0.15,
    num_circle_pts: int = 40
) -> np.ndarray:
    """
    Projects a circular cone frustum from a gripper and masks the region in the image.
    The cone starts at the gripper's origin and points along -Z in the gripper frame.
    """

    # Decompose pose
    pos = gripper_pose[:3]
    axis_angle = gripper_pose[3:]
    R = st.Rotation.from_rotvec(axis_angle)

    # Sample points on the base circle (in local gripper frame)
    angles = np.linspace(0, 2*np.pi, num_circle_pts, endpoint=False)
    top_circle_local = np.stack([
        cone_radius_top * np.cos(angles),
        cone_radius * np.sin(angles),
        -np.ones_like(angles) * cone_length_top
    ], axis=1)
    base_circle_local = np.stack([
        cone_radius * np.cos(angles),
        cone_radius * np.sin(angles),
        -np.ones_like(angles) * cone_length  # at end of cone along -Z
    ], axis=1)  # shape (N, 3)

    # Transform to world
    base_circle_world = pos[None] + (R @ base_circle_local.T).T  # shape (N, 3)
    top_circle_world = pos[None] + (R @ top_circle_local.T).T

    # Combine all 3D points
    cone_pts_world = np.vstack([top_circle_world, base_circle_world])  # shape (N+1, 3)

    # Transform to camera frame
    cone_pts_cam = (world_to_cam[:3, :3] @ cone_pts_world.T + world_to_cam[:3, 3:4]).T

    # Project valid points
    cone_pts_cam = cone_pts_cam[cone_pts_cam[:, 2] > 0]
    if len(cone_pts_cam) < 3:
        return image  # not enough visible points

    proj_2d = (K @ cone_pts_cam.T).T
    proj_2d = proj_2d[:, :2] / proj_2d[:, 2:3]
    proj_2d = np.round(proj_2d).astype(np.int32)

    # Ensure valid coordinates
    h, w = image.shape[:2]
    proj_2d = np.clip(proj_2d, 0, [w - 1, h - 1])

    # Fill the polygon mask (excluding the origin for circular silhouette)
    if len(proj_2d) > 3:
        cv2.fillConvexPoly(image, proj_2d, color=(0, 0, 0))

    return image
