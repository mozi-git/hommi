import sys
import os
from typing import Dict, Callable, Tuple, List, Optional
import numpy as np
import torch
import cv2
# from diffusion_policy.common.cv2_util import get_image_transform
from hommi.common.cv_util import get_image_transform_with_border, depth2xyzmap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'deps'))
from FoundationStereo.core.utils.utils import InputPadder

_RECTIFY_MAP_CACHE = {}
_RECTIFY_DEBUG_COUNTER = 0


def _double_sphere_project(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        K: np.ndarray,
        xi: float,
        alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
    d1 = np.sqrt(x * x + y * y + z * z)
    z1 = xi * d1 + z
    d2 = np.sqrt(x * x + y * y + z1 * z1)
    denom = alpha * d2 + (1.0 - alpha) * z1
    valid = denom > 1e-8
    u = np.full_like(x, -1.0, dtype=np.float32)
    v = np.full_like(y, -1.0, dtype=np.float32)
    u[valid] = K[0, 0] * (x[valid] / denom[valid]) + K[0, 2]
    v[valid] = K[1, 1] * (y[valid] / denom[valid]) + K[1, 2]
    return u, v


def _double_sphere_rectify_maps(
        image_size: Tuple[int, int],
        K_rect: np.ndarray,
        K_cam: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
    # dist_coeffs are [xi, alpha] for double sphere.
    xi, alpha = float(dist_coeffs[0]), float(dist_coeffs[1])
    # image_size follows OpenCV convention: (width, height)
    W, H = image_size
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    x = (u - K_rect[0, 2]) / K_rect[0, 0]
    y = (v - K_rect[1, 2]) / K_rect[1, 1]
    z = np.ones_like(x, dtype=np.float32)
    map_x, map_y = _double_sphere_project(x, y, z, K_cam, xi, alpha)
    return map_x, map_y


def _rectify_stereo_batch(
        imgs_left: np.ndarray,
        imgs_right: np.ndarray,
        K: np.ndarray,
        dist_coeffs: np.ndarray,
        baseline: float,
        dist_model: str,
        K_right: Optional[np.ndarray] = None,
        dist_coeffs_right: Optional[np.ndarray] = None,
        K_rect: Optional[np.ndarray] = None,
        rect_R1: Optional[np.ndarray] = None,
        rect_R2: Optional[np.ndarray] = None,
        rect_P1: Optional[np.ndarray] = None,
        rect_P2: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    H, W = imgs_left[0].shape[:2]
    dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    dist_coeffs_right = (
        np.asarray(dist_coeffs_right, dtype=np.float64).reshape(-1)
        if dist_coeffs_right is not None
        else dist_coeffs
    )
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    K_right = np.asarray(K_right if K_right is not None else K, dtype=np.float64).reshape(3, 3)
    K_rect = np.asarray(K_rect if K_rect is not None else K, dtype=np.float64).reshape(3, 3)
    cache_key = (
        H,
        W,
        dist_model,
        tuple(np.round(K.flatten(), 12)),
        tuple(np.round(dist_coeffs.flatten(), 12)),
        tuple(np.round(K_right.flatten(), 12)),
        tuple(np.round(dist_coeffs_right.flatten(), 12)),
        tuple(np.round(K_rect.flatten(), 12)),
        float(baseline),
    )
    cache_entry = _RECTIFY_MAP_CACHE.get(cache_key)
    if cache_entry is None:
        image_size = (W, H)
        if dist_model == "double_sphere":
            if dist_coeffs.size < 2 or dist_coeffs_right.size < 2:
                raise ValueError("double_sphere requires dist_coeffs=[xi, alpha] for both cameras.")
            map1x, map1y = _double_sphere_rectify_maps(image_size, K_rect, K, dist_coeffs)
            map2x, map2y = _double_sphere_rectify_maps(image_size, K_rect, K_right, dist_coeffs_right)
            rectified_K = K_rect
        elif dist_model in ("pinhole", "plumb_bob", "opencv"):
            R = np.eye(3, dtype=np.float64)
            T = np.array([baseline, 0.0, 0.0], dtype=np.float64)
            R1, R2, P1, P2, _Q, _roi1, _roi2 = cv2.stereoRectify(
                K, dist_coeffs, K, dist_coeffs, image_size, R, T,
                flags=cv2.CALIB_ZERO_DISPARITY,
                alpha=0.0,
            )
            map1x, map1y = cv2.initUndistortRectifyMap(
                K, dist_coeffs, R1, P1, image_size, cv2.CV_32FC1
            )
            map2x, map2y = cv2.initUndistortRectifyMap(
                K, dist_coeffs, R2, P2, image_size, cv2.CV_32FC1
            )
            rectified_K = P1[:3, :3]
        elif dist_model == "fisheye":
            print("Using fisheye stereo rectification.")
            dist_coeffs = dist_coeffs.reshape(-1, 1)
            dist_coeffs_right = dist_coeffs_right.reshape(-1, 1)
            if rect_R1 is None or rect_R2 is None or rect_P1 is None or rect_P2 is None:
                R = np.eye(3, dtype=np.float64)
                T = np.array([baseline, 0.0, 0.0], dtype=np.float64)
                R1, R2, P1, P2, _Q = cv2.fisheye.stereoRectify(
                    K, dist_coeffs, K_right, dist_coeffs_right, image_size, R, T,
                    flags=cv2.CALIB_ZERO_DISPARITY,
                    balance=0.0,
                    fov_scale=1.0,
                )
            else:
                R1 = rect_R1
                R2 = rect_R2
                P1 = rect_P1
                P2 = rect_P2
            map1x, map1y = cv2.fisheye.initUndistortRectifyMap(
                K, dist_coeffs, R1, P1, image_size, cv2.CV_16SC2
            )
            map2x, map2y = cv2.fisheye.initUndistortRectifyMap(
                K_right, dist_coeffs_right, R2, P2, image_size, cv2.CV_16SC2
            )
            rectified_K = P1[:3, :3]
        else:
            raise ValueError(f"Unsupported dist_model '{dist_model}' for rectification.")

        cache_entry = (map1x, map1y, map2x, map2y, rectified_K)
        _RECTIFY_MAP_CACHE[cache_key] = cache_entry

    map1x, map1y, map2x, map2y, rectified_K = cache_entry
    imgs_left_rect = np.stack(
        [cv2.remap(img, map1x, map1y, cv2.INTER_LINEAR) for img in imgs_left], axis=0
    )
    imgs_right_rect = np.stack(
        [cv2.remap(img, map2x, map2y, cv2.INTER_LINEAR) for img in imgs_right], axis=0
    )
    return imgs_left_rect, imgs_right_rect, rectified_K

def get_real_obs_resolution(
        shape_meta: dict
        ) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co,ho,wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res


def get_real_obs_dict(
        env_obs: Dict[str, np.ndarray],
        shape_meta: dict,
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform_with_border(
                    input_res=(wi,hi),
                    output_res=(wo,ho),
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        elif type == 'low_dim':
            this_data_in = env_obs[key]
            obs_dict_np[key] = this_data_in
    return obs_dict_np


def rectify_image_batch(
        imgs: np.ndarray,
        K: np.ndarray,
        dist_coeffs: np.ndarray,
        dist_model: str,
        baseline: float,
        scale: float = 1.0,
        K_rect: Optional[np.ndarray] = None,
        calib_dim: Optional[Tuple[int, int]] = None,
        debug_print: bool = False,
        rect_R: Optional[np.ndarray] = None,
        rect_P: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
    imgs = [cv2.resize(img, dsize=None, fx=scale, fy=scale) for img in imgs]
    imgs = np.stack(imgs, axis=0)
    def _scale_K_to_image(K_in: np.ndarray, image_size: Tuple[int, int], calib_size: Optional[Tuple[int, int]]) -> np.ndarray:
        if calib_size is None:
            scale_factor = scale
            K_out = K_in.copy()
            K_out[:2] *= scale_factor
            return K_out
        scale_x = image_size[0] / float(calib_size[0])
        scale_y = image_size[1] / float(calib_size[1])
        K_out = K_in.copy()
        K_out[0, 0] *= scale_x
        K_out[1, 1] *= scale_y
        K_out[0, 2] *= scale_x
        K_out[1, 2] *= scale_y
        return K_out

    def _scale_P_to_image(P_in: Optional[np.ndarray], image_size: Tuple[int, int], calib_size: Optional[Tuple[int, int]]) -> Optional[np.ndarray]:
        if P_in is None or calib_size is None:
            return P_in
        scale_x = image_size[0] / float(calib_size[0])
        scale_y = image_size[1] / float(calib_size[1])
        P_out = P_in.copy()
        P_out[0, 0] *= scale_x
        P_out[0, 2] *= scale_x
        P_out[0, 3] *= scale_x
        P_out[1, 1] *= scale_y
        P_out[1, 2] *= scale_y
        P_out[1, 3] *= scale_y
        return P_out

    H, W = imgs[0].shape[:2]
    intrinsic = _scale_K_to_image(K, (W, H), calib_dim)
    imgs_rect, _imgs_right, rectified_K = _rectify_stereo_batch(
        imgs_left=imgs,
        imgs_right=imgs,
        K=intrinsic,
        dist_coeffs=dist_coeffs,
        baseline=baseline,
        dist_model=dist_model,
        K_right=intrinsic,
        dist_coeffs_right=dist_coeffs,
        K_rect=_scale_K_to_image(K_rect, (W, H), calib_dim) if K_rect is not None else None,
        rect_R1=rect_R,
        rect_R2=rect_R,
        rect_P1=_scale_P_to_image(rect_P, (W, H), calib_dim),
        rect_P2=_scale_P_to_image(rect_P, (W, H), calib_dim),
    )
    if debug_print:
        print(f"[stereo] rectified K (image_size=({W}, {H})):\n{rectified_K}")
    return imgs_rect, rectified_K

def get_pointmap_from_stereo(
        model,
        imgs_left: np.ndarray,
        imgs_right: np.ndarray,
        K: np.ndarray,
        baseline: float,
        scale: float = 1.0,
        dist_coeffs: Optional[np.ndarray] = None,
        dist_model: str = "pinhole",
        rectify: bool = False,
        K_right: Optional[np.ndarray] = None,
        dist_coeffs_right: Optional[np.ndarray] = None,
        K_rect: Optional[np.ndarray] = None,
        debug_dir: Optional[str] = None,
        calib_dim: Optional[Tuple[int, int]] = None,
        debug_print: bool = False,
        rect_R1: Optional[np.ndarray] = None,
        rect_R2: Optional[np.ndarray] = None,
        rect_P1: Optional[np.ndarray] = None,
        rect_P2: Optional[np.ndarray] = None,
    ):
    # use foundation stereo to estimate pointmap from stereo images
    # imgs_left: a batch of images [T, H, W, 3]
    imgs_left = [cv2.resize(img, dsize=None, fx=scale, fy=scale) for img in imgs_left]
    imgs_right = [cv2.resize(img, dsize=None, fx=scale, fy=scale) for img in imgs_right]
    imgs_left = np.stack(imgs_left, axis=0)   # [T, H, W, 3]
    imgs_right = np.stack(imgs_right, axis=0) # [T, H, W, 3]
    H, W = imgs_left[0].shape[:2]
    # print(f"Rectify stereo images: input size ({imgs_left.shape}), scaled to ({W}x{H})")
    def _scale_K_to_image(K_in: np.ndarray, image_size: Tuple[int, int], calib_size: Optional[Tuple[int, int]]) -> np.ndarray:
        if calib_size is None:
            scale_factor = scale
            K_out = K_in.copy()
            K_out[:2] *= scale_factor
            return K_out
        scale_x = image_size[0] / float(calib_size[0])
        scale_y = image_size[1] / float(calib_size[1])
        K_out = K_in.copy()
        K_out[0, 0] *= scale_x
        K_out[1, 1] *= scale_y
        K_out[0, 2] *= scale_x
        K_out[1, 2] *= scale_y
        return K_out

    def _scale_P_to_image(P_in: Optional[np.ndarray], image_size: Tuple[int, int], calib_size: Optional[Tuple[int, int]]) -> Optional[np.ndarray]:
        if P_in is None or calib_size is None:
            return P_in
        scale_x = image_size[0] / float(calib_size[0])
        scale_y = image_size[1] / float(calib_size[1])
        P_out = P_in.copy()
        P_out[0, 0] *= scale_x
        P_out[0, 2] *= scale_x
        P_out[0, 3] *= scale_x
        P_out[1, 1] *= scale_y
        P_out[1, 2] *= scale_y
        P_out[1, 3] *= scale_y
        return P_out

    intrinsic = _scale_K_to_image(K, (W, H), calib_dim)
    if debug_print:
        print(f"[stereo] image_size=({W}, {H}) baseline={baseline}")
        print(f"[stereo] intrinsic K:\n{intrinsic}")
    if rectify and dist_coeffs is not None:
        imgs_left, imgs_right, intrinsic = _rectify_stereo_batch(
            imgs_left=imgs_left,
            imgs_right=imgs_right,
            K=intrinsic,
            dist_coeffs=dist_coeffs,
            baseline=baseline,
            dist_model=dist_model,
            K_right=_scale_K_to_image(K_right, (W, H), calib_dim) if K_right is not None else None,
            dist_coeffs_right=dist_coeffs_right,
            K_rect=_scale_K_to_image(K_rect, (W, H), calib_dim) if K_rect is not None else None,
            rect_R1=rect_R1,
            rect_R2=rect_R2,
            rect_P1=_scale_P_to_image(rect_P1, (W, H), calib_dim),
            rect_P2=_scale_P_to_image(rect_P2, (W, H), calib_dim),
        )
        if debug_print:
            print(f"[stereo] rectified K (image_size=({W}, {H})):\n{intrinsic}")
        if debug_dir:
            global _RECTIFY_DEBUG_COUNTER
            os.makedirs(debug_dir, exist_ok=True)
            for i in range(imgs_left.shape[0]):
                left = imgs_left[i]
                right = imgs_right[i]
                if left.dtype != np.uint8:
                    left = np.clip(left * 255.0, 0, 255).astype(np.uint8)
                if right.dtype != np.uint8:
                    right = np.clip(right * 255.0, 0, 255).astype(np.uint8)
                stem = f"rectify_{_RECTIFY_DEBUG_COUNTER:06d}_t{i:03d}"
                cv2.imwrite(os.path.join(debug_dir, f"{stem}_left.png"), left)
                cv2.imwrite(os.path.join(debug_dir, f"{stem}_right.png"), right)
            _RECTIFY_DEBUG_COUNTER += 1
    imgs_left = torch.as_tensor(imgs_left).cuda().float().permute(0,3,1,2)    # [T, 3, H, W]
    imgs_right = torch.as_tensor(imgs_right).cuda().float().permute(0,3,1,2)  # [T, 3, H, W]
    # print(f"imgs_left shape: {imgs_left.shape}, imgs_right shape: {imgs_right.shape}")
    padder = InputPadder(imgs_left.shape, divis_by=32, force_square=False)
    imgs_left, imgs_right = padder.pad(imgs_left, imgs_right)
    # print(f"imgs_left shape: {imgs_left.shape}, imgs_right shape: {imgs_right.shape}")

    with torch.cuda.amp.autocast(True):
        disp = model.forward(imgs_left, imgs_right, iters=12, test_mode=True)   # [T, 1, H, W]
    disp = padder.unpad(disp.float())   # [T, 1, H, W]
    disp = disp.data.cpu().numpy()
    pointmaps = np.stack([
        depth2xyzmap(
            intrinsic[0, 0] * baseline / disp[i].reshape(H, W), intrinsic,
        ) for i in range(disp.shape[0])
    ], axis=0)  # [T, H, W, 3]
    return pointmaps
