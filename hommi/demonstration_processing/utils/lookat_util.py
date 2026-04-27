import numpy as np

def compute_center_lookat_from_pointmap(pointmap: np.ndarray, top_k: int = 50) -> np.ndarray:
    """Derive a look-at point from a pointmap by sampling the closest rays.

    Args:
        pointmap: Array shaped (H, W, 3) containing xyz coordinates in the camera frame.
        top_k: Number of closest points (by radial distance) to aggregate.

    Returns:
        A 3-D point (same dtype as input if floating) where x=y=0 and z is the
        median depth of the selected points. Returns zeros if no valid points exist.
    """
    if pointmap is None:
        raise ValueError("pointmap cannot be None")
    points = np.asarray(pointmap)
    dtype = points.dtype if np.issubdtype(points.dtype, np.floating) else np.float16
    if points.size == 0:
        return np.zeros(3, dtype=dtype)

    flat = points.reshape(-1, 3)
    valid_mask = (flat[:, 2] > 0) & np.isfinite(flat).all(axis=1)
    if not np.any(valid_mask):
        return np.zeros(3, dtype=dtype)

    valid_points = flat[valid_mask]
    distances = (valid_points[:, 0] ** 2) + (valid_points[:, 1] ** 2)
    k = min(top_k, len(valid_points))
    closest_idx = np.argpartition(distances, k - 1)[:k]
    median_z = np.median(valid_points[closest_idx, 2])

    result = np.zeros(3, dtype=np.float16)
    result[2] = float(median_z)
    return result.astype(dtype)
