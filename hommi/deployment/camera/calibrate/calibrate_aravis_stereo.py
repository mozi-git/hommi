#!/usr/bin/env python3
"""Stereo fisheye calibration using Aravis cameras and a chessboard.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple
import glob

import cv2
import numpy as np

HOMMI_ROOT = Path(__file__).resolve().parents[4]
if str(HOMMI_ROOT) not in sys.path:
    sys.path.insert(0, str(HOMMI_ROOT))

from hommi.deployment.camera.camera_stream import AravisCameraStreamer


def _ensure_u8_bgr(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Got None image")
    if img.dtype != np.uint8:
        arr = img.astype(np.float32)
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        img = arr
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _corners_valid(corners: np.ndarray, width: int, height: int) -> bool:
    if corners is None:
        return False
    if not np.isfinite(corners).all():
        return False
    coords = corners.reshape(-1, 2)
    x = coords[:, 0]
    y = coords[:, 1]
    if (x < 0).any() or (x >= width).any() or (y < 0).any() or (y >= height).any():
        return False
    return True


class AravisStereoCamera:
    def __init__(self, left_key: str, right_key: str, config_path: str | None = None):
        self.left_key = left_key
        self.right_key = right_key
        self.streamer = AravisCameraStreamer(config_path=config_path)
        available = list(getattr(self.streamer, "_camera_map", {}).keys())
        for key in (left_key, right_key):
            if key not in getattr(self.streamer, "_camera_map", {}):
                raise SystemExit(f"Camera key '{key}' not found. Available: {available}")
        self.started = False

    def start(self) -> None:
        if not self.started:
            self.streamer.start()
            self.streamer.wait_until_ready(min_frames=1, timeout=5.0)
            self.started = True

    def stop(self) -> None:
        if self.started:
            self.streamer.stop()
            self.started = False

    def receive_images(self) -> Tuple[np.ndarray, np.ndarray]:
        frames, _ts = self.streamer.get_observation_window(1, include_timestamps=True)
        left = _ensure_u8_bgr(frames[self.left_key][0])
        right = _ensure_u8_bgr(frames[self.right_key][0])
        return left, right


class StereoFisheyeCalibrator:
    def __init__(
        self,
        camera: AravisStereoCamera,
        camera_resolution: Optional[Tuple[int, int]] = None,
        latency: Optional[float] = 0.0,
        image_save_path: str = "./images",
        camera_calibration_save_path: str = "./camera_calibration",
    ):
        self.camera = camera
        self.camera_resolution = camera_resolution
        self.latency = latency or 0.0
        self.is_running = False

        self.image_save_path = image_save_path
        self.camera_calibration_save_path = camera_calibration_save_path
        os.makedirs(self.image_save_path, exist_ok=True)
        os.makedirs(self.camera_calibration_save_path, exist_ok=True)

        self.DIM = None
        self.K1 = np.eye(3, dtype=np.float64)
        self.D1 = np.zeros((4, 1), dtype=np.float64)
        self.K2 = np.eye(3, dtype=np.float64)
        self.D2 = np.zeros((4, 1), dtype=np.float64)
        self.R = np.eye(3, dtype=np.float64)
        self.T = np.zeros((3, 1), dtype=np.float64)

        self.objpoints = []
        self.imgpoints_L = []
        self.imgpoints_R = []
        self.capture_index = 0

    def load_cached_pairs(self, chessboard_size, square_size):
        left_paths = sorted(glob.glob(os.path.join(self.image_save_path, "left_*.png")))
        right_paths = sorted(glob.glob(os.path.join(self.image_save_path, "right_*.png")))
        if not left_paths or not right_paths:
            return
        pairs = {}
        for path in left_paths:
            key = os.path.basename(path).replace("left_", "").replace(".png", "")
            pairs.setdefault(key, {})["L"] = path
        for path in right_paths:
            key = os.path.basename(path).replace("right_", "").replace(".png", "")
            pairs.setdefault(key, {})["R"] = path
        objp = np.zeros((1, chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size
        subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        added = 0
        for key, files in sorted(pairs.items()):
            if "L" not in files or "R" not in files:
                continue
            L = _ensure_u8_bgr(cv2.imread(files["L"], cv2.IMREAD_COLOR))
            R = _ensure_u8_bgr(cv2.imread(files["R"], cv2.IMREAD_COLOR))
            gL = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
            gR = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
            retL, cornersL = cv2.findChessboardCorners(
                gL,
                chessboard_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            retR, cornersR = cv2.findChessboardCorners(
                gR,
                chessboard_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if not (retL and retR):
                continue
            cornersL = cv2.cornerSubPix(
                gL,
                cornersL,
                (3, 3),
                (-1, -1),
                subpix_criteria,
            )
            cornersR = cv2.cornerSubPix(
                gR,
                cornersR,
                (3, 3),
                (-1, -1),
                subpix_criteria,
            )
            if not _corners_valid(cornersL, gL.shape[1], gL.shape[0]):
                continue
            if not _corners_valid(cornersR, gR.shape[1], gR.shape[0]):
                continue
            self.objpoints.append(objp)
            self.imgpoints_L.append(cornersL)
            self.imgpoints_R.append(cornersR)
            added += 1
        if added:
            print(f"Loaded {added} cached stereo pairs from {self.image_save_path}")

    def start_streaming(self) -> None:
        self.is_running = True
        self.camera.start()

    def stop_streaming(self) -> None:
        self.is_running = False
        self.camera.stop()

    def get_camera_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self.is_running:
            return None
        try:
            left, right = self.camera.receive_images()
            if self.camera_resolution is not None:
                left = cv2.resize(left, self.camera_resolution)
                right = cv2.resize(right, self.camera_resolution)
            return left, right
        except Exception as exc:
            print(f"Error capturing stereo frame: {exc}")
            traceback.print_exc()
            return None

    def calibrate_camera(self, chessboard_size=(8, 5), square_size=0.025, num_images=40):
        print(
            f"Starting stereo calibration. Show {chessboard_size[0]}x{chessboard_size[1]} chessboard..."
        )

        calibration_flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            + cv2.fisheye.CALIB_FIX_SKEW
            + cv2.fisheye.CALIB_FIX_K4
        )

        objp = np.zeros((1, chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size

        self.load_cached_pairs(chessboard_size, square_size)
        self.capture_index = len(self.objpoints)

        self.start_streaming()
        captured_images = 0
        last_capture_time = time.time() - 2

        win = "Stereo Calibration"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        while captured_images < num_images:
            frames = self.get_camera_frame()
            if frames is None:
                continue
            left, right = frames
            if left.shape[:2] != right.shape[:2]:
                right = cv2.resize(right, (left.shape[1], left.shape[0]))
            gL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            gR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

            retL, cornersL = cv2.findChessboardCorners(
                gL,
                chessboard_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            retR, cornersR = cv2.findChessboardCorners(
                gR,
                chessboard_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )

            display = np.concatenate([left, right], axis=1)
            if retL:
                cornersL = cv2.cornerSubPix(
                    gL,
                    cornersL,
                    (3, 3),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
                )
                cv2.drawChessboardCorners(display[:, :left.shape[1]], chessboard_size, cornersL, retL)
            if retR:
                cornersR = cv2.cornerSubPix(
                    gR,
                    cornersR,
                    (3, 3),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
                )
                cv2.drawChessboardCorners(display[:, left.shape[1]:], chessboard_size, cornersR, retR)

            cv2.putText(
                display,
                f"Captured: {captured_images}/{num_images} | L:{int(retL)} R:{int(retR)} | q to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            if retL != retR:
                fail_side = "right" if retL else "left"
                cv2.putText(
                    display,
                    f"Detection failed: {fail_side}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            cv2.imshow(win, display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if retL and retR and (time.time() - last_capture_time) > 1.0:
                if not _corners_valid(cornersL, gL.shape[1], gL.shape[0]):
                    print("[skip] left corners invalid/out of bounds")
                    continue
                if not _corners_valid(cornersR, gR.shape[1], gR.shape[0]):
                    print("[skip] right corners invalid/out of bounds")
                    continue
                self.objpoints.append(objp)
                self.imgpoints_L.append(cornersL)
                self.imgpoints_R.append(cornersR)
                captured_images += 1
                lp = os.path.join(self.image_save_path, f"left_{self.capture_index:04d}.png")
                rp = os.path.join(self.image_save_path, f"right_{self.capture_index:04d}.png")
                cv2.imwrite(lp, left)
                cv2.imwrite(rp, right)
                self.capture_index += 1
                last_capture_time = time.time()
                print(f"Captured image {captured_images}/{num_images}")

        cv2.destroyWindow(win)
        self.stop_streaming()

        if captured_images == 0:
            print("No images captured. Aborting.")
            return

        print("Computing stereo calibration...")
        try:
            self.DIM = gL.shape[::-1]
            print(f"Image size (W,H): {self.DIM}")
            print(f"Frames: {len(self.objpoints)} | L corners: {len(self.imgpoints_L)} | R corners: {len(self.imgpoints_R)}")
            obj = [op.astype(np.float64) for op in self.objpoints]
            imgL = [ip.reshape(1, -1, 2).astype(np.float64) for ip in self.imgpoints_L]
            imgR = [ip.reshape(1, -1, 2).astype(np.float64) for ip in self.imgpoints_R]
            if imgL and imgR:
                print(f"First L corners shape: {imgL[0].shape} | First R corners shape: {imgR[0].shape}")
                print(f"First obj shape: {obj[0].shape}")
            if imgL:
                widths = []
                heights = []
                for corners in imgL:
                    pts = corners.reshape(-1, 2)
                    widths.append(float(pts[:, 0].max() - pts[:, 0].min()))
                    heights.append(float(pts[:, 1].max() - pts[:, 1].min()))
                print(f"L corner bbox w/h range: w[{min(widths):.1f}, {max(widths):.1f}] h[{min(heights):.1f}, {max(heights):.1f}]")
            if imgR:
                widths = []
                heights = []
                for corners in imgR:
                    pts = corners.reshape(-1, 2)
                    widths.append(float(pts[:, 0].max() - pts[:, 0].min()))
                    heights.append(float(pts[:, 1].max() - pts[:, 1].min()))
                print(f"R corner bbox w/h range: w[{min(widths):.1f}, {max(widths):.1f}] h[{min(heights):.1f}, {max(heights):.1f}]")

            left_json = os.path.join(self.camera_calibration_save_path, "fisheye_calibration_left.json")
            right_json = os.path.join(self.camera_calibration_save_path, "fisheye_calibration_right.json")
            if os.path.exists(left_json) and os.path.exists(right_json):
                with open(left_json, "r") as f:
                    left_data = json.load(f)
                with open(right_json, "r") as f:
                    right_data = json.load(f)
                self.K1 = np.array(left_data["K"], dtype=np.float64)
                self.D1 = np.array(left_data["D"], dtype=np.float64)
                self.K2 = np.array(right_data["K"], dtype=np.float64)
                self.D2 = np.array(right_data["D"], dtype=np.float64)
                print("Loaded intrinsics from fisheye_calibration_left/right.json")
                rmsL = float(left_data.get("rms", 0.0))
                rmsR = float(right_data.get("rms", 0.0))
            else:
                rmsL, self.K1, self.D1, *_ = cv2.fisheye.calibrate(
                    obj, imgL, self.DIM, np.eye(3), np.zeros((4, 1)), None, None,
                    flags=calibration_flags,
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
                )
                rmsR, self.K2, self.D2, *_ = cv2.fisheye.calibrate(
                    obj, imgR, self.DIM, np.eye(3), np.zeros((4, 1)), None, None,
                    flags=calibration_flags,
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
                )

            stereo_flags = cv2.fisheye.CALIB_FIX_INTRINSIC | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            if hasattr(cv2.fisheye, "CALIB_USE_INTRINSIC_GUESS"):
                stereo_flags |= cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
            R_init = np.eye(3, dtype=np.float64)
            T_init = np.array([[0.10], [0.0], [0.0]], dtype=np.float64)
            try:
                rmsS, _K1, _D1, _K2, _D2, self.R, self.T, _E, _F = cv2.fisheye.stereoCalibrate(
                    obj, imgL, imgR,
                    self.K1, self.D1, self.K2, self.D2,
                    self.DIM,
                    R_init, T_init,
                    flags=stereo_flags,
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7),
                )
            except cv2.error as exc:
                print(f"[warn] fisheye stereoCalibrate failed: {exc}")
                print("[warn] Falling back to pinhole stereoCalibrate for initial R/T guess.")
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
                rms_p, _K1p, _D1p, _K2p, _D2p, R_init, T_init, _E, _F = cv2.stereoCalibrate(
                    [op.reshape(-1, 3) for op in obj],
                    [ip.reshape(-1, 2) for ip in imgL],
                    [ip.reshape(-1, 2) for ip in imgR],
                    self.K1.copy(), np.zeros((5, 1)),
                    self.K2.copy(), np.zeros((5, 1)),
                    self.DIM,
                    flags=cv2.CALIB_FIX_INTRINSIC,
                    criteria=criteria,
                )
                rmsS, _K1, _D1, _K2, _D2, self.R, self.T, _E, _F = cv2.fisheye.stereoCalibrate(
                    obj, imgL, imgR,
                    self.K1, self.D1, self.K2, self.D2,
                    self.DIM,
                    R_init, T_init,
                    flags=stereo_flags,
                    criteria=criteria,
                )

            R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
                self.K1, self.D1, self.K2, self.D2,
                self.DIM, self.R, self.T,
                flags=cv2.CALIB_ZERO_DISPARITY,
                balance=0.0,
                fov_scale=1.0,
            )

            calibration_data = {
                "DIM": self.DIM,
                "left": {"K": self.K1.tolist(), "D": self.D1.tolist(), "rms": float(rmsL)},
                "right": {"K": self.K2.tolist(), "D": self.D2.tolist(), "rms": float(rmsR)},
                "stereo": {"rms": float(rmsS), "R": self.R.tolist(), "T": self.T.tolist()},
                "rectify": {"R1": R1.tolist(), "R2": R2.tolist(), "P1": P1.tolist(), "P2": P2.tolist(), "Q": Q.tolist()},
            }

            out_path = os.path.join(self.camera_calibration_save_path, "stereo_fisheye_calibration.json")
            with open(out_path, "w") as f:
                json.dump(calibration_data, f, indent=2)

            print("\nCalibration successful!")
            print(f"Left RMS: {float(rmsL):.2f} | Right RMS: {float(rmsR):.2f} | Stereo RMS: {float(rmsS):.2f}")
            print(f"Saved stereo calibration to {out_path}")

        except Exception as exc:
            print(f"Stereo calibration failed: {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate a stereo Aravis camera pair.")
    parser.add_argument("--camera-config", default=None, help="Path to camera YAML. Defaults to HOMMI_CAMERA_CONFIG or source checkout config.")
    parser.add_argument("--left-key", default="camera_head_main_rgb")
    parser.add_argument("--right-key", default="camera_head_main_right_rgb")
    args = parser.parse_args()

    left_key = args.left_key
    right_key = args.right_key

    camera = AravisStereoCamera(left_key, right_key, config_path=args.camera_config)
    cam = StereoFisheyeCalibrator(
        camera=camera,
        camera_resolution=None,
        latency=0.0,
        image_save_path=f"./images_{left_key}_{right_key}",
        camera_calibration_save_path=f"./camera_calibration_{left_key}_{right_key}",
    )

    # IMPORTANT: chessboard_size is INNER corners (cols, rows)
    cam.calibrate_camera(chessboard_size=(8, 5), square_size=0.025, num_images=40)
