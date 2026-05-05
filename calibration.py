# Calibration code examples found online with OpenCV, but were incomplete
# This code is for 2 cameras, and the calibration is recorded in a .npz file

# Calibration command:
# python stereopsis_fixed.py calibrate --left-dir left_calib --right-dir right_calib --pattern "*.jpg" --corners-x 6 --corners-y 9 --square-size 20.0 --output stereo_calibration.npz
# Test command:
# python stereopsis_fixed.py demo --left-image left.jpg --right-image right.jpg --calibration stereo_calibration.npz


import argparse
import glob
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


# Default checkerboard means 6 x 9 INNER corners.
CHECKERBOARD = (6, 9)
SQUARE_SIZE_MM = 23
CALIBRATION_FILE = "stereo_calibration.npz"


def _build_object_points(checkerboard: Tuple[int, int], square_size: float) -> np.ndarray:
    """Create the real-world checkerboard corner coordinates in a flat Z=0 plane."""
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp


def _find_image_pairs(left_dir: str, right_dir: str, pattern: str) -> List[Tuple[str, str]]:
    left_paths = sorted(glob.glob(os.path.join(left_dir, pattern)))
    right_paths = sorted(glob.glob(os.path.join(right_dir, pattern)))

    if not left_paths or not right_paths:
        raise FileNotFoundError(
            f"No images found. left_dir={left_dir!r}, right_dir={right_dir!r}, pattern={pattern!r}"
        )
    if len(left_paths) != len(right_paths):
        raise ValueError(
            f"Mismatched number of images: {len(left_paths)} left vs {len(right_paths)} right. "
            "Use synchronized stereo pairs with matching filenames/order."
        )

    return list(zip(left_paths, right_paths))


def collect_calibration_points(
    left_dir: str,
    right_dir: str,
    checkerboard: Tuple[int, int],
    square_size: float,
    pattern: str = "*.png",
    preview: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], Tuple[int, int]]:
    """Find checkerboard corners in stereo image pairs."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = _build_object_points(checkerboard, square_size)

    objpoints: List[np.ndarray] = []
    imgpoints_left: List[np.ndarray] = []
    imgpoints_right: List[np.ndarray] = []
    image_size = None

    pairs = _find_image_pairs(left_dir, right_dir, pattern)
    print(f"Found {len(pairs)} stereo pairs.")


    os.makedirs("debug/left", exist_ok=True)
    os.makedirs("debug/right", exist_ok=True)
    i = 0

    for left_path, right_path in pairs:
    
        i += 1
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        if left_img is None or right_img is None:
            print(f"Skipping unreadable pair: {left_path}, {right_path}")
            continue

        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = gray_left.shape[::-1]
        elif image_size != gray_left.shape[::-1] or image_size != gray_right.shape[::-1]:
            raise ValueError("All calibration images must have the same size.")

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, checkerboard, flags)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, checkerboard, flags)

        if ret_left and ret_right:
            corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)


            objpoints.append(objp.copy())
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)

            print(f"Accepted pair: {Path(left_path).name} / {Path(right_path).name}")

            if preview:
                left_vis = cv2.drawChessboardCorners(left_img.copy(), checkerboard, corners_left, ret_left)
                right_vis = cv2.drawChessboardCorners(right_img.copy(), checkerboard, corners_right, ret_right)
                cv2.imshow("left corners", left_vis)
                cv2.imshow("right corners", right_vis)
                cv2.waitKey(250)

                
                cv2.imwrite(f"debug/left/left_{i:04d}.png", left_vis)
                cv2.imwrite(f"debug/right/right_{i:04d}.png", right_vis)



        else:
            print(
                f"Rejected pair: {Path(left_path).name} / {Path(right_path).name} "
                f"(left_found={ret_left}, right_found={ret_right})"
            )

    if preview:
        cv2.destroyAllWindows()

    if len(objpoints) < 8:
        raise RuntimeError(
            f"Only {len(objpoints)} valid stereo pairs found. Try to get at least 10-15 good pairs."
        )
    assert image_size is not None
    return objpoints, imgpoints_left, imgpoints_right, image_size


def calibrate_stereo(
    left_dir: str,
    right_dir: str,
    checkerboard: Tuple[int, int] = CHECKERBOARD,
    square_size: float = SQUARE_SIZE_MM,
    pattern: str = "*.png",
    output_file: str = CALIBRATION_FILE,
    preview: bool = False,
) -> None:
    objpoints, imgpoints_left, imgpoints_right, image_size = collect_calibration_points(
        left_dir, right_dir, checkerboard, square_size, pattern, preview
    )

    # Calibrate each camera independently first.
    ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_left, image_size, None, None
    )
    ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_right, image_size, None, None
    )

    stereo_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    stereo_flags = cv2.CALIB_FIX_INTRINSIC

    ret_s, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        mtx_l,
        dist_l,
        mtx_r,
        dist_r,
        image_size,
        criteria=stereo_criteria,
        flags=stereo_flags,
    )

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, image_size, R, T, alpha=0
    )

    np.savez(
        output_file,
        image_size=np.array(image_size),
        checkerboard=np.array(checkerboard),
        square_size=np.array(square_size, dtype=np.float32),
        mtx_l=mtx_l,
        dist_l=dist_l,
        mtx_r=mtx_r,
        dist_r=dist_r,
        R=R,
        T=T,
        E=E,
        F=F,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        roi1=np.array(roi1),
        roi2=np.array(roi2),
        mono_rms_left=np.array(ret_l),
        mono_rms_right=np.array(ret_r),
        stereo_rms=np.array(ret_s),
    )

    print("Saved stereo calibration to:", output_file)
    print("Left RMS reprojection error:", ret_l)
    print("Right RMS reprojection error:", ret_r)
    print("Stereo RMS reprojection error:", ret_s)
    print("Baseline (same units as square_size):", float(np.linalg.norm(T)))


def load_calibration(calibration_file: str = CALIBRATION_FILE) -> dict:
    data = np.load(calibration_file)
    return {key: data[key] for key in data.files}


def rectify_pair(left_img: np.ndarray, right_img: np.ndarray, calib: dict) -> Tuple[np.ndarray, np.ndarray]:
    image_size = tuple(int(v) for v in calib["image_size"])
    h, w = left_img.shape[:2]
    if (w, h) != image_size:
        raise ValueError(f"Image size {(w, h)} does not match calibration size {image_size}.")

    map1x, map1y = cv2.initUndistortRectifyMap(
        calib["mtx_l"], calib["dist_l"], calib["R1"], calib["P1"], image_size, cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        calib["mtx_r"], calib["dist_r"], calib["R2"], calib["P2"], image_size, cv2.CV_32FC1
    )

    left_rect = cv2.remap(left_img, map1x, map1y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_img, map2x, map2y, cv2.INTER_LINEAR)
    return left_rect, right_rect


def compute_disparity(left_rect: np.ndarray, right_rect: np.ndarray) -> np.ndarray:
    gray_l = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY) if left_rect.ndim == 3 else left_rect
    gray_r = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY) if right_rect.ndim == 3 else right_rect

    # numDisparities must be divisible by 16.
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 8,
        blockSize=5,
        P1=8 * 3 * 5 ** 2,
        P2=32 * 3 * 5 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
    return disparity


def reconstruct_3d(disparity: np.ndarray, calib: dict) -> np.ndarray:
    return cv2.reprojectImageTo3D(disparity, calib["Q"])


def save_disparity_preview(disparity: np.ndarray, out_path: str = "disparity_preview.png") -> None:
    disp = disparity.copy()
    valid = disp > disp.min()
    if np.any(valid):
        disp_norm = np.zeros_like(disp, dtype=np.uint8)
        cv2.normalize(disp, disp_norm, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(out_path, disp_norm)
        print("Saved disparity preview to:", out_path)
    else:
        print("Disparity map does not contain valid values; preview not saved.")


def run_demo(left_image: str, right_image: str, calibration_file: str = CALIBRATION_FILE) -> None:
    calib = load_calibration(calibration_file)
    left_img = cv2.imread(left_image)
    right_img = cv2.imread(right_image)
    if left_img is None or right_img is None:
        raise FileNotFoundError("Could not read left or right demo image.")

    left_rect, right_rect = rectify_pair(left_img, right_img, calib)
    disparity = compute_disparity(left_rect, right_rect)
    points_3d = reconstruct_3d(disparity, calib)


    combined = np.hstack((left_rect, right_rect))

    for y in range(0, combined.shape[0], 40):
        cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)

    cv2.imshow("Rectified Pair", combined)
    cv2.waitKey(0)

    save_disparity_preview(disparity)
    np.save("points_3d.npy", points_3d)
    print("Saved 3D points to: points_3d.npy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stereo calibration and rectified disparity pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cal = subparsers.add_parser("calibrate", help="Calibrate a stereo camera pair from checkerboard images.")
    cal.add_argument("--left-dir", required=True, help="Directory containing left camera checkerboard images.")
    cal.add_argument("--right-dir", required=True, help="Directory containing right camera checkerboard images.")
    cal.add_argument("--pattern", default="*.png", help="Glob pattern for calibration images.")
    cal.add_argument("--corners-x", type=int, default=CHECKERBOARD[0], help="Number of inner corners along X.")
    cal.add_argument("--corners-y", type=int, default=CHECKERBOARD[1], help="Number of inner corners along Y.")
    cal.add_argument("--square-size", type=float, default=SQUARE_SIZE_MM, help="Checkerboard square size in mm.")
    cal.add_argument("--output", default=CALIBRATION_FILE, help="Path to save calibration .npz file.")
    cal.add_argument("--preview", action="store_true", help="Show detected checkerboard corners briefly.")

    demo = subparsers.add_parser("demo", help="Rectify one stereo pair and compute disparity/3D points.")
    demo.add_argument("--left-image", required=True, help="Left image path.")
    demo.add_argument("--right-image", required=True, help="Right image path.")
    demo.add_argument("--calibration", default=CALIBRATION_FILE, help="Calibration .npz file path.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "calibrate":
        calibrate_stereo(
            left_dir=args.left_dir,
            right_dir=args.right_dir,
            checkerboard=(args.corners_x, args.corners_y),
            square_size=args.square_size,
            pattern=args.pattern,
            output_file=args.output,
            preview=args.preview,
        )
    elif args.command == "demo":
        run_demo(args.left_image, args.right_image, args.calibration)


if __name__ == "__main__":
    main()
