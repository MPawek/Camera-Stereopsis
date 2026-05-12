"""
silhouette_stereo_measure_fullmask_clean.py

Purpose
-------
Generic stereo size check for an object segmented by a partner CAD/silhouette pipeline.

This script accepts either:
    1. full-frame filled masks, or
    2. full-frame edge/CAD overlay images that can be converted into filled masks.

It rectifies the left/right images and masks using a stereo calibration .npz file,
measures visible object width and height from the rectified masks, applies empirical
axis-specific correction factors, and prints PASS/FAIL for both dimensions.

Current empirical corrections from your setup:
    width-scale-correction  = 0.6831
    height-scale-correction = 0.8008

Notes
-----
- Width is visible silhouette width. For a cube at an angle, this may include a side face.
- For a sphere, width and height should both approximate diameter.
- For a cylinder, width approximates diameter and height approximates cylinder height.
- The padding fallback exists only because the calibration images were 1282x759 while
  current camera frames are 1280x720.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# -----------------------------
# Calibration / rectification
# -----------------------------


def load_calibration(path):
    calib = np.load(path)

    required = [
        "mtx_l", "dist_l", "mtx_r", "dist_r",
        "R1", "R2", "P1", "P2", "Q", "image_size",
    ]
    missing = [key for key in required if key not in calib]
    if missing:
        raise KeyError(
            f"Calibration file is missing keys: {missing}. "
            "Expected mtx_l, dist_l, mtx_r, dist_r, R1, R2, P1, P2, Q, image_size."
        )

    image_size = tuple(int(v) for v in calib["image_size"])  # OpenCV: (width, height)
    print(f"Using calibration image size: {image_size}")

    map_lx, map_ly = cv2.initUndistortRectifyMap(
        calib["mtx_l"], calib["dist_l"], calib["R1"], calib["P1"],
        image_size, cv2.CV_32FC1,
    )

    map_rx, map_ry = cv2.initUndistortRectifyMap(
        calib["mtx_r"], calib["dist_r"], calib["R2"], calib["P2"],
        image_size, cv2.CV_32FC1,
    )

    return {
        "image_size": image_size,
        "mapLx": map_lx,
        "mapLy": map_ly,
        "mapRx": map_rx,
        "mapRy": map_ry,
        "Q": calib["Q"],
    }


def rectify_image(img, map_x, map_y):
    return cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def rectify_mask(mask, map_x, map_y):
    rectified = cv2.remap(
        mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (rectified > 0).astype(np.uint8) * 255


def reproject_with_Q(Q, x, y, disparity):
    if disparity <= 0:
        return None

    point = np.array([x, y, disparity, 1.0], dtype=np.float64)
    point_h = Q @ point

    if abs(point_h[3]) < 1e-9:
        return None

    return point_h[:3] / point_h[3]


# -----------------------------
# Input preparation
# -----------------------------


def pad_raw_1280x720_to_calibration_frame(img, expected_shape):
    """
    Pad a raw 1280x720 image/mask/overlay to match the old calibration frame 1282x759.

    Based on the calibration screenshots, the old frame appears to include:
        left = 1, right = 1, top = 38, bottom = 1
    """
    h, w = img.shape[:2]

    if (h, w) == expected_shape:
        return img

    if (h, w) != (720, 1280):
        raise ValueError(
            f"Expected either calibration shape {expected_shape} or raw camera shape (720, 1280), "
            f"but got {(h, w)}."
        )

    padded = cv2.copyMakeBorder(
        img,
        38,  # top
        1,   # bottom
        1,   # left
        1,   # right
        cv2.BORDER_CONSTANT,
        value=0,
    )

    if padded.shape[:2] != expected_shape:
        raise ValueError(f"Padding produced {padded.shape[:2]}, expected {expected_shape}")

    return padded


def maybe_pad_to_expected(img, expected_shape, allow_padding, label):
    if img.shape[:2] == expected_shape:
        return img

    if not allow_padding:
        raise ValueError(f"{label} shape {img.shape[:2]} does not match calibration shape {expected_shape}")

    print(f"Padding {label} from {img.shape[:2]} to {expected_shape}")
    return pad_raw_1280x720_to_calibration_frame(img, expected_shape)


def read_image_required(path, flags, label):
    img = cv2.imread(str(path), flags)
    if img is None:
        raise FileNotFoundError(f"Could not read {label}: {path}")
    return img


def preprocess_to_filled_mask(mask_or_edge, min_area=500):
    """Convert a mask-like image into a single cleaned filled binary mask."""
    if mask_or_edge is None:
        raise ValueError("Mask image is None")

    if mask_or_edge.ndim == 3:
        if mask_or_edge.shape[2] == 4:
            gray = cv2.cvtColor(mask_or_edge[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(mask_or_edge, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask_or_edge.copy()

    binary = (gray > 0).astype(np.uint8) * 255

    kernel_close = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)

    if not contours:
        return filled

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
            break

    kernel_open = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel_open, iterations=1)


def edge_overlay_to_filled_mask(edge_img, expected_shape=None, min_area=1000, close_kernel=9):
    """Convert a full-frame edge/CAD overlay into a filled binary object mask."""
    if edge_img is None:
        raise ValueError("edge_overlay_to_filled_mask received None")

    if edge_img.ndim == 3:
        gray = cv2.cvtColor(edge_img, cv2.COLOR_BGR2GRAY)
    elif edge_img.ndim == 2:
        gray = edge_img.copy()
    else:
        raise ValueError(f"Unsupported edge image shape: {edge_img.shape}")

    if expected_shape is not None and gray.shape[:2] != expected_shape:
        raise ValueError(f"Edge overlay shape {gray.shape[:2]} does not match expected {expected_shape}")

    _, edge_binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
    closed = cv2.morphologyEx(edge_binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(closed, kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in edge overlay image")

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest = None
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            largest = contour
            break

    if largest is None:
        raise ValueError(
            f"No contour larger than min_area={min_area}. "
            f"Largest area was {cv2.contourArea(contours[0]):.2f}."
        )

    filled = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(filled, [largest], -1, 255, thickness=cv2.FILLED)
    return cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel, iterations=1)


def load_input_mask(mask_path, edge_overlay_path, expected_shape, allow_padding, min_area, edge_close_kernel, side_label):
    if edge_overlay_path is not None:
        edge = read_image_required(edge_overlay_path, cv2.IMREAD_UNCHANGED, f"{side_label} edge overlay")
        edge = maybe_pad_to_expected(edge, expected_shape, allow_padding, f"{side_label} edge overlay")
        return edge_overlay_to_filled_mask(
            edge,
            expected_shape=expected_shape,
            min_area=min_area,
            close_kernel=edge_close_kernel,
        )

    if mask_path is not None:
        mask_raw = read_image_required(mask_path, cv2.IMREAD_UNCHANGED, f"{side_label} mask")
        mask_raw = maybe_pad_to_expected(mask_raw, expected_shape, allow_padding, f"{side_label} mask")
        return preprocess_to_filled_mask(mask_raw, min_area=min_area)

    raise ValueError(f"Provide either --{side_label}-mask or --{side_label}-edge-overlay")


# -----------------------------
# Measurement helpers
# -----------------------------


def row_edges(mask, min_width_px=10):
    rows = []
    h, _ = mask.shape

    for y in range(h):
        xs = np.where(mask[y, :] > 0)[0]
        if len(xs) < min_width_px:
            continue

        x_left = int(xs[0])
        x_right = int(xs[-1])

        if x_right - x_left >= min_width_px:
            rows.append((y, x_left, x_right))

    return rows


def print_mask_bbox(name, mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        print(f"{name}: empty mask")
        return

    print(
        f"{name}: x={xs.min()} to {xs.max()}, "
        f"y={ys.min()} to {ys.max()}, "
        f"w={xs.max() - xs.min() + 1}, h={ys.max() - ys.min() + 1}"
    )


def estimate_generic_width_height_from_masks(
    mask_l,
    mask_r,
    Q,
    min_disparity=1.0,
    max_row_diff=3,
    width_band_start=0.45,
    width_band_end=0.55,
    height_band_start=0.45,
    height_band_end=0.55,
):
    """
    Object-agnostic visible width/height measurement from rectified masks.

    Width: row-wise silhouette left edge to right edge near vertical middle band.
    Height: column-wise top edge to bottom edge near horizontal middle band.
    """
    left_rows = row_edges(mask_l)
    right_rows = row_edges(mask_r)

    if not left_rows or not right_rows:
        return None

    # Width from middle horizontal rows.
    left_y_min = min(row[0] for row in left_rows)
    left_y_max = max(row[0] for row in left_rows)
    left_h = max(1, left_y_max - left_y_min)
    width_y_start = int(left_y_min + width_band_start * left_h)
    width_y_end = int(left_y_min + width_band_end * left_h)

    right_by_y = {y: (x_left, x_right) for y, x_left, x_right in right_rows}
    width_values = []
    width_debug_rows = []

    for y_l, xL_left, xL_right in left_rows:
        if not (width_y_start <= y_l <= width_y_end):
            continue

        best = None
        for y_r in range(y_l - max_row_diff, y_l + max_row_diff + 1):
            if y_r in right_by_y:
                best = (y_r, right_by_y[y_r])
                break

        if best is None:
            continue

        _, (xR_left, xR_right) = best
        d_left = xL_left - xR_left
        d_right = xL_right - xR_right

        if d_left <= min_disparity or d_right <= min_disparity:
            continue

        p_left = reproject_with_Q(Q, xL_left, y_l, d_left)
        p_right = reproject_with_Q(Q, xL_right, y_l, d_right)

        if p_left is None or p_right is None:
            continue

        width = np.linalg.norm(p_right - p_left)
        if np.isfinite(width):
            width_values.append(width)
            width_debug_rows.append((y_l, xL_left, xL_right, xR_left, xR_right, d_left, d_right, width))

    # Height from middle vertical columns.
    ys_l, xs_l = np.where(mask_l > 0)
    ys_r, xs_r = np.where(mask_r > 0)

    if len(xs_l) == 0 or len(xs_r) == 0:
        return None

    x_l_min = int(xs_l.min())
    x_l_max = int(xs_l.max())
    x_r_min = int(xs_r.min())
    x_r_max = int(xs_r.max())

    left_w = max(1, x_l_max - x_l_min)
    height_x_start = int(x_l_min + height_band_start * left_w)
    height_x_end = int(x_l_min + height_band_end * left_w)

    height_values = []
    height_debug_segments = []

    for xL in range(height_x_start, height_x_end + 1):
        ys_at_x = np.where(mask_l[:, xL] > 0)[0]
        if len(ys_at_x) < 2:
            continue

        yL_top = int(ys_at_x.min())
        yL_bottom = int(ys_at_x.max())

        # Approximate matching right-column by relative position in right bounding box.
        rel = (xL - x_l_min) / max(1, (x_l_max - x_l_min))
        xR = int(round(x_r_min + rel * (x_r_max - x_r_min)))

        if xR < 0 or xR >= mask_r.shape[1]:
            continue

        ys_r_at_x = np.where(mask_r[:, xR] > 0)[0]
        if len(ys_r_at_x) < 2:
            continue

        disparity = xL - xR
        if disparity <= min_disparity:
            continue

        p_top = reproject_with_Q(Q, xL, yL_top, disparity)
        p_bottom = reproject_with_Q(Q, xL, yL_bottom, disparity)

        if p_top is None or p_bottom is None:
            continue

        height = np.linalg.norm(p_bottom - p_top)
        if np.isfinite(height):
            height_values.append(height)
            height_debug_segments.append((xL, yL_top, yL_bottom, xR, disparity, height))

    if not width_values or not height_values:
        return None

    width_values = np.array(width_values, dtype=np.float64)
    height_values = np.array(height_values, dtype=np.float64)

    return {
        "raw_width": float(np.median(width_values)),
        "raw_width_mean": float(np.mean(width_values)),
        "raw_width_std": float(np.std(width_values)),
        "raw_width_min": float(np.min(width_values)),
        "raw_width_max": float(np.max(width_values)),
        "width_samples": int(len(width_values)),
        "width_debug_rows": width_debug_rows,
        "raw_height": float(np.median(height_values)),
        "raw_height_mean": float(np.mean(height_values)),
        "raw_height_std": float(np.std(height_values)),
        "raw_height_min": float(np.min(height_values)),
        "raw_height_max": float(np.max(height_values)),
        "height_samples": int(len(height_values)),
        "height_debug_segments": height_debug_segments,
    }


# -----------------------------
# Debug drawing
# -----------------------------


def draw_width_debug_rows(image, debug_rows, every=10):
    dbg = image.copy()
    if dbg.ndim == 2:
        dbg = cv2.cvtColor(dbg, cv2.COLOR_GRAY2BGR)

    for i, row in enumerate(debug_rows):
        if i % every != 0:
            continue
        y, xL_left, xL_right, _, _, _, _, _ = row
        cv2.circle(dbg, (int(xL_left), int(y)), 3, (0, 255, 0), -1)
        cv2.circle(dbg, (int(xL_right), int(y)), 3, (0, 0, 255), -1)
        cv2.line(dbg, (int(xL_left), int(y)), (int(xL_right), int(y)), (255, 0, 0), 1)

    return dbg


def draw_height_debug_segments(image, debug_segments, every=10):
    dbg = image.copy()
    if dbg.ndim == 2:
        dbg = cv2.cvtColor(dbg, cv2.COLOR_GRAY2BGR)

    for i, segment in enumerate(debug_segments):
        if i % every != 0:
            continue
        xL, y_top, y_bottom, _, _, _ = segment
        cv2.circle(dbg, (int(xL), int(y_top)), 3, (0, 255, 255), -1)
        cv2.circle(dbg, (int(xL), int(y_bottom)), 3, (255, 0, 255), -1)
        cv2.line(dbg, (int(xL), int(y_top)), (int(xL), int(y_bottom)), (0, 255, 255), 1)

    return dbg


# -----------------------------
# CLI / main
# -----------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic stereo width/height measurement from full-frame masks or edge overlays."
    )

    parser.add_argument("--calibration", required=True)
    parser.add_argument("--left-image", required=True)
    parser.add_argument("--right-image", required=True)

    parser.add_argument("--left-mask", default=None)
    parser.add_argument("--right-mask", default=None)
    parser.add_argument("--left-edge-overlay", default=None)
    parser.add_argument("--right-edge-overlay", default=None)

    parser.add_argument("--expected-width", type=float, required=True)
    parser.add_argument("--expected-height", type=float, required=True)
    parser.add_argument("--tolerance-percent", type=float, default=10.0)

    parser.add_argument("--width-scale-correction", type=float, default=0.6831)
    parser.add_argument("--height-scale-correction", type=float, default=0.8008)

    parser.add_argument("--min-area", type=int, default=500)
    parser.add_argument("--min-disparity", type=float, default=1.0)
    parser.add_argument("--max-row-diff", type=int, default=3)
    parser.add_argument("--edge-close-kernel", type=int, default=9)

    parser.add_argument("--no-padding", action="store_true", help="Disable 1280x720 to 1282x759 padding fallback.")
    parser.add_argument("--output-dir", default="debug_outputs", help="Directory for debug images and optional JSON output.")
    parser.add_argument("--json-output", default=None, help="Optional JSON file path for machine-readable results.")
    parser.add_argument("--fail-on-measurement-fail", action="store_true", help="Exit with code 2 if width or height fails tolerance.")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.left_edge_overlay is None and args.left_mask is None:
        raise ValueError("Provide either --left-mask or --left-edge-overlay")
    if args.right_edge_overlay is None and args.right_mask is None:
        raise ValueError("Provide either --right-mask or --right-edge-overlay")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    allow_padding = not args.no_padding

    calib = load_calibration(args.calibration)
    calib_w, calib_h = calib["image_size"]
    expected_shape = (calib_h, calib_w)

    left_img = read_image_required(args.left_image, cv2.IMREAD_COLOR, "left image")
    right_img = read_image_required(args.right_image, cv2.IMREAD_COLOR, "right image")

    left_img = maybe_pad_to_expected(left_img, expected_shape, allow_padding, "left image")
    right_img = maybe_pad_to_expected(right_img, expected_shape, allow_padding, "right image")

    left_filled = load_input_mask(
        args.left_mask,
        args.left_edge_overlay,
        expected_shape,
        allow_padding,
        args.min_area,
        args.edge_close_kernel,
        "left",
    )
    right_filled = load_input_mask(
        args.right_mask,
        args.right_edge_overlay,
        expected_shape,
        allow_padding,
        args.min_area,
        args.edge_close_kernel,
        "right",
    )

    if left_filled.shape[:2] != expected_shape:
        raise ValueError(f"Left filled mask shape {left_filled.shape[:2]} does not match {expected_shape}")
    if right_filled.shape[:2] != expected_shape:
        raise ValueError(f"Right filled mask shape {right_filled.shape[:2]} does not match {expected_shape}")

    print_mask_bbox("Left input mask", left_filled)
    print_mask_bbox("Right input mask", right_filled)

    rect_left_img = rectify_image(left_img, calib["mapLx"], calib["mapLy"])
    rect_right_img = rectify_image(right_img, calib["mapRx"], calib["mapRy"])
    rect_left_mask = rectify_mask(left_filled, calib["mapLx"], calib["mapLy"])
    rect_right_mask = rectify_mask(right_filled, calib["mapRx"], calib["mapRy"])

    rect_left_mask = preprocess_to_filled_mask(rect_left_mask, min_area=args.min_area)
    rect_right_mask = preprocess_to_filled_mask(rect_right_mask, min_area=args.min_area)

    print_mask_bbox("Left rectified mask", rect_left_mask)
    print_mask_bbox("Right rectified mask", rect_right_mask)

    result = estimate_generic_width_height_from_masks(
        rect_left_mask,
        rect_right_mask,
        calib["Q"],
        min_disparity=args.min_disparity,
        max_row_diff=args.max_row_diff,
    )

    cv2.imwrite(str(output_dir / "debug_input_left_mask_full.png"), left_filled)
    cv2.imwrite(str(output_dir / "debug_input_right_mask_full.png"), right_filled)
    cv2.imwrite(str(output_dir / "debug_rect_left.png"), rect_left_img)
    cv2.imwrite(str(output_dir / "debug_rect_right.png"), rect_right_img)
    cv2.imwrite(str(output_dir / "debug_rect_left_mask.png"), rect_left_mask)
    cv2.imwrite(str(output_dir / "debug_rect_right_mask.png"), rect_right_mask)

    if result is None:
        print("No valid generic width/height measurements found.")
        print("Check that masks are full-frame, rectified, non-empty, and overlapping.")
        if args.fail_on_measurement_fail:
            sys.exit(2)
        return

    width_debug = draw_width_debug_rows(rect_left_img, result["width_debug_rows"])
    height_debug = draw_height_debug_segments(rect_left_img, result["height_debug_segments"])
    cv2.imwrite(str(output_dir / "debug_width_rows.png"), width_debug)
    cv2.imwrite(str(output_dir / "debug_height_segments.png"), height_debug)

    raw_width = result["raw_width"]
    raw_height = result["raw_height"]
    corrected_width = raw_width * args.width_scale_correction
    corrected_height = raw_height * args.height_scale_correction

    width_abs_error = abs(corrected_width - args.expected_width)
    height_abs_error = abs(corrected_height - args.expected_height)
    width_percent_error = width_abs_error / args.expected_width * 100.0
    height_percent_error = height_abs_error / args.expected_height * 100.0

    width_passed = width_percent_error <= args.tolerance_percent
    height_passed = height_percent_error <= args.tolerance_percent
    overall_passed = width_passed and height_passed

    report = {
        "expected_width": args.expected_width,
        "expected_height": args.expected_height,
        "raw_width": raw_width,
        "raw_height": raw_height,
        "width_scale_correction": args.width_scale_correction,
        "height_scale_correction": args.height_scale_correction,
        "corrected_width": corrected_width,
        "corrected_height": corrected_height,
        "width_abs_error": width_abs_error,
        "height_abs_error": height_abs_error,
        "width_percent_error": width_percent_error,
        "height_percent_error": height_percent_error,
        "tolerance_percent": args.tolerance_percent,
        "width_passed": width_passed,
        "height_passed": height_passed,
        "overall_passed": overall_passed,
        "width_samples": result["width_samples"],
        "height_samples": result["height_samples"],
        "raw_width_mean": result["raw_width_mean"],
        "raw_width_std": result["raw_width_std"],
        "raw_width_min": result["raw_width_min"],
        "raw_width_max": result["raw_width_max"],
        "raw_height_mean": result["raw_height_mean"],
        "raw_height_std": result["raw_height_std"],
        "raw_height_min": result["raw_height_min"],
        "raw_height_max": result["raw_height_max"],
    }

    print()
    print("Generic object scale estimate")
    print("-----------------------------")
    print("Width measurement")
    print(f"  Expected width:           {args.expected_width:.2f}")
    print(f"  Raw measured width:       {raw_width:.2f}")
    print(f"  Width scale correction:   {args.width_scale_correction:.4f}")
    print(f"  Corrected width:          {corrected_width:.2f}")
    print(f"  Absolute error:           {width_abs_error:.2f}")
    print(f"  Percent error:            {width_percent_error:.2f}%")
    print(f"  Samples:                  {result['width_samples']}")
    print(f"  Raw mean/std:             {result['raw_width_mean']:.2f} / {result['raw_width_std']:.2f}")
    print(f"  Raw min/max:              {result['raw_width_min']:.2f} / {result['raw_width_max']:.2f}")
    print(f"  Width result:             {'PASS' if width_passed else 'FAIL'}")
    print()
    print("Height measurement")
    print(f"  Expected height:          {args.expected_height:.2f}")
    print(f"  Raw measured height:      {raw_height:.2f}")
    print(f"  Height scale correction:  {args.height_scale_correction:.4f}")
    print(f"  Corrected height:         {corrected_height:.2f}")
    print(f"  Absolute error:           {height_abs_error:.2f}")
    print(f"  Percent error:            {height_percent_error:.2f}%")
    print(f"  Samples:                  {result['height_samples']}")
    print(f"  Raw mean/std:             {result['raw_height_mean']:.2f} / {result['raw_height_std']:.2f}")
    print(f"  Raw min/max:              {result['raw_height_min']:.2f} / {result['raw_height_max']:.2f}")
    print(f"  Height result:            {'PASS' if height_passed else 'FAIL'}")
    print()
    print(f"Tolerance:                  {args.tolerance_percent:.2f}%")
    print(f"Overall result:             {'PASS' if overall_passed else 'FAIL'}")
    print(f"Debug output directory:     {output_dir}")

    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"JSON output:                {json_path}")

    if args.fail_on_measurement_fail and not overall_passed:
        sys.exit(2)


if __name__ == "__main__":
    main()
