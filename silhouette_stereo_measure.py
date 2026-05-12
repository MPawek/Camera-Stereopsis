"""
silhouette_stereo_measure_cube_cylinder.py

Stereo measurement script for the CSCI 612 cube/cylinder pipeline.

Purpose
-------
This script consumes stereo camera images plus either full-frame edge overlays
or full-frame masks from the CAD/silhouette matching pipeline. It rectifies the
inputs using stereo_calibration.npz, measures object width and height, applies
empirical correction factors, and reports PASS/FAIL against expected dimensions.

Supported objects
-----------------
- cube
- cylinder

This version intentionally does not include a sphere/generic-object mode.

Expected integration inputs
---------------------------
Preferred:
    --left-edge-overlay  <left full-frame CAD/edge overlay>
    --right-edge-overlay <right full-frame CAD/edge overlay>

Optional for debug/fallback mask generation:
    --left-mask  <left full-frame filled mask>
    --right-mask <right full-frame filled mask>

Images, masks, and overlays should match the calibration frame size. If the
current camera frames are 1280x720 and the calibration expects 1282x759, the
script can apply the known padding fallback unless --no-padding is used.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Point3D = np.ndarray


# -----------------------------------------------------------------------------
# Calibration / image utilities
# -----------------------------------------------------------------------------

def load_calibration(path: str) -> Dict[str, Any]:
    calib = np.load(path)

    required = [
        "mtx_l", "dist_l", "mtx_r", "dist_r",
        "R1", "R2", "P1", "P2", "Q", "image_size",
    ]
    missing = [key for key in required if key not in calib]
    if missing:
        raise KeyError(
            f"Calibration file is missing keys: {missing}. "
            "Expected keys: mtx_l, dist_l, mtx_r, dist_r, R1, R2, P1, P2, Q, image_size."
        )

    image_size = tuple(int(v) for v in calib["image_size"])  # OpenCV: (width, height)
    print("Calibration image size:", image_size)

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


def pad_raw_1280x720_to_calibration_frame(img: np.ndarray, expected_shape: Tuple[int, int]) -> np.ndarray:
    """
    Pad raw 1280x720 camera frames to the old 1282x759 calibration frame.

    Known observed calibration-frame padding:
        left=1, right=1, top=38, bottom=1
    """
    h, w = img.shape[:2]
    if (h, w) == expected_shape:
        return img

    if (h, w) != (720, 1280):
        raise ValueError(
            f"Expected either calibration shape {expected_shape} or raw camera shape (720, 1280); got {(h, w)}. "
            "Use --no-padding to fail immediately instead of applying the fallback."
        )

    padded = cv2.copyMakeBorder(
        img, 38, 1, 1, 1, cv2.BORDER_CONSTANT, value=0,
    )

    if padded.shape[:2] != expected_shape:
        raise ValueError(f"Padding produced {padded.shape[:2]}, expected {expected_shape}")

    return padded


def maybe_pad_to_expected(
    img: np.ndarray,
    expected_shape: Tuple[int, int],
    label: str,
    no_padding: bool,
) -> np.ndarray:
    if img.shape[:2] == expected_shape:
        return img
    if no_padding:
        raise ValueError(f"{label} shape {img.shape[:2]} does not match calibration shape {expected_shape}")
    print(f"Padding {label} from {img.shape[:2]} to {expected_shape}")
    return pad_raw_1280x720_to_calibration_frame(img, expected_shape)


def read_image(path: str, flags: int, label: str) -> np.ndarray:
    img = cv2.imread(path, flags)
    if img is None:
        raise FileNotFoundError(f"Could not read {label}: {path}")
    return img


def rectify_image(img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def rectify_mask(mask: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    rectified = cv2.remap(
        mask, map_x, map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (rectified > 0).astype(np.uint8) * 255


def reproject_with_Q(Q: np.ndarray, x: int, y: int, disparity: float) -> Optional[Point3D]:
    if disparity <= 0:
        return None

    point = np.array([x, y, disparity, 1.0], dtype=np.float64)
    point_h = Q @ point

    if abs(point_h[3]) < 1e-9:
        return None

    return point_h[:3] / point_h[3]


# -----------------------------------------------------------------------------
# Mask / edge utilities
# -----------------------------------------------------------------------------

def edge_overlay_to_binary(edge_img: np.ndarray) -> np.ndarray:
    if edge_img.ndim == 3:
        gray = cv2.cvtColor(edge_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = edge_img.copy()
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    return binary


def preprocess_to_filled_mask(mask_or_edge: np.ndarray, min_area: int = 500) -> np.ndarray:
    """Convert a full-frame mask/edge image to a single filled binary object mask."""
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


def edge_overlay_to_filled_mask(edge_img: np.ndarray, min_area: int = 500, close_kernel: int = 9) -> np.ndarray:
    binary = edge_overlay_to_binary(edge_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(closed, kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    if not contours:
        raise ValueError("No contours found in edge overlay image")

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
            return cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel, iterations=1)

    largest_area = cv2.contourArea(contours[0])
    raise ValueError(f"No contour larger than min_area={min_area}; largest area was {largest_area:.2f}")


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def print_mask_bbox(name: str, mask: np.ndarray) -> None:
    bbox = mask_bbox(mask)
    if bbox is None:
        print(f"{name}: empty mask")
        return
    x0, y0, x1, y1 = bbox
    print(f"{name}: x={x0} to {x1}, y={y0} to {y1}, w={x1 - x0 + 1}, h={y1 - y0 + 1}")


def row_edges(mask: np.ndarray, min_width_px: int = 10) -> List[Tuple[int, int, int]]:
    rows: List[Tuple[int, int, int]] = []
    for y in range(mask.shape[0]):
        xs = np.where(mask[y, :] > 0)[0]
        if len(xs) < min_width_px:
            continue
        x_left = int(xs[0])
        x_right = int(xs[-1])
        if x_right - x_left >= min_width_px:
            rows.append((y, x_left, x_right))
    return rows


def find_vertical_edge_columns(edge_binary: np.ndarray, min_column_pixels: int = 25, cluster_gap: int = 6) -> List[int]:
    if edge_binary.ndim != 2:
        raise ValueError("find_vertical_edge_columns expects a single-channel binary image")

    col_counts = np.count_nonzero(edge_binary > 0, axis=0)
    candidate_xs = np.where(col_counts >= min_column_pixels)[0]
    if len(candidate_xs) == 0:
        return []

    clusters: List[List[int]] = []
    current = [int(candidate_xs[0])]
    for x_raw in candidate_xs[1:]:
        x = int(x_raw)
        if x - current[-1] <= cluster_gap:
            current.append(x)
        else:
            clusters.append(current)
            current = [x]
    clusters.append(current)

    edge_columns: List[int] = []
    for cluster_list in clusters:
        cluster = np.array(cluster_list)
        weights = col_counts[cluster]
        if np.sum(weights) == 0:
            edge_columns.append(int(np.median(cluster)))
        else:
            edge_columns.append(int(round(np.average(cluster, weights=weights))))

    return sorted(edge_columns)


def choose_cube_front_face_columns(edge_columns: Sequence[int], face_side: str) -> Tuple[int, int]:
    if len(edge_columns) < 2:
        raise ValueError(f"Need at least 2 vertical edge columns, found {len(edge_columns)}: {edge_columns}")

    cols = sorted(int(x) for x in edge_columns)
    if face_side == "side-left":
        selected = cols[-2:]
    elif face_side == "side-right":
        selected = cols[:2]
    else:
        raise ValueError(f"Unknown face_side: {face_side}")
    return selected[0], selected[1]



# -----------------------------------------------------------------------------
# Measurement utilities
# -----------------------------------------------------------------------------

def sample_width_from_columns(
    edge_l: np.ndarray,
    edge_r: np.ndarray,
    Q: np.ndarray,
    xL_left: int,
    xL_right: int,
    xR_left: int,
    xR_right: int,
    min_disparity: float,
) -> Tuple[Optional[Dict[str, float]], List[Tuple[int, int, int, int, int, float, float, float]]]:
    """Measure 3D distance between two selected vertical boundary columns."""
    ys_l_left = np.where(edge_l[:, max(0, xL_left - 2):xL_left + 3] > 0)[0]
    ys_l_right = np.where(edge_l[:, max(0, xL_right - 2):xL_right + 3] > 0)[0]
    ys_r_left = np.where(edge_r[:, max(0, xR_left - 2):xR_left + 3] > 0)[0]
    ys_r_right = np.where(edge_r[:, max(0, xR_right - 2):xR_right + 3] > 0)[0]

    if any(len(ys) == 0 for ys in [ys_l_left, ys_l_right, ys_r_left, ys_r_right]):
        raise ValueError("Could not find enough edge pixels near selected width columns")

    y_min = max(int(np.min(ys_l_left)), int(np.min(ys_l_right)), int(np.min(ys_r_left)), int(np.min(ys_r_right)))
    y_max = min(int(np.max(ys_l_left)), int(np.max(ys_l_right)), int(np.max(ys_r_left)), int(np.max(ys_r_right)))
    if y_max <= y_min + 20:
        raise ValueError(f"No useful shared y-overlap for selected width columns: {y_min} to {y_max}")

    sample_ys = np.linspace(y_min + 10, y_max - 10, 25).astype(int)
    widths: List[float] = []
    debug_rows: List[Tuple[int, int, int, int, int, float, float, float]] = []

    d_left = xL_left - xR_left
    d_right = xL_right - xR_right
    if d_left <= min_disparity or d_right <= min_disparity:
        raise ValueError(f"Invalid selected column disparities: left={d_left}, right={d_right}")

    for y in sample_ys:
        p_left = reproject_with_Q(Q, xL_left, int(y), d_left)
        p_right = reproject_with_Q(Q, xL_right, int(y), d_right)
        if p_left is None or p_right is None:
            continue
        width = float(np.linalg.norm(p_right - p_left))
        if np.isfinite(width):
            widths.append(width)
            debug_rows.append((int(y), xL_left, xL_right, xR_left, xR_right, float(d_left), float(d_right), width))

    if not widths:
        return None, debug_rows

    arr = np.array(widths, dtype=np.float64)
    return {
        "raw": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "samples": int(len(arr)),
    }, debug_rows


def height_for_edge_pair(
    edge_l: np.ndarray,
    Q: np.ndarray,
    xL: int,
    xR: int,
    min_disparity: float,
) -> Optional[Dict[str, float]]:
    disparity = xL - xR
    if disparity <= min_disparity:
        return None

    x0 = max(0, xL - 2)
    x1 = min(edge_l.shape[1], xL + 3)
    ys = np.where(edge_l[:, x0:x1] > 0)[0]
    if len(ys) < 2:
        return None

    y_top = int(np.percentile(ys, 5))
    y_bottom = int(np.percentile(ys, 95))
    if y_bottom <= y_top:
        return None

    p_top = reproject_with_Q(Q, xL, y_top, disparity)
    p_bottom = reproject_with_Q(Q, xL, y_bottom, disparity)
    if p_top is None or p_bottom is None:
        return None

    height = float(np.linalg.norm(p_bottom - p_top))
    if not np.isfinite(height):
        return None

    return {
        "raw": height,
        "xL": int(xL),
        "xR": int(xR),
        "disparity": float(disparity),
        "y_top": int(y_top),
        "y_bottom": int(y_bottom),
    }


def estimate_edge_based_dimensions(
    rect_left_edge: np.ndarray,
    rect_right_edge: np.ndarray,
    Q: np.ndarray,
    left_face_side: str,
    right_face_side: str,
    min_disparity: float,
    manual_left_columns: Optional[Sequence[int]] = None,
    manual_right_columns: Optional[Sequence[int]] = None,
) -> Tuple[Dict[str, Any], List[Tuple[int, int, int, int, int, float, float, float]]]:
    """Measure width/height from rectified edge overlays for cube or cylinder."""
    cols_l = find_vertical_edge_columns(rect_left_edge)
    cols_r = find_vertical_edge_columns(rect_right_edge)

    print("Left vertical edge columns:", cols_l)
    print("Right vertical edge columns:", cols_r)

    if manual_left_columns is not None:
        xL_a, xL_b = int(manual_left_columns[0]), int(manual_left_columns[1])
        print("Using MANUAL left columns:", xL_a, xL_b)
    else:
        xL_a, xL_b = choose_cube_front_face_columns(cols_l, left_face_side)
        print("Using AUTO left measurement columns:", xL_a, xL_b)

    if manual_right_columns is not None:
        xR_a, xR_b = int(manual_right_columns[0]), int(manual_right_columns[1])
        print("Using MANUAL right columns:", xR_a, xR_b)
    else:
        xR_a, xR_b = choose_cube_front_face_columns(cols_r, right_face_side)
        print("Using AUTO right measurement columns:", xR_a, xR_b)


    print("Selected disparities:")
    print("  left/first edge: ", xL_a - xR_a)
    print("  right/second edge:", xL_b - xR_b)

    width_stats, debug_rows = sample_width_from_columns(
        rect_left_edge, rect_right_edge, Q, xL_a, xL_b, xR_a, xR_b, min_disparity
    )
    if width_stats is None:
        raise ValueError("No valid width measurements found from selected columns")

    h_a = height_for_edge_pair(rect_left_edge, Q, xL_a, xR_a, min_disparity)
    h_b = height_for_edge_pair(rect_left_edge, Q, xL_b, xR_b, min_disparity)
    heights = [h for h in [h_a, h_b] if h is not None]
    if not heights:
        raise ValueError("No valid height measurements found from selected columns")

    height_values = np.array([h["raw"] for h in heights], dtype=np.float64)
    height_stats = {
        "raw": float(np.median(height_values)),
        "mean": float(np.mean(height_values)),
        "std": float(np.std(height_values)),
        "min": float(np.min(height_values)),
        "max": float(np.max(height_values)),
        "samples": int(len(height_values)),
        "edge_details": heights,
    }

    result = {
        "left_columns": [int(xL_a), int(xL_b)],
        "right_columns": [int(xR_a), int(xR_b)],
        "width": width_stats,
        "height": height_stats,
    }
    return result, debug_rows


def draw_width_debug_rows(image: np.ndarray, debug_rows: Sequence[Tuple[int, int, int, int, int, float, float, float]], every: int = 5) -> np.ndarray:
    dbg = image.copy()
    if dbg.ndim == 2:
        dbg = cv2.cvtColor(dbg, cv2.COLOR_GRAY2BGR)

    for i, row in enumerate(debug_rows):
        if i % every != 0:
            continue
        y, xL_left, xL_right, _, _, _, _, _ = row
        cv2.circle(dbg, (int(xL_left), int(y)), 4, (0, 255, 0), -1)
        cv2.circle(dbg, (int(xL_right), int(y)), 4, (0, 0, 255), -1)
        cv2.line(dbg, (int(xL_left), int(y)), (int(xL_right), int(y)), (255, 0, 0), 1)

    return dbg


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stereo size measurement for cube/cylinder using rectified edge overlays."
    )

    parser.add_argument("--calibration", required=True)
    parser.add_argument("--left-image", required=True)
    parser.add_argument("--right-image", required=True)

    parser.add_argument("--left-edge-overlay", required=True, help="Full-frame left CAD/edge overlay image.")
    parser.add_argument("--right-edge-overlay", required=True, help="Full-frame right CAD/edge overlay image.")

    parser.add_argument("--left-mask", default=None, help="Optional full-frame left filled mask for debug output.")
    parser.add_argument("--right-mask", default=None, help="Optional full-frame right filled mask for debug output.")

    parser.add_argument("--expected-width", type=float, required=True)
    parser.add_argument("--expected-height", type=float, required=True)
    parser.add_argument("--tolerance-percent", type=float, default=10.0)

    parser.add_argument("--width-scale-correction", type=float, default=0.6831)
    parser.add_argument("--height-scale-correction", type=float, default=0.8008)

    parser.add_argument("--left-face-side", choices=["side-left", "side-right"], default="side-left")
    parser.add_argument("--right-face-side", choices=["side-left", "side-right"], default="side-right")

    parser.add_argument("--manual-left-columns", nargs=2, type=int, default=None, metavar=("X_LEFT", "X_RIGHT"))
    parser.add_argument("--manual-right-columns", nargs=2, type=int, default=None, metavar=("X_LEFT", "X_RIGHT"))

    parser.add_argument("--min-area", type=int, default=500)
    parser.add_argument("--min-disparity", type=float, default=1.0)
    parser.add_argument("--edge-close-kernel", type=int, default=9)
    parser.add_argument("--no-padding", action="store_true", help="Disable 1280x720 to 1282x759 padding fallback.")
    parser.add_argument("--output-dir", default="debug_outputs")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--fail-on-measurement-fail", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    calib = load_calibration(args.calibration)
    calib_w, calib_h = calib["image_size"]
    expected_shape = (calib_h, calib_w)

    left_img = read_image(args.left_image, cv2.IMREAD_COLOR, "left image")
    right_img = read_image(args.right_image, cv2.IMREAD_COLOR, "right image")
    left_edge = read_image(args.left_edge_overlay, cv2.IMREAD_UNCHANGED, "left edge overlay")
    right_edge = read_image(args.right_edge_overlay, cv2.IMREAD_UNCHANGED, "right edge overlay")

    left_img = maybe_pad_to_expected(left_img, expected_shape, "left image", args.no_padding)
    right_img = maybe_pad_to_expected(right_img, expected_shape, "right image", args.no_padding)
    left_edge = maybe_pad_to_expected(left_edge, expected_shape, "left edge overlay", args.no_padding)
    right_edge = maybe_pad_to_expected(right_edge, expected_shape, "right edge overlay", args.no_padding)

    left_edge_binary = edge_overlay_to_binary(left_edge)
    right_edge_binary = edge_overlay_to_binary(right_edge)

    if args.left_mask is not None:
        left_filled = read_image(args.left_mask, cv2.IMREAD_UNCHANGED, "left mask")
        left_filled = maybe_pad_to_expected(left_filled, expected_shape, "left mask", args.no_padding)
        left_filled = preprocess_to_filled_mask(left_filled, min_area=args.min_area)
    else:
        left_filled = edge_overlay_to_filled_mask(left_edge, min_area=args.min_area, close_kernel=args.edge_close_kernel)

    if args.right_mask is not None:
        right_filled = read_image(args.right_mask, cv2.IMREAD_UNCHANGED, "right mask")
        right_filled = maybe_pad_to_expected(right_filled, expected_shape, "right mask", args.no_padding)
        right_filled = preprocess_to_filled_mask(right_filled, min_area=args.min_area)
    else:
        right_filled = edge_overlay_to_filled_mask(right_edge, min_area=args.min_area, close_kernel=args.edge_close_kernel)

    print_mask_bbox("Left input filled mask", left_filled)
    print_mask_bbox("Right input filled mask", right_filled)

    rect_left_img = rectify_image(left_img, calib["mapLx"], calib["mapLy"])
    rect_right_img = rectify_image(right_img, calib["mapRx"], calib["mapRy"])
    rect_left_mask = rectify_mask(left_filled, calib["mapLx"], calib["mapLy"])
    rect_right_mask = rectify_mask(right_filled, calib["mapRx"], calib["mapRy"])
    rect_left_edge = rectify_mask(left_edge_binary, calib["mapLx"], calib["mapLy"])
    rect_right_edge = rectify_mask(right_edge_binary, calib["mapRx"], calib["mapRy"])

    print_mask_bbox("Left rectified mask", rect_left_mask)
    print_mask_bbox("Right rectified mask", rect_right_mask)

    measurement, debug_rows = estimate_edge_based_dimensions(
        rect_left_edge=rect_left_edge,
        rect_right_edge=rect_right_edge,
        Q=calib["Q"],
        left_face_side=args.left_face_side,
        right_face_side=args.right_face_side,
        min_disparity=args.min_disparity,
        manual_left_columns=args.manual_left_columns,
        manual_right_columns=args.manual_right_columns,
    )

    raw_width = float(measurement["width"]["raw"])
    raw_height = float(measurement["height"]["raw"])
    corrected_width = raw_width * args.width_scale_correction
    corrected_height = raw_height * args.height_scale_correction

    width_abs_error = abs(corrected_width - args.expected_width)
    height_abs_error = abs(corrected_height - args.expected_height)
    width_percent_error = width_abs_error / args.expected_width * 100.0
    height_percent_error = height_abs_error / args.expected_height * 100.0
    width_passed = width_percent_error <= args.tolerance_percent
    height_passed = height_percent_error <= args.tolerance_percent
    overall_passed = width_passed and height_passed

    cv2.imwrite(os.path.join(args.output_dir, "debug_rect_left.png"), rect_left_img)
    cv2.imwrite(os.path.join(args.output_dir, "debug_rect_right.png"), rect_right_img)
    cv2.imwrite(os.path.join(args.output_dir, "debug_rect_left_mask.png"), rect_left_mask)
    cv2.imwrite(os.path.join(args.output_dir, "debug_rect_right_mask.png"), rect_right_mask)
    cv2.imwrite(os.path.join(args.output_dir, "debug_rect_left_edge.png"), rect_left_edge)
    cv2.imwrite(os.path.join(args.output_dir, "debug_rect_right_edge.png"), rect_right_edge)
    cv2.imwrite(os.path.join(args.output_dir, "debug_width_rows.png"), draw_width_debug_rows(rect_left_img, debug_rows))

    output = {
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
        "left_columns": measurement["left_columns"],
        "right_columns": measurement["right_columns"],
        "width_stats": measurement["width"],
        "height_stats": measurement["height"],
    }

    print()
    print("Stereo measurement")
    print("--------------------------------")
    print("Width measurement")
    print(f"  Expected width:           {args.expected_width:.2f}")
    print(f"  Raw measured width:       {raw_width:.2f}")
    print(f"  Width scale correction:   {args.width_scale_correction:.4f}")
    print(f"  Corrected width:          {corrected_width:.2f}")
    print(f"  Absolute error:           {width_abs_error:.2f}")
    print(f"  Percent error:            {width_percent_error:.2f}%")
    print(f"  Samples:                  {measurement['width']['samples']}")
    print(f"  Raw mean/std:             {measurement['width']['mean']:.2f} / {measurement['width']['std']:.2f}")
    print(f"  Raw min/max:              {measurement['width']['min']:.2f} / {measurement['width']['max']:.2f}")
    print(f"  Width result:             {'PASS' if width_passed else 'FAIL'}")
    print()
    print("Height measurement")
    print(f"  Expected height:          {args.expected_height:.2f}")
    print(f"  Raw measured height:      {raw_height:.2f}")
    print(f"  Height scale correction:  {args.height_scale_correction:.4f}")
    print(f"  Corrected height:         {corrected_height:.2f}")
    print(f"  Absolute error:           {height_abs_error:.2f}")
    print(f"  Percent error:            {height_percent_error:.2f}%")
    print(f"  Samples:                  {measurement['height']['samples']}")
    print(f"  Raw mean/std:             {measurement['height']['mean']:.2f} / {measurement['height']['std']:.2f}")
    print(f"  Raw min/max:              {measurement['height']['min']:.2f} / {measurement['height']['max']:.2f}")
    print(f"  Height result:            {'PASS' if height_passed else 'FAIL'}")
    print()
    print(f"Tolerance:                  {args.tolerance_percent:.2f}%")
    print(f"Overall result:             {'PASS' if overall_passed else 'FAIL'}")
    print(f"Debug output directory:     {args.output_dir}")

    if args.json_output:
        json_dir = os.path.dirname(args.json_output)
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"JSON output:                {args.json_output}")

    if args.fail_on_measurement_fail and not overall_passed:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
