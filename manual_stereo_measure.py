import cv2
import numpy as np
import argparse


clicked_points = []


def mouse_callback(event, x, y, flags, param):
    global clicked_points

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Clicked point {len(clicked_points)}: x={x}, y={y}")


def load_calibration(calibration_path):
    data = np.load(calibration_path)

    required_keys = [
        "mapLx", "mapLy",
        "mapRx", "mapRy",
        "Q"
    ]

    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing '{key}' in calibration file.")

    return data


def compute_disparity(rectified_left, rectified_right):
    gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoBM_create(
        numDisparities=16 * 6,
        blockSize=15
    )

    disparity = stereo.compute(gray_left, gray_right)

    # Important: OpenCV StereoBM/SGBM returns fixed-point disparity scaled by 16.
    disparity = disparity.astype(np.float32) / 16.0

    return disparity


def valid_3d_point(point):
    return (
        np.all(np.isfinite(point)) and
        not np.any(np.abs(point) > 1e6)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True, help="Path to left image")
    parser.add_argument("--right", required=True, help="Path to right image")
    parser.add_argument("--calibration", required=True, help="Path to saved .npz calibration file")
    args = parser.parse_args()

    calib = load_calibration(args.calibration)

    left = cv2.imread(args.left)
    right = cv2.imread(args.right)

    if left is None:
        raise FileNotFoundError(f"Could not load left image: {args.left}")
    if right is None:
        raise FileNotFoundError(f"Could not load right image: {args.right}")

    mapLx = calib["mapLx"]
    mapLy = calib["mapLy"]
    mapRx = calib["mapRx"]
    mapRy = calib["mapRy"]
    Q = calib["Q"]

    rect_left = cv2.remap(left, mapLx, mapLy, cv2.INTER_LINEAR)
    rect_right = cv2.remap(right, mapRx, mapRy, cv2.INTER_LINEAR)

    disparity = compute_disparity(rect_left, rect_right)

    points_3d = cv2.reprojectImageTo3D(disparity, Q)

    display = rect_left.copy()

    print("\nClick two points on the LEFT rectified image.")
    print("Use points on the object whose real-world distance you know.")
    print("Press any key after selecting two points.\n")

    cv2.namedWindow("Click two measurement points")
    cv2.setMouseCallback("Click two measurement points", mouse_callback)

    while True:
        temp = display.copy()

        for i, (x, y) in enumerate(clicked_points):
            cv2.circle(temp, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(
                temp,
                f"P{i+1}",
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        cv2.imshow("Click two measurement points", temp)

        key = cv2.waitKey(20)
        if key != -1 or len(clicked_points) >= 2:
            break

    cv2.destroyAllWindows()

    if len(clicked_points) < 2:
        print("You did not select two points.")
        return

    (x1, y1), (x2, y2) = clicked_points[:2]

    p1 = points_3d[y1, x1]
    p2 = points_3d[y2, x2]

    print("\nSelected 3D points:")
    print(f"P1: {p1}")
    print(f"P2: {p2}")

    if not valid_3d_point(p1) or not valid_3d_point(p2):
        print("\nOne or both selected points have invalid 3D coordinates.")
        print("Try selecting points on a clearer, more textured region.")
        return

    distance = np.linalg.norm(p1 - p2)

    print(f"\nMeasured distance: {distance:.2f} mm")
    print("\nCompare this against the real-world distance measured with a ruler/calipers.")


if __name__ == "__main__":
    main()