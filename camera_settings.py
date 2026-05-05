import cv2


def configure_logitech_camera(
    cap,
    width=1280,
    height=720,
    fps=30,
    autofocus=False,
    focus_value=30,
    auto_exposure=False,
    exposure_value=-6,
):
    """
    Configure a USB webcam for stereo calibration/capture.

    Notes:
    - Not every webcam/driver honors every setting.
    - Always verify settings after applying them.
    """

    # Resolution + FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Disable autofocus if supported
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)

    # Set manual focus if supported
    if not autofocus:
        cap.set(cv2.CAP_PROP_FOCUS, focus_value)

    # Disable auto exposure if supported
    # On many Logitech/OpenCV backends:
    # 0.25 = manual exposure, 0.75 = auto exposure
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if auto_exposure else 0.25)

    # Set manual exposure if supported
    if not auto_exposure:
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)

    # Warm up camera
    for _ in range(10):
        cap.read()

    # Print actual values reported by camera
    print("Camera settings:")
    print("  Width:        ", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("  Height:       ", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("  FPS:          ", cap.get(cv2.CAP_PROP_FPS))
    print("  Autofocus:    ", cap.get(cv2.CAP_PROP_AUTOFOCUS))
    print("  Focus:        ", cap.get(cv2.CAP_PROP_FOCUS))
    print("  Auto exposure:", cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
    print("  Exposure:     ", cap.get(cv2.CAP_PROP_EXPOSURE))


# Use it like this:
capL = cv2.VideoCapture(0, cv2.CAP_DSHOW)
capR = cv2.VideoCapture(1, cv2.CAP_DSHOW)

configure_logitech_camera(
    capL,
    width=1280,
    height=720,
    fps=30,
    autofocus=False,
    focus_value=30,
    auto_exposure=False,
    exposure_value=-6,
)

configure_logitech_camera(
    capR,
    width=1280,
    height=720,
    fps=30,
    autofocus=False,
    focus_value=30,
    auto_exposure=False,
    exposure_value=-6,
)