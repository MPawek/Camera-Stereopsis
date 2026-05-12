#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_SCRIPT="$SCRIPT_DIR/stl-render/main.py"
CAMERAS_JSON="$SCRIPT_DIR/configs/cameras.json"
DEBUG_CAMERAS_JSON="$SCRIPT_DIR/configs/debug_cameras.json"
BACKGROUND_BUILDER="$SCRIPT_DIR/build_backgrounds"

# --------------------------------------------------------------------
# Stereo measurement integration.
#
# These paths assume your stereo script, calibration file, and partner
# output folders live relative to this script directory.
#
# Update the four image/overlay paths inside STEREO_CMD near the bottom
# if your partner's program writes files somewhere else.
# --------------------------------------------------------------------
STEREO_SCRIPT="$SCRIPT_DIR/silhouette_stereo_measure.py"
STEREO_CALIBRATION="$SCRIPT_DIR/stereo_calibration.npz"
STEREO_OUTPUT_DIR="$SCRIPT_DIR/stereo_debug_outputs"
STEREO_JSON_OUTPUT="$STEREO_OUTPUT_DIR/result.json"

LIVE_DEBUG=0
DEBUG_MODE=0
DEBUG_CAMERA=""
DEBUG_SHAPE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --live-debug)
            LIVE_DEBUG=1
            shift
            ;;
        --debug)
            DEBUG_MODE=1
            shift
            ;;
        --camera-name)
            DEBUG_CAMERA="$2"
            shift 2
            ;;
        --shape)
            DEBUG_SHAPE="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

if [ "$DEBUG_MODE" -eq 1 ]; then
    if [ -z "$DEBUG_CAMERA" ] || [ -z "$DEBUG_SHAPE" ]; then
        echo "Usage:"
        echo "  ./run_render.sh --debug --shape <shape> --camera-name <camera_name>"
        echo
        echo "Example:"
        echo "  ./run_render.sh --debug --shape cube --camera-name posy_posx"
        exit 1
    fi

    COMPARE_CMD=(
        "$SCRIPT_DIR/compare"
        --debug
        --camera-name "$DEBUG_CAMERA"
        --shape "$DEBUG_SHAPE"
        --cameras "$DEBUG_CAMERAS_JSON"
    )

    echo
    echo "Running debug comparison:"
    printf '%q ' "${COMPARE_CMD[@]}"
    echo
    echo

    "${COMPARE_CMD[@]}"
    exit $?
fi

if [ $# -lt 1 ]; then
    echo "Usage:"
    echo "  ./run_render.sh [--live-debug] <stl_file> [layer_height] [layer_numbers...]"
    echo
    echo "Examples:"
    echo "  ./run_render.sh stl-render/stl-files/cube.stl"
    echo "  ./run_render.sh stl-render/stl-files/cube.stl 0.2 50"
    echo "  ./run_render.sh stl-render/stl-files/cube.stl 0.2 10 20 30 40 50"
    exit 1
fi

if [ ! -x "$BACKGROUND_BUILDER" ]; then
    echo "Error: background builder not found or not executable:"
    echo "  $BACKGROUND_BUILDER"
    echo
    echo "Run:"
    echo "  make build-backgrounds"
    exit 1
fi

if [ ! -f "$STEREO_SCRIPT" ]; then
    echo "Warning: stereo measurement script not found:"
    echo "  $STEREO_SCRIPT"
    echo "Stereo measurement will be skipped even if comparison passes."
fi

if [ ! -f "$STEREO_CALIBRATION" ]; then
    echo "Warning: stereo calibration file not found:"
    echo "  $STEREO_CALIBRATION"
    echo "Stereo measurement will be skipped even if comparison passes."
fi

echo
echo "Capturing and averaging background frames..."
"$BACKGROUND_BUILDER" "$CAMERAS_JSON"

echo
echo "Backgrounds built."
echo "Place the object in view now, then press ENTER to continue..."
read -r

STL_FILE="$1"
shift

CMD=(
    python3
    "$PYTHON_SCRIPT"
    "$STL_FILE"
    --cameras-json "$CAMERAS_JSON"
)

COMPARE_LAYER=""

if [ $# -ge 1 ]; then
    LAYER_HEIGHT="$1"
    shift

    CMD+=(--layer-height "$LAYER_HEIGHT")

    if [ $# -eq 1 ]; then
        COMPARE_LAYER="$1"
        CMD+=(--layer-number "$1")
    elif [ $# -gt 1 ]; then
        COMPARE_LAYER="$1"
        CMD+=(--layers "$@")
    fi
fi

echo
echo "Running render command:"
printf '%q ' "${CMD[@]}"
echo
echo

"${CMD[@]}"

COMPARE_CMD=(
    "$SCRIPT_DIR/compare"
    --live
    --cameras "$CAMERAS_JSON"
)

if [ -n "$COMPARE_LAYER" ]; then
    COMPARE_CMD+=(--layer-number "$COMPARE_LAYER")
fi

if [ "$LIVE_DEBUG" -eq 1 ]; then
    COMPARE_CMD+=(--live-debug)
fi

echo
echo "Running live comparison:"
printf '%q ' "${COMPARE_CMD[@]}"
echo
echo

COMPARE_OUTPUT_FILE="$(mktemp)"

set +e
"${COMPARE_CMD[@]}" 2>&1 | tee "$COMPARE_OUTPUT_FILE"
COMPARE_STATUS=${PIPESTATUS[0]}
set -e

if [ "$COMPARE_STATUS" -ne 0 ]; then
    echo
    echo "Live comparison command failed with exit code $COMPARE_STATUS."
    echo "Stereo measurement will not run."
    rm -f "$COMPARE_OUTPUT_FILE"
    exit "$COMPARE_STATUS"
fi

if grep -q "PASS" "$COMPARE_OUTPUT_FILE"; then
    echo
    echo "Live comparison result: PASS"
    echo "Running stereo measurement..."

    if [ ! -f "$STEREO_SCRIPT" ]; then
        echo "Error: stereo measurement script not found:"
        echo "  $STEREO_SCRIPT"
        rm -f "$COMPARE_OUTPUT_FILE"
        exit 1
    fi

    if [ ! -f "$STEREO_CALIBRATION" ]; then
        echo "Error: stereo calibration file not found:"
        echo "  $STEREO_CALIBRATION"
        rm -f "$COMPARE_OUTPUT_FILE"
        exit 1
    fi

    mkdir -p "$STEREO_OUTPUT_DIR"

    # ----------------------------------------------------------------
    # IMPORTANT:
    # Update these four file paths pipeline writes
    # the full-frame original images and fitted CAD overlays elsewhere.
    # ----------------------------------------------------------------
    STEREO_CMD=(
        python3
        "$STEREO_SCRIPT"
        --calibration "$STEREO_CALIBRATION"
        --left-image "$SCRIPT_DIR/../full_frame/cam_2/full_frame_original.png"
        --right-image "$SCRIPT_DIR/../full_frame/cam_1/full_frame_original.png"
        --left-edge-overlay "$SCRIPT_DIR/../full_frame/cam_2/full_frame_fitted_cad_overlay.png"
        --right-edge-overlay "$SCRIPT_DIR/../full_frame/cam_1/full_frame_fitted_cad_overlay.png"
        --expected-width 50
        --expected-height 50
        --output-dir "$STEREO_OUTPUT_DIR"
        --json-output "$STEREO_JSON_OUTPUT"
    )

    echo
    echo "Running stereo command:"
    printf '%q ' "${STEREO_CMD[@]}"
    echo
    echo

    "${STEREO_CMD[@]}"

    echo
    echo "Stereo measurement complete."
    echo "Stereo JSON output:"
    echo "  $STEREO_JSON_OUTPUT"
else
    echo
    echo "Live comparison did not output PASS."
    echo "Stereo measurement will not run."
fi

rm -f "$COMPARE_OUTPUT_FILE"