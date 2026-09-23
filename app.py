from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_BLOCKTIME", "0")

from camera_server import HTTP_PORT, start_camera_servers_once
from face_utils import get_face_recognizer, run_validation_report


CAMERA_URL = f"http://127.0.0.1:{HTTP_PORT}/camera"
PRELOAD_TIMEOUT_SECONDS = 300.0


def print_validation_report(report: dict) -> None:
    metrics = report.get("metrics", {})
    print()
    print("=== FACE RECOGNITION VALIDATION ===")
    print(
        f"Genuine acceptance : "
        f"{metrics.get('genuine_acceptance_rate', 0.0):.2f}%"
    )
    print(
        f"Cross-user FAR     : "
        f"{metrics.get('cross_user_false_accept_rate', 0.0):.2f}%"
    )
    print(
        f"Unknown FAR        : "
        f"{metrics.get('unknown_false_accept_rate', 0.0):.2f}%"
    )
    print()
    print("validation_results.json created.")
    print(
        "Interpretation: prioritize low false-accept rates; "
        "do not force a cosmetic 75-80% accuracy target."
    )
    print()


def preload_face_models() -> None:
    """
    Fully preload the face-recognition stack before the camera server starts.

    ArcFace + YuNet + gallery preparation already runs through the public
    FaceRecognizer preparation API. Fasnet PAD is then loaded explicitly so
    the first browser camera request does not trigger the initial PAD model
    load.
    """
    recognizer = get_face_recognizer()

    print()
    print("==============================================")
    print("       PRELOADING FACE RECOGNITION MODELS")
    print("==============================================")
    print()

    # 1. Prepare YuNet + DeepFace + ArcFace + gallery.
    print("⏳ [ML] Preparing YuNet + ArcFace + face gallery...", flush=True)
    recognizer.prepare_process_async()

    if not recognizer.wait_until_ready(timeout=PRELOAD_TIMEOUT_SECONDS):
        status = recognizer.get_runtime_status()
        stage = status.get("stage", "Unknown")
        error = status.get("error") or recognizer.get_loading_error()
        detail = f" | error={error}" if error else ""
        raise RuntimeError(
            f"Face recognition preload failed/not ready: {stage}{detail}"
        )

    print("✅ [ML] YuNet + ArcFace + gallery ready.", flush=True)

    # 2. Explicitly load Fasnet PAD before opening the camera.
    print("⏳ [PAD] Loading DeepFace Fasnet anti-spoof model...", flush=True)
    pad_model = recognizer._load_anti_spoof_model()

    if pad_model is None:
        status = recognizer.get_runtime_status()
        stage = status.get("stage", "Unknown")
        error = recognizer.get_loading_error()
        detail = f" | error={error}" if error else ""
        raise RuntimeError(
            f"Fasnet PAD failed to load: {stage}{detail}"
        )

    print("✅ [PAD] Fasnet anti-spoof model loaded and READY.", flush=True)

    recognizer.print_runtime_status()
    print()
    print("✅ All face-recognition models are loaded before camera startup.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HR Attendance Face Recognition"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help=(
            "Run validation_faces tests and save validation_results.json "
            "without starting the camera server."
        ),
    )
    parser.add_argument(
        "--validate-and-start",
        action="store_true",
        help=(
            "Run validation first, then start the normal camera server."
        ),
    )
    args = parser.parse_args()

    if args.validate or args.validate_and_start:
        try:
            report = run_validation_report()
            print_validation_report(report)
        except Exception as exc:
            print(f"❌ Validation failed: {exc}")
            if args.validate:
                raise SystemExit(1)

    if args.validate:
        return

    print()
    print("==============================================")
    print("       HR ATTENDANCE FACE RECOGNITION")
    print("==============================================")
    print()

    # IMPORTANT:
    # Preload PAD here, in the app.py terminal process, before the camera
    # server starts. Therefore the browser does not pay the first-load cost
    # when /camera is opened.
    try:
        preload_face_models()
    except Exception as exc:
        print(f"❌ Face model preload failed: {exc}")
        raise SystemExit(1)

    print("🚀 Starting camera services...")

    start_camera_servers_once()

    print()
    print("✅ HR Attendance server READY.")
    print("🌐 Camera URL:")
    print(CAMERA_URL)
    print()
    print("👉 Klik URL di terminal untuk membuka camera.")
    print()
    print("⏳ Waiting for browser connection...")
    print()

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print()
        print("🛑 HR Attendance stopped.")


if __name__ == "__main__":
    main()
