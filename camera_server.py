
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from websockets.asyncio.server import serve

from face_utils import (
    get_face_recognizer,
    start_process_preload,
)


# ==========================================================
# CONFIG
# ==========================================================

HOST = os.getenv(
    "HR_CAMERA_BIND_HOST",
    "127.0.0.1",
)

HTTP_PORT = int(
    os.getenv(
        "HR_CAMERA_HTTP_PORT",
        "8766",
    )
)

WS_PORT = int(
    os.getenv(
        "HR_CAMERA_WS_PORT",
        "8765",
    )
)


# ==========================================================
# FRAME / RECOGNITION CONFIG
# ==========================================================

# YuNet target processing rate.
# Browser sends frames continuously.
FRAME_INTERVAL = 1.0 / 25.0

# SFace does not need to run on every frame.
RECOGNITION_INTERVAL = 0.55

FACE_TOLERANCE = 0.60


# ==========================================================
# PATH
# ==========================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

CAMERA_DIR = (
    BASE_DIR /
    "camera_web"
)

RESULT_FILE = (
    BASE_DIR /
    "face_recognition_result.json"
)


# ==========================================================
# RECOGNITION STATE
# ==========================================================

_last_name = "Unknown"

_last_similarity = 0.0

_last_recognition_time = 0.0


# ==========================================================
# RESULT STATE
# ==========================================================

_result_saved = False

_result_lock = threading.Lock()


# ==========================================================
# SERVER STATE
# ==========================================================

_servers_started = False

_servers_lock = threading.Lock()


# ==========================================================
# SFACE WORKER
# ==========================================================

_recognition_worker_started = False

_recognition_event = threading.Event()

_recognition_lock = threading.Lock()


_pending_frame = None

_pending_detection = None

_pending_generation = 0

_recognition_generation = 0


# ==========================================================
# HTTP FILE SERVER
# ==========================================================

class CameraHandler(BaseHTTPRequestHandler):

    def _send_file(
        self,
        filename: str,
        content_type: str,
    ) -> None:

        path = (
            CAMERA_DIR /
            filename
        )


        if (
            not path.exists()
            or not path.is_file()
        ):

            print(
                f"❌ [HTTP] "
                f"File not found: {path}",
                flush=True,
            )

            self.send_error(
                404,
                "Camera file not found.",
            )

            return


        try:

            data = path.read_bytes()

        except Exception as exc:

            print(
                f"❌ [HTTP] "
                f"Read error: {exc}",
                flush=True,
            )

            self.send_error(
                500,
                "Could not read file.",
            )

            return


        self.send_response(200)

        self.send_header(
            "Content-Type",
            content_type,
        )

        # Never cache JS/CSS while developing.
        self.send_header(
            "Cache-Control",
            "no-store, no-cache, "
            "must-revalidate, max-age=0",
        )

        self.send_header(
            "Pragma",
            "no-cache",
        )

        self.send_header(
            "Expires",
            "0",
        )

        self.send_header(
            "Content-Length",
            str(len(data)),
        )

        self.end_headers()

        self.wfile.write(data)


    def do_GET(
        self,
    ) -> None:

        path = self.path.split(
            "?",
            1,
        )[0]


        # Main camera page.
        # Public URL: http://127.0.0.1:8766/camera
        if path == "/camera":

            self._send_file(
                "camera_web.html",
                "text/html; charset=utf-8",
            )

            return

        # Redirect root URL to /camera.
        if path == "/":

            self.send_response(302)
            self.send_header(
                "Location",
                "/camera",
            )
            self.send_header(
                "Cache-Control",
                "no-store, no-cache, must-revalidate, max-age=0",
            )
            self.end_headers()

            return


        if path == "/camera_web.js":

            self._send_file(
                "camera_web.js",
                "application/javascript; charset=utf-8",
            )

            return


        if path == "/camera_web.css":

            self._send_file(
                "camera_web.css",
                "text/css; charset=utf-8",
            )

            return


        self.send_error(404)


    def log_message(
        self,
        fmt,
        *args,
    ):

        return


# ==========================================================
# SAVE RESULT
# ==========================================================

def save_result(
    name: str,
    similarity: float,
) -> None:

    global _result_saved


    if (
        not name
        or name == "Unknown"
    ):

        return


    with _result_lock:

        if _result_saved:
            return


        payload = {

            "name":
                name,

            "similarity":
                float(
                    similarity
                ),

            "timestamp":
                time.time(),
        }


        try:

            RESULT_FILE.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )


            _result_saved = True


            print(
                f"✅ [FACE] "
                f"Result saved: {name}",
                flush=True,
            )


        except Exception as exc:

            print(
                f"⚠️ [FACE] "
                f"Save result error: {exc}",
                flush=True,
            )


# ==========================================================
# SFACE BACKGROUND WORKER
# ==========================================================

def recognition_worker() -> None:

    global _pending_frame
    global _pending_detection
    global _pending_generation

    global _recognition_generation

    global _last_name
    global _last_similarity


    print(
        "✅ [SFACE] "
        "Background worker READY.",
        flush=True,
    )


    while True:

        _recognition_event.wait()

        _recognition_event.clear()


        while True:

            with _recognition_lock:

                frame = (
                    _pending_frame
                )

                detection = (
                    _pending_detection
                )

                generation = (
                    _pending_generation
                )


                _pending_frame = None

                _pending_detection = None


            if (
                frame is None
                or detection is None
            ):

                break


            try:

                recognizer = (
                    get_face_recognizer()
                )


                if recognizer is None:
                    continue


                (
                    name,
                    similarity,
                    top_mean,
                ) = (
                    recognizer
                    .recognize_detection(
                        frame,
                        detection,
                        tolerance=FACE_TOLERANCE,
                    )
                )


                cleaned_name = (

                    "Unknown"

                    if (
                        not name
                        or
                        name == "Preparing..."
                    )

                    else (
                        str(
                            name
                        ).strip()
                        or
                        "Unknown"
                    )
                )


                with _recognition_lock:

                    # Never accept an old SFace result.
                    if (
                        generation
                        !=
                        _recognition_generation
                    ):

                        continue


                    _last_name = (
                        cleaned_name
                    )

                    _last_similarity = (
                        float(
                            similarity
                        )
                    )


                print(
                    f"🔎 [SFACE] "
                    f"{_last_name} "
                    f"similarity="
                    f"{_last_similarity:.3f}",
                    flush=True,
                )


                save_result(
                    _last_name,
                    _last_similarity,
                )


            except Exception as exc:

                print(
                    f"❌ [SFACE] "
                    f"Recognition error: {exc}",
                    flush=True,
                )


# ==========================================================
# QUEUE LATEST SFACE REQUEST
# ==========================================================

def queue_recognition(
    frame,
    detection,
) -> None:

    global _pending_frame
    global _pending_detection
    global _pending_generation
    global _recognition_generation
    global _last_recognition_time


    now = time.monotonic()


    if (
        now -
        _last_recognition_time
        <
        RECOGNITION_INTERVAL
    ):

        return


    _last_recognition_time = now


    with _recognition_lock:

        _recognition_generation += 1


        _pending_generation = (
            _recognition_generation
        )


        # Only newest recognition frame survives.
        _pending_frame = (
            frame.copy()
        )


        _pending_detection = (
            dict(
                detection
            )
        )


    _recognition_event.set()


# ==========================================================
# PROCESS ONE FRAME
# ==========================================================

def process_frame(
    message: bytes,
) -> dict:

    # ------------------------------------------------------
    # Decode JPEG
    # ------------------------------------------------------

    try:

        encoded = np.frombuffer(
            message,
            dtype=np.uint8,
        )


        frame = cv2.imdecode(
            encoded,
            cv2.IMREAD_COLOR,
        )


    except Exception as exc:

        print(
            f"❌ [FRAME] "
            f"Decode error: {exc}",
            flush=True,
        )


        return {
            "name": _last_name,
            "similarity": _last_similarity,
            "box": None,
            "raw_box": None,
            "width": 0,
            "height": 0,
            "status": "invalid_frame",
        }


    if frame is None:

        return {
            "name": _last_name,
            "similarity": _last_similarity,
            "box": None,
            "raw_box": None,
            "width": 0,
            "height": 0,
            "status": "invalid_frame",
        }


    frame_height = int(
        frame.shape[0]
    )

    frame_width = int(
        frame.shape[1]
    )


    # ------------------------------------------------------
    # Get recognizer
    # ------------------------------------------------------

    try:

        recognizer = (
            get_face_recognizer()
        )

    except Exception as exc:

        print(
            f"❌ [ML] "
            f"Recognizer error: {exc}",
            flush=True,
        )


        return {
            "name": _last_name,
            "similarity": _last_similarity,
            "box": None,
            "raw_box": None,
            "width": frame_width,
            "height": frame_height,
            "status":
                "recognizer_unavailable",
        }


    if recognizer is None:

        return {
            "name": _last_name,
            "similarity": _last_similarity,
            "box": None,
            "raw_box": None,
            "width": frame_width,
            "height": frame_height,
            "status":
                "recognizer_unavailable",
        }


    # ------------------------------------------------------
    # Read states
    # ------------------------------------------------------

    detector_ready = (
        recognizer
        .is_detector_ready()
    )

    model_ready = (
        recognizer
        .is_model_ready()
    )

    embeddings_ready = (
        recognizer
        .is_embeddings_ready()
    )


    # ------------------------------------------------------
    # Detector not ready
    # ------------------------------------------------------

    if not detector_ready:

        return {
            "name": _last_name,
            "similarity":
                _last_similarity,
            "box": None,
            "raw_box": None,
            "width":
                frame_width,
            "height":
                frame_height,
            "status":
                "loading_yunet",
            "detector_ready":
                False,
            "model_ready":
                model_ready,
            "embeddings_ready":
                embeddings_ready,
            "recognition_ready":
                False,
        }


    # ------------------------------------------------------
    # YUNET
    # ------------------------------------------------------

    try:

        detection, stable_box = (
            recognizer
            .track_face(
                frame
            )
        )


    except Exception as exc:

        print(
            f"❌ [YUNET] "
            f"Detection error: {exc}",
            flush=True,
        )


        return {
            "name": _last_name,
            "similarity":
                _last_similarity,
            "box": None,
            "raw_box": None,
            "width":
                frame_width,
            "height":
                frame_height,
            "status":
                "yunet_error",
            "detector_ready":
                True,
            "model_ready":
                model_ready,
            "embeddings_ready":
                embeddings_ready,
        }


    # ------------------------------------------------------
    # No face in CURRENT frame
    # ------------------------------------------------------

    if detection is None:

        return {
            "name": _last_name,
            "similarity":
                _last_similarity,
            "box": None,
            "raw_box": None,
            "stable_box": None,
            "width":
                frame_width,
            "height":
                frame_height,
            "status":
                "searching",
            "detector_ready":
                True,
            "model_ready":
                model_ready,
            "embeddings_ready":
                embeddings_ready,
            "recognition_ready":
                (
                    model_ready
                    and
                    embeddings_ready
                ),
        }


    # ------------------------------------------------------
    # RAW BOX
    # ------------------------------------------------------

    raw_box = None


    try:

        raw_box = {

            "x":
                int(
                    detection["x"]
                ),

            "y":
                int(
                    detection["y"]
                ),

            "w":
                int(
                    detection["w"]
                ),

            "h":
                int(
                    detection["h"]
                ),
        }


    except Exception as exc:

        print(
            f"⚠️ [YUNET] "
            f"Invalid box: {exc}",
            flush=True,
        )


    # ------------------------------------------------------
    # Stable box only for diagnostics.
    # Main UI uses raw box.
    # ------------------------------------------------------

    stable_box_json = None


    if stable_box is not None:

        x, y, w, h = stable_box


        stable_box_json = {

            "x":
                int(x),

            "y":
                int(y),

            "w":
                int(w),

            "h":
                int(h),
        }


    # ------------------------------------------------------
    # Latest box
    # ------------------------------------------------------

    display_box = (
        raw_box
        if raw_box is not None
        else stable_box_json
    )


    # ------------------------------------------------------
    # Queue SFace
    #
    # Only if a CURRENT YuNet face exists.
    # ------------------------------------------------------

    if (
        model_ready
        and embeddings_ready
    ):

        queue_recognition(
            frame,
            detection,
        )


        status_value = (
            "face_detected"
        )

    else:

        status_value = (
            "face_detected_preparing"
        )


    # ------------------------------------------------------
    # Response
    # ------------------------------------------------------

    return {

        "name":
            _last_name,

        "similarity":
            _last_similarity,

        "box":
            display_box,

        "raw_box":
            raw_box,

        "stable_box":
            stable_box_json,

        "width":
            frame_width,

        "height":
            frame_height,

        "status":
            status_value,

        "detector_ready":
            True,

        "model_ready":
            model_ready,

        "embeddings_ready":
            embeddings_ready,

        "recognition_ready":
            (
                model_ready
                and
                embeddings_ready
            ),
    }


# ==========================================================
# WEBSOCKET
#
# IMPORTANT:
#
# RECEIVER and PROCESSOR are separated.
#
# Browser can keep sending frames while YuNet
# is processing another frame.
# ==========================================================

async def ws_handler(
    websocket,
) -> None:

    print(
        "✅ [WS] "
        "Browser camera connected.",
        flush=True,
    )


    frame_queue = asyncio.Queue(
        maxsize=1
    )


    async def receiver():

        try:

            async for message in websocket:

                if not isinstance(
                    message,
                    bytes,
                ):

                    continue


                # ------------------------------------------
                # Latest-frame queue.
                #
                # If queue already has an older frame,
                # discard the older one.
                # ------------------------------------------

                if frame_queue.full():

                    try:

                        frame_queue.get_nowait()

                    except asyncio.QueueEmpty:

                        pass


                try:

                    frame_queue.put_nowait(
                        message
                    )

                except asyncio.QueueFull:

                    pass


        except Exception as exc:

            print(
                f"ℹ️ [WS] "
                f"Receiver stopped: {exc}",
                flush=True,
            )


    async def processor():

        while True:

            message = (
                await frame_queue.get()
            )


            try:

                result = (
                    await asyncio.to_thread(
                        process_frame,
                        message,
                    )
                )


                await websocket.send(
                    json.dumps(
                        result
                    )
                )


            except Exception as exc:

                print(
                    f"❌ [WS] "
                    f"Processor error: {exc}",
                    flush=True,
                )


                break


    receiver_task = asyncio.create_task(
        receiver()
    )


    processor_task = asyncio.create_task(
        processor()
    )


    done, pending = (
        await asyncio.wait(
            [
                receiver_task,
                processor_task,
            ],
            return_when=
                asyncio.FIRST_COMPLETED,
        )
    )


    for task in pending:

        task.cancel()


    print(
        "ℹ️ [WS] "
        "Browser camera disconnected.",
        flush=True,
    )


# ==========================================================
# HTTP
# ==========================================================

def run_http_server():

    server = (
        ThreadingHTTPServer(
            (
                HOST,
                HTTP_PORT,
            ),
            CameraHandler,
        )
    )


    print(
        "✅ [HTTP] "
        f"http://127.0.0.1:{HTTP_PORT}"
        "/camera",
        flush=True,
    )


    server.serve_forever()


# ==========================================================
# WEBSOCKET SERVER
# ==========================================================

async def run_ws_server():

    print(
        "✅ [WS] "
        f"ws://127.0.0.1:{WS_PORT}",
        flush=True,
    )


    async with serve(
        ws_handler,
        HOST,
        WS_PORT,
        max_size=
            2 * 1024 * 1024,
    ):

        print(
            "✅ [WS] Server READY.",
            flush=True,
        )


        await asyncio.Future()


# ==========================================================
# ML READINESS
# ==========================================================

def monitor_ml_readiness():

    try:

        recognizer = (
            get_face_recognizer()
        )


        print(
            "⏳ [ML] "
            "Waiting for "
            "YuNet + SFace + gallery...",
            flush=True,
        )


        ready = (
            recognizer
            .wait_until_ready(
                timeout=None
            )
        )


        if ready:

            print(
                "✅ [ML] "
                "FULL FACE RECOGNITION READY.",
                flush=True,
            )

        else:

            print(
                "❌ [ML] "
                "Recognition stack not ready.",
                flush=True,
            )


    except Exception as exc:

        print(
            f"❌ [ML] "
            f"Readiness error: {exc}",
            flush=True,
        )


# ==========================================================
# START
# ==========================================================

def start_camera_servers_once():

    global _servers_started

    global _recognition_worker_started


    with _servers_lock:

        if _servers_started:

            return


        _servers_started = True


    print(
        "🚀 [CAMERA] "
        "Starting services...",
        flush=True,
    )


    # ------------------------------------------------------
    # Preload
    # ------------------------------------------------------

    start_process_preload()


    print(
        "✅ [ML] "
        "Background preload started.",
        flush=True,
    )


    # ------------------------------------------------------
    # SFace worker
    # ------------------------------------------------------

    if not _recognition_worker_started:

        _recognition_worker_started = True


        threading.Thread(
            target=
                recognition_worker,

            daemon=True,

            name=
                "sface-recognition-worker",
        ).start()


    # ------------------------------------------------------
    # HTTP
    # ------------------------------------------------------

    threading.Thread(
        target=
            run_http_server,

        daemon=True,

        name=
            "camera-http-server",
    ).start()


    # ------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------

    threading.Thread(
        target=
            lambda:
                asyncio.run(
                    run_ws_server()
                ),

        daemon=True,

        name=
            "camera-websocket-server",
    ).start()


    # ------------------------------------------------------
    # Readiness monitor
    # ------------------------------------------------------

    threading.Thread(
        target=
            monitor_ml_readiness,

        daemon=True,

        name=
            "face-ml-readiness-monitor",
    ).start()


    print(
        "✅ [CAMERA] "
        "HTTP + WebSocket + "
        "SFace worker started.",
        flush=True,
    )
