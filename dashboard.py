# ==========================================================
# dashboard.py
# ==========================================================

import os
import json
import re
import threading
import time
import textwrap
from pathlib import Path

import streamlit as st
import numpy as np  # initialize NumPy before fragment/WebRTC loading

# Eagerly import the lightweight media/runtime modules during dashboard
# module initialization. app.py also imports these at process startup;
# this fallback keeps dashboard.py independently usable.
try:
    import av  # noqa: F401
except Exception as exc:
    print(f"[AV] runtime import failed: {exc}")

# Import WebRTC early. The component is mounted when Dashboard is rendered
# and is kept PLAYING for camera prewarm. AI processing is gated separately
# by _camera_session_active and starts only after START.
try:
    from streamlit_webrtc import (
        WebRtcMode,
        webrtc_streamer,
    )

    WEBRTC_AVAILABLE = True
except Exception as exc:
    WebRtcMode = None
    webrtc_streamer = None
    WEBRTC_AVAILABLE = False
    print(f"[WebRTC] runtime import failed: {exc}")

# ==========================================================
# CONFIG
# ==========================================================

CAMERA_KEY = "hr-face-camera"


def _get_camera_key():
    """Return a stable WebRTC key that changes only after logout.

    The UI/layout remains unchanged. A new key after logout prevents a stale
    browser RTCPeerConnection from being reused by the next login session.
    """
    instance = int(st.session_state.get("camera_instance", 0))
    return f"{CAMERA_KEY}-{instance}"


FACE_TOLERANCE = 0.60
PROCESS_INTERVAL = 0.55
FRAME_SCALE = 0.50

BASE_DIR = Path(__file__).resolve().parent
KNOWN_FACES_DIR = BASE_DIR / "known_faces"

# Persistent live recognition result.
# This is separate from .face_embeddings_cache.pkl, which stores gallery embeddings.
RECOGNITION_RESULT_FILE = BASE_DIR / "face_recognition_result.json"

EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

# ==========================================================
# SESSION
# ==========================================================

DEFAULTS = {
    "logged_in": False,
    "user_data": None,

    # Recognition persistence:
    # exactly ONE successful green-box recognition per login.
    "recognition_login_id": 0,
    "previous_logged_in": False,

    "show_camera": False,
    # True = sidebar is visible on the first dashboard render.
    # False = hidden, and the hamburger can open it again.
    "sidebar_open": True,
    "reload_message": None,
    "camera_instance": 0,

    # internal camera UI state
    "camera_last_playing": False,
    "camera_session_id": 0,
    # Camera UI state retained for lifecycle compatibility.
    "camera_view_ready": False,

    # WebRTC device recovery state.
    "camera_retry_count": 0,

    # START/STOP controls recognition only.
    # WebRTC itself stays prewarmed while Dashboard is open.
    "camera_requested": False,
    "camera_prepared": False,
}


def init_dashboard_session_state():
    """Initialize dashboard session state only during an active Streamlit run."""
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            if isinstance(value, dict):
                st.session_state[key] = value.copy()
            elif isinstance(value, list):
                st.session_state[key] = value.copy()
            elif isinstance(value, set):
                st.session_state[key] = value.copy()
            else:
                st.session_state[key] = value


def begin_new_login_session():
    """
    Start a new logical login session.

    Rule:
        1 login = maximum 1 saved successful (green-box) recognition.
    """

    st.session_state["recognition_login_id"] = (
        int(
            st.session_state.get(
                "recognition_login_id",
                0,
            )
        ) + 1
    )

    processor = get_processor()

    if processor is not None:
        with processor.lock:
            processor.result_saved = False
            processor.login_id = (
                st.session_state["recognition_login_id"]
            )

    print(
        "=========================================="
    )
    print(
        "[LOGIN] New recognition login session:",
        st.session_state["recognition_login_id"],
    )
    print(
        "=========================================="
    )


def sync_login_session():
    """
    Detect a real login transition: False -> True.
    The recognition save slot is reset ONLY here.
    """

    current_logged_in = bool(
        st.session_state.get(
            "logged_in",
            False,
        )
    )

    previous_logged_in = bool(
        st.session_state.get(
            "previous_logged_in",
            False,
        )
    )

    if current_logged_in and not previous_logged_in:
        begin_new_login_session()

    st.session_state["previous_logged_in"] = current_logged_in


# ==========================================================
# GLOBAL RUNTIME
# ==========================================================

_face_recognizer = None
_face_recognizer_lock = threading.Lock()

_processor = None
_processor_lock = threading.Lock()

# Lightweight camera lifecycle signals. Heavy ML state belongs to face_utils.py
# and is preloaded once at process startup by app.py.
_first_camera_frame = threading.Event()
_camera_session_active = threading.Event()

# Monotonic camera generation. Every new Camera session gets a new token.
# Old WebRTC callbacks are ignored after STOP/LOGOUT/new START.
_camera_generation = 0
_camera_generation_lock = threading.Lock()

# Camera lifecycle state is intentionally lightweight.
_camera_ui_lock = threading.Lock()

# ==========================================================
# CAMERA PERFORMANCE TIMING (TERMINAL ONLY)
# ==========================================================
# Diagnostic instrumentation only. It does not change layout, UI, model
# initialization, recognition settings, or camera behavior.
_camera_timing_lock = threading.Lock()
_camera_start_time = None
_camera_signalling_time = None
_camera_webrtc_playing_time = None
_camera_first_frame_time = None
_camera_first_recognition_time = None
_camera_timing_session = 0
_camera_first_frame_logged = False
_camera_first_recognition_logged = False
_ml_ready_logged = False


def reset_camera_timing(session_id):
    global _camera_start_time
    global _camera_signalling_time
    global _camera_webrtc_playing_time
    global _camera_first_frame_time
    global _camera_first_recognition_time
    global _camera_timing_session
    global _camera_first_frame_logged
    global _camera_first_recognition_logged

    with _camera_timing_lock:
        _camera_start_time = time.perf_counter()
        _camera_signalling_time = None
        _camera_webrtc_playing_time = None
        _camera_first_frame_time = None
        _camera_first_recognition_time = None
        _camera_timing_session = session_id
        _camera_first_frame_logged = False
        _camera_first_recognition_logged = False


def log_camera_timing(event_name, elapsed=None):
    if elapsed is None:
        print(f"[CAMERA TIMING] {event_name}")
    else:
        print(f"[CAMERA TIMING] {event_name}: {elapsed:.3f}s")


# ==========================================================
# HTML
# ==========================================================

def render_html(content):
    """Render trusted local HTML without Markdown indentation issues."""
    if not content:
        return

    content = textwrap.dedent(str(content)).strip()

    try:
        if hasattr(st, "html"):
            st.html(content)
        else:
            st.markdown(
                content,
                unsafe_allow_html=True,
            )
    except Exception:
        try:
            st.markdown(
                content,
                unsafe_allow_html=True,
            )
        except Exception:
            pass


# ==========================================================
# DATASET
# ==========================================================

def clean_face_name(name):
    if not name:
        return "Unknown"

    value = os.path.splitext(
        os.path.basename(str(name))
    )[0]

    value = re.sub(
        r"_\d+$",
        "",
        value,
    )

    value = value.replace(
        "_",
        " ",
    )

    return (
            " ".join(value.split()).strip()
            or "Unknown"
    )


def get_dataset_files():
    try:
        if not KNOWN_FACES_DIR.exists():
            return []

        return sorted(
            (
                path
                for path in KNOWN_FACES_DIR.iterdir()
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in EXTENSIONS
            )
            ),
            key=lambda path: path.name.lower(),
        )

    except Exception as exc:
        print(
            f"Dataset scan error: {exc}"
        )
        return []


def get_dataset_count():
    return len(
        get_dataset_files()
    )


def get_dataset_names():
    names = []

    for path in get_dataset_files():

        name = clean_face_name(
            path.name
        )

        if (
                name != "Unknown"
                and name not in names
        ):
            names.append(name)

    return names


# ==========================================================
# PERSISTENT RECOGNITION RESULT
# ==========================================================

RECOGNITION_RESULT_LOCK = threading.Lock()


def save_recognition_result(
    name,
    similarity,
    top_mean,
    login_id,
):
    """
    Append one successful green-box recognition to JSON.

    - Unknown / red box is never passed here.
    - Existing records are preserved.
    - Each logical login calls this at most once.
    """

    payload = {
        "login_id": int(login_id),
        "name": clean_face_name(name),
        "similarity": float(similarity),
        "top3_mean": float(top_mean),
        "recognized_at": time.time(),
    }

    temp_path = RECOGNITION_RESULT_FILE.with_name(
        RECOGNITION_RESULT_FILE.name + ".tmp"
    )

    with RECOGNITION_RESULT_LOCK:
        try:
            records = []

            if RECOGNITION_RESULT_FILE.exists():
                try:
                    with RECOGNITION_RESULT_FILE.open(
                        "r",
                        encoding="utf-8",
                    ) as handle:
                        existing = json.load(handle)
                except (json.JSONDecodeError, OSError) as exc:
                    print(
                        "⚠️ Existing recognition JSON could not be read: "
                        f"{exc}"
                    )
                    existing = []

                if isinstance(existing, list):
                    records = existing
                elif isinstance(existing, dict):
                    # Backward compatibility with the old one-record format.
                    records = [existing]

            # Append only; never replace existing history.
            records.append(payload)

            # Atomic write: write temp file first, then replace target.
            with temp_path.open(
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    records,
                    handle,
                    indent=4,
                    ensure_ascii=False,
                )

            temp_path.replace(RECOGNITION_RESULT_FILE)

            print(
                "✅ Recognition saved | "
                f"login_id={login_id} | "
                f"name={payload['name']} | "
                f"similarity={payload['similarity']:.4f}"
            )

            return True

        except Exception as exc:
            print(
                f"❌ Recognition result save error: {exc}"
            )

            try:
                temp_path.unlink(
                    missing_ok=True
                )
            except Exception:
                pass

            return False


def load_recognition_result():
    """
    Load the latest saved recognition result.

    Returns:
        dict | None
    """
    try:
        if not RECOGNITION_RESULT_FILE.exists():
            return None

        with RECOGNITION_RESULT_FILE.open(
            "r",
            encoding="utf-8",
        ) as handle:
            payload = json.load(handle)

        if isinstance(payload, list):
            return payload[-1] if payload else None

        if isinstance(payload, dict):
            return payload

        return None

    except Exception as exc:
        print(
            f"⚠️ Recognition result load error: {exc}"
        )
        return None


# ==========================================================
# DATABASE - LAZY
# ==========================================================

@st.cache_data(
    ttl=30,
    show_spinner=False,
)
def get_dashboard_data():
    try:

        from database import (
            get_attendance_summary,
        )

        return get_attendance_summary()

    except Exception as exc:

        print(
            f"Database error: {exc}"
        )

        return {
            "total_employees": 0,
            "today_attendance": 0,
            "absent": 0,
        }


# ==========================================================
# FACE RECOGNIZER - LAZY
# ==========================================================

def get_face_recognizer():
    global _face_recognizer

    with _face_recognizer_lock:
        if _face_recognizer is None:
            from face_utils import (
                face_recognizer,
            )

            _face_recognizer = (
                face_recognizer
            )

    return _face_recognizer


# ==========================================================
# PROCESSOR
# ==========================================================

class FaceRecognitionProcessor:

    def __init__(self):
        self.lock = threading.Lock()

        self.last_submit_time = 0.0
        self.generation = 0

        self.locations = []
        self.names = []
        self.ids = []

        # Fast tracker state is updated every WebRTC frame.
        self.tracking_location = None
        self.tracking_detection = None
        self.tracking_name = "Unknown"

        # Save one successful recognition per logical login.
        # The actual result is persisted on disk.
        self.result_saved = False
        self.login_id = 0

        # Only the newest frame is retained for the SFace worker.
        self.pending_frame = None
        self.pending_detection = None
        self.pending_generation = 0

        self._wake_event = threading.Event()
        self._worker_started = False

        self._start_worker()

    def _start_worker(self):
        with self.lock:
            if self._worker_started:
                return
            self._worker_started = True

        threading.Thread(
            target=self._recognition_worker,
            daemon=True,
            name="face-recognition-worker",
        ).start()

    def get_result(self):
        with self.lock:
            return (
                list(self.locations),
                list(self.names),
                list(self.ids),
            )

    def clear(self):
        with self.lock:
            self.locations.clear()
            self.names.clear()
            self.ids.clear()

            self.tracking_location = None
            self.tracking_detection = None
            self.tracking_name = "Unknown"

            # IMPORTANT:
            # Do NOT reset result_saved here.
            # clear() is called by START/STOP/reload/navigation.
            # The save slot must remain locked until LOGOUT -> LOGIN.

            self.last_submit_time = 0.0
            self.generation += 1

            self.pending_frame = None
            self.pending_detection = None
            self.pending_generation = self.generation

        self._wake_event.set()

    def process(self, image):
        """
        Low-latency live path.

        YuNet tracks on every camera frame. SFace recognition is throttled and
        performed asynchronously, so the bounding box never waits for identity.
        """
        if (
            image is None
            or not _camera_session_active.is_set()
        ):
            return self.get_result()

        recognizer = get_face_recognizer()
        if recognizer is None:
            return self.get_result()

        try:
            detection, stable_box = recognizer.track_face(image)
        except Exception as exc:
            print(f"Fast tracking callback error: {exc}")
            detection, stable_box = None, None

        location = None
        if stable_box is not None:
            x, y, w, h = stable_box
            location = (
                y,
                x + w,
                y + h,
                x,
            )

        now = time.monotonic()

        with self.lock:
            if location is not None:
                self.tracking_location = location
            else:
                location = self.tracking_location

            current_name = self.tracking_name

            # Queue recognition only when enough time has elapsed AND a real
            # detection exists. The newest frame/detection always wins.
            if (
                detection is not None
                and now - self.last_submit_time >= PROCESS_INTERVAL
            ):
                self.last_submit_time = now
                self.pending_frame = image.copy()
                self.pending_detection = dict(detection)
                self.pending_generation = self.generation

        self._wake_event.set()

        if location is None:
            return [], [], []

        return (
            [location],
            [current_name],
            [None],
        )

    def _recognition_worker(self):
        """
        Background identity recognition.

        YuNet has already detected the face in the callback, so this worker
        does not perform a second detector pass.
        """
        while True:
            self._wake_event.wait()
            self._wake_event.clear()

            with self.lock:
                frame = self.pending_frame
                detection = self.pending_detection
                generation = self.pending_generation
                self.pending_frame = None
                self.pending_detection = None

            if frame is None or detection is None:
                continue

            if not _camera_session_active.is_set():
                continue

            try:
                recognizer = get_face_recognizer()

                if recognizer is None:
                    continue

                if not recognizer.is_model_ready():
                    continue

                recognition_started = time.perf_counter()

                name, similarity, top_mean = (
                    recognizer.recognize_detection(
                        frame,
                        detection,
                        tolerance=FACE_TOLERANCE,
                    )
                )

                recognition_finished = time.perf_counter()
                recognition_processing = (
                    recognition_finished
                    - recognition_started
                )

                global _camera_first_recognition_time
                global _camera_first_recognition_logged

                with _camera_timing_lock:
                    if (
                        _camera_start_time is not None
                        and not _camera_first_recognition_logged
                    ):
                        _camera_first_recognition_time = (
                            recognition_finished
                        )
                        total_elapsed = (
                            recognition_finished
                            - _camera_start_time
                        )
                        _camera_first_recognition_logged = True

                        log_camera_timing(
                            "First recognition completed",
                            total_elapsed,
                        )
                        log_camera_timing(
                            "First recognition processing time",
                            recognition_processing,
                        )

                        if _camera_first_frame_time is not None:
                            log_camera_timing(
                                "First recognition after first frame",
                                recognition_finished
                                - _camera_first_frame_time,
                            )

                        if _camera_webrtc_playing_time is not None:
                            log_camera_timing(
                                "First recognition after WebRTC playing",
                                recognition_finished
                                - _camera_webrtc_playing_time,
                            )

                cleaned_name = (
                    "Unknown"
                    if not name or name == "Unknown"
                    else clean_face_name(name)
                )

                with self.lock:
                    if generation != self.generation:
                        continue

                    self.tracking_name = cleaned_name

                    # GREEN BOX VALIDATION:
                    # draw_face() uses exactly the same rule:
                    #   Unknown -> RED
                    #   known name -> GREEN
                    #
                    # Save only the first GREEN recognition of this login.
                    # Unknown is never saved.
                    if (
                        cleaned_name != "Unknown"
                        and not self.result_saved
                    ):
                        if save_recognition_result(
                            cleaned_name,
                            similarity,
                            top_mean,
                            login_id=self.login_id,
                        ):
                            self.result_saved = True

                    if self.tracking_location is not None:
                        self.locations = [
                            self.tracking_location
                        ]
                    else:
                        self.locations = []

                    self.names = [cleaned_name]
                    self.ids = [None]

            except Exception as exc:
                print(
                    f"Recognition worker error: {exc}"
                )


# ==========================================================

def get_processor():
    global _processor

    with _processor_lock:
        if _processor is None:
            _processor = (
                FaceRecognitionProcessor()
            )

    return _processor


def clear_processor():
    global _processor

    with _processor_lock:
        if _processor is not None:
            _processor.clear()


# Start the tiny recognition worker during module initialization.
# This does not load SFace/YuNet; those are owned by face_utils and
# preloaded by app.py. It only removes the first-frame thread-creation cost
# after START. Keeping session-state initialization out of module scope lets
# app.py import this module safely in its startup warm-up thread.
try:
    get_processor()
    print("✅ [Processor prewarm] recognition worker READY and idle.")
except Exception as exc:
    print(f"[Processor prewarm] failed: {exc}")


# ==========================================================
# DRAW FACE
# ==========================================================

def draw_face(
        image,
        location,
        name,
):
    import cv2

    top, right, bottom, left = location

    # Locations are now full-resolution coordinates from the fast tracker.
    top = int(top)
    right = int(right)
    bottom = int(bottom)
    left = int(left)

    height, width = (
        image.shape[:2]
    )

    left = max(
        0,
        min(left, width - 1),
    )

    right = max(
        0,
        min(right, width - 1),
    )

    top = max(
        0,
        min(top, height - 1),
    )

    bottom = max(
        0,
        min(bottom, height - 1),
    )

    color = (
        (0, 0, 255)
        if name == "Unknown"
        else (0, 180, 0)
    )

    cv2.rectangle(
        image,
        (left, top),
        (right, bottom),
        color,
        2,
    )

    label = clean_face_name(
        name
    )

    (
        (text_width, text_height),
        _,
    ) = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        2,
    )

    label_top = max(
        0,
        top - text_height - 10,
    )

    cv2.rectangle(
        image,
        (left, label_top),
        (
            left + text_width + 12,
            top,
        ),
        color,
        -1,
    )

    cv2.putText(
        image,
        label,
        (
            left + 6,
            max(
                text_height + 4,
                top - 5,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


# ==========================================================
# WEBRTC CALLBACK
# ==========================================================

def video_frame_callback(frame):
    """Low-latency WebRTC callback.

    PREWARM frames are returned immediately without ndarray conversion because
    AI is disabled before START. After START, the existing recognition path is
    unchanged.
    """
    # PREWARM: skip ndarray conversion and VideoFrame allocation.
    if not _camera_session_active.is_set():
        return frame

    import av

    image = frame.to_ndarray(format="bgr24")

    _first_camera_frame.set()

    global _camera_first_frame_time
    global _camera_first_frame_logged

    with _camera_timing_lock:
        if (
            _camera_start_time is not None
            and not _camera_first_frame_logged
        ):
            _camera_first_frame_time = time.perf_counter()
            elapsed = (
                _camera_first_frame_time
                - _camera_start_time
            )
            _camera_first_frame_logged = True
            log_camera_timing(
                "First video frame",
                elapsed,
            )

    locations = []
    names = []
    ids = []

    try:
        locations, names, ids = (
            get_processor().process(image)
        )
    except Exception as exc:
        print(
            f"Recognition callback error: {exc}"
        )

    if locations:
        for location, name, _ in zip(
            locations,
            names,
            ids,
        ):
            draw_face(
                image,
                location,
                name,
            )

    import cv2

    cv2.putText(
        image,
        f"Faces: {len(locations)}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return av.VideoFrame.from_ndarray(
        image,
        format="bgr24",
    )


# ==========================================================
# NAVIGATION
# ==========================================================

def _next_camera_generation():
    global _camera_generation

    with _camera_generation_lock:
        _camera_generation += 1
        return _camera_generation


def _get_camera_generation():
    with _camera_generation_lock:
        return _camera_generation


def go_dashboard():
    # Invalidate any existing WebRTC callback first.
    _next_camera_generation()

    _camera_session_active.clear()
    _first_camera_frame.clear()
    clear_processor()

    st.session_state["show_camera"] = False
    st.session_state["camera_last_playing"] = False
    st.session_state["camera_view_ready"] = False
    st.session_state["camera_retry_count"] = 0
    st.session_state["camera_requested"] = False
    st.session_state["camera_prepared"] = False
    st.session_state["reload_message"] = None


def go_camera():
    """Open the camera page without loading ML synchronously.

    The Dashboard mounts the persistent WebRTC component immediately and
    keeps it prewarmed in the background. START only reveals the same stream
    and enables the AI processing path.
    """
    current = st.session_state.get("show_camera", False)

    if not current:
        session_id = _next_camera_generation()
        st.session_state["camera_instance"] = (
                st.session_state.get("camera_instance", 0) + 1
        )
        st.session_state["camera_session_id"] = session_id
        st.session_state["camera_view_ready"] = False
        st.session_state["camera_requested"] = False
        st.session_state["camera_last_playing"] = False
        st.session_state["camera_retry_count"] = 0
        _first_camera_frame.clear()
        _camera_session_active.clear()
        clear_processor()
        st.session_state["show_camera"] = True

    st.session_state["reload_message"] = None


def toggle_sidebar():
    st.session_state[
        "sidebar_open"
    ] = not st.session_state.get(
        "sidebar_open",
        True,
    )


def logout():
    # Invalidate the current WebRTC callback immediately.
    _next_camera_generation()

    # Force a fresh WebRTC component after the next login.
    # This is lifecycle state only; it does not change the UI/layout.
    st.session_state["camera_instance"] = (
        int(st.session_state.get("camera_instance", 0)) + 1
    )

    _camera_session_active.clear()
    _first_camera_frame.clear()
    clear_processor()

    # Explicitly remove every camera/loading state before changing auth state.
    st.session_state["camera_last_playing"] = False
    st.session_state["camera_session_id"] = 0
    st.session_state["camera_view_ready"] = False
    st.session_state["camera_retry_count"] = 0
    st.session_state["camera_requested"] = False
    st.session_state["camera_prepared"] = False
    st.session_state["show_camera"] = False
    st.session_state["reload_message"] = None
    st.session_state["dashboard_loading_seen"] = False
    st.session_state["logged_in"] = False
    st.session_state["user_data"] = None

    # Mark the authentication transition as ended.
    # The next False -> True transition creates a new login session.
    st.session_state["previous_logged_in"] = False

    # Do not force st.rerun(). The callback itself triggers the normal rerun.


# ==========================================================
# CSS
# ==========================================================
# ==========================================================

def apply_dashboard_css():
    sidebar_open = st.session_state.get(
        "sidebar_open",
        True,
    )

    sidebar_display = (
        "block"
        if sidebar_open
        else "none"
    )

    content_margin = (
        "220px"
        if sidebar_open
        else "0px"
    )

    # Live Camera is embedded directly below Live Location on Dashboard.
    # WebRTC stays mounted/prewarmed, but the actual video is visually hidden
    # until START. This preserves the existing UI while allowing instant reveal.
    camera_host_css = "display:block;"

    render_html(
        f"""
        <style>

        /* ==================================================
           GLOBAL
           ================================================== */

        *,
        *::before,
        *::after {{
            box-sizing:border-box;
        }}

        html,
        body {{
            margin:0 !important;
            padding:0 !important;

            overflow-x:hidden !important;

            background:#f2f8f4 !important;
        }}

        body,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {{
            background:
                linear-gradient(
                    180deg,
                    #f8fcf9 0%,
                    #edf7ef 100%
                )
                !important;
        }}

        /* ==================================================
           HEADER
           ================================================== */

        [data-testid="stHeader"] {{
            height:60px !important;
            min-height:60px !important;

            background:
                linear-gradient(
                    90deg,
                    #ffffff 0%,
                    #eff9f1 100%
                )
                !important;

            border-bottom:
                1px solid #cfe5d3
                !important;

            position:fixed !important;

            top:0 !important;
            left:0 !important;
            right:0 !important;

            z-index:1000 !important;
        }}

        [data-testid="stHeader"] > div {{
            height:60px !important;
        }}

        /* ==================================================
           HIDE STREAMLIT CONTROLS
           ================================================== */

        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        [data-testid="stAppDeployButton"],
        .stDeployButton,
        footer,
        #MainMenu,
        .stAppFooter,
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapsedControl"],
        button[aria-label="Open sidebar"],
        button[aria-label="Close sidebar"] {{
            display:none !important;
        }}

        /* ==================================================
           MAIN
           ================================================== */

        [data-testid="stMainBlockContainer"] {{
            width:
                calc(100% - {content_margin})
                !important;

            max-width:none !important;

            margin-left:
                {content_margin}
                !important;

            margin-right:0 !important;

            padding:
                84px 34px 42px
                !important;

            min-height:100vh !important;

            background:transparent !important;
        }}

        [data-testid="stMainBlockContainer"] > div {{
            width:100% !important;

            max-width:none !important;

            margin:0 !important;
            padding:0 !important;
        }}

        /* ==================================================
           SIDEBAR
           ================================================== */

        [data-testid="stSidebar"] {{
            display:block !important;
            visibility:visible !important;
            pointer-events:auto !important;

            position:fixed !important;

            top:60px !important;
            left:0 !important;
            bottom:0 !important;

            width:220px !important;
            min-width:220px !important;
            max-width:220px !important;

            transform:
                translateX({"0" if sidebar_open else "-100%"})
                !important;
            transition:transform .18s ease !important;

            height:
                calc(100vh - 60px)
                !important;

            background:
                linear-gradient(
                    180deg,
                    #ffffff 0%,
                    #f3faf4 42%,
                    #e4f4e7 100%
                )
                !important;

            border-right:
                1px solid #cfe5d3
                !important;

            box-shadow:
                3px 0 20px
                rgba(38,112,55,.055)
                !important;

            overflow:hidden !important;

            z-index:900 !important;
        }}

        [data-testid="stSidebarHeader"] {{
            display:none !important;
        }}

        [data-testid="stSidebarContent"] {{
            display:block !important;
            visibility:visible !important;
            width:100% !important;
            height:100% !important;

            padding:
                27px 14px 22px
                !important;

            margin:0 !important;

            background:transparent !important;

            overflow:hidden !important;
        }}

        [data-testid="stSidebarUserContent"] {{
            display:block !important;
            visibility:visible !important;
            width:100% !important;

            margin:0 !important;
            padding:0 !important;
        }}

        /* ==================================================
           BRAND
           ================================================== */

        .sidebar-brand {{
            width:100% !important;

            display:flex !important;
            align-items:center !important;

            gap:10px !important;

            padding:
                0 8px
                !important;

            margin:
                0 0 17px
                !important;
        }}

        .sidebar-brand-icon {{
            width:38px !important;
            height:38px !important;

            flex:0 0 38px !important;

            display:flex !important;
            align-items:center !important;
            justify-content:center !important;

            border-radius:
                11px
                !important;

            background:
                linear-gradient(
                    135deg,
                    #58be62 0%,
                    #27883a 100%
                )
                !important;

            color:#ffffff !important;

            font-size:11px !important;
            font-weight:800 !important;

            box-shadow:
                0 6px 14px
                rgba(39,136,58,.20)
                !important;
        }}

        .sidebar-brand-title {{
            color:#184a29 !important;

            font-size:13px !important;
            font-weight:750 !important;

            line-height:18px !important;
        }}

        .sidebar-brand-subtitle {{
            color:#74857a !important;

            font-size:10px !important;
            font-weight:500 !important;

            line-height:14px !important;
        }}

        .sidebar-divider {{
            width:100% !important;
            height:1px !important;

            background:#d8ebdc !important;

            margin:
                0 0 19px
                !important;
        }}

        /* ==================================================
           SIDEBAR TITLES
           ================================================== */

        .sidebar-menu-title,
        .sidebar-section-title {{
            width:100% !important;

            padding:
                0 10px
                !important;

            color:#5f7c68 !important;

            font-size:10px !important;

            font-weight:800 !important;

            line-height:18px !important;

            letter-spacing:
                1px
                !important;

            text-transform:
                uppercase
                !important;

            text-align:left !important;
        }}

        .sidebar-menu-title {{
            margin:
                0 0 8px
                !important;
        }}

        .sidebar-section-title {{
            margin:
                23px 0 8px
                !important;
        }}

        .sidebar-logout-space {{
            height:31px !important;
        }}

        /* ==================================================
           SIDEBAR BUTTON
           ================================================== */

        [data-testid="stSidebar"]
        .stButton {{
            width:100% !important;

            margin:0 !important;
            padding:0 !important;
        }}

        [data-testid="stSidebar"]
        .stButton > button {{
            position:relative !important;

            width:100% !important;

            height:44px !important;
            min-height:44px !important;

            margin:
                0 0 7px
                !important;

            padding:
                0 12px
                !important;

            border:
                1px solid transparent
                !important;

            border-radius:
                11px
                !important;

            background:
                transparent
                !important;

            color:
                #405449
                !important;

            box-shadow:
                none
                !important;

            display:flex !important;

            align-items:center !important;

            justify-content:
                flex-start
                !important;

            text-align:left !important;

            overflow:hidden !important;

            cursor:pointer !important;

            transition:
                background .15s ease,
                color .15s ease,
                border-color .15s ease,
                box-shadow .15s ease
                !important;
        }}

        /* ==================================================
           ICON + TEXT
           ================================================== */

        [data-testid="stSidebar"]
        .stButton > button > div {{
            width:100% !important;
            height:100% !important;

            display:flex !important;

            align-items:center !important;

            justify-content:
                flex-start
                !important;

            gap:11px !important;

            margin:0 !important;
            padding:0 !important;

            overflow:visible !important;
        }}

        [data-testid="stSidebar"]
        .stButton
        [data-testid="stIconMaterial"] {{
            width:20px !important;

            min-width:20px !important;
            max-width:20px !important;

            height:20px !important;

            min-height:20px !important;

            flex:0 0 20px !important;

            display:flex !important;

            align-items:center !important;

            justify-content:center !important;

            margin:0 !important;
            padding:0 !important;
        }}

        [data-testid="stSidebar"]
        .stButton > button p {{
            flex:1 1 auto !important;

            width:auto !important;

            margin:0 !important;
            padding:0 !important;

            color:inherit !important;

            font-size:13px !important;

            font-weight:550 !important;

            line-height:20px !important;

            text-align:left !important;

            white-space:nowrap !important;
        }}

        /* ==================================================
           HOVER
           ================================================== */

        [data-testid="stSidebar"]
        .stButton > button:hover {{
            background:
                linear-gradient(
                    90deg,
                    #daf1df 0%,
                    #ebf8ed 100%
                )
                !important;

            color:#216f36 !important;

            border-color:
                #c5e3cb
                !important;

            box-shadow:
                0 4px 11px
                rgba(39,136,58,.08)
                !important;
        }}

        [data-testid="stSidebar"]
        .stButton > button:hover p {{
            color:#216f36 !important;
        }}

        [data-testid="stSidebar"]
        .stButton > button:hover
        [data-testid="stIconMaterial"] {{
            color:#216f36 !important;
        }}

        /* ==================================================
           ACTIVE
           ================================================== */

        [data-testid="stSidebar"]
        .stButton >
        button[data-testid="baseButton-primary"],
        [data-testid="stSidebar"]
        .stButton >
        button[kind="primary"],
        [data-testid="stSidebar"]
        .stButton >
        button[class*="primary"] {{
            background:
                linear-gradient(
                    135deg,
                    #4db657 0%,
                    #25833a 100%
                )
                !important;

            color:#ffffff !important;

            border:
                1px solid #3ea54b
                !important;

            border-radius:
                11px
                !important;

            box-shadow:
                0 7px 18px
                rgba(37,131,58,.25)
                !important;
        }}

        /* active indicator */

        [data-testid="stSidebar"]
        .stButton >
        button[data-testid="baseButton-primary"]::before,
        [data-testid="stSidebar"]
        .stButton >
        button[kind="primary"]::before,
        [data-testid="stSidebar"]
        .stButton >
        button[class*="primary"]::before {{
            content:"" !important;

            position:absolute !important;

            left:0 !important;

            top:7px !important;
            bottom:7px !important;

            width:4px !important;

            background:#e1f9e5 !important;

            border-radius:
                0 5px 5px 0
                !important;
        }}

        /* active text */

        [data-testid="stSidebar"]
        .stButton >
        button[data-testid="baseButton-primary"] p,
        [data-testid="stSidebar"]
        .stButton >
        button[kind="primary"] p,
        [data-testid="stSidebar"]
        .stButton >
        button[class*="primary"] p {{
            color:#ffffff !important;

            font-weight:700 !important;
        }}

        /* active icon */

        [data-testid="stSidebar"]
        .stButton >
        button[data-testid="baseButton-primary"]
        [data-testid="stIconMaterial"],
        [data-testid="stSidebar"]
        .stButton >
        button[kind="primary"]
        [data-testid="stIconMaterial"],
        [data-testid="stSidebar"]
        .stButton >
        button[class*="primary"]
        [data-testid="stIconMaterial"] {{
            color:#ffffff !important;
        }}

        /* ==================================================
           LOGOUT
           ================================================== */

        [data-testid="stSidebar"]
        .sidebar-logout-space
        + .stButton
        > button {{
            background:
                rgba(255,255,255,.55)
                !important;

            color:#617168 !important;
        }}

        [data-testid="stSidebar"]
        .sidebar-logout-space
        + .stButton
        > button:hover {{
            background:
                linear-gradient(
                    90deg,
                    #fff0f0,
                    #fff8f8
                )
                !important;

            color:#bd5050 !important;

            border-color:#eed6d6 !important;

            box-shadow:none !important;
        }}

        [data-testid="stSidebar"]
        .sidebar-logout-space
        + .stButton
        > button:hover
        [data-testid="stIconMaterial"] {{
            color:#bd5050 !important;
        }}

        /* ==================================================
           HAMBURGER
           ================================================== */

        div.st-key-sidebar_toggle_container {{
            position:fixed !important;

            top:8px !important;
            left:8px !important;

            width:44px !important;
            height:44px !important;

            margin:0 !important;
            padding:0 !important;

            z-index:2000 !important;
        }}

        div.st-key-sidebar_toggle_container .stButton {{
            width:44px !important;
            height:44px !important;

            margin:0 !important;
            padding:0 !important;
        }}

        div.st-key-sidebar_toggle_container button {{
            width:44px !important;
            height:44px !important;
            min-height:44px !important;

            margin:0 !important;
            padding:0 !important;

            border:
                1px solid #cde4d2
                !important;

            border-radius:
                10px
                !important;

            background:
                rgba(255,255,255,.96)
                !important;

            color:#35533d !important;

            box-shadow:
                0 3px 12px
                rgba(39,111,53,.10)
                !important;

            display:flex !important;

            align-items:center !important;

            justify-content:center !important;

            font-size:23px !important;
        }}

        div.st-key-sidebar_toggle_container button:hover {{
            background:#e2f5e6 !important;

            color:#24743a !important;

            border-color:
                #bddfc5
                !important;
        }}


        /* ==================================================
           CAMERA COMPONENT
           ================================================== */

        .camera-webrtc-shell {{
            width:100% !important;
            min-height:360px !important;
            background:#ffffff !important;
            border:1px solid #d7e8dc !important;
            border-radius:14px !important;
            overflow:hidden !important;
            contain:layout paint !important;
        }}

        /* Full-width camera panel aligned with Live Location. */
        div.st-key-camera_prewarm_shell {{
            position:relative !important;
            width:100% !important;
            max-width:none !important;
            min-height:360px !important;
            height:360px !important;
            margin:12px 0 14px !important;
            padding:0 !important;
            border:1px solid #d7e8dc !important;
            border-radius:14px !important;
            overflow:hidden !important;
            background:#ffffff !important;
            contain:layout paint !important;
            box-shadow:0 4px 16px rgba(38,112,55,.045) !important;
        }}

        /* Keep one persistent WebRTC host mounted for prewarm. */
        div.st-key-camera_webrtc_host {{
            position:absolute !important;
            inset:0 !important;
            width:100% !important;
            height:100% !important;
            max-width:none !important;
            margin:0 !important;
            padding:0 !important;
            border-radius:14px !important;
            overflow:hidden !important;
            z-index:1 !important;
            transition:opacity .08s linear !important;
            display:block !important;
        }}

        div.st-key-camera_webrtc_host [data-testid="stCustomComponentV1"],
        div.st-key-camera_webrtc_host [data-testid="stCustomComponentV1"] > div,
        div.st-key-camera_webrtc_host iframe {{
            width:100% !important;
            height:360px !important;
            min-height:360px !important;
            max-width:none !important;
            border:0 !important;
            margin:0 !important;
            padding:0 !important;
            border-radius:14px !important;
            background:#ffffff !important;
        }}

        /* Idle PREWARM: stream remains active but is visually hidden. */
        div.st-key-camera_prewarm_shell:has(.camera-prewarm-idle)
        div.st-key-camera_webrtc_host {{
            opacity:0 !important;
            pointer-events:none !important;
        }}

        /* STARTED: reveal the same already-running stream. */
        div.st-key-camera_prewarm_shell:has(.camera-prewarm-live)
        div.st-key-camera_webrtc_host {{
            opacity:1 !important;
            pointer-events:auto !important;
        }}

        /* Placeholder fills the exact same full-width camera panel. */
        div.st-key-camera_prewarm_shell:has(.camera-prewarm-idle)
        .camera-prewarm-overlay {{
            display:flex !important;
        }}

        .camera-prewarm-overlay {{
            position:absolute !important;
            inset:0 !important;
            z-index:20 !important;
            width:100% !important;
            height:360px !important;
            min-height:360px !important;
            margin:0 !important;
            padding:0 !important;
            display:none !important;
            align-items:center !important;
            justify-content:center !important;
            border-radius:14px !important;
            background:#ffffff !important;
            border:0 !important;
            pointer-events:none !important;
            box-sizing:border-box !important;
        }}

        .camera-placeholder-inner {{
            width:min(330px,84%) !important;
            text-align:center !important;
            color:#617168 !important;
            display:flex !important;
            flex-direction:column !important;
            align-items:center !important;
            justify-content:center !important;
            margin:0 !important;
            padding:0 !important;
            transform:none !important;
        }}

        .camera-placeholder-monitor {{
            width:110px !important;
            height:68px !important;
            margin:0 0 16px !important;
            border:3px solid #7ca78a !important;
            border-radius:9px !important;
            background:#ffffff !important;
            box-shadow:0 4px 12px rgba(39,111,53,.05) !important;
            position:relative !important;
            flex:0 0 auto !important;
        }}

        .camera-placeholder-monitor::before {{
            content:"" !important;
            position:absolute !important;
            width:66px !important;
            height:6px !important;
            left:50% !important;
            bottom:-14px !important;
            transform:translateX(-50%) !important;
            background:#7ca78a !important;
            border-radius:0 0 7px 7px !important;
        }}

        .camera-placeholder-monitor::after {{
            content:"📷" !important;
            position:absolute !important;
            inset:0 !important;
            display:flex !important;
            align-items:center !important;
            justify-content:center !important;
            font-size:28px !important;
        }}

        .camera-placeholder-title {{
            color:#245a34 !important;
            font-size:14px !important;
            font-weight:750 !important;
            margin:0 0 6px !important;
            text-align:center !important;
        }}

        .camera-placeholder-text {{
            color:#728078 !important;
            font-size:11px !important;
            line-height:1.5 !important;
            text-align:center !important;
        }}

        [data-testid="stCustomComponentV1"],
        [data-testid="stCustomComponentV1"] > div,
        [data-testid="stElementContainer"]:has(
            [data-testid="stCustomComponentV1"]
        ) {{
            width:100% !important;
            max-width:100% !important;
            margin:0 !important;
            padding:0 !important;
            border:0 !important;
            background:#ffffff !important;
            box-shadow:none !important;
            overflow:hidden !important;
        }}

        [data-testid="stCustomComponentV1"] iframe {{
            display:block !important;
            width:100% !important;
            height:360px !important;
            min-height:360px !important;
            border:0 !important;
            background:#ffffff !important;
        }}

        /* ==================================================
           DESKTOP CAMERA STATES
           ==================================================
           BEFORE START:
             - full-width panel stays aligned with the other desktop cards.
           AFTER START:
             - only the live video panel becomes smaller and centered.
           This block is desktop-only and does not alter the mobile rules.
        */
        @media (min-width:641px) {{

            /* Keep PREWARM / idle panel full width like the other sections. */
            div.st-key-camera_prewarm_shell:has(.camera-prewarm-idle) {{
                width:100% !important;
                max-width:none !important;
                height:360px !important;
                min-height:360px !important;
                margin:12px 0 14px !important;
            }}

            /* LIVE: compact, centered 4:3 frame matching the 640x480 source. */
            div.st-key-camera_prewarm_shell:has(.camera-prewarm-live) {{
                position:relative !important;
                width:min(100%, 480px) !important;
                max-width:480px !important;
                height:360px !important;
                min-height:360px !important;
                max-height:360px !important;
                margin:14px auto 16px !important;
                overflow:hidden !important;
                border-radius:14px !important;
                box-sizing:border-box !important;
            }}

            div.st-key-camera_prewarm_shell:has(.camera-prewarm-live)
            div.st-key-camera_webrtc_host {{
                position:absolute !important;
                inset:0 !important;
                width:100% !important;
                height:360px !important;
                max-width:100% !important;
                min-height:360px !important;
                margin:0 !important;
                padding:0 !important;
                border-radius:14px !important;
                overflow:hidden !important;
            }}

            div.st-key-camera_prewarm_shell:has(.camera-prewarm-live)
            div.st-key-camera_webrtc_host [data-testid="stCustomComponentV1"],
            div.st-key-camera_prewarm_shell:has(.camera-prewarm-live)
            div.st-key-camera_webrtc_host [data-testid="stCustomComponentV1"] > div,
            div.st-key-camera_prewarm_shell:has(.camera-prewarm-live)
            div.st-key-camera_webrtc_host iframe,
            div.st-key-camera_prewarm_shell:has(.camera-prewarm-live)
            div.st-key-camera_webrtc_host video {{
                width:100% !important;
                height:360px !important;
                min-width:0 !important;
                min-height:360px !important;
                max-width:100% !important;
                max-height:360px !important;
                display:block !important;
                margin:0 !important;
                padding:0 !important;
            }}

            div.st-key-camera_prewarm_shell:has(.camera-prewarm-live)
            div.st-key-camera_webrtc_host video {{
                object-fit:contain !important;
                object-position:center center !important;
                transform:none !important;
                background:#000000 !important;
            }}

            /* Center the control area under the compact live frame. */
            div.st-key-camera_controls {{
                width:min(100%, 480px) !important;
                max-width:480px !important;
                margin:0 auto !important;
            }}

            /* Keep the idle placeholder centered inside the full-width panel. */
            div.st-key-camera_prewarm_shell:has(.camera-prewarm-idle)
            .camera-prewarm-overlay {{
                width:100% !important;
                height:360px !important;
                min-height:360px !important;
            }}
        }}

        /* ==================================================
           ROBUST MOBILE CAMERA LAYOUT
           ==================================================
           The Streamlit element wrapper is explicitly sized so the
           iframe cannot collapse to a thin horizontal strip.
           The live state is used only after START.
        */

        @media (max-width:640px) {{

            /* The element container that owns the camera shell. */
            [data-testid="stElementContainer"]:has(
                div.st-key-camera_prewarm_shell
            ) {{
                width:100% !important;
                max-width:100% !important;
                min-width:0 !important;
                margin:0 !important;
                padding:0 !important;
            }}

            [data-testid="stElementContainer"]:has(
                div.st-key-camera_prewarm_shell
            ) > div {{
                width:100% !important;
                max-width:100% !important;
                margin:0 !important;
                padding:0 !important;
            }}

            /* AFTER START only:
               centered camera with a portrait-friendly 4:3 frame. */
            [data-testid="stElementContainer"]:has(
                div.st-key-camera_prewarm_shell .camera-prewarm-live
            ) {{
                display:flex !important;
                justify-content:center !important;
                align-items:center !important;
                width:100% !important;
                min-height:0 !important;
                margin:10px 0 14px !important;
                padding:0 !important;
            }}

            [data-testid="stElementContainer"]:has(
                div.st-key-camera_prewarm_shell .camera-prewarm-live
            ) div.st-key-camera_prewarm_shell {{
                position:relative !important;
                display:block !important;
                flex:0 0 auto !important;
                width:min(100%, 340px) !important;
                max-width:340px !important;
                height:min(calc((100vw - 28px) * 0.75), 255px) !important;
                min-height:0 !important;
                max-height:255px !important;
                margin:0 auto !important;
                padding:0 !important;
                overflow:hidden !important;
                border-radius:12px !important;
                box-sizing:border-box !important;
            }}

            [data-testid="stElementContainer"]:has(
                div.st-key-camera_prewarm_shell .camera-prewarm-live
            ) div.st-key-camera_prewarm_shell > div,
            [data-testid="stElementContainer"]:has(
                div.st-key-camera_prewarm_shell .camera-prewarm-live
            ) div.st-key-camera_prewarm_shell > div > div {{
                width:100% !important;
                height:100% !important;
                min-height:0 !important;
                max-height:100% !important;
                margin:0 !important;
                padding:0 !important;
            }}

            [data-testid="stElementContainer"]:has(
                div.st-key-camera_prewarm_shell .camera-prewarm-live
            ) div.st-key-camera_webrtc_host {{
                position:absolute !important;
                inset:0 !important;
                width:100% !important;
                height:100% !important;
                min-height:0 !important;
                max-height:100% !important;
                margin:0 !important;
                padding:0 !important;
                overflow:hidden !important;
                border-radius:12px !important;
            }}

            [data-testid="stElementContainer"]:has(
                div.st-key-camera_prewarm_shell .camera-prewarm-live
            ) div.st-key-camera_webrtc_host > div,
            [data-testid="stElementContainer"]:has(
                div.st-key-camera_prewarm_shell .camera-prewarm-live
            ) div.st-key-camera_webrtc_host [data-testid="stCustomComponentV1"],
            [data-testid="stElementContainer"]:has(
                div.st-key-camera_prewarm_shell .camera-prewarm-live
            ) div.st-key-camera_webrtc_host [data-testid="stCustomComponentV1"] > div,
            [data-testid="stElementContainer"]:has(
                div.st-key-camera_prewarm_shell .camera-prewarm-live
            ) div.st-key-camera_webrtc_host iframe {{
                width:100% !important;
                height:100% !important;
                min-height:0 !important;
                max-height:100% !important;
                margin:0 !important;
                padding:0 !important;
                border:0 !important;
                border-radius:12px !important;
                overflow:hidden !important;
            }}

            [data-testid="stElementContainer"]:has(
                div.st-key-camera_prewarm_shell .camera-prewarm-live
            ) div.st-key-camera_webrtc_host video {{
                width:100% !important;
                height:100% !important;
                min-width:0 !important;
                min-height:0 !important;
                max-width:100% !important;
                max-height:100% !important;
                display:block !important;
                object-fit:contain !important;
                object-position:center center !important;
                background:#000000 !important;
            }}
        }}

        /* ==================================================
           CAMERA TEXT
           ================================================== */

        .camera-page-title {{
            color:#174a29 !important;

            font-size:25px !important;

            font-weight:750 !important;

            line-height:1.25 !important;

            margin:0 !important;
        }}

        .camera-page-subtitle {{
            color:#728078 !important;

            font-size:13px !important;

            margin:
                5px 0 14px
                !important;
        }}

        .camera-info,
        .camera-guide {{
            background:
                linear-gradient(
                    135deg,
                    #effaf1 0%,
                    #e4f6e8 100%
                )
                !important;

            border:
                1px solid #cee7d4
                !important;

            border-radius:
                11px
                !important;

            padding:
                12px 14px
                !important;

            margin:
                10px 0 12px
                !important;

            color:
                #57655c
                !important;

            font-size:
                12px
                !important;

            line-height:
                1.65
                !important;

            box-shadow:
                0 2px 9px
                rgba(39,111,53,.035)
                !important;
        }}

        .camera-guide {{
            margin-top:
                12px
                !important;
        }}

        /* ==================================================
           DASHBOARD HERO
           ================================================== */

        .hero-dashboard {{
            position:relative !important;

            overflow:hidden !important;

            background:
                linear-gradient(
                    135deg,
                    #ffffff 0%,
                    #f0faf3 52%,
                    #dcf2e1 100%
                )
                !important;

            border:
                1px solid #c8e3cf
                !important;

            border-radius:
                20px
                !important;

            padding:
                30px
                !important;

            margin:
                0 0 22px
                !important;

            box-shadow:
                0 7px 24px
                rgba(38,112,55,.07)
                !important;
        }}

        .hero-dashboard::after {{
            content:"" !important;

            position:absolute !important;

            width:
                220px
                !important;

            height:
                220px
                !important;

            right:
                -95px
                !important;

            top:
                -110px
                !important;

            border-radius:
                50%
                !important;

            background:
                rgba(76,175,80,.11)
                !important;

            pointer-events:none !important;
        }}

        .hero-kicker {{
            display:inline-block !important;

            background:
                #daf1df
                !important;

            color:
                #25713b
                !important;

            border:
                1px solid #c0e1c7
                !important;

            border-radius:
                999px
                !important;

            padding:
                5px 11px
                !important;

            font-size:
                10px
                !important;

            font-weight:
                800
                !important;

            letter-spacing:
                .3px
                !important;

            margin:
                0 0 10px
                !important;
        }}

        .hero-title {{
            color:
                #173f27
                !important;

            font-size:
                25px
                !important;

            font-weight:
                750
                !important;

            line-height:
                1.3
                !important;

            margin:
                0 0 8px
                !important;
        }}

        .hero-text {{
            color:
                #69776f
                !important;

            font-size:
                13px
                !important;

            line-height:
                1.7
                !important;

            margin:
                0
                !important;

            max-width:
                820px
                !important;
        }}

        /* ==================================================
           SECTION
           ================================================== */

        .section-heading {{
            display:flex !important;

            align-items:center !important;

            gap:
                9px
                !important;

            color:
                #205a33
                !important;

            font-size:
                15px
                !important;

            font-weight:
                750
                !important;

            margin:
                3px 0 13px 2px
                !important;
        }}

        .section-heading .dot {{
            width:
                8px
                !important;

            height:
                8px
                !important;

            flex:
                0 0 8px
                !important;

            border-radius:
                50%
                !important;

            background:
                #45ad50
                !important;

            box-shadow:
                0 0 0 4px
                #daf2de
                !important;
        }}

        /* ==================================================
           METRIC
           ================================================== */

        .metric-card {{
            position:relative !important;

            overflow:hidden !important;

            background:
                #ffffff
                !important;

            min-height:
                160px
                !important;

            padding:
                22px 19px
                !important;

            border:
                1px solid #d3e7d8
                !important;

            border-radius:
                16px
                !important;

            display:flex !important;

            flex-direction:
                column
                !important;

            align-items:
                center
                !important;

            justify-content:
                center
                !important;

            box-shadow:
                0 4px 14px
                rgba(38,112,55,.055)
                !important;

            transition:
                transform .16s ease,
                box-shadow .16s ease,
                border-color .16s ease
                !important;
        }}

        .metric-card:hover {{
            transform:
                translateY(-3px)
                !important;

            border-color:
                #b8d9bf
                !important;

            box-shadow:
                0 8px 22px
                rgba(38,112,55,.10)
                !important;
        }}

        .metric-card::before {{
            content:"" !important;

            position:absolute !important;

            top:0 !important;
            left:0 !important;
            right:0 !important;

            height:
                4px
                !important;

            background:
                linear-gradient(
                    90deg,
                    #44aa50,
                    #7bc984
                )
                !important;
        }}

        .metric-icon {{
            width:
                48px
                !important;

            height:
                48px
                !important;

            display:flex !important;

            align-items:center !important;
            justify-content:center !important;

            border-radius:
                13px
                !important;

            background:
                #e2f5e6
                !important;

            margin-bottom:
                9px
                !important;
        }}

        .metric-card .icon {{
            font-size:
                27px
                !important;
        }}

        .metric-card .value {{
            font-size:
                31px
                !important;

            font-weight:
                750
                !important;

            color:
                #194c2c
                !important;

            line-height:
                1.1
                !important;
        }}

        .metric-card .label {{
            margin-top:
                7px
                !important;

            font-size:
                12px
                !important;

            color:
                #78847d
                !important;

            font-weight:
                600
                !important;
        }}

        /* ==================================================
           CONTENT CARDS
           ================================================== */

        .dashboard-card,
        .status-panel {{
            background:
                #ffffff
                !important;

            border:
                1px solid #d4e8d9
                !important;

            border-radius:
                16px
                !important;

            box-shadow:
                0 4px 14px
                rgba(38,112,55,.05)
                !important;

            padding:
                26px 28px
                !important;
        }}

        .status-panel {{
            background:
                linear-gradient(
                    180deg,
                    #fbfefb 0%,
                    #edf8ef 100%
                )
                !important;

            border-color:
                #cce5d2
                !important;
        }}

        .dashboard-card h3,
        .status-panel-title {{
            color:
                #194c2c
                !important;

            font-size:
                17px
                !important;

            font-weight:
                750
                !important;

            margin:
                0 0 14px
                !important;
        }}

        .dashboard-card p,
        .dashboard-card li,
        .status-row {{
            color:
                #68756d
                !important;

            font-size:
                13px
                !important;

            line-height:
                1.7
                !important;
        }}

        .dashboard-card ul {{
            list-style:
                none
                !important;

            padding:
                0
                !important;

            margin:
                15px 0 0
                !important;
        }}

        .dashboard-card li {{
            padding:
                8px 0
                !important;

            border-bottom:
                1px solid #ebf1ed
                !important;
        }}

        .status-row {{
            display:flex !important;

            align-items:center !important;

            justify-content:
                space-between
                !important;

            padding:
                9px 0
                !important;

            border-bottom:
                1px solid #e0ebe3
                !important;
        }}

        .status-value {{
            color:
                #28733e
                !important;

            font-weight:
                750
                !important;
        }}

        .status-warning {{
            color:
                #b85c00
                !important;
        }}

        .status-footer {{
            margin-top:
                15px
                !important;

            padding:
                10px 12px
                !important;

            background:
                #e3f5e7
                !important;

            border-radius:
                9px
                !important;

            color:
                #2e6840
                !important;

            font-size:
                11px
                !important;
        }}

        .system-note {{
            background:
                #edf8f0
                !important;

            border:
                1px solid #d4eada
                !important;

            border-radius:
                10px
                !important;

            padding:
                12px 14px
                !important;

            margin-top:
                17px
                !important;

            color:
                #59665e
                !important;

            font-size:
                12px
                !important;

            line-height:
                1.65
                !important;
        }}

        /* ==================================================
           LIVE LOCATION
           ================================================== */

        .location-section-heading {{
            margin-bottom:13px !important;
        }}

        /* ==================================================
           RESPONSIVE
           ================================================== */

        @media(max-width:900px) {{

            [data-testid="stMainBlockContainer"] {{
                width:100% !important;

                margin-left:0 !important;

                padding:
                    76px 20px 32px
                    !important;
            }}

            [data-testid="stSidebar"] {{
                width:210px !important;

                min-width:210px !important;

                max-width:210px !important;
            }}

            .hero-title {{
                font-size:
                    23px
                    !important;
            }}

            .metric-card {{
                min-height:
                    140px
                    !important;
            }}
        }}

        @media(max-width:640px) {{

            [data-testid="stMainBlockContainer"] {{
                padding:
                    72px 14px 28px
                    !important;
            }}

            [data-testid="stSidebar"] {{
                width:205px !important;

                min-width:205px !important;

                max-width:205px !important;
            }}

            .sidebar-brand {{
                padding:
                    0 5px
                    !important;
            }}

            .hero-dashboard {{
                padding:
                    22px 20px
                    !important;
            }}

            .hero-title {{
                font-size:
                    21px
                    !important;
            }}

            .dashboard-card,
            .status-panel {{
                padding:
                    20px
                    !important;
            }}
        }}

        /* ==================================================
           ADDITIONAL RESPONSIVE SAFETY
           ==================================================
           This block only controls responsive layout. It does not change
           recognition logic, geofence logic, or camera lifecycle.
        */

        [data-testid="stMainBlockContainer"],
        [data-testid="stMainBlockContainer"] > div,
        [data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"],
        [data-testid="stMainBlockContainer"] [data-testid="column"] {{
            min-width:0 !important;
            box-sizing:border-box !important;
        }}

        .hero-dashboard,
        .dashboard-card,
        .status-panel,
        .camera-info,
        .camera-guide {{
            max-width:100% !important;
            overflow-wrap:anywhere !important;
        }}

        .camera-info,
        .camera-guide {{
            width:100% !important;
        }}

        div.st-key-camera_controls {{
            width:100% !important;
            max-width:100% !important;
            margin:10px 0 0 !important;
            padding:0 !important;
        }}

        div.st-key-camera_controls [data-testid="stHorizontalBlock"] {{
            width:100% !important;
            min-width:0 !important;
            gap:10px !important;
        }}

        div.st-key-camera_controls [data-testid="column"] {{
            min-width:0 !important;
            padding:0 !important;
        }}

        div.st-key-camera_controls button {{
            min-height:44px !important;
            width:100% !important;
            white-space:nowrap !important;
        }}

        @media (max-width:900px) {{
            [data-testid="stMainBlockContainer"] {{
                padding-left:20px !important;
                padding-right:20px !important;
            }}

            .hero-dashboard {{
                width:100% !important;
            }}

            .section-heading {{
                max-width:100% !important;
                overflow-wrap:anywhere !important;
            }}

            div.st-key-camera_prewarm_shell {{
                max-width:100% !important;
            }}
        }}

        @media (max-width:640px) {{
            [data-testid="stMainBlockContainer"] {{
                width:100% !important;
                max-width:100% !important;
                padding:72px 14px 28px !important;
                margin:0 !important;
            }}

            .hero-dashboard {{
                padding:18px 16px !important;
                border-radius:14px !important;
            }}

            .hero-kicker {{
                font-size:9px !important;
                letter-spacing:.7px !important;
            }}

            .hero-title {{
                font-size:20px !important;
                line-height:1.3 !important;
                margin-top:5px !important;
            }}

            .hero-text {{
                font-size:12px !important;
                line-height:1.55 !important;
                margin-top:7px !important;
            }}

            .section-heading {{
                font-size:13px !important;
                line-height:1.35 !important;
            }}

            .section-heading .dot {{
                flex:0 0 auto !important;
            }}

            .dashboard-card,
            .status-panel {{
                padding:16px !important;
                border-radius:12px !important;
            }}

            [data-testid="stSidebar"] {{
                width:min(82vw, 280px) !important;
                min-width:min(82vw, 280px) !important;
                max-width:min(82vw, 280px) !important;
                box-shadow:6px 0 22px rgba(0,0,0,.10) !important;
            }}

            [data-testid="stSidebarContent"] {{
                padding:20px 12px 18px !important;
                overflow-y:auto !important;
                -webkit-overflow-scrolling:touch !important;
            }}

            .sidebar-brand-title {{
                font-size:12px !important;
            }}

            .sidebar-brand-subtitle {{
                font-size:9px !important;
            }}

            [data-testid="stSidebar"] .stButton > button {{
                height:46px !important;
                min-height:46px !important;
                font-size:13px !important;
                border-radius:11px !important;
            }}

            div.st-key-sidebar_toggle_container {{
                top:7px !important;
                left:7px !important;
                width:42px !important;
                height:42px !important;
            }}

            div.st-key-sidebar_toggle_container .stButton,
            div.st-key-sidebar_toggle_container button {{
                width:42px !important;
                height:42px !important;
                min-height:42px !important;
            }}

            /* Mobile START/STOP controls: always stack vertically. */
            div.st-key-camera_controls [data-testid="stHorizontalBlock"] {{
                display:grid !important;
                grid-template-columns:minmax(0, 1fr) !important;
                width:100% !important;
                gap:9px !important;
            }}

            div.st-key-camera_controls [data-testid="column"] {{
                width:100% !important;
                min-width:0 !important;
                flex:none !important;
            }}

            div.st-key-camera_controls button {{
                height:46px !important;
                min-height:46px !important;
                font-size:13px !important;
            }}

            /* MOBILE LIVE CAMERA ONLY. Desktop CSS remains unchanged. */
            [data-testid="stElementContainer"]:has(
                div.st-key-camera_prewarm_shell .camera-prewarm-live
            ) div.st-key-camera_prewarm_shell {{
                width:min(calc(100vw - 28px), 340px) !important;
                max-width:340px !important;
                aspect-ratio:4 / 3 !important;
                height:auto !important;
                min-height:0 !important;
                max-height:none !important;
                margin:10px auto 14px !important;
                overflow:hidden !important;
            }}

            [data-testid="stElementContainer"]:has(
                div.st-key-camera_prewarm_shell .camera-prewarm-live
            ) div.st-key-camera_webrtc_host,
            [data-testid="stElementContainer"]:has(
                div.st-key-camera_prewarm_shell .camera-prewarm-live
            ) div.st-key-camera_webrtc_host [data-testid="stCustomComponentV1"],
            [data-testid="stElementContainer"]:has(
                div.st-key-camera_prewarm_shell .camera-prewarm-live
            ) div.st-key-camera_webrtc_host iframe,
            [data-testid="stElementContainer"]:has(
                div.st-key-camera_prewarm_shell .camera-prewarm-live
            ) div.st-key-camera_webrtc_host video {{
                width:100% !important;
                height:100% !important;
                max-width:100% !important;
                max-height:100% !important;
                min-width:0 !important;
                min-height:0 !important;
                display:block !important;
                margin:0 !important;
                padding:0 !important;
            }}

            [data-testid="stElementContainer"]:has(
                div.st-key-camera_prewarm_shell .camera-prewarm-live
            ) div.st-key-camera_webrtc_host video {{
                object-fit:contain !important;
                object-position:center center !important;
                transform:none !important;
                transform-origin:center center !important;
            }}

            /* Camera text must never create horizontal overflow. */
            .camera-info,
            .camera-guide {{
                font-size:11px !important;
                line-height:1.55 !important;
            }}

            /* Native select is full-width and touch-friendly. */
            [data-testid="stCustomComponentV1"],
            [data-testid="stCustomComponentV1"] iframe {{
                max-width:100% !important;
            }}
        }}

        /* ==================================================
           MOBILE UI SAFETY
           ================================================== */

        button,
        input,
        select,
        textarea {{
            max-width:100% !important;
        }}

        img,
        svg,
        video,
        iframe {{
            max-width:100% !important;
        }}

        /* ==================================================
           DESKTOP ATTENDANCE DROPDOWN / MAP IFRAME
           ==================================================
           Desktop only: enlarge the map iframe so the dropdown can
           extend downward outside the location card.
           Mobile keeps the original 580px iframe height.
        */
        @media (min-width:701px) {{
            div.st-key-location_map_iframe {{
                width:100% !important;
                max-width:100% !important;
                overflow:visible !important;
            }}

            div.st-key-location_map_iframe iframe {{
                width:100% !important;
                min-height:760px !important;
                height:760px !important;
                max-height:760px !important;
                border:0 !important;
            }}
        }}

        @media (max-width:700px) {{
            div.st-key-location_map_iframe {{
                width:100% !important;
                max-width:100% !important;
            }}

            div.st-key-location_map_iframe iframe {{
                min-height:580px !important;
                height:580px !important;
                max-height:580px !important;
            }}
        }}

        @media (max-width:640px) {{
            .status-row {{
                gap:8px !important;
                flex-wrap:wrap !important;
            }}

            .status-row > * {{
                min-width:0 !important;
            }}

            .camera-info,
            .camera-guide,
            .system-note {{
                overflow-wrap:anywhere !important;
                word-break:break-word !important;
            }}
        }}

        @media (max-width:380px) {{
            [data-testid="stMainBlockContainer"] {{
                padding-left:10px !important;
                padding-right:10px !important;
            }}

            .hero-dashboard {{
                padding-left:13px !important;
                padding-right:13px !important;
            }}

            div.st-key-camera_prewarm_shell {{
                border-radius:10px !important;
            }}

            [data-testid="stElementContainer"]:has(
                div.st-key-camera_prewarm_shell .camera-prewarm-live
            ) div.st-key-camera_prewarm_shell {{
                width:100% !important;
                max-width:360px !important;
            }}
        }}
        </style>
        """
    )


# ==========================================================
# DASHBOARD VIEW
# ==========================================================

def show_dashboard():
    """Dashboard with original visual order and early WebRTC prewarm.

    The old attendance summary and lower information/status panels are intentionally
    removed from the dashboard. Live GPS remains client-side, while the selected
    attendance location is geocoded in the browser and shown as a red marker/circle.
    """

    render_html(
        """
        <div class="hero-dashboard">

            <div class="hero-kicker">
                ● HR ATTENDANCE SYSTEM
            </div>

            <div class="hero-title">
                Selamat datang di Dashboard 👋
            </div>

            <p class="hero-text">
                Pantau kehadiran karyawan dan
                aktivitas face recognition
                dalam satu tampilan yang sederhana,
                rapi, dan mudah dipantau.
            </p>

        </div>
        """
    )

    # ==========================================================
    # ORIGINAL VISUAL ORDER
    # ==========================================================
    # Live Location Tracking stays first. Live Camera Detection
    # stays directly below the location/map section.
    render_html(
        """
        <div class="section-heading location-section-heading">
            <span class="dot"></span>
            Live Location Tracking
        </div>
        """
    )

    # All map/dropdown interaction stays inside the browser iframe.
    # The section heading above is rendered only once.
    # Keep the complete map UI inside a trusted iframe.
    # Leaflet + geolocation run entirely in the browser.
    map_html = r"""
            <!DOCTYPE html>
            <html lang="id">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">

                <link
                    rel="stylesheet"
                    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
                    crossorigin=""
                >

                <style>
                    * {
                        box-sizing: border-box;
                    }

                    html,
                    body {
                        margin: 0;
                        padding: 0;
                        background: transparent;
                        font-family: Arial, sans-serif;
                    }

                    body {
                        overflow: auto;
                    }

                    .location-card {
                        width: 100%;
                        background: #ffffff;
                        border: 1px solid #d4e8d9;
                        border-radius: 16px;
                        box-shadow: 0 4px 14px rgba(38,112,55,.05);
                        overflow: hidden;
                    }

                    /* Desktop only: allow the custom dropdown to extend below
                       the location card instead of being clipped by it. */
                    @media (min-width: 701px) {
                        .location-card {
                            overflow: visible;
                        }

                        .attendance-dropdown {
                            z-index: 9999;
                        }

                        .attendance-dropdown-menu {
                            z-index: 99999;
                            max-height: 360px;
                        }
                    }

                    .location-top {
                        min-height: 70px;
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                        gap: 12px;
                        padding: 16px 22px;
                    }

                    .location-title {
                        display: flex;
                        align-items: center;
                        gap: 10px;
                        color: #194c2c;
                        font-size: 20px;
                        font-weight: 750;
                    }

                    .location-target {
                        width: 34px;
                        height: 34px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        border-radius: 10px;
                        background: #e3f5e7;
                        color: #28733e;
                        font-size: 18px;
                        flex: 0 0 34px;
                    }

                    .live-badge {
                        display: inline-flex;
                        align-items: center;
                        gap: 6px;
                        border: 1px solid #cce5d2;
                        border-radius: 999px;
                        padding: 6px 11px;
                        color: #2b7c41;
                        background: #effaf1;
                        font-size: 11px;
                        font-weight: 800;
                        letter-spacing: .5px;
                    }

                    .live-dot {
                        width: 7px;
                        height: 7px;
                        border-radius: 50%;
                        background: #45ad50;
                        box-shadow: 0 0 0 4px #daf2de;
                    }

                    #map {
                        width: 100%;
                        height: 365px;
                        background: #edf4ef;
                    }

                    .location-form {
                        padding: 16px 22px 20px;
                        background: #ffffff;
                    }

                    .location-label {
                        display: block;
                        margin-bottom: 8px;
                        color: #1f5130;
                        font-size: 15px;
                        font-weight: 750;
                    }

                    /* ------------------------------------------------------
                       RESPONSIVE CUSTOM ATTENDANCE LOCATION DROPDOWN
                       The native <select> is kept hidden only as a compatibility
                       value holder. The visible control is fully HTML/CSS/JS
                       and therefore stays inside the iframe/card on mobile.
                       ------------------------------------------------------ */

                    .attendance-dropdown {
                        position: relative;
                        width: 100%;
                        z-index: 1000;
                    }

                    .attendance-select-hidden {
                        position: absolute !important;
                        width: 1px !important;
                        height: 1px !important;
                        opacity: 0 !important;
                        pointer-events: none !important;
                        overflow: hidden !important;
                    }

                    .attendance-dropdown-trigger {
                        width: 100%;
                        min-height: 46px;
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                        gap: 10px;
                        border: 1px solid #dbe7ed;
                        border-radius: 9px;
                        padding: 0 13px 0 14px;
                        background: #f6f9fb;
                        color: #6f7c86;
                        font-size: 15px;
                        font-family: Arial, sans-serif;
                        text-align: left;
                        cursor: pointer;
                        outline: none;
                        box-shadow: none;
                        touch-action: manipulation;
                    }

                    .attendance-dropdown-trigger:hover {
                        border-color: #b9d9c0;
                        background: #f8fbf9;
                    }

                    .attendance-dropdown.open .attendance-dropdown-trigger,
                    .attendance-dropdown-trigger:focus-visible {
                        border-color: #8bc997;
                        box-shadow: 0 0 0 3px rgba(69,173,80,.10);
                    }

                    .attendance-dropdown-value {
                        flex: 1 1 auto;
                        min-width: 0;
                        overflow: hidden;
                        white-space: nowrap;
                        text-overflow: ellipsis;
                    }

                    .attendance-dropdown-arrow {
                        flex: 0 0 auto;
                        font-size: 14px;
                        color: #6b7c72;
                        transition: transform .16s ease;
                    }

                    .attendance-dropdown.open .attendance-dropdown-arrow {
                        transform: rotate(180deg);
                    }

                    .attendance-dropdown-menu {
                        position: absolute;
                        left: 0;
                        right: 0;
                        top: calc(100% + 6px);
                        width: 100%;
                        max-height: min(250px, 42vh);
                        overflow-y: auto;
                        -webkit-overflow-scrolling: touch;
                        overscroll-behavior: contain;
                        padding: 5px;
                        border: 1px solid #d9e7dd;
                        border-radius: 10px;
                        background: #ffffff;
                        box-shadow: 0 10px 28px rgba(27,68,38,.16);
                        display: none;
                        z-index: 5000;
                    }

                    .attendance-dropdown.open .attendance-dropdown-menu {
                        display: block;
                    }

                    .attendance-dropdown-option {
                        width: 100%;
                        min-height: 42px;
                        display: flex;
                        align-items: center;
                        padding: 9px 11px;
                        border: 0;
                        border-radius: 8px;
                        background: transparent;
                        color: #25313a;
                        font-size: 14px;
                        line-height: 1.3;
                        text-align: left;
                        cursor: pointer;
                    }

                    .attendance-dropdown-option:hover,
                    .attendance-dropdown-option:focus-visible {
                        background: #edf8ef;
                        color: #236f37;
                        outline: none;
                    }

                    .attendance-dropdown-option.selected {
                        background: #e5f4e8;
                        color: #236f37;
                        font-weight: 700;
                    }

                    @media (max-width: 480px) {
                        .attendance-dropdown-trigger {
                            min-height: 48px;
                            height: 48px;
                            border-radius: 10px;
                            padding: 0 12px;
                            font-size: 16px;
                        }

                        .attendance-dropdown-menu {
                            top: calc(100% + 5px);
                            max-height: min(245px, 45vh);
                            padding: 4px;
                            border-radius: 10px;
                        }

                        .attendance-dropdown-option {
                            min-height: 44px;
                            padding: 10px 11px;
                            font-size: 16px;
                        }
                    }

                    @media (orientation: landscape) and (max-width: 900px) {
                        .attendance-dropdown-menu {
                            max-height: min(220px, 60vh);
                        }
                    }

                    .geofence-note {
                        margin-top: 10px;
                        padding: 9px 11px;
                        border-radius: 9px;
                        background: #fff7f7;
                        border: 1px solid #fee2e2;
                        color: #7f1d1d;
                        font-size: 12px;
                        line-height: 1.5;
                    }

                    .leaflet-popup-content {
                        margin: 11px 14px;
                        font-family: Arial, sans-serif;
                    }

                    @media(max-width:700px) {
                        #map {
                            height: 320px;
                        }

                        .location-top,
                        .location-form {
                            padding-left: 16px;
                            padding-right: 16px;
                        }

                        .location-title {
                            font-size: 17px;
                        }
                    }
                    @media(max-width:480px) {
                        .location-card {
                            border-radius: 13px;
                        }

                        .location-top {
                            min-height: 64px;
                            padding: 13px 14px;
                            flex-wrap: wrap;
                            row-gap: 9px;
                        }

                        .location-title {
                            font-size: 16px;
                            gap: 8px;
                            min-width: 0;
                        }

                        .location-target {
                            width: 32px;
                            height: 32px;
                            flex: 0 0 32px;
                        }

                        .live-badge {
                            font-size: 10px;
                            padding: 5px 9px;
                        }

                        #map {
                            height: clamp(260px, 70vw, 320px);
                        }

                        .location-form {
                            padding: 14px;
                        }

                        .location-label {
                            font-size: 14px;
                            margin-bottom: 7px;
                        }

                        #attendanceLocation {
                            height: 48px;
                            min-height: 48px;
                            padding: 0 38px 0 12px;
                            font-size: 16px;
                            border-radius: 10px;
                        }

                        #attendanceLocation option {
                            font-size: 16px;
                        }

                        .geofence-note {
                            font-size: 11px;
                            padding: 8px 10px;
                        }

                        .leaflet-popup-content {
                            max-width: calc(100vw - 80px);
                            margin: 9px 10px;
                            font-size: 12px;
                        }
                    }

                    @media (orientation: landscape) and (max-width: 900px) {
                        #map {
                            height: clamp(240px, 58vh, 340px);
                        }
                    }
                </style>
            </head>

            <body>
                <div class="location-card">

                    <div class="location-top">
                        <div class="location-title">
                            <span class="location-target">⌖</span>
                            <span>Lokasi Perangkat</span>
                        </div>

                        <div class="live-badge">
                            <span class="live-dot"></span>
                            LIVE
                        </div>
                    </div>

                    <div id="map"></div>

                    <div class="location-form">

                        <label class="location-label" for="attendanceLocation">
                            Pilih Lokasi Absen
                        </label>

                        <!-- Hidden native select kept only as the existing value/event bridge. -->
                        <select id="attendanceLocation" class="attendance-select-hidden" aria-hidden="true" tabindex="-1">
                            <option value="">-- Pilih lokasi absen --</option>

                            <option value="Klinik Utama Medialys">Klinik Utama Medialys</option>
                            <option value="Mandiri DM 12">Mandiri DM 12</option>
                            <option value="Medika DM 9">Medika DM 9</option>
                            <option value="ISS DM 15">ISS DM 15</option>
                            <option value="Klinik Medilab Pekalongan">Klinik Medilab Pekalongan</option>
                            <option value="Treasury Office Tower">Treasury Office Tower</option>
                            <option value="IMS">IMS</option>
                            <option value="Sonic Service Center">Sonic Service Center</option>
                            <option value="Kamal">Kamal</option>
                            <option value="Logistik Medika DM 11">Logistik Medika DM 11</option>
                            <option value="ITS Green Lake">ITS Green Lake</option>
                            <option value="Pabrik Cikande">Pabrik Cikande</option>
                        </select>

                        <div class="attendance-dropdown" id="attendanceDropdown">
                            <button
                                type="button"
                                class="attendance-dropdown-trigger"
                                id="attendanceDropdownTrigger"
                                aria-haspopup="listbox"
                                aria-expanded="false"
                            >
                                <span class="attendance-dropdown-value" id="attendanceDropdownValue">
                                    -- Pilih lokasi absen --
                                </span>
                                <span class="attendance-dropdown-arrow" aria-hidden="true">⌄</span>
                            </button>

                            <div
                                class="attendance-dropdown-menu"
                                id="attendanceDropdownMenu"
                                role="listbox"
                                aria-label="Daftar lokasi absen"
                            >
                                <button type="button" class="attendance-dropdown-option" data-value="">
                                    -- Pilih lokasi absen --
                                </button>
                                <button type="button" class="attendance-dropdown-option" data-value="Klinik Utama Medialys">
                                    Klinik Utama Medialys
                                </button>
                                <button type="button" class="attendance-dropdown-option" data-value="Mandiri DM 12">
                                    Mandiri DM 12
                                </button>
                                <button type="button" class="attendance-dropdown-option" data-value="Medika DM 9">
                                    Medika DM 9
                                </button>
                                <button type="button" class="attendance-dropdown-option" data-value="ISS DM 15">
                                    ISS DM 15
                                </button>
                                <button type="button" class="attendance-dropdown-option" data-value="Klinik Medilab Pekalongan">
                                    Klinik Medilab Pekalongan
                                </button>
                                <button type="button" class="attendance-dropdown-option" data-value="Treasury Office Tower">
                                    Treasury Office Tower
                                </button>
                                <button type="button" class="attendance-dropdown-option" data-value="IMS">
                                    IMS
                                </button>
                                <button type="button" class="attendance-dropdown-option" data-value="Sonic Service Center">
                                    Sonic Service Center
                                </button>
                                <button type="button" class="attendance-dropdown-option" data-value="Kamal">
                                    Kamal
                                </button>
                                <button type="button" class="attendance-dropdown-option" data-value="Logistik Medika DM 11">
                                    Logistik Medika DM 11
                                </button>
                                <button type="button" class="attendance-dropdown-option" data-value="ITS Green Lake">
                                    ITS Green Lake
                                </button>
                                <button type="button" class="attendance-dropdown-option" data-value="Pabrik Cikande">
                                    Pabrik Cikande
                                </button>
                            </div>
                        </div>

                        <div
                            id="geofenceStatus"
                            style="
                                margin-top:12px;
                                font-size:14px;
                                font-weight:600;
                                color:#64748b;
                            "
                        >
                            🔵 Pilih lokasi absensi
                        </div>

                        <div
                            id="geofenceDistance"
                            style="
                                margin-top:4px;
                                font-size:13px;
                                color:#64748b;
                            "
                        >
                            Radius zona absensi: 60 meter
                        </div>

                        <div class="geofence-note">
                            🔴 Lingkaran merah = zona absensi dengan radius tepat
                            <b>60 meter</b> dari lokasi yang dipilih.
                        </div>

                        <div style="margin-top:10px;">
                            <a
                                id="googleMapsLocationLink"
                                href="#"
                                target="_blank"
                                rel="noopener noreferrer"
                                style="
                                    display:inline-block;
                                    padding:7px 10px;
                                    border-radius:8px;
                                    background:#f1f5f9;
                                    color:#334155;
                                    font-size:12px;
                                    text-decoration:none;
                                    border:1px solid #dbe4ea;
                                "
                            >
                                📍 Buka lokasi di Google Maps
                            </a>
                        </div>

                    </div>
                </div>

                <!-- Leaflet JS -->
                <script
                    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
                    crossorigin=""
                ></script>

                <script>
                    // ==========================================================
                    // MAP CONFIGURATION
                    // ==========================================================

                    const GEOFENCE_RADIUS_METERS = 7000;

                    /*
                     * IMPORTANT:
                     * Both locations below have FIXED coordinates.
                     * No geocoding API is required.
                     */

                    const ATTENDANCE_LOCATIONS = {
                        "Klinik Utama Medialys": {
                            lat: -6.174145501357489,
                            lng: 106.85686667301668,
                            address:
                                "504D, Jl. Letjen Suprapto No. 504C, " +
                                "Cemp. Putih Bar., Kec. Johar Baru, Jakarta 10530"
                        },

                        "Mandiri DM 12": {
                            lat: -6.152604321020155,
                            lng: 106.6940247365544,
                            address: "Mandiri DM 12"
                        },

                        "Medika DM 9": {
                            lat: -6.153692357716074,
                            lng: 106.69404619422662,
                            address: "Medika DM 9"
                        },

                        "ISS DM 15": {
                            lat: -6.151194,
                            lng: 106.695256,
                            address: "ISS DM 15"
                        }
                    };

                    const DEFAULT_CENTER = [
                        -6.174145501357489,
                        106.85686667301668
                    ];

                    // ==========================================================
                    // MAP CREATION
                    // ==========================================================

                    let map = null;
                    let userMarker = null;
                    let userWatchId = null;

                    let attendanceMarker = null;
                    let attendanceCircle = null;
                    let selectedAttendancePoint = null;
                    let userCentered = false;

                    function initMap() {

                        if (typeof L === "undefined") {
                            console.error(
                                "Leaflet failed to load from CDN."
                            );

                            const mapElement =
                                document.getElementById("map");

                            if (mapElement) {
                                mapElement.innerHTML =
                                    "<div style='" +
                                    "padding:30px;" +
                                    "text-align:center;" +
                                    "color:#b91c1c;" +
                                    "font-family:Arial,sans-serif;" +
                                    "'>" +
                                    "<b>Map gagal dimuat.</b><br>" +
                                    "Periksa koneksi internet / akses CDN Leaflet." +
                                    "</div>";
                            }

                            return false;
                        }

                        map = L.map(
                            "map",
                            {
                                zoomControl: true,
                                attributionControl: true,
                                preferCanvas: true
                            }
                        ).setView(
                            DEFAULT_CENTER,
                            17
                        );

                        L.tileLayer(
                            "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                            {
                                maxZoom: 19,
                                detectRetina: true,
                                attribution:
                                    "&copy; OpenStreetMap contributors"
                            }
                        ).addTo(map);

                        setTimeout(
                            function () {
                                map.invalidateSize(true);
                            },
                            250
                        );

                        setTimeout(
                            function () {
                                map.invalidateSize(true);
                            },
                            800
                        );

                        return true;
                    }

                    // ==========================================================
                    // USER ICON
                    // ==========================================================

                    function userIcon() {

                        return L.icon({
                            iconUrl:
                                "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",

                            iconRetinaUrl:
                                "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",

                            shadowUrl:
                                "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",

                            iconSize: [25, 41],
                            iconAnchor: [12, 41],
                            popupAnchor: [1, -34],
                            shadowSize: [41, 41]
                        });
                    }

                    // ==========================================================
                    // HAVERSINE DISTANCE
                    // ==========================================================

                    function calculateDistanceMeters(
                        lat1,
                        lon1,
                        lat2,
                        lon2
                    ) {

                        const R = 6371000;

                        const dLat =
                            (lat2 - lat1) *
                            Math.PI / 180;

                        const dLon =
                            (lon2 - lon1) *
                            Math.PI / 180;

                        const a =
                            Math.sin(dLat / 2) *
                            Math.sin(dLat / 2) +

                            Math.cos(
                                lat1 * Math.PI / 180
                            ) *

                            Math.cos(
                                lat2 * Math.PI / 180
                            ) *

                            Math.sin(dLon / 2) *
                            Math.sin(dLon / 2);

                        const c =
                            2 *
                            Math.atan2(
                                Math.sqrt(a),
                                Math.sqrt(1 - a)
                            );

                        return R * c;
                    }

                    // ==========================================================
                    // GEOFENCE STATUS
                    // ==========================================================

                    // ==========================================================
                    // CAMERA START GATE
                    // ==========================================================
                    // The map remains inside the original iframe. Because this iframe is
                    // same-origin with the Streamlit app, it can update the existing
                    // Streamlit START button directly without replacing the map UI.
                    let geofenceAllowsCamera = false;

                    function setCameraStartAccess(allowed) {

                        geofenceAllowsCamera = Boolean(allowed);

                        try {
                            const parentDocument = window.parent.document;
                            const buttons = parentDocument.querySelectorAll("button");

                            buttons.forEach(function(button) {
                                const text = (button.innerText || "").trim();

                                if (text === "START") {
                                    button.disabled = !geofenceAllowsCamera;
                                    button.setAttribute(
                                        "aria-disabled",
                                        geofenceAllowsCamera ? "false" : "true"
                                    );

                                    if (geofenceAllowsCamera) {
                                        button.title = "Webcam dapat dimulai di dalam zona absensi.";
                                    } else {
                                        button.title = "START terkunci. Anda harus berada di dalam zona merah.";
                                    }
                                }
                            });
                        } catch (error) {
                            console.warn("Unable to update camera START button:", error);
                        }
                    }

                    function updateGeofenceStatus(
                        distance
                    ) {

                        const statusElement =
                            document.getElementById(
                                "geofenceStatus"
                            );

                        const distanceElement =
                            document.getElementById(
                                "geofenceDistance"
                            );

                        if (
                            !statusElement ||
                            !distanceElement
                        ) {
                            return;
                        }

                        const roundedDistance =
                            Math.round(distance);

                        distanceElement.textContent =
                            "Jarak dari lokasi absen: " +
                            roundedDistance +
                            " meter";

                        const allowed =
                            distance <=
                            GEOFENCE_RADIUS_METERS;

                        setCameraStartAccess(allowed);

                        if (allowed) {

                            statusElement.innerHTML =
                                "🟢 <b>Dalam zona absensi</b>";

                            statusElement.style.color =
                                "#15803d";

                        } else {

                            statusElement.innerHTML =
                                "🔴 <b>Di luar zona absensi</b>";

                            statusElement.style.color =
                                "#dc2626";
                        }
                    }

                    // ==========================================================
                    // USER LOCATION
                    // ==========================================================

                    function updateUserPosition(
                        position
                    ) {

                        const latitude =
                            position.coords.latitude;

                        const longitude =
                            position.coords.longitude;

                        const point = [
                            latitude,
                            longitude
                        ];

                        if (!userMarker) {

                            userMarker =
                                L.marker(
                                    point,
                                    {
                                        icon: userIcon(),
                                        zIndexOffset: 1000
                                    }
                                )
                                .addTo(map)
                                .bindPopup(
                                    "<b>You are here</b>"
                                );

                        } else {

                            userMarker.setLatLng(
                                point
                            );
                        }

                        /*
                         * Center on the user only on the
                         * first GPS fix, before a location
                         * is selected.
                         */
                        if (
                            !userCentered &&
                            !selectedAttendancePoint
                        ) {

                            map.setView(
                                point,
                                16,
                                {
                                    animate: false
                                }
                            );

                            userCentered = true;
                        }

                        /*
                         * Continuously evaluate geofence.
                         */
                        if (
                            selectedAttendancePoint
                        ) {

                            const distance =
                                calculateDistanceMeters(
                                    latitude,
                                    longitude,
                                    selectedAttendancePoint[0],
                                    selectedAttendancePoint[1]
                                );

                            updateGeofenceStatus(
                                distance
                            );
                        }
                    }

                    function handleUserLocationError(
                        error
                    ) {

                        console.warn(
                            "Geolocation error:",
                            error
                        );

                        const statusElement =
                            document.getElementById(
                                "geofenceStatus"
                            );

                        setCameraStartAccess(false);

                        if (statusElement) {

                            statusElement.innerHTML =
                                "⚠️ <b>Lokasi perangkat belum tersedia</b>";

                            statusElement.style.color =
                                "#b45309";
                        }
                    }

                    function startLiveLocation() {

                        if (
                            !navigator.geolocation
                        ) {

                            console.warn(
                                "Geolocation is not supported."
                            );

                            return;
                        }

                        if (
                            userWatchId !== null
                        ) {

                            navigator.geolocation.clearWatch(
                                userWatchId
                            );
                        }

                        userWatchId =
                            navigator.geolocation.watchPosition(
                                updateUserPosition,
                                handleUserLocationError,
                                {
                                    enableHighAccuracy: true,
                                    maximumAge: 3000,
                                    timeout: 10000
                                }
                            );
                    }

                    // ==========================================================
                    // CLEAR RED ZONE
                    // ==========================================================

                    function clearAttendanceMarker() {

                        if (
                            attendanceMarker
                        ) {

                            map.removeLayer(
                                attendanceMarker
                            );

                            attendanceMarker = null;
                        }

                        if (
                            attendanceCircle
                        ) {

                            map.removeLayer(
                                attendanceCircle
                            );

                            attendanceCircle = null;
                        }

                        selectedAttendancePoint = null;
                        setCameraStartAccess(false);

                        const statusElement =
                            document.getElementById(
                                "geofenceStatus"
                            );

                        const distanceElement =
                            document.getElementById(
                                "geofenceDistance"
                            );

                        if (statusElement) {

                            statusElement.innerHTML =
                                "🔵 Pilih lokasi absensi";

                            statusElement.style.color =
                                "#64748b";
                        }

                        if (distanceElement) {

                            distanceElement.textContent =
                                "Radius zona absensi: " +
                                GEOFENCE_RADIUS_METERS +
                                " meter";
                        }
                    }

                    // ==========================================================
                    // SHOW SELECTED ATTENDANCE LOCATION
                    // ==========================================================

                    function showAttendanceLocation(
                        placeName
                    ) {

                        clearAttendanceMarker();

                        if (!placeName) {

                            setCameraStartAccess(false);

                            if (userMarker) {

                                map.setView(
                                    userMarker.getLatLng(),
                                    16,
                                    {
                                        animate: true
                                    }
                                );

                            } else {

                                map.setView(
                                    DEFAULT_CENTER,
                                    17,
                                    {
                                        animate: true
                                    }
                                );
                            }

                            return;
                        }

                        const location =
                            ATTENDANCE_LOCATIONS[
                                placeName
                            ];

                        if (!location) {

                            setCameraStartAccess(false);

                            const statusElement =
                                document.getElementById(
                                    "geofenceStatus"
                                );

                            const distanceElement =
                                document.getElementById(
                                    "geofenceDistance"
                                );

                            if (statusElement) {

                                statusElement.innerHTML =
                                    "⚠️ <b>Koordinat lokasi belum tersedia</b>";

                                statusElement.style.color =
                                    "#b45309";
                            }

                            if (distanceElement) {

                                distanceElement.textContent =
                                    "Lokasi ini belum memiliki koordinat tetap.";
                            }

                            return;
                        }

                        const point = [
                            location.lat,
                            location.lng
                        ];

                        selectedAttendancePoint =
                            point;

                        // ======================================================
                        // RED ZONE - EXACTLY 60 METERS
                        // ======================================================

                        attendanceCircle =
                            L.circle(
                                point,
                                {
                                    radius:
                                        GEOFENCE_RADIUS_METERS,

                                    color:
                                        "#ef4444",

                                    weight:
                                        3,

                                    opacity:
                                        1,

                                    fillColor:
                                        "#ef4444",

                                    fillOpacity:
                                        0.25,

                                    interactive:
                                        false
                                }
                            ).addTo(map);

                        // ======================================================
                        // RED CENTER MARKER
                        // ======================================================

                        attendanceMarker =
                            L.circleMarker(
                                point,
                                {
                                    radius: 9,

                                    color:
                                        "#991b1b",

                                    weight:
                                        3,

                                    fillColor:
                                        "#ef4444",

                                    fillOpacity:
                                        1,

                                    pane:
                                        "markerPane"
                                }
                            )
                            .addTo(map);

                        // ======================================================
                        // POPUP
                        // ======================================================

                        attendanceMarker.bindPopup(
                            `
                            <div style="
                                min-width:225px;
                                font-family:Arial,sans-serif;
                            ">
                                <div style="
                                    font-size:15px;
                                    font-weight:700;
                                    color:#194c2c;
                                    margin-bottom:7px;
                                ">
                                    ${placeName}
                                </div>

                                <div style="
                                    font-size:12px;
                                    color:#475569;
                                    line-height:1.55;
                                ">
                                    ${location.address}
                                </div>

                                <div style="
                                    margin-top:9px;
                                    font-size:12px;
                                    color:#dc2626;
                                    font-weight:700;
                                ">
                                    Zona absensi: 60 meter
                                </div>

                                <div style="
                                    margin-top:4px;
                                    font-size:11px;
                                    color:#64748b;
                                ">
                                    ${location.lat},
                                    ${location.lng}
                                </div>
                            </div>
                            `
                        );

                        // ======================================================
                        // STATUS
                        // ======================================================

                        const statusElement =
                            document.getElementById(
                                "geofenceStatus"
                            );

                        const distanceElement =
                            document.getElementById(
                                "geofenceDistance"
                            );

                        if (statusElement) {

                            statusElement.innerHTML =
                                "📍 <b>" +
                                placeName +
                                " dipilih</b>";

                            statusElement.style.color =
                                "#2563eb";
                        }

                        if (distanceElement) {

                            distanceElement.textContent =
                                "Zona absensi memiliki radius tepat " +
                                GEOFENCE_RADIUS_METERS +
                                " meter";
                        }

                        // ======================================================
                        // GOOGLE MAPS LINK
                        // ======================================================

                        const googleMapsLocationLink =
                            document.getElementById(
                                "googleMapsLocationLink"
                            );

                        if (
                            googleMapsLocationLink
                        ) {

                            googleMapsLocationLink.href =
                                "https://www.google.com/maps/search/?api=1&query=" +
                                encodeURIComponent(
                                    location.lat +
                                    "," +
                                    location.lng
                                );

                            googleMapsLocationLink.textContent =
                                "📍 Buka " +
                                placeName +
                                " di Google Maps";
                        }

                        // ======================================================
                        // FOCUS ON SELECTED PLACE
                        // ======================================================

                        map.setView(
                            point,
                            17,
                            {
                                animate: true
                            }
                        );

                        // ======================================================
                        // INITIAL DISTANCE CHECK
                        // ======================================================

                        if (userMarker) {

                            const userPoint =
                                userMarker.getLatLng();

                            const distance =
                                calculateDistanceMeters(
                                    userPoint.lat,
                                    userPoint.lng,
                                    point[0],
                                    point[1]
                                );

                            updateGeofenceStatus(
                                distance
                            );
                        }
                    }

                    // ==========================================================
                    // DOM ELEMENTS
                    // ==========================================================

                    const attendanceLocationSelect =
                        document.getElementById(
                            "attendanceLocation"
                        );

                    const attendanceDropdown =
                        document.getElementById(
                            "attendanceDropdown"
                        );

                    const attendanceDropdownTrigger =
                        document.getElementById(
                            "attendanceDropdownTrigger"
                        );

                    const attendanceDropdownMenu =
                        document.getElementById(
                            "attendanceDropdownMenu"
                        );

                    const attendanceDropdownValue =
                        document.getElementById(
                            "attendanceDropdownValue"
                        );

                    function closeAttendanceDropdown() {
                        if (!attendanceDropdown) {
                            return;
                        }

                        attendanceDropdown.classList.remove("open");

                        if (attendanceDropdownTrigger) {
                            attendanceDropdownTrigger.setAttribute(
                                "aria-expanded",
                                "false"
                            );
                        }
                    }

                    function openAttendanceDropdown() {
                        if (!attendanceDropdown) {
                            return;
                        }

                        attendanceDropdown.classList.add("open");

                        if (attendanceDropdownTrigger) {
                            attendanceDropdownTrigger.setAttribute(
                                "aria-expanded",
                                "true"
                            );
                        }
                    }

                    function syncAttendanceDropdown(value) {
                        if (!attendanceDropdownValue) {
                            return;
                        }

                        const selectedOption =
                            attendanceDropdownMenu
                                ? attendanceDropdownMenu.querySelector(
                                    '[data-value="' +
                                    CSS.escape(value) +
                                    '"]'
                                )
                                : null;

                        attendanceDropdownValue.textContent =
                            selectedOption
                                ? selectedOption.textContent.trim()
                                : "-- Pilih lokasi absen --";

                        if (attendanceDropdownMenu) {
                            attendanceDropdownMenu
                                .querySelectorAll(
                                    ".attendance-dropdown-option"
                                )
                                .forEach(function(option) {
                                    option.classList.toggle(
                                        "selected",
                                        option.getAttribute("data-value") === value
                                    );
                                });
                        }
                    }

                    if (
                        attendanceLocationSelect
                    ) {

                        attendanceLocationSelect.addEventListener(
                            "change",
                            function (event) {
                                syncAttendanceDropdown(
                                    event.target.value
                                );

                                showAttendanceLocation(
                                    event.target.value
                                );
                            }
                        );
                    }

                    if (
                        attendanceDropdownTrigger
                    ) {
                        attendanceDropdownTrigger.addEventListener(
                            "click",
                            function () {
                                if (
                                    attendanceDropdown &&
                                    attendanceDropdown.classList.contains("open")
                                ) {
                                    closeAttendanceDropdown();
                                } else {
                                    openAttendanceDropdown();
                                }
                            }
                        );
                    }

                    if (
                        attendanceDropdownMenu &&
                        attendanceLocationSelect
                    ) {
                        attendanceDropdownMenu
                            .querySelectorAll(
                                ".attendance-dropdown-option"
                            )
                            .forEach(function(option) {

                                option.addEventListener(
                                    "click",
                                    function () {
                                        const value =
                                            option.getAttribute(
                                                "data-value"
                                            ) || "";

                                        attendanceLocationSelect.value =
                                            value;

                                        attendanceLocationSelect.dispatchEvent(
                                            new Event(
                                                "change",
                                                { bubbles: true }
                                            )
                                        );

                                        closeAttendanceDropdown();
                                    }
                                );
                            });
                    }

                    document.addEventListener(
                        "click",
                        function (event) {
                            if (
                                attendanceDropdown &&
                                !attendanceDropdown.contains(
                                    event.target
                                )
                            ) {
                                closeAttendanceDropdown();
                            }
                        }
                    );

                    document.addEventListener(
                        "keydown",
                        function (event) {
                            if (event.key === "Escape") {
                                closeAttendanceDropdown();
                            }
                        }
                    );

                    // ==========================================================
                    // INITIALIZE
                    // ==========================================================

                    function initializeLocationPage() {

                        const mapReady =
                            initMap();

                        if (!mapReady) {
                            return;
                        }

                        // Start browser GPS.
                        setCameraStartAccess(false);
                        startLiveLocation();

                        // Streamlit can replace the START button DOM after reruns.
                        // Re-apply the current geofence decision without changing UI/layout.
                        setInterval(function () {
                            setCameraStartAccess(geofenceAllowsCamera);
                        }, 500);

                        // Make sure Leaflet recalculates iframe dimensions.
                        setTimeout(
                            function () {
                                map.invalidateSize(true);
                            },
                            300
                        );

                        setTimeout(
                            function () {
                                map.invalidateSize(true);
                            },
                            1000
                        );

                        /*
                         * IMPORTANT:
                         * Some browsers restore the last <select> value after
                         * a Streamlit rerender. If that happens, immediately
                         * rebuild the corresponding 60 m red zone.
                         */
                        if (
                            attendanceLocationSelect
                        ) {
                            syncAttendanceDropdown(
                                attendanceLocationSelect.value
                            );

                            if (
                                attendanceLocationSelect.value
                            ) {
                                showAttendanceLocation(
                                    attendanceLocationSelect.value
                                );
                            }
                        }
                    }

                    function refreshMapSize() {
                        if (typeof map !== "undefined" && map) {
                            setTimeout(function () {
                                map.invalidateSize(true);
                            }, 120);
                        }
                    }

                    window.addEventListener("resize", refreshMapSize);
                    window.addEventListener("orientationchange", function () {
                        setTimeout(refreshMapSize, 250);
                    });

                    window.addEventListener(
                        "beforeunload",
                        function () {

                            if (
                                userWatchId !== null &&
                                navigator.geolocation
                            ) {

                                navigator.geolocation.clearWatch(
                                    userWatchId
                                );

                                userWatchId = null;
                            }
                        }
                    );

                    /*
                     * Wait until the iframe DOM has fully loaded.
                     * This avoids initializing Leaflet before #map exists.
                     */
                    if (
                        document.readyState ===
                        "loading"
                    ) {

                        document.addEventListener(
                            "DOMContentLoaded",
                            initializeLocationPage,
                            { once: true }
                        );

                    } else {

                        initializeLocationPage();
                    }
                </script>
            </body>
            </html>
    """

    # Desktop gets extra iframe height so the dropdown can open downward
    # outside the card without being clipped by the iframe viewport.
    with st.container(key="location_map_iframe"):
        st.iframe(
            map_html,
            height=760,
        )

    render_html('<div style="height:18px;"></div>')

    render_html('<div style="height:18px;"></div>')

    # Live Camera Detection remains below Live Location.
    render_camera_section()

# CAMERA VIEW
# ==========================================================

def request_camera_stop():
    """Stop the webcam immediately without any STOP modal."""
    new_session_id = _next_camera_generation()

    # Invalidate all callbacks/recognition work from the old session.
    _camera_session_active.clear()
    _first_camera_frame.clear()
    clear_processor()

    # Keep the same WebRTC component PLAYING so the next START is instant.
    # STOP only hides the stream and disables recognition.
    st.session_state["camera_session_id"] = new_session_id
    st.session_state["camera_requested"] = False
    st.session_state["camera_view_ready"] = False
    st.session_state["camera_last_playing"] = False
    st.session_state["camera_current_playing"] = False


def _start_camera():
    """Reveal the prewarmed webcam and request START-time AI preparation."""
    session_id = _next_camera_generation()

    # Start timing exactly when the Streamlit START callback executes.
    reset_camera_timing(session_id)
    log_camera_timing(f"START clicked | session={session_id}")

    # Request YuNet initialization now. The call is non-blocking; the already
    # running WebRTC stream remains independent from model loading.
    recognizer = get_face_recognizer()
    recognizer.prepare_camera_async()

    # Reset processor state before enabling recognition so the new session
    # cannot inherit tracking/result state from the previous session.
    clear_processor()
    _first_camera_frame.clear()

    st.session_state["camera_session_id"] = session_id
    st.session_state["camera_requested"] = True
    st.session_state["camera_view_ready"] = False
    st.session_state["camera_last_playing"] = False
    st.session_state["camera_current_playing"] = False

    # Do not enable AI here yet. START may be clicked before the browser has
    # finished WebRTC negotiation. render_camera_section() will enable the AI
    # path only after ctx.state.playing becomes True.
    _camera_session_active.clear()


# Auto-rerun only this camera fragment so WebRTC readiness is polled
# independently of the full dashboard. This does not alter the existing UI.
# While the Camera page is mounted, the fragment checks WebRTC state frequently
# enough that START can wait for PLAYING instead of racing the negotiation.
@st.fragment(run_every=0.25)
def render_camera_section():
    """Persistent WebRTC prewarm + AI recognition gate.

    WebRTC is mounted for the current camera_instance and kept PLAYING while
    the Dashboard is open. Before START, an overlay covers the video pixels.
    START reveals the already-running video and enables YuNet/SFace processing.
    """
    if not WEBRTC_AVAILABLE:
        st.error(
            "WebRTC tidak tersedia. Pastikan streamlit-webrtc terpasang."
        )
        return

    session_id = st.session_state.get(
        "camera_session_id",
        _get_camera_generation(),
    )
    camera_requested = bool(
        st.session_state.get("camera_requested", False)
    )

    # WebRTC prewarm is independent from AI readiness. The recognizer object
    # is only needed when START enables the processing path.
    recognizer = get_face_recognizer()

    render_html(
        """
        <div class="section-heading location-section-heading">
            <span class="dot"></span>
            Live Camera Detection
        </div>
        """
    )

    render_html(
        """
        <div class="camera-info">
            <b>Live Webcam</b>
        </div>
        """
    )

    # ----------------------------------------------------------
    # PERSISTENT WEBRTC HOST + OVERLAY
    # ----------------------------------------------------------
    # Critical design:
    #   1. WebRTC is ALWAYS rendered.
    #   2. desired_playing_state=True keeps it prewarmed.
    #   3. The placeholder is an OVERLAY above the already-playing stream.
    #   4. On START the overlay is removed, exposing the SAME video element.
    # This avoids hide/show of the iframe itself, which can trigger delayed
    # component remounting and was the reason multiple clicks could be needed.
    try:
        # IMPORTANT:
        # The WebRTC host stays mounted and PLAYING whenever the browser has
        # actually reached the PLAYING state. START never creates a new key.
        # If START was clicked too early, keep the existing idle overlay in
        # place until WebRTC reports PLAYING.
        #
        # This is the key race-condition fix: requested != ready.
        # requested means the user asked to start; ready means WebRTC is live.
        # The same WebRTC component is reused in both states.
        current_playing = bool(
            st.session_state.get("camera_current_playing", False)
        )
        camera_live_ready = bool(camera_requested and current_playing)

        shell_state_class = (
            "camera-prewarm-live"
            if camera_live_ready
            else "camera-prewarm-idle"
        )

        with st.container(
            key="camera_prewarm_shell",
            border=False,
        ):
            # The class is applied through a tiny marker element. CSS selectors
            # below are intentionally scoped to the stable Streamlit key.
            render_html(
                f'<div class="{shell_state_class}" style="display:none"></div>'
            )

            with st.container(key="camera_webrtc_host"):
                ctx = webrtc_streamer(
                    key=_get_camera_key(),
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration={
                        "iceServers": []
                    },
                    media_stream_constraints={
                        "video": {
                            "width": {"ideal": 640, "max": 640},
                            "height": {"ideal": 480, "max": 480},
                            "frameRate": {"ideal": 10, "max": 10},
                            "facingMode": {"ideal": "user"},
                        },
                        "audio": False,
                    },
                    video_frame_callback=video_frame_callback,
                    async_processing=True,
                    media_toggle_controls=False,
                    # TRUE CAMERA PREWARM: WebRTC negotiation/media startup
                    # happens while Dashboard is open. Recognition is still
                    # gated OFF by _camera_session_active until START.
                    desired_playing_state=True,
                    video_html_attrs={
                        "autoPlay": True,
                        "playsInline": True,
                        "muted": True,
                    },
                )

            # Overlay only covers the already-playing video. It never hides,
            # disables, or remounts the WebRTC component itself.
            if not camera_live_ready:
                render_html(
                    """
                    <div class="camera-prewarm-overlay">
                        <div class="camera-placeholder-inner">
                            <div class="camera-placeholder-monitor">
                                <div class="camera-placeholder-lens"></div>
                            </div>
                            <div class="camera-placeholder-title">
                                Webcam siap digunakan
                            </div>
                            <div class="camera-placeholder-text">
                                Kamera sedang dipersiapkan di background.<br>
                                Klik <b>START</b> untuk menampilkan live camera.
                            </div>
                        </div>
                    </div>
                    """
                )
    except Exception as exc:
        error_text = str(exc)
        if (
            "NotReadableError" in error_text
            or "Could not start video source" in error_text
        ):
            st.error(
                "Webcam tidak dapat dibuka. Pastikan kamera tidak sedang "
                "digunakan aplikasi/tab lain, lalu refresh halaman."
            )
        else:
            st.error(f"Camera error: {exc}")
        return

    # Read the browser/WebRTC state after the component has had a chance to
    # negotiate. This is the authoritative readiness signal for START.
    current_playing = bool(ctx.state.playing)
    st.session_state["camera_prepared"] = current_playing

    current_signalling = bool(
        getattr(ctx.state, "signalling", False)
    )
    frame_ready = _first_camera_frame.is_set()

    # ----------------------------------------------------------
    # TIMING
    # ----------------------------------------------------------
    global _camera_signalling_time
    global _camera_webrtc_playing_time

    if current_signalling:
        with _camera_timing_lock:
            if (
                _camera_start_time is not None
                and _camera_signalling_time is None
            ):
                _camera_signalling_time = time.perf_counter()
                log_camera_timing(
                    "WebRTC signalling",
                    _camera_signalling_time - _camera_start_time,
                )

    if current_playing:
        with _camera_timing_lock:
            if (
                _camera_start_time is not None
                and _camera_webrtc_playing_time is None
            ):
                _camera_webrtc_playing_time = time.perf_counter()
                log_camera_timing(
                    "WebRTC playing",
                    _camera_webrtc_playing_time - _camera_start_time,
                )

                if _camera_signalling_time is not None:
                    log_camera_timing(
                        "Signalling -> WebRTC playing",
                        _camera_webrtc_playing_time - _camera_signalling_time,
                    )

    st.session_state["camera_current_playing"] = current_playing

    # START can only expose the video and enable AI after WebRTC is actually
    # PLAYING. Before that point the existing prewarm placeholder remains in
    # place, so a user click cannot race the browser negotiation.
    camera_live_ready = bool(
        camera_requested and current_playing
    )
    st.session_state["camera_view_ready"] = camera_live_ready

    if camera_requested and not current_playing:
        print(
            "[CAMERA READY] START requested; waiting for WebRTC PLAYING "
            "without recreating the camera component."
        )

    # ----------------------------------------------------------
    # START / STOP
    # ----------------------------------------------------------
    with st.container(key="camera_controls"):
        control_left, control_right = st.columns([1, 8], gap="small")

        with control_left:
            if camera_requested:
                st.button(
                    "STOP",
                    key="camera_stop_button",
                    on_click=request_camera_stop,
                    width="stretch",
                )
            else:
                st.button(
                    "START",
                    key="camera_start_button",
                    on_click=_start_camera,
                    width="stretch",
                )

    # ----------------------------------------------------------
    # AI GATE
    # ----------------------------------------------------------
    # Before START, WebRTC is only a webcam prewarm. No ndarray conversion,
    # YuNet detection, bounding box, or SFace recognition is executed.
    # START enables the existing recognition path. YuNet is requested
    # asynchronously at START and recognition begins as soon as it is ready.
    if camera_live_ready:
        # WebRTC is now truly ready. Only now do we allow ndarray conversion,
        # YuNet tracking, bounding boxes, and SFace recognition to run.
        _camera_session_active.set()
    else:
        # START may already have been clicked, but if WebRTC is still
        # negotiating we deliberately keep AI disabled. The same component
        # remains mounted and the fragment will check again automatically.
        _camera_session_active.clear()
        st.session_state["camera_last_playing"] = False

    # ----------------------------------------------------------
    # DATASET TOOLS
    # ----------------------------------------------------------
    total_faces = get_dataset_count()
    names = get_dataset_names()

    render_html(
        f"""
        <div class="camera-info">
            🔍 <b>{total_faces}</b> file wajah testing tersedia.
            <br>
            📁 Dataset aktif:
            <b style="color:#28733e;">
                {", ".join(names) if names else "-"}
            </b>
        </div>
        """
    )

    if st.button(
        "Reload Faces",
        icon=":material/refresh:",
        key="camera_reload",
    ):
        try:
            count = get_face_recognizer().reload()
            clear_processor()
            st.session_state["reload_message"] = (
                f"✅ Dataset berhasil dimuat ulang: {count} file."
            )
        except Exception as exc:
            st.session_state["reload_message"] = (
                f"❌ Reload gagal: {exc}"
            )

    message = st.session_state.get("reload_message")
    if message:
        if message.startswith("✅"):
            st.success(message)
        else:
            st.error(message)
        st.session_state["reload_message"] = None

    render_html(
        """
        <div class="camera-guide">
            <b>Format dataset</b><br><br>
            darren_1.jpeg → <b>darren</b><br>
            darren_2.jpeg → <b>darren</b><br>
            budi_1.jpg → <b>budi</b><br><br>
            Wajah cocok → nama dataset.<br>
            Wajah tidak cocok → <b>Unknown</b>.
        </div>
        """
    )


# SIDEBAR
# ==========================================================

def render_sidebar():
    dashboard_active = True

    with st.sidebar:
        # ==================================================
        # BRAND
        # ==================================================

        render_html(
            """
            <div class="sidebar-brand">

                <div class="sidebar-brand-icon">
                    HR
                </div>

                <div>

                    <div class="sidebar-brand-title">
                        HR Attendance
                    </div>

                    <div class="sidebar-brand-subtitle">
                        Management System
                    </div>

                </div>

            </div>

            <div class="sidebar-divider"></div>

            <div class="sidebar-menu-title">
                MENU
            </div>
            """
        )

        # ==================================================
        # DASHBOARD
        # ==================================================

        st.button(
            "Dashboard",
            icon=":material/home:",
            icon_position="left",
            width="stretch",
            key="menu_dashboard",
            type=(
                "primary"
                if dashboard_active
                else "secondary"
            ),
            on_click=go_dashboard,
        )

        # ==================================================
        # LOGOUT
        # ==================================================

        render_html(
            """
            <div class="sidebar-logout-space"></div>
            """
        )

        st.button(
            "Logout",
            icon=":material/logout:",
            icon_position="left",
            width="stretch",
            key="menu_logout",
            type="secondary",
            on_click=logout,
        )


# ==========================================================
# MAIN
# ==========================================================

def main():
    init_dashboard_session_state()

    # Detect a new authentication session before rendering anything.
    sync_login_session()

    if not st.session_state.get(
            "logged_in",
            False,
    ):
        st.stop()

    # ======================================================
    # CSS
    # ======================================================

    apply_dashboard_css()

    # ======================================================
    # HAMBURGER
    # ======================================================

    with st.container(
            key="sidebar_toggle_container"
    ):
        st.button(
            "☰",
            key="toggle_sidebar",
            help="Buka / tutup sidebar",
            type="secondary",
            on_click=toggle_sidebar,
        )

    # ======================================================
    # SIDEBAR
    # ======================================================

    render_sidebar()

    # ======================================================
    # DASHBOARD
    # ======================================================
    # Live Location and Live Camera are rendered on the same page.
    # Live Camera is embedded directly below Live Location.
    show_dashboard()


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()
