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

# Import WebRTC early so the runtime is available instantly when the camera
# frame is first mounted (which now happens as soon as the dashboard is
# shown - see _render_camera_frame()).
try:
    from streamlit_webrtc import (
        WebRtcMode,
        webrtc_streamer,
    )

    WEBRTC_AVAILABLE = True
    print("✅ [WEBRTC] Runtime READY. Waiting for browser camera connection...")
except Exception as exc:
    WebRtcMode = None
    webrtc_streamer = None
    WEBRTC_AVAILABLE = False
    print(f"[WebRTC] runtime import failed: {exc}")

# ==========================================================
# CONFIG
# ==========================================================

CAMERA_KEY = "hr-face-camera"

# Shared camera / recognition runtime state. These must exist at module load
# time because Streamlit fragments can call the camera functions before any
# navigation action initializes them.
_camera_generation_lock = threading.Lock()
_camera_generation = 0

_face_recognizer_lock = threading.Lock()
_face_recognizer = None

_processor_lock = threading.Lock()
_processor = None

_camera_session_active = threading.Event()
_first_camera_frame = threading.Event()
_webrtc_status_lock = threading.Lock()
_webrtc_running_logged = False


def _get_camera_key():
    """Return a stable WebRTC key that changes only when a genuinely new
    browser peer connection is required.

    The key is intentionally NOT bumped on every Start Camera click anymore:
    the whole point of prewarming is that the same connection, opened while
    the user was still looking at the idle panel, is reused when they press
    Start. It is only bumped after an explicit Stop (see
    request_camera_stop()) so the next prewarm cycle starts clean.
    """
    instance = int(st.session_state.get("camera_instance", 0))
    return f"{CAMERA_KEY}-{instance}"


def _log_webrtc_running(source: str = "browser"):
    """Print the WebRTC running status once per camera lifecycle."""
    global _webrtc_running_logged
    with _webrtc_status_lock:
        if _webrtc_running_logged:
            return
        _webrtc_running_logged = True
    print("✅ [WEBRTC] Browser peer connected.")
    print(f"✅ [WEBRTC] Camera stream RUNNING ({source}).")


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
    # Recognition persistence:
    # exactly ONE successful green-box recognition per login.
    # The camera and AI recognition path are mounted automatically on Dashboard.
    "show_camera": True,
    # True = sidebar is visible on the first dashboard render.
    # False = hidden, and the hamburger can open it again.
    "sidebar_open": True,
    "reload_message": None,
    "camera_instance": 1,

    # internal camera UI state
    "camera_last_playing": False,
    "camera_session_id": 1,
    # Camera UI state retained for lifecycle compatibility.
    "camera_view_ready": False,

    # WebRTC device recovery state.
    "camera_retry_count": 0,

    # Always-on camera state. WebRTC and recognition are active while the Dashboard renders.
    "camera_requested": True,
    "camera_started": True,
    "camera_prepared": False,
    "camera_startup_started_at": 0.0,
    "camera_startup_error": None,
    "camera_current_playing": False,
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
            # The save slot remains locked until the next logical recognition session.

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
        if image is None:
            return self.get_result()

        # The first real WebRTC frame is proof that the browser camera is
        # already streaming. Enable the AI path immediately instead of waiting
        # for a fragment rerun to update session state.
        _camera_session_active.set()
        _first_camera_frame.set()

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

                # SFace recognition uses the SAME known_faces gallery that
                # face_utils.py builds. The face was already detected by YuNet
                # in process(), so this call only performs alignment, embedding,
                # and gallery matching; it does not run another detector pass.
                recognized_name, similarity, top_mean = recognizer.recognize_detection(
                    frame,
                    detection,
                    tolerance=FACE_TOLERANCE,
                )

                # The UI only exposes two identity states. While the ML stack
                # is still warming, do not show a third label such as
                # "Preparing..."; keep the box red as Unknown until a real
                # gallery match is available.
                if recognized_name in (None, "", "Preparing..."):
                    cleaned_name = "Unknown"
                else:
                    cleaned_name = str(recognized_name).strip() or "Unknown"

                with self.lock:
                    if generation != self.generation:
                        continue

                    self.tracking_name = cleaned_name

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

def clean_face_name(name):
    """Normalize the displayed recognition label."""
    if not name or name in {"Unknown", "Preparing..."}:
        return "Unknown"
    value = str(name).strip()
    return value or "Unknown"

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
    """Low-latency WebRTC callback with a separate PREWARM/START gate.

    The browser camera is mounted (and this callback is already running)
    well before the Dashboard's AI path is switched on. In that idle state
    we just return the frame untouched. The Start Camera button only flips
    the process-level recognition event - it does not affect whether this
    callback itself is running.
    """
    import av

    image = frame.to_ndarray(format="bgr24")

    # A callback invocation proves that the browser is already delivering
    # camera frames. This is safe to set from the WebRTC worker thread.
    _first_camera_frame.set()
    _log_webrtc_running("first frame received")

    # Recognition is gated only by the thread-safe Event.
    # Do NOT read st.session_state from the WebRTC worker thread.
    if not _camera_session_active.is_set():
        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24",
        )

    # ======================================================
    # ACTIVE RECOGNITION MODE
    # ======================================================
    locations = []
    names = []
    ids = []

    try:
        locations, names, ids = get_processor().process(image)
    except Exception as exc:
        print(f"Recognition callback error: {exc}")

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
    """Reset only the recognition state.

    The prewarmed camera connection is intentionally left mounted here -
    navigating back to the Dashboard should not force a new WebRTC
    connection. Only the AI path and the idle/overlay UI are reset, so the
    next Start Camera click stays instant.
    """
    _camera_session_active.clear()
    _first_camera_frame.clear()
    clear_processor()

    st.session_state["show_camera"] = True
    st.session_state["camera_started"] = True
    st.session_state["camera_view_ready"] = False
    st.session_state["camera_current_playing"] = False
    st.session_state["camera_last_playing"] = False
    st.session_state["camera_retry_count"] = 0
    st.session_state["reload_message"] = None


def go_camera():
    """Kept for backward compatibility with any external navigation calls.

    The camera is always mounted on the dashboard now (prewarm), so opening
    "the camera page" is just the same reset as go_dashboard().
    """
    go_dashboard()


def toggle_sidebar():
    st.session_state[
        "sidebar_open"
    ] = not st.session_state.get(
        "sidebar_open",
        True,
    )


# ==========================================================
# CSS
# ==========================================================
# ==========================================================

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
            st.markdown(content, unsafe_allow_html=True)
    except Exception:
        try:
            st.markdown(content, unsafe_allow_html=True)
        except Exception:
            pass


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
            overflow-anchor:none !important;

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
            overflow-anchor:none !important;

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
           CAMERA TEXT (info/guide boxes elsewhere in the page)
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

            width:100% !important;
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
           recognition logic or camera lifecycle.
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

            /* Mobile automatic recognition: always stack vertically. */
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

            /* Camera text must never create horizontal overflow. */
            .camera-info,
            .camera-guide {{
                font-size:11px !important;
                line-height:1.55 !important;
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
        svg {{
            max-width:100% !important;
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

        }}

        </style>
        """
    )


# ==========================================================
# DASHBOARD VIEW
# ==========================================================

def show_dashboard():
    """Dashboard with a single persistent, prewarmed camera frame."""

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
    # CAMERA - always-on native WebRTC
    # ==========================================================
    render_camera_section()


# CAMERA VIEW
# ==========================================================

def request_camera_stop():
    """Legacy compatibility hook; keep WebRTC alive and clear only transient recognition state."""
    _camera_session_active.set()
    _first_camera_frame.set()
    st.session_state["camera_started"] = True
    st.session_state["camera_requested"] = True
    clear_processor()


def _start_camera():
    """Backward-compatible helper: camera is always on, so this only keeps AI active."""
    st.session_state["camera_started"] = True
    st.session_state["camera_requested"] = True
    st.session_state["camera_startup_error"] = None
    st.session_state["camera_startup_started_at"] = time.monotonic()
    _camera_session_active.set()
    _first_camera_frame.set()


def _render_camera_frame():
    """Mount native streamlit-webrtc once per Streamlit render; camera is always playing."""
    if not WEBRTC_AVAILABLE:
        st.error(
            "WebRTC tidak tersedia. Pastikan streamlit-webrtc terpasang."
        )
        return

    try:
        ctx = webrtc_streamer(
            key=_get_camera_key(),
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640, "max": 640},
                    "height": {"ideal": 480, "max": 480},
                    "frameRate": {"ideal": 15, "max": 15},
                    "facingMode": {"ideal": "user"},
                },
                "audio": False,
            },
            video_frame_callback=video_frame_callback,
            async_processing=True,
            media_toggle_controls=False,
            # Always-on camera mode. Browser WebRTC is requested immediately
            # when the dashboard renders. There is no Start/Stop gate.
            desired_playing_state=True,
            video_html_attrs={
                "autoPlay": True,
                "playsInline": True,
                "muted": True,
                "controls": False,
            },
        )
    except Exception as exc:
        st.session_state["camera_startup_error"] = str(exc)
        st.session_state["camera_current_playing"] = False
        st.session_state["camera_view_ready"] = False
        _camera_session_active.clear()
        st.error(f"Camera error: {exc}")
        return

    current_playing = bool(getattr(ctx.state, "playing", False))
    st.session_state["camera_current_playing"] = current_playing
    st.session_state["camera_view_ready"] = current_playing
    st.session_state["camera_last_playing"] = current_playing

    if current_playing:
        st.session_state["camera_startup_error"] = None
        _first_camera_frame.set()

    # Always-on mode: camera and recognition stay active while Dashboard is open.
    _camera_session_active.set()


def render_camera_section():
    """Render the native WebRTC camera directly; no Start/Stop button is needed."""
    if not WEBRTC_AVAILABLE:
        st.error(
            "WebRTC tidak tersedia. Pastikan streamlit-webrtc terpasang."
        )
        return

    # Camera is always requested and AI remains enabled.
    st.session_state["camera_started"] = True
    st.session_state["camera_requested"] = True
    _camera_session_active.set()

    get_face_recognizer()
    _render_camera_frame()


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



# ==========================================================
# MAIN
# ==========================================================

def main():
    init_dashboard_session_state()

    # Direct-dashboard build: authentication is disabled.
    st.session_state["logged_in"] = True

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
    show_dashboard()


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()

