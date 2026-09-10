from __future__ import annotations

# ==========================================================
# app.py - ONE PROCESS BOOTSTRAPPER
# ==========================================================
# Run only:
#     streamlit run app.py
#
# STARTUP STRATEGY
# ----------------
# 1. Configure TensorFlow/OpenMP BEFORE TensorFlow/DeepFace is imported.
# 2. Import lightweight media/WebRTC runtime once.
# 3. Import the process-level face_recognizer singleton.
# 4. Import dashboard once so its lightweight recognition worker exists before START.
# 5. Start the COMPLETE face-model preload in one background worker:
#       DeepFace -> SFace -> embedding cache/gallery -> YuNet -> SFace warm-up
#    Nothing is waited on from the Streamlit request thread.
# 6. Dashboard mounts WebRTC immediately when Dashboard starts.
#    Before START, frames are passed through untouched without AI processing.
# 7. START reveals the already-prewarmed webcam immediately.
# 8. YuNet/SFace recognition uses the already-resident process singleton.
# 9. START/STOP never recreate the heavy models.
# ==========================================================

import os

# ==========================================================
# RUNTIME CONFIG - MUST BE BEFORE TENSORFLOW / DEEPFACE
# ==========================================================

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_BLOCKTIME", "0")

import streamlit as st

st.set_page_config(
    page_title="HR Attendance System",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# SESSION
# ==========================================================

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "user_data" not in st.session_state:
    st.session_state["user_data"] = None


# ==========================================================
# PROCESS BOOTSTRAP
# ==========================================================

@st.cache_resource(show_spinner=False)
def _bootstrap_process_once():
    """Initialize process-level runtime and start the complete preload once."""

    import time

    started = time.perf_counter()

    # ------------------------------------------------------
    # LIGHTWEIGHT MEDIA / WEBRTC RUNTIME
    # ------------------------------------------------------

    import numpy  # noqa: F401
    import av  # noqa: F401

    from streamlit_webrtc import (
        WebRtcMode,       # noqa: F401
        webrtc_streamer,  # noqa: F401
    )

    # ------------------------------------------------------
    # PROCESS-LEVEL FACE SINGLETON
    # ------------------------------------------------------

    from face_utils import (
        face_recognizer,
        start_process_preload,
    )

    print("🚀 [BOOTSTRAP] Process bootstrap started.")
    print("   FaceRecognizer singleton acquired.")

    # ------------------------------------------------------
    # DASHBOARD IMPORT ONCE
    # ------------------------------------------------------

    import dashboard

    print("✅ [BOOTSTRAP] Dashboard module imported.")
    print(
        "✅ [BOOTSTRAP] Recognition worker exists: "
        f"{getattr(dashboard, '_processor', None) is not None}"
    )

    # ------------------------------------------------------
    # START COMPLETE FACE PRELOAD - NON BLOCKING
    # ------------------------------------------------------
    # The process-level worker loads the full resident AI stack.
    # The webcam itself is prewarmed separately by dashboard.py.

    start_process_preload()

    elapsed = time.perf_counter() - started

    print(
        "⏳ [BOOTSTRAP] Complete face preload STARTED in "
        f"{elapsed:.3f}s"
    )
    print(
        "   DeepFace -> SFace -> embedding cache/gallery -> YuNet -> warm-up"
    )
    print(
        "   Camera WebRTC prewarm starts separately as soon as Dashboard is rendered."
    )
    print(
        "   Login is rendered immediately; preload continues in background."
    )

    return {
        "dashboard_imported": True,
        "preload_started": True,
        "bootstrap_seconds": elapsed,
    }


# ==========================================================
# START PROCESS BOOTSTRAP IMMEDIATELY
# ==========================================================

_BOOTSTRAP = _bootstrap_process_once()


# ==========================================================
# LOGIN MODULE
# ==========================================================

try:
    import login
except Exception as exc:
    st.error(f"Login module gagal dimuat: {exc}")
    st.stop()


# ==========================================================
# ROUTING
# ==========================================================

if not st.session_state.get("logged_in", False):
    # Login is intentionally available immediately. The process-level
    # SFace preload continues in the background.
    login.main()
else:
    # Dashboard mounts the persistent WebRTC prewarm component immediately.
    import dashboard
    dashboard.main()
