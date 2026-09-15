from __future__ import annotations

import os
import time

# ==========================================================
# RUNTIME / ML PERFORMANCE
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
# SESSION BOOTSTRAP
# ==========================================================
st.session_state.setdefault("logged_in", True)
st.session_state.setdefault(
    "user_data",
    {"id": None, "name": "Guest", "email": None},
)


@st.cache_resource(show_spinner=False)
def _bootstrap_process_once():
    """
    Long-lived server bootstrap.

    SFace, embedding gallery and YuNet are preloaded asynchronously here.
    The Dashboard mounts the native browser WebRTC camera automatically;
    there is no Start/Stop camera action.
    """
    started = time.perf_counter()

    print("🚀 [BOOTSTRAP] Process bootstrap started.")

    # Lightweight runtime imports.
    import numpy  # noqa: F401
    import av  # noqa: F401

    from streamlit_webrtc import WebRtcMode, webrtc_streamer  # noqa: F401

    print("✅ [WEBRTC] Runtime READY.")
    print("⏳ [WEBRTC] Dashboard will request native camera automatically.")

    # face_utils owns the singleton recognizer and background ML preload.
    from face_utils import start_process_preload

    import dashboard

    print("✅ [BOOTSTRAP] Dashboard module imported.")
    print(
        "✅ [BOOTSTRAP] Recognition worker exists: "
        f"{getattr(dashboard, '_processor', None) is not None}"
    )

    # Start SFace + embedding gallery + YuNet preload asynchronously.
    start_process_preload()

    elapsed = time.perf_counter() - started

    print(f"✅ [BOOTSTRAP] ML preload STARTED in {elapsed:.3f}s")
    print("   ├─ SFace: background preload")
    print("   ├─ Embedding gallery: background preload")
    print("   └─ YuNet: background preload")
    print("⏳ [WEBRTC] Waiting for browser peer + camera frames...")

    return {
        "dashboard_imported": True,
        "preload_started": True,
        "bootstrap_seconds": elapsed,
    }


# ==========================================================
# START ONCE
# ==========================================================
_BOOTSTRAP = _bootstrap_process_once()

# ==========================================================
# DASHBOARD
# ==========================================================
import dashboard

dashboard.main()
