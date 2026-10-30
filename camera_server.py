from __future__ import annotations

import asyncio
from collections import deque
import json
import math
import os
import re
import ssl
import threading
import time
import uuid
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

import cv2
import numpy as np
from websockets.asyncio.server import serve

from face_utils import (
    get_face_recognizer,
    start_process_preload,
)


# Load Python-side secrets/config from .env next to this file.
load_dotenv(Path(__file__).resolve().parent / ".env")


# ============================================================
# CONFIG
# ============================================================
HOST = os.getenv("HR_CAMERA_BIND_HOST", "127.0.0.1")
HTTP_PORT = int(os.getenv("HR_CAMERA_HTTP_PORT", "8766"))
WS_PORT = int(os.getenv("HR_CAMERA_WS_PORT", "8765"))

# Laravel -> Python face registration API.
# This is a server-to-server HTTP endpoint only; it does NOT create a
# separate Python UI.
FACE_REGISTRATION_PATH = "/register-face"
FACE_REGISTRATION_STATUS_PATH = "/face-registration-status"
FACE_REGISTRATION_MAX_IMAGE_BYTES = int(
    os.getenv("FACE_REGISTRATION_MAX_IMAGE_BYTES", str(512 * 1024))
)
FACE_REGISTRATION_MAX_REQUEST_BYTES = int(
    os.getenv(
        "FACE_REGISTRATION_MAX_REQUEST_BYTES",
        str((5 * FACE_REGISTRATION_MAX_IMAGE_BYTES) + (1024 * 1024)),
    )
)
FACE_REGISTRATION_JPEG_QUALITY = int(
    os.getenv("FACE_REGISTRATION_JPEG_QUALITY", "95")
)

CAMERA_DIR = Path(__file__).resolve().parent / "camera_web"
# Legacy local result file kept for compatibility with older callers.
# The current attendance evidence flow does NOT use this JSON file.
RESULT_FILE = Path(__file__).resolve().parent / "face_recognition_result.json"
RESULT_WRITE_LOCK = threading.RLock()

# YuNet runs on every current frame. SFace is throttled for CPU usage.
FRAME_INTERVAL = 1.0 / 25.0
RECOGNITION_INTERVAL = 0.08
RECOGNITION_STALE_AFTER = 0.80

# One-shot attendance confirmation. The first valid HRIS + face match
# is enough to start the evidence upload and browser redirect.
CONFIRMATION_WINDOW = 1
CONFIRMATION_REQUIRED = 1

# Kept for compatibility with app.py; recognition thresholds are owned by
# face_utils.py and are calibrated from target + impostor gallery evidence.
FACE_TOLERANCE = 0.40

# Laravel SSO.
HRIS_FACE_AUTH_EXCHANGE_URL = os.getenv(
    "HRIS_FACE_AUTH_EXCHANGE_URL",
    "https://hris-medialys.test/api/face-auth/exchange",
).strip()
HRIS_FACE_ATTENDANCE_EVIDENCE_URL = os.getenv(
    "HRIS_FACE_ATTENDANCE_EVIDENCE_URL",
    "https://hris-medialys.test/api/face-auth/attendance-evidence",
).strip()
FACE_RECOGNITION_TOKEN = os.getenv(
    "FACE_RECOGNITION_TOKEN",
    "",
).strip()
HRIS_RETURN_URL = os.getenv(
    "HRIS_RETURN_URL",
    "https://hris-medialys.test/",
).strip()
HRIS_FACE_CA_BUNDLE = os.getenv(
    "HRIS_FACE_CA_BUNDLE",
    "",
).strip()
SSO_MAX_LEASE_SECONDS = 20.0


# ============================================================
# GLOBAL RECOGNITION STATE
# ============================================================
_last_name = "Unknown"
_last_candidate = "Unknown"
_last_similarity = 0.0
_last_face_confident = False
_last_recognition_time = 0.0
_last_quality_ok = False
_last_quality_score = 0.0

_confirmation_history = deque(maxlen=CONFIRMATION_WINDOW)
_attendance_confirmed = False
_attendance_confirmed_name = "Unknown"
_attendance_confirmed_similarity = 0.0

_state_lock = threading.RLock()

_hris_user: dict | None = None
_hris_user_lock = threading.Lock()

_result_saved = False
_result_file_path: str | None = None
_result_lock = threading.Lock()

# Set as soon as the one-shot confirmed attendance upload is launched.
# This prevents the browser redirect from racing the upload startup.
_attendance_upload_started = False
# True after the first full face + HRIS match is accepted for this session.
# Recognition is terminal after this flag becomes True.
_attendance_match_locked = False

_recognition_worker_started = False
_recognition_event = threading.Event()
_recognition_lock = threading.Lock()
_pending_frame = None
_pending_detection = None
_pending_generation = 0
_recognition_generation = 0
_last_queue_time = 0.0
_pending_session_id: str | None = None

# Only one authenticated browser session owns the global recognition state at a time.
# A newer login supersedes the older session safely.
_active_session_id: str | None = None
_active_session_lock = threading.Lock()

_auth_lease_expires_monotonic = 0.0
_auth_lease_lock = threading.Lock()

_servers_started = False
_servers_lock = threading.Lock()
_http_server: ThreadingHTTPServer | None = None
_ws_loop = None

# Prevent concurrent registration requests from choosing the same file index.
FACE_REGISTRATION_LOCK = threading.Lock()


class FaceRegistrationLockedError(ValueError):
    """Raised when an employee already has a registered face dataset."""


# ============================================================
# HRIS STATE
# ============================================================
def format_display_name(value: str | None) -> str:
    value = str(value or "").strip()
    if not value:
        return "Unknown"
    return " ".join(
        word[:1].upper() + word[1:].lower()
        for word in value.split()
        if word
    ) or "Unknown"


def normalize_person_name(value: str | None) -> str:
    value = format_display_name(value)
    return " ".join(value.casefold().split())


def set_hris_user(user: dict | None) -> None:
    global _hris_user
    with _hris_user_lock:
        _hris_user = dict(user) if user else None


def get_hris_user() -> dict | None:
    with _hris_user_lock:
        return dict(_hris_user) if _hris_user else None


def clear_hris_user() -> None:
    set_hris_user(None)


def build_laravel_ssl_context():
    if not HRIS_FACE_CA_BUNDLE:
        return None
    ca_path = Path(HRIS_FACE_CA_BUNDLE).expanduser()
    if not ca_path.exists() or not ca_path.is_file():
        raise RuntimeError(f"HRIS_FACE_CA_BUNDLE does not exist: {ca_path}")
    return ssl.create_default_context(cafile=str(ca_path))


def exchange_code_with_laravel(code: str) -> dict:
    if not HRIS_FACE_AUTH_EXCHANGE_URL:
        raise RuntimeError("HRIS_FACE_AUTH_EXCHANGE_URL is not configured.")
    if not FACE_RECOGNITION_TOKEN:
        raise RuntimeError("FACE_RECOGNITION_TOKEN is not configured.")

    code = str(code or "").strip()
    if not re.fullmatch(r"[a-fA-F0-9]{64}", code):
        raise RuntimeError("Invalid one-time authorization code.")

    payload = json.dumps({"code": code}, ensure_ascii=False).encode("utf-8")
    request = Request(
        HRIS_FACE_AUTH_EXCHANGE_URL,
        data=payload,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {FACE_RECOGNITION_TOKEN}",
        },
    )

    try:
        context = build_laravel_ssl_context()
        kwargs = {"context": context} if context is not None else {}
        with urlopen(request, timeout=8, **kwargs) as response:
            raw = response.read().decode("utf-8", errors="replace")
            remaining_header = response.headers.get(
                "X-Face-Auth-Remaining",
                "",
            ).strip()
            result_token = response.headers.get(
                "X-Face-Auth-Result-Token",
                "",
            ).strip()
        result = json.loads(raw)
    except HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise RuntimeError(f"Laravel exchange HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot connect to Laravel exchange API: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Laravel returned invalid JSON.") from exc
    except Exception as exc:
        raise RuntimeError(f"Laravel exchange request failed: {exc}") from exc

    if not result.get("success"):
        raise RuntimeError(
            str(result.get("message", "Laravel rejected authorization code."))
        )

    try:
        expires_in = float(remaining_header)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Laravel did not return a valid X-Face-Auth-Remaining header."
        ) from exc

    if (
        not math.isfinite(expires_in)
        or expires_in <= 0
        or expires_in > SSO_MAX_LEASE_SECONDS + 1.0
    ):
        raise RuntimeError(
            "Laravel returned an invalid SSO remaining lifetime."
        )

    if not re.fullmatch(r"[a-fA-F0-9]{64}", result_token):
        raise RuntimeError(
            "Laravel did not return a valid X-Face-Auth-Result-Token."
        )

    data = result.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("Laravel response does not contain user data.")

    user = {
        "name": format_display_name(data.get("name")),
        "employee_no": str(data.get("employee_no") or "").strip(),
        "email": str(data.get("email") or "").strip(),
        # Laravel exchange JSON intentionally contains no return_url/action.
        # Python uses its fixed server-side return URL instead.
        "return_url": HRIS_RETURN_URL,
        "sso_expires_in": expires_in,
        "result_token": result_token,
    }

    if not user["name"] or user["name"] == "Unknown":
        raise RuntimeError("Laravel returned empty name.")

    set_auth_lease(expires_in)
    return user


def set_auth_lease(seconds: float) -> None:
    global _auth_lease_expires_monotonic

    seconds = float(seconds)
    if (
        not math.isfinite(seconds)
        or seconds <= 0
        or seconds > SSO_MAX_LEASE_SECONDS + 1.0
    ):
        raise RuntimeError("Invalid SSO lease duration.")

    with _auth_lease_lock:
        _auth_lease_expires_monotonic = time.monotonic() + seconds


def clear_auth_lease() -> None:
    global _auth_lease_expires_monotonic
    with _auth_lease_lock:
        _auth_lease_expires_monotonic = 0.0


def get_auth_lease_remaining() -> float:
    with _auth_lease_lock:
        expires_at = _auth_lease_expires_monotonic

    if expires_at <= 0:
        return 0.0

    return max(0.0, expires_at - time.monotonic())


def auth_lease_valid() -> bool:
    return get_auth_lease_remaining() > 0.0


# ============================================================
# SESSION OWNERSHIP
# ============================================================
def begin_session(session_id: str) -> None:
    """Make this browser the active recognition session and reset all state."""
    global _active_session_id

    with _active_session_lock:
        _active_session_id = session_id
        clear_hris_user()
        reset_result_state(remove_file=True)

    print(
        f"🔄 [SSO] Active session switched to {session_id}",
        flush=True,
    )


def is_active_session(session_id: str) -> bool:
    with _active_session_lock:
        return _active_session_id == session_id


def end_session_if_active(session_id: str) -> bool:
    """Clear global recognition state only if this session still owns it."""
    global _active_session_id

    with _active_session_lock:
        if _active_session_id != session_id:
            return False

        _active_session_id = None
        clear_hris_user()
        reset_result_state(remove_file=True)
        return True


# ============================================================
# RESULT JSON
# ============================================================
def reset_result_state(remove_file: bool = False) -> None:
    clear_auth_lease()
    global _result_saved, _attendance_confirmed, _attendance_upload_started
    global _attendance_match_locked
    global _attendance_confirmed_name, _attendance_confirmed_similarity
    global _last_name, _last_candidate, _last_similarity
    global _last_face_confident, _last_recognition_time
    global _last_quality_ok, _last_quality_score
    global _recognition_generation, _pending_frame, _pending_detection
    global _last_queue_time, _pending_session_id, _result_file_path

    with _result_lock:
        _result_saved = False
        _result_file_path = None
        _attendance_upload_started = False
        _attendance_match_locked = False
        with _state_lock:
            _confirmation_history.clear()
            _attendance_confirmed = False
            _attendance_confirmed_name = "Unknown"
            _attendance_confirmed_similarity = 0.0
            _last_name = "Unknown"
            _last_candidate = "Unknown"
            _last_similarity = 0.0
            _last_face_confident = False
            _last_recognition_time = 0.0
            _last_quality_ok = False
            _last_quality_score = 0.0

        with _recognition_lock:
            # Generation changes only when a logical login/logout session resets.
            # It does NOT change for every camera frame.
            _recognition_generation += 1
            _pending_frame = None
            _pending_detection = None
            _pending_session_id = None
            _last_queue_time = 0.0

        # `remove_file` is kept for backwards compatibility with existing
        # callers, but successful recognition results are historical records
        # and are NEVER deleted when a camera session ends or a new session starts.


def update_attendance_confirmation(
    name: str,
    similarity: float,
    hris_match: bool,
) -> tuple[bool, str, float, int]:
    """
    Confirm attendance in ONE-SHOT mode.

    The first valid frame where the strict face_utils identity equals the
    authenticated HRIS user is enough to confirm attendance. The helper keeps
    the existing history structure for compatibility, but the production
    requirement is exactly one valid match.
    """
    global _attendance_confirmed, _attendance_confirmed_name, _attendance_match_locked
    global _attendance_confirmed_similarity

    clean_name = format_display_name(name)

    with _state_lock:
        if hris_match and clean_name != "Unknown":
            # A valid match confirms attendance immediately in one-shot mode.
            _confirmation_history.append(
                (clean_name, float(similarity), time.monotonic())
            )

            # Keep the existing history shape for compatibility; max length is 1.
            if any(item[0] != clean_name for item in _confirmation_history):
                _confirmation_history.clear()
                _confirmation_history.append(
                    (clean_name, float(similarity), time.monotonic())
                )
        else:
            # Any non-match clears the one-shot confirmation state.
            _confirmation_history.clear()

        valid = [
            item
            for item in _confirmation_history
            if item[0] == clean_name and item[0] != "Unknown"
        ]

        confirmed = (
            hris_match
            and clean_name != "Unknown"
            and len(valid) >= CONFIRMATION_REQUIRED
        )

        if confirmed:
            scores = sorted(item[1] for item in valid)
            median_score = scores[len(scores) // 2]
            _attendance_confirmed = True
            _attendance_confirmed_name = clean_name
            _attendance_confirmed_similarity = float(median_score)
        else:
            _attendance_confirmed = False
            _attendance_confirmed_name = "Unknown"
            _attendance_confirmed_similarity = 0.0

        return (
            _attendance_confirmed,
            _attendance_confirmed_name,
            _attendance_confirmed_similarity,
            len(valid),
        )




def _attendance_terminal_state() -> tuple[bool, bool, str, float, str, str]:
    """Return terminal one-shot attendance state without running recognition."""
    with _result_lock:
        saved = bool(_result_saved)
        upload_started = bool(_attendance_upload_started)

    with _state_lock:
        locked = bool(_attendance_match_locked)
        name = format_display_name(_attendance_confirmed_name)
        similarity = float(_attendance_confirmed_similarity)

    terminal = locked or upload_started or saved
    if saved:
        status = "attendance_saved"
    elif terminal:
        status = "attendance_processing"
    else:
        status = ""

    return terminal, saved, name, similarity, HRIS_RETURN_URL, status

def _load_result_history() -> list[dict]:
    """Load the persistent JSON history from face_recognition_result.json."""
    if not RESULT_FILE.exists():
        return []

    try:
        raw = RESULT_FILE.read_text(encoding="utf-8").strip()
        if not raw:
            return []

        data = json.loads(raw)

        # Current format: a JSON array of recognition records.
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]

        # Backward compatibility: if an older build stored one object,
        # preserve that object as the first history item instead of losing it.
        if isinstance(data, dict):
            return [data]

        print(
            "⚠️ [FACE] Existing face_recognition_result.json has an unexpected format. "
            "Starting a new history array.",
            flush=True,
        )
        return []

    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"⚠️ [FACE] Could not read existing result history: {exc}. "
            "Starting a new history array.",
            flush=True,
        )
        return []


def _write_result_history(history: list[dict]) -> None:
    """Write the full history atomically so the JSON file is never left half-written."""
    tmp_file = RESULT_FILE.with_name(
        f".{RESULT_FILE.name}.{uuid.uuid4().hex}.tmp"
    )

    try:
        with tmp_file.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as fp:
            json.dump(
                history,
                fp,
                ensure_ascii=False,
                indent=2,
            )
            fp.write("\n")
            fp.flush()
            os.fsync(fp.fileno())

        # Same-directory os.replace is atomic on normal local filesystems.
        os.replace(tmp_file, RESULT_FILE)

    finally:
        try:
            if tmp_file.exists():
                tmp_file.unlink()
        except OSError:
            pass



def _build_multipart_body(
    field_name: str,
    filename: str,
    content_type: str,
    content: bytes,
) -> tuple[bytes, str]:
    """Build a minimal multipart/form-data body without adding a dependency."""
    boundary = f"----HRFaceBoundary{uuid.uuid4().hex}"
    boundary_bytes = boundary.encode("ascii")

    parts = [
        b"--" + boundary_bytes + b"\r\n",
        (
            f'Content-Disposition: form-data; name="{field_name}"; '
            f'filename="{filename}"\r\n'
        ).encode("utf-8"),
        f"Content-Type: {content_type}\r\n\r\n".encode("ascii"),
        content,
        b"\r\n--" + boundary_bytes + b"--\r\n",
    ]

    return b"".join(parts), boundary


def post_attendance_evidence(
    frame: np.ndarray,
    detection: dict,
    name: str,
    similarity: float,
    session_id: str | None = None,
    user_snapshot: dict | None = None,
    allow_background_completion: bool = False,
) -> bool:
    """
    Upload exactly one clean attendance evidence image after one-shot confirmation.

    The saved evidence keeps the COMPLETE original camera frame at its native
    resolution. It does not crop, resize, blur, sharpen, draw a bounding box,
    or add recognition diagnostics over the face. The only added element is a
    clean employee-name header above the original frame.

    ``detection`` is intentionally retained for API compatibility with the
    existing recognition pipeline, but it is not used to crop the saved image.
    Laravel receives the result as a multipart JPEG.
    """
    global _result_saved, _result_file_path

    if not HRIS_FACE_ATTENDANCE_EVIDENCE_URL:
        print(
            "❌ [ATTENDANCE] Evidence URL is not configured.",
            flush=True,
        )
        return False

    if not FACE_RECOGNITION_TOKEN:
        print(
            "❌ [ATTENDANCE] FACE_RECOGNITION_TOKEN is not configured.",
            flush=True,
        )
        return False

    if not allow_background_completion and not auth_lease_valid():
        print(
            "⚠️ [ATTENDANCE] SSO lease expired. Evidence not uploaded.",
            flush=True,
        )
        return False

    if not allow_background_completion:
        with _active_session_lock:
            if session_id is not None and _active_session_id != session_id:
                return False

    # When one-shot attendance has already been confirmed, keep a snapshot of the authenticated
    # HRIS user/result token so the JPEG upload can finish even if the browser
    # redirects and closes the WebSocket immediately afterward.
    user = (
        dict(user_snapshot)
        if isinstance(user_snapshot, dict)
        else (get_hris_user() or {})
    )

    result_token = str(
        user.get("result_token") or ""
    ).strip()

    if not re.fullmatch(r"[a-fA-F0-9]{64}", result_token):
        print(
            "❌ [ATTENDANCE] Result token is missing or invalid.",
            flush=True,
        )
        return False

    # Prevent a second local upload attempt in the same authenticated session.
    with _result_lock:
        if _result_saved:
            return True

    try:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            print(
                "❌ [ATTENDANCE] Camera frame is empty.",
                flush=True,
            )
            return False

        # IMPORTANT:
        # Keep the exact camera frame. Do not use the YuNet bounding box to
        # crop the evidence because that makes the stored image overly zoomed
        # and can cut hair, chin, forehead, shoulders, or background context.
        evidence_frame = frame.copy()

        display_name = format_display_name(name)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.90
        font_thickness = 2
        text_color = (0, 255, 0)

        # A simple header is added ABOVE the original frame. The original camera
        # pixels below it remain untouched.
        frame_height, frame_width = evidence_frame.shape[:2]

        (text_w, text_h), baseline = cv2.getTextSize(
            display_name,
            font,
            font_scale,
            font_thickness,
        )

        header_height = max(64, text_h + baseline + 24)
        header = np.zeros(
            (header_height, frame_width, 3),
            dtype=np.uint8,
        )

        text_x = max(10, (frame_width - text_w) // 2)
        text_y = max(text_h + 8, header_height - 16)

        cv2.putText(
            header,
            display_name,
            (text_x, text_y),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

        # Vertical concatenation preserves the original frame width and all
        # original camera pixels. No resize/interpolation is applied.
        evidence = np.vstack((header, evidence_frame))

        # High-quality JPEG for storage. Laravel receives standard JPEG bytes
        # and the existing 5 MB validation remains compatible.
        encoded_ok, encoded = cv2.imencode(
            ".jpg",
            evidence,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                97,
                cv2.IMWRITE_JPEG_PROGRESSIVE,
                1,
                cv2.IMWRITE_JPEG_OPTIMIZE,
                1,
            ],
        )

        if not encoded_ok:
            print(
                "❌ [ATTENDANCE] Failed to encode evidence JPEG.",
                flush=True,
            )
            return False

        encoded_bytes = encoded.tobytes()
        print(
            "📸 [ATTENDANCE] Evidence frame: "
            f"{frame_width}x{frame_height} → {frame_width}x{frame_height + header_height}, "
            f"JPEG={len(encoded_bytes) / 1024:.1f} KB, quality=97",
            flush=True,
        )

        body, boundary = _build_multipart_body(
            field_name="image",
            filename="attendance.jpeg",
            content_type="image/jpeg",
            content=encoded_bytes,
        )

        request = Request(
            HRIS_FACE_ATTENDANCE_EVIDENCE_URL,
            data=body,
            method="POST",
            headers={
                "Accept": "application/json",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Authorization": f"Bearer {FACE_RECOGNITION_TOKEN}",
                "X-Face-Auth-Result-Token": result_token,
            },
        )

        context = build_laravel_ssl_context()
        kwargs = {"context": context} if context is not None else {}

        with urlopen(
            request,
            timeout=10,
            **kwargs,
        ) as response:
            raw = response.read().decode(
                "utf-8",
                errors="replace",
            )
            status_code = int(response.status)

        try:
            result = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            result = {}

        if status_code < 200 or status_code >= 300:
            print(
                f"❌ [ATTENDANCE] Laravel evidence HTTP {status_code}: {raw}",
                flush=True,
            )
            return False

        if not result.get("success"):
            print(
                f"❌ [ATTENDANCE] Laravel rejected evidence: {raw}",
                flush=True,
            )
            return False

        # Do not let a previous browser session's background completion
        # contaminate a newly authenticated session.
        still_active = (
            session_id is None
            or is_active_session(session_id)
        )

        if still_active:
            with _result_lock:
                _result_saved = True
                _result_file_path = str(
                    result.get("path") or ""
                ) or None

        print(
            "✅ [ATTENDANCE] Evidence uploaded to Laravel.",
            flush=True,
        )
        print(
            f"   Name       : {format_display_name(name)}",
            flush=True,
        )
        print(
            f"   Similarity : {float(similarity):.3f}",
            flush=True,
        )
        print(
            f"   Evidence   : {result.get('path', '(path not returned)')}",
            flush=True,
        )
        return True

    except HTTPError as exc:
        try:
            body_text = exc.read().decode(
                "utf-8",
                errors="replace",
            )
        except Exception:
            body_text = ""

        print(
            f"❌ [ATTENDANCE] Laravel evidence HTTP {exc.code}: {body_text}",
            flush=True,
        )
        return False
    except URLError as exc:
        print(
            f"❌ [ATTENDANCE] Cannot connect to Laravel evidence API: {exc}",
            flush=True,
        )
        return False
    except Exception as exc:
        print(
            f"❌ [ATTENDANCE] Evidence upload failed: {exc}",
            flush=True,
        )
        return False


def save_result(
    name: str,
    similarity: float,
    identity_confirmed: bool,
    hris_match: bool,
    session_id: str | None = None,
) -> bool:
    """
    Persist exactly once after the one-shot identity + HRIS confirmation.

    IMPORTANT:
    - The output filename is ALWAYS exactly ``face_recognition_result.json``.
    - Every successful recognition is appended as a NEW object in the JSON array.
    - Existing records are never deleted or replaced logically.
    - The browser still receives ``result_saved=true`` and can redirect to HRIS.
    """
    global _result_saved, _result_file_path

    name = format_display_name(name)

    if (
        name == "Unknown"
        or not identity_confirmed
        or not hris_match
        or not auth_lease_valid()
    ):
        return False

    with _active_session_lock:
        if session_id is not None and _active_session_id != session_id:
            return False

        user = get_hris_user() or {}

        with _result_lock:
            # One persisted attendance result per authenticated browser session.
            if _result_saved:
                return False

            employee_no = str(user.get("employee_no") or "").strip()
            email = str(user.get("email") or "").strip()
            hris_name = format_display_name(user.get("name"))
            timestamp = time.time()
            result_id = uuid.uuid4().hex

            payload = {
                "result_id": result_id,
                "name": name,
                "hris_name": hris_name,
                "employee_no": employee_no,
                "email": email,
                "similarity": round(float(similarity), 6),
                "face_confident": True,
                "face_match_confirmed": True,
                "confirmation_count": CONFIRMATION_REQUIRED,
                "hris_match": True,
                "session_id": session_id or "",
                "timestamp": timestamp,
                "timestamp_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(timestamp),
                ),
                "result_file": RESULT_FILE.name,
            }

            try:
                # One process-wide writer lock prevents two recognition threads
                # from reading/writing the same history simultaneously.
                with RESULT_WRITE_LOCK:
                    history = _load_result_history()
                    history.append(payload)
                    _write_result_history(history)

                _result_saved = True
                _result_file_path = str(RESULT_FILE)

                print(
                    "✅ [FACE] Result appended successfully.",
                    flush=True,
                )
                print(
                    f"   Name        : {name}",
                    flush=True,
                )
                print(
                    f"   Employee No : {employee_no}",
                    flush=True,
                )
                print(
                    f"   Similarity  : {float(similarity):.3f}",
                    flush=True,
                )
                print(
                    f"   JSON File   : {_result_file_path}",
                    flush=True,
                )
                print(
                    f"   Total Records: {len(history)}",
                    flush=True,
                )
                return True

            except Exception as exc:
                print(
                    f"⚠️ [FACE] Save result error: {exc}",
                    flush=True,
                )
                return False


# ============================================================
# ML STATUS LOGGER
# ============================================================
def print_ml_status(prefix: str = "[ML]") -> None:
    try:
        recognizer = get_face_recognizer()
        status = recognizer.get_runtime_status()
        print(
            f"🧠 {prefix} "
            f"DeepFace={'LOADED' if status['deepface_loaded'] else 'NOT LOADED'} | "
            f"SFace={'LOADED' if status['sface_loaded'] else 'NOT LOADED'} | "
            f"YuNet={'LOADED' if status['yunet_loaded'] else 'NOT LOADED'} | "
            f"Gallery={'LOADED' if status['embeddings_loaded'] else 'NOT LOADED'} | "
            f"Ready={'YES' if status['ready'] else 'NO'} | "
            f"Stage={status['stage']}",
            flush=True,
        )
    except Exception as exc:
        print(f"❌ [ML] Status error: {exc}", flush=True)


# ============================================================
# RECOGNITION WORKER
# ============================================================
def _start_recognition_worker() -> None:
    global _recognition_worker_started
    with _recognition_lock:
        if _recognition_worker_started:
            return
        _recognition_worker_started = True

    threading.Thread(
        target=recognition_worker,
        daemon=True,
        name="arcface-recognition-worker",
    ).start()


def recognition_worker() -> None:
    global _pending_frame, _pending_detection, _pending_generation
    global _attendance_upload_started, _attendance_match_locked
    global _last_name, _last_candidate, _last_similarity
    global _last_face_confident, _last_recognition_time
    global _last_quality_ok, _last_quality_score
    global _recognition_generation, _pending_session_id

    print("✅ [ARCFACE] Background worker READY.", flush=True)

    while True:
        _recognition_event.wait()
        _recognition_event.clear()

        while True:
            with _recognition_lock:
                frame = _pending_frame
                detection = _pending_detection
                generation = _pending_generation
                pending_session_id = _pending_session_id
                _pending_frame = None
                _pending_detection = None
                _pending_session_id = None

            if frame is None or detection is None:
                break

            try:
                if not pending_session_id or not is_active_session(pending_session_id):
                    continue

                if not auth_lease_valid():
                    continue

                # ONE-SHOT TERMINAL GATE: after the first accepted match,
                # never run another ArcFace/PAD recognition job.
                with _result_lock:
                    terminal_match = bool(
                        _attendance_upload_started or _result_saved
                    )
                with _state_lock:
                    terminal_match = terminal_match or bool(_attendance_match_locked)

                if terminal_match:
                    continue

                recognizer = get_face_recognizer()
                hris_user = get_hris_user() or {}
                target_identity = format_display_name(hris_user.get("name"))

                candidate, similarity, face_confident = recognizer.recognize_detection(
                    frame,
                    detection,
                    target_identity=target_identity,
                    tolerance=FACE_TOLERANCE,
                )

                if not auth_lease_valid():
                    continue

                metrics = recognizer.get_last_match_metrics()

                # face_utils performs full-gallery classification internally,
                # then returns only the logged-in target or Unknown to this UI
                # layer. Non-target identities never become browser candidates.
                clean_candidate = format_display_name(candidate)
                quality_ok = bool(metrics.get("quality_ok", False))
                quality_score = float(metrics.get("quality_score", 0.0))

                with _recognition_lock:
                    if generation != _recognition_generation:
                        continue

                if not is_active_session(pending_session_id):
                    continue

                with _state_lock:
                    _last_candidate = clean_candidate
                    _last_name = clean_candidate
                    _last_similarity = float(similarity)
                    _last_face_confident = bool(face_confident)
                    _last_quality_ok = quality_ok
                    _last_quality_score = quality_score
                    _last_recognition_time = time.monotonic()

                hris_name = target_identity

                # A valid match requires BOTH:
                # 1) the face_utils strict identity/quality gate passed, and
                # 2) the strict candidate equals the authenticated HRIS user.
                #
                # Example:
                #   HRIS = Hosea
                #   Camera candidate = Hosea
                #   => hris_match = True
                #
                hris_match = (
                    bool(face_confident)
                    and clean_candidate != "Unknown"
                    and normalize_person_name(clean_candidate)
                    == normalize_person_name(hris_name)
                )

                (
                    attendance_confirmed,
                    attendance_name,
                    attendance_similarity,
                    confirmation_count,
                ) = update_attendance_confirmation(
                    clean_candidate,
                    float(similarity),
                    hris_match,
                )

                print(
                    f"🔎 [ARCFACE] Candidate={clean_candidate} "
                    f"similarity={similarity:.3f} "
                    f"threshold={metrics.get('threshold', 0.60):.3f} "
                    f"quality={'OK' if quality_ok else 'FAIL'} "
                    f"quality_score={quality_score:.2f} "
                    f"target={target_identity} "
                    f"top3={metrics.get('topk_mean', 0.0):.3f} "
                    f"prototype={metrics.get('prototype_similarity', 0.0):.3f} "
                    f"negative={metrics.get('negative_similarity', 0.0):.3f} "
                    f"negative_gate={metrics.get('negative_gate', 0.0):.3f} "
                    f"margin={metrics.get('identity_margin', 0.0):.3f} "
                    f"gallery_best={metrics.get('gallery_best_identity', 'Unknown')} "
                    f"gallery_best_sim={metrics.get('gallery_best_similarity', 0.0):.3f} "
                    f"gallery_best_score={metrics.get('gallery_best_score', 0.0):.3f} "
                    f"gallery_runner={metrics.get('gallery_runner_up_identity', 'Unknown')} "
                    f"gallery_runner_sim={metrics.get('gallery_runner_up_similarity', 0.0):.3f} "
                    f"gallery_runner_score={metrics.get('gallery_runner_up_score', 0.0):.3f} "
                    f"gallery_margin={metrics.get('gallery_margin', 0.0):.3f} "
                    f"tta={'YES' if metrics.get('tta_used', False) else 'NO'} "
                    f"target_samples={metrics.get('target_sample_count', 0)} "
                    f"face_confident={'YES' if face_confident else 'NO'} "
                    f"hris_match={'YES' if hris_match else 'NO'} "
                    f"confirm={confirmation_count}/{CONFIRMATION_REQUIRED} "
                    f"attendance={'YES' if attendance_confirmed else 'NO'}",
                    flush=True,
                )

                if clean_candidate == "Unknown":
                    print(
                        f"ℹ️ [FACE] HRIS comparison skipped: face is Unknown.\n"
                        f"   HRIS : {hris_name}\n"
                        f"   Face : Unknown",
                        flush=True,
                    )
                elif hris_match:
                    print(
                        "✅ [FACE] HRIS NAME MATCH\n"
                        f"   HRIS : {hris_name}\n"
                        f"   Face : {clean_candidate}",
                        flush=True,
                    )
                else:
                    print(
                        "❌ [FACE] HRIS NAME MISMATCH\n"
                        f"   HRIS : {hris_name}\n"
                        f"   Face : {clean_candidate}",
                        flush=True,
                    )

                with _result_lock:
                    attendance_result_already_saved = bool(_result_saved)
                    attendance_upload_already_started = bool(
                        _attendance_upload_started
                    )

                if (
                    not attendance_result_already_saved
                    and not attendance_upload_already_started
                    and attendance_confirmed
                    and attendance_name != "Unknown"
                    and normalize_person_name(attendance_name)
                    == normalize_person_name(hris_name)
                ):
                    # Snapshot all information needed by the uploader BEFORE
                    # the browser is allowed to redirect and close the WebSocket.
                    upload_user_snapshot = get_hris_user() or {}

                    with _result_lock:
                        if not _result_saved and not _attendance_upload_started:
                            _attendance_upload_started = True
                            launch_upload = True
                        else:
                            launch_upload = False

                    if launch_upload:
                        # Lock recognition BEFORE starting the upload.
                        # From this point onward only the Laravel save/status
                        # path may continue; no more ArcFace/PAD calls.
                        with _state_lock:
                            _attendance_match_locked = True
                            _attendance_confirmed = True
                            _attendance_confirmed_name = attendance_name
                            _attendance_confirmed_similarity = float(attendance_similarity)

                        with _recognition_lock:
                            _pending_frame = None
                            _pending_detection = None
                            _pending_session_id = None

                        print(
                            "🟢 [FACE] ONE-SHOT HRIS + FACE MATCH CONFIRMED — LOCKED.",
                            flush=True,
                        )
                        print(
                            "📤 [ATTENDANCE] Starting background JPEG upload "
                            "before browser redirect.",
                            flush=True,
                        )

                        threading.Thread(
                            target=post_attendance_evidence,
                            args=(
                                frame.copy(),
                                dict(detection),
                                attendance_name,
                                attendance_similarity,
                                pending_session_id,
                            ),
                            kwargs={
                                "user_snapshot": upload_user_snapshot,
                                "allow_background_completion": True,
                            },
                            daemon=True,
                            name="attendance-evidence-upload",
                        ).start()

                        print(
                            "⏳ [ATTENDANCE] Recognition is locked. "
                            "Browser will redirect only after Laravel returns result_saved=true.",
                            flush=True,
                        )
                else:
                    print(
                        "⚠️ [FACE] Not saved/uploaded yet. "
                        f"HRIS match={'YES' if hris_match else 'NO'} | "
                        f"confirm={confirmation_count}/{CONFIRMATION_REQUIRED}",
                        flush=True,
                    )

            except Exception as exc:
                print(f"❌ [ARCFACE] Recognition error: {exc}", flush=True)


def queue_recognition(frame: np.ndarray, detection: dict) -> None:
    global _last_queue_time, _pending_frame, _pending_detection
    global _pending_generation, _pending_session_id

    session_id = None
    with _active_session_lock:
        session_id = _active_session_id

    if not session_id:
        return

    if not auth_lease_valid():
        return

    with _result_lock:
        if _attendance_upload_started or _result_saved:
            return
    with _state_lock:
        if _attendance_match_locked:
            return

    now = time.monotonic()
    if now - _last_queue_time < RECOGNITION_INTERVAL:
        return
    _last_queue_time = now

    with _recognition_lock:
        # IMPORTANT: do not increment _recognition_generation here.
        # A newer camera frame must not invalidate an in-flight recognition.
        _pending_generation = _recognition_generation
        _pending_session_id = session_id
        _pending_frame = frame.copy()
        _pending_detection = dict(detection)

    _recognition_event.set()


# ============================================================
# CURRENT FRAME PROCESSING
# ============================================================
def _session_result_meta() -> tuple[bool, str]:
    """Return the current persisted-result flag + fixed HRIS redirect URL."""
    with _result_lock:
        saved = bool(_result_saved)
    return saved, HRIS_RETURN_URL


def process_frame(message: bytes, session_id: str | None = None) -> dict:
    global _last_name, _last_candidate, _last_similarity, _last_face_confident
    global _attendance_confirmed, _attendance_confirmed_name
    global _attendance_confirmed_similarity
    global _recognition_generation

    try:
        encoded = np.frombuffer(message, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except Exception:
        frame = None

    if frame is None:
        result_saved, return_url = _session_result_meta()
        return {
            "name": "Unknown",
            "candidate": "Unknown",
            "similarity": 0.0,
            "face_confident": False,
            "hris_match": False,
            "recognition_fresh": False,
            "box": None,
            "status": "invalid_frame",
            "result_saved": result_saved,
            "return_url": return_url,
        }

    if session_id is not None and not is_active_session(session_id):
        return {
            "name": "Unknown",
            "candidate": "Unknown",
            "similarity": 0.0,
            "face_confident": False,
            "hris_match": False,
            "recognition_fresh": False,
            "box": None,
            "status": "session_superseded",
            "result_saved": False,
            "return_url": "",
        }

    if not auth_lease_valid():
        if session_id is not None:
            end_session_if_active(session_id)
        return {
            "name": "Unknown",
            "candidate": "Unknown",
            "similarity": 0.0,
            "face_confident": False,
            "hris_match": False,
            "recognition_fresh": False,
            "box": None,
            "status": "sso_expired",
            "result_saved": False,
            "return_url": HRIS_RETURN_URL,
        }

    recognizer = get_face_recognizer()
    hris_user = get_hris_user() or {}
    target_identity = format_display_name(hris_user.get("name"))

    if target_identity == "Unknown":
        result_saved, return_url = _session_result_meta()
        return {
            "name": "Unknown",
            "candidate": "Unknown",
            "similarity": 0.0,
            "face_confident": False,
            "hris_match": False,
            "recognition_fresh": False,
            "box": None,
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "status": "waiting_hris_auth",
            "result_saved": result_saved,
            "return_url": return_url,
            "detector_ready": recognizer.is_detector_ready(),
            "model_ready": recognizer.is_model_ready(),
            "embeddings_ready": recognizer.is_embeddings_ready(),
            "recognition_ready": recognizer.is_recognition_ready(),
        }

    # ONE-SHOT TERMINAL STATE: do not run YuNet/ArcFace/PAD again after a
    # valid face/HRIS match. Keep sending only a lightweight status so the
    # browser can wait for Laravel to finish saving evidence/attendance.
    terminal_locked, terminal_saved, terminal_name, terminal_similarity, return_url, terminal_status = _attendance_terminal_state()
    if terminal_locked:
        hris_match_terminal = bool(
            terminal_name != "Unknown"
            and normalize_person_name(terminal_name) == normalize_person_name(target_identity)
        )
        return {
            "name": terminal_name if terminal_name != "Unknown" else target_identity,
            "candidate": terminal_name if terminal_name != "Unknown" else target_identity,
            "similarity": float(terminal_similarity),
            "face_confident": True,
            "hris_match": hris_match_terminal,
            "recognition_fresh": False,
            "attendance_confirmed": bool(terminal_saved),
            "confirmation_count": CONFIRMATION_REQUIRED,
            "confirmation_required": CONFIRMATION_REQUIRED,
            "attendance_mode": "one-shot-terminal",
            "result_saved": bool(terminal_saved),
            "return_url": return_url,
            "target_identity": target_identity,
            "box": None,
            "raw_box": None,
            "stable_box": None,
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "status": terminal_status,
            "detector_ready": recognizer.is_detector_ready(),
            "model_ready": recognizer.is_model_ready(),
            "embeddings_ready": recognizer.is_embeddings_ready(),
            "recognition_ready": recognizer.is_recognition_ready(),
        }

    if not recognizer.is_detector_ready():
        print_ml_status("WAIT")
        result_saved, return_url = _session_result_meta()
        return {
            "name": "Unknown",
            "candidate": "Unknown",
            "similarity": 0.0,
            "face_confident": False,
            "hris_match": False,
            "recognition_fresh": False,
            "box": None,
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "status": "loading_yunet",
            "result_saved": result_saved,
            "return_url": return_url,
            "detector_ready": False,
            "model_ready": recognizer.is_model_ready(),
            "embeddings_ready": recognizer.is_embeddings_ready(),
        }

    detection, stable_box = recognizer.track_face(frame)
    if detection is None or stable_box is None:
        result_saved, return_url = _session_result_meta()
        with _state_lock:
            _last_name = "Unknown"
            _last_candidate = "Unknown"
            _last_similarity = 0.0
            _last_face_confident = False
            _confirmation_history.clear()
            _attendance_confirmed = False
            _attendance_confirmed_name = "Unknown"
            _attendance_confirmed_similarity = 0.0
        with _recognition_lock:
            # Drop a queued frame because there is currently no face.
            # Do not change session generation here.
            globals()["_pending_frame"] = None
            globals()["_pending_detection"] = None
            globals()["_pending_session_id"] = None
        return {
            "name": "Unknown",
            "candidate": "Unknown",
            "similarity": 0.0,
            "face_confident": False,
            "hris_match": False,
            "recognition_fresh": False,
            "box": None,
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "status": "searching",
            "result_saved": result_saved,
            "return_url": return_url,
            "detector_ready": True,
            "model_ready": recognizer.is_model_ready(),
            "embeddings_ready": recognizer.is_embeddings_ready(),
            "recognition_ready": recognizer.is_recognition_ready(),
        }

    queue_recognition(frame, detection)

    with _result_lock:
        result_saved_now = bool(_result_saved)

    with _state_lock:
        age = time.monotonic() - _last_recognition_time
        fresh = age <= RECOGNITION_STALE_AFTER

        # UI is deliberately binary: only the logged-in target or Unknown.
        # Internally, face_utils has already classified the frame against the
        # entire gallery and verified that the gallery winner is the target.
        candidate_name = (
            _last_candidate
            if fresh and _last_candidate == target_identity
            else "Unknown"
        )

        similarity = _last_similarity if fresh else 0.0
        face_confident = bool(fresh and _last_face_confident)
        confirmed = bool(
            fresh
            and _attendance_confirmed
            and result_saved_now
            and candidate_name != "Unknown"
            and normalize_person_name(candidate_name)
            == normalize_person_name(_attendance_confirmed_name)
        )

    hris_name = target_identity

    # Browser/UI match is valid only when the recognition layer passed its
    # strict identity + quality gate and the displayed identity equals HRIS.
    hris_match = bool(
        face_confident
        and candidate_name != "Unknown"
        and normalize_person_name(candidate_name)
        == normalize_person_name(hris_name)
    )

    with _state_lock:
        confirmation_count = len(
            [
                item
                for item in _confirmation_history
                if item[0] == target_identity
            ]
        )

    result_saved, return_url = _session_result_meta()

    return {
        "name": candidate_name,
        "candidate": candidate_name,
        "similarity": float(similarity),
        "face_confident": face_confident,
        "hris_match": hris_match,
        "recognition_fresh": fresh,
        "attendance_confirmed": bool(confirmed),
        "confirmation_count": int(confirmation_count),
        "confirmation_required": CONFIRMATION_REQUIRED,
        "attendance_mode": "one-shot",
        "result_saved": result_saved,
        "return_url": return_url,
        "target_identity": target_identity,
        # IMPORTANT: browser uses raw YuNet coordinates for lowest latency.
        # camera_web.js applies visual smoothing at render time.
        "box": {
            "x": int(detection["x"]),
            "y": int(detection["y"]),
            "w": int(detection["w"]),
            "h": int(detection["h"]),
        },
        "raw_box": {
            "x": int(detection["x"]),
            "y": int(detection["y"]),
            "w": int(detection["w"]),
            "h": int(detection["h"]),
        },
        "stable_box": {
            "x": int(stable_box[0]),
            "y": int(stable_box[1]),
            "w": int(stable_box[2]),
            "h": int(stable_box[3]),
        },
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "status": "face_detected",
        "detector_ready": True,
        "model_ready": recognizer.is_model_ready(),
        "embeddings_ready": recognizer.is_embeddings_ready(),
        "recognition_ready": recognizer.is_recognition_ready(),
    }


# ============================================================
# FACE REGISTRATION HTTP HELPERS
# ============================================================
def _send_json_response(
    handler: BaseHTTPRequestHandler,
    status_code: int,
    payload: dict,
) -> None:
    data = json.dumps(
        payload,
        ensure_ascii=False,
    ).encode("utf-8")

    handler.send_response(status_code)
    handler.send_header(
        "Content-Type",
        "application/json; charset=utf-8",
    )
    handler.send_header(
        "Cache-Control",
        "no-store, no-cache, must-revalidate, max-age=0",
    )
    handler.send_header(
        "Pragma",
        "no-cache",
    )
    handler.send_header(
        "Expires",
        "0",
    )
    handler.send_header(
        "Content-Length",
        str(len(data)),
    )
    handler.end_headers()
    handler.wfile.write(data)


def _sanitize_registration_name(value: str) -> str:
    value = str(value or "").strip()

    value = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        value,
    )

    value = value.strip("_")

    return value or "Unknown"


def _first_name(value: str) -> str:
    value = str(value or "").strip()

    if not value:
        return "Unknown"

    return value.split()[0]


def _parse_registration_multipart(
    content_type: str,
    body: bytes,
) -> tuple[dict[str, str], list[dict]]:
    """
    Parse Laravel's multipart/form-data request.

    Expected scalar fields:
      - employee_name
      - employee_no
      - user_uid

    Expected files:
      - photos[] x 5
    """
    header_blob = (
        f"Content-Type: {content_type}\r\n"
        "MIME-Version: 1.0\r\n"
        "\r\n"
    ).encode("utf-8")

    message = BytesParser(
        policy=default,
    ).parsebytes(
        header_blob + body
    )

    if not message.is_multipart():
        raise ValueError(
            "Request bukan multipart/form-data."
        )

    fields: dict[str, str] = {}
    files: list[dict] = []

    for part in message.iter_parts():
        disposition = part.get(
            "Content-Disposition",
            "",
        )

        name_match = re.search(
            r'name="([^"]+)"',
            disposition,
        )

        if not name_match:
            continue

        field_name = name_match.group(1)
        filename = part.get_filename()
        payload = (
            part.get_payload(
                decode=True
            )
            or b""
        )

        if filename:
            files.append(
                {
                    "field": field_name,
                    "filename": filename,
                    "content": payload,
                }
            )
        else:
            fields[field_name] = payload.decode(
                "utf-8",
                errors="replace",
            )

    return fields, files


def _get_next_registration_index(
    folder: Path,
    base_name: str,
) -> int:
    """
    Find the next number for one employee.

    Example:
        Darren_222222_1.jpg
        Darren_222222_2.jpg
        Darren_222222_5.jpg

    returns 6.
    """
    pattern = re.compile(
        rf"^{re.escape(base_name)}_(\d+)\.jpg$",
        re.IGNORECASE,
    )

    highest = 0

    if not folder.exists():
        return 1

    for path in folder.iterdir():
        if not path.is_file():
            continue

        match = pattern.match(
            path.name
        )

        if not match:
            continue

        try:
            number = int(
                match.group(1)
            )
        except ValueError:
            continue

        highest = max(
            highest,
            number,
        )

    return highest + 1


def _save_registration_images(
    files: list[dict],
    target_folder: Path,
    base_name: str,
) -> tuple[list[str], int]:
    """
    Decode and save exactly five uploaded images as JPEG files.

    Returns:
        (saved filenames, first assigned index)
    """
    start_index = _get_next_registration_index(
        target_folder,
        base_name,
    )

    saved_files: list[str] = []

    try:
        for offset, item in enumerate(files):
            image_bytes = item["content"]

            if not image_bytes:
                raise ValueError(
                    f"Foto ke-{offset + 1} kosong."
                )

            if len(image_bytes) > FACE_REGISTRATION_MAX_IMAGE_BYTES:
                raise ValueError(
                    f"Foto ke-{offset + 1} melebihi "
                    f"{FACE_REGISTRATION_MAX_IMAGE_BYTES // 1024} KB."
                )

            encoded = np.frombuffer(
                image_bytes,
                dtype=np.uint8,
            )

            image = cv2.imdecode(
                encoded,
                cv2.IMREAD_COLOR,
            )

            if image is None:
                raise ValueError(
                    f"Foto ke-{offset + 1} bukan gambar yang valid."
                )

            if image.size == 0:
                raise ValueError(
                    f"Foto ke-{offset + 1} kosong setelah decode."
                )

            number = start_index + offset
            filename = (
                f"{base_name}_{number}.jpg"
            )

            output_path = (
                target_folder /
                filename
            )

            ok = cv2.imwrite(
                str(output_path),
                image,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    FACE_REGISTRATION_JPEG_QUALITY,
                ],
            )

            if not ok:
                raise RuntimeError(
                    f"Gagal menyimpan {filename}."
                )

            saved_files.append(
                filename
            )

        return saved_files, start_index

    except Exception:
        # Roll back only files created by this request.
        for filename in saved_files:
            try:
                (
                    target_folder /
                    filename
                ).unlink(
                    missing_ok=True
                )
            except Exception:
                pass

        raise


def _register_face_images(
    employee_name: str,
    employee_no: str,
    user_uid: str,
    files: list[dict],
) -> dict:
    """
    Save five images into:
        known_faces/<FullName>/<FirstName>_<EmployeeNo>_<N>.jpg

    Then refresh the FaceRecognizer gallery.
    """
    if not employee_name:
        raise ValueError(
            "employee_name tidak dikirim oleh Laravel."
        )

    if not employee_no:
        raise ValueError(
            "employee_no tidak dikirim oleh Laravel."
        )

    if not user_uid:
        raise ValueError(
            "user_uid tidak dikirim oleh Laravel."
        )

    image_files = [
        item
        for item in files
        if item.get("field") == "photos[]"
    ]

    if len(image_files) != 5:
        raise ValueError(
            f"Python menerima {len(image_files)} foto; "
            "harus tepat 5 foto."
        )

    first_name = _sanitize_registration_name(
        _first_name(
            employee_name
        )
    )

    safe_employee_no = _sanitize_registration_name(
        employee_no
    )

    base_name = (
        f"{first_name}_{safe_employee_no}"
    )

    recognizer = get_face_recognizer()
    known_faces_dir = recognizer.known_faces_dir

    target_folder = known_faces_dir / _sanitize_registration_name(
        format_display_name(employee_name)
    )

    target_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    with FACE_REGISTRATION_LOCK:
        existing_files = sorted(target_folder.glob(f"{base_name}_*.jpg"))
        legacy_folder = known_faces_dir / first_name
        if legacy_folder != target_folder:
            existing_files += sorted(legacy_folder.glob(f"{base_name}_*.jpg"))

        if existing_files:
            raise FaceRegistrationLockedError(
                "Wajah user sudah terdaftar. "
                "Registrasi wajah hanya dapat dilakukan satu kali."
            )

        saved_files, start_index = (
            _save_registration_images(
                image_files,
                target_folder,
                base_name,
            )
        )

        try:
            print(
                "📸 [REGISTRATION] "
                f"Saved {len(saved_files)} images.",
                flush=True,
            )
            print(
                f"   User        : {employee_name}",
                flush=True,
            )
            print(
                f"   Employee No : {employee_no}",
                flush=True,
            )
            print(
                f"   Folder      : {target_folder}",
                flush=True,
            )
            print(
                f"   Files       : {', '.join(saved_files)}",
                flush=True,
            )

            # face_utils.py already exposes reload(), which re-indexes
            # known_faces and rebuilds the gallery asynchronously.
            print(
                "🔄 [REGISTRATION] Reloading face gallery...",
                flush=True,
            )
            gallery_count = recognizer.reload()

            # Give the new gallery a chance to finish preparation.
            gallery_ready = recognizer.wait_until_ready(
                timeout=120.0
            )

            if not gallery_ready:
                raise RuntimeError(
                    "Gallery reload belum READY setelah registration."
                )

            print(
                "✅ [REGISTRATION] "
                f"Gallery READY | embeddings={gallery_count}",
                flush=True,
            )

            return {
                "success": True,
                "user_uid": user_uid,
                "folder": target_folder.name,
                "path": str(target_folder),
                "files": saved_files,
                "start_index": start_index,
                "end_index": start_index + 4,
                "gallery_count": gallery_count,
            }

        except Exception:
            # Delete this request's images if gallery preparation failed.
            for filename in saved_files:
                try:
                    (
                        target_folder /
                        filename
                    ).unlink(
                        missing_ok=True
                    )
                except Exception:
                    pass

            raise


def _get_face_registration_status(
    employee_name: str,
    employee_no: str,
) -> dict:
    """
    Check registration state using the Python filesystem only.
    """
    if not employee_name:
        raise ValueError(
            "employee_name tidak dikirim oleh Laravel."
        )

    if not employee_no:
        raise ValueError(
            "employee_no tidak dikirim oleh Laravel."
        )

    first_name = _sanitize_registration_name(
        _first_name(employee_name)
    )

    safe_employee_no = _sanitize_registration_name(
        employee_no
    )

    base_name = (
        f"{first_name}_{safe_employee_no}"
    )

    recognizer = get_face_recognizer()

    target_folder = recognizer.known_faces_dir / _sanitize_registration_name(
        format_display_name(employee_name)
    )
    existing_files = sorted(target_folder.glob(f"{base_name}_*.jpg"))
    legacy_folder = recognizer.known_faces_dir / first_name
    if legacy_folder != target_folder:
        existing_files += sorted(legacy_folder.glob(f"{base_name}_*.jpg"))

    registered = bool(existing_files)

    return {
        "success": True,
        "registered": registered,
        "photo_count": len(existing_files),
        "files": [
            file.name
            for file in existing_files
        ],
    }


# ============================================================
# HTTP FILE SERVER
# ============================================================
class CameraHandler(BaseHTTPRequestHandler):
    def _send_file(self, filename: str, content_type: str) -> None:
        path = CAMERA_DIR / filename
        if not path.exists() or not path.is_file():
            self.send_error(404, "Camera file not found.")
            return
        try:
            data = path.read_bytes()
        except Exception:
            self.send_error(500, "Could not read camera file.")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            self.send_response(302)
            self.send_header("Location", "/camera")
            self.end_headers()
            return
        if path == "/camera":
            return self._send_file("camera_web.html", "text/html; charset=utf-8")
        if path == "/camera_web.js":
            return self._send_file("camera_web.js", "application/javascript; charset=utf-8")
        if path == "/camera_web.css":
            return self._send_file("camera_web.css", "text/css; charset=utf-8")
        if path == "/session-status":
            user = get_hris_user()
            payload = {
                "authenticated": bool(user),
                "user": user.get("name") if user else None,
            }
            data = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if path == FACE_REGISTRATION_STATUS_PATH:
            try:
                expected_token = str(
                    FACE_RECOGNITION_TOKEN or ""
                ).strip()

                authorization = str(
                    self.headers.get(
                        "Authorization",
                        "",
                    )
                    or ""
                ).strip()

                provided_token = ""

                token_match = re.match(
                    r"^Bearer\s+(.+)$",
                    authorization,
                    re.IGNORECASE,
                )

                if token_match:
                    provided_token = (
                        token_match.group(1).strip()
                    )

                if (
                    not expected_token
                    or not provided_token
                    or provided_token != expected_token
                ):
                    _send_json_response(
                        self,
                        401,
                        {
                            "success": False,
                            "message": "Unauthorized.",
                        },
                    )
                    return

                employee_name = str(
                    self.headers.get(
                        "X-Employee-Name",
                        "",
                    )
                    or ""
                ).strip()

                employee_no = str(
                    self.headers.get(
                        "X-Employee-No",
                        "",
                    )
                    or ""
                ).strip()

                result = _get_face_registration_status(
                    employee_name=employee_name,
                    employee_no=employee_no,
                )

                _send_json_response(
                    self,
                    200,
                    result,
                )
                return

            except ValueError as exc:
                _send_json_response(
                    self,
                    422,
                    {
                        "success": False,
                        "message": str(exc),
                    },
                )
                return

            except Exception as exc:
                print(
                    f"❌ [REGISTRATION STATUS] Error: {exc}",
                    flush=True,
                )

                _send_json_response(
                    self,
                    500,
                    {
                        "success": False,
                        "message":
                            "Python gagal membaca status registration.",
                    },
                )
                return

        self.send_error(404)

    def do_POST(self) -> None:
        """
        Laravel -> Python face registration endpoint.

        POST /register-face
        Content-Type: multipart/form-data

        Fields:
          employee_name
          employee_no
          user_uid

        Files:
          photos[] x 5
        """
        path = self.path.split("?", 1)[0]

        if path != FACE_REGISTRATION_PATH:
            _send_json_response(
                self,
                404,
                {
                    "success": False,
                    "message": "Endpoint not found.",
                },
            )
            return

        try:
            # ----------------------------------------------------
            # 1. Server-to-server token
            # ----------------------------------------------------
            expected_token = str(
                FACE_RECOGNITION_TOKEN or ""
            ).strip()

            authorization = str(
                self.headers.get(
                    "Authorization",
                    "",
                )
                or ""
            ).strip()

            provided_token = ""
            token_match = re.match(
                r"^Bearer\s+(.+)$",
                authorization,
                re.IGNORECASE,
            )

            if token_match:
                provided_token = (
                    token_match.group(1).strip()
                )

            if not expected_token:
                _send_json_response(
                    self,
                    500,
                    {
                        "success": False,
                        "message":
                            "FACE_RECOGNITION_TOKEN belum dikonfigurasi di Python.",
                    },
                )
                return

            if (
                not provided_token
                or provided_token != expected_token
            ):
                _send_json_response(
                    self,
                    401,
                    {
                        "success": False,
                        "message": "Unauthorized.",
                    },
                )
                return

            # ----------------------------------------------------
            # 2. Multipart request
            # ----------------------------------------------------
            content_type = str(
                self.headers.get(
                    "Content-Type",
                    "",
                )
                or ""
            )

            if not content_type.lower().startswith(
                "multipart/form-data"
            ):
                _send_json_response(
                    self,
                    400,
                    {
                        "success": False,
                        "message":
                            "Content-Type harus multipart/form-data.",
                    },
                )
                return

            try:
                content_length = int(
                    self.headers.get(
                        "Content-Length",
                        "0",
                    )
                )
            except ValueError:
                _send_json_response(
                    self,
                    400,
                    {
                        "success": False,
                        "message":
                            "Content-Length tidak valid.",
                    },
                )
                return

            if content_length <= 0:
                _send_json_response(
                    self,
                    400,
                    {
                        "success": False,
                        "message":
                            "Request body kosong.",
                    },
                )
                return

            if (
                content_length >
                FACE_REGISTRATION_MAX_REQUEST_BYTES
            ):
                _send_json_response(
                    self,
                    413,
                    {
                        "success": False,
                        "message":
                            "Ukuran request registration terlalu besar.",
                    },
                )
                return

            body = self.rfile.read(
                content_length
            )

            fields, files = (
                _parse_registration_multipart(
                    content_type,
                    body,
                )
            )

            # ----------------------------------------------------
            # 3. Identity supplied by Laravel
            # ----------------------------------------------------
            employee_name = str(
                fields.get(
                    "employee_name",
                    "",
                )
                or ""
            ).strip()

            employee_no = str(
                fields.get(
                    "employee_no",
                    "",
                )
                or ""
            ).strip()

            user_uid = str(
                fields.get(
                    "user_uid",
                    "",
                )
                or ""
            ).strip()

            # Never trust browser-supplied identity.
            # These values are expected to originate from Laravel's
            # authenticated user record.

            # ----------------------------------------------------
            # 4. Save + reload
            # ----------------------------------------------------
            result = _register_face_images(
                employee_name=employee_name,
                employee_no=employee_no,
                user_uid=user_uid,
                files=files,
            )

            _send_json_response(
                self,
                200,
                result,
            )

        except FaceRegistrationLockedError as exc:
            print(
                f"🔒 [REGISTRATION] Locked: {exc}",
                flush=True,
            )

            _send_json_response(
                self,
                409,
                {
                    "success": False,
                    "message": str(exc),
                },
            )

        except ValueError as exc:
            print(
                f"⚠️ [REGISTRATION] Validation error: {exc}",
                flush=True,
            )

            _send_json_response(
                self,
                422,
                {
                    "success": False,
                    "message": str(exc),
                },
            )

        except Exception as exc:
            print(
                f"❌ [REGISTRATION] Error: {exc}",
                flush=True,
            )

            _send_json_response(
                self,
                500,
                {
                    "success": False,
                    "message":
                        "Python gagal menyimpan dataset wajah.",
                },
            )


    def log_message(self, fmt, *args):
        # Suppress default BaseHTTPRequestHandler request logging.
        return


# ============================================================
# WEBSOCKET
# ============================================================
async def ws_handler(websocket) -> None:
    import uuid

    session_id = uuid.uuid4().hex

    print(
        f"✅ [WS] Browser camera connected | session={session_id}",
        flush=True,
    )
    _start_recognition_worker()

    frame_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    session_user = None
    auth_completed = False

    async def receiver():
        nonlocal session_user, auth_completed

        try:
            async for message in websocket:
                if isinstance(message, str):
                    try:
                        payload = json.loads(message)
                    except json.JSONDecodeError:
                        continue

                    message_type = str(
                        payload.get("type") or ""
                    ).strip().lower()

                    if message_type == "logout":
                        session_user = None
                        auth_completed = False

                        if end_session_if_active(session_id):
                            print(
                                f"🔒 [SSO] Session logged out | session={session_id}",
                                flush=True,
                            )
                        else:
                            print(
                                f"ℹ️ [SSO] Logout ignored for inactive session | session={session_id}",
                                flush=True,
                            )

                        await websocket.send(json.dumps({
                            "type": "logout_result",
                            "success": True,
                            "message": "Python face-recognition session reset.",
                        }))
                        continue

                    if message_type != "auth":
                        continue

                    if auth_completed:
                        continue

                    code = str(
                        payload.get("code") or ""
                    ).strip()

                    if not code:
                        await websocket.send(json.dumps({
                            "type": "auth_result",
                            "success": False,
                            "message": "Authorization code is missing.",
                        }))
                        continue

                    try:
                        # Claim the global recognition state for this browser.
                        # Any older browser session becomes stale, but its cleanup
                        # cannot clear the new user's state.
                        begin_session(session_id)

                        authenticated_user = await asyncio.to_thread(
                            exchange_code_with_laravel,
                            code,
                        )

                        if not is_active_session(session_id):
                            raise RuntimeError(
                                "This camera session was superseded by another login."
                            )

                        session_user = authenticated_user
                        set_hris_user(authenticated_user)
                        auth_completed = True

                        print(
                            "✅ [SSO] HRIS user authenticated.",
                            flush=True,
                        )
                        print(
                            f"   Name        : {authenticated_user['name']}",
                            flush=True,
                        )
                        print(
                            f"   Employee No : {authenticated_user['employee_no']}",
                            flush=True,
                        )
                        print(
                            f"   Email       : {authenticated_user['email']}",
                            flush=True,
                        )

                        try:
                            target_status = get_face_recognizer().get_target_identity_status(
                                authenticated_user["name"]
                            )
                            if target_status["samples"] > 0:
                                print(
                                    "🎯 [TARGET] HRIS target set to "
                                    f"{target_status['target']} | "
                                    f"positive_samples={target_status['samples']} | "
                                    f"threshold={target_status['threshold']:.3f} | "
                                    "full gallery used for identity competition",
                                    flush=True,
                                )
                            else:
                                print(
                                    "❌ [TARGET] Logged-in identity is not present in known_faces: "
                                    f"{authenticated_user['name']}",
                                    flush=True,
                                )
                        except Exception as exc:
                            print(
                                f"⚠️ [TARGET] Could not inspect target gallery: {exc}",
                                flush=True,
                            )

                        await websocket.send(json.dumps({
                            "type": "auth_result",
                            "success": True,
                            "message": "HRIS authentication successful.",
                            "user": authenticated_user.get("name", ""),
                            "return_url": HRIS_RETURN_URL,
                            "sso_expires_in": authenticated_user.get(
                                "sso_expires_in",
                                0,
                            ),
                        }))

                        print(
                            f"✅ [SSO] WebSocket session authenticated | session={session_id}",
                            flush=True,
                        )

                    except Exception as exc:
                        session_user = None
                        auth_completed = False

                        # Only the current owner may be cleared.
                        end_session_if_active(session_id)

                        print(
                            f"❌ [SSO] Authorization failed: {exc}",
                            flush=True,
                        )

                        await websocket.send(json.dumps({
                            "type": "auth_result",
                            "success": False,
                            "message": "HRIS authentication failed.",
                        }))

                    continue

                if not isinstance(message, bytes):
                    continue

                if not auth_completed or session_user is None:
                    continue

                if not is_active_session(session_id):
                    # A newer login now owns the global recognition state.
                    continue

                while frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                try:
                    frame_queue.put_nowait(message)
                except asyncio.QueueFull:
                    pass

        except Exception as exc:
            print(
                f"⚠️ [WS] Receiver ended: {exc}",
                flush=True,
            )

    async def processor():
        # These variables belong to ws_handler().
        # processor() also updates them, so they must be declared nonlocal.
        nonlocal session_user, auth_completed

        while True:
            message = await frame_queue.get()

            if not auth_completed or session_user is None:
                continue

            if not is_active_session(session_id):
                continue

            if not auth_lease_valid():
                # SSO expired before the next frame could be processed.
                # Do not save/post anything. Tell the browser to return to HRIS.
                try:
                    await websocket.send(json.dumps({
                        "type": "sso_expired",
                        "success": False,
                        "status": "sso_expired",
                        "result_saved": False,
                        "return_url": HRIS_RETURN_URL,
                        "message": "SSO authorization expired.",
                    }, ensure_ascii=False))
                except Exception:
                    pass

                session_user = None
                auth_completed = False
                end_session_if_active(session_id)
                break

            try:
                result = await asyncio.to_thread(
                    process_frame,
                    message,
                    session_id,
                )
            except Exception as exc:
                print(
                    f"❌ [FRAME] process_frame error: {exc}",
                    flush=True,
                )
                continue

            if not is_active_session(session_id):
                continue

            if result.get("status") == "sso_expired":
                try:
                    await websocket.send(
                        json.dumps(result, ensure_ascii=False)
                    )
                except Exception:
                    pass

                session_user = None
                auth_completed = False
                end_session_if_active(session_id)
                break

            try:
                await websocket.send(
                    json.dumps(result, ensure_ascii=False)
                )
            except Exception as exc:
                print(
                    f"ℹ️ [WS] Send stopped: {exc}",
                    flush=True,
                )
                break

    receiver_task = asyncio.create_task(receiver())
    processor_task = asyncio.create_task(processor())

    try:
        done, pending = await asyncio.wait(
            {receiver_task, processor_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

        await asyncio.gather(
            *pending,
            return_exceptions=True,
        )

    finally:
        session_user = None
        auth_completed = False

        if end_session_if_active(session_id):
            print(
                f"🧹 [SSO] Active session cleared | session={session_id}",
                flush=True,
            )
        else:
            print(
                f"ℹ️ [SSO] Old session disconnected; active session preserved | session={session_id}",
                flush=True,
            )

        print(
            "🛑 [WS] Browser camera disconnected.",
            flush=True,
        )


# ============================================================
# SERVER STARTUP / COMPATIBILITY API FOR app.py
# ============================================================
def _run_http_server() -> None:
    global _http_server
    _http_server = ThreadingHTTPServer((HOST, HTTP_PORT), CameraHandler)
    print(f"✅ [HTTP] Camera server: http://{HOST}:{HTTP_PORT}/camera", flush=True)
    _http_server.serve_forever(poll_interval=0.25)


async def _run_ws_server() -> None:
    print(f"✅ [WS] WebSocket server: ws://{HOST}:{WS_PORT}", flush=True)
    async with serve(
        ws_handler,
        HOST,
        WS_PORT,
        max_size=8 * 1024 * 1024,
        max_queue=4,
        ping_interval=20,
        ping_timeout=20,
        close_timeout=5,
    ):
        await asyncio.Future()


def _ws_thread_main() -> None:
    global _ws_loop
    _ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_ws_loop)
    try:
        _ws_loop.run_until_complete(_run_ws_server())
    finally:
        _ws_loop.close()


def start_camera_servers_once() -> None:
    """
    Compatibility entrypoint used by app.py.
    Starts HTTP + WebSocket servers exactly once in daemon threads.
    """
    global _servers_started
    with _servers_lock:
        if _servers_started:
            return
        _servers_started = True

        print("🚀 [ML] Starting face-recognition preload...", flush=True)
        # Preload is asynchronous and non-blocking. An empty known_faces/
        # directory is a valid first-run state because registration can happen
        # after the service starts.
        try:
            start_process_preload()
        except Exception as exc:
            print(
                f"⚠️ [ML] Preload scheduling warning: {exc}",
                flush=True,
            )
        print_ml_status("START")

        http_thread = threading.Thread(
            target=_run_http_server,
            daemon=True,
            name="camera-http-server",
        )
        ws_thread = threading.Thread(
            target=_ws_thread_main,
            daemon=True,
            name="camera-ws-server",
        )
        http_thread.start()
        ws_thread.start()


def start_camera_servers() -> None:
    start_camera_servers_once()


def stop_camera_servers() -> None:
    global _servers_started, _http_server
    with _servers_lock:
        if _http_server is not None:
            try:
                _http_server.shutdown()
            except Exception:
                pass
            try:
                _http_server.server_close()
            except Exception:
                pass
            _http_server = None
        _servers_started = False


# Start preloading when module is imported only if explicitly requested by app.py.
__all__ = [
    "start_camera_servers_once",
    "start_camera_servers",
    "stop_camera_servers",
    "start_process_preload",
    "get_face_recognizer",
    "process_frame",
    "save_result",
    "post_attendance_evidence",
    "reset_result_state",
    "exchange_code_with_laravel",
    "set_hris_user",
    "get_hris_user",
    "clear_hris_user",
    "CameraHandler",
]



