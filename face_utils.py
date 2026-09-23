from __future__ import annotations

from collections import deque
import math
import os
import pickle
import re
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any


class FaceRecognizer:
    """
    Simple realtime face recognition for HRIS attendance.

    Design:
        YuNet detection
            -> frontal-pose gate
            -> passive anti-spoof / PAD
            -> 5-point frontal alignment (112x112)
            -> DeepFace ArcFace embedding
            -> cosine similarity against target + competing gallery samples
            -> target top-3 agreement + best-similarity + identity margin
            -> TARGET or Unknown

    Identity recognition remains similarity-only and keeps the existing
    ArcFace/gallery decision logic unchanged. A separate temporal liveness
    gate is added before identity acceptance. The liveness gate uses only
    YuNet's existing five facial landmarks and face-box geometry over time;
    it does not alter ArcFace embeddings, gallery thresholds, top-3 logic,
    or identity-margin logic.

    The liveness gate is designed to reject a static photo/screenshot by
    requiring a short natural head-motion sequence and a return toward the
    starting position.

    The identity decision itself does not use:
        - pose buckets
        - prototype embeddings
        - temporal embedding averaging
        - learned classifiers
        - non-face behavioral signals

    The main anti-false-accept change is that one unusually similar gallery
    photo is no longer sufficient. The logged-in HRIS target must:
        1. reach a minimum ArcFace similarity;
        2. have support from its top-3 reference similarities; and
        3. beat the strongest competing identity by a minimum margin.

    camera_server.py remains responsible for the existing 3/3 temporal
    identity confirmation before attendance is finally saved. That 3/3
    confirmation does not require head movement.

    Gallery:
        known_faces/
        ├── Darren/
        │   ├── ...
        └── Hosea/
            ├── ...

    In the user's current setup, the intended reference set is:
        Darren: 14 photos
        Hosea : 10 photos
        Total  : 24 photos

    The code does not delete or force a fixed count. It loads every valid image
    on disk and prints the actual count at startup.
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    # ------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------
    MODEL_NAME = "ArcFace"
    MATCH_METRIC = "cosine"
    ARC_FACE_INPUT_SIZE = 112
    ARC_FACE_EMBEDDING_SIZE = 512

    # Expected dataset counts are diagnostics only.
    EXPECTED_REFERENCE_COUNTS = {
        "Darren": 14,
        "Hosea": 10,
    }

    # ------------------------------------------------------------------
    # YuNet
    # ------------------------------------------------------------------
    YUNET_MODEL_FILENAME = "face_detection_yunet_2023mar_int8.onnx"
    YUNET_DOWNLOAD_URL = (
        "https://huggingface.co/opencv/opencv_zoo/resolve/main/"
        "models/face_detection_yunet/face_detection_yunet_2023mar_int8.onnx"
    )

    YUNET_SCORE_THRESHOLD = 0.40
    YUNET_NMS_THRESHOLD = 0.30
    YUNET_TOP_K = 10
    MAX_FACES = 1

    # Keep detection inexpensive even when the browser sends 1080p/720p frames.
    # YuNet is run on a resized working frame and the coordinates are mapped
    # back to the original camera resolution.
    YUNET_FAST_MAX_SIDE = 640
    YUNET_FALLBACK_MAX_SIDE = 960
    SMALL_FACE_WIDTH = 70
    SMALL_FACE_HEIGHT = 70
    BOX_SMOOTH_ALPHA = 0.92

    # ------------------------------------------------------------------
    # FACE QUALITY
    # ------------------------------------------------------------------
    MIN_FACE_WIDTH = 85
    MIN_FACE_HEIGHT = 85
    MIN_BLUR_SCORE = 55.0
    MIN_QUALITY_SCORE = 0.62
    MIN_BRIGHTNESS = 30.0
    MAX_BRIGHTNESS = 280.0
    MIN_DETECTOR_SCORE = 0.58

    # ------------------------------------------------------------------
    # PASSIVE ANTI-SPOOF / PRESENTATION ATTACK DETECTION (PAD)
    # ------------------------------------------------------------------
    # The user does NOT need to move. The production gate is passive:
    # the face must be frontal and the anti-spoofing model must classify
    # several consecutive camera evaluations as a real face.
    #
    # The previous landmark-motion liveness settings remain in the class
    # for compatibility, but they are no longer used as the security gate.
    LIVENESS_ENABLED = False

    ANTI_SPOOF_ENABLED = True
    # IMPORTANT:
    # Do not call DeepFace.extract_faces(... detector_backend="opencv")
    # for PAD. That path can invoke OpenCV Haar cascade and fail when
    # cv2/data/haarcascade_frontalface_default.xml is unavailable.
    #
    # PAD now calls DeepFace's Fasnet anti-spoof model directly with the
    # already-known YuNet bounding box. YuNet remains the only face detector.
    ANTI_SPOOF_MODEL_NAME = "Fasnet"
    ANTI_SPOOF_REQUIRED_CONSECUTIVE = 3
    # Do not run the relatively heavy PAD network on every camera frame.
    # Three real PAD inferences are still required, but they are spaced out
    # so the browser camera remains responsive.
    ANTI_SPOOF_INTERVAL_SECONDS = 0.35

    # Use a padded crop from the original camera frame so the PAD model gets
    # facial detail plus a small amount of surrounding context.
    ANTI_SPOOF_PAD_X = 0.25
    ANTI_SPOOF_PAD_Y = 0.30

    # ------------------------------------------------------------------
    # SIMILARITY-ONLY TARGET VERIFICATION
    # ------------------------------------------------------------------
    # A single high match to one gallery photo is not enough. The live face
    # must be supported by the strongest few target samples and must separate
    # from the strongest competing identity.
    TOP_K_TARGET_SAMPLES = 3

    # Thresholds are calibrated PER IDENTITY using ONLY known_faces/.
    #
    # The calibration is leave-one-out:
    #   - a gallery image is used as the query;
    #   - its own embedding is excluded from the genuine comparison;
    #   - the query is compared against the target identity and competing
    #     identities;
    #   - the three production gates (best similarity, top-3 support, margin)
    #     are calibrated jointly.
    #
    # This is an internal gallery calibration, not a held-out production
    # accuracy estimate. No validation_faces/ directory is required and no
    # validation JSON file is generated.
    CALIBRATE_THRESHOLDS = True

    # Security-first joint calibration:
    # among candidate threshold triples that retain at least this fraction
    # of known-face genuine leave-one-out queries, minimize false accepts.
    CALIBRATION_MIN_GENUINE_TPR = 0.70
    CALIBRATION_SECURITY_FIRST = True

    # Conservative floors: automatic calibration may increase these values,
    # but never lower them.
    MIN_TARGET_MAX_SIMILARITY = 0.60
    MIN_TARGET_TOPK_MEAN = 0.55
    MIN_TARGET_MARGIN = 0.08

    MAX_CALIBRATED_MAX_SIMILARITY = 0.90
    MAX_CALIBRATED_TOPK_MEAN = 0.90
    MAX_CALIBRATED_MARGIN = 0.50

    DEFAULT_MAX_SIMILARITY = 0.65
    DEFAULT_TOPK_MEAN = 0.58
    DEFAULT_MARGIN = 0.08

    # ------------------------------------------------------------------
    # FRONTAL ALIGNMENT
    # ------------------------------------------------------------------
    # YuNet landmark order:
    # right eye, left eye, nose, right mouth, left mouth.
    #
    # This template is the standard ArcFace 112x112 landmark layout with the
    # corresponding order used by YuNet.
    ALIGN_TARGET = (
        (73.5318, 51.5014),  # right eye
        (38.2946, 51.6963),  # left eye
        (56.0252, 71.7366),  # nose
        (70.7299, 92.2041),  # right mouth
        (41.5493, 92.3655),  # left mouth
    )

    # A second, simple crop is used only as an optional fallback for webcam
    # framing. It is NOT fused into an embedding. We compare both and keep the
    # higher ArcFace similarity.
    USE_FRONTAL_ALT_CROP = True
    ALT_PAD_X = 0.16
    ALT_PAD_Y = 0.18

    # ------------------------------------------------------------------
    # CACHE
    # ------------------------------------------------------------------
    CACHE_VERSION = 402
    CACHE_FILENAME = ".face_embeddings_cache.pkl"

    _runtime_configured = False
    _runtime_config_lock = threading.Lock()

    def __init__(self) -> None:
        self.base_dir = Path(__file__).resolve().parent
        self.known_faces_dir = self.base_dir / "known_faces"
        self.models_dir = self.base_dir / "models"

        self.known_faces_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.yunet_model_path = self.models_dir / self.YUNET_MODEL_FILENAME
        self.embedding_cache_path = self.base_dir / self.CACHE_FILENAME

        self._deepface: Any = None
        self._arcface_model: Any = None
        self._anti_spoof_model: Any = None
        self._yunet: Any = None
        self._yunet_input_size: tuple[int, int] | None = None

        self._model_ready = False
        self._detector_ready = False
        self._embeddings_ready = False

        self._embedding_names: list[str] = []
        self._embedding_paths: list[str] = []
        self._embedding_matrix: Any = None
        self._identity_to_indices: dict[str, list[int]] = {}
        self._identity_names: list[str] = []

        self._identity_thresholds: dict[str, float] = {}
        self._identity_calibration: dict[str, dict[str, float]] = {}
        self._identity_topk_thresholds: dict[str, float] = {}
        self._identity_margin_thresholds: dict[str, float] = {}
        self._calibration_source = "known_faces_leave_one_out"

        self._embedding_signature: Any = None

        self._loading = False
        self._loading_stage = "Idle"
        self._loading_error: str | None = None
        self._prepare_thread: threading.Thread | None = None
        self._ready_event = threading.Event()
        self._lock = threading.RLock()

        self._smoothed_box: tuple[float, float, float, float] | None = None
        self._last_detection: dict | None = None

        # Temporal liveness state. This is deliberately separate from the
        # ArcFace identity state so existing recognition thresholds stay intact.
        self._liveness_target = "Unknown"
        self._liveness_started_at = 0.0
        self._liveness_last_seen = 0.0
        self._liveness_baseline: dict[str, float] | None = None
        self._liveness_phase = "idle"
        self._liveness_reason = "not_started"
        self._liveness_live_until = 0.0
        self._liveness_samples: deque[dict[str, float]] = deque(maxlen=12)

        # Passive anti-spoof runtime state.
        self._anti_spoof_target = "Unknown"
        self._anti_spoof_consecutive = 0
        self._anti_spoof_last_score = 0.0
        self._anti_spoof_last_ok = False
        self._anti_spoof_last_reason = "not_started"
        self._anti_spoof_last_at = 0.0
        self._anti_spoof_last_inference_at = 0.0

        # Last recognition diagnostics.
        self._last_match_name = "Unknown"
        self._last_similarity = 0.0
        self._last_face_confident = False
        self._last_quality_ok = False
        self._last_quality_score = 0.0
        self._last_target_identity = "Unknown"
        self._last_target_threshold = self.DEFAULT_MAX_SIMILARITY
        self._last_target_sample_count = 0
        self._last_target_topk_mean = 0.0
        self._last_target_margin = 0.0
        self._last_competitor_best = 0.0
        self._last_target_topk_threshold = self.DEFAULT_TOPK_MEAN
        self._last_target_margin_threshold = self.DEFAULT_MARGIN
        self._last_gallery_best_identity = "Unknown"
        self._last_gallery_best_similarity = 0.0
        self._last_gallery_runner_up_identity = "Unknown"
        self._last_gallery_runner_up_similarity = 0.0
        self._last_gallery_margin = 0.0
        self._last_decision_reason = "not_evaluated"
        self._last_decision_mode = "similarity_only"

        self._last_liveness_ok = False
        self._last_liveness_phase = "idle"
        self._last_liveness_reason = "not_started"
        self._last_liveness_score = 0.0

        self.known_names: list[str] = []
        self.known_paths: list[str] = []
        self._index_dataset()

    # ================================================================
    # Naming / dataset
    # ================================================================

    @staticmethod
    def format_person_name(value: str | None) -> str:
        value = str(value or "").strip()
        if not value:
            return "Unknown"
        return " ".join(
            word[:1].upper() + word[1:].lower()
            for word in value.replace("_", " ").split()
            if word
        ) or "Unknown"

    @classmethod
    def clean_name_from_folder(cls, folder: str | Path) -> str:
        return cls.format_person_name(Path(folder).name)

    def _get_gallery_files(self) -> list[tuple[str, Path]]:
        result: list[tuple[str, Path]] = []
        if not self.known_faces_dir.exists():
            return result

        for person_dir in sorted(
            self.known_faces_dir.iterdir(),
            key=lambda p: p.name.lower(),
        ):
            if not person_dir.is_dir():
                continue

            identity = self.clean_name_from_folder(person_dir)
            if identity == "Unknown":
                continue

            for path in sorted(
                person_dir.rglob("*"),
                key=lambda p: str(p).lower(),
            ):
                if path.is_file() and path.suffix.lower() in self.EXTENSIONS:
                    result.append((identity, path))

        return result

    def _index_dataset(self) -> None:
        files = self._get_gallery_files()
        with self._lock:
            self.known_paths = [str(path) for _, path in files]
            self.known_names = [identity for identity, _ in files]

    def get_dataset_file_count(self) -> int:
        return len(self._get_gallery_files())

    def get_known_names(self) -> list[str]:
        return sorted({identity for identity, _ in self._get_gallery_files()})

    def get_known_paths(self) -> list[str]:
        return [str(path) for _, path in self._get_gallery_files()]

    def get_dataset_names_from_disk(self) -> list[str]:
        return self.get_known_names()

    def get_total_encodings(self) -> int:
        with self._lock:
            return len(self._embedding_names)

    # Compatibility helpers retained so existing dashboard/app code does not
    # break. This final build intentionally does not use validation_faces/.
    def get_validation_file_count(self) -> int:
        return 0

    def get_validation_names(self) -> list[str]:
        return []

    # ================================================================
    # Status
    # ================================================================

    def _set_stage(self, stage: str) -> None:
        changed = False
        with self._lock:
            if self._loading_stage != stage:
                self._loading_stage = stage
                changed = True
        if changed:
            print(f"🧠 [ML] {stage}", flush=True)

    def is_loading(self) -> bool:
        with self._lock:
            return self._loading

    def get_loading_status(self) -> str:
        with self._lock:
            return self._loading_error or self._loading_stage

    def get_loading_error(self) -> str | None:
        with self._lock:
            return self._loading_error

    def get_loading_progress(self) -> int:
        status = self.get_loading_status()
        if status == "Face recognition ready":
            return 100

        if status.startswith("Building ArcFace embeddings"):
            match = re.search(r"\((\d+)/(\d+)\)", status)
            if match:
                current = int(match.group(1))
                total = max(1, int(match.group(2)))
                return min(98, 80 + int(18 * current / total))
            return 80

        mapping = {
            "Idle": 0,
            "Preparing ArcFace + gallery...": 10,
            "Checking face detector...": 5,
            "Downloading YuNet detector...": 10,
            "Loading YuNet detector...": 18,
            "Importing DeepFace...": 30,
            "Loading ArcFace model...": 55,
            "Loading cached ArcFace embeddings...": 75,
            "Building ArcFace embeddings...": 80,
            "Calibrating similarity thresholds...": 94,
            "Face recognition ready": 100,
        }
        return mapping.get(status, 40)

    def is_model_ready(self) -> bool:
        with self._lock:
            return self._model_ready

    def is_detector_ready(self) -> bool:
        with self._lock:
            return self._detector_ready

    def is_embeddings_ready(self) -> bool:
        with self._lock:
            return self._embeddings_ready

    def is_ready(self) -> bool:
        with self._lock:
            return (
                self._model_ready
                and self._detector_ready
                and self._embeddings_ready
            )

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        if self.is_ready():
            return True
        self._ready_event.wait(timeout=timeout)
        return self.is_ready()

    def get_runtime_status(self) -> dict:
        with self._lock:
            return {
                "deepface_loaded": self._deepface is not None,
                "arcface_loaded": self._model_ready and self._arcface_model is not None,
                # Keep this legacy key so existing UI does not break.
                "sface_loaded": False,
                "yunet_loaded": self._detector_ready and self._yunet is not None,
                "embeddings_loaded": self._embeddings_ready,
                "ready": self.is_ready(),
                "loading": self._loading,
                "stage": self._loading_error or self._loading_stage,
                "gallery_samples": len(self._embedding_names),
                "gallery_identities": len(set(self._embedding_names)),
                "model_name": self.MODEL_NAME,
                "similarity_metric": self.MATCH_METRIC,
                "calibration_source": self._calibration_source,
                "thresholds": dict(self._identity_thresholds),
                # Legacy liveness keys remain so existing UI does not break.
                "liveness_enabled": False,
                "liveness_phase": "disabled_compatibility",
                "liveness_ok": bool(self._last_liveness_ok),
                "anti_spoofing_enabled": bool(self.ANTI_SPOOF_ENABLED),
                "anti_spoofing_ready": self._anti_spoof_model is not None,
                "anti_spoof_ok": bool(self._anti_spoof_last_ok),
                "anti_spoof_score": float(self._anti_spoof_last_score),
                "anti_spoof_consecutive": int(self._anti_spoof_consecutive),
                "anti_spoof_required_consecutive": int(self.ANTI_SPOOF_REQUIRED_CONSECUTIVE),
                "anti_spoof_interval_seconds": float(self.ANTI_SPOOF_INTERVAL_SECONDS),
                "anti_spoof_reason": self._anti_spoof_last_reason,
                "yunet_fast_max_side": int(self.YUNET_FAST_MAX_SIDE),
            }

    def print_runtime_status(self) -> None:
        status = self.get_runtime_status()
        print(
            "🧠 [ML STATUS] "
            f"DeepFace={'LOADED' if status['deepface_loaded'] else 'NOT LOADED'} | "
            f"ArcFace={'LOADED' if status['arcface_loaded'] else 'NOT LOADED'} | "
            f"YuNet={'LOADED' if status['yunet_loaded'] else 'NOT LOADED'} | "
            f"Gallery={'LOADED' if status['embeddings_loaded'] else 'NOT LOADED'} | "
            f"Samples={status['gallery_samples']} | "
            f"Ready={'YES' if status['ready'] else 'NO'}",
            flush=True,
        )

    # ================================================================
    # Runtime / images
    # ================================================================

    @classmethod
    def _configure_runtime(cls) -> None:
        if cls._runtime_configured:
            return

        with cls._runtime_config_lock:
            if cls._runtime_configured:
                return

            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
            os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("KMP_BLOCKTIME", "0")

            cls._runtime_configured = True

    @staticmethod
    def _read_image(path: str) -> Any:
        try:
            import cv2

            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is not None:
                return image

            import numpy as np

            raw = np.fromfile(path, dtype=np.uint8)
            if raw.size == 0:
                return None
            return cv2.imdecode(raw, cv2.IMREAD_COLOR)
        except Exception:
            return None

    # ================================================================
    # YuNet
    # ================================================================

    def _find_local_yunet(self) -> Path | None:
        candidates = [
            self.yunet_model_path,
            self.base_dir / self.YUNET_MODEL_FILENAME,
        ]
        for path in candidates:
            try:
                if path.exists() and path.is_file() and path.stat().st_size > 10_000:
                    return path
            except OSError:
                pass
        return None

    def _download_yunet(self) -> Path | None:
        existing = self._find_local_yunet()
        if existing:
            return existing

        self._set_stage("Downloading YuNet detector...")
        target = self.yunet_model_path
        temp = target.with_suffix(target.suffix + ".tmp")

        try:
            request = urllib.request.Request(
                self.YUNET_DOWNLOAD_URL,
                headers={"User-Agent": "HR-Attendance-System/3.0"},
            )
            with urllib.request.urlopen(request, timeout=30) as response, temp.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)

            if not temp.exists() or temp.stat().st_size <= 10_000:
                raise RuntimeError("YuNet download incomplete")

            temp.replace(target)
            return target

        except Exception as exc:
            print(f"⚠️ YuNet download error: {exc}", flush=True)
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass
            return None

    def _create_yunet(self, model_path: Path) -> Any:
        import cv2

        factory = getattr(cv2, "FaceDetectorYN_create", None)
        if factory:
            return factory(
                str(model_path),
                "",
                (320, 320),
                self.YUNET_SCORE_THRESHOLD,
                self.YUNET_NMS_THRESHOLD,
                self.YUNET_TOP_K,
            )

        detector_cls = getattr(cv2, "FaceDetectorYN", None)
        if detector_cls is None or not hasattr(detector_cls, "create"):
            raise RuntimeError("OpenCV FaceDetectorYN tidak tersedia")

        return detector_cls.create(
            str(model_path),
            "",
            (320, 320),
            self.YUNET_SCORE_THRESHOLD,
            self.YUNET_NMS_THRESHOLD,
            self.YUNET_TOP_K,
        )

    def _load_yunet(self) -> bool:
        with self._lock:
            if self._detector_ready and self._yunet is not None:
                return True

        model_path = self._find_local_yunet() or self._download_yunet()
        if model_path is None:
            with self._lock:
                self._loading_error = "YuNet model tidak ditemukan."
            return False

        try:
            self._set_stage("Loading YuNet detector...")
            detector = self._create_yunet(model_path)
            with self._lock:
                self._yunet = detector
                self._yunet_input_size = None
                self._detector_ready = True
                self._loading_error = None

            print(f"✅ [YUNET] Detector loaded: {model_path}", flush=True)
            return True

        except Exception as exc:
            with self._lock:
                self._yunet = None
                self._detector_ready = False
                self._loading_error = f"YuNet gagal dimuat: {exc}"
            print(f"❌ YuNet error: {exc}", flush=True)
            return False

    def _detect_on_image(
        self,
        frame: Any,
        max_side: int | None = None,
    ) -> list[dict]:
        """Run YuNet on a bounded working resolution and map boxes back."""
        import cv2

        with self._lock:
            detector = self._yunet

        if detector is None or frame is None:
            return []

        try:
            height, width = frame.shape[:2]
            work = frame
            scale = 1.0

            limit = int(max_side or self.YUNET_FAST_MAX_SIDE)
            largest = max(width, height)
            if largest > limit:
                scale = float(limit) / float(largest)
                work = cv2.resize(
                    frame,
                    (max(1, int(round(width * scale))),
                     max(1, int(round(height * scale)))),
                    interpolation=cv2.INTER_AREA,
                )

            work_h, work_w = work.shape[:2]
            with self._lock:
                last_size = getattr(self, "_yunet_input_size", None)
            current_size = (int(work_w), int(work_h))
            if last_size != current_size:
                detector.setInputSize(current_size)
                with self._lock:
                    self._yunet_input_size = current_size

            _, faces = detector.detect(work)
            if faces is None:
                return []

            results: list[dict] = []

            inv_scale = 1.0 / scale if scale != 1.0 else 1.0
            for row in faces:
                if len(row) < 15:
                    continue

                score = float(row[-1])
                if score < self.YUNET_SCORE_THRESHOLD:
                    continue

                x, y, w, h = map(float, row[:4])
                x *= inv_scale
                y *= inv_scale
                w *= inv_scale
                h *= inv_scale

                if w <= 0 or h <= 0:
                    continue

                landmarks = [
                    (float(row[4]) * inv_scale, float(row[5]) * inv_scale),
                    (float(row[6]) * inv_scale, float(row[7]) * inv_scale),
                    (float(row[8]) * inv_scale, float(row[9]) * inv_scale),
                    (float(row[10]) * inv_scale, float(row[11]) * inv_scale),
                    (float(row[12]) * inv_scale, float(row[13]) * inv_scale),
                ]

                x1 = max(0, min(int(round(x)), width - 1))
                y1 = max(0, min(int(round(y)), height - 1))
                x2 = max(x1 + 1, min(int(round(x + w)), width))
                y2 = max(y1 + 1, min(int(round(y + h)), height))

                results.append(
                    {
                        "x": x1,
                        "y": y1,
                        "w": x2 - x1,
                        "h": y2 - y1,
                        "score": score,
                        "landmarks": landmarks,
                    }
                )

            results.sort(key=lambda item: item["score"], reverse=True)
            return results[: self.MAX_FACES]

        except Exception as exc:
            print(f"⚠️ YuNet detection error: {exc}", flush=True)
            return []

    def _detect_face_best_scale(self, frame: Any) -> list[dict]:
        """Fast first pass, then one higher-resolution fallback only if needed."""
        native = self._detect_on_image(
            frame,
            max_side=self.YUNET_FAST_MAX_SIDE,
        )
        native_best = native[0] if native else None

        if native_best is not None and (
            native_best["score"] >= self.MIN_DETECTOR_SCORE
            and native_best["w"] >= self.SMALL_FACE_WIDTH
            and native_best["h"] >= self.SMALL_FACE_HEIGHT
        ):
            return native

        # Fallback does NOT resize the image 2x and then resize it back again.
        # Instead, it simply gives YuNet a larger working side once.
        fallback = self._detect_on_image(
            frame,
            max_side=self.YUNET_FALLBACK_MAX_SIDE,
        )
        fallback_best = fallback[0] if fallback else None

        if fallback_best is not None:
            if native_best is None:
                return fallback
            if fallback_best["score"] > native_best["score"]:
                return fallback

        return native

    # ================================================================
    # Tracking
    # ================================================================

    def _smooth_box(
        self,
        detection: dict,
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int]:
        height, width = frame_shape[:2]
        current = (
            float(detection["x"]),
            float(detection["y"]),
            float(detection["w"]),
            float(detection["h"]),
        )

        previous = self._smoothed_box
        if previous is None:
            result = current
        else:
            alpha = self.BOX_SMOOTH_ALPHA
            result = tuple(
                alpha * current[i] + (1.0 - alpha) * previous[i]
                for i in range(4)
            )

        x, y, w, h = result
        x = max(0.0, min(x, float(width - 1)))
        y = max(0.0, min(y, float(height - 1)))
        w = max(1.0, min(w, float(width) - x))
        h = max(1.0, min(h, float(height) - y))

        self._smoothed_box = (x, y, w, h)
        return int(round(x)), int(round(y)), int(round(w)), int(round(h))

    def track_face(
        self,
        frame: Any,
    ) -> tuple[dict | None, tuple[int, int, int, int] | None]:
        if frame is None or not self.is_detector_ready():
            return None, None

        detections = self._detect_face_best_scale(frame)

        if not detections:
            self._smoothed_box = None
            self._last_detection = None
            with self._lock:
                self._reset_liveness_locked(reason="face_lost")
                self._reset_anti_spoof_locked(reason="face_lost")
            return None, None

        detection = detections[0]
        self._last_detection = dict(detection)

        stable = self._smooth_box(detection, frame.shape)
        return detection, stable

    # ================================================================
    # PASSIVE ANTI-SPOOF / FRONTAL GATE
    # ================================================================

    @staticmethod
    def _frontal_pose_check(
        detection: dict,
    ) -> tuple[bool, float, str]:
        """
        Passive frontal gate. No head movement is required.

        This is intentionally a moderate frontal policy: the user only has
        to look at the camera, not achieve mathematically perfect symmetry.
        """
        try:
            import math

            landmarks = detection.get("landmarks", [])
            if not isinstance(landmarks, (list, tuple)) or len(landmarks) != 5:
                return False, 0.0, "landmarks_invalid"

            right_eye, left_eye, nose, right_mouth, left_mouth = landmarks

            rx, ry = float(right_eye[0]), float(right_eye[1])
            lx, ly = float(left_eye[0]), float(left_eye[1])
            nx, ny = float(nose[0]), float(nose[1])
            rmx, rmy = float(right_mouth[0]), float(right_mouth[1])
            lmx, lmy = float(left_mouth[0]), float(left_mouth[1])

            eye_dx = lx - rx
            eye_dy = ly - ry
            eye_span = math.hypot(eye_dx, eye_dy)
            if eye_span < 4.0:
                return False, 0.0, "eye_span_too_small"

            eye_cx = (rx + lx) * 0.5
            mouth_cx = (rmx + lmx) * 0.5

            roll_deg = abs(math.degrees(math.atan2(eye_dy, eye_dx)))
            nose_offset = abs(nx - eye_cx) / eye_span
            mouth_offset = abs(mouth_cx - eye_cx) / eye_span

            max_roll = 8.0
            max_nose_offset = 0.16
            max_mouth_offset = 0.20

            score = (
                max(0.0, 1.0 - roll_deg / max_roll)
                + max(0.0, 1.0 - nose_offset / max_nose_offset)
                + max(0.0, 1.0 - mouth_offset / max_mouth_offset)
            ) / 3.0

            if roll_deg > max_roll:
                return False, float(score), "roll_not_frontal"
            if nose_offset > max_nose_offset:
                return False, float(score), "nose_not_centered"
            if mouth_offset > max_mouth_offset:
                return False, float(score), "mouth_not_centered"

            return True, float(score), "frontal_ok"

        except Exception:
            return False, 0.0, "frontal_check_error"

    def _anti_spoof_crop(self, frame: Any, detection: dict) -> Any:
        """Crop the original camera frame around the YuNet face."""
        try:
            height, width = frame.shape[:2]
            x = float(detection["x"])
            y = float(detection["y"])
            w = float(detection["w"])
            h = float(detection["h"])

            pad_x = self.ANTI_SPOOF_PAD_X * w
            pad_y = self.ANTI_SPOOF_PAD_Y * h

            left = max(0, int(round(x - pad_x)))
            top = max(0, int(round(y - pad_y)))
            right = min(width, int(round(x + w + pad_x)))
            bottom = min(height, int(round(y + h + pad_y)))

            crop = frame[top:bottom, left:right]
            if crop is None or crop.size == 0:
                return None

            return crop

        except Exception:
            return None

    def _reset_anti_spoof_locked(
        self,
        target: str | None = None,
        reason: str = "reset",
    ) -> None:
        self._anti_spoof_target = (
            self.format_person_name(target)
            if target is not None
            else "Unknown"
        )
        self._anti_spoof_consecutive = 0
        self._anti_spoof_last_score = 0.0
        self._anti_spoof_last_ok = False
        self._anti_spoof_last_reason = reason
        self._anti_spoof_last_at = 0.0
        self._anti_spoof_last_inference_at = 0.0

    def reset_anti_spoof(self, reason: str = "manual_reset") -> None:
        with self._lock:
            self._reset_anti_spoof_locked(reason=reason)

    def _load_anti_spoof_model(self) -> Any:
        """
        Load DeepFace Fasnet directly.

        This deliberately bypasses DeepFace.extract_faces(), because that
        function may build the OpenCV Haar detector when detector_backend="opencv".
        Our YuNet detector already supplied the face bounding box, so there is
        no reason to run a second detector for PAD.

        DeepFace's current detection module itself uses:
            modeling.build_model(task="spoofing", model_name="Fasnet")
        followed by:
            antispoof_model.analyze(img=img, facial_area=(x, y, w, h))

        Therefore this path uses the same official Fasnet PAD model without
        the OpenCV Haar cascade dependency.
        """
        with self._lock:
            if self._anti_spoof_model is not None:
                return self._anti_spoof_model

        DeepFace = self._load_deepface()
        if DeepFace is None:
            return None

        try:
            print(
                f"⏳ [PAD] Loading DeepFace {self.ANTI_SPOOF_MODEL_NAME} model...",
                flush=True,
            )

            model = DeepFace.build_model(
                task="spoofing",
                model_name=self.ANTI_SPOOF_MODEL_NAME,
            )

            with self._lock:
                self._anti_spoof_model = model

            print(
                f"✅ [PAD] {self.ANTI_SPOOF_MODEL_NAME} anti-spoof model loaded.",
                flush=True,
            )
            return model

        except Exception as exc:
            with self._lock:
                self._anti_spoof_model = None

            print(
                f"❌ [PAD] Failed to load {self.ANTI_SPOOF_MODEL_NAME}: {exc}",
                flush=True,
            )
            return None

    def _check_anti_spoof(
        self,
        frame: Any,
        detection: dict,
        target: str,
    ) -> tuple[bool, float, str]:
        """
        Run passive PAD directly through DeepFace Fasnet.

        No head movement is required.

        Critical difference from the previous implementation:
          * YuNet is the only face detector.
          * DeepFace.extract_faces() is NOT called.
          * OpenCV Haar cascade is NOT needed for PAD.
          * Fasnet receives the original camera frame plus the YuNet bbox.

        This follows DeepFace's own current anti-spoof implementation pattern.
        """
        if not self.ANTI_SPOOF_ENABLED:
            return True, 1.0, "disabled"

        target = self.format_person_name(target)
        now = time.monotonic()

        with self._lock:
            if self._anti_spoof_target != target:
                self._reset_anti_spoof_locked(
                    target=target,
                    reason="target_changed",
                )

            # Avoid running the relatively heavy PAD model on every camera
            # evaluation. Cached results do not increase the consecutive count.
            elapsed = now - self._anti_spoof_last_inference_at
            if (
                self._anti_spoof_last_inference_at > 0
                and elapsed < self.ANTI_SPOOF_INTERVAL_SECONDS
            ):
                cached_reason = (
                    "pad_passed_cached"
                    if self._anti_spoof_last_ok
                    else self._anti_spoof_last_reason
                )
                return (
                    bool(self._anti_spoof_last_ok),
                    float(self._anti_spoof_last_score),
                    cached_reason,
                )

        # --------------------------------------------------------------
        # Validate the YuNet bbox. No secondary detector is used.
        # --------------------------------------------------------------
        try:
            x = int(round(float(detection["x"])))
            y = int(round(float(detection["y"])))
            w = int(round(float(detection["w"])))
            h = int(round(float(detection["h"])))

            frame_h, frame_w = frame.shape[:2]
            x = max(0, min(x, frame_w - 1))
            y = max(0, min(y, frame_h - 1))
            w = max(1, min(w, frame_w - x))
            h = max(1, min(h, frame_h - y))

        except Exception:
            with self._lock:
                self._anti_spoof_consecutive = 0
                self._anti_spoof_last_ok = False
                self._anti_spoof_last_reason = "pad_bbox_invalid"
                self._anti_spoof_last_score = 0.0
                self._anti_spoof_last_at = now
                self._anti_spoof_last_inference_at = now

            return False, 0.0, "pad_bbox_invalid"

        if w < 40 or h < 40:
            with self._lock:
                self._anti_spoof_consecutive = 0
                self._anti_spoof_last_ok = False
                self._anti_spoof_last_reason = "pad_face_too_small"
                self._anti_spoof_last_score = 0.0
                self._anti_spoof_last_at = now
                self._anti_spoof_last_inference_at = now

            return False, 0.0, "pad_face_too_small"

        anti_spoof_model = self._load_anti_spoof_model()
        if anti_spoof_model is None:
            with self._lock:
                self._anti_spoof_consecutive = 0
                self._anti_spoof_last_ok = False
                self._anti_spoof_last_reason = "pad_model_not_loaded"
                self._anti_spoof_last_score = 0.0
                self._anti_spoof_last_at = now
                self._anti_spoof_last_inference_at = now

            return False, 0.0, "pad_model_not_loaded"

        try:
            # DeepFace's Fasnet internally makes its own enlarged crops from
            # this exact facial rectangle (two crop scales), so using the
            # original frame + YuNet bbox is preferable to running a second
            # face detector on an already-cropped image.
            is_real, score = anti_spoof_model.analyze(
                img=frame,
                facial_area=(x, y, w, h),
            )

            is_real = bool(is_real)

            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0

            reason = "pad_real" if is_real else "pad_spoof"

            with self._lock:
                self._anti_spoof_last_score = float(score)
                self._anti_spoof_last_at = now
                self._anti_spoof_last_inference_at = now

                if is_real:
                    self._anti_spoof_consecutive += 1
                else:
                    self._anti_spoof_consecutive = 0

                ok = (
                    self._anti_spoof_consecutive
                    >= self.ANTI_SPOOF_REQUIRED_CONSECUTIVE
                )

                self._anti_spoof_last_ok = bool(ok)
                self._anti_spoof_last_reason = (
                    "pad_passed"
                    if ok
                    else (
                        "pad_real_warming_up"
                        if is_real
                        else reason
                    )
                )

                consecutive = self._anti_spoof_consecutive
                last_reason = self._anti_spoof_last_reason

            print(
                "🛡️ [PAD] "
                f"target={target} "
                f"real={'YES' if is_real else 'NO'} "
                f"score={score:.3f} "
                f"consecutive={consecutive}/"
                f"{self.ANTI_SPOOF_REQUIRED_CONSECUTIVE} "
                f"reason={last_reason}",
                flush=True,
            )

            return bool(ok), float(score), last_reason

        except Exception as exc:
            with self._lock:
                self._anti_spoof_consecutive = 0
                self._anti_spoof_last_ok = False
                self._anti_spoof_last_reason = (
                    f"pad_error:{type(exc).__name__}"
                )
                self._anti_spoof_last_score = 0.0
                self._anti_spoof_last_at = now
                self._anti_spoof_last_inference_at = now

            print(
                f"⚠️ [PAD] Fasnet inference failed closed: {exc}",
                flush=True,
            )

            return False, 0.0, "pad_error"

    # ================================================================
    # TEMPORAL LIVENESS / ANTI-SPOOF
    # ================================================================

    @staticmethod
    def _liveness_features(detection: dict) -> dict[str, float] | None:
        """
        Convert existing YuNet landmarks/box into normalized temporal features.

        No new detector is introduced. The features are:
          - nose position relative to eye center and eye span;
          - eye-line roll angle;
          - face scale derived from bounding-box area.

        These features are used only to determine whether a short, natural
        temporal movement occurred. ArcFace identity scoring is untouched.
        """
        try:
            landmarks = detection.get("landmarks", [])
            if not isinstance(landmarks, (list, tuple)) or len(landmarks) != 5:
                return None

            right_eye, left_eye, nose, _right_mouth, _left_mouth = landmarks

            rx, ry = float(right_eye[0]), float(right_eye[1])
            lx, ly = float(left_eye[0]), float(left_eye[1])
            nx, ny = float(nose[0]), float(nose[1])

            eye_dx = lx - rx
            eye_dy = ly - ry
            eye_span = math.hypot(eye_dx, eye_dy)
            if eye_span < 4.0:
                return None

            eye_cx = (rx + lx) * 0.5
            eye_cy = (ry + ly) * 0.5

            nose_x = (nx - eye_cx) / eye_span
            nose_y = (ny - eye_cy) / eye_span
            roll_deg = math.degrees(math.atan2(eye_dy, eye_dx))

            width = max(float(detection.get("w", 0.0)), 1.0)
            height = max(float(detection.get("h", 0.0)), 1.0)
            scale = math.sqrt(width * height)

            values = {
                "nose_x": float(nose_x),
                "nose_y": float(nose_y),
                "roll_deg": float(roll_deg),
                "scale": float(scale),
            }

            if not all(math.isfinite(v) for v in values.values()):
                return None

            return values

        except Exception:
            return None

    @staticmethod
    def _median_feature_values(
        samples: list[dict[str, float]],
    ) -> dict[str, float]:
        if not samples:
            return {
                "nose_x": 0.0,
                "nose_y": 0.0,
                "roll_deg": 0.0,
                "scale": 1.0,
            }

        result: dict[str, float] = {}
        for key in ("nose_x", "nose_y", "roll_deg", "scale"):
            values = sorted(float(sample[key]) for sample in samples)
            middle = len(values) // 2
            if len(values) % 2:
                result[key] = values[middle]
            else:
                result[key] = 0.5 * (values[middle - 1] + values[middle])
        return result

    @staticmethod
    def _safe_scale_ratio(current: float, baseline: float) -> float:
        if current <= 0 or baseline <= 0:
            return 0.0
        return abs(current / baseline - 1.0)

    def _reset_liveness_locked(
        self,
        target: str | None = None,
        reason: str = "reset",
    ) -> None:
        target_name = (
            self.format_person_name(target)
            if target is not None
            else self._liveness_target
        )

        self._liveness_target = target_name
        self._liveness_started_at = time.monotonic()
        self._liveness_last_seen = 0.0
        self._liveness_baseline = None
        self._liveness_phase = "warming_up"
        self._liveness_reason = reason
        self._liveness_live_until = 0.0
        self._liveness_samples.clear()

        self._last_liveness_ok = False
        self._last_liveness_phase = self._liveness_phase
        self._last_liveness_reason = reason
        self._last_liveness_score = 0.0

    def reset_liveness(self, reason: str = "manual_reset") -> None:
        with self._lock:
            self._reset_liveness_locked(reason=reason)
            self._reset_anti_spoof_locked(reason=reason)

    def _update_liveness(
        self,
        detection: dict,
        target: str,
    ) -> tuple[bool, str, str, float]:
        """
        Temporal liveness gate.

        Phase 1: collect a short baseline.
        Phase 2: require measurable natural movement.
        Phase 3: require a return toward the baseline.
        Phase 4: hold the live result briefly so the existing 3/3
                 ArcFace confirmation can complete without extra UX changes.

        A static photo normally remains in phase 1/2 and therefore cannot
        make face_confident=True.
        """
        if not self.LIVENESS_ENABLED:
            return True, "live", "disabled", 1.0

        now = time.monotonic()
        target = self.format_person_name(target)

        with self._lock:
            if self._liveness_target != target or self._liveness_phase == "idle":
                self._reset_liveness_locked(
                    target=target,
                    reason="baseline_start",
                )

            if (
                self._liveness_phase == "live"
                and now > self._liveness_live_until
            ):
                self._reset_liveness_locked(
                    target=target,
                    reason="liveness_expired",
                )

            features = self._liveness_features(detection)
            if features is None:
                self._last_liveness_ok = False
                self._last_liveness_phase = "invalid_features"
                self._last_liveness_reason = "landmarks_invalid"
                self._last_liveness_score = 0.0
                return False, "invalid_features", "landmarks_invalid", 0.0

            self._liveness_last_seen = now
            self._liveness_samples.append(features)

            if (
                self._liveness_started_at > 0
                and now - self._liveness_started_at > self.LIVENESS_WINDOW_SECONDS
                and self._liveness_phase != "live"
            ):
                self._reset_liveness_locked(
                    target=target,
                    reason="window_timeout_restart",
                )
                self._liveness_samples.append(features)

            if self._liveness_phase == "warming_up":
                if len(self._liveness_samples) >= self.LIVENESS_BASELINE_SAMPLES:
                    baseline_samples = list(
                        self._liveness_samples
                    )[: self.LIVENESS_BASELINE_SAMPLES]
                    self._liveness_baseline = self._median_feature_values(
                        baseline_samples
                    )
                    self._liveness_phase = "waiting_for_motion"
                    self._liveness_reason = "baseline_ready"

            baseline = self._liveness_baseline
            if baseline is None:
                self._last_liveness_ok = False
                self._last_liveness_phase = self._liveness_phase
                self._last_liveness_reason = self._liveness_reason
                self._last_liveness_score = 0.0
                return False, self._liveness_phase, self._liveness_reason, 0.0

            nose_dx = abs(features["nose_x"] - baseline["nose_x"])
            nose_dy = abs(features["nose_y"] - baseline["nose_y"])
            roll_delta = abs(features["roll_deg"] - baseline["roll_deg"])
            scale_ratio = self._safe_scale_ratio(
                features["scale"],
                baseline["scale"],
            )

            movement_flags = {
                "nose_x": nose_dx >= self.LIVENESS_MIN_NOSE_X_SHIFT,
                "nose_y": nose_dy >= self.LIVENESS_MIN_NOSE_Y_SHIFT,
                "roll": roll_delta >= self.LIVENESS_MIN_ROLL_SHIFT_DEG,
                "scale": scale_ratio >= self.LIVENESS_MIN_SCALE_RATIO,
            }

            movement_count = sum(bool(v) for v in movement_flags.values())

            movement_score = min(
                1.0,
                max(
                    nose_dx / self.LIVENESS_MIN_NOSE_X_SHIFT,
                    nose_dy / self.LIVENESS_MIN_NOSE_Y_SHIFT,
                    roll_delta / self.LIVENESS_MIN_ROLL_SHIFT_DEG,
                    scale_ratio / self.LIVENESS_MIN_SCALE_RATIO,
                ) / 2.0,
            )

            if self._liveness_phase == "waiting_for_motion":
                if movement_count >= 1:
                    self._liveness_phase = "movement_detected"
                    self._liveness_reason = "movement_detected"
                else:
                    self._last_liveness_ok = False
                    self._last_liveness_phase = self._liveness_phase
                    self._last_liveness_reason = "move_face_naturally"
                    self._last_liveness_score = float(movement_score)
                    return (
                        False,
                        self._liveness_phase,
                        "move_face_naturally",
                        float(movement_score),
                    )

            if self._liveness_phase == "movement_detected":
                return_nose_x = abs(
                    features["nose_x"] - baseline["nose_x"]
                ) <= self.LIVENESS_RETURN_NOSE_TOLERANCE
                return_nose_y = abs(
                    features["nose_y"] - baseline["nose_y"]
                ) <= self.LIVENESS_RETURN_NOSE_TOLERANCE
                return_roll = (
                    abs(features["roll_deg"] - baseline["roll_deg"])
                    <= self.LIVENESS_RETURN_ROLL_TOLERANCE_DEG
                )
                return_scale = (
                    self._safe_scale_ratio(
                        features["scale"],
                        baseline["scale"],
                    )
                    <= self.LIVENESS_RETURN_SCALE_RATIO
                )

                return_count = sum(
                    bool(v)
                    for v in (
                        return_nose_x,
                        return_nose_y,
                        return_roll,
                        return_scale,
                    )
                )

                if return_count >= 2:
                    self._liveness_phase = "live"
                    self._liveness_reason = "liveness_passed"
                    self._liveness_live_until = (
                        now + self.LIVENESS_LIVE_HOLD_SECONDS
                    )
                    self._last_liveness_ok = True
                    self._last_liveness_phase = "live"
                    self._last_liveness_reason = "liveness_passed"
                    self._last_liveness_score = 1.0
                    print(
                        "🛡️ [LIVENESS] PASSED | natural motion + return confirmed.",
                        flush=True,
                    )
                    return True, "live", "liveness_passed", 1.0

                self._last_liveness_ok = False
                self._last_liveness_phase = self._liveness_phase
                self._last_liveness_reason = "return_to_baseline"
                self._last_liveness_score = float(movement_score)
                return (
                    False,
                    self._liveness_phase,
                    "return_to_baseline",
                    float(movement_score),
                )

            if self._liveness_phase == "live":
                self._last_liveness_ok = True
                self._last_liveness_phase = "live"
                self._last_liveness_reason = "liveness_passed"
                self._last_liveness_score = 1.0
                return True, "live", "liveness_passed", 1.0

            self._last_liveness_ok = False
            self._last_liveness_phase = self._liveness_phase
            self._last_liveness_reason = self._liveness_reason
            self._last_liveness_score = float(movement_score)
            return (
                False,
                self._liveness_phase,
                self._liveness_reason,
                float(movement_score),
            )

    # ================================================================
    # DeepFace / ArcFace
    # ================================================================

    def _load_deepface(self) -> Any:
        with self._lock:
            if self._deepface is not None:
                return self._deepface

        try:
            self._set_stage("Importing DeepFace...")
            self._configure_runtime()
            from deepface import DeepFace

            with self._lock:
                self._deepface = DeepFace

            print("✅ [DEEPFACE] Imported successfully.", flush=True)
            return DeepFace

        except Exception as exc:
            with self._lock:
                self._loading_error = f"DeepFace gagal dimuat: {exc}"
            print(f"❌ DeepFace import error: {exc}", flush=True)
            return None

    def _load_model(self) -> bool:
        DeepFace = self._load_deepface()
        if DeepFace is None:
            return False

        with self._lock:
            if self._model_ready and self._arcface_model is not None:
                return True

        try:
            self._set_stage("Loading ArcFace model...")
            print("⏳ [ARCFACE] Loading DeepFace ArcFace model...", flush=True)

            model = DeepFace.build_model(
                task="facial_recognition",
                model_name=self.MODEL_NAME,
            )

            with self._lock:
                self._arcface_model = model
                self._model_ready = True
                self._loading_error = None

            print("✅ [ARCFACE] Model loaded and ready.", flush=True)
            return True

        except Exception as exc:
            with self._lock:
                self._arcface_model = None
                self._model_ready = False
                self._loading_error = f"ArcFace gagal dimuat: {exc}"

            print(f"❌ ArcFace model failed: {exc}", flush=True)
            return False

    @classmethod
    def _target_points(cls) -> Any:
        import numpy as np
        return np.asarray(cls.ALIGN_TARGET, dtype=np.float32)

    @staticmethod
    def _landmarks_are_usable(landmarks: Any) -> bool:
        import numpy as np

        points = np.asarray(landmarks, dtype=np.float32)

        if points.shape != (5, 2) or not np.isfinite(points).all():
            return False

        right_eye, left_eye, nose, right_mouth, left_mouth = points

        eye_span = float(np.linalg.norm(right_eye - left_eye))
        mouth_span = float(np.linalg.norm(right_mouth - left_mouth))

        if eye_span < 4.0 or mouth_span < 2.0:
            return False

        if float(np.linalg.norm(nose - ((right_eye + left_eye) / 2.0))) < 2.0:
            return False

        return True

    def _align_face(self, frame: Any, detection: dict) -> Any:
        """
        Deterministic five-point similarity alignment for ArcFace 112x112.

        No pose buckets, no RANSAC selection and no additional identity logic.
        """
        try:
            import cv2
            import numpy as np

            landmarks = np.asarray(
                detection.get("landmarks", []),
                dtype=np.float32,
            )

            if not self._landmarks_are_usable(landmarks):
                return self._crop_face(frame, detection)

            transform, _ = cv2.estimateAffinePartial2D(
                landmarks,
                self._target_points(),
                method=0,
                refineIters=10,
            )

            if transform is None or not np.isfinite(transform).all():
                return self._crop_face(frame, detection)

            aligned = cv2.warpAffine(
                frame,
                transform,
                (self.ARC_FACE_INPUT_SIZE, self.ARC_FACE_INPUT_SIZE),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

            if aligned is None or aligned.size == 0:
                return self._crop_face(frame, detection)

            return aligned

        except Exception:
            return self._crop_face(frame, detection)

    def _frontal_alt_crop(self, frame: Any, detection: dict) -> Any:
        try:
            import cv2

            x = float(detection["x"])
            y = float(detection["y"])
            w = float(detection["w"])
            h = float(detection["h"])

            height, width = frame.shape[:2]

            pad_x = self.ALT_PAD_X * w
            pad_y = self.ALT_PAD_Y * h

            left = max(0, int(round(x - pad_x)))
            top = max(0, int(round(y - pad_y)))
            right = min(width, int(round(x + w + pad_x)))
            bottom = min(height, int(round(y + h + pad_y)))

            crop = frame[top:bottom, left:right]
            if crop is None or crop.size == 0:
                return None

            return cv2.resize(
                crop,
                (self.ARC_FACE_INPUT_SIZE, self.ARC_FACE_INPUT_SIZE),
                interpolation=cv2.INTER_AREA,
            )

        except Exception:
            return None

    def _crop_face(self, frame: Any, detection: dict) -> Any:
        try:
            import cv2

            x = int(detection["x"])
            y = int(detection["y"])
            w = int(detection["w"])
            h = int(detection["h"])

            height, width = frame.shape[:2]

            mx = int(w * 0.12)
            my = int(h * 0.12)

            left = max(0, x - mx)
            top = max(0, y - my)
            right = min(width, x + w + mx)
            bottom = min(height, y + h + my)

            crop = frame[top:bottom, left:right]
            if crop is None or crop.size == 0:
                return None

            return cv2.resize(
                crop,
                (self.ARC_FACE_INPUT_SIZE, self.ARC_FACE_INPUT_SIZE),
                interpolation=cv2.INTER_AREA,
            )

        except Exception:
            return None

    def _get_embedding(self, face: Any) -> Any:
        DeepFace = self._load_deepface()

        if DeepFace is None or face is None:
            return None

        try:
            import numpy as np

            if face.size == 0:
                return None

            result = DeepFace.represent(
                img_path=face,
                model_name=self.MODEL_NAME,
                detector_backend="skip",
                enforce_detection=False,
                align=False,
                normalization="ArcFace",
                l2_normalize=True,
            )

            if not result:
                return None

            vector = result[0].get("embedding", [])
            if not vector:
                return None

            vector = np.asarray(vector, dtype=np.float32)

            norm = float(np.linalg.norm(vector))
            if norm <= 0:
                return None

            return (vector / norm).astype(np.float32, copy=False)

        except TypeError:
            # Compatibility fallback if the installed DeepFace version does not
            # expose normalization/l2_normalize in represent().
            try:
                import numpy as np

                result = DeepFace.represent(
                    img_path=face,
                    model_name=self.MODEL_NAME,
                    detector_backend="skip",
                    enforce_detection=False,
                    align=False,
                )

                if not result:
                    return None

                vector = np.asarray(
                    result[0].get("embedding", []),
                    dtype=np.float32,
                )

                norm = float(np.linalg.norm(vector))
                if norm <= 0:
                    return None

                return (vector / norm).astype(np.float32, copy=False)

            except Exception as exc:
                print(f"⚠️ ArcFace embedding error: {exc}", flush=True)
                return None

        except Exception as exc:
            print(f"⚠️ ArcFace embedding error: {exc}", flush=True)
            return None

    # ================================================================
    # Quality
    # ================================================================

    def _quality_check(
        self,
        frame: Any,
        detection: dict,
        aligned: Any,
    ) -> tuple[bool, float]:
        try:
            import cv2
            import numpy as np

            w = float(detection.get("w", 0))
            h = float(detection.get("h", 0))
            detector_score = float(detection.get("score", 0.0))

            if w < self.MIN_FACE_WIDTH or h < self.MIN_FACE_HEIGHT:
                return False, 0.0

            if detector_score < self.MIN_DETECTOR_SCORE:
                return False, 0.0

            gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

            blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness = float(np.mean(gray))

            if blur < self.MIN_BLUR_SCORE:
                return False, 0.0

            if brightness < self.MIN_BRIGHTNESS or brightness > self.MAX_BRIGHTNESS:
                return False, 0.0

            blur_score = min(1.0, blur / 180.0)
            brightness_score = 1.0 - min(
                1.0,
                abs(brightness - 128.0) / 128.0,
            )
            size_score = min(1.0, min(w, h) / 180.0)

            quality_score = float(
                0.45 * blur_score
                + 0.30 * brightness_score
                + 0.25 * size_score
            )

            return bool(quality_score >= self.MIN_QUALITY_SCORE), quality_score

        except Exception:
            return False, 0.0

    def _enrollment_quality_ok(self, detection: dict, aligned: Any) -> bool:
        # Gallery/reference quality is slightly permissive because identity is
        # determined by ArcFace similarity across multiple reference photos.
        try:
            import cv2

            w = float(detection.get("w", 0))
            h = float(detection.get("h", 0))
            score = float(detection.get("score", 0.0))

            if w < 70 or h < 70:
                return False
            if score < 0.55:
                return False
            if aligned is None or getattr(aligned, "size", 0) == 0:
                return False

            gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness = float(gray.mean())

            if blur < 35.0:
                return False
            if brightness < 30.0 or brightness > 235.0:
                return False

            return True

        except Exception:
            return False

    # ================================================================
    # Similarity
    # ================================================================

    @staticmethod
    def _cosine_similarity(query: Any, matrix: Any) -> Any:
        import numpy as np

        q = np.asarray(query, dtype=np.float32)
        m = np.asarray(matrix, dtype=np.float32)

        q_norm = float(np.linalg.norm(q))
        if q_norm <= 0:
            return np.zeros((m.shape[0],), dtype=np.float32)

        q = q / q_norm
        return np.dot(m, q)

    @staticmethod
    def _normalize_vector(vector: Any) -> Any:
        import numpy as np

        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            return None

        return (vector / norm).astype(np.float32, copy=False)

    # ================================================================
    # Gallery cache
    # ================================================================

    def _file_signature(self) -> tuple:
        result = []

        for identity, path in self._get_gallery_files():
            try:
                st = path.stat()
                result.append(
                    (
                        identity,
                        str(path),
                        st.st_mtime_ns,
                        st.st_size,
                    )
                )
            except OSError:
                pass

        return tuple(result)

    def _cache_signature(self) -> dict:
        return {
            "version": self.CACHE_VERSION,
            "model": self.MODEL_NAME,
            "metric": self.MATCH_METRIC,
            "preprocess": "yunet-5point-arcface-112-target-top3-margin-liveness-knownfaces-loo-v501",
            "calibration": {
                "source": "known_faces_leave_one_out",
                "min_genuine_tpr": float(self.CALIBRATION_MIN_GENUINE_TPR),
                "security_first": bool(self.CALIBRATION_SECURITY_FIRST),
            },
            "files": self._file_signature(),
        }

    def _save_embedding_cache(self) -> None:
        payload = {
            "signature": self._cache_signature(),
            "names": list(self._embedding_names),
            "paths": list(self._embedding_paths),
            "matrix": self._embedding_matrix,
            "thresholds": dict(self._identity_thresholds),
            "topk_thresholds": dict(self._identity_topk_thresholds),
            "margin_thresholds": dict(self._identity_margin_thresholds),
            "calibration": dict(self._identity_calibration),
        }

        temp = self.embedding_cache_path.with_suffix(".tmp")

        try:
            with temp.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temp.replace(self.embedding_cache_path)
        except Exception as exc:
            print(f"⚠️ [ARCFACE] Cache save failed: {exc}", flush=True)
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass

    def _load_embedding_cache(self) -> bool:
        if not self.embedding_cache_path.exists():
            return False

        try:
            with self.embedding_cache_path.open("rb") as handle:
                payload = pickle.load(handle)

            if payload.get("signature") != self._cache_signature():
                return False

            names = list(payload.get("names", []))
            paths = list(payload.get("paths", []))
            matrix = payload.get("matrix")

            if not names or not paths or matrix is None:
                return False

            if len(names) != len(paths):
                return False

            with self._lock:
                self._embedding_names = names
                self._embedding_paths = paths
                self._embedding_matrix = matrix
                self._identity_thresholds = dict(payload.get("thresholds", {}))
                self._identity_topk_thresholds = dict(payload.get("topk_thresholds", {}))
                self._identity_margin_thresholds = dict(payload.get("margin_thresholds", {}))
                self._identity_calibration = dict(payload.get("calibration", {}))
                self._rebuild_identity_index_locked()
                self._embeddings_ready = True
                self._embedding_signature = self._cache_signature()

            print(
                f"✅ [ARCFACE] Loaded cached gallery: "
                f"{len(names)} embeddings | "
                f"{len(set(names))} identities",
                flush=True,
            )
            return True

        except Exception as exc:
            print(f"⚠️ [ARCFACE] Cache load ignored: {exc}", flush=True)
            return False

    def _rebuild_identity_index_locked(self) -> None:
        self._identity_to_indices = {}
        for index, name in enumerate(self._embedding_names):
            self._identity_to_indices.setdefault(name, []).append(index)
        self._identity_names = sorted(self._identity_to_indices)

    # ================================================================
    # Calibration: similarity-only target verification
    # ================================================================

    @staticmethod
    def _choose_scalar_threshold(
        positives: list[float],
        negatives: list[float],
        floor: float,
        ceiling: float,
        default: float,
    ) -> tuple[float, dict[str, float]]:
        """Pick a scalar threshold from gallery-derived positive/negative scores.

        This calibration is only a diagnostic/starting point because the same
        gallery is also used for enrollment. Production performance must be
        checked with held-out photos of enrolled people and truly unseen people.
        """
        import numpy as np

        if not positives:
            return float(default), {
                "positive_p10": 0.0,
                "negative_p95": 0.0,
                "balanced_accuracy": 0.0,
                "overlap": 0.0,
            }

        pos = np.asarray(positives, dtype=np.float32)
        neg = np.asarray(negatives if negatives else [0.0], dtype=np.float32)
        pos = np.clip(pos, -1.0, 1.0)
        neg = np.clip(neg, -1.0, 1.0)

        candidates = np.unique(
            np.clip(
                np.concatenate([pos, neg]),
                floor,
                ceiling,
            )
        )
        if candidates.size == 0:
            candidates = np.asarray([default], dtype=np.float32)
        if candidates.size > 1:
            mids = (candidates[:-1] + candidates[1:]) / 2.0
            candidates = np.unique(np.concatenate([candidates, mids]))

        best_t = float(default)
        best_balanced = -1.0
        for threshold in candidates:
            tpr = float(np.mean(pos >= threshold))
            tnr = float(np.mean(neg < threshold))
            balanced = 0.5 * (tpr + tnr)
            if balanced > best_balanced:
                best_balanced = balanced
                best_t = float(threshold)

        best_t = max(float(floor), min(best_t, float(ceiling)))
        p10 = float(np.percentile(pos, 10))
        p95 = float(np.percentile(neg, 95))
        return best_t, {
            "positive_p10": p10,
            "negative_p95": p95,
            "balanced_accuracy": float(best_balanced),
            "overlap": float(max(0.0, p95 - p10)),
        }

    def _calibrate_thresholds_from_gallery(self) -> None:
        """Calibrate max-similarity, top-k support, and identity-margin floors.

        Each gallery image is treated as a pseudo-query. Genuine examples are
        compared against the remaining samples of the same identity. Impostor
        examples are compared against the target identity while the strongest
        other identity acts as the competitor.
        """
        import numpy as np

        with self._lock:
            matrix = np.asarray(self._embedding_matrix, dtype=np.float32)
            identity_to_indices = {
                name: list(indices)
                for name, indices in self._identity_to_indices.items()
            }

        thresholds: dict[str, float] = {}
        topk_thresholds: dict[str, float] = {}
        margin_thresholds: dict[str, float] = {}
        calibration: dict[str, dict[str, float]] = {}

        if matrix.size == 0:
            return

        all_indices = list(range(len(matrix)))

        for identity, own_indices in identity_to_indices.items():
            genuine_max: list[float] = []
            genuine_topk: list[float] = []
            genuine_margin: list[float] = []

            impostor_max: list[float] = []
            impostor_topk: list[float] = []
            impostor_margin: list[float] = []

            # -------- Genuine pseudo-queries --------
            for query_index in own_indices:
                own_other = [i for i in own_indices if i != query_index]
                if not own_other:
                    continue

                q = matrix[query_index]
                own_scores = np.dot(matrix[own_other], q)
                own_sorted = np.sort(own_scores)[::-1]
                own_best = float(own_sorted[0])
                own_topk = float(
                    np.mean(own_sorted[: min(self.TOP_K_TARGET_SAMPLES, own_sorted.size)])
                )

                competitor_indices = [
                    i
                    for name, group in identity_to_indices.items()
                    if name != identity
                    for i in group
                ]
                if competitor_indices:
                    competitor_scores = np.dot(matrix[competitor_indices], q)
                    competitor_sorted = np.sort(competitor_scores)[::-1]
                    competitor_best = float(competitor_sorted[0])
                else:
                    competitor_best = -1.0

                genuine_max.append(own_best)
                genuine_topk.append(own_topk)
                genuine_margin.append(own_best - competitor_best)

            # -------- Impostor pseudo-queries --------
            other_query_indices = [i for i in all_indices if i not in own_indices]
            target_matrix = matrix[own_indices]

            for query_index in other_query_indices:
                q = matrix[query_index]
                target_scores = np.dot(target_matrix, q)
                target_sorted = np.sort(target_scores)[::-1]
                target_best = float(target_sorted[0])
                target_topk = float(
                    np.mean(target_sorted[: min(self.TOP_K_TARGET_SAMPLES, target_sorted.size)])
                )

                competitor_pool = [
                    i
                    for name, group in identity_to_indices.items()
                    if name != identity
                    for i in group
                ]
                if competitor_pool:
                    competitor_scores = np.dot(matrix[competitor_pool], q)
                    competitor_best = float(np.max(competitor_scores))
                else:
                    competitor_best = -1.0

                impostor_max.append(target_best)
                impostor_topk.append(target_topk)
                impostor_margin.append(target_best - competitor_best)

            max_t, max_details = self._choose_scalar_threshold(
                genuine_max,
                impostor_max,
                self.MIN_TARGET_MAX_SIMILARITY,
                self.MAX_CALIBRATED_MAX_SIMILARITY,
                self.DEFAULT_MAX_SIMILARITY,
            )
            topk_t, topk_details = self._choose_scalar_threshold(
                genuine_topk,
                impostor_topk,
                self.MIN_TARGET_TOPK_MEAN,
                self.MAX_CALIBRATED_TOPK_MEAN,
                self.DEFAULT_TOPK_MEAN,
            )
            margin_t, margin_details = self._choose_scalar_threshold(
                genuine_margin,
                impostor_margin,
                self.MIN_TARGET_MARGIN,
                self.MAX_CALIBRATED_MARGIN,
                self.DEFAULT_MARGIN,
            )

            thresholds[identity] = max_t
            topk_thresholds[identity] = topk_t
            margin_thresholds[identity] = margin_t
            calibration[identity] = {
                "genuine_max_p10": float(max_details["positive_p10"]),
                "impostor_max_p95": float(max_details["negative_p95"]),
                "max_balanced_accuracy": float(max_details["balanced_accuracy"]),
                "genuine_topk_p10": float(topk_details["positive_p10"]),
                "impostor_topk_p95": float(topk_details["negative_p95"]),
                "topk_balanced_accuracy": float(topk_details["balanced_accuracy"]),
                "genuine_margin_p10": float(margin_details["positive_p10"]),
                "impostor_margin_p95": float(margin_details["negative_p95"]),
                "margin_balanced_accuracy": float(margin_details["balanced_accuracy"]),
                "max_overlap": float(max_details["overlap"]),
                "topk_overlap": float(topk_details["overlap"]),
                "margin_overlap": float(margin_details["overlap"]),
            }

            print(
                f"🧪 [ARCFACE CAL] {identity}: "
                f"max_p10={max_details['positive_p10']:.3f} "
                f"max_imp95={max_details['negative_p95']:.3f} "
                f"max_thr={max_t:.3f} "
                f"top3_p10={topk_details['positive_p10']:.3f} "
                f"top3_imp95={topk_details['negative_p95']:.3f} "
                f"top3_thr={topk_t:.3f} "
                f"margin_thr={margin_t:.3f}",
                flush=True,
            )

            if (
                max_details["overlap"] > 0
                or topk_details["overlap"] > 0
                or margin_details["overlap"] > 0
            ):
                print(
                    f"⚠️ [ARCFACE CAL] {identity}: "
                    "gallery-derived positive/impostor distributions overlap; "
                    "held-out validation is required before claiming production accuracy.",
                    flush=True,
                )

        with self._lock:
            self._identity_thresholds = thresholds
            self._identity_calibration = calibration
            self._identity_topk_thresholds = topk_thresholds
            self._identity_margin_thresholds = margin_thresholds

    # ================================================================
    # Gallery build
    # ================================================================

    def _build_embeddings(self) -> bool:
        import numpy as np

        files = self._get_gallery_files()

        if not files:
            with self._lock:
                self._loading_error = "known_faces/ tidak berisi foto."
            return False

        counts: dict[str, int] = {}
        for identity, _ in files:
            counts[identity] = counts.get(identity, 0) + 1

        print(
            "📚 [ARCFACE] Reference images: "
            + " | ".join(
                f"{name}={counts.get(name, 0)}"
                for name in sorted(counts)
            )
            + f" | total={len(files)}",
            flush=True,
        )

        for expected_name, expected_count in self.EXPECTED_REFERENCE_COUNTS.items():
            actual = counts.get(expected_name, 0)
            if actual != expected_count:
                print(
                    f"⚠️ [ARCFACE] Expected {expected_name}={expected_count} "
                    f"but found {actual}. All valid images will still be used.",
                    flush=True,
                )

        vectors: list[Any] = []
        names: list[str] = []
        paths: list[str] = []

        total = len(files)

        for number, (identity, path) in enumerate(files, start=1):
            self._set_stage(f"Building ArcFace embeddings... ({number}/{total})")

            try:
                image = self._read_image(str(path))
                if image is None:
                    print(
                        f"⚠️ [ARCFACE] Skip unreadable: {path.name}",
                        flush=True,
                    )
                    continue

                detections = self._detect_face_best_scale(image)
                if not detections:
                    print(
                        f"⚠️ [ARCFACE] Skip no face: {path.name}",
                        flush=True,
                    )
                    continue

                detection = detections[0]
                aligned = self._align_face(image, detection)

                if aligned is None:
                    print(
                        f"⚠️ [ARCFACE] Skip alignment: {path.name}",
                        flush=True,
                    )
                    continue

                if not self._enrollment_quality_ok(detection, aligned):
                    print(
                        f"⚠️ [ARCFACE] Skip low quality: {path.name}",
                        flush=True,
                    )
                    continue

                vector = self._get_embedding(aligned)

                if vector is None:
                    print(
                        f"⚠️ [ARCFACE] Skip embedding: {path.name}",
                        flush=True,
                    )
                    continue

                vectors.append(vector)
                names.append(identity)
                paths.append(str(path))

            except Exception as exc:
                print(
                    f"⚠️ [ARCFACE] Error {path.name}: {exc}",
                    flush=True,
                )

        if not vectors:
            with self._lock:
                self._loading_error = "Tidak ada embedding ArcFace yang berhasil dibuat."
            return False

        matrix = np.vstack(vectors).astype(np.float32, copy=False)

        with self._lock:
            self._embedding_names = names
            self._embedding_paths = paths
            self._embedding_matrix = matrix
            self._rebuild_identity_index_locked()
            self._embedding_signature = self._cache_signature()

        self._set_stage("Calibrating similarity thresholds...")
        if self.CALIBRATE_THRESHOLDS:
            self._calibrate_thresholds_auto()
        else:
            with self._lock:
                self._identity_thresholds = {
                    identity: self.DEFAULT_MAX_SIMILARITY
                    for identity in self._identity_names
                }
                self._identity_topk_thresholds = {
                    identity: self.DEFAULT_TOPK_MEAN
                    for identity in self._identity_names
                }
                self._identity_margin_thresholds = {
                    identity: self.DEFAULT_MARGIN
                    for identity in self._identity_names
                }

        self._save_embedding_cache()

        # IMPORTANT: mark the in-memory gallery as ready after a fresh build.
        # Previously this flag was only set when loading an existing cache,
        # which caused is_ready() to remain False on the first run even though
        # ArcFace had already built the full gallery successfully.
        with self._lock:
            self._embeddings_ready = True
            self._loading_error = None

        print(
            f"✅ [ARCFACE] Gallery ready: "
            f"{len(names)} valid embeddings | "
            f"{len(set(names))} identities",
            flush=True,
        )

        return True

    # ================================================================
    # Similarity-only gallery classification
    # ================================================================

    def _classify_query(self, query: Any) -> dict:
        import numpy as np

        with self._lock:
            matrix = self._embedding_matrix
            names = list(self._embedding_names)
            identity_to_indices = {
                name: list(indices)
                for name, indices in self._identity_to_indices.items()
            }

        if matrix is None or not names:
            return {
                "best_identity": "Unknown",
                "best_similarity": 0.0,
                "best_topk_mean": 0.0,
                "runner_up_identity": "Unknown",
                "runner_up_similarity": 0.0,
                "gallery_margin": 0.0,
                "identity_scores": {},
            }

        similarities = self._cosine_similarity(query, matrix)
        identity_scores: dict[str, dict[str, float]] = {}

        for identity, indices in identity_to_indices.items():
            scores = np.asarray([similarities[i] for i in indices], dtype=np.float32)
            scores = np.sort(scores)[::-1]
            topk = scores[: min(self.TOP_K_TARGET_SAMPLES, scores.size)]
            identity_scores[identity] = {
                "best": float(scores[0]) if scores.size else 0.0,
                "topk_mean": float(np.mean(topk)) if topk.size else 0.0,
                "second": float(scores[1]) if scores.size >= 2 else float(scores[0]) if scores.size else 0.0,
                "third": float(scores[2]) if scores.size >= 3 else float(scores[-1]) if scores.size else 0.0,
                "sample_count": int(scores.size),
            }

        ranked = sorted(
            identity_scores.items(),
            key=lambda item: (item[1]["best"], item[1]["topk_mean"]),
            reverse=True,
        )

        if not ranked:
            return {
                "best_identity": "Unknown",
                "best_similarity": 0.0,
                "best_topk_mean": 0.0,
                "runner_up_identity": "Unknown",
                "runner_up_similarity": 0.0,
                "gallery_margin": 0.0,
                "identity_scores": {},
            }

        best_identity, best = ranked[0]
        if len(ranked) > 1:
            runner_identity, runner = ranked[1]
        else:
            runner_identity, runner = "Unknown", {"best": 0.0, "topk_mean": 0.0}

        return {
            "best_identity": best_identity,
            "best_similarity": float(best["best"]),
            "best_topk_mean": float(best["topk_mean"]),
            "best_second_similarity": float(best["second"]),
            "best_third_similarity": float(best["third"]),
            "runner_up_identity": runner_identity,
            "runner_up_similarity": float(runner["best"]),
            "runner_up_topk_mean": float(runner.get("topk_mean", 0.0)),
            "gallery_margin": float(best["best"] - runner.get("best", 0.0)),
            "identity_scores": identity_scores,
        }


    # ================================================================
    # Known-faces-only threshold calibration
    # ================================================================

    @staticmethod
    def _candidate_thresholds(
        values: list[float],
        floor: float,
        ceiling: float,
        max_candidates: int = 19,
    ) -> list[float]:
        """Build a compact threshold grid from observed gallery scores."""
        import numpy as np

        if not values:
            return [float(floor)]

        clipped = np.clip(
            np.asarray(values, dtype=np.float32),
            float(floor),
            float(ceiling),
        )
        unique = np.unique(clipped)

        candidates = [float(value) for value in unique]
        if unique.size > 1:
            mids = (unique[:-1] + unique[1:]) / 2.0
            candidates.extend(float(value) for value in mids)

        candidates.extend([float(floor), float(ceiling)])
        candidates = sorted(set(candidates))

        if len(candidates) <= max_candidates:
            return candidates

        quantiles = np.linspace(0.0, 1.0, max_candidates)
        sampled = np.quantile(
            np.asarray(candidates, dtype=np.float32),
            quantiles,
        )
        sampled = np.clip(
            sampled,
            float(floor),
            float(ceiling),
        )
        return sorted(set(float(value) for value in sampled))

    def _gallery_calibration_samples(
        self,
        target: str,
    ) -> list[dict[str, Any]]:
        """Build leave-one-out positive/impostor samples from known_faces/.

        Each gallery image is used once as a pseudo-query.

        Genuine query:
          - target pool excludes the query itself;
          - competitor pool contains other identities.

        Impostor query:
          - target pool contains all target references;
          - competitor pool excludes the query itself so the query does not
            create an artificial similarity of exactly 1.0.

        This produces the same three statistics used in live recognition:
          best target similarity, target top-3 mean, target-vs-competitor margin.
        """
        import numpy as np

        target = self.format_person_name(target)

        with self._lock:
            matrix = np.asarray(self._embedding_matrix, dtype=np.float32)
            identity_to_indices = {
                name: list(indices)
                for name, indices in self._identity_to_indices.items()
            }

        if matrix.size == 0:
            return []

        target_indices = list(identity_to_indices.get(target, []))
        if not target_indices:
            return []

        all_indices = list(range(len(matrix)))
        samples: list[dict[str, Any]] = []

        for query_index in all_indices:
            query = matrix[query_index]

            is_positive = query_index in target_indices

            if is_positive:
                target_pool = [
                    i for i in target_indices
                    if i != query_index
                ]
            else:
                target_pool = list(target_indices)

            if not target_pool:
                continue

            competitor_pool = [
                i
                for identity, group in identity_to_indices.items()
                if identity != target
                for i in group
                if i != query_index
            ]

            target_scores = np.dot(matrix[target_pool], query)
            target_scores = np.sort(
                np.asarray(target_scores, dtype=np.float32)
            )[::-1]

            if not target_scores.size:
                continue

            target_best = float(target_scores[0])
            topk = target_scores[
                : min(self.TOP_K_TARGET_SAMPLES, target_scores.size)
            ]
            target_topk_mean = (
                float(np.mean(topk))
                if topk.size
                else 0.0
            )

            if competitor_pool:
                competitor_scores = np.dot(
                    matrix[competitor_pool],
                    query,
                )
                competitor_best = float(
                    np.max(np.asarray(competitor_scores, dtype=np.float32))
                )
            else:
                competitor_best = -1.0

            samples.append(
                {
                    "query_index": int(query_index),
                    "positive": bool(is_positive),
                    "best": float(target_best),
                    "topk": float(target_topk_mean),
                    "margin": float(target_best - competitor_best),
                    "competitor_best": float(competitor_best),
                    "query_identity": (
                        self._embedding_names[query_index]
                        if query_index < len(self._embedding_names)
                        else "Unknown"
                    ),
                }
            )

        return samples

    def _choose_joint_gallery_thresholds(
        self,
        samples: list[dict[str, Any]],
    ) -> tuple[float, float, float, dict[str, float]]:
        """Choose max/top3/margin thresholds jointly from gallery samples.

        The production gate is an AND condition, so calibration is also done
        jointly. Security-first objective:
          1) keep at least CALIBRATION_MIN_GENUINE_TPR positives;
          2) minimize observed pseudo-impostor false accepts;
          3) maximize genuine acceptance;
          4) use the lowest threshold sum as deterministic tie-break.
        """
        import numpy as np

        positives = [
            item for item in samples
            if bool(item.get("positive"))
        ]
        negatives = [
            item for item in samples
            if not bool(item.get("positive"))
        ]

        if not positives:
            return (
                self.DEFAULT_MAX_SIMILARITY,
                self.DEFAULT_TOPK_MEAN,
                self.DEFAULT_MARGIN,
                {
                    "genuine_total": 0.0,
                    "impostor_total": float(len(negatives)),
                    "genuine_acceptance_rate": 0.0,
                    "false_accept_rate": 0.0,
                    "balanced_accuracy": 0.0,
                },
            )

        max_candidates = self._candidate_thresholds(
            [float(item["best"]) for item in samples],
            self.MIN_TARGET_MAX_SIMILARITY,
            self.MAX_CALIBRATED_MAX_SIMILARITY,
        )
        topk_candidates = self._candidate_thresholds(
            [float(item["topk"]) for item in samples],
            self.MIN_TARGET_TOPK_MEAN,
            self.MAX_CALIBRATED_TOPK_MEAN,
        )
        margin_candidates = self._candidate_thresholds(
            [float(item["margin"]) for item in samples],
            self.MIN_TARGET_MARGIN,
            self.MAX_CALIBRATED_MARGIN,
        )

        pos_best = np.asarray(
            [float(item["best"]) for item in positives],
            dtype=np.float32,
        )
        pos_topk = np.asarray(
            [float(item["topk"]) for item in positives],
            dtype=np.float32,
        )
        pos_margin = np.asarray(
            [float(item["margin"]) for item in positives],
            dtype=np.float32,
        )

        neg_count = len(negatives)

        if neg_count:
            neg_best = np.asarray(
                [float(item["best"]) for item in negatives],
                dtype=np.float32,
            )
            neg_topk = np.asarray(
                [float(item["topk"]) for item in negatives],
                dtype=np.float32,
            )
            neg_margin = np.asarray(
                [float(item["margin"]) for item in negatives],
                dtype=np.float32,
            )
        else:
            neg_best = np.empty((0,), dtype=np.float32)
            neg_topk = np.empty((0,), dtype=np.float32)
            neg_margin = np.empty((0,), dtype=np.float32)

        best_key: tuple[Any, ...] | None = None
        best_result: tuple[float, float, float, float, float, float] | None = None

        for threshold_max in max_candidates:
            pos_ok_max = pos_best >= threshold_max
            neg_ok_max = (
                neg_best >= threshold_max
                if neg_count
                else np.array([], dtype=bool)
            )

            for threshold_topk in topk_candidates:
                pos_ok_topk = pos_topk >= threshold_topk
                neg_ok_topk = (
                    neg_topk >= threshold_topk
                    if neg_count
                    else np.array([], dtype=bool)
                )

                pos_partial = pos_ok_max & pos_ok_topk
                neg_partial = (
                    neg_ok_max & neg_ok_topk
                    if neg_count
                    else np.array([], dtype=bool)
                )

                for threshold_margin in margin_candidates:
                    pos_accept = pos_partial & (
                        pos_margin >= threshold_margin
                    )
                    pos_tpr = float(
                        np.mean(pos_accept)
                        if len(pos_best)
                        else 0.0
                    )

                    if neg_count:
                        neg_accept = neg_partial & (
                            neg_margin >= threshold_margin
                        )
                        false_accept = float(np.mean(neg_accept))
                    else:
                        false_accept = 0.0

                    true_negative = 1.0 - false_accept
                    balanced = 0.5 * (pos_tpr + true_negative)

                    eligible = (
                        pos_tpr >= self.CALIBRATION_MIN_GENUINE_TPR
                    )

                    if self.CALIBRATION_SECURITY_FIRST:
                        if not eligible:
                            continue
                        key = (
                            false_accept,
                            -pos_tpr,
                            float(
                                threshold_max
                                + threshold_topk
                                + threshold_margin
                            ),
                        )
                    else:
                        key = (
                            -balanced,
                            false_accept,
                            -pos_tpr,
                            float(
                                threshold_max
                                + threshold_topk
                                + threshold_margin
                            ),
                        )

                    if best_key is None or key < best_key:
                        best_key = key
                        best_result = (
                            float(threshold_max),
                            float(threshold_topk),
                            float(threshold_margin),
                            pos_tpr,
                            false_accept,
                            balanced,
                        )

        if best_result is None:
            # The minimum genuine-retention constraint could not be satisfied.
            # Fall back to the least permissive thresholds within the allowed
            # range rather than silently lowering any floor.
            return (
                self.MAX_CALIBRATED_MAX_SIMILARITY,
                self.MAX_CALIBRATED_TOPK_MEAN,
                self.MAX_CALIBRATED_MARGIN,
                {
                    "genuine_total": float(len(positives)),
                    "impostor_total": float(neg_count),
                    "genuine_acceptance_rate": 0.0,
                    "false_accept_rate": 0.0,
                    "balanced_accuracy": 0.0,
                },
            )

        return (
            best_result[0],
            best_result[1],
            best_result[2],
            {
                "genuine_total": float(len(positives)),
                "impostor_total": float(neg_count),
                "genuine_acceptance_rate": float(best_result[3]),
                "false_accept_rate": float(best_result[4]),
                "balanced_accuracy": float(best_result[5]),
            },
        )

    def _calibrate_thresholds_from_gallery(self) -> None:
        """Calibrate per-identity thresholds from known_faces/ only."""
        with self._lock:
            identities = list(self._identity_names)

        thresholds: dict[str, float] = {}
        topk_thresholds: dict[str, float] = {}
        margin_thresholds: dict[str, float] = {}
        calibration: dict[str, dict[str, float]] = {}

        if not identities:
            return

        print("", flush=True)
        print("=" * 78, flush=True)
        print(
            "🧪 [ARCFACE CAL] KNOWN-FACES-ONLY LEAVE-ONE-OUT CALIBRATION",
            flush=True,
        )
        print("=" * 78, flush=True)
        print(
            "   Source : known_faces/ only",
            flush=True,
        )
        print(
            "   Note   : diagnostic/internal calibration; "
            "not a held-out production accuracy estimate.",
            flush=True,
        )
        print("", flush=True)

        all_samples_by_identity: dict[str, list[dict[str, Any]]] = {}

        for identity in identities:
            samples = self._gallery_calibration_samples(identity)
            all_samples_by_identity[identity] = samples

            positive_count = sum(
                1 for item in samples
                if bool(item.get("positive"))
            )
            negative_count = sum(
                1 for item in samples
                if not bool(item.get("positive"))
            )

            print(
                f"   [{identity}] pseudo-samples: "
                f"genuine={positive_count} | impostor={negative_count}",
                flush=True,
            )

        for identity in identities:
            samples = all_samples_by_identity.get(identity, [])

            max_t, topk_t, margin_t, metrics = (
                self._choose_joint_gallery_thresholds(samples)
            )

            thresholds[identity] = max_t
            topk_thresholds[identity] = topk_t
            margin_thresholds[identity] = margin_t

            calibration[identity] = {
                "source": "known_faces_leave_one_out",
                "genuine_total": float(metrics["genuine_total"]),
                "impostor_total": float(metrics["impostor_total"]),
                "genuine_acceptance_rate": float(
                    metrics["genuine_acceptance_rate"]
                ),
                "false_accept_rate": float(
                    metrics["false_accept_rate"]
                ),
                "balanced_accuracy": float(
                    metrics["balanced_accuracy"]
                ),
                "threshold_max_similarity": float(max_t),
                "threshold_topk_mean": float(topk_t),
                "threshold_margin": float(margin_t),
            }

            print(
                f"   [{identity}] "
                f"sim>={max_t:.3f} | "
                f"top3>={topk_t:.3f} | "
                f"margin>={margin_t:.3f} | "
                f"genuine={int(metrics['genuine_total'])} | "
                f"impostor={int(metrics['impostor_total'])} | "
                f"pseudo-TAR={metrics['genuine_acceptance_rate'] * 100.0:.1f}% | "
                f"pseudo-FAR={metrics['false_accept_rate'] * 100.0:.1f}% | "
                f"balanced={metrics['balanced_accuracy'] * 100.0:.1f}%",
                flush=True,
            )

        with self._lock:
            self._identity_thresholds = thresholds
            self._identity_topk_thresholds = topk_thresholds
            self._identity_margin_thresholds = margin_thresholds
            self._identity_calibration = calibration
            self._calibration_source = "known_faces_leave_one_out"

        print("", flush=True)
        print(
            "✅ [ARCFACE CAL] Known-faces-only calibration complete.",
            flush=True,
        )

    def _calibrate_thresholds_auto(self) -> None:
        if not self.CALIBRATE_THRESHOLDS:
            with self._lock:
                self._identity_thresholds = {
                    identity: self.DEFAULT_MAX_SIMILARITY
                    for identity in self._identity_names
                }
                self._identity_topk_thresholds = {
                    identity: self.DEFAULT_TOPK_MEAN
                    for identity in self._identity_names
                }
                self._identity_margin_thresholds = {
                    identity: self.DEFAULT_MARGIN
                    for identity in self._identity_names
                }
                self._identity_calibration = {
                    identity: {
                        "source": "static_defaults",
                        "threshold_max_similarity": self.DEFAULT_MAX_SIMILARITY,
                        "threshold_topk_mean": self.DEFAULT_TOPK_MEAN,
                        "threshold_margin": self.DEFAULT_MARGIN,
                    }
                    for identity in self._identity_names
                }
            return

        self._calibrate_thresholds_from_gallery()

    def print_threshold_profile(self) -> None:
        """Print the active per-identity thresholds to the Python terminal."""
        with self._lock:
            identities = list(self._identity_names)
            thresholds = dict(self._identity_thresholds)
            topk_thresholds = dict(self._identity_topk_thresholds)
            margin_thresholds = dict(self._identity_margin_thresholds)
            calibration = dict(self._identity_calibration)

        print("", flush=True)
        print("🧾 [ARCFACE] ACTIVE THRESHOLD PROFILE", flush=True)

        if not identities:
            print("   No enrolled identities.", flush=True)
            return

        for identity in identities:
            cal = calibration.get(identity, {})
            source = cal.get(
                "source",
                "known_faces_leave_one_out",
            )
            print(
                f"   {identity}: "
                f"similarity>={thresholds.get(identity, self.DEFAULT_MAX_SIMILARITY):.3f} | "
                f"top3>={topk_thresholds.get(identity, self.DEFAULT_TOPK_MEAN):.3f} | "
                f"margin>={margin_thresholds.get(identity, self.DEFAULT_MARGIN):.3f} | "
                f"source={source}",
                flush=True,
            )

    def _target_threshold(self, target: str) -> float:
        target = self.format_person_name(target)

        with self._lock:
            return float(
                self._identity_thresholds.get(
                    target,
                    self.DEFAULT_MAX_SIMILARITY,
                )
            )

    def _target_topk_threshold(self, target: str) -> float:
        target = self.format_person_name(target)

        with self._lock:
            return float(
                self._identity_topk_thresholds.get(
                    target,
                    self.DEFAULT_TOPK_MEAN,
                )
            )

    def _target_margin_threshold(self, target: str) -> float:
        target = self.format_person_name(target)

        with self._lock:
            return float(
                self._identity_margin_thresholds.get(
                    target,
                    self.DEFAULT_MARGIN,
                )
            )

    # ================================================================
    # Main recognition API
    # ================================================================

    def recognize_detection(
        self,
        frame: Any,
        detection: dict | None,
        target_identity: str | None = None,
        tolerance: float | None = None,
        require_liveness: bool = True,
    ) -> tuple[str, float, bool]:
        """Verify the logged-in HRIS target using the existing ArcFace rules.

        Existing identity acceptance remains unchanged:
          * quality gate passes;
          * the HRIS target is the strongest identity by best similarity;
          * target best similarity >= calibrated target threshold;
          * target top-3 mean similarity >= calibrated target support threshold;
          * target-vs-runner margin >= calibrated margin threshold.

        Additional live-camera gate:
          * when require_liveness=True, temporal liveness must pass before
            face_confident can become True.

        The liveness gate does not alter ArcFace embeddings, thresholds,
        gallery ranking, top-3 support, or identity-margin calculations.
        """
        del tolerance

        if frame is None or detection is None:
            return "Unknown", 0.0, False

        if not self.is_ready():
            self._last_decision_reason = "recognizer_not_ready"
            return "Unknown", 0.0, False

        target = self.format_person_name(target_identity)
        if target == "Unknown":
            with self._lock:
                self._last_target_identity = target
                self._last_target_threshold = self.DEFAULT_MAX_SIMILARITY
                self._last_target_topk_threshold = self.DEFAULT_TOPK_MEAN
                self._last_target_margin_threshold = self.DEFAULT_MARGIN
                self._last_decision_reason = "target_unknown"
            return "Unknown", 0.0, False

        # Reject impossible target early if this identity is not actually in
        # the loaded gallery.
        with self._lock:
            target_indices = list(self._identity_to_indices.get(target, []))
        if not target_indices:
            with self._lock:
                self._last_target_identity = target
                self._last_target_sample_count = 0
                self._last_target_threshold = self.DEFAULT_MAX_SIMILARITY
                self._last_target_topk_threshold = self.DEFAULT_TOPK_MEAN
                self._last_target_margin_threshold = self.DEFAULT_MARGIN
                self._last_decision_reason = "target_not_in_gallery"
            return "Unknown", 0.0, False

        aligned = self._align_face(frame, detection)
        if aligned is None:
            with self._lock:
                self._last_decision_reason = "alignment_failed"
            return "Unknown", 0.0, False

        quality_ok, quality_score = self._quality_check(frame, detection, aligned)
        if not quality_ok:
            with self._lock:
                self._last_match_name = "Unknown"
                self._last_similarity = 0.0
                self._last_face_confident = False
                self._last_quality_ok = False
                self._last_quality_score = float(quality_score)
                self._last_target_identity = target
                self._last_target_sample_count = len(target_indices)
                self._last_target_topk_mean = 0.0
                self._last_target_margin = 0.0
                self._last_competitor_best = 0.0
                self._last_target_topk_threshold = self._target_topk_threshold(target)
                self._last_target_margin_threshold = self._target_margin_threshold(target)
                self._last_decision_reason = "quality_failed"
                self._last_decision_mode = "target_top3_margin"
            return "Unknown", 0.0, False

        # --------------------------------------------------------------
        # FRONTAL GATE FIRST, THEN ARC FACE, THEN PASSIVE PAD
        # --------------------------------------------------------------
        # IMPORTANT: similarity is computed before PAD so the UI/logs show the
        # actual identity score even while PAD is warming up. Security is not
        # weakened: face_confident can only become True after PAD also passes.
        frontal_ok, frontal_score, frontal_reason = self._frontal_pose_check(
            detection
        )

        if not frontal_ok:
            with self._lock:
                self._last_match_name = "Unknown"
                self._last_similarity = 0.0
                self._last_face_confident = False
                self._last_quality_ok = True
                self._last_quality_score = float(quality_score)
                self._last_target_identity = target
                self._last_target_sample_count = len(target_indices)
                self._last_target_topk_mean = 0.0
                self._last_target_margin = 0.0
                self._last_competitor_best = 0.0
                self._last_target_topk_threshold = self._target_topk_threshold(target)
                self._last_target_margin_threshold = self._target_margin_threshold(target)
                self._last_decision_reason = f"frontal_{frontal_reason}"
                self._last_decision_mode = "frontal+pad+target_top3_margin"
                self._last_liveness_ok = False
                self._last_liveness_phase = "frontal_required"
                self._last_liveness_reason = frontal_reason
                self._last_liveness_score = float(frontal_score)

            return "Unknown", 0.0, False

        query = self._get_embedding(aligned)
        if query is None:
            with self._lock:
                self._last_match_name = "Unknown"
                self._last_similarity = 0.0
                self._last_face_confident = False
                self._last_quality_ok = True
                self._last_quality_score = float(quality_score)
                self._last_target_identity = target
                self._last_target_sample_count = len(target_indices)
                self._last_target_topk_mean = 0.0
                self._last_target_margin = 0.0
                self._last_competitor_best = 0.0
                self._last_target_topk_threshold = self._target_topk_threshold(target)
                self._last_target_margin_threshold = self._target_margin_threshold(target)
                self._last_decision_reason = "embedding_failed"
                self._last_decision_mode = "frontal+pad+target_top3_margin"
            return "Unknown", 0.0, False

        classification = self._classify_query(query)
        best_identity = self.format_person_name(classification["best_identity"])
        best_similarity = float(classification["best_similarity"])
        best_topk_mean = float(classification.get("best_topk_mean", 0.0))
        runner_identity = self.format_person_name(classification["runner_up_identity"])
        runner_similarity = float(classification["runner_up_similarity"])
        gallery_margin = float(classification["gallery_margin"])

        # IMPORTANT: compare the actual HRIS target specifically. The gallery
        # winner is retained only as a diagnostic. This avoids accepting a
        # different enrolled person under the target's login.
        identity_scores = classification.get("identity_scores", {})
        target_stats = identity_scores.get(
            target,
            {"best": 0.0, "topk_mean": 0.0, "second": 0.0, "third": 0.0, "sample_count": 0},
        )
        target_best = float(target_stats.get("best", 0.0))
        target_topk_mean = float(target_stats.get("topk_mean", 0.0))
        target_second = float(target_stats.get("second", 0.0))
        target_third = float(target_stats.get("third", 0.0))

        competing_scores = [
            float(stats.get("best", 0.0))
            for identity, stats in identity_scores.items()
            if identity != target
        ]
        competitor_best = max(competing_scores) if competing_scores else 0.0
        target_margin = target_best - competitor_best

        max_threshold = self._target_threshold(target)
        topk_threshold = self._target_topk_threshold(target)
        margin_threshold = self._target_margin_threshold(target)

        # The target must also be the identity winner. If another enrolled
        # identity scores higher, this is a hard reject for the current HRIS.
        target_is_gallery_winner = best_identity == target

        # Passive anti-spoof gate. PAD is throttled internally so this call
        # remains responsive even when camera_server submits several frames/sec.
        pad_ok = True
        pad_score = 1.0
        pad_reason = "disabled_or_bypassed"
        if require_liveness and self.ANTI_SPOOF_ENABLED:
            pad_ok, pad_score, pad_reason = self._check_anti_spoof(
                frame,
                detection,
                target,
            )

        accepted = bool(
            quality_ok
            and frontal_ok
            and pad_ok
            and target_is_gallery_winner
            and target_best >= max_threshold
            and target_topk_mean >= topk_threshold
            and target_margin >= margin_threshold
        )

        candidate = target if accepted else "Unknown"
        returned_similarity = target_best

        if not quality_ok:
            reason = "quality_failed"
        elif not frontal_ok:
            reason = f"frontal_{frontal_reason}"
        elif not pad_ok:
            reason = f"anti_spoof_{pad_reason}"
        elif not target_is_gallery_winner:
            reason = "target_not_gallery_winner"
        elif target_best < max_threshold:
            reason = "target_best_below_threshold"
        elif target_topk_mean < topk_threshold:
            reason = "target_top3_support_below_threshold"
        elif target_margin < margin_threshold:
            reason = "target_margin_below_threshold"
        else:
            reason = "accepted_target_similarity"

        with self._lock:
            self._last_match_name = candidate
            self._last_similarity = returned_similarity
            self._last_face_confident = accepted
            self._last_quality_ok = bool(quality_ok)
            self._last_quality_score = float(quality_score)
            self._last_target_identity = target
            self._last_target_threshold = float(max_threshold)
            self._last_target_sample_count = len(target_indices)
            self._last_target_topk_mean = float(target_topk_mean)
            self._last_target_margin = float(target_margin)
            self._last_competitor_best = float(competitor_best)
            self._last_target_topk_threshold = float(topk_threshold)
            self._last_target_margin_threshold = float(margin_threshold)
            self._last_gallery_best_identity = best_identity
            self._last_gallery_best_similarity = best_similarity
            self._last_gallery_runner_up_identity = runner_identity
            self._last_gallery_runner_up_similarity = runner_similarity
            self._last_gallery_margin = gallery_margin
            self._last_decision_reason = reason
            self._last_decision_mode = (
                "frontal+pad+target_top3_margin"
                if (require_liveness and self.ANTI_SPOOF_ENABLED)
                else "target_top3_margin"
            )
            self._last_liveness_ok = bool(pad_ok)
            self._last_liveness_phase = "pad_passed" if pad_ok else "pad"
            self._last_liveness_reason = pad_reason
            self._last_liveness_score = float(pad_score)

        print(
            "🔎 [ARCFACE] "
            f"Candidate={candidate} "
            f"similarity={returned_similarity:.3f} "
            f"threshold={max_threshold:.3f} "
            f"top3={target_topk_mean:.3f}/{topk_threshold:.3f} "
            f"margin={target_margin:.3f}/{margin_threshold:.3f} "
            f"quality={'OK' if quality_ok else 'FAIL'} "
            f"quality_score={quality_score:.2f} "
            f"target={target} "
            f"target_best={target_best:.3f} "
            f"target_second={target_second:.3f} "
            f"target_third={target_third:.3f} "
            f"competitor={runner_identity} "
            f"competitor_best={competitor_best:.3f} "
            f"gallery_best={best_identity} "
            f"gallery_best_sim={best_similarity:.3f} "
            f"gallery_runner={runner_identity} "
            f"gallery_runner_sim={runner_similarity:.3f} "
            f"gallery_margin={gallery_margin:.3f} "
            f"target_samples={len(target_indices)} "
            f"reason={reason} "
            f"mode=frontal+pad+target_top3_margin",
            flush=True,
        )

        return candidate, returned_similarity, accepted

    def get_last_match_metrics(self) -> dict:
        with self._lock:
            target = self._last_target_identity
            max_threshold = float(self._last_target_threshold)
            topk_threshold = float(self._identity_topk_thresholds.get(target, self.DEFAULT_TOPK_MEAN))
            margin_threshold = float(self._identity_margin_thresholds.get(target, self.DEFAULT_MARGIN))
            return {
                # Current ArcFace fields.
                "candidate": self._last_match_name,
                "similarity": float(self._last_similarity),
                "target_threshold": max_threshold,
                "topk_mean": float(self._last_target_topk_mean),
                "target_topk_threshold": float(self._last_target_topk_threshold),
                "target_margin": float(self._last_target_margin),
                "competitor_best": float(self._last_competitor_best),
                "target_margin_threshold": float(self._last_target_margin_threshold),
                "face_confident": bool(self._last_face_confident),
                "quality_ok": bool(self._last_quality_ok),
                "quality_score": float(self._last_quality_score),
                "target": target,
                "target_samples": int(self._last_target_sample_count),
                "gallery_best": self._last_gallery_best_identity,
                "gallery_best_sim": float(self._last_gallery_best_similarity),
                "gallery_runner": self._last_gallery_runner_up_identity,
                "gallery_runner_sim": float(self._last_gallery_runner_up_similarity),
                "gallery_margin": float(self._last_gallery_margin),
                "decision_reason": self._last_decision_reason,
                "decision_mode": self._last_decision_mode,
                "liveness_enabled": bool(self.LIVENESS_ENABLED),
                "liveness_ok": bool(self._last_liveness_ok),
                "liveness_phase": self._last_liveness_phase,
                "liveness_reason": self._last_liveness_reason,
                "liveness_score": float(self._last_liveness_score),

                # Passive anti-spoof / PAD diagnostics.
                "anti_spoofing_enabled": bool(self.ANTI_SPOOF_ENABLED),
                "anti_spoof_ok": bool(self._anti_spoof_last_ok),
                "anti_spoof_score": float(self._anti_spoof_last_score),
                "anti_spoof_consecutive": int(self._anti_spoof_consecutive),
                "anti_spoof_required_consecutive": int(self.ANTI_SPOOF_REQUIRED_CONSECUTIVE),
                "anti_spoof_reason": self._anti_spoof_last_reason,
                "anti_spoof_model": self.ANTI_SPOOF_MODEL_NAME,

                "model": self.MODEL_NAME,
                "metric": self.MATCH_METRIC,
                # Legacy camera_server.py keys so the existing UI/logger does
                # not crash while it is still being migrated from SFace.
                "threshold": max_threshold,
                "target_sample_count": int(self._last_target_sample_count),
                "identity_margin": float(self._last_target_margin),
                "negative_similarity": float(self._last_competitor_best),
                "negative_gate": float(self._last_target_margin_threshold),
                "gallery_best_identity": self._last_gallery_best_identity,
                "gallery_best_similarity": float(self._last_gallery_best_similarity),
                "gallery_best_score": float(self._last_gallery_best_similarity),
                "gallery_runner_up_identity": self._last_gallery_runner_up_identity,
                "gallery_runner_up_similarity": float(self._last_gallery_runner_up_similarity),
                "gallery_runner_up_score": float(self._last_gallery_runner_up_similarity),
                "tta_used": False,
                "prototype_similarity": 0.0,
            }

    def get_target_identity_status(
        self,
        target_identity: str | None,
    ) -> dict:
        target = self.format_person_name(target_identity)

        with self._lock:
            return {
                "target": target,
                "samples": len(self._identity_to_indices.get(target, [])),
                "threshold": float(
                    self._identity_thresholds.get(
                        target,
                        self.DEFAULT_MAX_SIMILARITY,
                    )
                ),
                "topk_threshold": float(
                    self._identity_topk_thresholds.get(
                        target,
                        self.DEFAULT_TOPK_MEAN,
                    )
                ),
                "margin_threshold": float(
                    self._identity_margin_thresholds.get(
                        target,
                        self.DEFAULT_MARGIN,
                    )
                ),
                "calibration": dict(
                    self._identity_calibration.get(target, {})
                ),
                "model": self.MODEL_NAME,
                "metric": self.MATCH_METRIC,
                "mode": "frontal+pad+target_top3_margin",
                "liveness_enabled": False,
                "anti_spoofing_enabled": bool(self.ANTI_SPOOF_ENABLED),
                "anti_spoof_required_consecutive": int(self.ANTI_SPOOF_REQUIRED_CONSECUTIVE),
                "anti_spoof_model": self.ANTI_SPOOF_MODEL_NAME,
            }

    # ================================================================
    # Async preparation
    # ================================================================

    def _finish_loading(self, success: bool) -> None:
        with self._lock:
            self._loading = False
            if success:
                self._loading_stage = "Face recognition ready"
                self._loading_error = None
            self._ready_event.set()

    def _prepare_worker(self) -> None:
        success = False

        try:
            self._set_stage("Preparing ArcFace + gallery...")

            if not self._load_yunet():
                return

            if not self._load_model():
                return

            self._set_stage("Loading cached ArcFace embeddings...")

            if not self._load_embedding_cache():
                success = self._build_embeddings()
            else:
                        # Cache already contains calibrated thresholds.
                success = True

            if success:
                self._index_dataset()
                self.print_runtime_status()
                self.print_threshold_profile()

        except Exception as exc:
            with self._lock:
                self._loading_error = (
                    f"Face recognition preload gagal: {exc}"
                )
            print(
                f"❌ [ARCFACE] Preload error: {exc}",
                flush=True,
            )

        finally:
            self._finish_loading(success)

    def prepare_all_async(self) -> None:
        with self._lock:
            if self._model_ready and self._detector_ready and self._embeddings_ready:
                return

            if self._loading and self._prepare_thread and self._prepare_thread.is_alive():
                return

            self._loading = True
            self._loading_error = None
            self._loading_stage = "Preparing ArcFace + gallery..."
            self._ready_event.clear()

            worker = threading.Thread(
                target=self._prepare_worker,
                daemon=True,
                name="arcface-model-loader",
            )
            self._prepare_thread = worker

        worker.start()

    # Compatibility aliases used by existing dashboard/camera code.
    def prepare_process_async(self) -> None:
        self.prepare_all_async()

    def prepare_camera_async(self) -> None:
        self.prepare_all_async()

    def prepare_model_async(self) -> None:
        self.prepare_all_async()

    def prepare_login_async(self) -> None:
        self.prepare_all_async()

    def prepare_cached_gallery_async(self) -> None:
        self.prepare_all_async()

    def prepare_async(self) -> None:
        self.prepare_all_async()

    # ================================================================
    # Reload
    # ================================================================

    def clear_embeddings(self) -> None:
        with self._lock:
            self._embedding_names = []
            self._embedding_paths = []
            self._embedding_matrix = None
            self._identity_to_indices = {}
            self._identity_names = []
            self._identity_thresholds = {}
            self._identity_topk_thresholds = {}
            self._identity_margin_thresholds = {}
            self._identity_calibration = {}
            self._embedding_signature = None
            self._embeddings_ready = False
            self._reset_liveness_locked(reason="embeddings_cleared")
            self._reset_anti_spoof_locked(reason="embeddings_cleared")
            self._ready_event.clear()

        try:
            self.embedding_cache_path.unlink(missing_ok=True)
        except OSError:
            pass

    def reload(self) -> int:
        self.clear_embeddings()
        self._index_dataset()

        with self._lock:
            self._smoothed_box = None
            self._last_detection = None
            self._last_match_name = "Unknown"
            self._last_similarity = 0.0
            self._last_face_confident = False
            self._last_target_identity = "Unknown"
            self._last_target_topk_mean = 0.0
            self._last_target_margin = 0.0
            self._last_competitor_best = 0.0
            self._last_target_topk_threshold = self.DEFAULT_TOPK_MEAN
            self._last_target_margin_threshold = self.DEFAULT_MARGIN
            self._last_decision_reason = "not_evaluated"
            self._last_decision_mode = "similarity_only"
            self._reset_liveness_locked(reason="reload")
            self._reset_anti_spoof_locked(reason="reload")

        self.prepare_all_async()
        return self.get_dataset_file_count()


# ----------------------------------------------------------------------
# Process-level singleton / public compatibility API
# ----------------------------------------------------------------------

_recognizer = FaceRecognizer()
_recognizer_lock = threading.Lock()


def get_face_recognizer() -> FaceRecognizer:
    global _recognizer
    with _recognizer_lock:
        if _recognizer is None:
            _recognizer = FaceRecognizer()
    return _recognizer


def start_process_preload() -> None:
    get_face_recognizer().prepare_process_async()


def prepare_async() -> None:
    get_face_recognizer().prepare_all_async()


def is_model_ready() -> bool:
    return get_face_recognizer().is_model_ready()


def is_detector_ready() -> bool:
    return get_face_recognizer().is_detector_ready()


def is_embeddings_ready() -> bool:
    return get_face_recognizer().is_embeddings_ready()


def is_ready() -> bool:
    return get_face_recognizer().is_ready()


def get_runtime_status() -> dict:
    return get_face_recognizer().get_runtime_status()


def print_runtime_status() -> None:
    get_face_recognizer().print_runtime_status()


def is_loading() -> bool:
    return get_face_recognizer().is_loading()


def get_loading_status() -> str:
    return get_face_recognizer().get_loading_status()


def get_loading_progress() -> int:
    return get_face_recognizer().get_loading_progress()


def detect_faces(frame: Any) -> list[dict]:
    return get_face_recognizer()._detect_face_best_scale(frame)


def reset_liveness(reason: str = "manual_reset") -> None:
    get_face_recognizer().reset_liveness(reason)


def reset_anti_spoof(reason: str = "manual_reset") -> None:
    get_face_recognizer().reset_anti_spoof(reason)


def reload() -> int:
    return get_face_recognizer().reload()


def clear_embeddings() -> None:
    get_face_recognizer().clear_embeddings()


def get_dataset_file_count() -> int:
    return get_face_recognizer().get_dataset_file_count()


def get_known_names() -> list[str]:
    return get_face_recognizer().get_known_names()


def get_known_paths() -> list[str]:
    return get_face_recognizer().get_known_paths()


def get_validation_file_count() -> int:
    # Compatibility only. This final build does not use validation_faces/.
    return 0


def get_validation_names() -> list[str]:
    # Compatibility only. This final build does not use validation_faces/.
    return []


# Keep the public singleton name used by older code.
face_recognizer = _recognizer



def _validation_score_query(
    recognizer: FaceRecognizer,
    matrix: Any,
    names: list[str],
    identity_to_indices: dict[str, list[int]],
    query_index: int,
    target: str,
    thresholds: dict[str, float],
    topk_thresholds: dict[str, float],
    margin_thresholds: dict[str, float],
) -> dict[str, Any]:
    """Evaluate one gallery embedding against one HRIS target.

    This reproduces the identity-only production gates while correctly
    excluding the query image from its own target gallery for genuine tests.
    PAD is intentionally not part of this static-image diagnostic.
    """
    import numpy as np

    target = recognizer.format_person_name(target)
    query = matrix[query_index]

    target_indices = list(identity_to_indices.get(target, []))
    is_positive = query_index in target_indices

    if is_positive:
        target_pool = [i for i in target_indices if i != query_index]
    else:
        target_pool = target_indices

    competitor_identity = "Unknown"
    competitor_best = -1.0

    # All non-target identities form the competitor/impostor pool.
    competitor_pools: dict[str, list[int]] = {}
    for identity, group in identity_to_indices.items():
        if identity == target:
            continue
        pool = [i for i in group if i != query_index]
        if pool:
            competitor_pools[identity] = pool

    if not target_pool:
        return {
            "valid": False,
            "positive": is_positive,
            "actual_identity": names[query_index] if query_index < len(names) else "Unknown",
            "target": target,
            "candidate": "Unknown",
            "best": 0.0,
            "topk": 0.0,
            "margin": 0.0,
            "competitor": "Unknown",
            "competitor_best": 0.0,
            "reason": "target_gallery_pool_empty",
            "gallery_best_identity": "Unknown",
            "gallery_best_similarity": 0.0,
        }

    target_scores = np.sort(
        np.asarray(np.dot(matrix[target_pool], query), dtype=np.float32)
    )[::-1]

    best = float(target_scores[0])
    topk = target_scores[
        : min(recognizer.TOP_K_TARGET_SAMPLES, target_scores.size)
    ]
    topk_mean = float(np.mean(topk)) if topk.size else 0.0

    gallery_scores: dict[str, float] = {target: best}

    for identity, pool in competitor_pools.items():
        score = float(
            np.max(
                np.asarray(
                    np.dot(matrix[pool], query),
                    dtype=np.float32,
                )
            )
        )
        gallery_scores[identity] = score

        if score > competitor_best:
            competitor_best = score
            competitor_identity = identity

    if competitor_best < 0.0:
        competitor_best = 0.0

    margin = best - competitor_best

    gallery_best_identity = max(
        gallery_scores,
        key=lambda identity: (gallery_scores[identity], identity),
    )
    gallery_best_similarity = float(gallery_scores[gallery_best_identity])

    threshold = float(
        thresholds.get(target, recognizer.DEFAULT_MAX_SIMILARITY)
    )
    topk_threshold = float(
        topk_thresholds.get(target, recognizer.DEFAULT_TOPK_MEAN)
    )
    margin_threshold = float(
        margin_thresholds.get(target, recognizer.DEFAULT_MARGIN)
    )

    target_is_gallery_winner = gallery_best_identity == target

    accepted = bool(
        target_is_gallery_winner
        and best >= threshold
        and topk_mean >= topk_threshold
        and margin >= margin_threshold
        and best >= competitor_best
    )

    if not target_is_gallery_winner:
        reason = "target_not_gallery_winner"
    elif best < threshold:
        reason = "target_best_below_threshold"
    elif topk_mean < topk_threshold:
        reason = "target_top3_support_below_threshold"
    elif margin < margin_threshold:
        reason = "target_margin_below_threshold"
    else:
        reason = "accepted_target_identity"

    return {
        "valid": True,
        "positive": is_positive,
        "actual_identity": names[query_index] if query_index < len(names) else "Unknown",
        "target": target,
        "candidate": target if accepted else "Unknown",
        "accepted": accepted,
        "best": best,
        "topk": topk_mean,
        "margin": margin,
        "competitor": competitor_identity,
        "competitor_best": competitor_best,
        "threshold": threshold,
        "topk_threshold": topk_threshold,
        "margin_threshold": margin_threshold,
        "reason": reason,
        "gallery_best_identity": gallery_best_identity,
        "gallery_best_similarity": gallery_best_similarity,
    }


def run_validation_report(
    validation_root: str | Path | None = None,
    output_path: str | Path | None = None,
    timeout: float = 300.0,
    target_identity: str | None = None,
) -> dict:
    """Run identity validation using the real known_faces gallery.

    The test is deliberately leave-one-out:
      - each gallery image is queried;
      - that exact image is removed from its own genuine target pool;
      - same-identity queries are EXPECTED ACCEPT;
      - different enrolled identities are EXPECTED REJECT.

    This measures gallery identity separation only. It is not a held-out
    production accuracy estimate because the thresholds are also calibrated
    from known_faces/ and the same gallery is used for enrollment.

    validation_root is kept only for backward compatibility and is ignored.
    output_path is optional; when provided, a JSON report is written there.
    """
    del validation_root

    recognizer = get_face_recognizer()
    recognizer.prepare_process_async()

    if not recognizer.wait_until_ready(timeout=timeout):
        status = recognizer.get_runtime_status()
        raise RuntimeError(
            "Face recognition model is not ready: "
            f"{status.get('stage', 'Unknown')}"
        )

    import numpy as np

    with recognizer._lock:
        matrix = np.asarray(
            recognizer._embedding_matrix,
            dtype=np.float32,
        )
        names = list(recognizer._embedding_names)
        paths = list(recognizer._embedding_paths)
        identity_to_indices = {
            name: list(indices)
            for name, indices in recognizer._identity_to_indices.items()
        }
        thresholds = dict(recognizer._identity_thresholds)
        topk_thresholds = dict(recognizer._identity_topk_thresholds)
        margin_thresholds = dict(recognizer._identity_margin_thresholds)

    if matrix.size == 0 or not names:
        raise RuntimeError("known_faces tidak menghasilkan embedding.")

    identities = sorted(identity_to_indices)

    if target_identity:
        normalized_target = recognizer.format_person_name(target_identity)
        if normalized_target not in identity_to_indices:
            raise ValueError(
                f"Target '{normalized_target}' tidak ditemukan di known_faces."
            )
        identities = [normalized_target]

    print("", flush=True)
    print("=" * 96, flush=True)
    print(
        "🧪 [VALIDATION] KNOWN-FACES LEAVE-ONE-OUT IDENTITY TEST",
        flush=True,
    )
    print("=" * 96, flush=True)
    print(
        "⚠️ Scope     : ArcFace identity matching + top-3 + target margin",
        flush=True,
    )
    print(
        "⚠️ Not test : live PAD, browser camera transport, attendance saving",
        flush=True,
    )
    print(
        "⚠️ Meaning   : internal gallery separation, NOT held-out production accuracy",
        flush=True,
    )
    print(
        f"📚 Gallery   : {len(names)} embeddings | "
        f"identities={', '.join(sorted(identity_to_indices))}",
        flush=True,
    )
    print("", flush=True)

    summary: dict[str, dict[str, Any]] = {
        identity: {
            "genuine_total": 0,
            "genuine_accept": 0,
            "genuine_reject": 0,
            "impostor_total": 0,
            "impostor_false_accept": 0,
            "impostor_reject": 0,
            "examples": [],
        }
        for identity in identities
    }

    all_results: list[dict[str, Any]] = []

    for target in identities:
        print("-" * 96, flush=True)
        print(
            f"🎯 TARGET = {target} | "
            f"sim>={thresholds.get(target, recognizer.DEFAULT_MAX_SIMILARITY):.3f} | "
            f"top3>={topk_thresholds.get(target, recognizer.DEFAULT_TOPK_MEAN):.3f} | "
            f"margin>={margin_thresholds.get(target, recognizer.DEFAULT_MARGIN):.3f}",
            flush=True,
        )
        print("-" * 96, flush=True)

        for query_index, actual_identity in enumerate(names):
            result = _validation_score_query(
                recognizer=recognizer,
                matrix=matrix,
                names=names,
                identity_to_indices=identity_to_indices,
                query_index=query_index,
                target=target,
                thresholds=thresholds,
                topk_thresholds=topk_thresholds,
                margin_thresholds=margin_thresholds,
            )

            if not result["valid"]:
                continue

            image_name = (
                Path(paths[query_index]).name
                if query_index < len(paths)
                else f"embedding_{query_index}"
            )

            result["query_index"] = query_index
            result["query_path"] = paths[query_index]
            all_results.append(result)

            same_identity = (
                actual_identity.casefold() == target.casefold()
            )

            if same_identity:
                summary[target]["genuine_total"] += 1

                if result["accepted"]:
                    summary[target]["genuine_accept"] += 1
                else:
                    summary[target]["genuine_reject"] += 1

                print(
                    f"   [GENUINE] "
                    f"actual={actual_identity:<14} "
                    f"query={image_name:<28} "
                    f"target={target:<14} "
                    f"sim={result['best']:.3f} "
                    f"top3={result['topk']:.3f} "
                    f"margin={result['margin']:.3f} "
                    f"gallery={result['gallery_best_identity']:<14} "
                    f"result={'ACCEPT' if result['accepted'] else 'REJECT':<6} "
                    f"reason={result['reason']}",
                    flush=True,
                )
            else:
                summary[target]["impostor_total"] += 1

                if result["accepted"]:
                    summary[target]["impostor_false_accept"] += 1
                else:
                    summary[target]["impostor_reject"] += 1

                print(
                    f"   [IMPOSTOR] "
                    f"actual={actual_identity:<14} "
                    f"query={image_name:<28} "
                    f"target={target:<14} "
                    f"sim={result['best']:.3f} "
                    f"top3={result['topk']:.3f} "
                    f"margin={result['margin']:.3f} "
                    f"competitor={result['competitor']:<14} "
                    f"result={'FALSE_ACCEPT' if result['accepted'] else 'REJECT':<11} "
                    f"reason={result['reason']}",
                    flush=True,
                )

    def pct(a: int, b: int) -> float:
        return (a / b * 100.0) if b else 0.0

    total_genuine = sum(v["genuine_total"] for v in summary.values())
    total_genuine_accept = sum(v["genuine_accept"] for v in summary.values())
    total_impostor = sum(v["impostor_total"] for v in summary.values())
    total_false_accept = sum(v["impostor_false_accept"] for v in summary.values())

    print("", flush=True)
    print("=" * 96, flush=True)
    print("📊 [VALIDATION] SUMMARY", flush=True)
    print("=" * 96, flush=True)

    for identity, data in summary.items():
        genuine_rate = pct(data["genuine_accept"], data["genuine_total"])
        far_rate = pct(
            data["impostor_false_accept"],
            data["impostor_total"],
        )
        print(
            f"   {identity:<14} | "
            f"GENUINE {data['genuine_accept']}/{data['genuine_total']} "
            f"({genuine_rate:.1f}%) | "
            f"CROSS-ID FALSE ACCEPT "
            f"{data['impostor_false_accept']}/{data['impostor_total']} "
            f"({far_rate:.1f}%)",
            flush=True,
        )

    overall_genuine_rate = pct(total_genuine_accept, total_genuine)
    overall_far = pct(total_false_accept, total_impostor)

    print("", flush=True)
    print(
        f"   OVERALL genuine acceptance : "
        f"{overall_genuine_rate:.1f}% "
        f"({total_genuine_accept}/{total_genuine})",
        flush=True,
    )
    print(
        f"   OVERALL cross-ID FAR       : "
        f"{overall_far:.1f}% "
        f"({total_false_accept}/{total_impostor})",
        flush=True,
    )
    print("=" * 96, flush=True)

    result_payload = {
        "source": "known_faces_leave_one_out",
        "scope": "identity_only_internal_diagnostic",
        "held_out_accuracy": False,
        "gallery_embeddings": len(names),
        "identities": sorted(identity_to_indices),
        "genuine_acceptance_rate_percent": overall_genuine_rate,
        "cross_identity_false_accept_rate_percent": overall_far,
        "summary": summary,
        "thresholds": {
            identity: {
                "similarity": float(
                    thresholds.get(
                        identity,
                        recognizer.DEFAULT_MAX_SIMILARITY,
                    )
                ),
                "top3": float(
                    topk_thresholds.get(
                        identity,
                        recognizer.DEFAULT_TOPK_MEAN,
                    )
                ),
                "margin": float(
                    margin_thresholds.get(
                        identity,
                        recognizer.DEFAULT_MARGIN,
                    )
                ),
            }
            for identity in sorted(identity_to_indices)
        },
        "queries": all_results,
    }

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"📝 Validation JSON saved: {output.resolve()}",
            flush=True,
        )

    print(
        "✅ Validation selesai. Ini adalah diagnostic internal; "
        "belum merupakan held-out production accuracy.",
        flush=True,
    )

    return result_payload


def run_validation_image_test(
    image_path: str | Path,
    target_identity: str,
    timeout: float = 300.0,
) -> dict:
    """Test one arbitrary image against the production identity gate.

    Useful for checking:
        Darren target + Darren image   -> expected ACCEPT
        Darren target + friend image   -> expected REJECT / Unknown

    PAD is intentionally disabled here because a static image is not a
    live-camera PAD test. The ArcFace identity decision is kept unchanged.
    """
    recognizer = get_face_recognizer()
    recognizer.prepare_process_async()

    if not recognizer.wait_until_ready(timeout=timeout):
        status = recognizer.get_runtime_status()
        raise RuntimeError(
            "Face recognition model is not ready: "
            f"{status.get('stage', 'Unknown')}"
        )

    import cv2

    path = Path(image_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Image tidak ditemukan: {path}")

    frame = recognizer._read_image(str(path))
    if frame is None:
        raise ValueError(f"Image gagal dibaca: {path}")

    detections = recognizer._detect_face_best_scale(frame)
    if not detections:
        raise ValueError(f"Tidak ada wajah terdeteksi: {path}")

    detection = detections[0]

    candidate, similarity, accepted = recognizer.recognize_detection(
        frame=frame,
        detection=detection,
        target_identity=target_identity,
        require_liveness=False,
    )

    metrics = recognizer.get_last_match_metrics()

    print("", flush=True)
    print("=" * 96, flush=True)
    print("🖼️ [VALIDATION IMAGE] SINGLE-IMAGE IDENTITY TEST", flush=True)
    print("=" * 96, flush=True)
    print(f"Image            : {path.resolve()}", flush=True)
    print(f"Target HRIS      : {metrics['target']}", flush=True)
    print(f"Result           : {'ACCEPT' if accepted else 'REJECT / Unknown'}", flush=True)
    print(f"Candidate        : {candidate}", flush=True)
    print(f"Similarity       : {similarity:.4f}", flush=True)
    print(f"Threshold        : {metrics['target_threshold']:.4f}", flush=True)
    print(f"Top-3 mean       : {metrics['topk_mean']:.4f}", flush=True)
    print(f"Top-3 threshold  : {metrics['target_topk_threshold']:.4f}", flush=True)
    print(f"Target margin    : {metrics['target_margin']:.4f}", flush=True)
    print(f"Margin threshold : {metrics['target_margin_threshold']:.4f}", flush=True)
    print(f"Competitor best  : {metrics['competitor_best']:.4f}", flush=True)
    print(f"Gallery winner   : {metrics['gallery_best']}", flush=True)
    print(f"Gallery winner sim: {metrics['gallery_best_sim']:.4f}", flush=True)
    print(f"Reason           : {metrics['decision_reason']}", flush=True)
    print("=" * 96, flush=True)

    return {
        "image": str(path.resolve()),
        "target": metrics["target"],
        "candidate": candidate,
        "accepted": bool(accepted),
        "similarity": float(similarity),
        "metrics": metrics,
    }


def run_validation_camera_test(
    target_identity: str,
    camera_index: int = 0,
    timeout: float = 300.0,
) -> None:
    """Interactive live-camera identity test with PAD, no attendance saving.

    Use this for:
        - Darren target + Darren in front of camera
        - Darren target + Darren's friend in front of camera

    The test intentionally does not call camera_server attendance/evidence APIs.
    """
    recognizer = get_face_recognizer()
    recognizer.prepare_process_async()

    if not recognizer.wait_until_ready(timeout=timeout):
        status = recognizer.get_runtime_status()
        raise RuntimeError(
            "Face recognition model is not ready: "
            f"{status.get('stage', 'Unknown')}"
        )

    import cv2
    import time as _time

    target = recognizer.format_person_name(target_identity)
    if target not in recognizer.get_known_names():
        raise ValueError(
            f"Target '{target}' tidak ada di known_faces. "
            f"Pilihan: {', '.join(recognizer.get_known_names())}"
        )

    cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        raise RuntimeError(
            f"Camera index {camera_index} tidak dapat dibuka."
        )

    # Prefer the same webcam quality recommended for production evidence.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    last_run = 0.0
    last_result = {
        "candidate": "Unknown",
        "similarity": 0.0,
        "accepted": False,
        "metrics": {},
    }

    print("", flush=True)
    print("=" * 96, flush=True)
    print("📷 [VALIDATION CAMERA] LIVE IDENTITY TEST", flush=True)
    print("=" * 96, flush=True)
    print(f"Target HRIS : {target}", flush=True)
    print("Q / ESC     : keluar", flush=True)
    print(
        "Expected    : target asli -> ACCEPT | orang lain -> REJECT / Unknown",
        flush=True,
    )
    print("=" * 96, flush=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("⚠️ Camera frame gagal dibaca.", flush=True)
                break

            detection, _stable_box = recognizer.track_face(frame)

            now = _time.monotonic()
            if detection is None:
                last_result = {
                    "candidate": "Unknown",
                    "similarity": 0.0,
                    "accepted": False,
                    "metrics": {
                        "decision_reason": "face_not_detected",
                        "target": target,
                    },
                }
            elif now - last_run >= 0.22:
                last_run = now

                candidate, similarity, accepted = recognizer.recognize_detection(
                    frame=frame,
                    detection=detection,
                    target_identity=target,
                    require_liveness=True,
                )

                last_result = {
                    "candidate": candidate,
                    "similarity": float(similarity),
                    "accepted": bool(accepted),
                    "metrics": recognizer.get_last_match_metrics(),
                }

            metrics = last_result.get("metrics", {})
            candidate = last_result.get("candidate", "Unknown")
            similarity = float(last_result.get("similarity", 0.0))
            accepted = bool(last_result.get("accepted", False))

            h, w = frame.shape[:2]

            status_text = (
                "ACCEPT" if accepted else "REJECT / UNKNOWN"
            )
            status_scale = 0.95
            status_color = (
                (40, 200, 40) if accepted else (40, 40, 220)
            )

            cv2.rectangle(
                frame,
                (10, 10),
                (min(w - 10, 800), min(h - 150, 180)),
                (20, 20, 20),
                -1,
            )

            cv2.putText(
                frame,
                f"TARGET: {target}",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                status_text,
                (25, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                status_scale,
                status_color,
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"sim={similarity:.3f} | "
                f"top3={float(metrics.get('topk_mean', 0.0)):.3f} | "
                f"margin={float(metrics.get('target_margin', 0.0)):.3f}",
                (25, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"gallery={metrics.get('gallery_best', 'Unknown')} | "
                f"competitor={metrics.get('gallery_runner', 'Unknown')}",
                (25, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"reason={metrics.get('decision_reason', 'not_evaluated')}",
                (25, 173),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Face Recognition Validation - No Attendance", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        recognizer.reset_anti_spoof("validation_camera_exit")


def _main_cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="HR Attendance ArcFace validation utilities."
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run known_faces leave-one-out identity validation.",
    )
    parser.add_argument(
        "--validate-image",
        action="append",
        default=[],
        help="Test an arbitrary image. Repeat the option for multiple images.",
    )
    parser.add_argument(
        "--validate-camera",
        action="store_true",
        help="Open a live camera identity test without saving attendance.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="HRIS identity target, e.g. Darren.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index for --validate-camera.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path for --validate.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Model/gallery preload timeout in seconds.",
    )

    args = parser.parse_args()

    if args.validate_camera:
        if not args.target:
            parser.error("--validate-camera membutuhkan --target, contoh: --target Darren")
        run_validation_camera_test(
            target_identity=args.target,
            camera_index=args.camera,
            timeout=args.timeout,
        )
        return 0

    if args.validate_image:
        if not args.target:
            parser.error("--validate-image membutuhkan --target, contoh: --target Darren")
        for image in args.validate_image:
            run_validation_image_test(
                image_path=image,
                target_identity=args.target,
                timeout=args.timeout,
            )
        return 0

    if args.validate:
        run_validation_report(
            output_path=args.output,
            timeout=args.timeout,
            target_identity=args.target,
        )
        return 0

    parser.print_help()
    return 0


__all__ = [
    "FaceRecognizer",
    "face_recognizer",
    "get_face_recognizer",
    "start_process_preload",
    "prepare_async",
    "is_model_ready",
    "is_detector_ready",
    "is_embeddings_ready",
    "is_ready",
    "get_runtime_status",
    "print_runtime_status",
    "is_loading",
    "get_loading_status",
    "get_loading_progress",
    "detect_faces",
    "reset_liveness",
    "reset_anti_spoof",
    "reload",
    "clear_embeddings",
    "get_dataset_file_count",
    "get_known_names",
    "get_known_paths",
    "get_validation_file_count",
    "get_validation_names",
    "run_validation_report",
    "run_validation_image_test",
    "run_validation_camera_test",
]


if __name__ == "__main__":
    raise SystemExit(_main_cli())



