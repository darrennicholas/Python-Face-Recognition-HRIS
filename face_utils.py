# ==========================================================
# face_utils.py
# FINAL STABLE + LIGHTWEIGHT FACE RECOGNITION
#
# Detector    : OpenCV YuNet INT8
# Recognition : DeepFace + SFace
# Faces       : exactly 1 best face
# Lifecycle   : one background preload, resident for the process
#
# Main improvements:
#   1. Native OpenCV YuNet detection.
#   2. 1-face-only policy.
#   3. One controlled 2x fallback for far/small faces.
#   4. YuNet 5-point landmarks retained for SFace alignment.
#   5. Smooth/stable bounding box using temporal EMA.
#   6. Short missed-detection hold to reduce box flicker.
#   7. SFace recognition uses aligned face crops.
#   8. Per-person prototype + best-sample matching.
#   9. Stronger Unknown rejection than the old 0.60 distance rule.
#  10. Known-face embedding cache.
#  11. Process-level one-time preload for local/server deployment.
#
# Compatible public API:
#   face_recognizer
#   get_face_recognizer()
#   prepare_async()
#   is_model_ready()
#   is_detector_ready()
#   is_embeddings_ready()
#   is_ready()
#   is_loading()
#   get_loading_status()
#   get_loading_progress()
#   detect_faces()
#   reload()
#   clear_embeddings()
#   get_dataset_file_count()
#   get_known_names()
#   get_known_paths()
# ==========================================================

from __future__ import annotations

import os
import pickle
import re
import threading
import urllib.request
from pathlib import Path
from typing import Any


class FaceRecognizer:
    """
    One-person, low-CPU face-recognition engine.

    Roles:
      OpenCV YuNet  -> locate exactly one face + bounding box landmarks
      OpenCV        -> crop/alignment image operations
      SFace         -> identity embedding
      cosine score  -> Darren / Unknown

    The gallery is built from known_faces/.
    Files such as:
        darren_1.jpg
        darren_2.jpg
        ...
        darren_10.jpg
    are normalized to the identity name "Darren".
    """

    # ======================================================
    # CONFIG
    # ======================================================

    EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    MODEL_NAME = "SFace"
    MODEL_LABEL = "DeepFace + SFace"

    YUNET_MODEL_FILENAME = (
        "face_detection_yunet_2023mar_int8.onnx"
    )

    YUNET_DOWNLOAD_URL = (
        "https://huggingface.co/opencv/opencv_zoo/resolve/main/"
        "models/face_detection_yunet/"
        "face_detection_yunet_2023mar_int8.onnx"
    )

    # Detector settings.
    # Slightly permissive so a face farther from the camera
    # is not rejected too early.
    YUNET_SCORE_THRESHOLD = 0.45
    YUNET_NMS_THRESHOLD = 0.30
    YUNET_TOP_K = 10
    MAX_FACES = 1

    # Fast live-camera tracking.
    FAST_DETECT_SCALE = 0.50
    FAST_FALLBACK_INTERVAL = 6

    # One fallback pass for small/far faces.
    FAR_FACE_UPSCALE = 2.0
    SMALL_FACE_WIDTH = 60
    SMALL_FACE_HEIGHT = 60

    # Bounding-box smoothing.
    BOX_SMOOTH_ALPHA = 0.88
    MAX_MISSED_DETECTIONS = 5

    # Identity smoothing.
    IDENTITY_SWITCH_CONFIRMATIONS = 2

    # ------------------------------------------------------
    # SFace matching.
    #
    # The old implementation required similarity >= 0.60.
    # For a one-person gallery that can be unnecessarily strict.
    #
    # We use a lower baseline and then strengthen it with
    # multi-sample consistency from Darren's 10 images.
    # ------------------------------------------------------
    MIN_SIMILARITY = 0.45
    MIN_TOP3_MEAN_SIMILARITY = 0.48
    MAX_TOP3_SPREAD = 0.25

    # Keep all known samples, but use the strongest few
    # for the final stability score.
    TOP_K_SAMPLES = 3
    CALIBRATION_MARGIN = 0.025
    DEFAULT_CALIBRATED_SIMILARITY = 0.50
    ABSOLUTE_MIN_SIMILARITY = 0.42
    MAX_CALIBRATED_SIMILARITY = 0.70

    ALIGN_SIZE = 160

    CACHE_VERSION = 15
    CACHE_FILENAME = (
        ".face_embeddings_cache.pkl"
    )

    # ======================================================
    # INIT
    # ======================================================

    def __init__(self) -> None:

        self.base_dir = (
            Path(__file__).resolve().parent
        )

        self.known_faces_dir = (
            self.base_dir / "known_faces"
        )

        self.known_faces_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.models_dir = (
            self.base_dir / "models"
        )

        self.models_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.yunet_model_path = (
            self.models_dir
            / self.YUNET_MODEL_FILENAME
        )

        self.embedding_cache_path = (
            self.base_dir
            / self.CACHE_FILENAME
        )

        self.model_name = self.MODEL_NAME
        self.tolerance = 0.60

        # Heavy models.
        self._deepface: Any = None
        self._facenet_model: Any = None  # internal attribute kept for API stability; stores SFace
        self._yunet: Any = None

        self._model_ready = False
        self._detector_ready = False

        # Known gallery.
        self._embedding_names: list[str] = []
        self._embedding_matrix: Any = None
        self._embeddings_ready = False
        self._embedding_signature: Any = None

        # One prototype per identity.
        self._identity_names: list[str] = []
        self._identity_prototypes: Any = None

        # Async loading state (one background worker shared by the process).
        self._loading = False
        self._loading_stage = "Idle"
        self._loading_error: str | None = None
        self._prepare_thread: threading.Thread | None = None
        self._camera_prepare_requested = False
        # Set only when the full SFace + embedding gallery + YuNet stack is ready.
        self._ready_event = threading.Event()

        self._lock = threading.RLock()

        # Temporal tracking state.
        self._smoothed_box: tuple[
            float,
            float,
            float,
            float,
        ] | None = None

        self._missed_detections = 0
        self._fast_track_frame = 0

        self._last_identity = "Unknown"
        self._pending_identity = "Unknown"
        self._pending_identity_count = 0

        # Last match metrics are useful for debugging threshold.
        self._last_similarity = 0.0
        self._last_top3_mean = 0.0
        self._last_match_name = "Unknown"
        self._calibrated_similarity = (
            self.DEFAULT_CALIBRATED_SIMILARITY
        )
        self._calibration_samples = 0

        self.known_paths: list[str] = []
        self.known_names: list[str] = []

        self._index_dataset()

    # ======================================================
    # DATASET
    # ======================================================

    @staticmethod
    def clean_name(
        filename: str | os.PathLike | None,
    ) -> str:

        if not filename:
            return "Unknown"

        value = os.path.splitext(
            os.path.basename(str(filename))
        )[0]

        # darren_1 / darren_10 -> Darren
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
            " ".join(
                value.split()
            ).strip()
            or "Unknown"
        )

    def _get_files(self) -> list[Path]:

        try:

            if not self.known_faces_dir.exists():
                return []

            return sorted(
                [
                    path
                    for path in self.known_faces_dir.iterdir()
                    if (
                        path.is_file()
                        and path.suffix.lower()
                        in self.EXTENSIONS
                    )
                ],
                key=lambda path:
                path.name.lower(),
            )

        except Exception as exc:

            print(
                f"⚠️ Dataset scan error: {exc}"
            )

            return []

    def _index_dataset(self) -> None:

        files = self._get_files()

        with self._lock:

            self.known_paths = [
                str(path)
                for path in files
            ]

            self.known_names = [
                self.clean_name(path.name)
                for path in files
            ]

    def get_dataset_file_count(
        self,
    ) -> int:

        return len(
            self._get_files()
        )

    def get_total_encodings(
        self,
    ) -> int:

        with self._lock:
            return len(
                self._embedding_names
            )

    def get_known_names(
        self,
    ) -> list[str]:

        names: list[str] = []

        for path in self._get_files():

            name = self.clean_name(
                path.name
            )

            if (
                name != "Unknown"
                and name not in names
            ):

                names.append(name)

        return names

    def get_known_paths(
        self,
    ) -> list[str]:

        return [
            str(path)
            for path in self._get_files()
        ]

    def get_dataset_names_from_disk(
        self,
    ) -> list[str]:

        return self.get_known_names()

    # ======================================================
    # STATUS
    # ======================================================

    def _set_stage(self, stage: str) -> None:
        with self._lock:
            self._loading_stage = stage

    def get_loading_status(self) -> str:

        with self._lock:

            if self._loading_error:
                return self._loading_error

            return self._loading_stage

    def get_loading_error(
        self,
    ) -> str | None:

        with self._lock:
            return self._loading_error

    def get_loading_progress(self) -> int:

        with self._lock:
            stage = self._loading_stage

        mapping = {
            "Idle": 0,
            "Preparing SFace + gallery...": 15,
            "Face model + gallery ready": 88,
            "Face model ready - gallery waits for START": 70,
            "Loading face gallery...": 88,
            "Checking face detector...": 5,
            "Downloading YuNet detector...": 10,
            "Loading YuNet detector...": 18,
            "Importing DeepFace...": 28,
            "Loading SFace model...": 55,
            "SFace model ready": 65,
            "Loading cached embeddings...": 75,
            "Building face embeddings...": 80,
            "Loading YuNet for START...": 35,
            "Face recognition ready": 100,
        }

        if stage.startswith(
            "Building face embeddings"
        ):

            match = re.search(
                r"\((\d+)/(\d+)\)",
                stage,
            )

            if match:

                current = int(
                    match.group(1)
                )

                total = max(
                    int(match.group(2)),
                    1,
                )

                return min(
                    98,
                    80
                    + int(
                        18
                        * current
                        / total
                    ),
                )

            return 80

        return mapping.get(
            stage,
            40,
        )

    # ======================================================
    # IMAGE
    # ======================================================

    @staticmethod
    def _read_image(path: str) -> Any:

        try:

            import cv2

            image = cv2.imread(
                path,
                cv2.IMREAD_COLOR,
            )

            if image is not None:
                return image

            import numpy as np

            data = np.fromfile(
                path,
                dtype=np.uint8,
            )

            if data.size == 0:
                return None

            return cv2.imdecode(
                data,
                cv2.IMREAD_COLOR,
            )

        except Exception as exc:

            print(
                f"⚠️ Image read error {path}: {exc}"
            )

            return None

    # ======================================================
    # YUNET
    # ======================================================

    def _find_local_yunet(
        self,
    ) -> Path | None:

        candidates = [
            self.yunet_model_path,
            self.base_dir
            / self.YUNET_MODEL_FILENAME,
        ]

        for path in candidates:

            try:

                if (
                    path.exists()
                    and path.is_file()
                    and path.stat().st_size
                    > 10_000
                ):

                    return path

            except OSError:
                continue

        return None

    def _download_yunet(
        self,
    ) -> Path | None:

        existing = self._find_local_yunet()

        if existing is not None:
            return existing

        self._set_stage(
            "Downloading YuNet detector..."
        )

        target = self.yunet_model_path
        temp = target.with_suffix(
            target.suffix + ".tmp"
        )

        try:

            print(
                "⏳ Downloading YuNet INT8 detector..."
            )

            request = urllib.request.Request(
                self.YUNET_DOWNLOAD_URL,
                headers={
                    "User-Agent":
                    "HR-Attendance-System/1.0"
                },
            )

            with urllib.request.urlopen(
                request,
                timeout=30,
            ) as response:

                with temp.open(
                    "wb"
                ) as output:

                    while True:

                        chunk = response.read(
                            1024 * 1024
                        )

                        if not chunk:
                            break

                        output.write(
                            chunk
                        )

            if (
                not temp.exists()
                or temp.stat().st_size
                <= 10_000
            ):

                raise RuntimeError(
                    "YuNet model download incomplete."
                )

            temp.replace(
                target
            )

            print(
                "✅ YuNet model ready on disk."
            )

            return target

        except Exception as exc:

            print(
                f"⚠️ YuNet download error: {exc}"
            )

            try:
                temp.unlink(
                    missing_ok=True
                )
            except Exception:
                pass

            return None

    def _create_yunet(
        self,
        model_path: Path,
    ) -> Any:

        import cv2

        factory = getattr(
            cv2,
            "FaceDetectorYN_create",
            None,
        )

        if factory is not None:

            return factory(
                str(model_path),
                "",
                (320, 320),
                self.YUNET_SCORE_THRESHOLD,
                self.YUNET_NMS_THRESHOLD,
                self.YUNET_TOP_K,
            )

        detector_cls = getattr(
            cv2,
            "FaceDetectorYN",
            None,
        )

        if (
            detector_cls is None
            or not hasattr(
                detector_cls,
                "create",
            )
        ):

            raise RuntimeError(
                "OpenCV FaceDetectorYN tidak tersedia."
            )

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

            if self._detector_ready:
                return True

            if self._yunet is not None:

                self._detector_ready = True
                return True

        model_path = (
            self._find_local_yunet()
            or self._download_yunet()
        )

        if model_path is None:

            with self._lock:
                self._loading_error = (
                    "YuNet model tidak ditemukan."
                )

            return False

        try:

            self._set_stage(
                "Loading YuNet detector..."
            )

            detector = self._create_yunet(
                model_path
            )

            with self._lock:

                self._yunet = detector
                self._detector_ready = True
                self._loading_error = None

            print(
                "✅ OpenCV YuNet initialized."
            )

            return True

        except Exception as exc:

            with self._lock:

                self._yunet = None
                self._detector_ready = False
                self._loading_error = (
                    f"YuNet gagal dimuat: {exc}"
                )

            print(
                f"❌ YuNet error: {exc}"
            )

            return False

    # ======================================================
    # DEEPFACE / SFace
    # ======================================================

    _runtime_configured = False
    _runtime_config_lock = threading.Lock()

    @classmethod
    def _configure_runtime(cls) -> None:
        """
        Keep TensorFlow/OpenMP from spawning excessive CPU workers on
        low-spec laptops. This is primarily a responsiveness optimization:
        it prevents model loading/inference from monopolizing the machine.
        """
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

    def _load_deepface(self) -> Any:

        with self._lock:

            if self._deepface is not None:
                return self._deepface

        try:

            self._set_stage(
                "Importing DeepFace..."
            )

            # Configure CPU limits BEFORE TensorFlow/DeepFace is imported.
            # This avoids excessive CPU contention on low-spec machines.
            self._configure_runtime()

            from deepface import DeepFace

            with self._lock:
                self._deepface = DeepFace

            print(
                "✅ DeepFace imported."
            )

            return DeepFace

        except Exception as exc:

            with self._lock:

                self._loading_error = (
                    f"DeepFace gagal dimuat: {exc}"
                )

            print(
                f"❌ DeepFace import error: {exc}"
            )

            return None

    def _load_model(self) -> bool:

        DeepFace = self._load_deepface()

        if DeepFace is None:
            return False

        with self._lock:

            if self._model_ready:
                return True

            if self._facenet_model is not None:

                self._model_ready = True
                return True

        try:

            self._set_stage(
                "Loading SFace model..."
            )

            print(
                "⏳ Loading SFace..."
            )

            model = DeepFace.build_model(
                task="facial_recognition",
                model_name=self.model_name,
            )

            with self._lock:

                self._facenet_model = model
                self._model_ready = True
                self._loading_error = None

            print(
                "✅ SFace ready."
            )

            return True

        except Exception as exc:

            with self._lock:

                self._facenet_model = None
                self._model_ready = False
                self._loading_error = (
                    f"SFace gagal dimuat: {exc}"
                )

            print(
                f"❌ SFace error: {exc}"
            )

            return False

    # ======================================================
    # CACHE
    # ======================================================

    def _file_signature(self) -> tuple:

        result = []

        for path in self._get_files():

            try:

                stat = path.stat()

                result.append(
                    (
                        str(path),
                        stat.st_mtime_ns,
                        stat.st_size,
                    )
                )

            except OSError:
                continue

        return tuple(result)

    def _detector_signature(self) -> tuple:

        path = self._find_local_yunet()

        if path is None:

            return (
                self.YUNET_MODEL_FILENAME,
                "missing",
            )

        try:

            stat = path.stat()

            return (
                str(path),
                stat.st_mtime_ns,
                stat.st_size,
            )

        except OSError:

            return (
                str(path),
                "unknown",
            )

    def _cache_signature(self) -> dict:
        # Cache validity depends on the fixed model/preprocessing version and
        # known-face files, not on the detector file mtime/path. This lets a
        # valid cache be restored before YuNet is initialized.
        return {
            "version": self.CACHE_VERSION,
            "face_model": self.model_name,
            "detector": "opencv-yunet-int8-v1",
            "preprocess": "yunet-5point-align-160",
            "files": self._file_signature(),
        }

    # ======================================================
    # EMBEDDING CACHE
    # ======================================================

    def _install_embeddings(
        self,
        names: list[str],
        matrix: Any,
        signature: dict,
    ) -> None:

        import numpy as np

        matrix = np.asarray(
            matrix,
            dtype=np.float32,
        )

        if matrix.size == 0:

            matrix = np.empty(
                (0, 0),
                dtype=np.float32,
            )

        elif matrix.ndim == 1:

            matrix = matrix.reshape(
                1,
                -1,
            )

        matrix = np.ascontiguousarray(
            matrix,
            dtype=np.float32,
        )

        identity_names: list[str] = []
        identity_vectors: list[Any] = []

        grouped: dict[
            str,
            list[Any]
        ] = {}

        for name, vector in zip(
            names,
            matrix,
        ):

            grouped.setdefault(
                self.clean_name(name),
                [],
            ).append(vector)

        for name, vectors in (
            grouped.items()
        ):

            if not vectors:
                continue

            prototype = np.mean(
                np.vstack(vectors),
                axis=0,
            ).astype(
                np.float32
            )

            norm = float(
                np.linalg.norm(
                    prototype
                )
            )

            if norm <= 0:
                continue

            prototype = (
                prototype / norm
            ).astype(
                np.float32,
                copy=False,
            )

            identity_names.append(
                name
            )

            identity_vectors.append(
                prototype
            )

        if identity_vectors:

            identity_matrix = (
                np.ascontiguousarray(
                    np.vstack(
                        identity_vectors
                    ).astype(
                        np.float32,
                        copy=False,
                    )
                )
            )

        else:

            identity_matrix = np.empty(
                (0, 0),
                dtype=np.float32,
            )

        with self._lock:

            self._embedding_names = list(
                names
            )

            self._embedding_matrix = matrix

            self._identity_names = list(
                identity_names
            )

            self._identity_prototypes = (
                identity_matrix
            )

            self._embedding_signature = (
                signature
            )

            self._embeddings_ready = True

    def _load_embedding_cache(
        self,
        signature: dict,
    ) -> bool:

        self._set_stage(
            "Loading cached embeddings..."
        )

        try:

            if not self.embedding_cache_path.exists():
                return False

            with self.embedding_cache_path.open(
                "rb"
            ) as handle:

                payload = pickle.load(
                    handle
                )

            if not isinstance(
                payload,
                dict,
            ):
                return False

            if (
                payload.get("signature")
                != signature
            ):
                return False

            names = payload.get(
                "names"
            )

            matrix = payload.get(
                "matrix"
            )

            if (
                not isinstance(
                    names,
                    list,
                )
                or matrix is None
            ):

                legacy = payload.get(
                    "embeddings"
                )

                if not isinstance(
                    legacy,
                    list,
                ):
                    return False

                names = []
                vectors = []

                for item in legacy:

                    if (
                        isinstance(
                            item,
                            (list, tuple),
                        )
                        and len(item) == 2
                    ):

                        names.append(
                            str(item[0])
                        )

                        vectors.append(
                            item[1]
                        )

                matrix = vectors

            self._install_embeddings(
                names,
                matrix,
                signature,
            )

            print(
                "✅ Embedding cache loaded: "
                f"{len(names)} samples."
            )

            return True

        except Exception as exc:

            print(
                f"⚠️ Embedding cache error: {exc}"
            )

            return False

    def _save_embedding_cache(
        self,
        signature: dict,
        names: list[str],
        matrix: Any,
    ) -> None:

        temp = (
            self.embedding_cache_path.with_name(
                self.embedding_cache_path.name
                + ".tmp"
            )
        )

        try:

            payload = {
                "version":
                    self.CACHE_VERSION,
                "signature":
                    signature,
                "names":
                    list(names),
                "matrix":
                    matrix,
            }

            with temp.open(
                "wb"
            ) as handle:

                pickle.dump(
                    payload,
                    handle,
                    protocol=
                    pickle.HIGHEST_PROTOCOL,
                )

            temp.replace(
                self.embedding_cache_path
            )

        except Exception as exc:

            print(
                f"⚠️ Cache save error: {exc}"
            )

            try:
                temp.unlink(
                    missing_ok=True
                )
            except Exception:
                pass

    # ======================================================
    # YUNET DETECTION
    # ======================================================

    def _detect_on_image(
        self,
        frame: Any,
    ) -> list[dict]:

        with self._lock:
            detector = self._yunet

        if detector is None:
            return []

        try:

            height, width = (
                frame.shape[:2]
            )

            detector.setInputSize(
                (
                    int(width),
                    int(height),
                )
            )

            _, faces = detector.detect(
                frame
            )

            if faces is None:
                return []

            results: list[dict] = []

            for row in faces:

                if len(row) < 15:
                    continue

                x = float(row[0])
                y = float(row[1])
                w = float(row[2])
                h = float(row[3])

                score = float(
                    row[-1]
                )

                if (
                    score
                    < self.YUNET_SCORE_THRESHOLD
                ):
                    continue

                if (
                    w <= 0
                    or h <= 0
                ):
                    continue

                landmarks = [
                    (
                        float(row[4]),
                        float(row[5]),
                    ),
                    (
                        float(row[6]),
                        float(row[7]),
                    ),
                    (
                        float(row[8]),
                        float(row[9]),
                    ),
                    (
                        float(row[10]),
                        float(row[11]),
                    ),
                    (
                        float(row[12]),
                        float(row[13]),
                    ),
                ]

                x1 = max(
                    0,
                    min(
                        int(round(x)),
                        width - 1,
                    ),
                )

                y1 = max(
                    0,
                    min(
                        int(round(y)),
                        height - 1,
                    ),
                )

                x2 = max(
                    x1 + 1,
                    min(
                        int(round(x + w)),
                        width,
                    ),
                )

                y2 = max(
                    y1 + 1,
                    min(
                        int(round(y + h)),
                        height,
                    ),
                )

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

            results.sort(
                key=lambda item:
                item["score"],
                reverse=True,
            )

            return results[
                : self.MAX_FACES
            ]

        except Exception as exc:

            print(
                f"⚠️ YuNet detection error: {exc}"
            )

            return []

    def _detect_face_best_scale(
        self,
        frame: Any,
    ) -> list[dict]:

        import cv2

        native = self._detect_on_image(
            frame
        )

        if native:

            best = native[0]

            if (
                best["w"]
                >= self.SMALL_FACE_WIDTH
                and best["h"]
                >= self.SMALL_FACE_HEIGHT
            ):

                return native

        # One fallback pass for far/small faces.
        try:

            height, width = (
                frame.shape[:2]
            )

            scale = (
                self.FAR_FACE_UPSCALE
            )

            enlarged = cv2.resize(
                frame,
                (
                    int(width * scale),
                    int(height * scale),
                ),
                interpolation=
                cv2.INTER_CUBIC,
            )

            large = self._detect_on_image(
                enlarged
            )

            if not large:
                return native

            converted = []

            for item in large:

                converted.append(
                    {
                        "x":
                            int(
                                round(
                                    item["x"] / scale
                                )
                            ),
                        "y":
                            int(
                                round(
                                    item["y"] / scale
                                )
                            ),
                        "w":
                            max(
                                1,
                                int(
                                    round(
                                        item["w"]
                                        / scale
                                    )
                                ),
                            ),
                        "h":
                            max(
                                1,
                                int(
                                    round(
                                        item["h"]
                                        / scale
                                    )
                                ),
                            ),
                        "score":
                            item["score"],
                        "landmarks":
                            [
                                (
                                    lx / scale,
                                    ly / scale,
                                )
                                for lx, ly in
                                item["landmarks"]
                            ],
                    }
                )

            return converted[
                : self.MAX_FACES
            ]

        except Exception as exc:

            print(
                f"⚠️ YuNet far-face error: {exc}"
            )

            return native

    # ======================================================
    # BOUNDING BOX STABILITY
    # ======================================================

    def _smooth_box(
        self,
        detection: dict,
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int]:

        height, width = (
            frame_shape[:2]
        )

        current = (
            float(detection["x"]),
            float(detection["y"]),
            float(detection["w"]),
            float(detection["h"]),
        )

        previous = (
            self._smoothed_box
        )

        if previous is None:

            result = current

        else:

            alpha = (
                self.BOX_SMOOTH_ALPHA
            )

            result = tuple(
                (
                    alpha * current[i]
                    + (1.0 - alpha)
                    * previous[i]
                )
                for i in range(4)
            )

        x, y, w, h = result

        x = max(
            0.0,
            min(
                x,
                float(width - 1),
            ),
        )

        y = max(
            0.0,
            min(
                y,
                float(height - 1),
            ),
        )

        w = max(
            1.0,
            min(
                w,
                float(width) - x,
            ),
        )

        h = max(
            1.0,
            min(
                h,
                float(height) - y,
            ),
        )

        self._smoothed_box = (
            x,
            y,
            w,
            h,
        )

        self._missed_detections = 0

        return (
            int(round(x)),
            int(round(y)),
            int(round(w)),
            int(round(h)),
        )

    def _get_stable_box(
        self,
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int] | None:

        if self._smoothed_box is None:
            return None

        self._missed_detections += 1

        if (
            self._missed_detections
            > self.MAX_MISSED_DETECTIONS
        ):

            self._smoothed_box = None
            self._missed_detections = 0

            return None

        x, y, w, h = (
            self._smoothed_box
        )

        height, width = (
            frame_shape[:2]
        )

        x = max(
            0.0,
            min(
                x,
                float(width - 1),
            ),
        )

        y = max(
            0.0,
            min(
                y,
                float(height - 1),
            ),
        )

        w = max(
            1.0,
            min(
                w,
                float(width) - x,
            ),
        )

        h = max(
            1.0,
            min(
                h,
                float(height) - y,
            ),
        )

        return (
            int(round(x)),
            int(round(y)),
            int(round(w)),
            int(round(h)),
        )

    def _reset_tracking_state(
        self,
    ) -> None:

        self._smoothed_box = None
        self._missed_detections = 0
        self._fast_track_frame = 0

        self._last_identity = "Unknown"
        self._pending_identity = "Unknown"
        self._pending_identity_count = 0

        self._last_similarity = 0.0
        self._last_top3_mean = 0.0
        self._last_match_name = "Unknown"

    # ======================================================
    # FACE ALIGNMENT
    # ======================================================

    def _align_face(
        self,
        frame: Any,
        detection: dict,
    ) -> Any:

        try:

            import cv2
            import numpy as np

            landmarks = np.asarray(
                detection.get(
                    "landmarks",
                    [],
                ),
                dtype=np.float32,
            )

            if landmarks.shape != (
                5,
                2,
            ):

                return self._crop_face(
                    frame,
                    detection,
                )

            # Reference points for SFace alignment.
            target = np.asarray(
                [
                    [46.0, 58.0],
                    [114.0, 58.0],
                    [80.0, 86.0],
                    [51.0, 116.0],
                    [109.0, 116.0],
                ],
                dtype=np.float32,
            )

            transform, _ = (
                cv2.estimateAffinePartial2D(
                    landmarks,
                    target,
                    method=cv2.LMEDS,
                )
            )

            if transform is None:

                return self._crop_face(
                    frame,
                    detection,
                )

            aligned = cv2.warpAffine(
                frame,
                transform,
                (
                    self.ALIGN_SIZE,
                    self.ALIGN_SIZE,
                ),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

            if (
                aligned is None
                or getattr(
                    aligned,
                    "size",
                    0,
                ) == 0
            ):

                return self._crop_face(
                    frame,
                    detection,
                )

            return aligned

        except Exception:

            return self._crop_face(
                frame,
                detection,
            )

    def _crop_face(
        self,
        frame: Any,
        detection: dict,
    ) -> Any:

        try:

            import cv2

            x = int(
                detection["x"]
            )

            y = int(
                detection["y"]
            )

            w = int(
                detection["w"]
            )

            h = int(
                detection["h"]
            )

            height, width = (
                frame.shape[:2]
            )

            margin_x = int(
                w * 0.10
            )

            margin_y = int(
                h * 0.10
            )

            left = max(
                0,
                x - margin_x,
            )

            top = max(
                0,
                y - margin_y,
            )

            right = min(
                width,
                x + w + margin_x,
            )

            bottom = min(
                height,
                y + h + margin_y,
            )

            crop = frame[
                top:bottom,
                left:right,
            ]

            if (
                crop is None
                or getattr(
                    crop,
                    "size",
                    0,
                ) == 0
            ):

                return None

            return cv2.resize(
                crop,
                (
                    self.ALIGN_SIZE,
                    self.ALIGN_SIZE,
                ),
                interpolation=
                cv2.INTER_AREA,
            )

        except Exception:

            return None

    # ======================================================
    # FACE EMBEDDING
    # ======================================================

    def _get_embedding(
        self,
        face: Any,
    ) -> Any:

        DeepFace = self._load_deepface()

        if (
            DeepFace is None
            or face is None
        ):
            return None

        try:

            if face.size == 0:
                return None

        except Exception:

            return None

        try:

            import numpy as np

            result = DeepFace.represent(
                img_path=face,
                model_name=self.model_name,
                detector_backend="skip",
                enforce_detection=False,
                align=False,
            )

            if not result:
                return None

            vector = result[0].get(
                "embedding",
                [],
            )

            if not vector:
                return None

            vector = np.asarray(
                vector,
                dtype=np.float32,
            )

            norm = float(
                np.linalg.norm(vector)
            )

            if norm <= 0:
                return None

            return (
                vector / norm
            ).astype(
                np.float32,
                copy=False,
            )

        except Exception as exc:

            print(
                f"⚠️ SFace embedding error: "
                f"{exc}"
            )

            return None

    # ======================================================
    # BUILD GALLERY
    # ======================================================

    def _build_embeddings(
        self,
    ) -> bool:

        import numpy as np

        # --------------------------------------------------
        # STEP 1: cache first.
        # A valid cache means YuNet does NOT need to be initialized here.
        # --------------------------------------------------
        files = self._get_files()
        signature = self._cache_signature()

        with self._lock:
            if (
                self._embeddings_ready
                and self._embedding_signature == signature
            ):
                return self._load_model()

        if self._load_embedding_cache(signature):
            self._set_stage(
                "Loading SFace model..."
            )

            # Cache is already in RAM. The remaining heavy task is only
            # loading SFace for live recognition.
            return self._load_model()

        # --------------------------------------------------
        # STEP 2: no valid cache -> build once.
        # Load SFace first, then YuNet, then build the gallery.
        # --------------------------------------------------
        if not self._load_model():
            return False

        if not files:
            empty = np.empty(
                (0, 0),
                dtype=np.float32,
            )

            self._install_embeddings(
                [],
                empty,
                signature,
            )

            self._set_stage(
                "Face recognition ready"
            )

            return True

        if not self._load_yunet():
            return False

        self._set_stage(
            "Building face embeddings..."
        )

        print(
            "⏳ Building known-face gallery..."
        )

        names: list[str] = []
        vectors: list[Any] = []
        total = len(files)

        for index, path in enumerate(
            files,
            start=1,
        ):
            try:
                image = self._read_image(
                    str(path)
                )

                if image is None:
                    continue

                detections = (
                    self._detect_face_best_scale(
                        image
                    )
                )

                if not detections:
                    print(
                        "⚠️ No face in "
                        f"{path.name}"
                    )
                    continue

                face = self._align_face(
                    image,
                    detections[0],
                )

                vector = self._get_embedding(
                    face
                )

                if vector is None:
                    continue

                names.append(
                    self.clean_name(
                        path.name
                    )
                )

                vectors.append(vector)

                self._set_stage(
                    "Building face embeddings... "
                    f"({index}/{total})"
                )

            except Exception as exc:
                print(
                    f"⚠️ Dataset error "
                    f"{path.name}: {exc}"
                )

        if vectors:
            matrix = np.vstack(
                vectors
            ).astype(
                np.float32,
                copy=False,
            )
        else:
            matrix = np.empty(
                (0, 0),
                dtype=np.float32,
            )

        self._install_embeddings(
            names,
            matrix,
            signature,
        )

        self._save_embedding_cache(
            signature,
            names,
            matrix,
        )

        # Calibration is cheap for a 10-image gallery, but do it here once
        # instead of waiting for the first live recognition frame.
        self._calibrate_threshold()

        self._set_stage(
            "Face recognition ready"
        )

        print(
            "✅ Gallery ready: "
            f"{len(names)} samples."
        )

        return True

    # ======================================================
    # ASYNC PRELOAD / LIFECYCLE
    # ======================================================

    def _run_async_once(self, target, initial_stage: str) -> None:
        with self._lock:
            if (
                self._loading
                and self._prepare_thread is not None
                and self._prepare_thread.is_alive()
            ):
                return

            self._ready_event.clear()
            self._loading = True
            self._loading_error = None
            self._loading_stage = initial_stage

            worker = threading.Thread(
                target=target,
                daemon=True,
                name="face-model-loader",
            )
            self._prepare_thread = worker

        worker.start()

    def _finish_loading(self, success: bool, default_error: str) -> None:
        with self._lock:
            self._loading = False
            if success and self._detector_ready and self._model_ready and self._embeddings_ready:
                self._ready_event.set()
            else:
                self._ready_event.clear()
                if not success and not self._loading_error:
                    self._loading_error = default_error

    def _prepare_base_worker(self) -> None:
        """Preload only DeepFace/SFace and any valid embedding cache.

        YuNet is intentionally NOT initialized here. This keeps process startup
        lightweight and reserves detector initialization for the user's START
        action.
        """
        success = False
        try:
            if not self._load_model():
                return

            signature = self._cache_signature()
            cache_ready = self._load_embedding_cache(signature)

            if cache_ready:
                self._calibrate_threshold()
                success = True
                with self._lock:
                    self._loading_stage = "Face model + gallery ready"
            else:
                # Cache is optional. The gallery can be built after START
                # because that build needs YuNet for face alignment.
                success = True
                with self._lock:
                    self._loading_stage = (
                        "Face model ready - gallery waits for START"
                    )

            # If START arrived while base preload was running, immediately
            # continue in this same worker instead of creating a second loader.
            with self._lock:
                live_requested = self._camera_prepare_requested

            if live_requested:
                success = self._prepare_camera_steps()

        except Exception as exc:
            success = False
            with self._lock:
                self._loading_error = (
                    f"Face recognition preload gagal: {exc}"
                )
            print(f"⚠️ Face recognition preload error: {exc}")
        finally:
            self._finish_loading(
                success,
                "Face recognition gagal disiapkan.",
            )

    def _prepare_full_worker(self) -> None:
        """Load the complete live-camera recognition stack."""
        success = False
        try:
            success = self._prepare_camera_steps()
        except Exception as exc:
            with self._lock:
                self._loading_error = (
                    f"Face recognition preload gagal: {exc}"
                )
            print(f"⚠️ Face recognition preload error: {exc}")
        finally:
            self._finish_loading(
                success,
                "Face recognition gagal disiapkan.",
            )

    def prepare_login_async(self) -> None:
        """Compatibility API for the lightweight process preload."""
        self.prepare_process_async()

    def _prepare_camera_steps(self) -> bool:
        """Load YuNet on START, then build/restore the face gallery."""
        try:
            # SFace is normally already resident from process preload.
            if not self._load_model():
                return False

            signature = self._cache_signature()

            with self._lock:
                cache_ready = (
                    self._embeddings_ready
                    and self._embedding_signature == signature
                )

            if not cache_ready:
                cache_ready = self._load_embedding_cache(signature)

            # YuNet is deliberately the START-time detector initialization.
            self._set_stage("Loading YuNet detector...")
            if not self._load_yunet():
                return False

            if cache_ready:
                self._calibrate_threshold()
            else:
                if not self._build_embeddings():
                    return False

            with self._lock:
                self._loading_stage = "Face recognition ready"
                self._loading_error = None

            return True

        except Exception as exc:
            with self._lock:
                self._loading_error = (
                    f"Camera preparation gagal: {exc}"
                )
            print(f"⚠️ Camera preparation error: {exc}")
            return False

    def _prepare_camera_worker(self) -> None:
        self._prepare_full_worker()

    def prepare_process_async(self) -> None:
        """Preload SFace and cached gallery, but never initialize YuNet."""
        with self._lock:
            if self._model_ready and self._embeddings_ready:
                self._loading_stage = "Face model + gallery ready"
                return

            if (
                self._loading
                and self._prepare_thread is not None
                and self._prepare_thread.is_alive()
            ):
                return

        self._run_async_once(
            self._prepare_base_worker,
            "Preparing SFace + gallery...",
        )

    def prepare_camera_async(self) -> None:
        """Request START-time YuNet and complete live recognition preparation."""
        with self._lock:
            self._camera_prepare_requested = True

            if (
                self._model_ready
                and self._detector_ready
                and self._embeddings_ready
            ):
                self._camera_prepare_requested = False
                self._loading_stage = "Face recognition ready"
                return

            if (
                self._loading
                and self._prepare_thread is not None
                and self._prepare_thread.is_alive()
            ):
                return

        self._run_async_once(
            self._prepare_full_worker,
            "Loading YuNet for START...",
        )

    # Backwards-compatible API aliases.
    def prepare_all_async(self) -> None:
        """Start full live-camera ML preparation in the background."""
        self.prepare_camera_async()

    def prepare_model_async(self) -> None:
        self.prepare_all_async()

    def _prepare_model_worker(self) -> None:
        self.prepare_all_async()

    def prepare_cached_gallery_async(self) -> None:
        with self._lock:
            if self._embeddings_ready or self._loading:
                return

        self._run_async_once(
            self._prepare_cached_gallery_worker,
            "Loading cached embeddings...",
        )

    def _prepare_cached_gallery_worker(self) -> None:
        success = False

        try:
            signature = self._cache_signature()
            success = self._load_embedding_cache(signature)

            if success:
                self._calibrate_threshold()

                with self._lock:
                    self._loading_stage = (
                        "Face recognition ready"
                        if self._model_ready and self._detector_ready
                        else "Embedding cache ready"
                    )
            else:
                # Missing cache is not an error. Full preload can rebuild it.
                success = True

        except Exception as exc:
            with self._lock:
                self._loading_error = (
                    f"Cache preparation gagal: {exc}"
                )

            print(
                f"⚠️ Cache preparation error: {exc}"
            )

        finally:
            self._finish_loading(
                success,
                "Embedding cache gagal dimuat.",
            )

    def prepare_async(self) -> None:
        self.prepare_all_async()

    # READINESS
    # ======================================================

    def is_detector_ready(self) -> bool:

        with self._lock:
            return self._detector_ready

    def is_model_ready(self) -> bool:

        with self._lock:
            return self._model_ready

    def is_embeddings_ready(self) -> bool:

        with self._lock:
            return self._embeddings_ready

    def is_ready(self) -> bool:

        with self._lock:

            return (
                self._detector_ready
                and self._model_ready
                and self._embeddings_ready
            )

    def is_loading(self) -> bool:

        with self._lock:
            return self._loading

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """Wait until the complete ML stack is resident and ready."""
        if self.is_ready():
            return True

        # The event is set only after SFace, embeddings, and YuNet are all ready.
        if timeout is None:
            self._ready_event.wait()
        else:
            self._ready_event.wait(timeout=max(0.0, float(timeout)))

        return self.is_ready()

    # ======================================================
    # MATCHING
    # ======================================================

    def _calibrate_threshold(self) -> None:
        """
        Calibrate the minimum cosine similarity from known_faces.

        Each known image is compared to the other images belonging to
        the same normalized identity name. With 10 Darren images this
        estimates how far genuine Darren-vs-Darren embeddings can vary.
        """
        import numpy as np

        with self._lock:
            names = list(self._embedding_names)
            matrix = self._embedding_matrix

        if (
            matrix is None
            or getattr(matrix, "size", 0) == 0
            or len(names) < 2
        ):
            with self._lock:
                self._calibrated_similarity = (
                    self.DEFAULT_CALIBRATED_SIMILARITY
                )
                self._calibration_samples = 0
            return

        genuine_scores: list[float] = []

        groups: dict[str, list[int]] = {}

        for index, name in enumerate(names):
            groups.setdefault(
                self.clean_name(name),
                [],
            ).append(index)

        for indices in groups.values():

            if len(indices) < 2:
                continue

            for query_index in indices:

                other_indices = [
                    index
                    for index in indices
                    if index != query_index
                ]

                if not other_indices:
                    continue

                query = matrix[query_index]

                scores = np.dot(
                    matrix[other_indices],
                    query,
                )

                # Best genuine match for this sample.
                genuine_scores.append(
                    float(np.max(scores))
                )

        if not genuine_scores:

            calibrated = (
                self.DEFAULT_CALIBRATED_SIMILARITY
            )

        else:

            genuine_scores.sort()

            # Lower genuine tail. This avoids using only the easiest
            # Darren-to-Darren comparisons.
            index = max(
                0,
                int(
                    0.20
                    * (len(genuine_scores) - 1)
                ),
            )

            lower_tail = float(
                genuine_scores[index]
            )

            calibrated = (
                lower_tail
                - self.CALIBRATION_MARGIN
            )

            calibrated = max(
                self.ABSOLUTE_MIN_SIMILARITY,
                min(
                    calibrated,
                    self.MAX_CALIBRATED_SIMILARITY,
                ),
            )

        with self._lock:

            self._calibrated_similarity = float(
                calibrated
            )

            self._calibration_samples = len(
                genuine_scores
            )

        print(
            "✅ Darren/known-face threshold: "
            f"{calibrated:.3f}"
        )

    def _match_identity(
        self,
        query: Any,
        tolerance: float,
    ) -> tuple[str, float, float]:

        import numpy as np

        # Calibrate lazily. This avoids changing the model-loading
        # lifecycle and works with the existing dashboard.
        with self._lock:
            matrix = self._embedding_matrix
            names = list(self._embedding_names)
            calibrated = self._calibrated_similarity

        if calibrated <= 0 or not names:
            self._calibrate_threshold()

            with self._lock:
                calibrated = self._calibrated_similarity

        if (
            matrix is None
            or getattr(matrix, "size", 0) == 0
            or not names
        ):
            return (
                "Unknown",
                0.0,
                0.0,
            )

        # Cosine similarity because stored and query embeddings
        # are normalized.
        scores = np.dot(
            matrix,
            query,
        )

        order = np.argsort(
            scores
        )[::-1]

        top_indices = order[
            : min(
                self.TOP_K_SAMPLES,
                len(order),
            )
        ]

        top_scores = [
            float(scores[index])
            for index in top_indices
        ]

        best_similarity = (
            top_scores[0]
        )

        top_mean = float(
            sum(top_scores)
            / len(top_scores)
        )

        best_name = self.clean_name(
            names[
                int(top_indices[0])
            ]
        )

        # Dashboard passes tolerance as cosine distance.
        tolerance_floor = (
            1.0 - float(tolerance)
        )

        required_similarity = max(
            calibrated,
            tolerance_floor,
            self.ABSOLUTE_MIN_SIMILARITY,
        )

        # Main decision:
        # the live face needs to resemble a genuine gallery sample.
        accepted = (
            best_similarity
            >= required_similarity
        )

        # Do not over-reject a genuine Darren just because
        # every photo is not equally similar. Only reject when
        # the top-3 evidence is clearly inconsistent.
        if (
            accepted
            and len(top_scores) >= 3
            and top_mean
            < max(
                self.ABSOLUTE_MIN_SIMILARITY,
                calibrated - 0.10,
            )
        ):
            accepted = False

        if accepted:

            return (
                best_name,
                best_similarity,
                top_mean,
            )

        return (
            "Unknown",
            best_similarity,
            top_mean,
        )

    def _stabilize_identity(
        self,
        predicted: str,
    ) -> str:

        if (
            predicted
            == self._last_identity
        ):

            self._pending_identity = (
                predicted
            )

            self._pending_identity_count = 0

            return predicted

        if (
            predicted
            == self._pending_identity
        ):

            self._pending_identity_count += 1

        else:

            self._pending_identity = (
                predicted
            )

            self._pending_identity_count = 1

        if (
            self._pending_identity_count
            >= self.IDENTITY_SWITCH_CONFIRMATIONS
        ):

            self._last_identity = (
                predicted
            )

            self._pending_identity_count = 0

        return self._last_identity


    # ======================================================
    # FAST LIVE-CAMERA TRACKING
    # ======================================================

    def _scale_detection_to_frame(
        self,
        detection: dict,
        scale: float,
    ) -> dict:
        """Convert a reduced-frame detection to full-frame coordinates."""
        if scale == 1.0:
            return dict(detection)

        converted = dict(detection)
        converted["x"] = int(round(float(detection["x"]) / scale))
        converted["y"] = int(round(float(detection["y"]) / scale))
        converted["w"] = max(
            1,
            int(round(float(detection["w"]) / scale)),
        )
        converted["h"] = max(
            1,
            int(round(float(detection["h"]) / scale)),
        )
        converted["landmarks"] = [
            (
                float(x) / scale,
                float(y) / scale,
            )
            for x, y in detection.get("landmarks", [])
        ]
        return converted

    def track_face(
        self,
        frame: Any,
    ) -> tuple[dict | None, tuple[int, int, int, int] | None]:
        """
        Fast YuNet-only tracking for every live-camera frame.

        It intentionally does not run SFace. The expensive far-face fallback
        is sampled periodically, while the normal path runs on a reduced frame.
        """
        if frame is None:
            return None, None

        try:
            if frame.size == 0:
                return None, None
        except Exception:
            return None, None

        if not self.is_detector_ready():
            return None, self._get_stable_box(frame.shape)

        try:
            import cv2

            self._fast_track_frame += 1

            height, width = frame.shape[:2]
            scale = self.FAST_DETECT_SCALE

            if scale < 1.0:
                small = cv2.resize(
                    frame,
                    (
                        max(1, int(round(width * scale))),
                        max(1, int(round(height * scale))),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                small = frame

            detections = self._detect_on_image(small)

            # Only sample the full-resolution fallback occasionally.
            if not detections and (
                self._fast_track_frame
                % self.FAST_FALLBACK_INTERVAL
                == 0
            ):
                detections = self._detect_face_best_scale(frame)
                if detections:
                    detection = detections[0]
                    stable = self._smooth_box(
                        detection,
                        frame.shape,
                    )
                    return detection, stable

            if not detections:
                stable = self._get_stable_box(frame.shape)
                return None, stable

            detection = self._scale_detection_to_frame(
                detections[0],
                scale,
            )

            stable = self._smooth_box(
                detection,
                frame.shape,
            )

            return detection, stable

        except Exception as exc:
            print(f"⚠️ Fast face tracking error: {exc}")
            return None, self._get_stable_box(frame.shape)

    def recognize_detection(
        self,
        frame: Any,
        detection: dict | None,
        tolerance: float | None = None,
    ) -> tuple[str, float, float]:
        """
        Recognize an already-detected face with SFace.

        No YuNet detection is repeated here; alignment + embedding + matching
        are all that remain on the background recognition worker.
        """
        if frame is None or detection is None:
            return (
                self._last_identity,
                self._last_similarity,
                self._last_top3_mean,
            )

        with self._lock:
            embeddings_ready = self._embeddings_ready

        if not embeddings_ready:
            return "Preparing...", 0.0, 0.0

        aligned_face = self._align_face(
            frame,
            detection,
        )

        query = self._get_embedding(
            aligned_face
        )

        if query is None:
            return (
                self._last_identity,
                self._last_similarity,
                self._last_top3_mean,
            )

        public_tolerance = (
            self.tolerance
            if tolerance is None
            else float(tolerance)
        )

        predicted, similarity, top_mean = (
            self._match_identity(
                query,
                public_tolerance,
            )
        )

        self._last_similarity = similarity
        self._last_top3_mean = top_mean
        self._last_match_name = predicted

        stable_identity = self._stabilize_identity(
            predicted
        )

        return (
            stable_identity,
            similarity,
            top_mean,
        )

    # ======================================================
    # DETECT + RECOGNIZE ONE FACE
    # ======================================================

    def detect_faces(
        self,
        frame: Any,
        tolerance: float | None = None,
    ) -> tuple[list, list[str], list]:

        if frame is None:
            return [], [], []

        try:

            if frame.size == 0:
                return [], [], []

        except Exception:

            return [], [], []

        public_tolerance = (
            self.tolerance
            if tolerance is None
            else float(tolerance)
        )

        # --------------------------------------------------
        # Detection is independent from embeddings.
        # --------------------------------------------------

        # IMPORTANT:
        # Never initialize YuNet synchronously inside the video callback.
        # The camera START lifecycle owns detector preparation. This keeps
        # the browser/video stream responsive even on slower CPUs.
        if not self.is_detector_ready():
            return [], [], []

        detections = (
            self._detect_face_best_scale(
                frame
            )
        )

        if not detections:

            stable = self._get_stable_box(
                frame.shape
            )

            if stable is None:

                self._reset_tracking_state()

                return [], [], []

            location = (
                stable[1],
                stable[0] + stable[2],
                stable[1] + stable[3],
                stable[0],
            )

            return (
                [location],
                [
                    self._last_identity
                ],
                [None],
            )

        detection = detections[0]

        stable = self._smooth_box(
            detection,
            frame.shape,
        )

        location = (
            stable[1],
            stable[0] + stable[2],
            stable[1] + stable[3],
            stable[0],
        )

        # --------------------------------------------------
        # If gallery is not ready, box still displays.
        # --------------------------------------------------

        with self._lock:

            embeddings_ready = (
                self._embeddings_ready
            )

        if not embeddings_ready:

            # START owns gallery preparation. Never spawn heavy model/gallery
            # loading from the per-frame callback.
            return (
                [location],
                ["Preparing..."],
                [None],
            )

        # --------------------------------------------------
        # SFace recognition.
        # --------------------------------------------------

        aligned_face = (
            self._align_face(
                frame,
                detection,
            )
        )

        query = self._get_embedding(
            aligned_face
        )

        if query is None:

            return (
                [location],
                [
                    self._last_identity
                    if self._last_identity
                    != "Unknown"
                    else "Unknown"
                ],
                [None],
            )

        predicted, similarity, top_mean = (
            self._match_identity(
                query,
                public_tolerance,
            )
        )

        self._last_similarity = (
            similarity
        )

        self._last_top3_mean = (
            top_mean
        )

        self._last_match_name = (
            predicted
        )

        stable_identity = (
            self._stabilize_identity(
                predicted
            )
        )

        return (
            [location],
            [stable_identity],
            [None],
        )

    # ======================================================
    # DIAGNOSTIC
    # ======================================================

    def get_last_match_metrics(
        self,
    ) -> dict:

        return {
            "name":
                self._last_match_name,
            "similarity":
                round(
                    self._last_similarity,
                    4,
                ),
            "top3_mean":
                round(
                    self._last_top3_mean,
                    4,
                ),
            "calibrated_similarity":
                round(
                    self._calibrated_similarity,
                    4,
                ),
            "calibration_samples":
                self._calibration_samples,
            "min_top3_mean":
                self.MIN_TOP3_MEAN_SIMILARITY,
        }

    # ======================================================
    # RELOAD
    # ======================================================

    def reload(
        self,
    ) -> int:

        with self._lock:

            self._index_dataset()

            self._embedding_names = []
            self._embedding_matrix = None

            self._identity_names = []
            self._identity_prototypes = None

            self._embedding_signature = None
            self._embeddings_ready = False
            self._loading_error = None

        try:

            self.embedding_cache_path.unlink(
                missing_ok=True
            )

        except OSError:
            pass

        self._reset_tracking_state()

        count = (
            self.get_dataset_file_count()
        )

        print(
            "✅ Dataset reloaded: "
            f"{count} files"
        )

        # Rebuild the gallery asynchronously so the next START can reuse the
        # resident SFace/YuNet objects and a ready embedding cache.
        try:
            self.prepare_all_async()
        except Exception as exc:
            print(
                f"⚠️ Background reload preload error: {exc}"
            )

        return count

    def clear_embeddings(
        self,
    ) -> None:

        with self._lock:

            self._embedding_names = []
            self._embedding_matrix = None

            self._identity_names = []
            self._identity_prototypes = None

            self._embedding_signature = None
            self._embeddings_ready = False

        self._reset_tracking_state()


# ==========================================================
# SINGLETON
# ==========================================================

_instance: FaceRecognizer | None = None
_instance_lock = threading.Lock()


def get_face_recognizer() -> FaceRecognizer:
    global _instance

    with _instance_lock:
        if _instance is None:
            _instance = FaceRecognizer()

        return _instance


# ==========================================================
# SHARED INSTANCE
# ==========================================================

# Creating the singleton is lightweight. Heavy model initialization is started
# by start_process_preload(); app.py can optionally wait until the stack is READY.
face_recognizer = get_face_recognizer()


# ==========================================================
# PROCESS-LEVEL PRELOAD
# ==========================================================

_process_preload_lock = threading.Lock()
_process_preload_started = False


def start_process_preload() -> None:
    """Start the complete face-model preload once per Python process.

    Startup loads DeepFace/SFace, the embedding gallery, and YuNet in the
    background. The WebRTC webcam prewarm is handled independently by
    dashboard.py and does not run AI inference before START.
    """
    global _process_preload_started

    with _process_preload_lock:
        if _process_preload_started:
            return
        _process_preload_started = True

    try:
        get_face_recognizer().prepare_all_async()
    except Exception:
        with _process_preload_lock:
            _process_preload_started = False
        raise

def wait_for_process_preload(timeout: float | None = None) -> bool:
    """Block until the process-level face-recognition stack is fully ready."""
    return get_face_recognizer().wait_until_ready(timeout=timeout)
