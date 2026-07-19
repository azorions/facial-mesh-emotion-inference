"""MediaPipe structural tracking with asynchronous DeepFace emotion inference.

Design
------
* :class:`FacialAnalyzer` runs MediaPipe Face Mesh synchronously on **every**
  frame — landmark extraction is cheap and must stay smooth.
* :class:`EmotionWorker` is a daemon thread that owns DeepFace. The analyzer
  hands it a face crop every ``emotion_frame_interval`` frames (frame-skip),
  and the worker keeps only the newest pending crop so it can never build up
  a backlog. The heavy CNN therefore never blocks the render loop.
* Both MediaPipe and DeepFace are imported lazily: MediaPipe on
  :meth:`FacialAnalyzer.start`, DeepFace inside the worker thread — so model
  loading happens **asynchronously** while the video pipeline is already
  running, and the unit tests can inject fakes without either library.

Error-handling strategy
-----------------------
1. **Frame tracking dropouts** — when the user steps out of frame, the
   analyzer reports an empty face list plus a :class:`TrackingState`
   (``LOST`` inside a grace window, then ``SEARCHING``). Nothing raises;
   no thread dies.
2. **Extreme lighting / landmark initialization timeouts** — if MediaPipe
   yields no landmarks (or raises) for ``landmark_timeout_frames``
   consecutive frames, the Face Mesh graph is torn down and re-initialized
   to recover from a wedged tracker, with a lighting hint logged once per
   recovery.
3. **First-run DeepFace weight download** — before loading, the worker
   checks ``~/.deepface/weights``; if the weights are absent it prints an
   explicit status update that a one-time download is in progress, then
   reports readiness (or a clear failure) without crashing the app.
"""

from __future__ import annotations

import enum
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Callable

import cv2
import numpy as np

from config.settings import AnalyzerSettings

InferenceFn = Callable[[np.ndarray], dict[str, float]]
"""Maps a BGR face crop to raw per-emotion scores in percent."""

FaceMeshFactory = Callable[[], Any]
"""Builds an object exposing MediaPipe's ``process(rgb)`` / ``close()`` API."""


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
class TrackingState(enum.Enum):
    """Structural-tracking status reported with every frame.

    Attributes:
        SEARCHING: No face has been seen recently; scanning for one.
        TRACKING: At least one face is actively tracked this frame.
        LOST: A tracked face vanished within the dropout grace window.
    """

    SEARCHING = "searching"
    TRACKING = "tracking"
    LOST = "lost"


@dataclass(frozen=True)
class FaceLandmarks:
    """Structural landmarks for one tracked face, in pixel space.

    Attributes:
        points: Array of shape ``(N, 3)`` — pixel ``x``, pixel ``y`` and
            MediaPipe's relative depth ``z`` per landmark (N = 468/478).
        bbox: Padded pixel bounding box ``(x_min, y_min, x_max, y_max)``,
            clamped to the frame bounds.
    """

    points: np.ndarray
    bbox: tuple[int, int, int, int]

    def as_dict(self) -> dict[int, tuple[float, float, float]]:
        """Returns the landmarks as ``{index: (x, y, z)}``."""
        return {i: (float(x), float(y), float(z))
                for i, (x, y, z) in enumerate(self.points)}


@dataclass(frozen=True)
class EmotionPrediction:
    """One smoothed emotion prediction.

    Attributes:
        label: Dominant emotion name (e.g. ``"happy"``).
        confidence: Score of the dominant emotion, in percent [0, 100].
        scores: Per-emotion scores in percent, keyed by emotion name.
        timestamp: ``time.monotonic()`` at which the result was produced.
    """

    label: str
    confidence: float
    scores: dict[str, float]
    timestamp: float


@dataclass(frozen=True)
class FrameAnalysis:
    """Combined per-frame output of :class:`FacialAnalyzer`.

    Attributes:
        faces: All faces tracked in this frame (may be empty).
        emotion: Most recent (possibly slightly stale) emotion prediction,
            or ``None`` if the worker has not produced one yet.
        tracking_state: Structural-tracking status for this frame.
        engine_status: Human-readable status of the emotion engine, for
            the HUD (e.g. ``"downloading weights"``, ``"online"``).
    """

    faces: list[FaceLandmarks] = field(default_factory=list)
    emotion: EmotionPrediction | None = None
    tracking_state: TrackingState = TrackingState.SEARCHING
    engine_status: str = "offline"


# --------------------------------------------------------------------------- #
# Asynchronous emotion worker
# --------------------------------------------------------------------------- #
class EmotionWorker(threading.Thread):
    """Daemon thread running emotion inference off the render loop.

    The producer hands over at most one pending crop; newer submissions
    replace older un-processed ones, so the worker always analyzes the
    freshest face and can never accumulate a backlog.
    """

    _WEIGHTS_DIR: Path = Path.home() / ".deepface" / "weights"

    def __init__(
        self,
        settings: AnalyzerSettings,
        inference_fn: InferenceFn | None = None,
    ) -> None:
        """Initializes the worker (does not start the thread).

        Args:
            settings: Analyzer configuration (smoothing, intervals).
            inference_fn: Optional injected inference callable. When
                ``None`` (production), DeepFace is loaded lazily inside
                the thread; tests inject a lightweight fake here.
        """
        super().__init__(daemon=True, name="EmotionWorker")
        self._settings: AnalyzerSettings = settings
        self._inference_fn: InferenceFn | None = inference_fn
        self._condition: threading.Condition = threading.Condition()
        self._pending: np.ndarray | None = None
        self._latest: EmotionPrediction | None = None
        self._smoothed: dict[str, float] | None = None
        self._stop_event: threading.Event = threading.Event()
        self._status_lock: threading.Lock = threading.Lock()
        self._status: str = "loading model"

    # ------------------------------ public API ------------------------------ #
    @property
    def status(self) -> str:
        """Current engine status string (thread-safe)."""
        with self._status_lock:
            return self._status

    def submit(self, face_crop: np.ndarray) -> None:
        """Queues a face crop for analysis, replacing any pending one.

        Args:
            face_crop: BGR image containing one face.
        """
        with self._condition:
            self._pending = face_crop
            self._condition.notify()

    def latest(self) -> EmotionPrediction | None:
        """Returns the most recent prediction, or ``None`` if none exists."""
        with self._condition:
            return self._latest

    def stop(self) -> None:
        """Signals the thread to exit and waits briefly for it."""
        self._stop_event.set()
        with self._condition:
            self._condition.notify()
        if self.is_alive():
            self.join(timeout=3.0)

    # ------------------------------ thread body ----------------------------- #
    def run(self) -> None:
        """Loads the model (if needed), then serves inference requests."""
        if self._inference_fn is None:
            try:
                self._inference_fn = self._load_deepface()
            except Exception as exc:  # noqa: BLE001 — thread must not die
                self._set_status(f"failed: {exc}")
                print(
                    "[EMOTION] Could not initialize DeepFace. Structural "
                    "tracking continues without emotion inference.\n"
                    f"          Details: {exc}",
                    file=sys.stderr,
                )
                return
        self._set_status("online")

        while not self._stop_event.is_set():
            with self._condition:
                while self._pending is None and not self._stop_event.is_set():
                    self._condition.wait(timeout=0.1)
                crop = self._pending
                self._pending = None
            if crop is None or self._stop_event.is_set():
                continue
            try:
                raw_scores = self._inference_fn(crop)
            except Exception as exc:  # noqa: BLE001 — a bad crop is survivable
                print(f"[EMOTION] Inference failed: {exc}", file=sys.stderr)
                continue
            prediction = self._smooth(raw_scores)
            if prediction is not None:
                with self._condition:
                    self._latest = prediction

    # ------------------------------ internals ------------------------------- #
    def _set_status(self, status: str) -> None:
        """Updates the engine status string (thread-safe)."""
        with self._status_lock:
            self._status = status

    def _load_deepface(self) -> InferenceFn:
        """Imports DeepFace and warms up the emotion model inside the thread.

        On a pristine machine DeepFace downloads its standard model weights
        (VGG-Face family / facial-expression weights) into
        ``~/.deepface/weights`` on first use; the user is told explicitly
        so a silent multi-hundred-MB download is never mistaken for a hang.

        Returns:
            A callable mapping a BGR crop to raw per-emotion scores.

        Raises:
            RuntimeError: If the model cannot be built or downloaded.
        """
        first_run = not any(self._WEIGHTS_DIR.glob("*.h5")) \
            if self._WEIGHTS_DIR.exists() else True
        if first_run:
            self._set_status("downloading weights")
            print(
                "[EMOTION] First run detected — DeepFace is downloading its "
                "standard model weights (VGG-Face / facial-expression) into "
                f"{self._WEIGHTS_DIR}. This is a one-time setup and may take "
                "a few minutes depending on your connection..."
            )
        else:
            print("[EMOTION] Loading DeepFace emotion model...")

        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        try:
            from deepface import DeepFace
        except ImportError as exc:
            raise RuntimeError(
                "DeepFace is not installed. Fix with: "
                "pip install deepface tf-keras"
            ) from exc

        def _infer(face_crop: np.ndarray) -> dict[str, float]:
            analysis = DeepFace.analyze(
                img_path=face_crop,
                actions=("emotion",),
                enforce_detection=False,
                detector_backend="skip",
                silent=True,
            )
            record: dict[str, Any] | None
            if isinstance(analysis, list):
                record = analysis[0] if analysis else None
            else:
                record = analysis
            if not record or "emotion" not in record:
                return {}
            return {str(k): float(v) for k, v in record["emotion"].items()}

        # Warm-up on a dummy crop: triggers the weight download / graph
        # build here, surfacing network or disk failures immediately.
        try:
            _infer(np.zeros((96, 96, 3), dtype=np.uint8))
        except Exception as exc:
            raise RuntimeError(
                "DeepFace model initialization failed. First run requires "
                "internet access to download weights into "
                f"{self._WEIGHTS_DIR}. If a previous download was "
                "interrupted, delete that folder and retry. "
                f"Original error: {exc}"
            ) from exc

        print("[EMOTION] DeepFace emotion model ready.")
        return _infer

    def _smooth(self, raw_scores: dict[str, float]) -> EmotionPrediction | None:
        """Applies exponential smoothing and picks the dominant emotion.

        Args:
            raw_scores: Fresh per-emotion scores in percent.

        Returns:
            The smoothed prediction, or ``None`` for empty score sets.
        """
        if not raw_scores:
            return None
        alpha = self._settings.emotion_smoothing
        if self._smoothed is None:
            self._smoothed = dict(raw_scores)
        else:
            self._smoothed = {
                name: alpha * raw_scores.get(name, 0.0)
                + (1.0 - alpha) * previous
                for name, previous in self._smoothed.items()
            }
        label, confidence = max(
            self._smoothed.items(), key=lambda item: item[1]
        )
        return EmotionPrediction(
            label=label,
            confidence=confidence,
            scores=dict(self._smoothed),
            timestamp=time.monotonic(),
        )


# --------------------------------------------------------------------------- #
# Facial analyzer (structure every frame, emotion every N frames)
# --------------------------------------------------------------------------- #
class FacialAnalyzer:
    """Owns the MediaPipe Face Mesh and the asynchronous emotion engine.

    Example:
        >>> with FacialAnalyzer(AnalyzerSettings()) as analyzer:
        ...     analysis = analyzer.process(frame_bgr)
    """

    def __init__(
        self,
        settings: AnalyzerSettings,
        face_mesh_factory: FaceMeshFactory | None = None,
        inference_fn: InferenceFn | None = None,
    ) -> None:
        """Stores configuration; resources are created in :meth:`start`.

        Args:
            settings: Analyzer configuration.
            face_mesh_factory: Optional injected Face Mesh builder (tests);
                ``None`` loads the real MediaPipe graph lazily.
            inference_fn: Optional injected emotion inference callable
                (tests); ``None`` loads DeepFace in the worker thread.
        """
        self._settings: AnalyzerSettings = settings
        self._face_mesh_factory: FaceMeshFactory = (
            face_mesh_factory or self._default_face_mesh_factory
        )
        self._inference_fn: InferenceFn | None = inference_fn
        self._face_mesh: Any = None
        self._worker: EmotionWorker | None = None
        self._frame_counter: int = 0
        self._missed_frames: int = 0
        self._had_face: bool = False

    # ------------------------------ lifecycle ------------------------------- #
    def start(self) -> "FacialAnalyzer":
        """Initializes Face Mesh and launches the emotion worker thread.

        The worker loads DeepFace *asynchronously*: structural tracking is
        live immediately while the CNN spins up in the background.

        Returns:
            This ``FacialAnalyzer`` instance.
        """
        self._face_mesh = self._face_mesh_factory()
        print("[ANALYZER] Face Mesh initialized.")
        self._worker = EmotionWorker(self._settings, self._inference_fn)
        self._worker.start()
        return self

    def close(self) -> None:
        """Stops the worker thread and releases the Face Mesh graph."""
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
        print("[ANALYZER] Shut down.")

    def __enter__(self) -> "FacialAnalyzer":
        """Starts the analyzer when entering a ``with`` block."""
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Releases all resources when leaving a ``with`` block."""
        self.close()

    # ------------------------------ per frame ------------------------------- #
    def process(self, frame_bgr: np.ndarray) -> FrameAnalysis:
        """Extracts structural landmarks and orchestrates emotion inference.

        Runs MediaPipe on every call; forwards a face crop to the emotion
        worker every ``emotion_frame_interval`` calls. Never raises for
        tracking dropouts or Face Mesh hiccups — those are folded into the
        returned :class:`FrameAnalysis`.

        Args:
            frame_bgr: The current BGR video frame.

        Returns:
            The combined structural + emotional analysis of this frame.

        Raises:
            RuntimeError: If called before :meth:`start`.
        """
        if self._face_mesh is None or self._worker is None:
            raise RuntimeError(
                "FacialAnalyzer.process() called before start()."
            )
        self._frame_counter += 1

        faces = self._extract_faces(frame_bgr)
        state = self._update_tracking_state(bool(faces))

        if faces and (
            self._frame_counter % self._settings.emotion_frame_interval == 0
        ):
            self._submit_primary_face(frame_bgr, faces[0])

        return FrameAnalysis(
            faces=faces,
            emotion=self._worker.latest(),
            tracking_state=state,
            engine_status=self._worker.status,
        )

    # ------------------------------ internals ------------------------------- #
    def _default_face_mesh_factory(self) -> Any:
        """Builds the real MediaPipe Face Mesh graph (lazy import)."""
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed. Fix with: pip install mediapipe"
            ) from exc
        return mp.solutions.face_mesh.FaceMesh(
            max_num_faces=self._settings.max_num_faces,
            refine_landmarks=self._settings.refine_landmarks,
            min_detection_confidence=self._settings.min_detection_confidence,
            min_tracking_confidence=self._settings.min_tracking_confidence,
        )

    def _extract_faces(self, frame_bgr: np.ndarray) -> list[FaceLandmarks]:
        """Runs Face Mesh on one frame, intercepting graph failures.

        Args:
            frame_bgr: The current BGR video frame.

        Returns:
            All tracked faces (possibly empty — never raises for tracking
            or lighting problems).
        """
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            output = self._face_mesh.process(frame_rgb)
        except Exception as exc:  # noqa: BLE001 — graph hiccup, not fatal
            print(f"[ANALYZER] Face Mesh error intercepted: {exc}",
                  file=sys.stderr)
            return []

        landmark_lists = getattr(output, "multi_face_landmarks", None)
        if not landmark_lists:
            return []

        height, width = frame_bgr.shape[:2]
        faces: list[FaceLandmarks] = []
        for landmark_list in landmark_lists:
            points = np.array(
                [(point.x * width, point.y * height, point.z)
                 for point in landmark_list.landmark],
                dtype=np.float32,
            )
            faces.append(
                FaceLandmarks(
                    points=points,
                    bbox=self._padded_bbox(points, width, height),
                )
            )
        return faces

    def _update_tracking_state(self, face_present: bool) -> TrackingState:
        """Advances the dropout / lighting-timeout state machine.

        Args:
            face_present: Whether this frame contained any landmarks.

        Returns:
            The tracking state to report for this frame.
        """
        if face_present:
            self._missed_frames = 0
            self._had_face = True
            return TrackingState.TRACKING

        self._missed_frames += 1

        # Lighting-timeout recovery: a long landmark drought can mean the
        # tracker is wedged by extreme lighting — rebuild the graph.
        if self._missed_frames >= self._settings.landmark_timeout_frames:
            print(
                "[ANALYZER] No landmarks for "
                f"{self._missed_frames} frames — re-initializing Face Mesh. "
                "If this persists, adjust lighting (avoid strong backlight "
                "or near-darkness)."
            )
            try:
                self._face_mesh.close()
            except Exception:  # noqa: BLE001 — closing a wedged graph
                pass
            self._face_mesh = self._face_mesh_factory()
            self._missed_frames = 0
            self._had_face = False
            return TrackingState.SEARCHING

        if self._had_face and (
            self._missed_frames <= self._settings.dropout_grace_frames
        ):
            return TrackingState.LOST
        return TrackingState.SEARCHING

    def _submit_primary_face(
        self, frame_bgr: np.ndarray, face: FaceLandmarks
    ) -> None:
        """Crops the primary face and hands it to the emotion worker.

        Args:
            frame_bgr: The full BGR frame.
            face: The face observation to crop.
        """
        assert self._worker is not None
        x_min, y_min, x_max, y_max = face.bbox
        if (x_max - x_min) < self._settings.min_crop_size:
            return
        if (y_max - y_min) < self._settings.min_crop_size:
            return
        crop = frame_bgr[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return
        self._worker.submit(np.ascontiguousarray(crop))

    def _padded_bbox(
        self, points: np.ndarray, width: int, height: int
    ) -> tuple[int, int, int, int]:
        """Computes a padded pixel bounding box around the landmarks.

        Args:
            points: Pixel-space landmark array of shape ``(N, 3)``.
            width: Frame width in pixels.
            height: Frame height in pixels.

        Returns:
            ``(x_min, y_min, x_max, y_max)`` clamped to the frame bounds.
        """
        x_min = float(points[:, 0].min())
        x_max = float(points[:, 0].max())
        y_min = float(points[:, 1].min())
        y_max = float(points[:, 1].max())
        pad_x = (x_max - x_min) * self._settings.face_crop_padding
        pad_y = (y_max - y_min) * self._settings.face_crop_padding
        return (
            max(int(x_min - pad_x), 0),
            max(int(y_min - pad_y), 0),
            min(int(x_max + pad_x), width - 1),
            min(int(y_max + pad_y), height - 1),
        )
