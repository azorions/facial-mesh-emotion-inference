#!/usr/bin/env python3
"""Real-time facial-structure mapping with asynchronous emotion inference.

Pipeline
--------
1. ``VideoSource``    – context-managed webcam capture (OpenCV).
2. ``FacialAnalyzer`` – MediaPipe Face Mesh runs on *every* frame for smooth
   structural tracking, while DeepFace emotion inference runs on a background
   worker thread fed with a cropped face every N frames, so the heavy CNN
   never blocks the render loop.
3. ``UIOverlay``      – draws the mesh tesselation/contours, bounding box,
   dominant emotion, per-emotion probability panel, and FPS counter.

Controls
--------
``q`` / ``ESC`` quit · ``m`` toggle mesh · ``c`` toggle contours ·
``p`` toggle probability panel

Run:
    python facial_mesh_emotion.py
"""

from __future__ import annotations

import os
import sys
import time
import threading
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, NoReturn

# Silence TensorFlow's C++ banner before DeepFace pulls it in.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _fail_import(package: str, pip_hint: str, error: ImportError) -> NoReturn:
    """Exit with an actionable message when a dependency is missing.

    Args:
        package: Human-readable package name that failed to import.
        pip_hint: Exact ``pip install`` argument(s) that fix the problem.
        error: The original :class:`ImportError`.
    """
    print(
        f"[FATAL] Missing dependency '{package}'.\n"
        f"        Fix with:  pip install {pip_hint}\n"
        f"        Details:   {error}",
        file=sys.stderr,
    )
    sys.exit(1)


try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment guard
    _fail_import("numpy", "numpy", exc)

try:
    import cv2
except ImportError as exc:  # pragma: no cover - environment guard
    _fail_import("OpenCV", "opencv-python", exc)

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - environment guard
    _fail_import("MediaPipe", "mediapipe", exc)

try:
    from deepface import DeepFace
except ImportError as exc:  # pragma: no cover - environment guard
    _fail_import("DeepFace", "deepface tf-keras", exc)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AppConfig:
    """Central configuration for the whole application.

    Attributes:
        camera_index: OpenCV device index of the webcam.
        frame_width: Requested capture width in pixels.
        frame_height: Requested capture height in pixels.
        flip_horizontal: Mirror the feed so it behaves like a mirror.
        max_num_faces: Maximum simultaneous faces tracked by MediaPipe.
        refine_landmarks: Enable iris refinement (478 landmarks vs 468).
        min_detection_confidence: MediaPipe face-detection threshold [0, 1].
        min_tracking_confidence: MediaPipe landmark-tracking threshold [0, 1].
        emotion_frame_interval: Submit a face crop for emotion inference
            every N rendered frames (frame-skipping strategy).
        emotion_smoothing: EMA weight for new emotion scores [0, 1];
            higher reacts faster, lower is more stable.
        face_crop_padding: Extra margin around the landmark bounding box,
            as a fraction of the box size, before cropping for DeepFace.
        min_crop_size: Minimum crop side length (px) worth analyzing.
        window_name: Title of the OpenCV preview window.
        draw_tesselation: Draw the full triangular mesh at startup.
        draw_contours: Draw eye/lip/oval contour lines at startup.
        draw_irises: Draw iris rings at startup (needs refine_landmarks).
        show_probability_panel: Show the per-emotion bar panel at startup.
        max_consecutive_read_failures: Grabs allowed to fail before aborting.
    """

    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    flip_horizontal: bool = True

    max_num_faces: int = 1
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    emotion_frame_interval: int = 15
    emotion_smoothing: float = 0.5
    face_crop_padding: float = 0.25
    min_crop_size: int = 48

    window_name: str = "Facial Mesh + Emotion Inference"
    draw_tesselation: bool = True
    draw_contours: bool = True
    draw_irises: bool = True
    show_probability_panel: bool = True

    max_consecutive_read_failures: int = 30


EMOTION_ORDER: tuple[str, ...] = (
    "happy", "neutral", "surprise", "sad", "angry", "fear", "disgust",
)

EMOTION_COLORS: dict[str, tuple[int, int, int]] = {
    "happy": (80, 200, 80),
    "neutral": (200, 200, 200),
    "surprise": (0, 200, 255),
    "sad": (200, 120, 60),
    "angry": (60, 60, 230),
    "fear": (200, 60, 200),
    "disgust": (60, 180, 180),
}


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EmotionResult:
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
class FaceObservation:
    """A single tracked face within one frame.

    Attributes:
        landmarks: MediaPipe ``NormalizedLandmarkList`` for this face.
        bbox: Pixel-space bounding box ``(x_min, y_min, x_max, y_max)``
            derived from the landmarks (padding already applied).
    """

    landmarks: Any
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class AnalysisResult:
    """Combined per-frame output of :class:`FacialAnalyzer`.

    Attributes:
        faces: All faces tracked in the frame (may be empty).
        emotion: Most recent emotion prediction, or ``None`` if the worker
            has not produced one yet.
    """

    faces: list[FaceObservation] = field(default_factory=list)
    emotion: EmotionResult | None = None


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
class FpsTracker:
    """Exponentially smoothed frames-per-second estimator."""

    def __init__(self, smoothing: float = 0.9) -> None:
        """Initializes the tracker.

        Args:
            smoothing: EMA weight of the previous FPS estimate [0, 1].
        """
        self._smoothing: float = smoothing
        self._last: float = time.perf_counter()
        self._fps: float = 0.0

    def tick(self) -> float:
        """Registers one rendered frame.

        Returns:
            The current smoothed FPS estimate.
        """
        now = time.perf_counter()
        delta = now - self._last
        self._last = now
        if delta <= 0.0:
            return self._fps
        instantaneous = 1.0 / delta
        if self._fps == 0.0:
            self._fps = instantaneous
        else:
            self._fps = (
                self._smoothing * self._fps
                + (1.0 - self._smoothing) * instantaneous
            )
        return self._fps


# --------------------------------------------------------------------------- #
# Video capture
# --------------------------------------------------------------------------- #
class VideoSource:
    """Context-managed webcam capture with resolution negotiation.

    Example:
        >>> with VideoSource(AppConfig()) as source:
        ...     frame = source.read()
    """

    def __init__(self, config: AppConfig) -> None:
        """Stores configuration; the device is opened in :meth:`__enter__`.

        Args:
            config: Application configuration.
        """
        self._config: AppConfig = config
        self._capture: cv2.VideoCapture | None = None

    def __enter__(self) -> "VideoSource":
        """Opens the camera and applies capture settings.

        Returns:
            This ``VideoSource`` instance, ready for :meth:`read`.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        capture = cv2.VideoCapture(self._config.camera_index, backend)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Could not open camera index {self._config.camera_index}. "
                "Check that a webcam is connected, not in use by another "
                "application, and that the OS camera-privacy setting allows "
                "desktop apps to access it."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.frame_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.frame_height)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._capture = capture

        actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Camera opened at {actual_w}x{actual_h}.")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Releases the camera device."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        print("[INFO] Camera released.")

    def read(self) -> np.ndarray | None:
        """Grabs one frame from the camera.

        Returns:
            The BGR frame (mirrored if configured), or ``None`` if the grab
            failed.

        Raises:
            RuntimeError: If called outside the context-manager block.
        """
        if self._capture is None:
            raise RuntimeError("VideoSource.read() called before __enter__.")
        success, frame = self._capture.read()
        if not success or frame is None:
            return None
        if self._config.flip_horizontal:
            frame = cv2.flip(frame, 1)
        return frame


# --------------------------------------------------------------------------- #
# Asynchronous emotion inference
# --------------------------------------------------------------------------- #
class _EmotionWorker(threading.Thread):
    """Daemon thread running DeepFace emotion inference off the render loop.

    The producer (main loop) hands over at most one pending crop; newer
    submissions replace older un-processed ones so the worker always analyzes
    the freshest face and can never build up a backlog.
    """

    def __init__(self, config: AppConfig) -> None:
        """Initializes the worker (does not start the thread).

        Args:
            config: Application configuration.
        """
        super().__init__(daemon=True, name="EmotionWorker")
        self._config: AppConfig = config
        self._condition = threading.Condition()
        self._pending: np.ndarray | None = None
        self._latest: EmotionResult | None = None
        self._smoothed_scores: dict[str, float] | None = None
        self._stop_event = threading.Event()

    def warm_up(self) -> None:
        """Builds the emotion model and triggers the weight download.

        DeepFace lazily downloads ``facial_expression_model_weights.h5`` to
        ``~/.deepface/weights`` on first use; doing it here surfaces network
        or disk failures before the video loop starts.

        Raises:
            RuntimeError: If the model cannot be built or weights cannot be
                downloaded.
        """
        dummy = np.zeros((96, 96, 3), dtype=np.uint8)
        try:
            DeepFace.analyze(
                img_path=dummy,
                actions=("emotion",),
                enforce_detection=False,
                detector_backend="skip",
                silent=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize the DeepFace emotion model. First run "
                "requires internet access to download weights into "
                "~/.deepface/weights. If a previous download was interrupted, "
                "delete that folder and retry. Original error: "
                f"{exc}"
            ) from exc
        print("[INFO] DeepFace emotion model ready.")

    def submit(self, face_crop: np.ndarray) -> None:
        """Queues a face crop for analysis, replacing any pending one.

        Args:
            face_crop: BGR image containing one face.
        """
        with self._condition:
            self._pending = face_crop
            self._condition.notify()

    def latest(self) -> EmotionResult | None:
        """Returns the most recent prediction, or ``None`` if none exists."""
        with self._condition:
            return self._latest

    def stop(self) -> None:
        """Signals the thread to exit and waits briefly for it."""
        self._stop_event.set()
        with self._condition:
            self._condition.notify()
        if self.is_alive():
            self.join(timeout=2.0)

    def run(self) -> None:
        """Thread body: waits for crops and runs inference on each."""
        while not self._stop_event.is_set():
            with self._condition:
                while self._pending is None and not self._stop_event.is_set():
                    self._condition.wait(timeout=0.1)
                crop = self._pending
                self._pending = None
            if crop is None or self._stop_event.is_set():
                continue
            try:
                result = self._analyze(crop)
            except Exception as exc:
                print(f"[WARN] Emotion inference failed: {exc}", file=sys.stderr)
                continue
            if result is not None:
                with self._condition:
                    self._latest = result

    def _analyze(self, face_crop: np.ndarray) -> EmotionResult | None:
        """Runs DeepFace on one crop and smooths the score distribution.

        Args:
            face_crop: BGR image containing one face.

        Returns:
            The smoothed :class:`EmotionResult`, or ``None`` if DeepFace
            returned nothing usable.
        """
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
            return None

        raw_scores = {str(k): float(v) for k, v in record["emotion"].items()}

        alpha = self._config.emotion_smoothing
        if self._smoothed_scores is None:
            self._smoothed_scores = raw_scores
        else:
            self._smoothed_scores = {
                name: alpha * raw_scores.get(name, 0.0)
                + (1.0 - alpha) * previous
                for name, previous in self._smoothed_scores.items()
            }

        label, confidence = max(
            self._smoothed_scores.items(), key=lambda item: item[1]
        )
        return EmotionResult(
            label=label,
            confidence=confidence,
            scores=dict(self._smoothed_scores),
            timestamp=time.monotonic(),
        )


# --------------------------------------------------------------------------- #
# Facial analysis (structure + emotion orchestration)
# --------------------------------------------------------------------------- #
class FacialAnalyzer:
    """Owns the MediaPipe Face Mesh and the asynchronous emotion engine.

    MediaPipe runs synchronously on every frame (it is cheap and must be
    smooth); DeepFace runs on :class:`_EmotionWorker` fed every
    ``emotion_frame_interval`` frames.
    """

    def __init__(self, config: AppConfig) -> None:
        """Stores configuration; resources are created in :meth:`__enter__`.

        Args:
            config: Application configuration.
        """
        self._config: AppConfig = config
        self._face_mesh: Any = None
        self._worker: _EmotionWorker | None = None
        self._frame_counter: int = 0

    def __enter__(self) -> "FacialAnalyzer":
        """Initializes MediaPipe and starts the emotion worker.

        Returns:
            This ``FacialAnalyzer`` instance.

        Raises:
            RuntimeError: If the emotion model cannot be initialized.
        """
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=self._config.max_num_faces,
            refine_landmarks=self._config.refine_landmarks,
            min_detection_confidence=self._config.min_detection_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
        )
        print("[INFO] MediaPipe Face Mesh initialized.")

        worker = _EmotionWorker(self._config)
        worker.warm_up()
        worker.start()
        self._worker = worker
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Stops the worker thread and releases MediaPipe resources."""
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
        print("[INFO] FacialAnalyzer shut down.")

    def analyze(self, frame_bgr: np.ndarray) -> AnalysisResult:
        """Tracks facial structure and orchestrates emotion inference.

        Args:
            frame_bgr: The current BGR video frame.

        Returns:
            An :class:`AnalysisResult` with all tracked faces and the most
            recent (possibly slightly stale) emotion prediction.

        Raises:
            RuntimeError: If called outside the context-manager block.
        """
        if self._face_mesh is None or self._worker is None:
            raise RuntimeError(
                "FacialAnalyzer.analyze() called before __enter__."
            )
        self._frame_counter += 1

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        mesh_output = self._face_mesh.process(frame_rgb)

        faces: list[FaceObservation] = []
        if mesh_output.multi_face_landmarks:
            for landmarks in mesh_output.multi_face_landmarks:
                bbox = self._landmark_bbox(landmarks, frame_bgr.shape)
                faces.append(FaceObservation(landmarks=landmarks, bbox=bbox))

        if faces and self._frame_counter % self._config.emotion_frame_interval == 0:
            self._submit_primary_face(frame_bgr, faces[0])

        return AnalysisResult(faces=faces, emotion=self._worker.latest())

    def _submit_primary_face(
        self, frame_bgr: np.ndarray, face: FaceObservation
    ) -> None:
        """Crops the primary face and hands it to the emotion worker.

        Args:
            frame_bgr: The full BGR frame.
            face: The face observation to crop.
        """
        assert self._worker is not None
        x_min, y_min, x_max, y_max = face.bbox
        if (x_max - x_min) < self._config.min_crop_size:
            return
        if (y_max - y_min) < self._config.min_crop_size:
            return
        crop = frame_bgr[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return
        self._worker.submit(np.ascontiguousarray(crop))

    def _landmark_bbox(
        self, landmarks: Any, frame_shape: tuple[int, ...]
    ) -> tuple[int, int, int, int]:
        """Computes a padded pixel bounding box around the landmarks.

        Args:
            landmarks: MediaPipe ``NormalizedLandmarkList`` for one face.
            frame_shape: ``frame.shape`` of the current frame.

        Returns:
            ``(x_min, y_min, x_max, y_max)`` clamped to the frame bounds.
        """
        height, width = frame_shape[:2]
        xs = [point.x for point in landmarks.landmark]
        ys = [point.y for point in landmarks.landmark]
        x_min = min(xs) * width
        x_max = max(xs) * width
        y_min = min(ys) * height
        y_max = max(ys) * height

        pad_x = (x_max - x_min) * self._config.face_crop_padding
        pad_y = (y_max - y_min) * self._config.face_crop_padding
        return (
            max(int(x_min - pad_x), 0),
            max(int(y_min - pad_y), 0),
            min(int(x_max + pad_x), width - 1),
            min(int(y_max + pad_y), height - 1),
        )


# --------------------------------------------------------------------------- #
# UI rendering
# --------------------------------------------------------------------------- #
class UIOverlay:
    """Draws the facial mesh, emotion readout, and HUD onto frames.

    The ``show_*`` attributes are mutable at runtime so the main loop can
    bind them to keyboard toggles.
    """

    _FONT: int = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, config: AppConfig) -> None:
        """Initializes drawing utilities and runtime toggles.

        Args:
            config: Application configuration (supplies toggle defaults).
        """
        self._config: AppConfig = config
        self._drawer = mp.solutions.drawing_utils
        self._styles = mp.solutions.drawing_styles
        self._mesh_module = mp.solutions.face_mesh

        self.show_tesselation: bool = config.draw_tesselation
        self.show_contours: bool = config.draw_contours
        self.show_irises: bool = config.draw_irises
        self.show_panel: bool = config.show_probability_panel

    def render(
        self, frame: np.ndarray, result: AnalysisResult, fps: float
    ) -> np.ndarray:
        """Draws all overlays for one frame (in place).

        Args:
            frame: The BGR frame to annotate.
            result: Structural and emotional analysis of this frame.
            fps: Current smoothed frames-per-second estimate.

        Returns:
            The same ``frame`` array, annotated.
        """
        for face in result.faces:
            self._draw_mesh(frame, face)
            self._draw_bbox(frame, face, result.emotion)

        if not result.faces:
            self._draw_text(frame, "No face detected", (10, 60), 0.8,
                            (60, 60, 230))
        elif result.emotion is None:
            self._draw_text(frame, "Analyzing emotion...", (10, 60), 0.8,
                            (200, 200, 200))

        if self.show_panel and result.emotion is not None:
            self._draw_probability_panel(frame, result.emotion)

        self._draw_text(frame, f"FPS: {fps:5.1f}", (10, 30), 0.7,
                        (255, 255, 255))
        self._draw_text(
            frame,
            "q:quit  m:mesh  c:contours  p:panel",
            (10, frame.shape[0] - 12),
            0.5,
            (180, 180, 180),
        )
        return frame

    def _draw_mesh(self, frame: np.ndarray, face: FaceObservation) -> None:
        """Draws the tesselation, contour, and iris geometry for one face.

        Args:
            frame: The BGR frame to annotate.
            face: The face whose landmarks are drawn.
        """
        if self.show_tesselation:
            self._drawer.draw_landmarks(
                image=frame,
                landmark_list=face.landmarks,
                connections=self._mesh_module.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=(
                    self._styles.get_default_face_mesh_tesselation_style()
                ),
            )
        if self.show_contours:
            self._drawer.draw_landmarks(
                image=frame,
                landmark_list=face.landmarks,
                connections=self._mesh_module.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=(
                    self._styles.get_default_face_mesh_contours_style()
                ),
            )
        if self.show_irises and self._config.refine_landmarks:
            self._drawer.draw_landmarks(
                image=frame,
                landmark_list=face.landmarks,
                connections=self._mesh_module.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=(
                    self._styles.get_default_face_mesh_iris_connections_style()
                ),
            )

    def _draw_bbox(
        self,
        frame: np.ndarray,
        face: FaceObservation,
        emotion: EmotionResult | None,
    ) -> None:
        """Draws the face bounding box, colored and labeled by emotion.

        Args:
            frame: The BGR frame to annotate.
            face: The face whose box is drawn.
            emotion: Latest emotion prediction, if any.
        """
        x_min, y_min, x_max, y_max = face.bbox
        color = (255, 255, 255)
        label = ""
        if emotion is not None:
            color = EMOTION_COLORS.get(emotion.label, (255, 255, 255))
            label = f"{emotion.label.upper()}  {emotion.confidence:4.1f}%"
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        if label:
            text_y = y_min - 12 if y_min - 12 > 20 else y_max + 28
            self._draw_text(frame, label, (x_min, text_y), 0.8, color)

    def _draw_probability_panel(
        self, frame: np.ndarray, emotion: EmotionResult
    ) -> None:
        """Draws horizontal bars for every emotion class.

        Args:
            frame: The BGR frame to annotate.
            emotion: The prediction whose score distribution is shown.
        """
        panel_x, panel_y = 10, 80
        row_height, bar_max = 24, 150
        label_width = 90
        panel_w = label_width + bar_max + 60
        panel_h = row_height * len(EMOTION_ORDER) + 12

        y_end = min(panel_y + panel_h, frame.shape[0])
        x_end = min(panel_x + panel_w, frame.shape[1])
        roi = frame[panel_y:y_end, panel_x:x_end]
        frame[panel_y:y_end, panel_x:x_end] = (roi * 0.35).astype(np.uint8)

        for row, name in enumerate(EMOTION_ORDER):
            score = emotion.scores.get(name, 0.0)
            base_y = panel_y + 10 + row * row_height
            color = EMOTION_COLORS.get(name, (255, 255, 255))
            self._draw_text(frame, name, (panel_x + 8, base_y + 12), 0.5,
                            color)
            bar_x = panel_x + label_width
            bar_len = int(bar_max * max(0.0, min(score, 100.0)) / 100.0)
            cv2.rectangle(
                frame,
                (bar_x, base_y + 2),
                (bar_x + bar_max, base_y + 14),
                (80, 80, 80),
                1,
            )
            if bar_len > 0:
                cv2.rectangle(
                    frame,
                    (bar_x, base_y + 2),
                    (bar_x + bar_len, base_y + 14),
                    color,
                    -1,
                )
            self._draw_text(
                frame,
                f"{score:4.1f}",
                (bar_x + bar_max + 8, base_y + 13),
                0.45,
                (220, 220, 220),
            )

    def _draw_text(
        self,
        frame: np.ndarray,
        text: str,
        origin: tuple[int, int],
        scale: float,
        color: tuple[int, int, int],
    ) -> None:
        """Draws outlined text for legibility on any background.

        Args:
            frame: The BGR frame to annotate.
            text: The string to draw.
            origin: Bottom-left corner of the text in pixels.
            scale: OpenCV font scale.
            color: BGR text color.
        """
        cv2.putText(frame, text, origin, self._FONT, scale, (0, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(frame, text, origin, self._FONT, scale, color, 1,
                    cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Application entry point
# --------------------------------------------------------------------------- #
def _handle_key(key: int, overlay: UIOverlay) -> bool:
    """Processes one keypress.

    Args:
        key: Masked result of ``cv2.waitKey``.
        overlay: The overlay whose toggles may be flipped.

    Returns:
        ``True`` if the application should keep running.
    """
    if key in (ord("q"), 27):  # 27 = ESC
        return False
    if key == ord("m"):
        overlay.show_tesselation = not overlay.show_tesselation
    elif key == ord("c"):
        overlay.show_contours = not overlay.show_contours
    elif key == ord("p"):
        overlay.show_panel = not overlay.show_panel
    return True


def main() -> int:
    """Runs the capture → analyze → render loop.

    Returns:
        Process exit code (0 on clean shutdown, 1 on fatal error).
    """
    config = AppConfig()
    overlay = UIOverlay(config)
    fps_tracker = FpsTracker()
    read_failures = 0

    try:
        with VideoSource(config) as source, FacialAnalyzer(config) as analyzer:
            print("[INFO] Running. Press 'q' or ESC in the window to quit.")
            while True:
                frame = source.read()
                if frame is None:
                    read_failures += 1
                    if read_failures >= config.max_consecutive_read_failures:
                        raise RuntimeError(
                            "Camera stopped delivering frames "
                            f"({read_failures} consecutive failures)."
                        )
                    continue
                read_failures = 0

                result = analyzer.analyze(frame)
                fps = fps_tracker.tick()
                overlay.render(frame, result, fps)

                cv2.imshow(config.window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if not _handle_key(key, overlay):
                    break
                if (
                    cv2.getWindowProperty(
                        config.window_name, cv2.WND_PROP_VISIBLE
                    )
                    < 1
                ):
                    break
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except RuntimeError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()

    print("[INFO] Clean shutdown complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
