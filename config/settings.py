"""Configuration dataclasses and constants for the facial mesh emotion tracker.

Every tunable of the application lives here, grouped into small frozen
dataclasses so each layer (video, analyzer, ui) receives only the settings
it needs. :class:`AppSettings` composes them into a single root object.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CameraSettings:
    """Webcam capture configuration.

    Attributes:
        camera_index: OpenCV device index of the webcam.
        frame_width: Requested capture width in pixels.
        frame_height: Requested capture height in pixels.
        flip_horizontal: Mirror the feed so it behaves like a mirror.
        queue_size: Depth of the frame buffer between the capture thread
            and the consumer. ``1`` keeps only the freshest frame, which
            eliminates hardware ingestion lag entirely.
        read_timeout: Seconds the consumer waits for a frame before
            treating the read as a miss.
        max_consecutive_failures: Hardware grabs allowed to fail in a row
            before the capture thread declares the device dead.
    """

    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    flip_horizontal: bool = True
    queue_size: int = 1
    read_timeout: float = 1.0
    max_consecutive_failures: int = 30


@dataclass(frozen=True)
class AnalyzerSettings:
    """MediaPipe Face Mesh and DeepFace emotion-engine configuration.

    Attributes:
        max_num_faces: Maximum simultaneous faces tracked by MediaPipe.
        refine_landmarks: Enable iris refinement (478 landmarks vs 468).
        min_detection_confidence: MediaPipe face-detection threshold [0, 1].
        min_tracking_confidence: MediaPipe landmark-tracking threshold [0, 1].
        emotion_frame_interval: Frame-skip number — submit a face crop for
            emotion inference every N processed frames.
        emotion_smoothing: EMA weight for new emotion scores [0, 1];
            higher reacts faster, lower is more stable.
        face_crop_padding: Extra margin around the landmark bounding box,
            as a fraction of the box size, before cropping for DeepFace.
        min_crop_size: Minimum crop side length (px) worth analyzing.
        landmark_timeout_frames: Consecutive frames without any landmarks
            (e.g. extreme lighting) after which the Face Mesh graph is
            re-initialized to recover from a wedged tracker.
        dropout_grace_frames: Frames a previously tracked face may vanish
            before the tracker reports ``SEARCHING`` instead of ``LOST``.
    """

    max_num_faces: int = 1
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    emotion_frame_interval: int = 15
    emotion_smoothing: float = 0.5
    face_crop_padding: float = 0.25
    min_crop_size: int = 48

    landmark_timeout_frames: int = 90
    dropout_grace_frames: int = 30


@dataclass(frozen=True)
class UISettings:
    """Rendering and heads-up-display configuration.

    Attributes:
        window_name: Title of the OpenCV preview window.
        mesh_alpha: Opacity of the futuristic mesh layer [0, 1].
        mesh_color_near: BGR color of mesh edges closest to the camera.
        mesh_color_far: BGR color of mesh edges furthest from the camera.
        node_interval: Draw a glowing node at every Nth landmark.
        hud_margin: Padding (px) between HUD elements and the frame edge.
    """

    window_name: str = "Facial Mesh // Emotion Inference"
    mesh_alpha: float = 0.75
    mesh_color_near: tuple[int, int, int] = (255, 255, 120)
    mesh_color_far: tuple[int, int, int] = (120, 70, 20)
    node_interval: int = 12
    hud_margin: int = 16


@dataclass(frozen=True)
class AppSettings:
    """Root configuration object composing every layer's settings.

    Attributes:
        camera: Webcam capture settings.
        analyzer: MediaPipe / DeepFace settings.
        ui: Rendering settings.
    """

    camera: CameraSettings = field(default_factory=CameraSettings)
    analyzer: AnalyzerSettings = field(default_factory=AnalyzerSettings)
    ui: UISettings = field(default_factory=UISettings)


EMOTION_LABELS: tuple[str, ...] = (
    "happy", "neutral", "surprise", "sad", "angry", "fear", "disgust",
)
"""Display order of DeepFace emotion classes."""

EMOTION_COLORS: dict[str, tuple[int, int, int]] = {
    "happy": (80, 220, 80),
    "neutral": (200, 200, 200),
    "surprise": (0, 200, 255),
    "sad": (220, 130, 60),
    "angry": (60, 60, 235),
    "fear": (220, 60, 220),
    "disgust": (60, 190, 190),
}
"""BGR accent color per emotion class."""
