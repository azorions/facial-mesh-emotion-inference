"""Futuristic mesh rendering and heads-up-display for the tracker.

:class:`UIOverlay` receives the raw BGR frame, the structural landmarks
produced by :class:`src.analyzer.FacialAnalyzer`, and the active emotion
prediction, and renders:

* a depth-shaded 3D wireframe grid over the face (MediaPipe tesselation
  topology, colored near→far so the mesh reads as three-dimensional),
* glowing node markers at regular landmark intervals,
* targeting brackets around the face bounding box,
* a HUD panel with the active emotion, its accuracy score, engine status,
  tracking state, and FPS.
"""

from __future__ import annotations

import time
from typing import Sequence

import cv2
import numpy as np

from config.settings import EMOTION_COLORS, UISettings
from src.analyzer import (
    EmotionPrediction,
    FaceLandmarks,
    FrameAnalysis,
    TrackingState,
)

_FONT: int = cv2.FONT_HERSHEY_SIMPLEX
_HUD_BG: tuple[int, int, int] = (30, 20, 10)
_HUD_ACCENT: tuple[int, int, int] = (255, 220, 100)
_HUD_TEXT: tuple[int, int, int] = (235, 235, 235)
_HUD_DIM: tuple[int, int, int] = (150, 150, 150)


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


class UIOverlay:
    """Draws the futuristic face-mesh grid and HUD onto video frames."""

    def __init__(self, settings: UISettings) -> None:
        """Loads the mesh topology and precomputes drawing state.

        Args:
            settings: Rendering configuration.

        Raises:
            RuntimeError: If MediaPipe (source of the mesh topology) is
                not installed.
        """
        self._settings: UISettings = settings
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed (required for the mesh "
                "topology). Fix with: pip install mediapipe"
            ) from exc
        mesh_module = mp.solutions.face_mesh
        self._edges: np.ndarray = np.array(
            sorted(mesh_module.FACEMESH_TESSELATION), dtype=np.int32
        )
        self._contour_edges: np.ndarray = np.array(
            sorted(mesh_module.FACEMESH_CONTOURS), dtype=np.int32
        )

    # ------------------------------ public API ------------------------------ #
    def render(
        self, frame: np.ndarray, analysis: FrameAnalysis, fps: float
    ) -> np.ndarray:
        """Draws all overlays for one frame (in place).

        Args:
            frame: The raw BGR frame to annotate.
            analysis: Structural landmarks + active emotion for this frame.
            fps: Current smoothed frames-per-second estimate.

        Returns:
            The same ``frame`` array, annotated.
        """
        accent = self._accent_color(analysis.emotion)
        if analysis.faces:
            mesh_layer = frame.copy()
            for face in analysis.faces:
                self._draw_mesh_grid(mesh_layer, face)
            alpha = self._settings.mesh_alpha
            cv2.addWeighted(mesh_layer, alpha, frame, 1.0 - alpha, 0.0, frame)
            for face in analysis.faces:
                self._draw_nodes(frame, face)
                self._draw_brackets(frame, face.bbox, accent)

        self._draw_hud(frame, analysis, fps, accent)
        return frame

    # ------------------------------ mesh layer ------------------------------ #
    def _draw_mesh_grid(self, frame: np.ndarray, face: FaceLandmarks) -> None:
        """Draws the depth-shaded 3D wireframe for one face.

        Edge color is interpolated between the configured *near* and *far*
        colors using the average landmark depth of each edge, so surface
        relief (nose forward, jawline back) is visible in the wireframe.

        Args:
            frame: The BGR layer to draw the wireframe on.
            face: The face whose landmarks are rendered.
        """
        points_xy = face.points[:, :2].astype(np.int32)
        depth = face.points[:, 2]
        z_min, z_max = float(depth.min()), float(depth.max())
        z_span = (z_max - z_min) or 1.0
        # 0 = closest to camera (smallest z), 1 = furthest.
        normalized = (depth - z_min) / z_span

        near = np.array(self._settings.mesh_color_near, dtype=np.float32)
        far = np.array(self._settings.mesh_color_far, dtype=np.float32)

        n_points = len(points_xy)
        for start, end in self._edges:
            if start >= n_points or end >= n_points:
                continue
            t = float((normalized[start] + normalized[end]) * 0.5)
            color = tuple(int(c) for c in (near * (1.0 - t) + far * t))
            cv2.line(
                frame,
                tuple(points_xy[start]),
                tuple(points_xy[end]),
                color,
                1,
                cv2.LINE_AA,
            )
        # Emphasize the structural contours (eyes, lips, oval) on top.
        for start, end in self._contour_edges:
            if start >= n_points or end >= n_points:
                continue
            cv2.line(
                frame,
                tuple(points_xy[start]),
                tuple(points_xy[end]),
                self._settings.mesh_color_near,
                1,
                cv2.LINE_AA,
            )

    def _draw_nodes(self, frame: np.ndarray, face: FaceLandmarks) -> None:
        """Draws glowing node markers at regular landmark intervals.

        Args:
            frame: The BGR frame to annotate.
            face: The face whose landmarks are rendered.
        """
        step = max(1, self._settings.node_interval)
        for x, y, _z in face.points[::step]:
            center = (int(x), int(y))
            cv2.circle(frame, center, 2, self._settings.mesh_color_near, -1,
                       cv2.LINE_AA)

    def _draw_brackets(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        color: tuple[int, int, int],
    ) -> None:
        """Draws sci-fi corner targeting brackets around the face box.

        Args:
            frame: The BGR frame to annotate.
            bbox: Pixel bounding box ``(x_min, y_min, x_max, y_max)``.
            color: BGR accent color (follows the active emotion).
        """
        x_min, y_min, x_max, y_max = bbox
        arm = max(12, (x_max - x_min) // 8)
        thickness = 2
        corners: Sequence[tuple[tuple[int, int], tuple[int, int],
                                tuple[int, int]]] = (
            ((x_min, y_min + arm), (x_min, y_min), (x_min + arm, y_min)),
            ((x_max - arm, y_min), (x_max, y_min), (x_max, y_min + arm)),
            ((x_max, y_max - arm), (x_max, y_max), (x_max - arm, y_max)),
            ((x_min + arm, y_max), (x_min, y_max), (x_min, y_max - arm)),
        )
        for a, corner, b in corners:
            cv2.line(frame, corner, a, color, thickness, cv2.LINE_AA)
            cv2.line(frame, corner, b, color, thickness, cv2.LINE_AA)

    # -------------------------------- HUD ----------------------------------- #
    def _draw_hud(
        self,
        frame: np.ndarray,
        analysis: FrameAnalysis,
        fps: float,
        accent: tuple[int, int, int],
    ) -> None:
        """Draws the heads-up display panel and status readouts.

        Args:
            frame: The BGR frame to annotate.
            analysis: This frame's analysis result.
            fps: Current smoothed FPS estimate.
            accent: BGR accent color for the active emotion.
        """
        margin = self._settings.hud_margin
        panel_w, panel_h = 300, 118
        self._fill_panel(frame, margin, margin, panel_w, panel_h)

        x = margin + 14
        self._text(frame, "EMOTION ANALYSIS", (x, margin + 24), 0.5,
                   _HUD_ACCENT)
        cv2.line(frame, (x, margin + 32), (margin + panel_w - 14, margin + 32),
                 _HUD_ACCENT, 1, cv2.LINE_AA)

        emotion = analysis.emotion
        if emotion is not None:
            self._text(frame, emotion.label.upper(), (x, margin + 66), 0.95,
                       accent, thickness=2)
            self._confidence_bar(
                frame, (x, margin + 80), panel_w - 28, emotion.confidence,
                accent,
            )
            self._text(frame, f"ACCURACY {emotion.confidence:5.1f}%",
                       (x, margin + 108), 0.45, _HUD_TEXT)
        else:
            self._text(frame, "STANDBY", (x, margin + 66), 0.95, _HUD_DIM,
                       thickness=2)
            self._text(frame, f"ENGINE: {analysis.engine_status.upper()}",
                       (x, margin + 108), 0.45, _HUD_DIM)

        # Top-right: FPS readout.
        fps_text = f"FPS {fps:5.1f}"
        (text_w, _), _ = cv2.getTextSize(fps_text, _FONT, 0.6, 1)
        self._text(frame, fps_text,
                   (frame.shape[1] - text_w - margin, margin + 24), 0.6,
                   _HUD_TEXT)

        # Bottom-left: tracking + engine status line.
        status = self._status_line(analysis)
        self._text(frame, status, (margin, frame.shape[0] - margin), 0.5,
                   accent if analysis.tracking_state is TrackingState.TRACKING
                   else _HUD_DIM)

        # Bottom-right: controls hint.
        hint = "[Q] QUIT"
        (hint_w, _), _ = cv2.getTextSize(hint, _FONT, 0.5, 1)
        self._text(frame, hint,
                   (frame.shape[1] - hint_w - margin,
                    frame.shape[0] - margin), 0.5, _HUD_DIM)

    def _status_line(self, analysis: FrameAnalysis) -> str:
        """Builds the bottom status string for this frame.

        Args:
            analysis: This frame's analysis result.

        Returns:
            A short uppercase status line for the HUD.
        """
        tracking = {
            TrackingState.TRACKING: "TRACKING LOCKED",
            TrackingState.LOST: "SIGNAL LOST — REACQUIRING",
            TrackingState.SEARCHING: "SCANNING FOR FACE",
        }[analysis.tracking_state]
        return f"{tracking}  |  ENGINE: {analysis.engine_status.upper()}"

    # ------------------------------ primitives ------------------------------ #
    @staticmethod
    def _fill_panel(
        frame: np.ndarray, x: int, y: int, width: int, height: int
    ) -> None:
        """Darkens a translucent rectangular HUD panel region in place.

        Args:
            frame: The BGR frame to annotate.
            x: Panel left edge (px).
            y: Panel top edge (px).
            width: Panel width (px).
            height: Panel height (px).
        """
        y_end = min(y + height, frame.shape[0])
        x_end = min(x + width, frame.shape[1])
        roi = frame[y:y_end, x:x_end].astype(np.float32)
        tint = np.array(_HUD_BG, dtype=np.float32)
        frame[y:y_end, x:x_end] = (roi * 0.30 + tint * 0.70).astype(np.uint8)
        cv2.rectangle(frame, (x, y), (x_end, y_end), _HUD_ACCENT, 1,
                      cv2.LINE_AA)

    @staticmethod
    def _confidence_bar(
        frame: np.ndarray,
        origin: tuple[int, int],
        width: int,
        percent: float,
        color: tuple[int, int, int],
    ) -> None:
        """Draws a horizontal accuracy bar.

        Args:
            frame: The BGR frame to annotate.
            origin: Top-left corner of the bar (px).
            width: Full bar width (px).
            percent: Fill amount [0, 100].
            color: BGR fill color.
        """
        x, y = origin
        height = 10
        cv2.rectangle(frame, (x, y), (x + width, y + height), (90, 90, 90), 1,
                      cv2.LINE_AA)
        fill = int(width * max(0.0, min(percent, 100.0)) / 100.0)
        if fill > 0:
            cv2.rectangle(frame, (x, y), (x + fill, y + height), color, -1)

    @staticmethod
    def _text(
        frame: np.ndarray,
        text: str,
        origin: tuple[int, int],
        scale: float,
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        """Draws outlined text for legibility on any background.

        Args:
            frame: The BGR frame to annotate.
            text: The string to draw.
            origin: Bottom-left corner of the text in pixels.
            scale: OpenCV font scale.
            color: BGR text color.
            thickness: Stroke thickness of the foreground pass.
        """
        cv2.putText(frame, text, origin, _FONT, scale, (0, 0, 0),
                    thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, origin, _FONT, scale, color, thickness,
                    cv2.LINE_AA)

    @staticmethod
    def _accent_color(
        emotion: EmotionPrediction | None,
    ) -> tuple[int, int, int]:
        """Picks the HUD accent color for the active emotion.

        Args:
            emotion: The active prediction, if any.

        Returns:
            The emotion's BGR accent color, or a neutral default.
        """
        if emotion is None:
            return _HUD_ACCENT
        return EMOTION_COLORS.get(emotion.label, _HUD_ACCENT)
