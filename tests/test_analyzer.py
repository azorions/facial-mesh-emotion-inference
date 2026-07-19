"""Unit tests for :mod:`src.analyzer`.

MediaPipe and DeepFace are never imported here: the analyzer accepts an
injected ``face_mesh_factory`` and ``inference_fn``, so every layer of the
frame-processing pipeline — landmark extraction, bounding-box math, the
frame-skip cadence, the dropout / lighting-timeout state machine, and the
asynchronous emotion worker — is exercised with lightweight fakes.

Run from the repository root:
    pytest tests/ -v
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pytest

from config.settings import AnalyzerSettings
from src.analyzer import (
    EmotionWorker,
    FaceLandmarks,
    FacialAnalyzer,
    FrameAnalysis,
    TrackingState,
)

FRAME_W, FRAME_H = 640, 480


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeLandmark:
    """Mimics one MediaPipe normalized landmark."""

    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


class FakeLandmarkList:
    """Mimics MediaPipe's ``NormalizedLandmarkList``."""

    def __init__(self, landmarks: list[FakeLandmark]) -> None:
        self.landmark = landmarks


class FakeMeshOutput:
    """Mimics the object returned by ``FaceMesh.process``."""

    def __init__(
        self, face_lists: list[FakeLandmarkList] | None
    ) -> None:
        self.multi_face_landmarks = face_lists


class FakeFaceMesh:
    """Scriptable stand-in for MediaPipe's ``FaceMesh`` graph.

    Attributes:
        script: Queue of outputs (or exceptions) returned by successive
            ``process`` calls; the last entry repeats forever.
        process_calls: Number of times ``process`` was invoked.
        closed: Whether ``close`` was called.
    """

    def __init__(
        self, script: list[FakeMeshOutput | Exception] | None = None
    ) -> None:
        self.script: list[FakeMeshOutput | Exception] = script or []
        self.process_calls: int = 0
        self.closed: bool = False

    def process(self, _rgb: np.ndarray) -> FakeMeshOutput:
        self.process_calls += 1
        if not self.script:
            return FakeMeshOutput(None)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self.closed = True


def make_face(
    x0: float = 0.3, y0: float = 0.3, x1: float = 0.7, y1: float = 0.7
) -> FakeLandmarkList:
    """Builds a fake face whose landmarks span a normalized rectangle.

    Args:
        x0: Normalized left edge.
        y0: Normalized top edge.
        x1: Normalized right edge.
        y1: Normalized bottom edge.

    Returns:
        A four-corner fake landmark list covering the rectangle.
    """
    return FakeLandmarkList([
        FakeLandmark(x0, y0, -0.02),
        FakeLandmark(x1, y0, -0.01),
        FakeLandmark(x0, y1, 0.01),
        FakeLandmark(x1, y1, 0.02),
    ])


def face_output() -> FakeMeshOutput:
    """One ``process`` result containing a single centered face."""
    return FakeMeshOutput([make_face()])


def empty_output() -> FakeMeshOutput:
    """One ``process`` result containing no faces (dropout frame)."""
    return FakeMeshOutput(None)


def frame() -> np.ndarray:
    """A blank BGR test frame."""
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


def wait_for(
    predicate: Callable[[], bool], timeout: float = 3.0
) -> bool:
    """Polls ``predicate`` until true or the timeout elapses.

    Args:
        predicate: Zero-argument condition to await.
        timeout: Maximum seconds to wait.

    Returns:
        ``True`` if the predicate became true in time.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def settings() -> AnalyzerSettings:
    """Small intervals so cadence tests stay fast."""
    return AnalyzerSettings(
        emotion_frame_interval=3,
        emotion_smoothing=0.5,
        min_crop_size=10,
        landmark_timeout_frames=5,
        dropout_grace_frames=2,
    )


def build_analyzer(
    settings: AnalyzerSettings,
    mesh: FakeFaceMesh,
    inference_fn: Callable[[np.ndarray], dict[str, float]] | None = None,
) -> FacialAnalyzer:
    """Constructs an analyzer wired to the given fakes (not started)."""
    return FacialAnalyzer(
        settings,
        face_mesh_factory=lambda: mesh,
        inference_fn=inference_fn or (lambda crop: {"neutral": 100.0}),
    )


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
class TestLifecycle:
    def test_process_before_start_raises(
        self, settings: AnalyzerSettings
    ) -> None:
        analyzer = build_analyzer(settings, FakeFaceMesh())
        with pytest.raises(RuntimeError, match="before start"):
            analyzer.process(frame())

    def test_close_stops_worker_and_mesh(
        self, settings: AnalyzerSettings
    ) -> None:
        mesh = FakeFaceMesh([face_output()])
        analyzer = build_analyzer(settings, mesh)
        with analyzer:
            worker = analyzer._worker
            assert worker is not None and worker.is_alive()
        assert mesh.closed
        assert not worker.is_alive()


# --------------------------------------------------------------------------- #
# Structural landmark extraction
# --------------------------------------------------------------------------- #
class TestLandmarkExtraction:
    def test_face_produces_pixel_landmarks_and_state(
        self, settings: AnalyzerSettings
    ) -> None:
        mesh = FakeFaceMesh([face_output()])
        with build_analyzer(settings, mesh) as analyzer:
            analysis = analyzer.process(frame())

        assert isinstance(analysis, FrameAnalysis)
        assert analysis.tracking_state is TrackingState.TRACKING
        assert len(analysis.faces) == 1
        face = analysis.faces[0]
        assert isinstance(face, FaceLandmarks)
        # Normalized 0.3..0.7 must scale to pixel space.
        assert face.points[:, 0].min() == pytest.approx(0.3 * FRAME_W)
        assert face.points[:, 1].max() == pytest.approx(0.7 * FRAME_H)

    def test_landmark_dict_view(self, settings: AnalyzerSettings) -> None:
        mesh = FakeFaceMesh([face_output()])
        with build_analyzer(settings, mesh) as analyzer:
            face = analyzer.process(frame()).faces[0]
        as_dict = face.as_dict()
        assert set(as_dict.keys()) == {0, 1, 2, 3}
        assert as_dict[0][0] == pytest.approx(0.3 * FRAME_W)

    def test_bbox_is_padded_and_clamped_to_frame(
        self, settings: AnalyzerSettings
    ) -> None:
        # Face hugging the top-left corner: padding must clamp at 0.
        mesh = FakeFaceMesh(
            [FakeMeshOutput([make_face(0.0, 0.0, 0.4, 0.4)])]
        )
        with build_analyzer(settings, mesh) as analyzer:
            x_min, y_min, x_max, y_max = analyzer.process(frame()).faces[0].bbox

        assert x_min == 0 and y_min == 0
        assert x_max <= FRAME_W - 1 and y_max <= FRAME_H - 1
        # Padding must extend beyond the raw landmark extent.
        assert x_max > int(0.4 * FRAME_W)


# --------------------------------------------------------------------------- #
# Error handling: dropouts and lighting timeouts
# --------------------------------------------------------------------------- #
class TestErrorHandling:
    def test_dropout_reports_lost_without_crashing(
        self, settings: AnalyzerSettings
    ) -> None:
        # One tracked frame, then the user steps away.
        mesh = FakeFaceMesh([face_output(), empty_output()])
        with build_analyzer(settings, mesh) as analyzer:
            assert analyzer.process(frame()).tracking_state \
                is TrackingState.TRACKING
            dropout = analyzer.process(frame())

        assert dropout.faces == []
        assert dropout.tracking_state is TrackingState.LOST

    def test_dropout_degrades_to_searching_after_grace(
        self, settings: AnalyzerSettings
    ) -> None:
        mesh = FakeFaceMesh([face_output(), empty_output()])
        with build_analyzer(settings, mesh) as analyzer:
            analyzer.process(frame())  # TRACKING
            states = [
                analyzer.process(frame()).tracking_state
                for _ in range(settings.dropout_grace_frames + 1)
            ]
        assert states[: settings.dropout_grace_frames] == \
            [TrackingState.LOST] * settings.dropout_grace_frames
        assert states[-1] is TrackingState.SEARCHING

    def test_mesh_exception_is_intercepted(
        self, settings: AnalyzerSettings
    ) -> None:
        mesh = FakeFaceMesh([RuntimeError("graph wedged"), face_output()])
        with build_analyzer(settings, mesh) as analyzer:
            crashed = analyzer.process(frame())     # must not raise
            recovered = analyzer.process(frame())

        assert crashed.faces == []
        assert len(recovered.faces) == 1

    def test_landmark_timeout_reinitializes_face_mesh(
        self, settings: AnalyzerSettings
    ) -> None:
        # Extreme-lighting scenario: no landmarks, ever.
        meshes: list[FakeFaceMesh] = []

        def factory() -> FakeFaceMesh:
            mesh = FakeFaceMesh([empty_output()])
            meshes.append(mesh)
            return mesh

        analyzer = FacialAnalyzer(
            settings,
            face_mesh_factory=factory,
            inference_fn=lambda crop: {},
        )
        with analyzer:
            for _ in range(settings.landmark_timeout_frames):
                analyzer.process(frame())

        assert len(meshes) == 2          # original + one recovery rebuild
        assert meshes[0].closed          # wedged graph was torn down


# --------------------------------------------------------------------------- #
# Frame-skip cadence into the emotion worker
# --------------------------------------------------------------------------- #
class TestFrameSkipCadence:
    def test_submits_every_nth_frame_only(
        self, settings: AnalyzerSettings
    ) -> None:
        mesh = FakeFaceMesh([face_output()])
        with build_analyzer(settings, mesh) as analyzer:
            submitted: list[np.ndarray] = []
            assert analyzer._worker is not None
            analyzer._worker.submit = submitted.append  # type: ignore[method-assign]

            for _ in range(9):
                analyzer.process(frame())

        # interval=3 over 9 frames → frames 3, 6, 9.
        assert len(submitted) == 3
        assert all(isinstance(crop, np.ndarray) for crop in submitted)
        assert all(crop.size > 0 for crop in submitted)

    def test_undersized_crop_is_not_submitted(
        self, settings: AnalyzerSettings
    ) -> None:
        # A face far smaller than min_crop_size (10 px): 4 px wide.
        tiny = FakeMeshOutput(
            [make_face(0.500, 0.500, 0.503, 0.503)]
        )
        mesh = FakeFaceMesh([tiny])
        with build_analyzer(settings, mesh) as analyzer:
            submitted: list[np.ndarray] = []
            assert analyzer._worker is not None
            analyzer._worker.submit = submitted.append  # type: ignore[method-assign]
            for _ in range(settings.emotion_frame_interval * 2):
                analyzer.process(frame())

        assert submitted == []

    def test_no_submission_without_a_face(
        self, settings: AnalyzerSettings
    ) -> None:
        mesh = FakeFaceMesh([empty_output()])
        with build_analyzer(settings, mesh) as analyzer:
            submitted: list[np.ndarray] = []
            assert analyzer._worker is not None
            analyzer._worker.submit = submitted.append  # type: ignore[method-assign]
            for _ in range(settings.emotion_frame_interval * 2):
                analyzer.process(frame())

        assert submitted == []


# --------------------------------------------------------------------------- #
# Asynchronous emotion worker
# --------------------------------------------------------------------------- #
class TestEmotionWorker:
    def test_publishes_prediction_after_submit(
        self, settings: AnalyzerSettings
    ) -> None:
        worker = EmotionWorker(
            settings, inference_fn=lambda crop: {"happy": 90.0, "sad": 10.0}
        )
        worker.start()
        try:
            assert worker.latest() is None
            worker.submit(np.zeros((32, 32, 3), dtype=np.uint8))
            assert wait_for(lambda: worker.latest() is not None)
            prediction = worker.latest()
            assert prediction is not None
            assert prediction.label == "happy"
            assert prediction.confidence == pytest.approx(90.0)
        finally:
            worker.stop()
        assert not worker.is_alive()

    def test_inference_failure_does_not_kill_thread(
        self, settings: AnalyzerSettings
    ) -> None:
        calls: list[int] = []

        def flaky(_crop: np.ndarray) -> dict[str, float]:
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("corrupt crop")
            return {"neutral": 100.0}

        worker = EmotionWorker(settings, inference_fn=flaky)
        worker.start()
        try:
            worker.submit(np.zeros((32, 32, 3), dtype=np.uint8))
            assert wait_for(lambda: len(calls) >= 1)
            assert worker.latest() is None       # failure produced nothing
            assert worker.is_alive()             # ...and the thread survived

            worker.submit(np.zeros((32, 32, 3), dtype=np.uint8))
            assert wait_for(lambda: worker.latest() is not None)
            prediction = worker.latest()
            assert prediction is not None and prediction.label == "neutral"
        finally:
            worker.stop()

    def test_scores_are_exponentially_smoothed(
        self, settings: AnalyzerSettings
    ) -> None:
        results = [
            {"happy": 100.0, "sad": 0.0},
            {"happy": 0.0, "sad": 100.0},
        ]
        worker = EmotionWorker(
            settings, inference_fn=lambda crop: results.pop(0)
        )
        worker.start()
        try:
            worker.submit(np.zeros((32, 32, 3), dtype=np.uint8))
            assert wait_for(lambda: worker.latest() is not None)
            first = worker.latest()
            assert first is not None and first.label == "happy"

            worker.submit(np.zeros((32, 32, 3), dtype=np.uint8))
            assert wait_for(
                lambda: (latest := worker.latest()) is not None
                and latest.timestamp != first.timestamp
            )
            second = worker.latest()
            assert second is not None
            # alpha = 0.5: happy = 0.5*0 + 0.5*100 = 50, sad likewise.
            assert second.scores["happy"] == pytest.approx(50.0)
            assert second.scores["sad"] == pytest.approx(50.0)
        finally:
            worker.stop()

    def test_status_reports_online_with_injected_engine(
        self, settings: AnalyzerSettings
    ) -> None:
        worker = EmotionWorker(settings, inference_fn=lambda crop: {})
        assert worker.status == "loading model"
        worker.start()
        try:
            assert wait_for(lambda: worker.status == "online")
        finally:
            worker.stop()

    def test_newer_submission_replaces_pending(
        self, settings: AnalyzerSettings
    ) -> None:
        seen: list[float] = []
        gate = time.monotonic() + 0.15

        def slow(crop: np.ndarray) -> dict[str, float]:
            seen.append(float(crop[0, 0, 0]))
            while time.monotonic() < gate:
                time.sleep(0.01)
            return {"neutral": 100.0}

        worker = EmotionWorker(settings, inference_fn=slow)
        worker.start()
        try:
            first = np.full((8, 8, 3), 1, dtype=np.uint8)
            worker.submit(first)
            assert wait_for(lambda: len(seen) == 1)
            # While the worker is busy, three newer crops arrive; only the
            # freshest may be analyzed afterwards.
            for value in (2, 3, 4):
                worker.submit(np.full((8, 8, 3), value, dtype=np.uint8))
            assert wait_for(lambda: len(seen) >= 2)
            time.sleep(0.1)  # allow any (incorrect) extra work to surface
            assert seen == [1.0, 4.0]
        finally:
            worker.stop()
