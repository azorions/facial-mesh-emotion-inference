"""Thread-safe webcam stream with a latest-frame buffer.

A dedicated capture thread pulls frames from the hardware as fast as the
driver delivers them and pushes each one into a bounded :class:`queue.Queue`.
When the buffer is full the *oldest* frame is discarded before the new one is
enqueued, so the consumer always receives the freshest frame the camera has
produced — the classic OpenCV problem of stale frames accumulating inside the
driver's internal buffer (hardware ingestion lag) is eliminated entirely.
"""

from __future__ import annotations

import queue
import sys
import threading
from types import TracebackType

import cv2
import numpy as np

from config.settings import CameraSettings


class VideoSource:
    """Context-managed, thread-safe webcam stream.

    Example:
        >>> with VideoSource(CameraSettings()) as source:
        ...     frame = source.read()
    """

    def __init__(self, settings: CameraSettings) -> None:
        """Stores configuration; the device is opened in :meth:`start`.

        Args:
            settings: Camera capture configuration.
        """
        self._settings: CameraSettings = settings
        self._capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._frames: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=max(1, settings.queue_size)
        )
        self._stop_event: threading.Event = threading.Event()
        self._error_lock: threading.Lock = threading.Lock()
        self._error: str | None = None

    @property
    def error(self) -> str | None:
        """Fatal capture-thread error message, or ``None`` while healthy."""
        with self._error_lock:
            return self._error

    def start(self) -> "VideoSource":
        """Opens the camera and launches the capture thread.

        Returns:
            This ``VideoSource`` instance, ready for :meth:`read`.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        capture = cv2.VideoCapture(self._settings.camera_index, backend)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Could not open camera index {self._settings.camera_index}. "
                "Check that a webcam is connected, not in use by another "
                "application, and that the OS camera-privacy setting allows "
                "desktop apps to access it."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._settings.frame_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._settings.frame_height)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._capture = capture

        actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[VIDEO] Camera opened at {actual_w}x{actual_h}.")

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, name="VideoCapture", daemon=True
        )
        self._thread.start()
        return self

    def read(self, timeout: float | None = None) -> np.ndarray | None:
        """Retrieves the freshest available frame.

        Args:
            timeout: Seconds to wait for a frame; defaults to the configured
                ``read_timeout``.

        Returns:
            The BGR frame (mirrored if configured), or ``None`` if no frame
            arrived within the timeout.

        Raises:
            RuntimeError: If :meth:`start` has not been called.
        """
        if self._thread is None:
            raise RuntimeError("VideoSource.read() called before start().")
        wait = self._settings.read_timeout if timeout is None else timeout
        try:
            return self._frames.get(timeout=wait)
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Stops the capture thread and releases the camera device."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        print("[VIDEO] Camera released.")

    def __enter__(self) -> "VideoSource":
        """Starts the stream when entering a ``with`` block."""
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Releases all resources when leaving a ``with`` block."""
        self.stop()

    # ------------------------------------------------------------------ #
    # Capture thread
    # ------------------------------------------------------------------ #
    def _capture_loop(self) -> None:
        """Thread body: continuously buffers the latest hardware frame."""
        assert self._capture is not None
        failures = 0
        while not self._stop_event.is_set():
            success, frame = self._capture.read()
            if not success or frame is None:
                failures += 1
                if failures >= self._settings.max_consecutive_failures:
                    with self._error_lock:
                        self._error = (
                            "Camera stopped delivering frames "
                            f"({failures} consecutive read failures)."
                        )
                    return
                continue
            failures = 0

            if self._settings.flip_horizontal:
                frame = cv2.flip(frame, 1)

            # Single-producer latest-frame policy: drop the stale frame
            # (if any) so the new one always fits without blocking.
            if self._frames.full():
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._frames.put_nowait(frame)
            except queue.Full:
                pass
