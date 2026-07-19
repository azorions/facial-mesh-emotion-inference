"""Application entry point for the facial mesh emotion tracker.

Orchestrates the three runtime layers:

* :class:`src.video.VideoSource`      — threaded, latest-frame webcam stream
* :class:`src.analyzer.FacialAnalyzer` — landmarks every frame, emotion async
* :class:`src.ui.UIOverlay`            — futuristic mesh grid + HUD

Run from the repository root:
    python -m src.main
"""

from __future__ import annotations

import sys

import cv2

from config.settings import AppSettings
from src.analyzer import FacialAnalyzer
from src.ui import FpsTracker, UIOverlay
from src.video import VideoSource

_QUIT_KEYS: frozenset[int] = frozenset({ord("q"), ord("Q"), 27})  # 27 = ESC


def run(settings: AppSettings) -> int:
    """Runs the capture → analyze → render loop until the user quits.

    Args:
        settings: Root application configuration.

    Returns:
        Process exit code (0 on clean shutdown, 1 on fatal error).
    """
    fps_tracker = FpsTracker()

    try:
        overlay = UIOverlay(settings.ui)
        with VideoSource(settings.camera) as source, \
                FacialAnalyzer(settings.analyzer) as analyzer:
            print("[MAIN] Running. Press 'q' or ESC in the window to quit.")
            while True:
                frame = source.read()
                if frame is None:
                    error = source.error
                    if error is not None:
                        raise RuntimeError(error)
                    continue

                analysis = analyzer.process(frame)
                fps = fps_tracker.tick()
                overlay.render(frame, analysis, fps)

                cv2.imshow(settings.ui.window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in _QUIT_KEYS:
                    print("[MAIN] Quit requested.")
                    break
                if cv2.getWindowProperty(
                    settings.ui.window_name, cv2.WND_PROP_VISIBLE
                ) < 1:
                    print("[MAIN] Window closed.")
                    break
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user.")
    except RuntimeError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()

    print("[MAIN] Clean shutdown complete.")
    return 0


def main() -> int:
    """Builds default settings and launches the application.

    Returns:
        Process exit code.
    """
    return run(AppSettings())


if __name__ == "__main__":
    sys.exit(main())
