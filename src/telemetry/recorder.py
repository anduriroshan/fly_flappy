"""MP4 video recorder built on cv2.VideoWriter."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class VideoRecorder:
    """Lazy-opens the writer on the first frame so we know the size."""

    def __init__(self, out_path: str | Path, fps: int = 30, fourcc: str = "mp4v"):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps)
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self._writer: Optional[cv2.VideoWriter] = None
        self._size: tuple[int, int] | None = None
        self.frames_written = 0

    def write(self, frame: np.ndarray) -> None:
        if frame is None:
            return
        h, w = frame.shape[:2]
        if self._writer is None:
            self._size = (w, h)
            self._writer = cv2.VideoWriter(str(self.out_path), self.fourcc,
                                           self.fps, self._size)
        elif (w, h) != self._size:
            # Frame size changed mid-run — resize rather than crash the recording.
            frame = cv2.resize(frame, self._size)
        self._writer.write(frame)
        self.frames_written += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()
