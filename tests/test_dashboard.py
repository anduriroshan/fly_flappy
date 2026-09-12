"""Sanity tests for the OpenCV dashboard + MP4 recorder."""
from pathlib import Path

import numpy as np

from src.telemetry import Dashboard, VideoRecorder


def test_dashboard_composite_shape():
    dash = Dashboard(panel_h=128, panel_w=128, heatmap_neurons=64)
    frame = (np.random.rand(288, 512, 3) * 255).astype(np.uint8)
    state = np.random.randn(500)
    out = dash.compose(frame, state, reward=0.1, step=1, action=1)
    assert out.shape == (128, 128 * 3, 3)
    assert out.dtype == np.uint8


def test_dashboard_handles_missing_inputs():
    dash = Dashboard(panel_h=64, panel_w=64, heatmap_neurons=32)
    out = dash.compose(None, None, reward=0.0, step=0, action=0)
    assert out.shape == (64, 64 * 3, 3)


def test_recorder_writes_mp4(tmp_path: Path):
    dash = Dashboard(panel_h=64, panel_w=64, heatmap_neurons=16)
    path = tmp_path / "out.mp4"
    with VideoRecorder(path, fps=10) as rec:
        for t in range(5):
            frame = dash.compose(None, np.random.randn(16), reward=0.0, step=t, action=0)
            rec.write(frame)
    assert path.exists() and path.stat().st_size > 0
