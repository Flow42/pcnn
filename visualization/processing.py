"""Frame collection, result selection, and browser-compatible video export."""

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Mapping

import imageio.v2 as imageio
import numpy as np

from .algorithms import Algorithm


@dataclass
class RunResult:
    frames: tuple[np.ndarray, ...]
    best_index: int
    video_path: str
    fps: int


def cleanup_result(result: RunResult | None):
    if result is not None:
        shutil.rmtree(Path(result.video_path).parent, ignore_errors=True)


def entropy(frame: np.ndarray) -> float:
    counts = np.unique(frame, return_counts=True)[1]
    probabilities = counts / frame.size
    return max(0.0, float(-np.sum(probabilities * np.log2(probabilities))))


def process_image(algorithm: Algorithm, image_path: str, values: Mapping,
                  fps: int = 8, progress: Callable | None = None) -> RunResult:
    if not image_path:
        raise ValueError("Upload an image before running the algorithm.")
    if not 1 <= fps <= 30 or int(fps) != fps:
        raise ValueError("Playback speed must be an integer between 1 and 30 fps.")
    frames = []
    for index, frame in enumerate(algorithm.frames(image_path, values)):
        if frame.dtype != np.uint8 or frame.ndim != 2 or not frame.size:
            raise ValueError("Algorithms must yield nonempty uint8 grayscale frames.")
        if frames and frame.shape != frames[0].shape:
            raise ValueError("All frames must have the same dimensions.")
        frames.append(frame.copy())
        if progress:
            progress(index + 1)
    if not frames:
        raise ValueError("The algorithm did not produce any frames.")
    best_index = max(range(len(frames)), key=lambda i: entropy(frames[i]))
    directory = Path(tempfile.mkdtemp(prefix="pcnn-viewer-"))
    video_path = directory / "pulses.mp4"
    try:
        # H.264/yuv420p plays in browsers. Pad odd dimensions without changing
        # the original frames used for inspection or entropy calculations.
        with imageio.get_writer(str(video_path), format="FFMPEG", mode="I",
                                fps=int(fps), codec="libx264", pixelformat="yuv420p",
                                macro_block_size=1, ffmpeg_log_level="error") as writer:
            for frame in frames:
                padded = np.pad(frame, ((0, frame.shape[0] % 2), (0, frame.shape[1] % 2)), mode="edge")
                writer.append_data(np.repeat(padded[..., None], 3, axis=2))
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return RunResult(tuple(frames), best_index, str(video_path), int(fps))


def inspect_frame(result: RunResult | None, index: int):
    """One-based frame navigation, clamped at both ends."""
    if result is None:
        return None, 1, "Run an image to inspect its frames."
    index = max(1, min(int(index), len(result.frames)))
    frame = result.frames[index - 1]
    return frame, index, f"Frame {index} / {len(result.frames)} · {(index - 1) / result.fps:.2f} s · entropy {entropy(frame):.3f} bits"
