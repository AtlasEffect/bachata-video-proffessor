"""Video processing utilities for Bachata analysis."""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Optional, Any
import yt_dlp
from tqdm import tqdm

from .config import AnalysisConfig


class VideoProcessor:
    """Handles video loading and frame extraction."""

    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 30.0  # Default fallback
        self.total_frames: int = 0
        self.width: int = 1920  # Default fallback
        self.height: int = 1080  # Default fallback

    def load_video(self, video_path: str) -> bool:
        """Load video from file path or YouTube URL."""
        if video_path.startswith(("http://", "https://")):
            video_path = self._download_youtube_video(video_path)

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)

        return True

    def _download_youtube_video(self, url: str) -> str:
        """Download video from YouTube URL."""
        cache_dir = self.config.cache_dir or Path("cache")
        cache_dir.mkdir(exist_ok=True)

        ydl_opts = {
            "format": "worst[width<=720]/worst",  # Prefer lower quality for CPU
            "outtmpl": str(cache_dir / "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore[arg-type]
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            return str(filename)  # Ensure we return str

    def get_frame_generator(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Generate frames at the target FPS."""
        if self.cap is None:
            raise ValueError("Video not loaded")

        frame_interval = max(1, int(self.fps / self.config.fps))
        frame_idx = 0
        output_frame_idx = 0

        # Calculate target dimensions
        if self.width > self.config.max_width:
            scale = self.config.max_width / self.width
            target_width = self.config.max_width
            target_height = int(self.height * scale)
        else:
            target_width = self.width
            target_height = self.height

        with tqdm(
            total=self.total_frames // frame_interval, desc="Processing frames"
        ) as pbar:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    # Resize if necessary
                    if (target_width, target_height) != (self.width, self.height):
                        frame = cv2.resize(frame, (target_width, target_height))

                    yield output_frame_idx, frame
                    output_frame_idx += 1
                    pbar.update(1)

                frame_idx += 1

    def get_duration(self) -> float:
        """Get video duration in seconds."""
        if self.cap is None:
            raise ValueError("Video not loaded")
        if self.fps == 0:
            return 0.0
        return self.total_frames / self.fps

    def get_frame_timestamp(self, frame_idx: int) -> float:
        """Get timestamp for a given frame index."""
        if self.fps == 0:
            raise ValueError("Video not loaded")
        return frame_idx / self.config.fps

    def release(self) -> None:
        """Release video capture resources."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "VideoProcessor":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()
