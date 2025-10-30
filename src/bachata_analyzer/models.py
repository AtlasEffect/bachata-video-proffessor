"""Data models for Bachata dance analysis."""

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class Keypoint(BaseModel):
    """Single keypoint with coordinates and confidence."""

    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


class PoseLandmarks(BaseModel):
    """Complete pose landmarks for a person."""

    keypoints: Dict[str, Keypoint]
    confidence: float
    bbox: Optional[Tuple[float, float, float, float]] = None  # x, y, w, h


class PersonTrack(BaseModel):
    """Tracking information for a person across frames."""

    track_id: int
    landmarks_history: List[PoseLandmarks] = Field(default_factory=list)
    frame_indices: List[int] = Field(default_factory=list)
    avg_confidence: float = 0.0
    persistence: float = 0.0  # How consistently this person appears


class SegmentFeatures(BaseModel):
    """Features extracted for a dance segment."""

    avg_speed: float = 0.0
    turns: bool = False
    dip: bool = False
    hand_distance_avg: float = 0.0
    torso_rotation_avg: float = 0.0
    step_cadence: float = 0.0
    freeze_frames: int = 0
    total_frames: int = 0


class DanceSegment(BaseModel):
    """A single dance combination/segment."""

    id: int
    start_sec: float
    end_sec: float
    start_frame: int
    end_frame: int
    roles: Dict[str, int]  # {"leader_track": 0, "follower_track": 1}
    leader_name: str = "Leader"
    follower_name: str = "Follower"
    features: SegmentFeatures
    tentative_name: str = ""


class AnalysisResult(BaseModel):
    """Complete analysis result for a video."""

    video_id: str
    video_path: str
    fps: int
    total_frames: int
    duration_sec: float
    segments: List[DanceSegment]
    tracks: List[PersonTrack]
    config: Dict

    def get_segment_count(self) -> int:
        """Get total number of detected segments."""
        return len(self.segments)

    def get_total_dance_time(self) -> float:
        """Get total time spent dancing (sum of all segments)."""
        return sum(seg.end_sec - seg.start_sec for seg in self.segments)
