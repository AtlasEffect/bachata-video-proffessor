"""Configuration management for Bachata analysis."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class AnalysisConfig(BaseModel):
    """Configuration for Bachata dance analysis."""

    model_config = ConfigDict(validate_assignment=True)

    # Video processing
    fps: int = Field(
        default=12, ge=1, le=60, description="Frames per second for analysis"
    )
    max_width: int = Field(
        default=1280, ge=480, le=1920, description="Maximum video width"
    )
    min_segment_sec: float = Field(
        default=4.0, ge=1.0, le=30.0, description="Minimum segment length in seconds"
    )

    # Pose detection
    pose_confidence: float = Field(
        default=0.5, ge=0.1, le=1.0, description="Minimum pose confidence"
    )
    tracking_confidence: float = Field(
        default=0.5, ge=0.1, le=1.0, description="Minimum tracking confidence"
    )

    # Segmentation
    change_point_sensitivity: float = Field(
        default=0.8,
        ge=0.1,
        le=2.0,
        description="Sensitivity for change point detection",
    )
    pause_threshold: float = Field(
        default=0.3, ge=0.1, le=1.0, description="Threshold for detecting pauses"
    )

    # Output
    output_dir: Path = Field(default=Path("output"), description="Output directory")
    create_video: bool = Field(
        default=True, description="Create annotated video output"
    )
    cache_dir: Optional[Path] = Field(
        default=Path("cache"), description="Cache directory for downloads"
    )

    # Performance
    use_temporal_smoothing: bool = Field(
        default=True, description="Apply temporal smoothing to keypoints"
    )
    smoothing_window: int = Field(
        default=5, ge=1, le=15, description="Window size for temporal smoothing"
    )
