"""Tests for configuration management."""

import pytest
from pydantic import ValidationError

from src.bachata_analyzer.config import AnalysisConfig


class TestAnalysisConfig:
    """Test cases for AnalysisConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AnalysisConfig()

        assert config.fps == 12
        assert config.max_width == 1280
        assert config.min_segment_sec == 4.0
        assert config.pose_confidence == 0.5
        assert config.tracking_confidence == 0.5
        assert config.create_video == True
        assert config.use_temporal_smoothing == True
        assert config.smoothing_window == 5

    def test_custom_config(self):
        """Test configuration with custom values."""
        config = AnalysisConfig(
            fps=8,
            max_width=720,
            min_segment_sec=3.0,
            pose_confidence=0.7,
            create_video=False,
        )

        assert config.fps == 8
        assert config.max_width == 720
        assert config.min_segment_sec == 3.0
        assert config.pose_confidence == 0.7
        assert config.create_video == False

    def test_fps_validation(self):
        """Test FPS validation constraints."""
        # Valid FPS values
        for fps in [1, 12, 30, 60]:
            config = AnalysisConfig(fps=fps)
            assert config.fps == fps

        # Invalid FPS values
        with pytest.raises(ValidationError):
            AnalysisConfig(fps=0)

        with pytest.raises(ValidationError):
            AnalysisConfig(fps=61)

    def test_max_width_validation(self):
        """Test max width validation constraints."""
        # Valid width values
        for width in [480, 720, 1280, 1920]:
            config = AnalysisConfig(max_width=width)
            assert config.max_width == width

        # Invalid width values
        with pytest.raises(ValidationError):
            AnalysisConfig(max_width=479)

        with pytest.raises(ValidationError):
            AnalysisConfig(max_width=1921)

    def test_min_segment_sec_validation(self):
        """Test minimum segment length validation."""
        # Valid values
        for seg_len in [1.0, 4.0, 10.0, 30.0]:
            config = AnalysisConfig(min_segment_sec=seg_len)
            assert config.min_segment_sec == seg_len

        # Invalid values
        with pytest.raises(ValidationError):
            AnalysisConfig(min_segment_sec=0.9)

        with pytest.raises(ValidationError):
            AnalysisConfig(min_segment_sec=30.1)

    def test_confidence_validation(self):
        """Test confidence parameter validation."""
        # Valid confidence values
        for conf in [0.1, 0.5, 0.8, 1.0]:
            config = AnalysisConfig(pose_confidence=conf, tracking_confidence=conf)
            assert config.pose_confidence == conf
            assert config.tracking_confidence == conf

        # Invalid confidence values
        with pytest.raises(ValidationError):
            AnalysisConfig(pose_confidence=0.05)

        with pytest.raises(ValidationError):
            AnalysisConfig(tracking_confidence=1.1)

    def test_smoothing_window_validation(self):
        """Test smoothing window validation."""
        # Valid values
        for window in [1, 5, 10, 15]:
            config = AnalysisConfig(smoothing_window=window)
            assert config.smoothing_window == window

        # Invalid values
        with pytest.raises(ValidationError):
            AnalysisConfig(smoothing_window=0)

        with pytest.raises(ValidationError):
            AnalysisConfig(smoothing_window=16)

    def test_path_conversion(self):
        """Test path parameter conversion."""
        config = AnalysisConfig(output_dir="custom_output")
        assert str(config.output_dir) == "custom_output"

        config = AnalysisConfig(cache_dir=None)
        assert config.cache_dir is None

    def test_config_dict_conversion(self):
        """Test conversion to dictionary."""
        config = AnalysisConfig(fps=15, create_video=False)
        config_dict = config.model_dump()

        assert isinstance(config_dict, dict)
        assert config_dict["fps"] == 15
        assert config_dict["create_video"] == False
        assert "output_dir" in config_dict

    def test_config_immutability(self):
        """Test that config validation works on assignment."""
        config = AnalysisConfig()

        # Valid assignment
        config.fps = 20
        assert config.fps == 20

        # Invalid assignment should raise error
        with pytest.raises(ValidationError):
            config.fps = 0
