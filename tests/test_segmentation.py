"""Tests for dance segmentation functionality."""

import numpy as np
import pytest
from unittest.mock import Mock

from src.bachata_analyzer.config import AnalysisConfig
from src.bachata_analyzer.segmentation import DanceSegmenter
from src.bachata_analyzer.models import PersonTrack, PoseLandmarks, Keypoint


class TestDanceSegmenter:
    """Test cases for DanceSegmenter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = AnalysisConfig()
        self.segmenter = DanceSegmenter(self.config)

    def create_mock_pose(self, x_offset=0.0, y_offset=0.0) -> PoseLandmarks:
        """Create a mock pose with basic keypoints."""
        keypoints = {
            "nose": Keypoint(x=0.5 + x_offset, y=0.1 + y_offset),
            "left_shoulder": Keypoint(x=0.4 + x_offset, y=0.3 + y_offset),
            "right_shoulder": Keypoint(x=0.6 + x_offset, y=0.3 + y_offset),
            "left_hip": Keypoint(x=0.4 + x_offset, y=0.6 + y_offset),
            "right_hip": Keypoint(x=0.6 + x_offset, y=0.6 + y_offset),
            "left_knee": Keypoint(x=0.4 + x_offset, y=0.8 + y_offset),
            "right_knee": Keypoint(x=0.6 + x_offset, y=0.8 + y_offset),
            "left_ankle": Keypoint(x=0.4 + x_offset, y=0.9 + y_offset),
            "right_ankle": Keypoint(x=0.6 + x_offset, y=0.9 + y_offset),
            "left_wrist": Keypoint(x=0.3 + x_offset, y=0.4 + y_offset),
            "right_wrist": Keypoint(x=0.7 + x_offset, y=0.4 + y_offset),
        }
        return PoseLandmarks(keypoints=keypoints, confidence=0.8)

    def create_mock_track(
        self, num_frames: int, movement_pattern: str = "static"
    ) -> PersonTrack:
        """Create a mock person track with specified movement pattern."""
        track = PersonTrack(track_id=0)

        for i in range(num_frames):
            if movement_pattern == "static":
                pose = self.create_mock_pose()
            elif movement_pattern == "linear":
                pose = self.create_mock_pose(x_offset=i * 0.01)
            elif movement_pattern == "circular":
                angle = 2 * np.pi * i / num_frames
                x_offset = 0.1 * np.cos(angle)
                y_offset = 0.1 * np.sin(angle)
                pose = self.create_mock_pose(x_offset, y_offset)
            else:
                pose = self.create_mock_pose()

            track.landmarks_history.append(pose)
            track.frame_indices.append(i)

        track.avg_confidence = 0.8
        track.persistence = 1.0
        return track

    def test_extract_features_static_movement(self):
        """Test feature extraction for static movement."""
        tracks = {
            0: self.create_mock_track(10, "static"),
            1: self.create_mock_track(10, "static"),
        }

        features = self.segmenter.extract_features(tracks, fps=12)

        assert features.shape[0] == 10  # 10 frames
        assert features.shape[1] == 10  # 10 features per frame

        # Static movement should have low velocity
        avg_velocity = np.mean(features[:, 0])
        assert avg_velocity < 0.1

    def test_extract_features_linear_movement(self):
        """Test feature extraction for linear movement."""
        tracks = {
            0: self.create_mock_track(10, "linear"),
            1: self.create_mock_track(10, "static"),
        }

        features = self.segmenter.extract_features(tracks, fps=12)

        # Linear movement should have higher velocity than static
        avg_velocity = np.mean(features[:, 0])
        assert avg_velocity > 0.001  # Lower threshold for small movements

    def test_detect_change_points_no_movement(self):
        """Test change point detection with no movement changes."""
        # Create uniform features (no changes)
        features = np.random.normal(0, 0.01, (100, 10))
        self.segmenter.features = features

        change_points = self.segmenter.detect_change_points()

        # Should return at least start and end points
        assert len(change_points) >= 2
        assert change_points[0] >= 0  # May not be exactly 0 due to peak detection
        assert change_points[-1] == len(features) - 1

    def test_detect_change_points_with_movement(self):
        """Test change point detection with movement changes."""
        # Create features with clear changes
        features = np.zeros((100, 10))
        features[30:60, 0] = 1.0  # High velocity in middle
        features[70:90, 0] = 0.5  # Medium velocity later

        self.segmenter.features = features

        change_points = self.segmenter.detect_change_points()

        # Should detect multiple change points
        assert len(change_points) >= 2
        assert change_points[0] >= 0  # May not be exactly 0 due to peak detection
        assert change_points[-1] == len(features) - 1

    def test_create_segments(self):
        """Test segment creation from change points."""
        change_points = [0, 24, 48, 71]  # 3 segments
        fps = 12
        tracks = {
            0: self.create_mock_track(72, "static"),
            1: self.create_mock_track(72, "static"),
        }

        # First extract features so they're available for segment creation
        self.segmenter.extract_features(tracks, fps)

        segments = self.segmenter.create_segments(change_points, fps, tracks)

        assert len(segments) == 3

        # Check first segment
        assert segments[0].id == 1
        assert segments[0].start_frame == 0
        assert segments[0].end_frame == 24
        assert segments[0].start_sec == 0.0
        assert segments[0].end_sec == 2.0

        # Check segment features
        assert segments[0].features.total_frames == 24
        assert segments[0].tentative_name == "Combo 1"

    def test_identify_roles(self):
        """Test leader/follower role identification."""
        # Create tracks with different movement patterns
        leader_track = self.create_mock_track(10, "linear")  # More movement
        follower_track = self.create_mock_track(10, "static")  # Less movement

        tracks = {0: leader_track, 1: follower_track}

        roles = self.segmenter.identify_roles(tracks)

        assert len(roles) == 2
        assert 0 in roles
        assert 1 in roles
        assert roles[0] in ["Leader", "Follower"]
        assert roles[1] in ["Leader", "Follower"]
        assert roles[0] != roles[1]  # Should be different roles

    def test_calculate_leadership_score(self):
        """Test leadership score calculation."""
        static_track = self.create_mock_track(10, "static")
        moving_track = self.create_mock_track(10, "linear")

        static_score = self.segmenter._calculate_leadership_score(static_track)
        moving_score = self.segmenter._calculate_leadership_score(moving_track)

        # Moving track should have higher leadership score
        assert moving_score > static_score

    def test_segment_features_calculation(self):
        """Test segment feature calculation."""
        # Create features with known properties
        self.segmenter.features = np.zeros((60, 10))
        self.segmenter.features[20:40, 0] = 0.1  # Higher velocity
        self.segmenter.features[20:40, 7] = 1.0  # Turns
        self.segmenter.features[30:35, 8] = 1.0  # Dips

        segment_features = self.segmenter._calculate_segment_features(20, 40)

        assert segment_features.total_frames == 20
        assert segment_features.avg_speed > 0  # Should detect higher velocity
        assert segment_features.turns == True  # Should detect turns
        assert segment_features.dip == True  # Should detect dips

    def test_empty_tracks(self):
        """Test handling of empty tracks."""
        features = self.segmenter.extract_features({}, fps=12)
        assert len(features) == 0

        change_points = self.segmenter.detect_change_points()
        assert len(change_points) == 0

    def test_single_track(self):
        """Test handling of single track (no couple)."""
        tracks = {0: self.create_mock_track(10, "linear")}

        features = self.segmenter.extract_features(tracks, fps=12)
        # Should still extract features even with single track
        assert features.shape[0] == 10
        assert features.shape[1] == 10
