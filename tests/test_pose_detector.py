"""Tests for pose detection and tracking functionality."""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from src.bachata_analyzer.config import AnalysisConfig
from src.bachata_analyzer.pose_detector import PoseDetector
from src.bachata_analyzer.models import PoseLandmarks, Keypoint


class TestPoseDetector:
    """Test cases for PoseDetector class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = AnalysisConfig()
        self.detector = PoseDetector(self.config)

    def create_mock_frame(self, width=640, height=480):
        """Create a mock video frame."""
        return np.zeros((height, width, 3), dtype=np.uint8)

    def create_mock_pose_landmarks(self):
        """Create mock MediaPipe pose landmarks."""
        mock_landmarks = Mock()
        mock_landmarks.landmark = []

        # Create 33 landmarks with default values
        for i in range(33):
            landmark = Mock()
            landmark.x = 0.5 + (i % 3) * 0.1
            landmark.y = 0.3 + (i % 5) * 0.1
            landmark.z = 0.0
            landmark.visibility = 0.8
            mock_landmarks.landmark.append(landmark)

        return mock_landmarks

    def test_convert_landmarks(self):
        """Test conversion of MediaPipe landmarks to our format."""
        mp_landmarks = self.create_mock_pose_landmarks()

        pose = self.detector._convert_landmarks(mp_landmarks)

        assert isinstance(pose, PoseLandmarks)
        assert len(pose.keypoints) == 33
        assert "nose" in pose.keypoints
        assert "left_shoulder" in pose.keypoints
        assert "right_shoulder" in pose.keypoints

        # Check confidence calculation
        assert 0 < pose.confidence <= 1.0

        # Check bounding box
        assert pose.bbox is not None
        assert len(pose.bbox) == 4

    def test_calculate_pose_distance(self):
        """Test distance calculation between two poses."""
        # Create two poses with known positions
        pose1 = PoseLandmarks(
            keypoints={
                "left_shoulder": Keypoint(x=0.4, y=0.3),
                "right_shoulder": Keypoint(x=0.6, y=0.3),
                "left_hip": Keypoint(x=0.4, y=0.6),
                "right_hip": Keypoint(x=0.6, y=0.6),
            },
            confidence=0.8,
        )

        pose2 = PoseLandmarks(
            keypoints={
                "left_shoulder": Keypoint(x=0.45, y=0.35),  # Slight offset
                "right_shoulder": Keypoint(x=0.65, y=0.35),
                "left_hip": Keypoint(x=0.45, y=0.65),
                "right_hip": Keypoint(x=0.65, y=0.65),
            },
            confidence=0.8,
        )

        distance = self.detector._calculate_pose_distance(pose1, pose2)

        assert distance > 0
        assert distance < 1.0  # Should be relatively small for similar poses

    def test_update_tracking_new_tracks(self):
        """Test tracking update with new poses (creates new tracks)."""
        poses = [
            PoseLandmarks(keypoints={"nose": Keypoint(x=0.5, y=0.1)}, confidence=0.8),
            PoseLandmarks(keypoints={"nose": Keypoint(x=0.3, y=0.1)}, confidence=0.7),
        ]

        self.detector.update_tracking(poses, frame_idx=0)

        assert len(self.detector.tracks) == 2
        assert 0 in self.detector.tracks
        assert 1 in self.detector.tracks

        # Check track properties
        track0 = self.detector.tracks[0]
        assert track0.track_id == 0
        assert len(track0.landmarks_history) == 1
        assert track0.frame_indices == [0]

    def test_update_tracking_existing_tracks(self):
        """Test tracking update with poses close to existing tracks."""
        # Create initial tracks with required keypoints for distance calculation
        poses1 = [
            PoseLandmarks(
                keypoints={
                    "nose": Keypoint(x=0.5, y=0.1),
                    "left_shoulder": Keypoint(x=0.4, y=0.3),
                    "right_shoulder": Keypoint(x=0.6, y=0.3),
                    "left_hip": Keypoint(x=0.4, y=0.6),
                    "right_hip": Keypoint(x=0.6, y=0.6),
                },
                confidence=0.8,
            )
        ]
        self.detector.update_tracking(poses1, frame_idx=0)

        # Update with similar poses (should match existing tracks)
        poses2 = [
            PoseLandmarks(
                keypoints={
                    "nose": Keypoint(x=0.51, y=0.11),  # Slight offset
                    "left_shoulder": Keypoint(x=0.41, y=0.31),
                    "right_shoulder": Keypoint(x=0.61, y=0.31),
                    "left_hip": Keypoint(x=0.41, y=0.61),
                    "right_hip": Keypoint(x=0.61, y=0.61),
                },
                confidence=0.8,
            )
        ]
        self.detector.update_tracking(poses2, frame_idx=1)

        assert len(self.detector.tracks) == 1
        track = self.detector.tracks[0]
        assert len(track.landmarks_history) == 2
        assert track.frame_indices == [0, 1]

    def test_update_track_statistics(self):
        """Test track statistics calculation."""
        # Create a track with multiple poses
        track = self.detector.tracks[0] = Mock()
        track.landmarks_history = [
            PoseLandmarks(keypoints={}, confidence=0.8),
            PoseLandmarks(keypoints={}, confidence=0.6),
            PoseLandmarks(keypoints={}, confidence=0.9),
        ]
        track.frame_indices = [0, 1, 2]

        self.detector._update_track_statistics()

        assert track.avg_confidence == (0.8 + 0.6 + 0.9) / 3
        assert track.persistence == 1.0  # 3 frames out of 3 total

    def test_select_primary_couple(self):
        """Test selection of primary dancing couple."""
        # Create multiple tracks with different qualities
        track1 = Mock()
        track1.avg_confidence = 0.9
        track1.persistence = 0.8

        track2 = Mock()
        track2.avg_confidence = 0.7
        track2.persistence = 0.9

        track3 = Mock()
        track3.avg_confidence = 0.5
        track3.persistence = 0.6

        self.detector.tracks = {0: track1, 1: track2, 2: track3}

        leader_id, follower_id = self.detector.select_primary_couple()

        # Should select the two best tracks
        assert leader_id in [0, 1]
        assert follower_id in [0, 1]
        assert leader_id != follower_id

    def test_select_primary_couple_insufficient_tracks(self):
        """Test primary couple selection with insufficient tracks."""
        # Test with no tracks
        self.detector.tracks = {}
        leader_id, follower_id = self.detector.select_primary_couple()
        assert leader_id is None
        assert follower_id is None

        # Test with single track
        self.detector.tracks = {0: Mock()}
        leader_id, follower_id = self.detector.select_primary_couple()
        assert leader_id is None
        assert follower_id is None

    @patch("src.bachata_analyzer.pose_detector.mp.solutions.pose")
    def test_detect_poses(self, mock_pose_class):
        """Test pose detection from frame."""
        # Mock MediaPipe pose detection
        mock_pose = Mock()
        mock_pose_class.return_value = mock_pose

        # Mock pose processing results
        mock_results = Mock()
        mock_results.pose_landmarks = self.create_mock_pose_landmarks()
        mock_pose.process.return_value = mock_results

        # Re-initialize detector with mocked pose
        self.detector.pose = mock_pose

        frame = self.create_mock_frame()
        poses = self.detector.detect_poses(frame, frame_idx=0)

        assert len(poses) == 1
        assert isinstance(poses[0], PoseLandmarks)

    def test_temporal_smoothing_insufficient_history(self):
        """Test temporal smoothing with insufficient frame history."""
        # Create minimal history
        self.detector.frame_history = [[PoseLandmarks(keypoints={}, confidence=0.8)]]

        # Should not raise error
        self.detector.apply_temporal_smoothing()

    def test_smooth_pose(self):
        """Test individual pose smoothing."""
        pose = PoseLandmarks(
            keypoints={"nose": Keypoint(x=0.5, y=0.1, visibility=0.8)}, confidence=0.8
        )

        # Create history with similar poses
        self.detector.frame_history = [
            [
                PoseLandmarks(
                    keypoints={"nose": Keypoint(x=0.49, y=0.09, visibility=0.8)},
                    confidence=0.8,
                )
            ],
            [
                PoseLandmarks(
                    keypoints={"nose": Keypoint(x=0.5, y=0.1, visibility=0.8)},
                    confidence=0.8,
                )
            ],
            [
                PoseLandmarks(
                    keypoints={"nose": Keypoint(x=0.51, y=0.11, visibility=0.8)},
                    confidence=0.8,
                )
            ],
        ]

        smoothed_pose = self.detector._smooth_pose(pose, idx=1, window_length=3)

        assert isinstance(smoothed_pose, PoseLandmarks)
        assert "nose" in smoothed_pose.keypoints
        # Smoothed position should be close to original
        assert abs(smoothed_pose.keypoints["nose"].x - pose.keypoints["nose"].x) < 0.1

    def test_close(self):
        """Test resource cleanup."""
        # Mock the pose object
        self.detector.pose = Mock()
        self.detector.pose.close = Mock()

        self.detector.close()
        self.detector.pose.close.assert_called_once()
