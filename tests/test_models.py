"""Tests for data models."""

import pytest
from pydantic import ValidationError

from src.bachata_analyzer.models import (
    Keypoint,
    PoseLandmarks,
    PersonTrack,
    SegmentFeatures,
    DanceSegment,
    AnalysisResult,
)


class TestKeypoint:
    """Test cases for Keypoint model."""

    def test_keypoint_creation(self):
        """Test keypoint creation with all parameters."""
        kp = Keypoint(x=0.5, y=0.3, z=0.1, visibility=0.8)

        assert kp.x == 0.5
        assert kp.y == 0.3
        assert kp.z == 0.1
        assert kp.visibility == 0.8

    def test_keypoint_defaults(self):
        """Test keypoint creation with default values."""
        kp = Keypoint(x=0.5, y=0.3)

        assert kp.x == 0.5
        assert kp.y == 0.3
        assert kp.z == 0.0
        assert kp.visibility == 1.0

    def test_keypoint_validation(self):
        """Test keypoint parameter validation."""
        # Should accept any numeric values
        kp = Keypoint(x=-1.0, y=2.0, z=0.5, visibility=0.0)
        assert kp.x == -1.0
        assert kp.y == 2.0


class TestPoseLandmarks:
    """Test cases for PoseLandmarks model."""

    def test_pose_landmarks_creation(self):
        """Test pose landmarks creation."""
        keypoints = {
            "nose": Keypoint(x=0.5, y=0.1),
            "left_shoulder": Keypoint(x=0.4, y=0.3),
        }

        pose = PoseLandmarks(keypoints=keypoints, confidence=0.8)

        assert len(pose.keypoints) == 2
        assert pose.confidence == 0.8
        assert "nose" in pose.keypoints
        assert pose.keypoints["nose"].x == 0.5

    def test_pose_landmarks_with_bbox(self):
        """Test pose landmarks with bounding box."""
        keypoints = {"nose": Keypoint(x=0.5, y=0.1)}
        bbox = (0.3, 0.0, 0.4, 0.8)

        pose = PoseLandmarks(keypoints=keypoints, confidence=0.9, bbox=bbox)

        assert pose.bbox == bbox
        assert len(pose.bbox) == 4

    def test_pose_landmarks_defaults(self):
        """Test pose landmarks with default values."""
        pose = PoseLandmarks(keypoints={}, confidence=0.5)

        assert len(pose.keypoints) == 0
        assert pose.confidence == 0.5
        assert pose.bbox is None


class TestPersonTrack:
    """Test cases for PersonTrack model."""

    def test_person_track_creation(self):
        """Test person track creation."""
        track = PersonTrack(track_id=5)

        assert track.track_id == 5
        assert len(track.landmarks_history) == 0
        assert len(track.frame_indices) == 0
        assert track.avg_confidence == 0.0
        assert track.persistence == 0.0

    def test_person_track_with_history(self):
        """Test person track with landmarks history."""
        keypoints = {"nose": Keypoint(x=0.5, y=0.1)}
        pose = PoseLandmarks(keypoints=keypoints, confidence=0.8)

        track = PersonTrack(
            track_id=1,
            landmarks_history=[pose],
            frame_indices=[0, 1],
            avg_confidence=0.7,
            persistence=0.9,
        )

        assert track.track_id == 1
        assert len(track.landmarks_history) == 1
        assert track.frame_indices == [0, 1]
        assert track.avg_confidence == 0.7
        assert track.persistence == 0.9


class TestSegmentFeatures:
    """Test cases for SegmentFeatures model."""

    def test_segment_features_creation(self):
        """Test segment features creation."""
        features = SegmentFeatures(
            avg_speed=0.05,
            turns=True,
            dip=False,
            hand_distance_avg=0.3,
            torso_rotation_avg=0.2,
            step_cadence=0.1,
            freeze_frames=5,
            total_frames=60,
        )

        assert features.avg_speed == 0.05
        assert features.turns == True
        assert features.dip == False
        assert features.hand_distance_avg == 0.3
        assert features.torso_rotation_avg == 0.2
        assert features.step_cadence == 0.1
        assert features.freeze_frames == 5
        assert features.total_frames == 60

    def test_segment_features_defaults(self):
        """Test segment features with default values."""
        features = SegmentFeatures()

        assert features.avg_speed == 0.0
        assert features.turns == False
        assert features.dip == False
        assert features.hand_distance_avg == 0.0
        assert features.torso_rotation_avg == 0.0
        assert features.step_cadence == 0.0
        assert features.freeze_frames == 0
        assert features.total_frames == 0


class TestDanceSegment:
    """Test cases for DanceSegment model."""

    def test_dance_segment_creation(self):
        """Test dance segment creation."""
        features = SegmentFeatures(avg_speed=0.05, turns=True)
        roles = {"leader_track": 0, "follower_track": 1}

        segment = DanceSegment(
            id=1,
            start_sec=0.0,
            end_sec=4.0,
            start_frame=0,
            end_frame=48,
            roles=roles,
            features=features,
            tentative_name="Test Combo",
        )

        assert segment.id == 1
        assert segment.start_sec == 0.0
        assert segment.end_sec == 4.0
        assert segment.start_frame == 0
        assert segment.end_frame == 48
        assert segment.roles == roles
        assert segment.features == features
        assert segment.tentative_name == "Test Combo"
        assert segment.leader_name == "Leader"
        assert segment.follower_name == "Follower"

    def test_dance_segment_defaults(self):
        """Test dance segment with default values."""
        features = SegmentFeatures()
        roles = {"leader_track": 0, "follower_track": 1}

        segment = DanceSegment(
            id=2,
            start_sec=5.0,
            end_sec=9.0,
            start_frame=60,
            end_frame=108,
            roles=roles,
            features=features,
        )

        assert segment.leader_name == "Leader"
        assert segment.follower_name == "Follower"
        assert segment.tentative_name == ""


class TestAnalysisResult:
    """Test cases for AnalysisResult model."""

    def test_analysis_result_creation(self):
        """Test analysis result creation."""
        segments = [
            DanceSegment(
                id=1,
                start_sec=0.0,
                end_sec=4.0,
                start_frame=0,
                end_frame=48,
                roles={"leader_track": 0, "follower_track": 1},
                features=SegmentFeatures(),
            )
        ]

        tracks = [
            PersonTrack(track_id=0, avg_confidence=0.8, persistence=0.9),
            PersonTrack(track_id=1, avg_confidence=0.7, persistence=0.8),
        ]

        result = AnalysisResult(
            video_id="test_video",
            video_path="/path/to/video.mp4",
            fps=12,
            total_frames=300,
            duration_sec=25.0,
            segments=segments,
            tracks=tracks,
            config={"fps": 12, "create_video": True},
        )

        assert result.video_id == "test_video"
        assert result.video_path == "/path/to/video.mp4"
        assert result.fps == 12
        assert result.total_frames == 300
        assert result.duration_sec == 25.0
        assert len(result.segments) == 1
        assert len(result.tracks) == 2

    def test_get_segment_count(self):
        """Test segment count calculation."""
        segments = [
            DanceSegment(
                id=1,
                start_sec=0.0,
                end_sec=4.0,
                start_frame=0,
                end_frame=48,
                roles={"leader_track": 0, "follower_track": 1},
                features=SegmentFeatures(),
            ),
            DanceSegment(
                id=2,
                start_sec=4.0,
                end_sec=8.0,
                start_frame=48,
                end_frame=96,
                roles={"leader_track": 0, "follower_track": 1},
                features=SegmentFeatures(),
            ),
        ]

        result = AnalysisResult(
            video_id="test",
            video_path="test.mp4",
            fps=12,
            total_frames=96,
            duration_sec=8.0,
            segments=segments,
            tracks=[],
            config={},
        )

        assert result.get_segment_count() == 2

    def test_get_total_dance_time(self):
        """Test total dance time calculation."""
        segments = [
            DanceSegment(
                id=1,
                start_sec=0.0,
                end_sec=4.0,
                start_frame=0,
                end_frame=48,
                roles={"leader_track": 0, "follower_track": 1},
                features=SegmentFeatures(),
            ),
            DanceSegment(
                id=2,
                start_sec=5.0,
                end_sec=9.0,
                start_frame=60,
                end_frame=108,
                roles={"leader_track": 0, "follower_track": 1},
                features=SegmentFeatures(),
            ),
        ]

        result = AnalysisResult(
            video_id="test",
            video_path="test.mp4",
            fps=12,
            total_frames=108,
            duration_sec=10.0,
            segments=segments,
            tracks=[],
            config={},
        )

        # Should be 4.0 + 4.0 = 8.0 seconds
        assert result.get_total_dance_time() == 8.0

    def test_get_total_dance_time_empty_segments(self):
        """Test total dance time with no segments."""
        result = AnalysisResult(
            video_id="test",
            video_path="test.mp4",
            fps=12,
            total_frames=0,
            duration_sec=0.0,
            segments=[],
            tracks=[],
            config={},
        )

        assert result.get_total_dance_time() == 0.0
