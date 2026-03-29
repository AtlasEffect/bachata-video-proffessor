"""Pose detection and tracking using MediaPipe."""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING, Any
from collections import defaultdict

if TYPE_CHECKING:
    from scipy.signal import savgol_filter

from .config import AnalysisConfig
from .models import PoseLandmarks, Keypoint, PersonTrack


class PoseDetector:
    """MediaPipe-based pose detection and tracking."""

    # MediaPipe pose landmark names
    LANDMARK_NAMES = [
        "nose",
        "left_eye_inner",
        "left_eye",
        "left_eye_outer",
        "right_eye_inner",
        "right_eye",
        "right_eye_outer",
        "left_ear",
        "right_ear",
        "mouth_left",
        "mouth_right",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_pinky",
        "right_pinky",
        "left_index",
        "right_index",
        "left_thumb",
        "right_thumb",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    ]

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Use lighter model for CPU
            enable_segmentation=False,
            min_detection_confidence=config.pose_confidence,
            min_tracking_confidence=config.tracking_confidence,
        )

        self.tracks: Dict[int, PersonTrack] = {}
        self.next_track_id = 0
        self.frame_history: List[List[PoseLandmarks]] = []

    def detect_poses(self, frame: np.ndarray, frame_idx: int) -> List[PoseLandmarks]:
        """Detect poses in a frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        poses = []
        if results.pose_landmarks:
            landmarks = self._convert_landmarks(results.pose_landmarks)
            poses.append(landmarks)

        # Store in history for temporal smoothing
        self.frame_history.append(poses)
        if len(self.frame_history) > self.config.smoothing_window:
            self.frame_history.pop(0)

        return poses

    def _convert_landmarks(self, mp_landmarks: Any) -> PoseLandmarks:
        """Convert MediaPipe landmarks to our format."""
        keypoints = {}
        visibility_sum = 0
        keypoint_count = 0

        for i, name in enumerate(self.LANDMARK_NAMES):
            landmark = mp_landmarks.landmark[i]
            keypoints[name] = Keypoint(
                x=landmark.x, y=landmark.y, z=landmark.z, visibility=landmark.visibility
            )
            visibility_sum += landmark.visibility
            keypoint_count += 1

        # Calculate bounding box
        x_coords = [kp.x for kp in keypoints.values()]
        y_coords = [kp.y for kp in keypoints.values()]
        bbox = (
            min(x_coords),
            min(y_coords),
            max(x_coords) - min(x_coords),
            max(y_coords) - min(y_coords),
        )

        return PoseLandmarks(
            keypoints=keypoints, confidence=visibility_sum / keypoint_count, bbox=bbox
        )

    def update_tracking(self, poses: List[PoseLandmarks], frame_idx: int) -> None:
        """Update person tracking with current frame poses."""
        if not poses:
            return

        # Simple tracking: assign poses to existing tracks based on proximity
        assigned_tracks = set()

        for pose in poses:
            best_track_id = None
            best_distance = float("inf")

            # Find closest existing track
            for track_id, track in self.tracks.items():
                if track_id in assigned_tracks:
                    continue

                if track.landmarks_history:
                    last_pose = track.landmarks_history[-1]
                    distance = self._calculate_pose_distance(pose, last_pose)

                    if (
                        distance < best_distance and distance < 0.3
                    ):  # Threshold for matching
                        best_distance = distance
                        best_track_id = track_id

            if best_track_id is not None:
                # Update existing track
                self.tracks[best_track_id].landmarks_history.append(pose)
                self.tracks[best_track_id].frame_indices.append(frame_idx)
                assigned_tracks.add(best_track_id)
            else:
                # Create new track
                new_track = PersonTrack(
                    track_id=self.next_track_id,
                    landmarks_history=[pose],
                    frame_indices=[frame_idx],
                )
                self.tracks[self.next_track_id] = new_track
                assigned_tracks.add(self.next_track_id)
                self.next_track_id += 1

        # Update track statistics
        self._update_track_statistics()

    def _calculate_pose_distance(
        self, pose1: PoseLandmarks, pose2: PoseLandmarks
    ) -> float:
        """Calculate distance between two poses based on key body parts."""
        # Use key joints for matching (shoulders, hips)
        key_joints = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]

        total_distance = 0
        count = 0

        for joint in key_joints:
            if joint in pose1.keypoints and joint in pose2.keypoints:
                kp1 = pose1.keypoints[joint]
                kp2 = pose2.keypoints[joint]
                distance = np.sqrt((kp1.x - kp2.x) ** 2 + (kp1.y - kp2.y) ** 2)
                total_distance += distance
                count += 1

        return total_distance / count if count > 0 else float("inf")

    def _update_track_statistics(self) -> None:
        """Update track statistics like confidence and persistence."""
        total_frames = (
            max(len(track.frame_indices) for track in self.tracks.values())
            if self.tracks
            else 1
        )

        for track in self.tracks.values():
            # Average confidence
            if track.landmarks_history:
                track.avg_confidence = sum(
                    pose.confidence for pose in track.landmarks_history
                ) / len(track.landmarks_history)

            # Persistence (how consistently this track appears)
            track.persistence = len(track.frame_indices) / total_frames

    def select_primary_couple(self) -> Tuple[Optional[int], Optional[int]]:
        """Select the primary dancing couple from all tracks."""
        if len(self.tracks) < 2:
            return None, None

        # Sort tracks by combined confidence and persistence
        scored_tracks = []
        for track_id, track in self.tracks.items():
            score = track.avg_confidence * track.persistence
            scored_tracks.append((score, track_id, track))

        scored_tracks.sort(reverse=True)

        # Take top 2 tracks as the primary couple
        if len(scored_tracks) >= 2:
            return scored_tracks[0][1], scored_tracks[1][1]

        return None, None

    def apply_temporal_smoothing(self) -> None:
        """Apply temporal smoothing to all tracked poses."""
        if not self.config.use_temporal_smoothing or len(self.frame_history) < 3:
            return

        window_length = min(len(self.frame_history), self.config.smoothing_window)
        if window_length % 2 == 0:
            window_length -= 1  # Make odd for savgol_filter

        # Apply smoothing to each track
        for track in self.tracks.values():
            if len(track.landmarks_history) < window_length:
                continue

            smoothed_landmarks = []
            for i, pose in enumerate(track.landmarks_history):
                smoothed_pose = self._smooth_pose(pose, i, window_length)
                smoothed_landmarks.append(smoothed_pose)

            track.landmarks_history = smoothed_landmarks

    def _smooth_pose(
        self, pose: PoseLandmarks, idx: int, window_length: int
    ) -> PoseLandmarks:
        """Apply smoothing to a single pose using neighboring frames."""
        # Get window around current frame
        start_idx = max(0, idx - window_length // 2)
        end_idx = min(len(self.frame_history), idx + window_length // 2 + 1)

        smoothed_keypoints = {}

        for keypoint_name in pose.keypoints:
            # Collect values from window
            x_values = []
            y_values = []
            z_values = []

            for frame_idx in range(start_idx, end_idx):
                if frame_idx < len(self.frame_history):
                    frame_poses = self.frame_history[frame_idx]
                    for frame_pose in frame_poses:
                        if keypoint_name in frame_pose.keypoints:
                            kp = frame_pose.keypoints[keypoint_name]
                            x_values.append(kp.x)
                            y_values.append(kp.y)
                            z_values.append(kp.z)
                            break

            # Apply Savitzky-Golay filter
            if len(x_values) >= 3:
                x_smooth = savgol_filter(
                    x_values, min(len(x_values), window_length), 2
                )[len(x_values) // 2]
                y_smooth = savgol_filter(
                    y_values, min(len(y_values), window_length), 2
                )[len(y_values) // 2]
                z_smooth = savgol_filter(
                    z_values, min(len(z_values), window_length), 2
                )[len(z_values) // 2]

                original_kp = pose.keypoints[keypoint_name]
                smoothed_keypoints[keypoint_name] = Keypoint(
                    x=x_smooth,
                    y=y_smooth,
                    z=z_smooth,
                    visibility=original_kp.visibility,
                )
            else:
                smoothed_keypoints[keypoint_name] = pose.keypoints[keypoint_name]

        return PoseLandmarks(
            keypoints=smoothed_keypoints, confidence=pose.confidence, bbox=pose.bbox
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self, "pose"):
            self.pose.close()
