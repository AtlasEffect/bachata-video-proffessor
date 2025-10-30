"""Dance segmentation and feature extraction."""

import numpy as np
from typing import List, Tuple, Dict
from scipy.signal import find_peaks
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from .config import AnalysisConfig
from .models import PersonTrack, DanceSegment, SegmentFeatures


class DanceSegmenter:
    """Segments dance video into individual combinations."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.features = []
        self.segments = []

    def extract_features(self, tracks: Dict[int, PersonTrack], fps: int) -> np.ndarray:
        """Extract motion features for segmentation."""
        if len(tracks) == 0:
            return np.array([])

        # Get the two primary tracks (leader and follower)
        track_ids = list(tracks.keys())
        leader_track = tracks[track_ids[0]]
        follower_track = tracks[track_ids[1]] if len(track_ids) > 1 else None

        features = []

        # Process each frame
        for frame_idx in range(len(leader_track.landmarks_history)):
            leader_pose = leader_track.landmarks_history[frame_idx]
            follower_pose = (
                follower_track.landmarks_history[frame_idx]
                if follower_track and frame_idx < len(follower_track.landmarks_history)
                else None
            )

            frame_features = self._extract_frame_features(
                leader_pose, follower_pose, frame_idx, fps
            )
            features.append(frame_features)

        self.features = np.array(features)
        return self.features

    def _extract_frame_features(
        self, leader_pose, follower_pose, frame_idx: int, fps: int
    ) -> List[float]:
        """Extract features for a single frame."""
        features = []

        if not leader_pose:
            return [0.0] * 10  # Return zeros if no pose

        # 1. Joint velocities (speed)
        velocity = self._calculate_velocity(leader_pose, frame_idx)
        features.append(velocity)

        # 2. Torso rotation
        torso_rotation = self._calculate_torso_rotation(leader_pose)
        features.append(torso_rotation)

        # 3. Hand distance (between partners)
        hand_distance = 0.0
        if follower_pose:
            hand_distance = self._calculate_hand_distance(leader_pose, follower_pose)
        features.append(hand_distance)

        # 4. Step cadence (leg movement)
        step_cadence = self._calculate_step_cadence(leader_pose)
        features.append(step_cadence)

        # 5. Vertical movement (bounces, dips)
        vertical_movement = self._calculate_vertical_movement(leader_pose)
        features.append(vertical_movement)

        # 6. Arm spread
        arm_spread = self._calculate_arm_spread(leader_pose)
        features.append(arm_spread)

        # 7. Hip movement
        hip_movement = self._calculate_hip_movement(leader_pose)
        features.append(hip_movement)

        # 8. Turn detection
        turn_indicator = self._detect_turn(leader_pose, frame_idx)
        features.append(turn_indicator)

        # 9. Dip detection
        dip_indicator = self._detect_dip(leader_pose, frame_idx)
        features.append(dip_indicator)

        # 10. Freeze/pause detection
        freeze_indicator = self._detect_freeze(leader_pose, frame_idx)
        features.append(freeze_indicator)

        return features

    def _calculate_velocity(self, pose, frame_idx: int) -> float:
        """Calculate overall movement velocity."""
        if frame_idx == 0 or not hasattr(self, "_prev_keypoints"):
            self._prev_keypoints = pose.keypoints
            return 0.0

        total_distance = 0.0
        count = 0

        for joint_name, current_kp in pose.keypoints.items():
            if joint_name in self._prev_keypoints:
                prev_kp = self._prev_keypoints[joint_name]
                distance = np.sqrt(
                    (current_kp.x - prev_kp.x) ** 2 + (current_kp.y - prev_kp.y) ** 2
                )
                total_distance += distance
                count += 1

        self._prev_keypoints = pose.keypoints
        return total_distance / count if count > 0 else 0.0

    def _calculate_torso_rotation(self, pose) -> float:
        """Calculate torso rotation angle."""
        try:
            left_shoulder = pose.keypoints["left_shoulder"]
            right_shoulder = pose.keypoints["right_shoulder"]
            left_hip = pose.keypoints["left_hip"]
            right_hip = pose.keypoints["right_hip"]

            # Calculate shoulder and hip vectors
            shoulder_vector = np.array(
                [right_shoulder.x - left_shoulder.x, right_shoulder.y - left_shoulder.y]
            )
            hip_vector = np.array([right_hip.x - left_hip.x, right_hip.y - left_hip.y])

            # Calculate angle between vectors
            cos_angle = np.dot(shoulder_vector, hip_vector) / (
                np.linalg.norm(shoulder_vector) * np.linalg.norm(hip_vector) + 1e-8
            )
            angle = np.arccos(np.clip(cos_angle, -1, 1))

            return float(angle)
        except KeyError:
            return 0.0

    def _calculate_hand_distance(self, leader_pose, follower_pose) -> float:
        """Calculate distance between partners' hands."""
        try:
            # Use right hand of leader and left hand of follower (typical dance hold)
            leader_hand = leader_pose.keypoints["right_wrist"]
            follower_hand = follower_pose.keypoints["left_wrist"]

            distance = np.sqrt(
                (leader_hand.x - follower_hand.x) ** 2
                + (leader_hand.y - follower_hand.y) ** 2
            )
            return float(distance)
        except KeyError:
            return 0.0

    def _calculate_step_cadence(self, pose) -> float:
        """Calculate leg movement cadence."""
        try:
            left_ankle = pose.keypoints["left_ankle"]
            right_ankle = pose.keypoints["right_ankle"]
            left_knee = pose.keypoints["left_knee"]
            right_knee = pose.keypoints["right_knee"]

            # Calculate leg angles
            left_leg_angle = self._calculate_joint_angle(
                left_hip=pose.keypoints["left_hip"],
                left_knee=left_knee,
                left_ankle=left_ankle,
            )
            right_leg_angle = self._calculate_joint_angle(
                left_hip=pose.keypoints["right_hip"],
                left_knee=right_knee,
                left_ankle=right_ankle,
            )

            # Cadence is the variation in leg angles
            cadence = abs(left_leg_angle - right_leg_angle)
            return float(cadence)
        except KeyError:
            return 0.0

    def _calculate_joint_angle(self, left_hip, left_knee, left_ankle) -> float:
        """Calculate angle at knee joint."""
        # Vector from knee to hip
        v1 = np.array([left_hip.x - left_knee.x, left_hip.y - left_knee.y])
        # Vector from knee to ankle
        v2 = np.array([left_ankle.x - left_knee.x, left_ankle.y - left_knee.y])

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        return float(angle)

    def _calculate_vertical_movement(self, pose) -> float:
        """Calculate vertical movement (y-axis changes)."""
        try:
            # Use hip center as reference point
            left_hip = pose.keypoints["left_hip"]
            right_hip = pose.keypoints["right_hip"]
            hip_center_y = (left_hip.y + right_hip.y) / 2

            if not hasattr(self, "_prev_hip_y"):
                self._prev_hip_y = hip_center_y
                return 0.0

            vertical_change = abs(hip_center_y - self._prev_hip_y)
            self._prev_hip_y = hip_center_y

            return float(vertical_change)
        except KeyError:
            return 0.0

    def _calculate_arm_spread(self, pose) -> float:
        """Calculate distance between hands."""
        try:
            left_wrist = pose.keypoints["left_wrist"]
            right_wrist = pose.keypoints["right_wrist"]

            distance = np.sqrt(
                (left_wrist.x - right_wrist.x) ** 2
                + (left_wrist.y - right_wrist.y) ** 2
            )
            return float(distance)
        except KeyError:
            return 0.0

    def _calculate_hip_movement(self, pose) -> float:
        """Calculate hip movement."""
        try:
            left_hip = pose.keypoints["left_hip"]
            right_hip = pose.keypoints["right_hip"]

            if not hasattr(self, "_prev_left_hip"):
                self._prev_left_hip = left_hip
                self._prev_right_hip = right_hip
                return 0.0

            # Calculate movement of hip center
            current_hip_center = np.array(
                [(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2]
            )
            prev_hip_center = np.array(
                [
                    (self._prev_left_hip.x + self._prev_right_hip.x) / 2,
                    (self._prev_left_hip.y + self._prev_right_hip.y) / 2,
                ]
            )

            movement = np.linalg.norm(current_hip_center - prev_hip_center)

            self._prev_left_hip = left_hip
            self._prev_right_hip = right_hip

            return float(movement)
        except KeyError:
            return 0.0

    def _detect_turn(self, pose, frame_idx: int) -> float:
        """Detect if person is turning."""
        try:
            if frame_idx == 0 or not hasattr(self, "_prev_shoulder_angle"):
                # Store initial shoulder orientation
                left_shoulder = pose.keypoints["left_shoulder"]
                right_shoulder = pose.keypoints["right_shoulder"]
                self._prev_shoulder_angle = np.arctan2(
                    right_shoulder.y - left_shoulder.y,
                    right_shoulder.x - left_shoulder.x,
                )
                return 0.0

            # Calculate current shoulder orientation
            left_shoulder = pose.keypoints["left_shoulder"]
            right_shoulder = pose.keypoints["right_shoulder"]
            current_angle = np.arctan2(
                right_shoulder.y - left_shoulder.y, right_shoulder.x - left_shoulder.x
            )

            # Calculate angle change
            angle_change = abs(current_angle - self._prev_shoulder_angle)
            self._prev_shoulder_angle = current_angle

            # Return 1.0 if significant turn detected
            return 1.0 if angle_change > 0.3 else 0.0
        except KeyError:
            return 0.0

    def _detect_dip(self, pose, frame_idx: int) -> float:
        """Detect if person is performing a dip."""
        try:
            # Check for significant vertical drop
            vertical_movement = self._calculate_vertical_movement(pose)

            # Also check arm position (often raised during dips)
            arm_spread = self._calculate_arm_spread(pose)

            # Dip is detected with significant vertical movement and arm spread
            return 1.0 if vertical_movement > 0.05 and arm_spread > 0.3 else 0.0
        except:
            return 0.0

    def _detect_freeze(self, pose, frame_idx: int) -> float:
        """Detect if person is frozen/paused."""
        velocity = self._calculate_velocity(pose, frame_idx)
        return 1.0 if velocity < self.config.pause_threshold else 0.0

    def detect_change_points(self) -> List[int]:
        """Detect change points in the feature sequence."""
        if len(self.features) == 0:
            return []

        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(self.features)

        # Calculate overall activity score
        activity_scores = np.mean(features_normalized, axis=1)

        # Find peaks in activity (potential change points)
        peaks, _ = find_peaks(
            activity_scores,
            height=np.percentile(activity_scores, 75),
            distance=int(
                self.config.min_segment_sec * 12
            ),  # Minimum distance between segments
        )

        # Add start and end points
        all_change_points = [0] + list(peaks) + [len(self.features) - 1]

        # Filter very short segments
        filtered_points = []
        for i in range(len(all_change_points) - 1):
            segment_length = all_change_points[i + 1] - all_change_points[i]
            if segment_length >= int(self.config.min_segment_sec * 12):
                filtered_points.append(all_change_points[i])

        filtered_points.append(all_change_points[-1])  # Always include end

        return filtered_points

    def create_segments(
        self, change_points: List[int], fps: int, tracks: Dict[int, PersonTrack]
    ) -> List[DanceSegment]:
        """Create dance segments from change points."""
        segments = []
        track_ids = list(tracks.keys())

        for i in range(len(change_points) - 1):
            start_frame = change_points[i]
            end_frame = change_points[i + 1]

            # Calculate segment features
            segment_features = self._calculate_segment_features(start_frame, end_frame)

            # Create segment
            segment = DanceSegment(
                id=i + 1,
                start_sec=start_frame / fps,
                end_sec=end_frame / fps,
                start_frame=start_frame,
                end_frame=end_frame,
                roles={
                    "leader_track": track_ids[0],
                    "follower_track": track_ids[1] if len(track_ids) > 1 else -1,
                },
                leader_name="Leader",
                follower_name="Follower",
                features=segment_features,
                tentative_name=f"Combo {i + 1}",
            )

            segments.append(segment)

        self.segments = segments
        return segments

    def _calculate_segment_features(
        self, start_frame: int, end_frame: int
    ) -> SegmentFeatures:
        """Calculate features for a segment."""
        if start_frame >= len(self.features) or end_frame > len(self.features):
            return SegmentFeatures()

        segment_features = self.features[start_frame:end_frame]

        # Calculate averages and statistics
        avg_speed = (
            np.mean(segment_features[:, 0]) if len(segment_features) > 0 else 0.0
        )
        turns = (
            np.any(segment_features[:, 7] > 0.5) if len(segment_features) > 0 else False
        )
        dip = (
            np.any(segment_features[:, 8] > 0.5) if len(segment_features) > 0 else False
        )
        hand_distance_avg = (
            np.mean(segment_features[:, 2]) if len(segment_features) > 0 else 0.0
        )
        torso_rotation_avg = (
            np.mean(segment_features[:, 1]) if len(segment_features) > 0 else 0.0
        )
        step_cadence = (
            np.mean(segment_features[:, 3]) if len(segment_features) > 0 else 0.0
        )
        freeze_frames = (
            int(np.sum(segment_features[:, 9] > 0.5))
            if len(segment_features) > 0
            else 0
        )
        total_frames = len(segment_features)

        return SegmentFeatures(
            avg_speed=float(avg_speed),
            turns=bool(turns),
            dip=bool(dip),
            hand_distance_avg=float(hand_distance_avg),
            torso_rotation_avg=float(torso_rotation_avg),
            step_cadence=float(step_cadence),
            freeze_frames=freeze_frames,
            total_frames=total_frames,
        )

    def identify_roles(self, tracks: Dict[int, PersonTrack]) -> Dict[int, str]:
        """Identify leader and follower roles based on movement patterns."""
        if len(tracks) < 2:
            return {}

        track_ids = list(tracks.keys())
        track1, track2 = tracks[track_ids[0]], tracks[track_ids[1]]

        # Analyze movement patterns to determine leader
        track1_leadership_score = self._calculate_leadership_score(track1)
        track2_leadership_score = self._calculate_leadership_score(track2)

        roles = {}
        if track1_leadership_score > track2_leadership_score:
            roles[track_ids[0]] = "Leader"
            roles[track_ids[1]] = "Follower"
        else:
            roles[track_ids[0]] = "Follower"
            roles[track_ids[1]] = "Leader"

        return roles

    def _calculate_leadership_score(self, track: PersonTrack) -> float:
        """Calculate leadership score based on movement patterns."""
        if not track.landmarks_history:
            return 0.0

        # Leaders typically initiate movements and have more controlled motion
        total_velocity = 0.0
        total_turns = 0
        total_frames = len(track.landmarks_history)

        for i, pose in enumerate(track.landmarks_history):
            # Velocity contributes to leadership (initiating movement)
            velocity = self._calculate_velocity(pose, i)
            total_velocity += velocity

            # Turns also indicate leadership
            turn_indicator = self._detect_turn(pose, i)
            total_turns += turn_indicator

        # Leadership score combines velocity and turn initiation
        leadership_score = (total_velocity / total_frames) + (
            total_turns / total_frames
        ) * 0.5

        return leadership_score
