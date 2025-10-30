"""Output generation for analysis results."""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import mediapipe as mp

from .config import AnalysisConfig
from .models import AnalysisResult, DanceSegment, PersonTrack, PoseLandmarks
from .pose_detector import PoseDetector


class OutputGenerator:
    """Generates various output formats from analysis results."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def save_json(self, result: AnalysisResult, output_path: Path):
        """Save analysis results as JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict for JSON serialization
        result_dict = {
            "video_id": result.video_id,
            "video_path": result.video_path,
            "fps": result.fps,
            "total_frames": result.total_frames,
            "duration_sec": result.duration_sec,
            "segments": [
                {
                    "id": seg.id,
                    "start_sec": seg.start_sec,
                    "end_sec": seg.end_sec,
                    "roles": seg.roles,
                    "leader_name": seg.leader_name,
                    "follower_name": seg.follower_name,
                    "features": {
                        "avg_speed": seg.features.avg_speed,
                        "turns": seg.features.turns,
                        "dip": seg.features.dip,
                        "hand_distance_avg": seg.features.hand_distance_avg,
                        "torso_rotation_avg": seg.features.torso_rotation_avg,
                        "step_cadence": seg.features.step_cadence,
                        "freeze_frames": seg.features.freeze_frames,
                        "total_frames": seg.features.total_frames,
                    },
                    "tentative_name": seg.tentative_name,
                }
                for seg in result.segments
            ],
        }

        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=2)

    def save_text_summary(self, result: AnalysisResult, output_path: Path):
        """Save human-readable text summary."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write("# Bachata Dance Analysis Summary\n\n")
            f.write(f"Video: {result.video_id}\n")
            f.write(f"Duration: {result.duration_sec:.1f} seconds\n")
            f.write(f"FPS: {result.fps}\n")
            f.write(f"Total combinations detected: {len(result.segments)}\n\n")

            f.write("## Dance Combinations\n\n")

            for i, segment in enumerate(result.segments, 1):
                f.write(f"### {segment.tentative_name}\n")
                f.write(
                    f"- **Time**: {segment.start_sec:.1f}s - {segment.end_sec:.1f}s "
                    f"({segment.end_sec - segment.start_sec:.1f}s)\n"
                )
                f.write(f"- **Leader**: {segment.leader_name}\n")
                f.write(f"- **Follower**: {segment.follower_name}\n")

                # Feature highlights
                features = segment.features
                feature_descs = []

                if features.turns:
                    feature_descs.append("turns")
                if features.dip:
                    feature_descs.append("dip")
                if features.avg_speed > 0.02:
                    feature_descs.append("fast-paced")
                elif features.avg_speed < 0.01:
                    feature_descs.append("slow-paced")

                if feature_descs:
                    f.write(f"- **Features**: {', '.join(feature_descs)}\n")

                f.write(f"- **Average speed**: {features.avg_speed:.4f}\n")
                f.write(f"- **Hand distance**: {features.hand_distance_avg:.4f}\n")
                f.write(f"- **Step cadence**: {features.step_cadence:.4f}\n")

                if features.freeze_frames > 0:
                    freeze_percentage = (
                        features.freeze_frames / features.total_frames
                    ) * 100
                    f.write(f"- **Pauses**: {freeze_percentage:.1f}% of segment\n")

                f.write("\n")

            # Summary statistics
            f.write("## Summary Statistics\n\n")
            if result.segments:
                total_dance_time = sum(
                    seg.end_sec - seg.start_sec for seg in result.segments
                )
                avg_combo_length = total_dance_time / len(result.segments)

                f.write(f"- Total dance time: {total_dance_time:.1f}s\n")
                f.write(f"- Average combination length: {avg_combo_length:.1f}s\n")
                f.write(
                    f"- Dance coverage: {(total_dance_time / result.duration_sec) * 100:.1f}%\n"
                )

                # Count features
                combos_with_turns = sum(
                    1 for seg in result.segments if seg.features.turns
                )
                combos_with_dips = sum(1 for seg in result.segments if seg.features.dip)

                f.write(
                    f"- Combinations with turns: {combos_with_turns}/{len(result.segments)}\n"
                )
                f.write(
                    f"- Combinations with dips: {combos_with_dips}/{len(result.segments)}\n"
                )

    def create_annotated_video(
        self,
        result: AnalysisResult,
        video_path: str,
        output_path: Path,
        tracks: Dict[int, PersonTrack],
    ) -> bool:
        """Create annotated video with skeleton overlays and segment labels."""
        if not self.config.create_video:
            return False

        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calculate frame sampling
            frame_interval = max(1, int(fps / result.fps))

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(output_path), fourcc, result.fps, (width, height))

            # Get primary couple track IDs
            track_ids = list(tracks.keys())
            leader_track_id = track_ids[0] if len(track_ids) > 0 else None
            follower_track_id = track_ids[1] if len(track_ids) > 1 else None

            frame_idx = 0
            output_frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    # Draw annotations
                    annotated_frame = self._draw_frame_annotations(
                        frame,
                        output_frame_idx,
                        result,
                        tracks,
                        leader_track_id,
                        follower_track_id,
                    )

                    out.write(annotated_frame)
                    output_frame_idx += 1

                frame_idx += 1

            cap.release()
            out.release()
            return True

        except Exception as e:
            print(f"Error creating annotated video: {e}")
            return False

    def _draw_frame_annotations(
        self,
        frame: np.ndarray,
        frame_idx: int,
        result: AnalysisResult,
        tracks: Dict[int, PersonTrack],
        leader_track_id: Optional[int],
        follower_track_id: Optional[int],
    ) -> np.ndarray:
        """Draw pose skeletons and segment labels on a frame."""
        annotated_frame = frame.copy()

        # Find current segment
        current_segment = None
        current_time = frame_idx / result.fps

        for segment in result.segments:
            if segment.start_frame <= frame_idx <= segment.end_frame:
                current_segment = segment
                break

        # Draw segment label
        if current_segment:
            label = f"{current_segment.tentative_name} ({current_time:.1f}s)"
            cv2.putText(
                annotated_frame,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        # Draw pose skeletons for primary couple
        if leader_track_id is not None and leader_track_id in tracks:
            leader_track = tracks[leader_track_id]
            if frame_idx < len(leader_track.landmarks_history):
                pose = leader_track.landmarks_history[frame_idx]
                self._draw_pose(annotated_frame, pose, (0, 255, 0), "Leader")

        if follower_track_id is not None and follower_track_id in tracks:
            follower_track = tracks[follower_track_id]
            if frame_idx < len(follower_track.landmarks_history):
                pose = follower_track.landmarks_history[frame_idx]
                self._draw_pose(annotated_frame, pose, (255, 0, 0), "Follower")

        # Draw other people in gray (if any)
        for track_id, track in tracks.items():
            if track_id not in [leader_track_id, follower_track_id]:
                if frame_idx < len(track.landmarks_history):
                    pose = track.landmarks_history[frame_idx]
                    self._draw_pose(annotated_frame, pose, (128, 128, 128), "")

        return annotated_frame

    def _draw_pose(
        self, frame: np.ndarray, pose: PoseLandmarks, color: tuple, label: str
    ):
        """Draw a single pose skeleton on the frame."""
        if not pose.keypoints:
            return

        # Create a simple skeleton drawing using OpenCV
        # Define connections between keypoints
        connections = [
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_hip", "left_knee"),
            ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"),
            ("right_knee", "right_ankle"),
        ]

        # Draw connections
        for start_joint, end_joint in connections:
            if start_joint in pose.keypoints and end_joint in pose.keypoints:
                start_kp = pose.keypoints[start_joint]
                end_kp = pose.keypoints[end_joint]

                # Convert normalized coordinates to pixel coordinates
                start_pt = (
                    int(start_kp.x * frame.shape[1]),
                    int(start_kp.y * frame.shape[0]),
                )
                end_pt = (
                    int(end_kp.x * frame.shape[1]),
                    int(end_kp.y * frame.shape[0]),
                )

                cv2.line(frame, start_pt, end_pt, color, 2)

        # Draw keypoints
        for joint_name, kp in pose.keypoints.items():
            if kp.visibility > 0.5:  # Only draw visible keypoints
                pt = (int(kp.x * frame.shape[1]), int(kp.y * frame.shape[0]))
                cv2.circle(frame, pt, 3, color, -1)

        # Draw label if provided
        if label and pose.bbox:
            x, y, w, h = pose.bbox
            cv2.putText(
                frame,
                label,
                (int(x * frame.shape[1]), int(y * frame.shape[0]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
