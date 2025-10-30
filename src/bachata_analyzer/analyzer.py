"""Main Bachata dance analyzer."""

import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import uuid
from urllib.parse import urlparse

from .config import AnalysisConfig
from .models import AnalysisResult, PersonTrack
from .video_processor import VideoProcessor
from .pose_detector import PoseDetector
from .segmentation import DanceSegmenter
from .output_generator import OutputGenerator


class BachataAnalyzer:
    """Main analyzer for Bachata dance combinations."""

    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.video_processor = VideoProcessor(self.config)
        self.pose_detector = PoseDetector(self.config)
        self.segmenter = DanceSegmenter(self.config)
        self.output_generator = OutputGenerator(self.config)

    def analyze(
        self, video_path: str, output_dir: Optional[Path] = None
    ) -> AnalysisResult:
        """Analyze a video for Bachata dance combinations."""
        if output_dir is None:
            output_dir = self.config.output_dir

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading video: {video_path}")
        self.video_processor.load_video(video_path)

        # Generate video ID
        video_id = self._generate_video_id(video_path)

        print("Detecting poses and tracking people...")
        tracks = self._process_video()

        if not tracks:
            print("No poses detected in video")
            return self._create_empty_result(video_path, video_id)

        print("Selecting primary dancing couple...")
        leader_track_id, follower_track_id = self.pose_detector.select_primary_couple()

        # Filter to only primary couple
        if leader_track_id is not None and follower_track_id is not None:
            primary_tracks = {
                leader_track_id: tracks[leader_track_id],
                follower_track_id: tracks[follower_track_id],
            }
        else:
            primary_tracks = tracks

        print("Extracting features and segmenting dance...")
        features = self.segmenter.extract_features(primary_tracks, self.config.fps)

        if len(features) == 0:
            print("No features extracted")
            return self._create_empty_result(video_path, video_id)

        change_points = self.segmenter.detect_change_points()
        segments = self.segmenter.create_segments(
            change_points, self.config.fps, primary_tracks
        )

        # Identify leader/follower roles
        roles = self.segmenter.identify_roles(primary_tracks)
        for segment in segments:
            if segment.roles["leader_track"] in roles:
                segment.leader_name = roles[segment.roles["leader_track"]]
            if segment.roles["follower_track"] in roles:
                segment.follower_name = roles[segment.roles["follower_track"]]

        print(f"Found {len(segments)} dance combinations")

        # Create analysis result
        result = AnalysisResult(
            video_id=video_id,
            video_path=video_path,
            fps=self.config.fps,
            total_frames=len(features),
            duration_sec=self.video_processor.get_duration(),
            segments=segments,
            tracks=list(primary_tracks.values()),
            config=self.config.dict(),
        )

        # Generate outputs
        print("Generating outputs...")
        self._generate_outputs(result, video_path, output_dir, primary_tracks)

        return result

    def _process_video(self) -> Dict[int, PersonTrack]:
        """Process video and detect poses."""
        frame_idx = 0

        for frame_idx, frame in self.video_processor.get_frame_generator():
            poses = self.pose_detector.detect_poses(frame, frame_idx)
            self.pose_detector.update_tracking(poses, frame_idx)

        # Apply temporal smoothing
        self.pose_detector.apply_temporal_smoothing()

        return self.pose_detector.tracks

    def _generate_video_id(self, video_path: str) -> str:
        """Generate a unique ID for the video."""
        if video_path.startswith(("http://", "https://")):
            # Extract YouTube video ID if possible
            parsed = urlparse(video_path)
            if "youtu.be" in parsed.netloc:
                return parsed.path.strip("/")
            elif "youtube.com" in parsed.netloc:
                from urllib.parse import parse_qs

                query = parse_qs(parsed.query)
                return query.get("v", ["unknown"])[0]

        # For local files, use filename + UUID
        filename = Path(video_path).stem
        return f"{filename}_{str(uuid.uuid4())[:8]}"

    def _create_empty_result(self, video_path: str, video_id: str) -> AnalysisResult:
        """Create an empty result when no poses are detected."""
        return AnalysisResult(
            video_id=video_id,
            video_path=video_path,
            fps=self.config.fps,
            total_frames=0,
            duration_sec=0.0,
            segments=[],
            tracks=[],
            config=self.config.dict(),
        )

    def _generate_outputs(
        self,
        result: AnalysisResult,
        video_path: str,
        output_dir: Path,
        tracks: Dict[int, PersonTrack],
    ):
        """Generate all output files."""
        # JSON output
        json_path = output_dir / "segments.json"
        self.output_generator.save_json(result, json_path)
        print(f"Saved JSON: {json_path}")

        # Text summary
        summary_path = output_dir / "summary.md"
        self.output_generator.save_text_summary(result, summary_path)
        print(f"Saved summary: {summary_path}")

        # Annotated video (if enabled)
        if self.config.create_video:
            video_output_path = output_dir / "annotated.mp4"
            success = self.output_generator.create_annotated_video(
                result, video_path, video_output_path, tracks
            )
            if success:
                print(f"Saved annotated video: {video_output_path}")
            else:
                print("Failed to create annotated video")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.video_processor.release()
        self.pose_detector.close()
