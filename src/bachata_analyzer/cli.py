"""Command-line interface for Bachata dance analysis."""

import argparse
import sys
from pathlib import Path

from .config import AnalysisConfig
from .analyzer import BachataAnalyzer


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Extract Bachata dance combinations from YouTube videos or local files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bachata-analyze "https://youtu.be/4V7EccGsSUI?si=m-hxgRejZcg-b1nD" --out output/
  bachata-analyze video.mp4 --out output/ --fps 12 --min-seg-sec 3
  bachata-analyze "https://youtu.be/OIEpCz8Q97A?si=k08QOEL5rkqALeDH" --out output/ --no-video
        """,
    )

    parser.add_argument("input", help="YouTube URL or local video file path")

    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output)",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="Frames per second for analysis (default: 12)",
    )

    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="Maximum video width (default: 1280)",
    )

    parser.add_argument(
        "--min-seg-sec",
        type=float,
        default=4.0,
        help="Minimum segment length in seconds (default: 4.0)",
    )

    parser.add_argument(
        "--no-video", action="store_true", help="Skip video output generation"
    )

    parser.add_argument(
        "--cache-dir", type=Path, help="Cache directory for downloads (default: cache)"
    )

    parser.add_argument(
        "--pose-confidence",
        type=float,
        default=0.5,
        help="Minimum pose confidence (default: 0.5)",
    )

    parser.add_argument(
        "--tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum tracking confidence (default: 0.5)",
    )

    parser.add_argument(
        "--change-point-sensitivity",
        type=float,
        default=0.8,
        help="Sensitivity for change point detection (default: 0.8)",
    )

    parser.add_argument(
        "--pause-threshold",
        type=float,
        default=0.3,
        help="Threshold for detecting pauses (default: 0.3)",
    )

    parser.add_argument(
        "--no-temporal-smoothing",
        action="store_true",
        help="Disable temporal smoothing of keypoints",
    )

    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Window size for temporal smoothing (default: 5)",
    )

    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Create configuration
    config = AnalysisConfig(
        fps=args.fps,
        max_width=args.max_width,
        min_segment_sec=args.min_seg_sec,
        create_video=not args.no_video,
        cache_dir=args.cache_dir,
        pose_confidence=args.pose_confidence,
        tracking_confidence=args.tracking_confidence,
        change_point_sensitivity=args.change_point_sensitivity,
        pause_threshold=args.pause_threshold,
        use_temporal_smoothing=not args.no_temporal_smoothing,
        smoothing_window=args.smoothing_window,
        output_dir=args.out,
    )

    # Validate input
    if not args.input.startswith(("http://", "https://")):
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file does not exist: {args.input}")
            sys.exit(1)

    try:
        # Run analysis
        with BachataAnalyzer(config) as analyzer:
            result = analyzer.analyze(args.input, args.out)

            # Print summary
            print(f"\nAnalysis complete!")
            print(f"Video: {result.video_id}")
            print(f"Duration: {result.duration_sec:.1f} seconds")
            print(f"Combinations detected: {len(result.segments)}")

            if result.segments:
                total_dance_time = sum(
                    seg.end_sec - seg.start_sec for seg in result.segments
                )
                print(f"Total dance time: {total_dance_time:.1f} seconds")
                print(
                    f"Dance coverage: {(total_dance_time / result.duration_sec) * 100:.1f}%"
                )

            print(f"\nOutput files saved to: {args.out}")
            print("- segments.json: Detailed analysis data")
            print("- summary.md: Human-readable summary")
            if config.create_video:
                print("- annotated.mp4: Video with pose overlays")

    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
