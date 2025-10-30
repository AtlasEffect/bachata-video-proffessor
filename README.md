# Bachata Video Professor

A CPU-only Python pipeline for extracting Bachata dance combinations from YouTube videos.

## Features

- **CPU-friendly**: Uses MediaPipe for pose detection on CPU
- **Multi-person tracking**: Automatically identifies and focuses on the primary dancing couple
- **Unsupervised segmentation**: Detects individual dance combinations using motion analysis
- **Leader/follower identification**: Heuristic-based role assignment
- **Multiple outputs**: JSON summary, text report, and optional annotated video
- **CLI interface**: Easy command-line usage
- **Notebook ready**: Works in Colab/Kaggle environments

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# Analyze a YouTube video
bachata-analyze "https://youtu.be/4V7EccGsSUI?si=m-hxgRejZcg-b1nD" --out output/

# Analyze a local video file
bachata-analyze "video.mp4" --out output/ --fps 12 --min-seg-sec 3

# Generate summaries only (no video output)
bachata-analyze "https://youtu.be/OIEpCz8Q97A?si=k08QOEL5rkqALeDH" --out output/ --no-video
```

### Output Files

- `segments.json`: Detailed JSON with all detected combinations
- `summary.md`: Human-readable summary of all combos
- `annotated.mp4`: Video with skeleton overlays and segment labels (if enabled)

## Requirements

- Python 3.8+
- CPU (no GPU required)
- ~10-20 minutes processing time for 3-4 minute videos

## Performance Tips

- Lower FPS (`--fps 8`) for faster processing
- Reduce resolution (`--max-width 720`) for slower CPUs
- Increase minimum segment length (`--min-seg-sec 5`) to reduce false positives

## Example Videos

The system has been tested with these YouTube videos:
- https://youtu.be/4V7EccGsSUI?si=m-hxgRejZcg-b1nD
- https://youtu.be/OIEpCz8Q97A?si=k08QOEL5rkqALeDH  
- https://youtu.be/6MqwgPIiQaQ?si=LE9Tp8s-1wQ-OhBX

## Development

See `notebooks/Bachata_Analysis_Demo.ipynb` for a complete walkthrough of the analysis pipeline.

## License

MIT License - see LICENSE file for details.