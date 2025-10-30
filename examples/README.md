# Example Videos for Bachata Analysis

This directory contains example YouTube URLs that have been tested with the Bachata Video Professor pipeline.

## Test Videos

1. **https://youtu.be/4V7EccGsSUI?si=m-hxgRejZcg-b1nD**
   - Typical Bachata demonstration
   - Multiple combinations
   - Clear view of dancing couple

2. **https://youtu.be/OIEpCz8Q97A?si=k08QOEL5rkqALeDH**
   - Social dancing context
   - Some background people
   - Good for testing audience filtering

3. **https://youtu.be/6MqwgPIiQaQ?si=LE9Tp8s-1wQ-OhBX**
   - Performance style
   - Complex combinations
   - Fast-paced movements

## Usage

```bash
# Test with all example videos
bachata-analyze "https://youtu.be/4V7EccGsSUI?si=m-hxgRejZcg-b1nD" --out output/video1/
bachata-analyze "https://youtu.be/OIEpCz8Q97A?si=k08QOEL5rkqALeDH" --out output/video2/
bachata-analyze "https://youtu.be/6MqwgPIiQaQ?si=LE9Tp8s-1wQ-OhBX" --out output/video3/
```

## Expected Results

- Each video should produce 15+ segments when combinations are present
- Primary couple should be correctly identified and tracked
- Background people should be ignored
- JSON and text summaries should be generated for all videos