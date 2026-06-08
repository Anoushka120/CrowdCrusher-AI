# YOLOv8 Crowd Intelligence Dashboard

A Streamlit web application for real-time crowd density analysis, risk assessment, and trend prediction using YOLOv8 person detection and frame occupancy analysis.

## What This Does

This application processes video footage to:
- **Detect people** using YOLOv8 neural network (class 0: person only)
- **Estimate crowd density** using both YOLO bounding boxes and frame occupancy analysis
- **Classify risk levels** based on density ratio and person count
- **Predict crowd trends** using polynomial regression
- **Visualize hotspots** with density heatmaps overlaid on video
- **Export analytics** as processed video and frame-by-frame CSV data

## Key Features

- 📊 **Dual Detection Method**: YOLO person detection + adaptive frame occupancy analysis for dense crowds
- 🎥 **Video Upload**: Support for local PC files or browser upload (MP4, AVI, MOV, MKV, WEBM)
- ⚙️ **Configurable Detection**: Adjustable confidence threshold, heatmap parameters, and prediction horizon
- 📈 **Live Monitoring**: Frame-by-frame crowd count, density ratio, and risk level display during processing
- 🗂️ **Batch Analytics**: Frame-level metrics table with crowd count, density, risk, and trend data
- 📥 **Data Export**: Processed video with annotations and CSV analytics for further analysis
- 🎨 **Professional Dashboard**: Separate charts for people count, density ratio, and risk level visualization

## Risk Classification

Risk is based primarily on **crowd density ratio** (frame occupancy) with person count as secondary:

| Risk Level | Criteria | Color |
|-----------|----------|-------|
| 🟢 **SAFE** | Density <15% AND People <20 | Green (#22a06b) |
| 🟡 **MODERATE** | Density >15% OR People >20 | Amber (#f59e0b) |
| 🟠 **HIGH** | Density >25% OR People >50 | Orange (#f97316) |
| 🔴 **CRITICAL** | Density >35% OR People >80 | Red (#dc2626) |

## Technical Details

### Detection Methods

1. **YOLO Detection**: Ultralytics YOLOv8 neural network detecting person class (class ID 0)
2. **Frame Occupancy**: Adaptive thresholding on grayscale frames (70% dark pixels + 30% edges)
3. **Density Ratio**: Maximum of (YOLO box area / frame area) and (frame occupancy estimate)

### Analytics Calculation

- **Smoothing**: Rolling average over 10 frames to stabilize metrics
- **Risk Calculation**: Density ratio (primary) + person count (secondary)
- **Trend Detection**: Linear trend of crowd count (Increasing/Stable/Decreasing)
- **Prediction**: Polynomial degree-2 regression on recent crowd counts

### Processing

- Frame-by-frame analysis at video frame rate
- Real-time visualization during processing
- Overlay includes: people count, density ratio, risk level, predicted risk
- Heatmap visualization using Gaussian blur and color mapping

## Installation

### Requirements
- Python 3.10+
- 8GB+ RAM
- GPU recommended for processing speed (optional)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd crowd-intelligence-dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Locally

```bash
streamlit run app.py
```

Then:
1. Select video source (local PC path or browser upload)
2. Configure detection parameters in sidebar
3. Click "Process Video"
4. View results in Overview, Video, Analytics, and Downloads tabs

### Configuration Options

**Detection Settings**:
- YOLOv8 model: nano (fast), small (balanced), medium (accurate)
- Confidence threshold: 0.05-0.95 (default: 0.25)

**Visualization**:
- Show density heatmap (enabled by default)
- Heatmap radius: 20-140 pixels
- Heatmap opacity: 10-80%

**Forecasting**:
- Future estimate horizon: 5-120 seconds
- Trend window: 2-60 seconds

**Alerting**:
- Alert level: MODERATE, HIGH, or CRITICAL
- Live preview frequency: 1-30 frames

## Output Format

### Processed Video
- Input frame with annotations:
  - Green bounding boxes around detected people
  - Density heatmap (if enabled)
  - Red zones for critical density areas
  - Overlay with: people count, density %, risk level, predicted risk

### Analytics CSV
Columns per frame:
- `frame`: Frame number
- `time_seconds`: Time in video
- `crowd_count`: People detected by YOLO
- `crowd_ratio`: Density as decimal (0-1)
- `risk_level`: Current risk classification
- `predicted_count`: Predicted people count N seconds ahead
- `predicted_risk`: Predicted risk level
- `trend`: Crowd trend (Increasing/Stable/Decreasing)

## Limitations

### Current Scope
- **Crowd Counter, Not Prevention System**: This is a visualization/analysis tool, not a real-time prevention system
- **No Bottleneck Detection**: Doesn't identify exit congestion or dangerous zones
- **No Crowd Flow Analysis**: Doesn't track movement direction or velocity
- **No Evacuation Planning**: Doesn't route people to exits or calculate capacity
- **Linear Prediction**: Polynomial degree-2 prediction may not capture complex surge patterns

### Technical Limitations
- **Dense Crowds**: YOLO struggles with heavily overlapped people (density >50% may underestimate count)
- **Lighting Conditions**: Frame occupancy analysis works best in controlled lighting
- **Resolution Dependent**: Performance varies with video resolution and frame rate
- **Single Camera**: No multi-camera aggregate analysis
- **Memory**: Processes entire video in RAM (500MB+ videos may cause issues)

## Performance

Typical processing speed on CPU:
- 1-2 FPS for FHD (1920x1080) video with YOLOv8 nano
- 0.5-1 FPS for FHD video with YOLOv8 small
- GPU acceleration: 5-10x faster

First run downloads YOLOv8 weights (~100MB).

## Deployment

### Streamlit Cloud

```bash
# Ensure requirements.txt uses pinned versions
# Use opencv-python-headless instead of opencv-python
streamlit deploy
```

### Docker

```bash
docker build -t crowd-dashboard .
docker run -p 8501:8501 crowd-dashboard
```

## Project Architecture

```
inputs:
  ├── Video file (MP4, AVI, MOV, MKV, WEBM)
  └── Configuration (model, thresholds, parameters)
         ↓
detection:
  ├── YOLO person detection
  ├── Frame occupancy estimation
  └── Density ratio calculation
         ↓
analytics:
  ├── Risk classification
  ├── Trend detection
  ├── Prediction (polynomial regression)
  └── Smoothing (10-frame rolling average)
         ↓
visualization:
  ├── Video overlay (annotations, heatmap)
  ├── Dashboard charts (count, density, risk)
  └── Analytics table (frame-level metrics)
         ↓
outputs:
  ├── Processed video with overlays
  └── CSV analytics export
```

## Future Improvements

- [ ] Bottleneck detection (spatial density clustering)
- [ ] Optical flow analysis (crowd movement tracking)
- [ ] Real-time alerts (email/SMS/webhook)
- [ ] Multi-camera aggregate analysis
- [ ] GPU acceleration optimization
- [ ] Mobile app for live monitoring
- [ ] Integration with emergency systems
- [ ] Evacuation route planning

## Known Issues

- Dense crowds (density >50%) may underestimate person count
- Dark lighting may increase false positives in occupancy estimation
- Very large videos (>2GB) may cause out-of-memory errors
- Some video codecs may have compatibility issues

## Contributing

Contributions welcome! Areas for improvement:
- Better dense crowd detection algorithms
- Optical flow implementation
- Performance optimization
- GPU support
- Multi-camera support

## License

MIT License - See LICENSE file for details

## Disclaimer

This tool is designed for **analysis and visualization** purposes only. It is **NOT** a certified crowd safety system. Do not rely solely on this application for critical safety decisions. Always consult with trained safety professionals for crowd management.

## Citation

If you use this project in research, please cite:
```
YOLOv8 Crowd Intelligence Dashboard (2026)
https://github.com/<your-repo>/crowd-intelligence-dashboard
```

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for person detection
- [Streamlit](https://streamlit.io/) for the web framework
- [OpenCV](https://opencv.org/) for video processing

