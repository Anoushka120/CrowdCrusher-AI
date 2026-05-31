# YOLOv8 Crowd Counter

A Streamlit application that uploads a video, processes it frame-by-frame with YOLOv8, detects only people, draws bounding boxes, displays a live crowd count, tracks crowd count across frames, predicts crowd growth trend, shows future risk estimates, overlays a crowd density heatmap, and outputs the processed video.

Dense zones are highlighted in red using detected person coordinates.

Future risk is estimated from a rolling linear trend over recent frame counts. Adjust the prediction horizon and trend window from the sidebar.

## Features

- Application-style dashboard with upload, configuration, live monitoring, result tabs, and exports
- Person-only YOLOv8 detection
- Crowd density heatmap with red dense-zone highlights
- Live crowd count and risk level
- Future crowd count and risk estimate
- Peak, average, trend, and predicted-risk summary metrics
- Frame-level analytics table
- Configurable risk alert level
- Processed video and CSV downloads

## Risk Levels

- 0-20 people: SAFE
- 21-50 people: MODERATE
- 51-99 people: HIGH
- 100+ people: CRITICAL

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the selected YOLOv8 model weights from Ultralytics.
