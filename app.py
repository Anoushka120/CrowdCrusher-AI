import csv
import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


PERSON_CLASS_ID = 0
SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

RISK_STYLES = {
    "SAFE": {"color": (40, 180, 80), "status": "success", "hex": "#22a06b", "score": 1},
    "MODERATE": {"color": (0, 190, 255), "status": "warning", "hex": "#f59e0b", "score": 2},
    "HIGH": {"color": (0, 120, 255), "status": "warning", "hex": "#f97316", "score": 3},
    "CRITICAL": {"color": (35, 35, 220), "status": "error", "hex": "#dc2626", "score": 4},
}


st.set_page_config(
    page_title="YOLOv8 Crowd Counter",
    page_icon="video_camera",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    .app-hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 36px 32px;
        margin-bottom: 28px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
    }
    .app-hero h1 {
        margin: 0 0 12px 0;
        font-size: 2.4rem;
        letter-spacing: -0.5px;
        color: white;
        font-weight: 800;
    }
    .app-hero p {
        margin: 0;
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.05rem;
        line-height: 1.5;
        font-weight: 300;
    }
    .risk-badge {
        display: inline-block;
        border-radius: 8px;
        padding: 8px 14px;
        color: white;
        font-weight: 700;
        letter-spacing: 0.5px;
        font-size: 0.9rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }
    .section-note {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: -8px;
        margin-bottom: 14px;
        font-weight: 500;
    }
    .sidebar-header {
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 24px;
        margin-bottom: 12px;
        color: #1e293b;
        letter-spacing: -0.3px;
    }
    .metric-card {
        background: #f8fafc;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e2e8f0;
    }
    .setup-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #fafbfc 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e2e8f0;
    }
    .tab-content {
        margin-top: 24px;
    }
    .success-message {
        border-left: 4px solid #22c55e;
        background: #f0fdf4;
        padding: 14px;
        border-radius: 6px;
    }
    /* Enhanced button styling */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
        transform: translateY(-2px) !important;
    }
    /* Improved metric styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        border-radius: 10px;
        padding: 16px !important;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    /* Better slider styling */
    .stSlider {
        padding: 10px 0;
    }
    /* Improved selectbox styling */
    .stSelectbox {
        padding: 8px 0;
    }
    /* Enhanced expander styling */
    .stExpander {
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
    }
    /* Better info/warning/error messages */
    .stAlert {
        border-radius: 8px !important;
        padding: 12px 16px !important;
        border-left: 4px solid !important;
    }
    /* Improved dataframe styling */
    [data-testid="dataframe"] {
        border-radius: 8px !important;
        overflow: hidden;
    }
    /* Tab styling enhancement */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px !important;
        background-color: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_model(model_name: str) -> YOLO:
    model = YOLO(model_name)
    # Verify person class is available
    if 0 not in range(len(model.names)) or model.names.get(0, "").lower() != "person":
        st.warning(f"⚠️ Warning: Class 0 may not be 'person'. Model has classes: {model.names}")
    return model


def calculate_crowd_ratio(boxes, frame_width: int, frame_height: int) -> float:
    """Calculate crowd density ratio as total box area / frame area."""
    if len(boxes) == 0:
        return 0.0
    
    frame_area = frame_width * frame_height
    if frame_area == 0:
        return 0.0
    
    total_box_area = 0
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        box_width = max(0, x2 - x1)
        box_height = max(0, y2 - y1)
        total_box_area += box_width * box_height
    
    return total_box_area / frame_area


def estimate_frame_occupancy(frame) -> float:
    """Estimate occupancy ratio using adaptive thresholding.
    
    Uses adaptive thresholding for robust detection across different lighting conditions.
    Returns value normalized between 0 and 1.
    """
    if frame is None or frame.size == 0:
        return 0.0
    
    height, width = frame.shape[:2]
    frame_area = height * width
    if frame_area == 0:
        return 0.0
    
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding handles different lighting conditions better
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        dark_pixels = np.sum(adaptive_thresh == 0)
        
        # Detect edges (crowd boundaries)
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges > 0)
        
        # Combine detections
        dark_occupancy = dark_pixels / frame_area
        edge_occupancy = edge_pixels / frame_area
        combined_occupancy = (dark_occupancy * 0.7) + (edge_occupancy * 0.3)
        
        return min(1.0, max(0.0, combined_occupancy))
    except Exception:
        return 0.0


def smooth_values(values: list[float], window_size: int = 10) -> list[float]:
    """Apply rolling average smoothing to stabilize values."""
    if len(values) < window_size:
        return values
    
    smoothed = []
    for i in range(len(values)):
        start_idx = max(0, i - window_size + 1)
        window = values[start_idx:i+1]
        smoothed.append(np.mean(window))
    
    return smoothed


def get_risk_level(person_count: int, crowd_ratio: float) -> str:
    """Determine risk level based on crowd density ratio (primary) and person count (secondary).
    
    Prioritizes density ratio since it's more reliable for dense top-view crowds.
    
    CRITICAL: crowd_ratio > 0.35 OR person_count > 80
    HIGH: crowd_ratio > 0.25 OR person_count > 50
    MODERATE: crowd_ratio > 0.15 OR person_count > 20
    SAFE: otherwise
    """
    if crowd_ratio > 0.35 or person_count > 80:
        return "CRITICAL"
    if crowd_ratio > 0.25 or person_count > 50:
        return "HIGH"
    if crowd_ratio > 0.15 or person_count > 20:
        return "MODERATE"
    return "SAFE"


def is_supported_video(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES


def get_highest_risk(risk_levels: list[str]) -> str:
    if not risk_levels:
        return "SAFE"
    return max(risk_levels, key=lambda risk: RISK_STYLES[risk]["score"])


def get_trend_label(count_history: list[int]) -> str:
    if len(count_history) < 3:
        return "Collecting data"

    recent_window = count_history[-min(len(count_history), 12):]
    delta = recent_window[-1] - recent_window[0]
    if delta >= 3:
        return "Increasing"
    if delta <= -3:
        return "Decreasing"
    return "Stable"


def predict_future_count(count_history: list[int], fps: float, seconds_ahead: int, window_seconds: int) -> int:
    """Predict future crowd count using polynomial regression for better accuracy."""
    if len(count_history) < 3:
        return count_history[-1] if count_history else 0

    window_size = max(3, int(fps * window_seconds))
    recent_counts = np.array(count_history[-window_size:], dtype=np.float32)
    frame_numbers = np.arange(len(recent_counts), dtype=np.float32)

    try:
        # Use polynomial degree 2 instead of linear for better nonlinear prediction
        coeffs = np.polyfit(frame_numbers, recent_counts, 2)
        poly = np.poly1d(coeffs)
        future_frame = len(recent_counts) + (fps * seconds_ahead)
        predicted_count = poly(future_frame)
        return max(0, int(round(predicted_count)))
    except Exception:
        return int(recent_counts[-1]) if len(recent_counts) > 0 else 0


def build_analytics_csv(analytics_rows: list[dict]) -> bytes:
    if not analytics_rows or len(analytics_rows) == 0:
        return b""

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=analytics_rows[0].keys())
    writer.writeheader()
    writer.writerows(analytics_rows)
    return output.getvalue().encode("utf-8")
    writer.writeheader()
    writer.writerows(analytics_rows)
    return output.getvalue().encode("utf-8")


def render_risk_badge(risk_level: str) -> None:
    color = RISK_STYLES[risk_level]["hex"]
    st.markdown(
        f'<span class="risk-badge" style="background:{color};">{risk_level}</span>',
        unsafe_allow_html=True,
    )


def show_risk_status(slot, risk_level: str) -> None:
    status = RISK_STYLES[risk_level]["status"]
    message = f"Risk level: {risk_level}"

    if status == "success":
        slot.success(message)
    elif status == "warning":
        slot.warning(message)
    else:
        slot.error(message)


def draw_person_detections(frame, boxes, confidences):
    for box, confidence in zip(boxes, confidences):
        x1, y1, x2, y2 = map(int, box)
        label = f"Person {confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (35, 190, 80), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (35, 190, 80),
            2,
            cv2.LINE_AA,
        )

    return frame


def draw_density_heatmap(frame, boxes, radius: int, opacity: float):
    if len(boxes) == 0:
        return frame

    height, width = frame.shape[:2]
    density = np.zeros((height, width), dtype=np.float32)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        center_x = min(max((x1 + x2) // 2, 0), width - 1)
        center_y = min(max((y1 + y2) // 2, 0), height - 1)
        cv2.circle(density, (center_x, center_y), radius, 1.0, -1)

    blur_size = max(3, radius * 2 + 1)
    if blur_size % 2 == 0:
        blur_size += 1

    density = cv2.GaussianBlur(density, (blur_size, blur_size), 0)
    density = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    colored_heatmap = cv2.applyColorMap(density, cv2.COLORMAP_JET)
    dense_zone_mask = density >= 120
    blended = cv2.addWeighted(frame, 1.0 - opacity, colored_heatmap, opacity, 0)
    frame[dense_zone_mask] = blended[dense_zone_mask]

    red_zone_mask = density >= 180
    red_highlight = np.zeros_like(frame)
    red_highlight[:, :] = (0, 0, 255)
    red_blend = cv2.addWeighted(frame, 0.55, red_highlight, 0.45, 0)
    frame[red_zone_mask] = red_blend[red_zone_mask]

    return frame


def draw_dashboard_overlay(
    frame,
    crowd_count: int,
    crowd_ratio: float,
    risk_level: str,
    predicted_count: int,
    predicted_risk: str,
    prediction_seconds: int,
):
    risk_color = RISK_STYLES[risk_level]["color"]
    predicted_risk_color = RISK_STYLES[predicted_risk]["color"]
    overlay_text = [
        f"People: {crowd_count} | Density: {crowd_ratio:.2%}",
        f"Risk: {risk_level}",
        f"{prediction_seconds}s ahead: {predicted_count} ({predicted_risk})",
    ]

    cv2.rectangle(frame, (16, 16), (640, 140), (20, 20, 20), -1)
    cv2.rectangle(frame, (16, 16), (640, 140), predicted_risk_color, 2)

    for index, text in enumerate(overlay_text):
        cv2.putText(
            frame,
            text,
            (30, 50 + index * 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            risk_color if index == 1 else predicted_risk_color if index == 2 else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


def process_video(
    input_path: Path,
    output_path: Path,
    model: YOLO,
    confidence_threshold: float,
    show_heatmap: bool,
    heatmap_radius: int,
    heatmap_opacity: float,
    prediction_seconds: int,
    trend_window_seconds: int,
    alert_level: str,
    preview_every: int,
):
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError("Could not open the uploaded video.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 24
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    progress = st.progress(0, text="Processing video...")
    frame_slot = st.empty()
    count_col, risk_col, future_col = st.columns(3)
    count_slot = count_col.empty()
    risk_slot = risk_col.empty()
    future_slot = future_col.empty()

    max_count = 0
    peak_risk = "SAFE"
    processed_frames = 0
    count_history = []
    crowd_ratio_history = []
    predicted_history = []
    risk_history = []
    analytics_rows = []
    last_alert_score = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            results = model.predict(
                source=frame,
                classes=[PERSON_CLASS_ID],
                conf=confidence_threshold,
                verbose=False,
            )

            boxes = []
            confidences = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()

            crowd_count = len(boxes)
            yolo_density = calculate_crowd_ratio(boxes, width, height)
            
            # Fallback: estimate density from frame occupancy (for dense top-view crowds)
            fallback_density = estimate_frame_occupancy(frame)
            
            # Use maximum of YOLO density and fallback occupancy
            crowd_ratio = max(yolo_density, fallback_density)
            
            count_history.append(crowd_count)
            crowd_ratio_history.append(crowd_ratio)
            risk_level = get_risk_level(crowd_count, crowd_ratio)
            
            predicted_count = predict_future_count(
                count_history=count_history,
                fps=fps,
                seconds_ahead=prediction_seconds,
                window_seconds=trend_window_seconds,
            )
            predicted_risk = get_risk_level(predicted_count, 0.0)
            trend_label = get_trend_label(count_history)
            predicted_history.append(predicted_count)
            risk_history.append(risk_level)
            if crowd_count > max_count:
                max_count = crowd_count
                peak_risk = risk_level

            annotated = frame.copy()
            if show_heatmap:
                annotated = draw_density_heatmap(
                    annotated,
                    boxes,
                    radius=heatmap_radius,
                    opacity=heatmap_opacity,
                )
            annotated = draw_person_detections(annotated, boxes, confidences)
            annotated = draw_dashboard_overlay(
                annotated,
                crowd_count,
                crowd_ratio,
                risk_level,
                predicted_count,
                predicted_risk,
                prediction_seconds,
            )

            writer.write(annotated)
            processed_frames += 1

            count_slot.metric("Live crowd count", crowd_count)
            show_risk_status(risk_slot, risk_level)
            future_slot.metric(
                f"{prediction_seconds}s future estimate",
                predicted_count,
                predicted_risk,
            )
            current_risk_score = RISK_STYLES[risk_level]["score"]
            alert_risk_score = RISK_STYLES[alert_level]["score"]
            if current_risk_score >= alert_risk_score and current_risk_score > last_alert_score:
                st.toast(f"{risk_level} crowd risk detected at frame {processed_frames:,}")
                last_alert_score = current_risk_score

            if processed_frames % preview_every == 0 or processed_frames == 1:
                frame_slot.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True,
                )

            analytics_rows.append(
                {
                    "frame": processed_frames,
                    "time_seconds": round(processed_frames / fps, 2),
                    "crowd_count": crowd_count,
                    "crowd_ratio": round(crowd_ratio, 4),
                    "risk_level": risk_level,
                    "predicted_count": predicted_count,
                    "predicted_risk": predicted_risk,
                    "trend": trend_label,
                }
            )

            if total_frames > 0:
                progress.progress(
                    min(processed_frames / total_frames, 1.0),
                    text=f"Processing frame {processed_frames:,} of {total_frames:,}",
                )

    finally:
        capture.release()
        writer.release()

    progress.progress(1.0, text="Processing complete.")
    
    # Apply smoothing to stabilize metrics
    smoothed_crowd_ratio = smooth_values(crowd_ratio_history, window_size=10)
    smoothed_count = smooth_values(count_history, window_size=10)
    
    summary = {
        "processed_frames": processed_frames,
        "max_count": max_count,
        "average_count": round(float(np.mean(count_history)), 1) if count_history else 0,
        "max_crowd_ratio": round(float(np.max(crowd_ratio_history)), 4) if crowd_ratio_history else 0.0,
        "average_crowd_ratio": round(float(np.mean(crowd_ratio_history)), 4) if crowd_ratio_history else 0.0,
        "peak_risk": peak_risk,
        "highest_predicted_risk": get_highest_risk([row["predicted_risk"] for row in analytics_rows]),
        "final_predicted_count": predicted_history[-1] if predicted_history else 0,
        "final_predicted_risk": get_risk_level(predicted_history[-1], 0.0) if predicted_history else "SAFE",
        "trend": get_trend_label(count_history),
        "duration_seconds": round(processed_frames / fps, 2) if fps else 0,
    }
    return summary, smoothed_count, smoothed_crowd_ratio, predicted_history, analytics_rows


st.markdown(
    """
    <div class="app-hero">
        <h1>🎥 YOLOv8 Crowd Intelligence Dashboard</h1>
        <p>Upload surveillance or event footage, detect people in real-time, monitor crowd density, visualize hot zones, and export detailed frame-by-frame analytics.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 🔍 Detection Settings")
    model_name = st.selectbox(
        "YOLOv8 model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        index=0,
        help="Larger models can be more accurate but process video more slowly.",
    )
    confidence_threshold = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05, help="Lower values detect more people (higher sensitivity). Default 0.25 for better crowd detection.")
    
    st.info("ℹ️ **Dual Detection Method**: YOLO person detection + fallback frame occupancy analysis for dense crowds where individuals may be heavily overlapped.")

    st.markdown("### 🌡️ Visualization")
    show_heatmap = st.toggle("Show density heatmap", value=True)
    if show_heatmap:
        heatmap_radius = st.slider("Heatmap radius", 20, 140, 70, 10, help="Size of influence around each detected person")
        heatmap_opacity = st.slider("Heatmap opacity", 0.10, 0.80, 0.45, 0.05, help="How visible the heatmap overlay is")

    st.markdown("### 📈 Forecast Settings")
    prediction_seconds = st.slider("Future estimate horizon", 5, 120, 30, 5, help="How many seconds ahead to predict")
    trend_window_seconds = st.slider("Trend window", 2, 60, 10, 1, help="Window for calculating trend")

    st.markdown("### ⚠️ Alerts & Display")
    alert_level = st.selectbox("Alert at risk level", ["MODERATE", "HIGH", "CRITICAL"], index=1, help="Show alerts when crowd reaches this risk level")
    preview_every = st.slider("Live preview every N frames", 1, 30, 3, 1, help="Preview frequency during processing")

    st.markdown("---")
    st.markdown("### 📊 Risk Levels (Density Ratio Primary)")
    st.markdown("""
    Risk is based primarily on **density ratio** (frame occupancy), with person count as a secondary factor.
    
    - 🟢 **SAFE**: Density <15% AND People <20
    - 🟡 **MODERATE**: Density >15% OR People >20
    - 🟠 **HIGH**: Density >25% OR People >50
    - 🔴 **CRITICAL**: Density >35% OR People >80
    
    *Frame occupancy is estimated from dark pixels and edge detection for dense top-view crowds.*
    """)

upload_col, status_col = st.columns([1.05, 0.95])

with upload_col:
    st.markdown("### 📁 Video Source")
    source_method = st.radio(
        "Select video source",
        ["Choose from this PC", "Browser upload"],
        horizontal=True,
    )
    uploaded_video = None
    local_video_path = None
    selected_video_path = None

    if source_method == "Choose from this PC":
        local_path_text = st.text_input(
            "Video file path on this computer",
            placeholder=r"C:\Users\YourName\Videos\crowd_video.mp4",
            help="Paste the full path of a video saved on this PC. This avoids the browser file picker.",
        )
        st.caption("💡 Tip: right-click your video file in File Explorer, choose Copy as path, then paste it here.")

        if local_path_text:
            local_video_path = Path(local_path_text.strip().strip('"'))
            if not local_video_path.exists():
                st.error("❌ That file path was not found. Check the path and try again.")
            elif not local_video_path.is_file():
                st.error("❌ That path is not a video file.")
            elif not is_supported_video(local_video_path):
                st.error("❌ Supported formats: MP4, AVI, MOV, MKV, WEBM.")
            else:
                selected_video_path = local_video_path
                st.success(f"✅ Selected: {local_video_path.name}")
    else:
        uploaded_video = st.file_uploader(
            "Upload video from your PC",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            help="Choose a local video file from your computer, then click Open in the file picker.",
        )
        st.caption("💡 If your browser shows an extra mobile option, ignore it and click Open after selecting your PC file.")

with status_col:
    st.markdown("### ⚙️ Analysis Configuration")
    st.markdown('<p class="section-note">Your current settings before processing.</p>', unsafe_allow_html=True)
    setup_cols = st.columns(3)
    setup_cols[0].metric("Model", model_name.replace(".pt", "").upper())
    setup_cols[1].metric("Threshold", f"{confidence_threshold:.2f}")
    setup_cols[2].metric("Horizon", f"{prediction_seconds}s")
    render_risk_badge(alert_level)

if selected_video_path is None and uploaded_video is None:
    st.info("🎬 Choose a local PC video path or upload a video to begin analysis.")
else:
    with st.expander("▶️ Original video preview", expanded=True):
        if selected_video_path is not None:
            st.video(str(selected_video_path))
        else:
            st.video(uploaded_video)

    col1, col2 = st.columns([1, 4])
    with col1:
        process_button = st.button("▶️ Process Video", type="primary", use_container_width=True)
    with col2:
        st.markdown("*This will analyze your video frame-by-frame and generate a processed output with crowd detection and risk assessment.*")

    if process_button:
        model = load_model(model_name)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            output_path = temp_dir_path / "processed_video.mp4"

            if selected_video_path is not None:
                input_path = selected_video_path
                # Check file size (max 500MB)
                file_size_mb = input_path.stat().st_size / (1024 * 1024)
                if file_size_mb > 500:
                    st.error(f"❌ File too large: {file_size_mb:.1f}MB (max 500MB). Please use a smaller video.")
                    st.stop()
            else:
                # Validate uploaded file size (max 500MB)
                file_size_mb = len(uploaded_video.getbuffer()) / (1024 * 1024)
                if file_size_mb > 500:
                    st.error(f"❌ File too large: {file_size_mb:.1f}MB (max 500MB). Please use a smaller video.")
                    st.stop()
                
                input_suffix = Path(uploaded_video.name).suffix or ".mp4"
                input_path = temp_dir_path / f"uploaded{input_suffix}"
                input_path.write_bytes(uploaded_video.getbuffer())

            try:
                summary, count_history, crowd_ratio_history, predicted_history, analytics_rows = process_video(
                    input_path=input_path,
                    output_path=output_path,
                    model=model,
                    confidence_threshold=confidence_threshold,
                    show_heatmap=show_heatmap,
                    heatmap_radius=heatmap_radius,
                    heatmap_opacity=heatmap_opacity,
                    prediction_seconds=prediction_seconds,
                    trend_window_seconds=trend_window_seconds,
                    alert_level=alert_level,
                    preview_every=preview_every,
                )
            except Exception as exc:
                st.error(f"❌ Video processing failed: {exc}")
            else:
                processed_video = output_path.read_bytes()
                analytics_csv = build_analytics_csv(analytics_rows)

                st.success("✅ Analysis complete! Processing finished successfully.")
                
                st.markdown("---")
                st.markdown("### 📊 Processing Summary")
                metric_cols = st.columns(5)
                metric_cols[0].metric("🎬 Frames", f"{summary['processed_frames']:,}")
                metric_cols[1].metric("👥 Peak Count", summary["max_count"])
                metric_cols[2].metric("📈 Avg Count", summary["average_count"])
                metric_cols[3].metric("🌡️ Max Density", f"{summary['max_crowd_ratio']:.2%}")
                metric_cols[4].metric("📉 Trend", summary["trend"])

                overview_tab, video_tab, analytics_tab, downloads_tab = st.tabs(
                    ["📊 Overview", "🎥 Processed Video", "📋 Analytics", "⬇️ Downloads"]
                )

                with overview_tab:
                    st.markdown("### 👥 Crowd Analysis Summary")
                    
                    # Key statistics
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    with stats_col1:
                        st.metric("Peak People Count", summary["max_count"])
                        st.metric("Average People Count", summary["average_count"])
                    with stats_col2:
                        st.metric("Max Density Ratio", f"{summary['max_crowd_ratio']:.2%}")
                        st.metric("Avg Density Ratio", f"{summary['average_crowd_ratio']:.2%}")
                    with stats_col3:
                        st.metric("Duration", f"{summary['duration_seconds']}s")
                        st.metric("Total Frames", f"{summary['processed_frames']:,}")
                    
                    st.markdown("---")
                    
                    # Check if dense crowd detected with low YOLO count
                    avg_yolo_count = np.mean(count_history) if count_history else 0
                    max_density_ratio = np.max(crowd_ratio_history) if crowd_ratio_history else 0
                    if avg_yolo_count < 30 and max_density_ratio > 0.25:
                        st.warning("⚠️ **Dense crowd detected even when YOLO person count is low.** This indicates a very dense crowd where individuals are heavily overlapped or occluded. Density ratio and frame occupancy analysis is being used to estimate actual crowd density.")
                    
                    # Risk badges
                    risk_col1, risk_col2 = st.columns([0.45, 0.55])
                    with risk_col1:
                        st.markdown("#### 🎯 Peak Risk Observed")
                        render_risk_badge(summary["peak_risk"])
                        st.markdown("#### 🔮 Highest Predicted Risk")
                        render_risk_badge(summary["highest_predicted_risk"])
                    with risk_col2:
                        st.markdown("#### 📊 Key Metrics")
                        st.markdown(f"**Avg YOLO Count**: {avg_yolo_count:.1f}")
                        st.markdown(f"**Max Density Ratio**: {max_density_ratio:.2%}")
                        st.markdown(f"**Trend**: {summary['trend']}")
                    
                    st.markdown("---")
                    st.markdown("#### 📈 People Count Over Time")
                    st.line_chart(count_history, height=250)
                    
                    st.markdown("#### 🌡️ Density Ratio Over Time (0-100%)")
                    st.line_chart([x * 100 for x in crowd_ratio_history], height=250)
                    
                    st.markdown("#### ⚠️ Risk Level Over Time")
                    # Convert risk levels to numeric scores for visualization
                    risk_scores = [RISK_STYLES[risk]["score"] for risk in [get_risk_level(int(c), float(d)) for c, d in zip(count_history, crowd_ratio_history)]]
                    st.line_chart(risk_scores, height=250)

                with video_tab:
                    st.markdown("### 🎥 Processed Video with Annotations")
                    st.video(processed_video)
                    st.caption("Green bounding boxes: detected people | Crowd ratio: % of frame covered by people | Risk level: based on count & density | Heatmap: density visualization | Red zones: critical density | Top-left overlay: real-time metrics including people count, density ratio, and predicted risk")

                with analytics_tab:
                    st.markdown("### 📋 Frame-by-Frame Analytics")
                    st.dataframe(analytics_rows, use_container_width=True, hide_index=True)
                    st.caption(f"Total frames analyzed: {len(analytics_rows)}")

                with downloads_tab:
                    st.markdown("### ⬇️ Export Results")
                    st.markdown("Save your analysis results for later use or sharing:")
                    download_cols = st.columns(2)
                    download_cols[0].download_button(
                        "🎬 Download Processed Video",
                        data=processed_video,
                        file_name="processed_crowd_count.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )
                    download_cols[1].download_button(
                        "📊 Download Analytics CSV",
                        data=analytics_csv,
                        file_name="crowd_analytics.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                    st.caption("💾 The processed video includes bounding boxes, crowd counts, risk indicators, and density heatmaps. The CSV file contains frame-by-frame metrics for further analysis.")
