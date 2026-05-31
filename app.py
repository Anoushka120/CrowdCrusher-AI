import csv
import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


PERSON_CLASS_ID = 0

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
        border: 1px solid #d7dde8;
        border-radius: 8px;
        padding: 20px 22px;
        background: #f8fafc;
        margin-bottom: 18px;
    }
    .app-hero h1 {
        margin: 0 0 6px 0;
        font-size: 2rem;
        letter-spacing: 0;
    }
    .app-hero p {
        margin: 0;
        color: #475569;
        font-size: 1rem;
    }
    .risk-badge {
        display: inline-block;
        border-radius: 6px;
        padding: 5px 10px;
        color: white;
        font-weight: 700;
        letter-spacing: 0;
    }
    .section-note {
        color: #64748b;
        font-size: 0.92rem;
        margin-top: -8px;
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_model(model_name: str) -> YOLO:
    return YOLO(model_name)


def get_risk_level(count: int) -> str:
    if count <= 20:
        return "SAFE"
    if count <= 50:
        return "MODERATE"
    if count < 100:
        return "HIGH"
    return "CRITICAL"


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
    if len(count_history) < 2:
        return count_history[-1] if count_history else 0

    window_size = max(2, int(fps * window_seconds))
    recent_counts = np.array(count_history[-window_size:], dtype=np.float32)
    frame_numbers = np.arange(len(recent_counts), dtype=np.float32)

    slope, intercept = np.polyfit(frame_numbers, recent_counts, 1)
    future_frame = len(recent_counts) + (fps * seconds_ahead)
    predicted_count = (slope * future_frame) + intercept

    return max(0, int(round(predicted_count)))


def build_analytics_csv(analytics_rows: list[dict]) -> bytes:
    if not analytics_rows:
        return b""

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=analytics_rows[0].keys())
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
    risk_level: str,
    predicted_count: int,
    predicted_risk: str,
    prediction_seconds: int,
):
    risk_color = RISK_STYLES[risk_level]["color"]
    predicted_risk_color = RISK_STYLES[predicted_risk]["color"]
    overlay_text = [
        f"Crowd count: {crowd_count}",
        f"Risk level: {risk_level}",
        f"{prediction_seconds}s estimate: {predicted_count} ({predicted_risk})",
    ]

    cv2.rectangle(frame, (16, 16), (560, 140), (20, 20, 20), -1)
    cv2.rectangle(frame, (16, 16), (560, 140), predicted_risk_color, 2)

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
            count_history.append(crowd_count)
            risk_level = get_risk_level(crowd_count)
            predicted_count = predict_future_count(
                count_history=count_history,
                fps=fps,
                seconds_ahead=prediction_seconds,
                window_seconds=trend_window_seconds,
            )
            predicted_risk = get_risk_level(predicted_count)
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
    summary = {
        "processed_frames": processed_frames,
        "max_count": max_count,
        "average_count": round(float(np.mean(count_history)), 1) if count_history else 0,
        "peak_risk": peak_risk,
        "highest_predicted_risk": get_highest_risk([row["predicted_risk"] for row in analytics_rows]),
        "final_predicted_count": predicted_history[-1] if predicted_history else 0,
        "final_predicted_risk": get_risk_level(predicted_history[-1]) if predicted_history else "SAFE",
        "trend": get_trend_label(count_history),
        "duration_seconds": round(processed_frames / fps, 2) if fps else 0,
    }
    return summary, count_history, predicted_history, analytics_rows


st.markdown(
    """
    <div class="app-hero">
        <h1>YOLOv8 Crowd Intelligence Dashboard</h1>
        <p>Upload surveillance or event footage, detect people, monitor crowd risk, visualize dense zones, and export frame-level analytics.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Detection")
    model_name = st.selectbox(
        "YOLOv8 model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        index=0,
        help="Larger models can be more accurate but process video more slowly.",
    )
    confidence_threshold = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)

    st.header("Visualization")
    show_heatmap = st.toggle("Show density heatmap", value=True)
    heatmap_radius = st.slider("Heatmap radius", 20, 140, 70, 10)
    heatmap_opacity = st.slider("Heatmap opacity", 0.10, 0.80, 0.45, 0.05)

    st.header("Forecast")
    prediction_seconds = st.slider("Future estimate horizon", 5, 120, 30, 5)
    trend_window_seconds = st.slider("Trend window", 2, 60, 10, 1)

    st.header("Operations")
    alert_level = st.selectbox("Alert at risk level", ["MODERATE", "HIGH", "CRITICAL"], index=1)
    preview_every = st.slider("Live preview every N frames", 1, 30, 3, 1)

    st.header("Risk levels")
    st.caption("0-20 SAFE | 21-50 MODERATE | 51-99 HIGH | 100+ CRITICAL")

upload_col, status_col = st.columns([1.05, 0.95])

with upload_col:
    st.subheader("Video Source")
    uploaded_video = st.file_uploader(
        "Upload video from your PC",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help="Choose a local video file from your computer, then click Open in the file picker.",
    )
    st.caption("Select a video saved on this computer. If your browser shows an extra mobile option, you can ignore it and use Open.")

with status_col:
    st.subheader("Analysis Setup")
    st.markdown('<p class="section-note">Current configuration before processing.</p>', unsafe_allow_html=True)
    setup_cols = st.columns(3)
    setup_cols[0].metric("Model", model_name.replace(".pt", ""))
    setup_cols[1].metric("Confidence", f"{confidence_threshold:.2f}")
    setup_cols[2].metric("Forecast", f"{prediction_seconds}s")
    render_risk_badge(alert_level)

if uploaded_video is None:
    st.info("Upload a video to begin analysis.")
else:
    with st.expander("Original video preview", expanded=True):
        st.video(uploaded_video)

    if st.button("Process video", type="primary"):
        model = load_model(model_name)

        input_suffix = Path(uploaded_video.name).suffix or ".mp4"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            input_path = temp_dir_path / f"uploaded{input_suffix}"
            output_path = temp_dir_path / "processed_video.mp4"

            input_path.write_bytes(uploaded_video.getbuffer())

            try:
                summary, count_history, predicted_history, analytics_rows = process_video(
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
                st.error(f"Video processing failed: {exc}")
            else:
                processed_video = output_path.read_bytes()
                analytics_csv = build_analytics_csv(analytics_rows)

                st.success("Analysis complete.")
                metric_cols = st.columns(5)
                metric_cols[0].metric("Frames", f"{summary['processed_frames']:,}")
                metric_cols[1].metric("Peak count", summary["max_count"])
                metric_cols[2].metric("Average count", summary["average_count"])
                metric_cols[3].metric("Trend", summary["trend"])
                metric_cols[4].metric("Future risk", summary["final_predicted_risk"])

                overview_tab, video_tab, analytics_tab, downloads_tab = st.tabs(
                    ["Overview", "Processed Video", "Analytics", "Downloads"]
                )

                with overview_tab:
                    st.subheader("Crowd Summary")
                    left, right = st.columns([0.45, 0.55])
                    with left:
                        st.write("Peak observed risk")
                        render_risk_badge(summary["peak_risk"])
                        st.write("Highest predicted risk")
                        render_risk_badge(summary["highest_predicted_risk"])
                    with right:
                        st.line_chart(
                            {
                                "Observed crowd count": count_history,
                                "Predicted future count": predicted_history,
                            },
                            height=280,
                        )

                with video_tab:
                    st.subheader("Processed Video")
                    st.video(processed_video)

                with analytics_tab:
                    st.subheader("Frame Analytics")
                    st.dataframe(analytics_rows, use_container_width=True, hide_index=True)

                with downloads_tab:
                    st.subheader("Export")
                    download_cols = st.columns(2)
                    download_cols[0].download_button(
                        "Download processed video",
                        data=processed_video,
                        file_name="processed_crowd_count.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )
                    download_cols[1].download_button(
                        "Download analytics CSV",
                        data=analytics_csv,
                        file_name="crowd_analytics.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
