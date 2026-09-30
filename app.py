"""AI Real-Time Object Detection - Streamlit Web Application.

A production-ready browser-based computer vision application powered by
Streamlit, Ultralytics YOLO, PyTorch, and OpenCV.
"""

from pathlib import Path
import sys
import time
from typing import List, Optional
import numpy as np
from PIL import Image
import streamlit as st

# Configure page layout and title
st.set_page_config(
    page_title="AI Real-Time Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from detector import DetectionResult, ObjectDetector
from utils.metrics import compute_detection_summary, format_latency_and_fps


@st.cache_resource(show_spinner="Loading YOLO model into memory...")
def get_detector(model_name: str, device: Optional[str] = None) -> ObjectDetector:
    """Load and cache the YOLO ObjectDetector across Streamlit sessions.

    Using @st.cache_resource guarantees the neural network weights are loaded
    into memory only once, avoiding expensive reload latency on UI reruns.
    """
    return ObjectDetector(
        model_name=model_name,
        device=device,
    )


def create_synthetic_demo_image() -> Image.Image:
    """Generate a clean synthetic sample frame with shapes for demo testing."""
    img_array = np.full((500, 800, 3), 245, dtype=np.uint8)
    import cv2
    # Draw sample elements
    cv2.rectangle(img_array, (100, 120), (350, 450), (45, 85, 230), -1)
    cv2.rectangle(img_array, (450, 200), (700, 420), (50, 180, 75), -1)
    cv2.putText(
        img_array,
        "DEMO TEST FRAME",
        (260, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    return Image.fromarray(img_array)


def render_sidebar(detector: ObjectDetector) -> tuple[float, float, List[str]]:
    """Render the sidebar controls and configuration options."""
    st.sidebar.title("⚙️ Detection Settings")
    st.sidebar.markdown("---")

    st.sidebar.subheader("🎯 Model Parameters")
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=config.DEFAULT_CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Filter out detections with confidence below this value.",
    )

    iou_threshold = st.sidebar.slider(
        "NMS IoU Threshold",
        min_value=0.10,
        max_value=1.00,
        value=config.DEFAULT_IOU_THRESHOLD,
        step=0.05,
        help="Intersection-over-Union threshold for Non-Maximum Suppression overlap removal.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Class Filter")

    available_classes = sorted(list(detector.class_names.values()))
    selected_classes = st.sidebar.multiselect(
        "Filter Specific Classes (leave empty for all)",
        options=available_classes,
        default=[],
        help="Select specific categories to detect, e.g., 'person', 'car', 'bottle'.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ System Status")
    st.sidebar.info(
        f"**Model**: `{detector.model_name}`\n\n"
        f"**Device**: `{detector.device.upper()}`\n\n"
        f"**Recognized Classes**: `{len(detector.class_names)}`"
    )

    return conf_threshold, iou_threshold, selected_classes


def main() -> None:
    """Main application routine."""
    # Top Header
    st.title("🎯 AI Real-Time Object Detection")
    st.markdown(
        "**Real-time computer vision using YOLO and OpenCV.** "
        "Capture live frames from your browser webcam or upload images for instant deep learning inference."
    )

    # Load Model with Error Handling
    try:
        detector = get_detector(config.MODEL_NAME, config.DEVICE)
    except Exception as e:
        st.error(
            f"❌ **Failed to load YOLO model.**\n\n"
            f"Details: `{e}`\n\n"
            f"Please ensure internet access for the initial weights download or verify `models/` directory."
        )
        return

    # Render Sidebar
    conf_threshold, iou_threshold, class_filter = render_sidebar(detector)

    # Input Method Tabs
    tab_webcam, tab_upload = st.tabs(["📷 Browser Webcam", "📁 Image Upload"])

    input_image: Optional[Image.Image] = None

    with tab_webcam:
        st.markdown(
            "#### 📷 Live Browser Camera Input\n"
            "*Your browser will prompt for camera access. Click **Take Photo** to capture and detect in real time.*"
        )
        camera_file = st.camera_input("Webcam Capture", label_visibility="collapsed")
        if camera_file is not None:
            try:
                input_image = Image.open(camera_file)
            except Exception as err:
                st.error(f"Error opening camera frame: {err}")

    with tab_upload:
        st.markdown("#### 📁 Upload an Image File")
        uploaded_file = st.file_uploader(
            "Upload image (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            try:
                input_image = Image.open(uploaded_file)
            except Exception as err:
                st.error(f"Error reading uploaded file: {err}")

    # Process and Display Results
    if input_image is not None:
        st.markdown("---")

        with st.spinner("Executing YOLO inference..."):
            try:
                detections, inference_time = detector.detect(
                    input_image,
                    conf_override=conf_threshold,
                )
            except Exception as err:
                st.error(f"❌ Inference error occurred: {err}")
                return

        # Apply class filter if specified
        if class_filter:
            detections = [d for d in detections if d["class_name"] in class_filter]

        summary = compute_detection_summary(detections)
        latency_ms, fps = format_latency_and_fps(inference_time)

        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Objects Detected", summary.total_count)
        with col2:
            st.metric("Inference Latency", f"{latency_ms} ms")
        with col3:
            st.metric("Processing Speed", f"{fps} FPS")
        with col4:
            avg_conf_pct = f"{int(summary.average_confidence * 100)}%" if summary.total_count > 0 else "0%"
            st.metric("Avg Confidence", avg_conf_pct)

        # Annotated Image Generation
        annotated_np = detector.annotate_image(input_image, detections)
        annotated_image = Image.fromarray(annotated_np)

        # Visual Comparison Display
        st.markdown("### 🖼️ Detection Visuals")
        view_col1, view_col2 = st.columns(2)

        with view_col1:
            st.markdown("**Original Image**")
            st.image(input_image, use_container_width=True)

        with view_col2:
            st.markdown(f"**Annotated Output ({summary.total_count} Objects)**")
            st.image(annotated_image, use_container_width=True)

        # Structured Detections Table
        st.markdown("### 📊 Detection Details")
        if summary.total_count > 0:
            det_data = [
                {
                    "Object": d["class_name"].title(),
                    "Confidence": f"{int(d['confidence'] * 100)}%",
                    "Class ID": d["class_id"],
                    "Bounding Box [x1, y1, x2, y2]": str(d["bbox"]),
                }
                for d in summary.top_detections
            ]
            st.dataframe(det_data, use_container_width=True)
        else:
            st.info("No objects detected above the selected confidence threshold. Try lowering the threshold in the sidebar.")
    else:
        st.info("👆 Use the webcam capture above or upload an image to begin detection.")

    # Footer
    st.markdown("---")
    st.caption("AI Real-Time Object Detection System | Powered by Streamlit, Ultralytics YOLO, PyTorch, and OpenCV.")


if __name__ == "__main__":
    main()
