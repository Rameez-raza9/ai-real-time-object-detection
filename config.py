"""Configuration module for AI Real-Time Object Detection Web Application.

Centralizes model settings, detection parameters, UI styling, and layout options.
"""

from pathlib import Path
from typing import List, Tuple

# Base Directories
BASE_DIR: Path = Path(__file__).resolve().parent
MODELS_DIR: Path = BASE_DIR / "models"
SCREENSHOTS_DIR: Path = BASE_DIR / "screenshots"

# Ensure required directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# Application Identity & Branding
# ==========================================
APP_TITLE: str = "AI Real-Time Object Detection"
APP_SUBTITLE: str = "Real-time computer vision using YOLO and OpenCV"
PAGE_ICON: str = "🎯"

# ==========================================
# Model Parameters
# ==========================================
# Ultralytics model weights. 'yolov8n.pt' is lightweight (~3.2M params),
# optimized for CPU / web hosting on Streamlit Community Cloud.
MODEL_NAME: str = "yolov8n.pt"

# Default confidence threshold (0.0 to 1.0)
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.50

# Default Intersection-over-Union (IoU) threshold for Non-Maximum Suppression (NMS)
DEFAULT_IOU_THRESHOLD: float = 0.45

# Inference dimension (square input dimension)
IMAGE_SIZE: int = 640

# Target compute device: None for automatic resolution ('cuda', 'mps', or 'cpu')
DEVICE: str | None = None

# ==========================================
# Visual & Bounding Box Styling
# ==========================================
BBOX_THICKNESS: int = 2
LABEL_FONT_SCALE: float = 0.55
LABEL_FONT_THICKNESS: int = 2

# Distinct high-contrast palette (BGR format for OpenCV drawing)
PALETTE_BGR: List[Tuple[int, int, int]] = [
    (56, 56, 255),    # Red-Orange
    (10, 249, 72),    # Lime Green
    (255, 143, 0),    # Deep Sky Blue
    (232, 35, 244),   # Purple
    (0, 215, 255),    # Amber
    (23, 204, 146),   # Mint Green
    (210, 0, 148),    # Hot Pink
    (187, 212, 0),    # Teal
    (29, 178, 255),   # Coral
    (124, 0, 186),    # Orchid
]


def get_color_for_class(class_id: int) -> Tuple[int, int, int]:
    """Retrieve consistent BGR color tuple for a given class ID."""
    return PALETTE_BGR[class_id % len(PALETTE_BGR)]
