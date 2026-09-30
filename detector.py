"""Object detection engine using Ultralytics YOLO.

Provides a modular ObjectDetector class handling model initialization,
image inference, structured output formatting, and clean bounding box annotation.
Separates deep learning computation completely from web UI logic.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union
import cv2
import numpy as np
from PIL import Image

# Suppress Ultralytics verbose output and prevent background auto-updates that can hang in cloud
os.environ["YOLO_VERBOSE"] = "False"
os.environ["ULTRALYTICS_AUTOINSTALL"] = "False"

try:
    import torch
    # Restrict PyTorch CPU worker threads to prevent thread contention on shared cloud vCPUs
    torch.set_num_threads(2)
    from ultralytics import YOLO
except ImportError as err:  # pragma: no cover
    raise ImportError(
        "Missing deep learning dependencies. Ensure 'ultralytics' and 'torch' are installed."
    ) from err

import config

logger = logging.getLogger(__name__)


class DetectionResult(TypedDict):
    """Schema representing a single detected object."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2] in pixel coordinates


class ObjectDetector:
    """Encapsulates YOLO object detection inference and annotation."""

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        conf_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = config.DEFAULT_IOU_THRESHOLD,
        device: Optional[str] = config.DEVICE,
        imgsz: int = config.IMAGE_SIZE,
    ) -> None:
        """Initialize detector with specified model parameters.

        Args:
            model_name: Ultralytics model file name or local weights path.
            conf_threshold: Minimum confidence score to retain detection (0.0 to 1.0).
            iou_threshold: IoU overlap threshold for Non-Maximum Suppression (NMS).
            device: Computing device ('cpu', 'cuda', 'mps', or None for auto).
            imgsz: Input dimension for model letterbox scaling.
        """
        self.model_name: str = model_name
        self.conf_threshold: float = conf_threshold
        self.iou_threshold: float = iou_threshold
        self.imgsz: int = imgsz

        self.device: str = self._resolve_device(device)
        self.model: YOLO = self._load_model(self.model_name)
        self.class_names: Dict[int, str] = self.model.names if hasattr(self.model, "names") else {}

        logger.info(
            "ObjectDetector initialized with model '%s' on device '%s' (conf=%.2f, iou=%.2f)",
            self.model_name,
            self.device,
            self.conf_threshold,
            self.iou_threshold,
        )

    def _resolve_device(self, requested_device: Optional[str]) -> str:
        """Resolve optimal available compute device."""
        if requested_device:
            return requested_device

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self, model_name: str) -> YOLO:
        """Load YOLO model weights with caching and fallback.

        Args:
            model_name: Weight file name or path.

        Returns:
            YOLO model instance.

        Raises:
            RuntimeError: If model cannot be loaded or downloaded.
        """
        try:
            candidates = [
                Path(model_name),
                config.MODELS_DIR / model_name,
                config.BASE_DIR / model_name,
            ]
            resolved_path: Optional[Path] = None
            for cand in candidates:
                if cand.is_file():
                    resolved_path = cand
                    break

            if resolved_path is not None:
                logger.info("Loading YOLO weights from local file: %s", resolved_path)
                model = YOLO(str(resolved_path))
            else:
                logger.info("Local weights not found. Loading via Ultralytics identifier: %s", model_name)
                model = YOLO(model_name)

            model.to(self.device)
            return model
        except Exception as e:
            msg = (
                f"Failed to load YOLO model '{model_name}'. "
                f"Verify internet connection for initial weight download or check file path. "
                f"Error: {e}"
            )
            logger.error(msg)
            raise RuntimeError(msg) from e

    def detect(
        self,
        image_input: Union[np.ndarray, Image.Image],
        conf_override: Optional[float] = None,
    ) -> Tuple[List[DetectionResult], float]:
        """Execute object detection inference on an image.

        Args:
            image_input: NumPy array (RGB or BGR) or PIL Image.
            conf_override: Optional confidence threshold to override default.

        Returns:
            Tuple of (detections_list, inference_time_seconds).

        Raises:
            ValueError: If input image is empty or invalid.
            RuntimeError: If model inference fails.
        """
        # Convert PIL to NumPy RGB if needed
        if isinstance(image_input, Image.Image):
            np_image = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            np_image = image_input
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

        if np_image is None or np_image.size == 0:
            raise ValueError("Input image is empty.")

        confidence_val = conf_override if conf_override is not None else self.conf_threshold

        try:
            import time

            start_time = time.perf_counter()
            results = self.model.predict(
                source=np_image,
                conf=confidence_val,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
            inference_time = time.perf_counter() - start_time

            detections: List[DetectionResult] = []
            if not results:
                return detections, inference_time

            result = results[0]
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                return detections, inference_time

            xyxy_array = boxes.xyxy.cpu().numpy()
            conf_array = boxes.conf.cpu().numpy()
            cls_array = boxes.cls.cpu().numpy().astype(int)

            h, w = np_image.shape[:2]

            for i in range(len(cls_array)):
                cls_id = int(cls_array[i])
                conf = float(conf_array[i])
                box = xyxy_array[i]

                x1, y1, x2, y2 = [int(round(c)) for c in box]
                # Clamp coordinates to image boundaries
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                class_name = self.class_names.get(cls_id, f"class_{cls_id}")

                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": round(conf, 4),
                        "bbox": [x1, y1, x2, y2],
                    }
                )

            return detections, inference_time

        except Exception as e:
            logger.error("Inference runtime error: %s", e)
            raise RuntimeError(f"Error during detection inference: {e}") from e

    def annotate_image(
        self,
        image_input: Union[np.ndarray, Image.Image],
        detections: List[DetectionResult],
    ) -> np.ndarray:
        """Draw bounding boxes and class labels with confidence on the image.

        Args:
            image_input: Base image (NumPy RGB or PIL Image).
            detections: List of detection result dictionaries.

        Returns:
            Annotated image as an RGB NumPy array.
        """
        if isinstance(image_input, Image.Image):
            base_img = np.array(image_input.convert("RGB"))
        else:
            base_img = image_input.copy()

        # Work in RGB
        annotated = base_img.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            class_id = det["class_id"]
            class_name = det["class_name"]
            confidence = det["confidence"]

            # config returns BGR, convert to RGB for PIL/Streamlit rendering
            b, g, r = config.get_color_for_class(class_id)
            color_rgb = (r, g, b)

            # Draw bounding box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color_rgb,
                thickness=config.BBOX_THICKNESS,
            )

            # Corner accents
            corner_len = min(18, max(5, int((x2 - x1) * 0.15)), max(5, int((y2 - y1) * 0.15)))
            accent_thickness = config.BBOX_THICKNESS + 1
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color_rgb, accent_thickness)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color_rgb, accent_thickness)
            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color_rgb, accent_thickness)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color_rgb, accent_thickness)
            cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), color_rgb, accent_thickness)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), color_rgb, accent_thickness)
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color_rgb, accent_thickness)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color_rgb, accent_thickness)

            # Format label badge: "Person 95%"
            pct = int(confidence * 100)
            label = f"{class_name.title()} {pct}%"

            (txt_w, txt_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                config.LABEL_FONT_SCALE,
                config.LABEL_FONT_THICKNESS,
            )

            # Position badge above box or inside if too close to top edge
            badge_y1 = max(0, y1 - txt_h - baseline - 6)
            badge_y2 = badge_y1 + txt_h + baseline + 6
            badge_x1 = x1
            badge_x2 = min(annotated.shape[1] - 1, x1 + txt_w + 12)

            cv2.rectangle(
                annotated,
                (badge_x1, badge_y1),
                (badge_x2, badge_y2),
                color_rgb,
                thickness=-1,
            )

            # High-contrast text color selection
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            txt_color = (15, 15, 15) if luminance > 140 else (255, 255, 255)

            cv2.putText(
                annotated,
                label,
                (badge_x1 + 6, badge_y2 - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.LABEL_FONT_SCALE,
                txt_color,
                config.LABEL_FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return annotated
