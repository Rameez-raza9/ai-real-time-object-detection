"""Performance metrics and detection aggregation utilities.

Provides helper functions for calculating inference latency, frames per second (FPS),
class frequency distributions, and formatted summary objects for Streamlit rendering.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class DetectionSummary:
    """Structured summary of detections for a single frame."""
    total_count: int
    class_counts: Dict[str, int]
    top_detections: List[Dict[str, Any]]
    average_confidence: float


def compute_detection_summary(detections: List[Dict[str, Any]]) -> DetectionSummary:
    """Calculate aggregated metrics from a list of structured detections.

    Args:
        detections: List of detection result dictionaries.

    Returns:
        DetectionSummary containing counts, class distribution, and average confidence.
    """
    if not detections:
        return DetectionSummary(
            total_count=0,
            class_counts={},
            top_detections=[],
            average_confidence=0.0,
        )

    total_count = len(detections)
    class_counts = dict(Counter(d["class_name"] for d in detections))
    sorted_detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    avg_conf = sum(d["confidence"] for d in detections) / total_count

    return DetectionSummary(
        total_count=total_count,
        class_counts=class_counts,
        top_detections=sorted_detections,
        average_confidence=avg_conf,
    )


def format_latency_and_fps(latency_seconds: float) -> Tuple[float, float]:
    """Compute latency in milliseconds and equivalent processing FPS.

    Args:
        latency_seconds: Elapsed processing time for inference in seconds.

    Returns:
        Tuple of (latency_ms, equivalent_fps).
    """
    latency_ms = max(0.001, latency_seconds * 1000.0)
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
    return round(latency_ms, 1), round(fps, 1)
