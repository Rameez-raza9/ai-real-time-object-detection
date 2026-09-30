# AI Real-Time Object Detection

A production-ready, interactive web application for real-time object detection built with **Python 3.11**, **Streamlit**, **Ultralytics YOLO**, **PyTorch**, and **OpenCV**.

Designed for cloud and local deployment, the system captures camera input directly through the web browser, processes frames with a single-shot neural network, and renders real-time bounding boxes, confidence tags, object counts, and inference metrics.

---

## Live Demo
* **Status**: Ready for Deployment on Streamlit Community Cloud
* **Live Demo URL**: Deploy following the instructions below to obtain your unique public URL (e.g., `https://ai-real-time-object-detection.streamlit.app`).

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Technology Stack](#technology-stack)
5. [Computer Vision Concepts Explained](#computer-vision-concepts-explained)
6. [Object Detection vs. Image Classification](#object-detection-vs-image-classification)
7. [Local Installation Guide (Windows & VS Code)](#local-installation-guide-windows--vs-code)
8. [Running the Application](#running-the-application)
9. [Deployment Guide (Streamlit Community Cloud)](#deployment-guide-streamlit-community-cloud)
10. [Troubleshooting Guide](#troubleshooting-guide)
11. [Resume-Ready Project Description](#resume-ready-project-description)
12. [60-Second Interview Explanation](#60-second-interview-explanation)
13. [20+ Technical Interview Questions & Answers](#20-technical-interview-questions--answers)
14. [Version 2 Roadmap](#version-2-roadmap)

---

## Project Overview

In real-world cloud environments, server-side camera capture using `cv2.VideoCapture(0)` fails because a remote Linux server cannot access a client's physical laptop webcam.

**AI Real-Time Object Detection** solves this architectural problem by decoupling video ingestion from backend processing. Streamlit's web components capture media directly inside the user's browser, transmit the frame buffer to the Python backend for deep learning inference, and display interactive visual feedback in milliseconds.

---

## Key Features

* **Browser-Native Webcam Input**: Seamless camera access via standard WebRTC browser permissions without native desktop drivers.
* **Instant Single-Shot Detection**: Powered by lightweight YOLOv8 Nano (~3.2M parameters) delivering 30+ FPS processing speeds on CPU.
* **Smart Resource Caching**: Uses `@st.cache_resource` to keep neural network weights in memory across reruns.
* **Interactive Threshold Controls**: Real-time slider adjustments for detection confidence and NMS IoU overlap.
* **Class Filtering**: Multi-select dropdown to isolate specific object categories (e.g. only people or cars).
* **Live Analytics HUD**: Metric cards showing object count, inference latency (ms), effective processing FPS, and average confidence.
* **Structured Data Inspection**: Interactive tabular display of detected objects, confidence scores, and pixel bounding box coordinates.
* **Flexible Input Methods**: Supports both live browser webcam capture and image file uploads (JPG, PNG, WEBP).

---

## System Architecture

```text
Browser User Interface
   │
   ▼
[1] Web Camera Ingestion (st.camera_input / HTML5 WebRTC)
   │ Transmits browser image buffer to Streamlit backend
   ▼
[2] Frame Preprocessing (PIL & NumPy)
   ├── Buffer conversion to NumPy array (H x W x 3)
   └── Normalization and letterbox scaling (640x640)
   │
   ▼
[3] YOLO Neural Network Inference (PyTorch & Ultralytics)
   ├── Single forward pass through CSPDarknet backbone
   └── Multi-scale feature extraction across 80 COCO classes
   │
   ▼
[4] Post-Processing & Filtering
   ├── Confidence Threshold filtering (User slider: 0.10 - 1.00)
   ├── Non-Maximum Suppression (NMS IoU: 0.45)
   └── Un-scaling coordinates to original image dimensions
   │
   ▼
[5] Visual Annotation & Metrics Computation
   ├── Draw color-coded bounding boxes with corner accents
   ├── Render adaptive-contrast class badges (e.g., "Person 95%")
   └── Compute aggregate latency and FPS metrics
   │
   ▼
[6] Streamlit Reactive Web UI
   └── Displays side-by-side comparison, metric tiles, and data table
```

---

## Technology Stack

| Technology | Purpose | Why It Was Chosen |
| :--- | :--- | :--- |
| **Python 3.11** | Core Runtime | High execution speed, modern typing, robust ecosystem support. |
| **Streamlit** | Web Framework | Rapid, reactive web UI designed for machine learning apps with browser webcam support. |
| **Ultralytics YOLO** | Detection Model | Modern single-stage detector with optimal balance between speed and precision. |
| **PyTorch** | Deep Learning Engine | Powers underlying tensor operations, model graph execution, and hardware acceleration hooks. |
| **OpenCV (Headless)** | Image Processing | Fast bounding box drawing and coordinate geometry without GUI display dependencies. |
| **NumPy** | Array Computing | High-performance vectorized matrix math on multi-dimensional image arrays. |
| **Pillow (PIL)** | Image I/O | Standard library for loading, formatting, and encoding browser image streams. |

---

## Computer Vision Concepts Explained

1. **Pixel**: The smallest individual unit of programmable color in a digital raster image.
2. **Image**: A discrete 2D grid of pixels represented as a 3D numerical matrix of shape `(Height, Width, Channels)`.
3. **Frame**: A single still image from a sequence of video images captured over time.
4. **Object Detection**: A computer vision task that predicts both **classification** (*what object is present*) and **localization** (*where is it in the image*).
5. **Bounding Box**: A rectangular coordinate boundary enclosing a detected target, expressed as $[x_1, y_1, x_2, y_2]$ (top-left and bottom-right pixel coordinates).
6. **Confidence Score**: A probability score between $0.0$ and $1.0$ expressing the model's certainty that a predicted bounding box contains a valid object.
7. **Intersection over Union (IoU)**: Mathematical measure of overlap between two boxes:
   $$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}$$
8. **Non-Maximum Suppression (NMS)**: An algorithm that eliminates redundant, overlapping bounding boxes that predict the same physical object, keeping only the highest-scoring candidate.
9. **Inference**: Running an input image through a trained neural network to generate predictions without modifying the network's weights.
10. **FPS (Frames Per Second)**: The frequency at which consecutive images can be processed:
    $$\text{FPS} = \frac{1000}{\text{Latency in milliseconds}}$$
11. **YOLO (You Only Look Once)**: A single-stage neural network that predicts bounding boxes and class probabilities simultaneously across the entire image grid in a single evaluation.

---

## Object Detection vs. Image Classification

| Dimension | Image Classification | Object Detection |
| :--- | :--- | :--- |
| **Question Answered** | *"What is the primary subject?"* | *"What objects are present and where are they located?"* |
| **Output** | Single class label + confidence score | Multiple bounding boxes $[x_1, y_1, x_2, y_2]$ + labels + confidences |
| **Multiple Items** | Cannot distinguish multiple distinct items | Detects and localizes multiple co-occurring items simultaneously |
| **Spatial Output** | None (no position coordinates) | Full pixel coordinates for every detected instance |
| **Use Case** | Identifying breed of a single dog | Autonomous vehicle detecting pedestrians, cars, and traffic lights |

---

## Local Installation Guide (Windows & VS Code)

### Step 1: Open Terminal
Open VS Code, press `Ctrl + ~` to open the integrated terminal.

### Step 2: Navigate to Project
```powershell
cd ai-real-time-object-detection
```

### Step 3: Create Virtual Environment
```powershell
py -3.11 -m venv venv
```

### Step 4: Activate Virtual Environment
* On Windows PowerShell:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
* On Windows Command Prompt:
  ```cmd
  venv\Scripts\activate.bat
  ```

### Step 5: Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Application

Launch the Streamlit web server:
```powershell
streamlit run app.py
```

Streamlit will print the local URL to the terminal:
```text
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```
Open `http://localhost:8501` in your browser. When prompted, allow camera permissions to capture frames.

---

## Deployment Guide (Streamlit Community Cloud)

Follow these steps to deploy your project to a public URL:

### Step 1: Push Project to GitHub
```bash
git init
git add .
git commit -m "Initial commit - real time object detection"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/ai-real-time-object-detection.git
git push -u origin main
```

### Step 2: Open Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
2. Click **Create app** (or **New app**).
3. Select your repository: `YOUR_GITHUB_USERNAME/ai-real-time-object-detection`.
4. Set **Branch**: `main`.
5. Set **Main file path**: `app.py`.
6. Click **Deploy**.

### Step 3: Access Public URL
Once built, Streamlit will assign a public domain:
`https://ai-real-time-object-detection.streamlit.app`

---

## Troubleshooting Guide

1. **Python Version**: Ensure Python 3.11 is used (`py -3.11 --version`). Python 3.6 or older is incompatible with modern PyTorch and Streamlit.
2. **Missing `libGL.so` on Linux / Cloud**: The project specifies `opencv-python-headless` in `requirements.txt` to avoid GUI dependencies on headless cloud servers.
3. **Camera Permission Denied**: If the webcam does not open in your browser, check your browser's site settings (`Settings > Privacy and security > Site settings > Camera`) and ensure camera permissions are allowed for `localhost` or your Streamlit domain.
4. **Model Download Interruption**: On first run, YOLOv8 Nano weights are downloaded from GitHub releases. If an internet timeout occurs, re-run `streamlit run app.py` or manually place `yolov8n.pt` in the project root.
5. **Port in Use**: If port 8501 is occupied, launch on another port:
   ```bash
   streamlit run app.py --server.port 8502
   ```

---

## Resume-Ready Project Description

> **"Developed a real-time computer vision application using YOLO, OpenCV, PyTorch, and Streamlit to detect and classify multiple objects from camera input. Implemented bounding-box visualization, confidence scoring, object counting, configurable detection thresholds, and an interactive web interface."**

---

## 60-Second Interview Explanation

> *"I designed and built an interactive AI Real-Time Object Detection Web Application using Python, Streamlit, Ultralytics YOLO, and PyTorch. The motivation was to build a computer vision service that works seamlessly in cloud environments without making faulty assumptions about server-side hardware access.
>
> In production, cloud servers cannot invoke `cv2.VideoCapture(0)` to read a client's camera. I resolved this by designing a browser-native ingestion pipeline using Streamlit's camera components. The image stream is captured via client-side WebRTC, passed to a modular `ObjectDetector` backend, and evaluated in a single forward pass by a cached YOLOv8 model.
>
> The system performs letterbox scaling, confidence thresholding, and Non-Maximum Suppression to deliver real-time bounding boxes and class metrics at 30+ FPS. The architecture is modular, completely typed, container-ready, and optimized for cloud hosting."*

---

## 20+ Technical Interview Questions & Answers

### 1. What is Computer Vision?
Computer Vision is an interdisciplinary field of artificial intelligence enabling algorithms to extract meaningful, high-level structured representations from digital visual inputs such as images and video streams.

### 2. What is object detection?
Object detection is a core computer vision task that combines classification and localization to identify instances of semantic objects belonging to predefined categories and compute spatial coordinates for each instance.

### 3. What is the fundamental difference between image classification and object detection?
Classification determines the primary category of an image as a whole without spatial context. Object detection identifies and determines precise bounding box coordinates for multiple co-occurring objects.

### 4. What is YOLO and how does it work?
YOLO (You Only Look Once) is an anchor-free single-stage object detector that reframes detection as a single regression problem. It passes an image through a convolutional backbone once, predicting bounding box offsets and class probabilities simultaneously across a feature grid.

### 5. Why choose YOLO over two-stage detectors like Faster R-CNN?
Two-stage detectors generate region proposals with an RPN before classifying each proposal, leading to high latency ($<10\text{ FPS}$). YOLO evaluates the entire frame in a single pass, delivering $30-100+\text{ FPS}$ speeds essential for real-time video.

### 6. What is a bounding box?
A bounding box is a rectangular envelope enclosing an object, defined by four coordinates: either top-left and bottom-right points $[x_1, y_1, x_2, y_2]$ or center and dimensions $[x_c, y_c, w, h]$.

### 7. What does the confidence score represent?
The confidence score equals the probability that an anchor point contains an object multiplied by the conditional class probability:
$$\text{Confidence} = P(\text{Object}) \times P(\text{Class}_i \mid \text{Object})$$

### 8. What is Intersection-over-Union (IoU)?
IoU measures the spatial overlap ratio between a predicted bounding box $B_p$ and a target box $B_g$:
$$\text{IoU} = \frac{\text{Area}(B_p \cap B_g)}{\text{Area}(B_p \cup B_g)}$$

### 9. What is Non-Maximum Suppression (NMS) and why is it required?
Convolutional networks often output multiple overlapping candidate boxes for the same physical object. NMS iteratively suppresses lower-confidence boxes that overlap with the top prediction above an IoU threshold (e.g. $0.45$).

### 10. What is neural network inference?
Inference is the forward evaluation pass where input data traverses a trained neural network to produce predictions, with gradient tracking disabled and weights fixed.

### 11. Why use OpenCV in this project?
OpenCV provides optimized, C++-accelerated computer vision primitives for coordinate transformations, array manipulation, and fast graphical bounding box annotations.

### 12. Why PyTorch?
PyTorch is the underlying deep learning engine powering Ultralytics YOLO. It handles hardware abstraction (CPU, CUDA, MPS) and dynamic tensor operations.

### 13. Why Streamlit for deployment?
Streamlit allows rapid development of reactive, web-native ML applications with built-in browser webcam support, eliminating complex frontend boilerplate.

### 14. How does the webcam input work in this web app?
The browser uses HTML5 / WebRTC APIs to request local camera permission. When a frame is captured, the browser encodes it into an image buffer and transmits it via HTTP/WebSocket to the Python server.

### 15. Why can't `cv2.VideoCapture(0)` directly access a user's webcam on a cloud server?
`cv2.VideoCapture(0)` queries local operating system device drivers (`/dev/video0` or DirectShow) attached physically to the machine executing the Python process. On a remote cloud VM (e.g. AWS or Streamlit Cloud), index 0 refers to the server's nonexistent hardware, not the user's laptop.

### 16. How did you optimize inference performance?
1. Selected YOLOv8 Nano (`yolov8n.pt`) with only ~3.2M parameters.
2. Cached the detector model using `@st.cache_resource` so weights stay resident in memory.
3. Used `imgsz=640` to balance resolution and latency.

### 17. How could you improve detection accuracy?
1. Fine-tune on custom domain datasets with transfer learning.
2. Upgrade to `yolov8s.pt` or `yolov8m.pt` if GPU hardware is available.
3. Apply test-time augmentation (TTA).

### 18. How do you train YOLO on a custom dataset?
1. Collect and annotate images in YOLO format (normalized $[class, x_c, y_c, w, h]$).
2. Create a `data.yaml` file defining train/val paths and class names.
3. Train with `yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640`.

### 19. How would you deploy this system at enterprise scale?
1. Wrap model inference inside a high-throughput FastAPI microservice running on GPU clusters.
2. Convert weights to NVIDIA TensorRT (`.engine`) with FP16/INT8 quantization.
3. Ingest video via RTSP or WebRTC streams orchestrated by Kubernetes.

### 20. What happens if the model fails to load?
The application employs defensive exception handling: if weights cannot be loaded or downloaded, the exception is trapped and a user-friendly error message is displayed in the UI rather than crashing the web server with an uncaught traceback.

---

## Version 2 Roadmap

1. **ByteTrack Multi-Object Tracking**: Maintain identity continuity across frames to count unique visitors rather than instantaneous detections.
2. **Virtual Tripwires**: Define boundary lines across the video to track entry/exit events.
3. **Custom Fine-Tuning**: Support custom-trained weights for specialized domains (e.g. solar panel defect inspection).
4. **TensorRT INT8 Export**: Optimize for microsecond latency on edge hardware like NVIDIA Jetson.
5. **Database Logging**: Store historical detection counts and peak activity trends in PostgreSQL.
