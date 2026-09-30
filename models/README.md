# Models Directory

This folder is designated for caching or storing deep learning model weights.

## Default Model: YOLOv8 Nano (`yolov8n.pt`)

The web application uses **Ultralytics YOLOv8 Nano** by default:
* **Architecture**: Anchor-free Single-Stage Convolutional Detector
* **Parameters**: ~3.2 Million
* **Weights Size**: ~6.2 MB
* **Inference Latency**: 15–35 ms on modern CPUs; < 5 ms on CUDA GPUs
* **Pretrained Dataset**: COCO (80 distinct categories)

## Automatic Download
When the application starts, Ultralytics automatically checks if `yolov8n.pt` is present. If missing, it downloads the official pre-trained weights from GitHub releases and places them in the application root or local cache.

## Custom Models
To deploy a custom fine-tuned model (e.g. PPE detection, manufacturing defect inspection), drop your `best.pt` file into this directory and update `MODEL_NAME` in `config.py`.
