# Gloved vs Ungloved Hand Detection

## Overview
This project aims to automatically detect whether a worker is wearing gloves in an image.  
It helps ensure safety compliance in factory or industrial settings by identifying two categories of hands:
- gloved_hand
- bare_hand

The model is trained and fine-tuned using YOLOv8 for robust and real-time object detection.

---

## Dataset

- **Dataset Name:** Gloves and Bare Hands Detection  
- **Source:** [Roboflow Universe Dataset](https://universe.roboflow.com/dolphin-nog9y/gloves-and-bare-hands-detection)  
- **Format:** YOLOv5/YOLOv8 compatible (PyTorch format)  
- **Path:** `/data/datasets/data/glove_detection`  
- **Classes:**
  - 0: gloved_hand  
  - 1: bare_hand  
- **Structure:**
  ```
  train/images, train/labels
  valid/images, valid/labels
  test/images, test/labels
  data.yaml
  ```

---

## Model Used

- **Base Model:** YOLOv8n (Ultralytics)  
- **Framework:** PyTorch (Ultralytics YOLOv8)  
- **Training Device:** NVIDIA RTX A6000 (CUDA)  
- **Weights File:** `runs/detect/glove_binary2/weights/best.pt`  
- **Parameters:** Approximately 3.0M (8.1 GFLOPs)  
- **Training Duration:** 30 epochs  

### Final Results
| Metric | Score |
|---------|--------|
| mAP@0.5 | 0.927 |
| mAP@0.5:0.95 | 0.672 |
| Precision | 0.895 |
| Recall | 0.878 |

---

## Preprocessing and Training Details

### Preprocessing
- Images resized to 640×640  
- Augmentation: random flip, rotation, brightness, and scaling (handled by YOLOv8)  
- Normalized image pixel values  
- Dataset simplified to two classes: gloved_hand and bare_hand  

### Training Command
```bash
yolo task=detect mode=train \
  model=yolov8n.pt \
  data=/data/datasets/data/glove_detection/data.yaml \
  epochs=30 \
  imgsz=640 \
  batch=16 \
  name=glove_binary2
```

### Validation Command
```bash
yolo task=detect mode=val \
  model=runs/detect/glove_binary2/weights/best.pt \
  data=/data/datasets/data/glove_detection/data.yaml \
  device=cuda:0
```

---

## What Worked Well
- Fine-tuning YOLOv8n achieved excellent accuracy while remaining computationally efficient.
- Using Roboflow augmentations improved generalization and reduced overfitting.
- Balanced dataset and 30-epoch training produced strong performance (mAP@0.5 ≈ 0.93).
- Model handled different glove colors, lighting conditions, and backgrounds effectively.

---

## What Did Not Work / Limitations
- Minor false negatives when hands are small or partially visible.
- Duplicate detections occasionally appeared when two boxes overlapped heavily.
- Some performance drop observed in motion-blurred or low-light images.
- Could be improved with higher-resolution data and longer training.

---

## How to Run the Script

### 1. Environment Setup
```bash
conda create -n glove_detection python=3.9 -y
conda activate glove_detection
pip install ultralytics==8.2.2 opencv-python torch torchvision tqdm numpy
```

### 2. Run Detection
```bash
python detection_script.py \
  --input /data/datasets/data/glove_detection/test/images \
  --output /data/m24csa029/MTP/glove_detection/output \
  --weights /data/m24csa029/MTP/glove_detection/runs/detect/glove_binary2/weights/best.pt \
  --conf 0.25 \
  --device cuda:0 \
  --imgsz 640
```

### 3. Output
- Annotated images are saved in `/output/annotated/`
- Detection logs (JSON) are saved in `/output/logs/`
- Each JSON log follows this structure:

```json
{
  "filename": "image1.jpg",
  "detections": [
    {"label": "gloved_hand", "confidence": 0.92, "bbox": [x1, y1, x2, y2]},
    {"label": "bare_hand", "confidence": 0.85, "bbox": [x1, y1, x2, y2]}
  ]
}
```

---

## Project Structure
```
Part_1_Glove_Detection/
├── detection_script.py
├── train_glove_detection.py
├── output/
│   ├── annotated/
│   └── logs/
├── runs/
│   └── detect/glove_binary2/weights/best.pt
├── scripts/
├── yolo5/
├── yolov8n.pt
├── yolo11n.pt
└── README.md
```

---

## How to Reproduce Training

### Clone the YOLOv8 Repository
```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
```

### Train Your Model
```bash
yolo task=detect mode=train \
  model=yolov8n.pt \
  data=/data/datasets/data/glove_detection/data.yaml \
  epochs=30 \
  imgsz=640 \
  batch=16 \
  name=glove_binary2
```

### Evaluate
```bash
yolo task=detect mode=val \
  model=runs/detect/glove_binary2/weights/best.pt \
  data=/data/datasets/data/glove_detection/data.yaml
```

---

## Author
- **Name:** Shivani Tiwari
- **Project:**  Gloved vs Ungloved Hand Detection
- **Institute:** IIT Jodhpur
