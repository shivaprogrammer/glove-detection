#!/usr/bin/env python3
"""
train_glove_detection.py
Simple script to train YOLOv8 on your dataset. If ultralytics is not installed,
it will print instructions for using the yolov5 repo.

Usage:
    python train_glove_detection.py --data data/data_glove.yaml --model yolov8n.pt --epochs 30 --imgsz 640 --batch 16 --project runs/train
"""
import argparse
import os
import sys
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/data_glove.yaml", help="path to data yaml")
    p.add_argument("--model", default="yolov8n.pt", help="pretrained model (yolov8n.pt) or path")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--project", default="runs/train", help="project folder for runs")
    p.add_argument("--name", default="glove_detector", help="experiment name")
    return p.parse_args()

def main():
    args = parse_args()

    # Try using ultralytics YOLO (YOLOv8)
    try:
        from ultralytics import YOLO
    except Exception as e:
        print("Ultralytics (YOLOv8) not available in this environment.")
        print("If you prefer to train with YOLOv5, clone the repo and run train.py:")
        print("  git clone https://github.com/ultralytics/yolov5.git && cd yolov5")
        print("  pip install -r requirements.txt")
        print(f"  python train.py --img {args.imgsz} --batch {args.batch} --epochs {args.epochs} --data {args.data} --weights yolov5s.pt --project {args.project} --name {args.name}")
        sys.exit(1)

    # Using YOLOv8 API
    print("Using ultralytics YOLOv8 API for training...")
    model = YOLO(args.model)  # loads yolov8n.pt or custom .pt
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project=args.project,
    )
    print("Training started. Check the runs/train directory for outputs.")

if __name__ == "__main__":
    main()
