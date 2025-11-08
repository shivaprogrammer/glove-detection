#!/usr/bin/env python3
"""
detection_script.py (YOLOv8 version)

Usage:
python /data/m24csa029/MTP/glove_detection/detection_script.py \
  --input /data/datasets/data/glove_detection/test/images \
  --output /data/m24csa029/MTP/glove_detection/output \
  --weights /data/m24csa029/MTP/glove_detection/runs/detect/glove_binary2/weights/best.pt \
  --conf 0.25 \
  --device cuda:0 \
  --imgsz 640
"""
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import cv2

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="input folder with images")
    p.add_argument("--output", required=True, help="output folder for annotated images and logs")
    p.add_argument("--weights", required=True, help="path to YOLOv8 .pt weights")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--device", default="cpu", help="device, e.g. cpu or cuda:0")
    p.add_argument("--imgsz", type=int, default=640, help="inference size")
    return p.parse_args()

def ensure_dir(p:Path):
    p.mkdir(parents=True, exist_ok=True)

def collect_images(folder:Path):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp")
    files = []
    for e in exts:
        files += sorted(folder.glob(e))
    return files

def draw_boxes(img, detections):
    for det in detections:
        x1,y1,x2,y2 = map(int, det["bbox"])
        label = f'{det["label"]} {det["confidence"]:.2f}'
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, label, (x1, max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return img

def main():
    args = parse_args()
    inp = Path(args.input)
    out = Path(args.output)
    ann = out / "annotated"
    logs = out / "logs"
    ensure_dir(out); ensure_dir(ann); ensure_dir(logs)

    images = collect_images(inp)
    if len(images) == 0:
        print("No images found in", inp)
        return

    # import here so script fails fast if ultralytics not installed
    try:
        from ultralytics import YOLO
    except Exception as e:
        print("Ultralytics (YOLOv8) not installed or failed to import. Install with: pip install ultralytics")
        raise

    model = YOLO(args.weights)
    # set confidence threshold for prediction
    # use predict(..., conf=args.conf) below too
    results = model.predict(source=[str(p) for p in images], conf=args.conf, device=args.device, imgsz=args.imgsz, verbose=False)

    # results is a list of Results objects (one per image)
    for path, res in zip(images, results):
        detections = []
        if hasattr(res, "boxes") and res.boxes is not None:
            # res.boxes.xyxy, res.boxes.conf, res.boxes.cls
            xyxy = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes.xyxy, "cpu") else res.boxes.xyxy
            confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes.conf, "cpu") else res.boxes.conf
            clss = res.boxes.cls.cpu().numpy() if hasattr(res.boxes.cls, "cpu") else res.boxes.cls
            names = model.model.names if hasattr(model, "model") and hasattr(model.model, "names") else {}
            for b, c, cl in zip(xyxy, confs, clss):
                x1,y1,x2,y2 = [float(v) for v in b]
                conf = float(c)
                cls = int(cl)
                label = names.get(cls, str(cls))
                detections.append({
                    "label": label,
                    "confidence": round(conf, 4),
                    "bbox": [round(x1,2), round(y1,2), round(x2,2), round(y2,2)]
                })

        # save JSON
        json_log = {"filename": path.name, "detections": detections}
        with open(logs / (path.stem + ".json"), "w") as f:
            json.dump(json_log, f, indent=2)

        # annotate and save image
        img = cv2.imread(str(path))
        img = draw_boxes(img, detections)
        cv2.imwrite(str(ann / path.name), img)

    print("Done. Annotated images:", ann, "Logs:", logs)

if __name__ == "__main__":
    main()
