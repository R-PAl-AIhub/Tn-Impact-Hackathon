"""Export backend/best.pt to web-demo/model/best.onnx for the browser demo.

The YOLO26n head is end-to-end (no NMS step), so the ONNX output is
[1, 300, 6] = x1, y1, x2, y2, score, class in 640x640 letterboxed pixels.

Usage (from the repo root):  python scripts/export_onnx.py
Needs: ultralytics==8.4.10, onnx, onnxslim
"""
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
onnx_path = YOLO(str(ROOT / "backend" / "best.pt")).export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False)
target = ROOT / "web-demo" / "model" / "best.onnx"
shutil.move(onnx_path, target)
print("wrote", target)
