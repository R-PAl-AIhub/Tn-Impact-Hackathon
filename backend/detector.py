from ultralytics import YOLO
import numpy as np
import os

class PPEDetector:
    def __init__(self,model_path:str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found {model_path}")
    
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load the model: {e}")
        
    def detect(self, image):

    # ---- HANDLE INPUT TYPES ----
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            source = image

        elif isinstance(image, np.ndarray):
            source = image

        else:
            raise TypeError("Input must be file path or numpy image")

        # ---- RUN INFERENCE ----
        try:
            results = self.model(source, conf=0.20)
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")

        # ---- PARSE RESULTS ----
        detections = []

        for r in results:
            for box in r.boxes:
                detections.append({
                    "class_id": int(box.cls[0]),
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()
                })

        return detections


