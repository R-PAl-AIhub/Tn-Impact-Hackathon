from fastapi import FastAPI, UploadFile, File
import cv2
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from detector import PPEDetector
from risk_engine import RiskEngine
from utils import is_overlap, object_belongs_to_person
import numpy as np
import os
import sys

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- PATH FIX FOR EXE ----
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DIST_PATH = os.path.join(BASE_DIR, "dist")

# Load YOLO model
detector = PPEDetector(os.path.join(BASE_DIR, "best.pt"))

# Persistent risk tracking per person
person_risks = {}
person_risks.clear()

CLASS_MAP = {
    0: "helmet",
    2: "vest",
    6: "person",
    7: "no_helmet"
}

# --- API ROUTES ---
@app.post("/analyse")
async def analyze_frame(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    detection = detector.detect(frame)

    persons = [d for d in detection if CLASS_MAP.get(d["class_id"]) == "person"]
    helmets = [d for d in detection if CLASS_MAP.get(d["class_id"]) == "helmet"]
    vests = [d for d in detection if CLASS_MAP.get(d["class_id"]) == "vest"]
    no_helmets = [d for d in detection if CLASS_MAP.get(d["class_id"]) == "no_helmet"]

    helmet_violation_count = 0
    vest_violation_count = 0

    for idx, person in enumerate(persons):
        person_id = f"person_{idx}"
        pbox = person["bbox"]

        if person_id not in person_risks:
            person_risks[person_id] = RiskEngine()

        engine = person_risks[person_id]

        helmet_found = any(object_belongs_to_person(pbox, h["bbox"]) for h in helmets)
        explicit_no_helmet = any(is_overlap(pbox, nh["bbox"]) for nh in no_helmets)

        if not helmet_found or explicit_no_helmet:
            helmet_violation_count += 1
            engine.add_violation("helmet")

        vest_found = any(is_overlap(pbox, v["bbox"]) for v in vests)

        if not vest_found:
            vest_violation_count += 1
            engine.add_violation("harness")

    return {
        "no_of_person": len(persons),
        "no_of_helmet_violation": helmet_violation_count,
        "no_of_vest_violation": vest_violation_count,
    }

# --- SERVE FRONTEND LAST ---
if os.path.exists(DIST_PATH):
    # Serve everything in dist as static files
    app.mount("/", StaticFiles(directory=DIST_PATH, html=True), name="static")
else:
    print("WARNING: dist folder not found at", DIST_PATH)


