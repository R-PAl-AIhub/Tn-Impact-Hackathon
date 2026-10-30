# Backend

FastAPI service that receives one camera frame at a time and reports how many people are in view and how many of them are missing a helmet or a safety vest.

| File | What it does |
|---|---|
| `main.py` | FastAPI app. `POST /analyse` takes a JPEG in the multipart field `file`. It also serves the built dashboard from `dist/` when that folder exists. |
| `detector.py` | `PPEDetector`: loads `best.pt` with Ultralytics and runs it at confidence 0.20. |
| `utils.py` | Geometry rules: box overlap, and "helmet centre in the upper half of the person box". |
| `risk_engine.py` | `RiskEngine`: per-person deque of violations with Unix timestamps, a risk score over a 300 s window, and an escalation level. |
| `run.py` | Starts the server on `127.0.0.1:8000` and opens the dashboard in the browser. Entry point for the executable. |
| `run.spec` | PyInstaller spec. Bundles `dist/` and `best.pt` into `dist/run/run.exe`. |
| `best.pt` | Trained weights (YOLO26n, 5.4 MB). See [`../training`](../training). |

## Run

```bash
pip install -r requirements.txt

# optional: build the dashboard into backend/dist so the server can serve it
cd ../frontend && npm install && npm run build:backend && cd ../backend

python run.py            # http://127.0.0.1:8000
```

Without `dist/`, the server prints a warning and serves only the API.

## API

```bash
curl -F "file=@frame.jpg" http://127.0.0.1:8000/analyse
```

```json
{ "no_of_person": 2, "no_of_helmet_violation": 2, "no_of_vest_violation": 1 }
```

## Decision rules

For each detected person (class 6):

- **Helmet violation** when no helmet (class 0) has its centre inside the person box and in the upper half of it, or when a `no_helmet` box (class 7) overlaps the person.
- **Vest violation** when no vest box (class 2) overlaps the person box.

The model detects 11 classes. The backend uses only these four.

## Build the Windows executable

```bash
pip install pyinstaller
cd ../frontend && npm run build:backend && cd ../backend
pyinstaller run.spec      # output: dist/run/run.exe
```

The PyInstaller output folder (about 850 MB, most of it PyTorch) is too large for git. The team's build is on [Google Drive](https://drive.google.com/drive/folders/1Mh6IURDg4hLgYRlpoS8jp2zb3SqvBxwS?usp=sharing).
