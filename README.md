<p align="center">
  <img src="assets/diagrams/banner.png" alt="Real-time PPE violation detection. 3rd place, TN-IMPACT 2026." width="100%">
</p>

<p align="center">
  <a href="https://r-pal-aihub.github.io/Tn-Impact-Hackathon/"><b>Live demo</b></a> &nbsp;|&nbsp;
  <a href="docs/TN-IMPACT-2026-report.pdf">Project report</a> &nbsp;|&nbsp;
  <a href="docs/TN-IMPACT-2026-presentation.pdf">Slides</a> &nbsp;|&nbsp;
  <a href="https://youtu.be/gSikuVNupWM">Demo video 1</a> &nbsp;|&nbsp;
  <a href="https://youtu.be/zrvYNF5XT34">Demo video 2</a> &nbsp;|&nbsp;
  <a href="https://drive.google.com/drive/folders/1Mh6IURDg4hLgYRlpoS8jp2zb3SqvBxwS?usp=sharing">Windows build</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="Ultralytics YOLO" src="https://img.shields.io/badge/Ultralytics-YOLO26n-111F68?style=flat-square">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-backend-009688?style=flat-square&logo=fastapi&logoColor=white">
  <img alt="React" src="https://img.shields.io/badge/React-19-20232A?style=flat-square&logo=react&logoColor=61DAFB">
  <img alt="ONNX Runtime Web" src="https://img.shields.io/badge/ONNX%20Runtime-web-005CED?style=flat-square&logo=onnx&logoColor=white">
</p>

Construction sites depend on supervisors to check that every worker wears a helmet and a safety vest. Manual checks are slow and miss things.

This project watches a site camera and checks every person in view once per second. When someone is missing a helmet or a vest, the dashboard keeps the first frame as evidence and starts a timer. If the violation lasts 15 seconds, the frame is saved as a violation report.

The team built it in 2026 for **TN-IMPACT 2026**, problem statement **TNI26154**, and won **3rd place**.

> **AI for Safety Gear Detection:** Build a computer vision model that uses site CCTV feeds to detect workers not wearing helmets or harnesses in real-time and automatically logs the safety violation.

## Contents

- [See it work](#see-it-work)
- [How it works](#how-it-works)
- [Model and results](#model-and-results)
- [Validation from the report](#validation-from-the-report)
- [Run it](#run-it)
- [Repository layout](#repository-layout)
- [Limitations and next steps](#limitations-and-next-steps)
- [Team](#team)

## See it work

<table>
<tr>
<td width="36%" valign="top">
<img src="assets/demo-original-app.gif" alt="Recording of the original dashboard: a worker takes off his helmet, the dashboard keeps the first frame of the violation and counts the seconds." width="100%">
</td>
<td valign="top">

**The original app (React + FastAPI), recorded by the team**

1. The worker wears a helmet and a vest. The counts read 1 person, 0 violations.
2. He takes the helmet off. The dashboard keeps that frame in a red panel and starts a timer.
3. A second person without PPE walks in. The counts change to 2 people, 2 helmet violations, 2 vest violations.
4. After 15 seconds the browser downloads `violation_report_<date>.jpg`.

Full recordings: [demo video 1](https://youtu.be/gSikuVNupWM) (rooftop) and [demo video 2](https://youtu.be/zrvYNF5XT34) (overhead street view).

</td>
</tr>
</table>

### Try it in your browser

The [live demo](https://r-pal-aihub.github.io/Tn-Impact-Hackathon/) runs the same trained weights in your browser. You can use the sample clip, your webcam, or your own video or photo. No video leaves your device.

<a href="https://r-pal-aihub.github.io/Tn-Impact-Hackathon/"><img src="assets/web-demo.jpg" alt="Browser demo: hazard-striped camera frame, boxes on two people marked no helmet, counts 2 people, 2 no helmet, 1 no vest, and the evidence panel." width="100%"></a>

The demo uses the backend's 0.20 confidence threshold and its person–PPE rules, ported line for line. It can also send frames to a running FastAPI backend instead of the browser model.

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/diagrams/architecture-dark.png">
  <img alt="Architecture: site cameras feed the React frontend, which sends one JPEG frame per second to POST /analyse on the FastAPI backend. The backend decodes the frame, runs PPEDetector (YOLO26n, confidence 0.20), splits detections by class, applies per-person helmet and vest rules, logs violations in RiskEngine and returns JSON counts. The frontend's violation state manager keeps the first frame, downloads a report after 15 s, or clears when resolved." src="assets/diagrams/architecture-light.png" width="100%">
</picture>

| Step | Where | What happens |
|---|---|---|
| Capture | [`frontend/src/Camera.jsx`](frontend/src/Camera.jsx) | Draws the camera frame to a canvas once per second and posts it as a JPEG to `/analyse` |
| Detect | [`backend/detector.py`](backend/detector.py) | Runs the trained YOLO model at confidence 0.20 |
| Decide | [`backend/main.py`](backend/main.py), [`backend/utils.py`](backend/utils.py) | Checks every detected person for a helmet and a vest (rules below) |
| Log | [`backend/risk_engine.py`](backend/risk_engine.py) | Stores each violation with its Unix timestamp and computes a risk score over a 300 s window |
| Alert | [`frontend/src/Camera.jsx`](frontend/src/Camera.jsx) | Shows the counts, keeps the evidence frame, and downloads a report after 15 s |

### Decision rules

The model detects 11 classes. The backend uses four: `helmet` (0), `vest` (2), `Person` (6) and `no_helmet` (7).

| Violation | Raised when |
|---|---|
| No helmet | No helmet has its centre inside the person's box **and** in the upper half of it, **or** a `no_helmet` box overlaps the person |
| No vest | No vest box overlaps the person's box |

The upper-half rule stops a helmet held in the hand, or lying on the ground, from counting as worn. The middle frame below shows this case.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/detections-dark.jpg">
  <img alt="Three frames from the sample clip with real model output. Left: helmet and vest on, green box. Middle: helmet in hand, red box, no helmet. Right: a second person without PPE walks in, two red boxes." src="assets/detections-light.jpg" width="100%">
</picture>
<sub>Output of <code>backend/best.pt</code> with the backend's rules, made by <a href="scripts/make_figures.py"><code>scripts/make_figures.py</code></a>. Thin white boxes are the detected helmets and vests.</sub>

### Alert flow

```mermaid
stateDiagram-v2
    direction LR
    [*] --> Clear
    Clear --> Violation: helmet or vest count > 0<br/>keep this frame, start timer
    Violation --> Clear: counts back to 0<br/>discard frame
    Violation --> Report: 15 s without a break
    Report --> Clear: download violation_report_date.jpg
```

### API

```bash
curl -F "file=@frame.jpg" http://127.0.0.1:8000/analyse
```

```json
{ "no_of_person": 2, "no_of_helmet_violation": 2, "no_of_vest_violation": 1 }
```

## Model and results

| | |
|---|---|
| Model | Ultralytics YOLO26n, fine-tuned (`backend/best.pt`, 5.4 MB, 2.38 M parameters) |
| Dataset | [Ultralytics Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe/): 1,132 train, 143 val, 141 test images, 11 classes |
| Training | 100 epochs, batch 8, 640 px, on an RTX 4050 laptop GPU. The kept weights are from epoch 78. |
| Details | [`training/`](training) has the notebook, arguments, per-epoch log and Ultralytics plots |

<table>
<tr>
<td width="50%">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/class-ap-dark.png">
  <img alt="AP at IoU 0.5 for the four classes the app uses. Person 0.90 val, 0.80 test. Helmet 0.78 val, 0.92 test. Vest 0.83 val, 0.86 test. no_helmet 0.22 val, 0.22 test." src="assets/class-ap-light.png" width="100%">
</picture>
</td>
<td width="50%">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/training-dark.png">
  <img alt="Validation mAP per epoch over 100 epochs. mAP at 0.5 rises to about 0.54. mAP 0.5 to 0.95 rises to about 0.27. The best epoch is 78." src="assets/training-light.png" width="100%">
</picture>
</td>
</tr>
</table>

| Class | val AP@0.5 | test AP@0.5 | test precision | test recall |
|---|---:|---:|---:|---:|
| Person | 0.901 | 0.799 | 0.792 | 0.807 |
| helmet | 0.784 | 0.922 | 0.838 | 0.911 |
| vest | 0.830 | 0.856 | 0.780 | 0.837 |
| no_helmet | 0.218 | 0.216 | 0.371 | 0.200 |
| All 11 classes (mAP@0.5) | 0.536 | 0.507 | | |

The model finds people, helmets and vests well. It finds the explicit `no_helmet` class poorly. The backend does not depend on that class: it raises a helmet violation whenever **no helmet is found** on a person, and uses `no_helmet` only as extra evidence.

The test split was not used in training. Both columns come from `model.val()` on `best.pt` ([`training/eval_results.json`](training/eval_results.json)).

### One clip, second by second

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/timeline-dark.png">
  <img alt="Per-second counts on the 34-second sample clip. One person throughout until second 22. Helmet violation from second 11, with a one-second gap at 13. A second person appears from second 22 to 28 with extra helmet and vest violations. Counts return to one compliant person from second 29." src="assets/timeline-light.png" width="100%">
</picture>

The backend's counts at one frame per second on the [sample clip](web-demo/media/rooftop.mp4). One missed frame, as at second 13, resets the evidence timer. Smoothing over a few frames, or tracking people across frames, would prevent that.

## Validation from the report

The team tested the full app with five scenarios (report §6.1). All five gave the expected output.

| # | Scenario | Expected output | Method |
|---|---|---|---|
| 1 | One person with helmet and vest | 0 violations | Static image |
| 2 | One person without helmet | `helmet_violation = 1` | Static image |
| 3 | One person without vest | `vest_violation = 1` | Static image |
| 4 | Two people, one compliant, one not | 1 of each violation type | Multi-person image |
| 5 | Empty frame | All counts 0 | Edge case |

<img src="assets/report-test-cases.jpg" alt="Screenshots of the dashboard for test cases 1, 2, 4 and 5 from the report." width="100%">

Performance measured by the team on their test device (report §6.4):

| Metric | Value | Notes |
|---|---|---|
| Average inference time | 20–60 ms | Per frame |
| First-frame inference | 120–150 ms (GPU), 360–450 ms (CPU only) | Model warm-up |
| Processing capability | 14–16 FPS | Steady state |
| Average API response | 105 ms | Includes processing |
| Input frame rate | 1 FPS | Set by the frontend |
| Model size | 5,275 KB | Trained weights file |

The packaged executable also ran on a second machine without Python or Node.js installed (report §7.6).

## Run it

### Windows executable

1. Download the ZIP from [Google Drive](https://drive.google.com/drive/folders/1Mh6IURDg4hLgYRlpoS8jp2zb3SqvBxwS?usp=sharing) and extract it.
2. Open `CONSTRUCTION-PPE-DETECTION-main` → `dist` → `run`.
3. Run `run.exe` (about 48 MB). The dashboard opens at `http://127.0.0.1:8000`.

Requirements: Windows 10 or later, 8 GB RAM, Intel i5 or equivalent. An NVIDIA RTX GPU gives the best speed.

### From source

```bash
git clone https://github.com/R-PAl-AIhub/Tn-Impact-Hackathon.git
cd Tn-Impact-Hackathon

# backend + model
pip install -r backend/requirements.txt

# dashboard, built into backend/dist
cd frontend && npm install && npm run build:backend && cd ..

# start: serves the dashboard and the API on http://127.0.0.1:8000
cd backend && python run.py
```

For frontend development, run `npm run dev` in `frontend/`. It talks to the backend on port 8000.

### Browser demo, locally

```bash
cd web-demo && python -m http.server 8080      # open http://localhost:8080
```

[`scripts/export_onnx.py`](scripts/export_onnx.py) rebuilds `web-demo/model/best.onnx` from `backend/best.pt`. On the 34 one-second frames of the sample clip, the browser model and the PyTorch backend gave the same counts on 32 frames. The two differences were detections close to the 0.20 threshold.

## Repository layout

```text
.
├── backend/          FastAPI service, detector, rules, RiskEngine, PyInstaller spec, best.pt
├── frontend/         React + Vite dashboard (the app shown in the demo videos)
├── web-demo/         Static browser demo: ONNX model, sample clip, no server needed
├── training/         Training notebook, arguments, per-epoch log, plots, evaluation
├── docs/             Project report, slides, certificate, report screenshots
├── scripts/          Figure generation, ONNX export, diagram sources
└── assets/           Images used in this README
```

The code comes from two earlier repositories, combined here:
[R-PAl-AIhub/CONSTRUCTION-PPE-DETECTION](https://github.com/R-PAl-AIhub/CONSTRUCTION-PPE-DETECTION) (backend) and
[TarunSagarSah/tn_impact_frontend](https://github.com/TarunSagarSah/tn_impact_frontend) (frontend).
The backend files here are the versions used to build the Windows executable.

## Limitations and next steps

From testing (report §6.3) and from the results above:

- **Person IDs reset every frame.** `person_0`, `person_1` are list positions, so the RiskEngine history does not follow a real person across frames. Cross-frame tracking is the planned next step.
- **Camera angle matters.** Overhead and oblique views reduce the accuracy of the upper-half helmet rule. In demo video 2, a worker seen from directly above is not detected at first.
- **Dark or unusual vests** are sometimes missed (report §7.2).
- **The risk score is not shown yet.** `RiskEngine.compute_risk_score()` and `escalation_level()` exist but the API does not return them.
- **Cameras 2–4** are placeholder tiles in the dashboard. Only Camera 1 is analysed.

Planned work (report §8): a separate identification module after the alert step, for example a face-recognition API, access-control matching, RFID badges or gate attendance logs. The report explains why identification is hard on real sites: PPE hides faces, CCTV footage is low quality, and visitors are unregistered.

## Team

**Team Data Minds**, Vellore Institute of Technology, Vellore

| Member | Programme |
|---|---|
| [Ratul Pal](https://github.com/R-PAl-AIhub) | B.Tech CSE (AI & ML) |
| [Tarun Sagar Sah](https://github.com/TarunSagarSah) | B.Tech CSE (Blockchain) |
| Dibbankur Dey | B.Tech CSE (Data Science) |

Mentor: Dr. Anindita Kundu, Assistant Professor, VIT Vellore.

### Recognition

<img src="docs/certificate.jpg" alt="Certificate of Appreciation: Ratul Pal, CSE AIML, Vellore Institute of Technology, third place in TN-IMPACT 2026, conducted by TANCAM, Chennai with Sona College of Technology, Salem, 28 February 2026." width="420" align="right">

**3rd place, TN-IMPACT 2026.** TANCAM's hackathon for Tamil Nadu Industry Modernization, Problem-solving and Advanced Creative Tech. It was conducted by the Tamil Nadu Centre of Excellence for Advanced Manufacturing (TANCAM), Chennai, with Sona College of Technology, Salem, on 28 February 2026.

<br clear="right">

## Acknowledgements

[Ultralytics](https://github.com/ultralytics/ultralytics) (YOLO and the Construction-PPE dataset), [FastAPI](https://fastapi.tiangolo.com/), [OpenCV](https://opencv.org/), [React](https://react.dev/), [Vite](https://vite.dev/) and [ONNX Runtime](https://onnxruntime.ai/).

The weights were trained with Ultralytics YOLO, which is released under AGPL-3.0. See the [Ultralytics licence](https://www.ultralytics.com/license) for how it applies to trained models.
