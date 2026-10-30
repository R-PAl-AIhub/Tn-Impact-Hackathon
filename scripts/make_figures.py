"""
Builds the result figures in assets/ from real model output and logged training data.

  1. assets/detections-*.png   best.pt on three frames of web-demo/media/rooftop.mp4,
                                with the per-person verdicts from backend/main.py
  2. assets/timeline-*.png     per-second counts on the same clip (1 frame per second,
                                the rate the frontend uses)
  3. assets/training-*.png     mAP per epoch, from training/results.csv
  4. assets/class-ap-*.png     AP50 per class on the val and test splits,
                                from training/eval_results.json (Ultralytics `model.val`)
  5. scripts/diagrams/banner-frame.jpg  annotated frame for the README banner

Usage (from the repo root):  python scripts/make_figures.py
Needs: ultralytics, opencv-python, matplotlib
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from utils import is_overlap, object_belongs_to_person  # noqa: E402  (the backend's own rules)
from ultralytics import YOLO  # noqa: E402

OUT = ROOT / "assets"
CLIP = ROOT / "web-demo" / "media" / "rooftop.mp4"
CLASS_MAP = {0: "helmet", 2: "vest", 6: "person", 7: "no_helmet"}

for f in Path("/usr/share/fonts/opentype/inter").glob("Inter-*.otf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams["font.family"] = ["Inter", "DejaVu Sans"]

THEMES = {
    "light": dict(bg="#ffffff", ink="#1d272d", ink2="#46525a", muted="#6a757c", grid="#e3e4e0",
                  axis="#b9bcb5", s=["#2a78d6", "#eb6834", "#1baf7a"], bad="#c8102e", ok="#2e7d32", shade="#fde7ea"),
    "dark": dict(bg="#0d1117", ink="#eef0ec", ink2="#c2c8c4", muted="#949d9a", grid="#262c33",
                 axis="#3a454c", s=["#3987e5", "#d95926", "#199e70"], bad="#ff5a6b", ok="#6cc070", shade="#3a1d22"),
}


# ---------------------------------------------------------------- model + backend rules
model = YOLO(str(ROOT / "backend" / "best.pt"))


def detect(frame):
    out = []
    for r in model(frame, conf=0.20, verbose=False):          # backend/detector.py
        for b in r.boxes:
            out.append(dict(class_id=int(b.cls[0]), confidence=float(b.conf[0]), bbox=b.xyxy[0].tolist()))
    return out


def analyse(dets):                                              # backend/main.py analyze_frame()
    pick = lambda n: [d for d in dets if CLASS_MAP.get(d["class_id"]) == n]
    persons, helmets, vests, nh = pick("person"), pick("helmet"), pick("vest"), pick("no_helmet")
    verdicts = []
    for p in persons:
        b = p["bbox"]
        no_helmet = not any(object_belongs_to_person(b, h["bbox"]) for h in helmets) or any(is_overlap(b, n["bbox"]) for n in nh)
        no_vest = not any(is_overlap(b, v["bbox"]) for v in vests)
        verdicts.append(dict(bbox=b, no_helmet=no_helmet, no_vest=no_vest))
    return dict(persons=len(persons), helmet=sum(v["no_helmet"] for v in verdicts),
                vest=sum(v["no_vest"] for v in verdicts), verdicts=verdicts, dets=dets)


def frame_at(t):
    cap = cv2.VideoCapture(str(CLIP))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * cap.get(cv2.CAP_PROP_FPS)))
    ok, f = cap.read()
    cap.release()
    return f


def clip_seconds():
    cap = cv2.VideoCapture(str(CLIP))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return n


# ---------------------------------------------------------------- 1. detections
def draw(frame, res):
    img = frame.copy()
    for d in res["dets"]:                                       # PPE items, thin
        name = CLASS_MAP.get(d["class_id"])
        if name in ("helmet", "vest", "no_helmet"):
            x1, y1, x2, y2 = map(int, d["bbox"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (225, 225, 225), 1, cv2.LINE_AA)
    for v in res["verdicts"]:                                   # people, coloured by verdict
        x1, y1, x2, y2 = map(int, v["bbox"])
        bad = v["no_helmet"] or v["no_vest"]
        col = (46, 16, 200) if bad else (50, 125, 46)
        label = " + ".join(t for t, f in (("no helmet", v["no_helmet"]), ("no vest", v["no_vest"])) if f) or "helmet + vest"
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1 - th - 8, 0)
        lx = max(0, min(x1, img.shape[1] - tw - 8))
        cv2.rectangle(img, (lx, ty), (lx + tw + 8, ty + th + 8), col, -1)
        cv2.putText(img, label, (lx + 4, ty + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def chart_detections(theme, picks):
    t = THEMES[theme]
    fig, axes = plt.subplots(1, len(picks), figsize=(11, 6.6), dpi=170)
    fig.patch.set_facecolor(t["bg"])
    for ax, (sec, title) in zip(axes, picks):
        res = analyse(detect(frame_at(sec)))
        ax.imshow(draw(frame_at(sec), res))
        ax.set_axis_off()
        ax.set_title(title, color=t["ink"], fontsize=12, fontweight="bold", loc="left", pad=8)
        verdict = (f"{res['persons']} {'person' if res['persons'] == 1 else 'people'} · "
                   f"no helmet {res['helmet']} · no vest {res['vest']}")
        ax.text(0, -0.035, verdict, transform=ax.transAxes, color=t["ink2"], fontsize=10.5, va="top")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.07, wspace=0.05)
    fig.savefig(OUT / f"detections-{theme}.jpg", facecolor=t["bg"], pil_kwargs={"quality": 86})
    plt.close(fig)


# ---------------------------------------------------------------- 2. timeline
def timeline_data():
    rows = []
    for s in range(clip_seconds()):
        r = analyse(detect(frame_at(s)))
        rows.append((s, r["persons"], r["helmet"], r["vest"]))
    return rows


def chart_timeline(theme, rows):
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=170)
    fig.patch.set_facecolor(t["bg"]); ax.set_facecolor(t["bg"])
    xs = [r[0] for r in rows]
    for s, *_ in [r for r in rows if r[2] or r[3]]:
        ax.axvspan(s - 0.5, s + 0.5, color=t["shade"], lw=0, zorder=0)
    series = [("People in view", 1, t["s"][0]), ("No helmet", 2, t["s"][1]), ("No vest", 3, t["s"][2])]
    for name, i, col in series:
        ax.step(xs, [r[i] for r in rows], where="mid", color=col, lw=2, label=name, zorder=3)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], length=0)
    ax.set_yticks([0, 1, 2, 3]); ax.yaxis.grid(True, color=t["grid"], lw=0.8); ax.set_axisbelow(True)
    ax.set_xlim(-0.5, xs[-1] + 0.5); ax.set_ylim(-0.15, 3.4)
    ax.set_xlabel("Seconds into the sample clip (one frame analysed per second)", color=t["muted"], fontsize=10)
    leg = ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=10, bbox_to_anchor=(0, 1.2))
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    fig.text(0.995, 0.93, "shaded = at least one violation", ha="right", color=t["muted"], fontsize=9.5)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.82, bottom=0.17)
    fig.savefig(OUT / f"timeline-{theme}.png", facecolor=t["bg"])
    plt.close(fig)


# ---------------------------------------------------------------- 3. training curves
def chart_training(theme):
    t = THEMES[theme]
    rows = list(csv.DictReader(open(ROOT / "training" / "results.csv")))
    ep = [int(r["epoch"]) for r in rows]
    m50 = [float(r["metrics/mAP50(B)"]) for r in rows]
    m5095 = [float(r["metrics/mAP50-95(B)"]) for r in rows]
    best = max(range(len(rows)), key=lambda i: m5095[i])
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=170)
    fig.patch.set_facecolor(t["bg"]); ax.set_facecolor(t["bg"])
    ax.plot(ep, m50, color=t["s"][0], lw=2, label="mAP@0.5")
    ax.plot(ep, m5095, color=t["s"][1], lw=2, label="mAP@0.5:0.95")
    ax.scatter([ep[best]], [m5095[best]], s=46, color=t["s"][1], zorder=4, edgecolor=t["bg"], linewidth=2)
    ax.scatter([ep[best]], [m50[best]], s=46, color=t["s"][0], zorder=4, edgecolor=t["bg"], linewidth=2)
    ax.annotate(f"best.pt, epoch {ep[best]}\nmAP@0.5 {m50[best]:.3f} · mAP@0.5:0.95 {m5095[best]:.3f}",
                (ep[best], m5095[best]), xytext=(0, -34), textcoords="offset points",
                ha="center", color=t["ink2"], fontsize=9.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], length=0)
    ax.yaxis.grid(True, color=t["grid"], lw=0.8); ax.set_axisbelow(True)
    ax.set_xlim(0, ep[-1] + 2); ax.set_ylim(0, 0.62)
    ax.set_xlabel("Epoch", color=t["muted"], fontsize=10)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=10)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    ax.set_title("Validation mAP, all 11 classes", loc="left", color=t["ink"], fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / f"training-{theme}.png", facecolor=t["bg"])
    plt.close(fig)


# ---------------------------------------------------------------- 4. per-class AP
def chart_class_ap(theme):
    t = THEMES[theme]
    ev = json.load(open(ROOT / "training" / "eval_results.json"))
    classes = [("Person", "person"), ("helmet", "helmet"), ("vest", "vest"), ("no_helmet", "no_helmet")]
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=170)
    fig.patch.set_facecolor(t["bg"]); ax.set_facecolor(t["bg"])
    h = 0.36
    for j, (split, col) in enumerate((("val", t["s"][0]), ("test", t["s"][2]))):
        vals = [ev[split]["per_class"][k]["ap50"] for k, _ in classes]
        ys = [i + (h / 2 + 0.02) * (-1 if j == 0 else 1) for i in range(len(classes))]
        ax.barh(ys, vals, height=h, color=col, label=f"{split} split ({143 if split == 'val' else 141} images)")
        for y, v in zip(ys, vals):
            ax.text(v + 0.012, y, f"{v:.2f}", va="center", color=t["ink"], fontsize=9.5)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels([c[1] for c in classes], color=t["ink"], fontsize=10.5)
    ax.invert_yaxis()
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], length=0)
    ax.xaxis.grid(True, color=t["grid"], lw=0.8); ax.set_axisbelow(True)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("AP@0.5", color=t["muted"], fontsize=10)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    ax.set_title("AP for the classes the app uses", loc="left", color=t["ink"], fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / f"class-ap-{theme}.png", facecolor=t["bg"])
    plt.close(fig)


def banner_frame(sec=24):
    """Annotated frame used inside the README banner (scripts/diagrams/banner.html)."""
    f = frame_at(sec)
    rgb = draw(f, analyse(detect(f)))
    cv2.imwrite(str(ROOT / "scripts" / "diagrams" / "banner-frame.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    banner_frame()
    picks = [(3, "Helmet and vest on"), (14, "Helmet in hand"), (24, "Second person walks in")]
    rows = timeline_data()
    json.dump(rows, open(OUT / "timeline.json", "w"))
    for th in THEMES:
        chart_detections(th, picks)
        chart_timeline(th, rows)
        chart_training(th)
        chart_class_ap(th)
    print("per-second counts (s, people, no helmet, no vest):", rows)
