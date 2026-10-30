/*
 * PPE Watch: browser version of the TN-IMPACT 2026 PPE violation detector.
 *
 * Detection: backend/best.pt (Ultralytics YOLO26n) exported to ONNX, run with onnxruntime-web.
 * Decision rules: a line-for-line port of backend/main.py and backend/utils.py.
 * Alert flow: the same as frontend/src/Camera.jsx (1 frame per second, first violation
 * frame kept as evidence, report after 15 s of continuous violation).
 */
"use strict";

// Hosts can set window.PPE_CONFIG before this script:
//   ortDist      folder with ort-wasm-simd-threaded.{mjs,wasm}
//   modelUrl     where the ONNX model is
//   modelBase64  true if modelUrl serves the model as base64 text (for hosts that refuse .onnx files)
const CFG = Object.assign(
  { ortDist: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/", modelUrl: "model/best.onnx", modelBase64: false },
  window.PPE_CONFIG || {}
);
const INPUT = 640;            // export size
const CONF = 0.20;            // backend/detector.py: self.model(source, conf=0.20)
const FRAME_MS = 1000;        // frontend/src/Camera.jsx: setInterval(sendFrameToBackend, 1000)
const REPORT_AFTER_S = 15;    // frontend/src/Camera.jsx: if (violationTime >= 15)

// backend/main.py CLASS_MAP
const CLASS_MAP = { 0: "helmet", 2: "vest", 6: "person", 7: "no_helmet" };

// ---------- backend/utils.py ----------
function isOverlap(a, b) {
  const xA = Math.max(a[0], b[0]), yA = Math.max(a[1], b[1]);
  const xB = Math.min(a[2], b[2]), yB = Math.min(a[3], b[3]);
  return xA < xB && yA < yB;
}
// A helmet belongs to a person if its centre is inside the person box and in the upper half.
function objectBelongsToPerson(p, h) {
  const hx = (h[0] + h[2]) / 2, hy = (h[1] + h[3]) / 2;
  if (!(p[0] <= hx && hx <= p[2] && p[1] <= hy && hy <= p[3])) return false;
  return (hy - p[1]) <= (p[3] - hy);
}

// ---------- backend/main.py analyze_frame() ----------
function analyse(dets) {
  const of = (name) => dets.filter((d) => CLASS_MAP[d.classId] === name);
  const persons = of("person"), helmets = of("helmet"), vests = of("vest"), noHelmets = of("no_helmet");
  let helmetViolations = 0, vestViolations = 0;
  const verdicts = persons.map((p) => {
    const helmetFound = helmets.some((h) => objectBelongsToPerson(p.bbox, h.bbox));
    const explicitNoHelmet = noHelmets.some((n) => isOverlap(p.bbox, n.bbox));
    const noHelmet = !helmetFound || explicitNoHelmet;
    const noVest = !vests.some((v) => isOverlap(p.bbox, v.bbox));
    if (noHelmet) helmetViolations++;
    if (noVest) vestViolations++;
    return { bbox: p.bbox, noHelmet, noVest };
  });
  return {
    no_of_person: persons.length,
    no_of_helmet_violation: helmetViolations,
    no_of_vest_violation: vestViolations,
    verdicts,
  };
}

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);
const video = $("video"), still = $("still"), overlay = $("overlay"), work = $("work");
const stage = $("stage"), statusEl = $("status");
const ui = {
  person: $("n-person"), helmet: $("n-helmet"), vest: $("n-vest"),
  evidence: $("evidence"), evidenceImg: $("evidence-img"), timer: $("timer"), meter: $("meter-fill"),
  banner: $("banner"), camLabel: $("cam-label"), logList: $("log-list"), logEmpty: $("log-empty"),
  showBoxes: $("show-boxes"), autoSave: $("auto-save"), playBtn: $("play-btn"),
};

let session = null;
let backendUrl = null;        // when set, frames go to the FastAPI backend instead
let busy = false;
let source = "sample";
let lastResult = null;
let lastFrameSize = null;
let lastMs = null;

// violation state (mirrors Camera.jsx)
let violationActive = false;
let violationURL = null;
let violationBlob = null;
let violationTime = 0;
let violationTimer = null;

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

// ---------- model ----------
async function loadModel() {
  if (!window.ort) throw new Error("onnxruntime-web did not load. Check your connection and reload the page.");
  ort.env.wasm.wasmPaths = CFG.ortDist;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  let model = CFG.modelUrl;
  if (CFG.modelBase64) {
    const text = await (await fetch(CFG.modelUrl)).text();
    const bin = atob(text.trim());
    model = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) model[i] = bin.charCodeAt(i);
  }
  session = await ort.InferenceSession.create(model, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
}

// Letterbox to 640x640 with grey (114) padding, the same as Ultralytics preprocessing.
function toTensor(srcCanvas) {
  const w = srcCanvas.width, h = srcCanvas.height;
  const s = Math.min(INPUT / w, INPUT / h);
  const nw = Math.round(w * s), nh = Math.round(h * s);
  const px = Math.floor((INPUT - nw) / 2), py = Math.floor((INPUT - nh) / 2);
  const c = toTensor.canvas || (toTensor.canvas = Object.assign(document.createElement("canvas"), { width: INPUT, height: INPUT }));
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, INPUT, INPUT);
  ctx.drawImage(srcCanvas, 0, 0, w, h, px, py, nw, nh);
  const { data } = ctx.getImageData(0, 0, INPUT, INPUT);
  const area = INPUT * INPUT, f = new Float32Array(3 * area);
  for (let i = 0, j = 0; i < area; i++, j += 4) {
    f[i] = data[j] / 255; f[i + area] = data[j + 1] / 255; f[i + 2 * area] = data[j + 2] / 255;
  }
  return { tensor: new ort.Tensor("float32", f, [1, 3, INPUT, INPUT]), s, px, py };
}

async function detect(srcCanvas) {
  const { tensor, s, px, py } = toTensor(srcCanvas);
  const out = await session.run({ [session.inputNames[0]]: tensor });
  const d = out[session.outputNames[0]].data; // [1, 300, 6]: x1 y1 x2 y2 score class
  const dets = [];
  for (let i = 0; i < d.length; i += 6) {
    const score = d[i + 4];
    if (score < CONF) continue;
    dets.push({
      classId: Math.round(d[i + 5]), confidence: score,
      bbox: [(d[i] - px) / s, (d[i + 1] - py) / s, (d[i + 2] - px) / s, (d[i + 3] - py) / s],
    });
  }
  return dets;
}

// ---------- frame capture ----------
function currentMedia() { return source === "file" && !still.hidden ? still : video; }

function captureFrame() {
  const m = currentMedia();
  const w = m.videoWidth || m.naturalWidth, h = m.videoHeight || m.naturalHeight;
  if (!w || !h) return null;
  work.width = w; work.height = h;
  work.getContext("2d").drawImage(m, 0, 0, w, h);
  return work;
}
const canvasToBlob = (c) => new Promise((res) => c.toBlob(res, "image/jpeg", 0.8));

// ---------- one tick: same flow as sendFrameToBackend() in Camera.jsx ----------
async function tick() {
  if (busy || document.hidden) return;
  if (!session && !backendUrl) return;
  const frame = captureFrame();
  if (!frame) return;
  busy = true;
  try {
    const blob = await canvasToBlob(frame);
    const t0 = performance.now();
    let result;
    if (backendUrl) {
      const form = new FormData();
      form.append("file", blob, "frame.jpg");
      const r = await fetch(backendUrl.replace(/\/$/, "") + "/analyse", { method: "POST", body: form });
      if (!r.ok) throw new Error(`Backend replied ${r.status}`);
      result = await r.json();
      result.verdicts = null;
    } else {
      result = analyse(await detect(frame));
    }
    lastMs = performance.now() - t0;
    lastResult = result;
    lastFrameSize = [frame.width, frame.height];
    render(result);
    updateViolation(result, blob);
    setStatus(`${backendUrl ? "Backend" : "Model"} ready. 1 frame per second. Last frame took ${Math.round(lastMs)} ms.`);
  } catch (err) {
    console.error(err);
    setStatus(backendUrl ? `Could not reach the backend at ${backendUrl}. Check that it is running, or switch back to the browser model.` : `Detection failed: ${err.message}`, true);
  } finally {
    busy = false;
  }
}

function render(r) {
  ui.person.textContent = r.no_of_person;
  ui.helmet.textContent = r.no_of_helmet_violation;
  ui.vest.textContent = r.no_of_vest_violation;
  ui.helmet.parentElement.classList.toggle("hot", r.no_of_helmet_violation > 0);
  ui.vest.parentElement.classList.toggle("hot", r.no_of_vest_violation > 0);
  const parts = [];
  if (r.no_of_helmet_violation) parts.push(`${r.no_of_helmet_violation} without helmet`);
  if (r.no_of_vest_violation) parts.push(`${r.no_of_vest_violation} without vest`);
  ui.banner.textContent = parts.length ? `Violation: ${parts.join(", ")}` : "";
  stage.dataset.state = parts.length ? "violation" : "clear";
  drawBoxes();
}

function drawBoxes() {
  const m = currentMedia();
  const rect = m.getBoundingClientRect(), host = m.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  overlay.style.left = `${rect.left - host.left}px`;
  overlay.style.top = `${rect.top - host.top}px`;
  overlay.style.width = `${rect.width}px`;
  overlay.style.height = `${rect.height}px`;
  overlay.width = Math.round(rect.width * dpr);
  overlay.height = Math.round(rect.height * dpr);
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!ui.showBoxes.checked || !lastResult || !lastResult.verdicts || !lastFrameSize) return;
  const k = (rect.width / lastFrameSize[0]) * dpr;
  ctx.lineWidth = 2.5 * dpr;
  ctx.font = `600 ${13 * dpr}px "IBM Plex Sans", sans-serif`;
  ctx.textBaseline = "top";
  for (const v of lastResult.verdicts) {
    const [x1, y1, x2, y2] = v.bbox.map((n) => n * k);
    const bad = v.noHelmet || v.noVest;
    const color = bad ? "#c8102e" : "#2e7d32";
    const label = bad ? [v.noHelmet && "no helmet", v.noVest && "no vest"].filter(Boolean).join(" + ") : "helmet + vest";
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const tw = ctx.measureText(label).width + 10 * dpr, th = 20 * dpr;
    const ty = y1 - th >= 0 ? y1 - th : y1;
    const lx = Math.max(0, Math.min(x1 - ctx.lineWidth / 2, overlay.width - tw));
    ctx.fillStyle = color;
    ctx.fillRect(lx, ty, tw, th);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, lx + 5 * dpr, ty + 3 * dpr);
  }
}

// ---------- violation state machine (Camera.jsx) ----------
function updateViolation(r, blob) {
  const hasViolation = r.no_of_helmet_violation > 0 || r.no_of_vest_violation > 0;
  if (hasViolation && !violationActive) {
    violationActive = true;
    violationBlob = blob;
    violationURL = URL.createObjectURL(blob);
    ui.evidenceImg.src = violationURL;
    ui.evidence.hidden = false;
    violationTime = 0;
    showTimer();
    violationTimer = setInterval(() => {
      violationTime++;
      showTimer();
      if (violationTime >= REPORT_AFTER_S) fileReport();
    }, 1000);
  } else if (!hasViolation && violationActive) {
    clearViolation(true);
  }
}
function showTimer() {
  ui.timer.textContent = `${violationTime} s`;
  ui.meter.style.width = `${Math.min(100, (violationTime / REPORT_AFTER_S) * 100)}%`;
}
function clearViolation(revoke) {
  clearInterval(violationTimer);
  violationTimer = null;
  violationActive = false;
  violationTime = 0;
  if (revoke && violationURL) URL.revokeObjectURL(violationURL);
  violationURL = null;
  violationBlob = null;
  ui.evidence.hidden = true;
}
// Save a report image as a normal browser download.
function saveImage(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
}

function fileReport() {
  const stamp = new Date();
  const name = `violation_report_${stamp.toString().replace(/[:]/g, "_")}.jpg`;
  const url = violationURL;
  const blob = violationBlob;
  const li = document.createElement("li");
  const r = lastResult || {};
  const what = [r.no_of_helmet_violation && "No helmet", r.no_of_vest_violation && "No vest"].filter(Boolean).join(", ") || "Violation";
  li.innerHTML = `<img alt="Evidence frame"><div><div class="what"></div><div class="when"></div><button type="button" class="link">Save image</button></div>`;
  li.querySelector("img").src = url;
  li.querySelector(".what").textContent = `${what}, lasted ${REPORT_AFTER_S} s`;
  li.querySelector(".when").textContent = stamp.toLocaleTimeString();
  li.querySelector("button").addEventListener("click", () => saveImage(blob, name));
  ui.logList.prepend(li);
  ui.logEmpty.hidden = true;
  if (ui.autoSave.checked) saveImage(blob, name);
  clearViolation(false); // keep the object URL alive for the report entry
}

// ---------- sources ----------
async function setSource(next) {
  document.querySelectorAll(".segmented button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.source === next)));
  if (next === "file") { $("file-input").click(); return; }
  stopWebcam();
  still.hidden = true; video.hidden = false;
  clearViolation(true);
  if (next === "sample") {
    source = "sample";
    video.srcObject = null;
    if (!video.currentSrc.includes("rooftop")) video.load();
    video.loop = true;
    ui.camLabel.textContent = "Camera 1: sample clip";
    await video.play().catch(() => {});
  } else if (next === "webcam") {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      source = "webcam";
      video.srcObject = stream;
      ui.camLabel.textContent = "Camera 1: your webcam";
      await video.play();
    } catch (err) {
      setStatus("Camera access was blocked. Allow camera access in the address bar, then choose Webcam again.", true);
      setSource("sample");
    }
  }
  ui.playBtn.textContent = "Pause";
}
function stopWebcam() {
  if (video.srcObject) { video.srcObject.getTracks().forEach((t) => t.stop()); video.srcObject = null; }
}
$("file-input").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) { setSource(source); return; }
  stopWebcam();
  clearViolation(true);
  source = "file";
  const url = URL.createObjectURL(f);
  ui.camLabel.textContent = `Camera 1: ${f.name}`;
  if (f.type.startsWith("image/")) {
    video.pause(); video.hidden = true;
    still.src = url; still.hidden = false;
    await still.decode().catch(() => {});
  } else {
    still.hidden = true; video.hidden = false;
    video.srcObject = null; video.src = url; video.loop = true;
    await video.play().catch(() => {});
  }
  e.target.value = "";
  tick();
});
document.querySelectorAll(".segmented button").forEach((b) => b.addEventListener("click", () => setSource(b.dataset.source)));
ui.playBtn.addEventListener("click", () => {
  if (video.hidden) return;
  if (video.paused) { video.play(); ui.playBtn.textContent = "Pause"; } else { video.pause(); ui.playBtn.textContent = "Play"; }
});
ui.showBoxes.addEventListener("change", drawBoxes);
window.addEventListener("resize", drawBoxes);

// ---------- optional FastAPI backend ----------
$("backend-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const v = $("backend-url").value.trim() || "http://127.0.0.1:8000";
  backendUrl = v;
  $("engine-name").textContent = `backend at ${v}`;
  $("use-browser").hidden = false;
  setStatus(`Sending frames to ${v}/analyse…`);
});
$("use-browser").addEventListener("click", () => {
  backendUrl = null;
  $("engine-name").textContent = "in this browser";
  $("use-browser").hidden = true;
  setStatus(session ? "Model ready." : "Loading the model…");
});

// ---------- start ----------
(async () => {
  video.play().catch(() => {});
  try {
    await loadModel();
    setStatus("Model ready. 1 frame per second.");
  } catch (err) {
    console.error(err);
    setStatus(`The model could not load: ${err.message}`, true);
  }
  setInterval(tick, FRAME_MS);
  tick();
})();

// exposed for automated checks
window.__ppe = { analyse, detect, captureFrame, get last() { return lastResult; } };
