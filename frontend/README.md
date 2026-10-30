# Frontend

React + Vite monitoring dashboard, written by [Tarun Sagar Sah](https://github.com/TarunSagarSah). Originally in [TarunSagarSah/tn_impact_frontend](https://github.com/TarunSagarSah/tn_impact_frontend).

All the logic is in [`src/Camera.jsx`](src/Camera.jsx):

1. Opens the webcam and shows it as Camera 1. Cameras 2–4 are placeholder tiles.
2. Every second, draws the current frame to a hidden canvas and sends it as a JPEG to `POST /analyse`.
3. Shows the returned counts: persons, helmet violations, vest violations.
4. When a violation starts, keeps that first frame in a red-framed panel and counts the seconds.
5. If the violation is still there after 15 seconds, downloads the frame as `violation_report_<date>.jpg` and resets.
6. If the violation clears earlier, discards the frame and resets the timer.

## Run

```bash
npm install
npm run dev              # http://localhost:5173, talks to the backend at http://127.0.0.1:8000
```

Set `VITE_API_URL` to use a different backend.

```bash
npm run build:backend    # builds into ../backend/dist, served by FastAPI at the same origin
```

A static build without a backend is deployed at [tnimpactfrontend.vercel.app](https://tnimpactfrontend.vercel.app/).
