import os
import threading
from pathlib import Path

import cv2
import supervision as sv
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template_string, request
from roboflow import Roboflow


BASE_DIR = Path(__file__).resolve().parent
TEST_VIDEO_DIR = (BASE_DIR.parent / "test_video").resolve()

app = Flask(__name__)
state_lock = threading.Lock()
stream_state = {
    "running": False,
    "done": False,
    "error": None,
    "video_name": "",
    "current_frame": 0,
    "makes": 0,
    "make_frames": [],
}


load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
rf = Roboflow(api_key=ROBOFLOW_API_KEY)

PLAYER_DETECTION_MODEL_ID = "basketball-player-detection-3-ycjdo"
project = rf.workspace().project(PLAYER_DETECTION_MODEL_ID)
PLAYER_DETECTION_MODEL = project.version(4).model

COLOR = sv.ColorPalette.from_hex(
    [
        "#ffff00",
        "#ff9b00",
        "#ff66ff",
        "#3399ff",
        "#ff66b2",
        "#ff8080",
        "#b266ff",
        "#9999ff",
        "#66ffff",
        "#33ff99",
        "#66ff66",
        "#99ff00",
    ]
)
box_annotator = sv.BoxAnnotator(color=COLOR, thickness=2)
label_annotator = sv.LabelAnnotator(color=COLOR, text_color=sv.Color.BLACK)


def is_ball_in_basket(detections):
    for detection in detections:
        if detection.get("class") == "ball-in-basket":
            return True
    return False


def list_test_videos():
    if not TEST_VIDEO_DIR.exists():
        return []
    return sorted([p.name for p in TEST_VIDEO_DIR.glob("*.mp4")])


def reset_state(video_name):
    with state_lock:
        stream_state["running"] = True
        stream_state["done"] = False
        stream_state["error"] = None
        stream_state["video_name"] = video_name
        stream_state["current_frame"] = 0
        stream_state["makes"] = 0
        stream_state["make_frames"] = []


def update_state(**kwargs):
    with state_lock:
        stream_state.update(kwargs)


def annotate_frame(frame, results, frame_idx, makes):
    detections = results.get("predictions", [])
    sv_detections = sv.Detections.from_inference(results)
    annotated = frame.copy()
    annotated = box_annotator.annotate(scene=annotated, detections=sv_detections)
    annotated = label_annotator.annotate(scene=annotated, detections=sv_detections)

    has_basket = any("basket" in d.get("class", "") for d in detections)
    has_ball = any("ball" in d.get("class", "") for d in detections)
    status_text = f"Frame: {frame_idx} | Basket: {'Y' if has_basket else 'N'} | Ball: {'Y' if has_ball else 'N'} | Makes: {makes}"
    cv2.putText(
        annotated,
        status_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
    )
    return annotated, detections


def generate_stream(video_name):
    video_path = (TEST_VIDEO_DIR / video_name).resolve()
    if not video_path.exists():
        update_state(
            running=False,
            done=True,
            error=f"Video not found: {video_name}",
        )
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        update_state(
            running=False,
            done=True,
            error=f"Failed to open video: {video_name}",
        )
        return

    frame_count = 0
    debounce_counter = 0
    make_frames = []
    makes = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = PLAYER_DETECTION_MODEL.predict(frame, confidence=0.35, overlap=0.5).json()
        annotated_frame, detections = annotate_frame(frame, results, frame_count, makes)

        if is_ball_in_basket(detections):
            debounce_counter += 1
            if debounce_counter >= 1:
                makes += 1
                make_frames.append(frame_count)
                debounce_counter = 0
        else:
            debounce_counter = 0

        # Overlay makes after update to keep on-screen value in sync.
        cv2.putText(
            annotated_frame,
            f"Makes: {makes}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
        )

        update_state(current_frame=frame_count, makes=makes, make_frames=make_frames)

        ok, buffer = cv2.imencode(".jpg", annotated_frame)
        if not ok:
            continue
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    cap.release()
    update_state(running=False, done=True, make_frames=make_frames, makes=makes)
    print(f"[INFO] {video_name} finished. Make frames: {make_frames}")


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Basketball Make Viewer</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background: #111; color: #eee; }
    .row { display: flex; gap: 12px; align-items: center; margin-bottom: 12px; }
    select, button { padding: 8px 12px; font-size: 14px; }
    #feed { width: 960px; max-width: 100%; border: 1px solid #555; background: #000; }
    #status { margin-top: 12px; white-space: pre-wrap; line-height: 1.5; }
  </style>
</head>
<body>
  <h2>Video Detection Stream</h2>
  <div class="row">
    <label for="videoSelect">Select test video:</label>
    <select id="videoSelect">
      {% for video in videos %}
        <option value="{{ video }}">{{ video }}</option>
      {% endfor %}
    </select>
    <button id="startBtn" onclick="startStream()">Start</button>
  </div>
  <img id="feed" alt="Annotated stream will appear here"/>
  <div id="status">Waiting to start...</div>

  <script>
    let pollTimer = null;
    function startStream() {
      const video = document.getElementById("videoSelect").value;
      const feed = document.getElementById("feed");
      const status = document.getElementById("status");
      status.textContent = "Starting...";
      feed.src = `/video_feed?video=${encodeURIComponent(video)}&t=${Date.now()}`;

      if (pollTimer) clearInterval(pollTimer);
      pollTimer = setInterval(async () => {
        const r = await fetch("/status");
        const data = await r.json();
        const lines = [
          `Video: ${data.video_name || "-"}`,
          `Running: ${data.running}`,
          `Current frame: ${data.current_frame}`,
          `Makes: ${data.makes}`,
          `Make frames: ${data.make_frames.length ? data.make_frames.join(", ") : "None"}`
        ];
        if (data.error) lines.push(`Error: ${data.error}`);
        if (data.done) lines.push("Done.");
        status.textContent = lines.join("\\n");
      }, 500);
    }
  </script>
</body>
</html>
"""


@app.get("/")
def index():
    videos = list_test_videos()
    return render_template_string(INDEX_HTML, videos=videos)


@app.get("/video_feed")
def video_feed():
    video_name = request.args.get("video", "").strip()
    if not video_name:
        return Response("Missing video parameter.", status=400)
    reset_state(video_name)
    return Response(
        generate_stream(video_name),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/status")
def status():
    with state_lock:
        return jsonify(stream_state)


if __name__ == "__main__":
    print("[INFO] Open http://127.0.0.1:5000 in your browser")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
