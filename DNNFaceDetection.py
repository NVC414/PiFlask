from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import time
from picamera2 import Picamera2
import threading
from datetime import datetime
import random
import os

try:
    from libcamera import controls
except Exception:
    controls = None

AVAILABLE_RESOLUTIONS = [
    (640, 480),
    (1296, 972),
    (2592, 1944),
]

MODE_SIZE = random.choice(AVAILABLE_RESOLUTIONS)

FILTER_MODE = random.choices(
    ["color", "mono"],
    weights=[80, 20],
    k=1
)[0]

AWB_PRESETS = ["auto", "daylight", "tungsten", "fluorescent", "indoor", "cloudy"]
AWB_MODE = random.choice(AWB_PRESETS)

MASK_FOLDER = "masks"
MASK_COUNT = 6

DNN_INPUT_SIZE = (300, 300)
CONF_THRESH = 0.60

SQUARE_SCALE = 1.35
SQUARE_Y_SHIFT = -0.10

AWB_SETTLE_SECONDS = 0.6
LOCK_AWB_GAINS_PER_SHOT = True

app = Flask(__name__)

STATE = {
    "picam2": None,
    "net": None,
    "masks": None,
    "lock": threading.Lock(),
}

HTML = """
<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pi Capture</title>
  <style>
    body { font-family: sans-serif; padding: 16px; }
    button { font-size: 18px; padding: 14px 18px; width: 100%; }
    img { width: 100%; margin-top: 16px; border-radius: 8px; }
    p { margin: 6px 0; }
  </style>
</head>
<body>
  <button onclick="capture()">Take photo</button>
  <p>Resolution: {{ res[0] }}x{{ res[1] }}</p>
  <p>Filter: {{ filt }}</p>
  <p>AWB: {{ awb }}</p>
  <img id="out" />
  <script>
    async function capture() {
      const btn = document.querySelector("button");
      btn.disabled = true;
      btn.textContent = "Capturing...";
      const r = await fetch("/capture", { method: "POST" });
      const blob = await r.blob();
      document.getElementById("out").src = URL.createObjectURL(blob);
      btn.disabled = false;
      btn.textContent = "Take photo";
    }
  </script>
</body>
</html>
"""

def overlay_rgba(bg_bgr, fg_rgba, x, y, w, h):
    H, W = bg_bgr.shape[:2]
    if w <= 0 or h <= 0:
        return bg_bgr

    fg = cv2.resize(fg_rgba, (w, h), interpolation=cv2.INTER_AREA)

    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, W)
    y2 = min(y + h, H)

    roi = bg_bgr[y1:y2, x1:x2]
    fg_crop = fg[y1 - y:y2 - y, x1 - x:x2 - x]

    alpha = fg_crop[:, :, 3].astype(np.float32) / 255.0
    fg_rgb = fg_crop[:, :, :3].astype(np.float32)

    roi[:] = fg_rgb * alpha[..., None] + roi.astype(np.float32) * (1.0 - alpha[..., None])
    return bg_bgr

def apply_filter_mode(frame):
    if FILTER_MODE == "mono":
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return frame

def load_masks():
    masks = []
    for i in range(1, MASK_COUNT + 1):
        path = os.path.join(MASK_FOLDER, f"mask{i}.png")
        m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if m is not None and m.shape[2] == 4:
            masks.append(m)
    if not masks:
        raise RuntimeError("No valid RGBA masks found")
    return masks

def awb_enum(mode):
    if controls is None:
        return None
    return {
        "auto": controls.AwbModeEnum.Auto,
        "daylight": controls.AwbModeEnum.Daylight,
        "tungsten": controls.AwbModeEnum.Tungsten,
        "fluorescent": controls.AwbModeEnum.Fluorescent,
        "indoor": controls.AwbModeEnum.Indoor,
        "cloudy": controls.AwbModeEnum.Cloudy,
    }.get(mode, controls.AwbModeEnum.Auto)

def init_once():
    if STATE["picam2"] is not None:
        return

    STATE["net"] = cv2.dnn.readNetFromCaffe(
        "deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel"
    )
    STATE["masks"] = load_masks()

    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(
        main={"format": "BGR888", "size": MODE_SIZE}
    )
    picam2.configure(cfg)
    picam2.start()

    if controls:
        picam2.set_controls({"AwbEnable": True, "AwbMode": awb_enum(AWB_MODE)})
    else:
        picam2.set_controls({"AwbEnable": True})

    STATE["picam2"] = picam2

def detect_faces(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, DNN_INPUT_SIZE, (104, 177, 123))
    STATE["net"].setInput(blob)
    det = STATE["net"].forward()

    boxes = []
    for i in range(det.shape[2]):
        if det[0, 0, i, 2] >= CONF_THRESH:
            x1 = int(det[0, 0, i, 3] * w)
            y1 = int(det[0, 0, i, 4] * h)
            x2 = int(det[0, 0, i, 5] * w)
            y2 = int(det[0, 0, i, 6] * h)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes

def lock_awb(picam2):
    picam2.set_controls({"AwbEnable": True})
    time.sleep(AWB_SETTLE_SECONDS)
    if LOCK_AWB_GAINS_PER_SHOT:
        cg = picam2.capture_metadata().get("ColourGains")
        if cg:
            picam2.set_controls({"AwbEnable": False, "ColourGains": cg})

def capture_with_overlay():
    init_once()
    picam2 = STATE["picam2"]

    lock_awb(picam2)
    frame = picam2.capture_array()
    frame = apply_filter_mode(frame)

    for (x, y, w, h) in detect_faces(frame):
        mask = random.choice(STATE["masks"])
        side = int(max(w, h) * SQUARE_SCALE)
        cx, cy = x + w // 2, y + h // 2
        cy += int(SQUARE_Y_SHIFT * side)
        frame = overlay_rgba(frame, mask, cx - side // 2, cy - side // 2, side, side)

    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    data = jpg.tobytes()

    with open(datetime.now().strftime("capture_%Y%m%d_%H%M%S.jpg"), "wb") as f:
        f.write(data)

    return data

@app.get("/")
def index():
    return render_template_string(HTML, res=MODE_SIZE, filt=FILTER_MODE, awb=AWB_MODE)

@app.post("/capture")
def capture():
    with STATE["lock"]:
        data = capture_with_overlay()
    return Response(data, mimetype="image/jpeg")

if __name__ == "__main__":
    print("Resolution:", MODE_SIZE)
    print("Filter:", FILTER_MODE)
    print("AWB:", AWB_MODE)
    app.run(host="0.0.0.0", port=5000, threaded=True)
