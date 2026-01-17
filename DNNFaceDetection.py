from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import time
from picamera2 import Picamera2
import threading
from datetime import datetime
import random
import os

AVAILABLE_RESOLUTIONS = [
    (640, 480),
    (1296, 972),
    (1920, 1080),
    (2592, 1944),
]

AVAILABLE_FILTER_MODES = [
    "color",
    "mono",
]

MODE_SIZE = random.choice(AVAILABLE_RESOLUTIONS)
FILTER_MODE = random.choice(AVAILABLE_FILTER_MODES)

MASK_FOLDER = "masks"
MASK_COUNT = 6

DNN_INPUT_SIZE = (300, 300)
CONF_THRESH = 0.60

SQUARE_SCALE = 1.35
SQUARE_Y_SHIFT = -0.10

AWB_SETTLE_SECONDS = 0.6

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
      p { margin: 8px 0; }
    </style>
  </head>
  <body>
    <button onclick="capture()">Take photo</button>
    <p>Resolution: {{ res[0] }}x{{ res[1] }}</p>
    <p>Filter mode: {{ filt }}</p>
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

    fg_x1 = x1 - x
    fg_y1 = y1 - y
    fg_x2 = fg_x1 + (x2 - x1)
    fg_y2 = fg_y1 + (y2 - y1)

    if x1 >= x2 or y1 >= y2:
        return bg_bgr

    roi = bg_bgr[y1:y2, x1:x2]
    fg_crop = fg[fg_y1:fg_y2, fg_x1:fg_x2]

    alpha = fg_crop[:, :, 3].astype(np.float32) / 255.0
    alpha = alpha[:, :, None]

    fg_rgb = fg_crop[:, :, :3].astype(np.float32)
    roi_f = roi.astype(np.float32)

    out = fg_rgb * alpha + roi_f * (1.0 - alpha)
    bg_bgr[y1:y2, x1:x2] = out.astype(np.uint8)
    return bg_bgr

def apply_filter_mode(frame_bgr):
    if FILTER_MODE == "mono":
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return frame_bgr

def load_masks():
    masks = []
    for i in range(1, MASK_COUNT + 1):
        path = os.path.join(MASK_FOLDER, f"mask{i}.png")
        m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if m is None or len(m.shape) != 3 or m.shape[2] != 4:
            continue
        masks.append(m)
    if not masks:
        raise RuntimeError("No valid RGBA masks found in masks/ as mask1.png..maskN.png")
    return masks

def init_once():
    if STATE["picam2"] is not None:
        return

    net = cv2.dnn.readNetFromCaffe(
        "deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel"
    )

    masks = load_masks()

    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(
        main={"format": "BGR888", "size": MODE_SIZE}
    )
    picam2.configure(cfg)
    picam2.start()
    picam2.set_controls({"AwbEnable": True})

    STATE["masks"] = masks
    STATE["net"] = net
    STATE["picam2"] = picam2

def detect_faces_dnn(frame_bgr, net, conf_thresh):
    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame_bgr,
        1.0,
        DNN_INPUT_SIZE,
        (104.0, 177.0, 123.0),
        swapRB=False,
        crop=False
    )
    net.setInput(blob)
    det = net.forward()

    boxes = []
    for i in range(det.shape[2]):
        conf = float(det[0, 0, i, 2])
        if conf < conf_thresh:
            continue
        x1 = int(det[0, 0, i, 3] * w)
        y1 = int(det[0, 0, i, 4] * h)
        x2 = int(det[0, 0, i, 5] * w)
        y2 = int(det[0, 0, i, 6] * h)
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        bw = x2 - x1
        bh = y2 - y1
        if bw > 0 and bh > 0:
            boxes.append((x1, y1, bw, bh))
    return boxes

def lock_awb_for_shot(picam2):
    picam2.set_controls({"AwbEnable": True})
    time.sleep(AWB_SETTLE_SECONDS)
    md = picam2.capture_metadata()
    cg = md.get("ColourGains")
    if cg is not None:
        picam2.set_controls({
            "AwbEnable": False,
            "ColourGains": (float(cg[0]), float(cg[1]))
        })

def capture_with_overlay():
    init_once()
    picam2 = STATE["picam2"]
    net = STATE["net"]
    masks = STATE["masks"]

    lock_awb_for_shot(picam2)

    frame = picam2.capture_array()
    frame = apply_filter_mode(frame)

    faces = detect_faces_dnn(frame, net, CONF_THRESH)

    for (x, y, w, h) in faces:
        mask = random.choice(masks)
        cx = x + w // 2
        cy = y + h // 2
        side = int(max(w, h) * SQUARE_SCALE)
        cy = int(cy + SQUARE_Y_SHIFT * side)
        x0 = cx - side // 2
        y0 = cy - side // 2
        frame = overlay_rgba(frame, mask, x0, y0, side, side)

    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError("JPEG encode failed")

    data = jpg.tobytes()
    name = datetime.now().strftime("capture_%Y%m%d_%H%M%S.jpg")
    with open(name, "wb") as f:
        f.write(data)

    return data

@app.get("/")
def index():
    return render_template_string(HTML, res=MODE_SIZE, filt=FILTER_MODE)

@app.post("/capture")
def capture():
    with STATE["lock"]:
        data = capture_with_overlay()
    r = Response(data, mimetype="image/jpeg")
    r.headers["Cache-Control"] = "no-store"
    return r

if __name__ == "__main__":
    print("Random resolution selected:", MODE_SIZE)
    print("Random filter mode selected:", FILTER_MODE)
    app.run(host="0.0.0.0", port=5000, threaded=True)
