from flask import Flask, Response, render_template_string, jsonify
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

MODE_SIZE = (1296, 972)

AWB_PRESETS = ["auto", "daylight", "tungsten", "fluorescent", "indoor", "cloudy"]

MASK_FOLDER = "masks"
TEXT_FOLDER = "text"
OVERLAY_FOLDER = "overlays"

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
    "texts": None,
    "overlays": None,
    "lock": threading.Lock(),
    "filter_mode": "color",
    "awb_mode": "auto",
}

HTML = """
<!doctype html>
<html>
  <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Pi Capture</title>
    <style>
      body { font-family: sans-serif; padding: 16px; }
      button { font-size: 18px; padding: 14px 18px; width: 100%; margin-bottom: 10px; }
      img { width: 100%; margin-top: 16px; border-radius: 8px; }
      p { margin: 6px 0; }
      .row { display: grid; gap: 10px; }
    </style>
  </head>
  <body>
    <div class="row">
      <button onclick="capture()">Take photo</button>
      <button onclick="reroll()">Randomize modes</button>
    </div>

    <p>Resolution: {{ res[0] }}x{{ res[1] }}</p>
    <p>Filter: {{ filt }}</p>
    <p>AWB: {{ awb }}</p>

    <img id="out" />

    <script>
      async function capture() {
        const btns = document.querySelectorAll("button");
        btns.forEach(b => b.disabled = true);
        btns[0].textContent = "Capturing...";
        const r = await fetch("/capture", { method: "POST" });
        const blob = await r.blob();
        document.getElementById("out").src = URL.createObjectURL(blob);
        btns[0].textContent = "Take photo";
        btns.forEach(b => b.disabled = false);
      }

      async function reroll() {
        const btns = document.querySelectorAll("button");
        btns.forEach(b => b.disabled = true);
        btns[1].textContent = "Randomizing...";
        await fetch("/reroll", { method: "POST" });
        location.reload();
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

    if x1 >= x2 or y1 >= y2:
        return bg_bgr

    roi = bg_bgr[y1:y2, x1:x2]
    fg_crop = fg[(y1 - y):(y2 - y), (x1 - x):(x2 - x)]

    alpha = fg_crop[:, :, 3].astype(np.float32) / 255.0
    fg_rgb = fg_crop[:, :, :3].astype(np.float32)
    roi_f = roi.astype(np.float32)

    roi[:] = (fg_rgb * alpha[..., None] + roi_f * (1.0 - alpha[..., None])).astype(np.uint8)
    return bg_bgr

def apply_filter_mode(frame_bgr):
    if STATE["filter_mode"] == "mono":
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return frame_bgr

def load_rgba_pngs_from_folder(folder):
    if not os.path.isdir(folder):
        return []
    out = []
    for name in os.listdir(folder):
        if not name.lower().endswith(".png"):
            continue
        path = os.path.join(folder, name)
        m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        if len(m.shape) == 3 and m.shape[2] == 4:
            out.append(m)
    return out

def load_masks():
    masks = load_rgba_pngs_from_folder(MASK_FOLDER)
    if not masks:
        raise RuntimeError("No valid RGBA PNG masks found in masks/")
    return masks

def load_texts():
    return load_rgba_pngs_from_folder(TEXT_FOLDER)

def load_overlays():
    return load_rgba_pngs_from_folder(OVERLAY_FOLDER)

def awb_enum(mode):
    if controls is None:
        return None
    m = mode.lower()
    if m == "auto":
        return controls.AwbModeEnum.Auto
    if m == "daylight":
        return controls.AwbModeEnum.Daylight
    if m == "tungsten":
        return controls.AwbModeEnum.Tungsten
    if m == "fluorescent":
        return controls.AwbModeEnum.Fluorescent
    if m == "indoor":
        return controls.AwbModeEnum.Indoor
    if m == "cloudy":
        return controls.AwbModeEnum.Cloudy
    return controls.AwbModeEnum.Auto

def apply_awb_mode(picam2):
    if controls is not None:
        picam2.set_controls({"AwbEnable": True, "AwbMode": awb_enum(STATE["awb_mode"])})
    else:
        picam2.set_controls({"AwbEnable": True})

def init_once():
    if STATE["picam2"] is not None:
        return

    base = os.path.dirname(os.path.abspath(__file__))
    proto = os.path.join(base, "deploy.prototxt")
    model = os.path.join(base, "res10_300x300_ssd_iter_140000.caffemodel")
    net = cv2.dnn.readNetFromCaffe(proto, model)

    masks = load_masks()
    texts = load_texts()
    overlays = load_overlays()

    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(main={"format": "BGR888", "size": MODE_SIZE})
    picam2.configure(cfg)
    picam2.start()

    STATE["net"] = net
    STATE["masks"] = masks
    STATE["texts"] = texts
    STATE["overlays"] = overlays
    STATE["picam2"] = picam2

    apply_awb_mode(picam2)

def detect_faces_dnn(frame_bgr, net, conf_thresh):
    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, DNN_INPUT_SIZE, (104.0, 177.0, 123.0), swapRB=False, crop=False)
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
    apply_awb_mode(picam2)
    time.sleep(AWB_SETTLE_SECONDS)
    if not LOCK_AWB_GAINS_PER_SHOT:
        return
    md = picam2.capture_metadata()
    cg = md.get("ColourGains")
    if cg is not None:
        picam2.set_controls({"AwbEnable": False, "ColourGains": (float(cg[0]), float(cg[1]))})

def reroll_modes_internal():
    STATE["filter_mode"] = random.choices(["color", "mono"], weights=[80, 20], k=1)[0]
    STATE["awb_mode"] = random.choice(AWB_PRESETS)
    if STATE["picam2"] is not None:
        apply_awb_mode(STATE["picam2"])

def capture_with_layers():
    init_once()
    picam2 = STATE["picam2"]
    net = STATE["net"]
    masks = STATE["masks"]
    texts = STATE["texts"] or []
    overlays = STATE["overlays"] or []

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

    H, W = frame.shape[:2]

    if texts:
        tx = random.choice(texts)
        frame = overlay_rgba(frame, tx, 0, 0, W, H)

    if overlays:
        ov = random.choice(overlays)
        frame = overlay_rgba(frame, ov, 0, 0, W, H)

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
    with STATE["lock"]:
        reroll_modes_internal()
        return render_template_string(
            HTML,
            res=MODE_SIZE,
            filt=STATE["filter_mode"],
            awb=STATE["awb_mode"],
        )

@app.post("/reroll")
def reroll():
    with STATE["lock"]:
        reroll_modes_internal()
        return jsonify({"filter": STATE["filter_mode"], "awb": STATE["awb_mode"]})

@app.post("/capture")
def capture():
    with STATE["lock"]:
        data = capture_with_layers()
    r = Response(data, mimetype="image/jpeg")
    r.headers["Cache-Control"] = "no-store"
    return r

if __name__ == "__main__":
    print("Resolution fixed:", MODE_SIZE)
    app.run(host="0.0.0.0", port=5000, threaded=True)
