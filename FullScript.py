from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import time
from picamera2 import Picamera2
import threading
from datetime import datetime

MODE_SIZE = (1296, 972)

app = Flask(__name__)

STATE = {
    "picam2": None,
    "cascade": None,
    "mask": None,
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
    </style>
  </head>
  <body>
    <button onclick="capture()">Take photo</button>
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

def load_cascade():
    paths = [
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
    ]
    for p in paths:
        c = cv2.CascadeClassifier(p)
        if not c.empty():
            return c
    raise RuntimeError("Could not load Haar cascade")

def init_camera_once():
    if STATE["picam2"] is not None:
        return

    mask = cv2.imread("mask.png", cv2.IMREAD_UNCHANGED)
    if mask is None or mask.shape[2] != 4:
        raise RuntimeError("mask.png must be RGBA")

    cascade = load_cascade()

    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(main={"format": "BGR888", "size": MODE_SIZE})
    picam2.configure(cfg)
    picam2.start()

    time.sleep(2)
    md = picam2.capture_metadata()
    cg = md.get("ColourGains")
    if cg is not None:
        picam2.set_controls({"AwbEnable": False, "ColourGains": cg})

    STATE["mask"] = mask
    STATE["cascade"] = cascade
    STATE["picam2"] = picam2

def capture_frame_with_overlay():
    init_camera_once()

    picam2 = STATE["picam2"]
    cascade = STATE["cascade"]
    mask = STATE["mask"]

    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        pad_w = int(0.10 * w)
        pad_h = int(0.25 * h)
        frame = overlay_rgba(frame, mask, x - pad_w, y - pad_h, w + 2 * pad_w, h + 2 * pad_h)

    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError("JPEG encode failed")

    data = jpg.tobytes()
    name = datetime.now().strftime("capture_%Y%m%d_%H%M%S.jpg")
    with open(name, "wb") as f:
        f.write(data)

    return data, name

@app.get("/")
def index():
    return render_template_string(HTML)

@app.post("/capture")
def capture():
    with STATE["lock"]:
        data, _ = capture_frame_with_overlay()
    return Response(data, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
