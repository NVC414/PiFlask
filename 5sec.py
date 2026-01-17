import cv2
import numpy as np
import time
from picamera2 import Picamera2

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
    raise RuntimeError("Haar cascade not found")

def main():
    mask = cv2.imread("mask.png", cv2.IMREAD_UNCHANGED)
    if mask is None or mask.shape[2] != 4:
        raise RuntimeError("mask.png must be RGBA")

    cascade = load_cascade()

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "BGR888", "size": (1280, 720)}
    )
    picam2.configure(config)
    picam2.start()

    start_time = time.time()
    captured = False
    countdown_seconds = 5

    while True:
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
            frame = overlay_rgba(
                frame,
                mask,
                x - pad_w,
                y - pad_h,
                w + 2 * pad_w,
                h + 2 * pad_h
            )

        elapsed = time.time() - start_time
        remaining = int(countdown_seconds - elapsed)

        if remaining > 0:
            cv2.putText(
                frame,
                f"Taking photo in {remaining}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )
        elif not captured:
            cv2.imwrite("capture.jpg", frame)
            captured = True

        cv2.imshow("Face Overlay", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
