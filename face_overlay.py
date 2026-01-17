import cv2
import numpy as np

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
    raise RuntimeError("Could not load Haar cascade. Ensure opencv-data is installed.")

def main():
    mask = cv2.imread("mask.png", cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError("mask.png not found in the current folder.")
    if len(mask.shape) != 3 or mask.shape[2] != 4:
        raise RuntimeError("mask.png must be a transparent PNG with alpha (RGBA).")

    face_cascade = load_cascade()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened via /dev/video0. If libcamera-hello works but /dev/video0 is missing, you need a different capture method.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            pad_w = int(0.10 * w)
            pad_h = int(0.25 * h)
            x0 = x - pad_w
            y0 = y - pad_h
            w0 = w + 2 * pad_w
            h0 = h + 2 * pad_h
            frame = overlay_rgba(frame, mask, x0, y0, w0, h0)

        cv2.imshow("Face Overlay", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
