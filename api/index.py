import io, os, tempfile
import numpy as np
import cv2
from flask import Flask, request, send_file, jsonify
from PIL import Image
from dotenv import load_dotenv

load_dotenv()  # baca file .env saat lokal; di Vercel pakai Environment Variables

MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "4"))   # batas Vercel ~4.5MB
MAX_SIDE = int(os.getenv("MAX_SIDE", "720"))
FRAMES = int(os.getenv("FRAMES", "20"))
FPS = int(os.getenv("FPS", "10"))
CLIP_SECONDS = float(os.getenv("CLIP_SECONDS", "3"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024


def fit(img):
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def apply_flash(img, i, n, on):
    """Efek blitz putih di awal animasi, memudar ke normal."""
    if not on:
        return img
    k = max(1, int(n * 0.25))
    if i < k:
        white = Image.new("RGB", img.size, (255, 255, 255))
        return Image.blend(img, white, 0.9 * (1 - i / k))
    return img


def photo_to_frames(img):
    w, h = img.size
    frames = []
    for i in range(FRAMES):
        t = i / (FRAMES - 1)
        z = 1 + 0.08 * t                      # zoom pelan + geser tipis
        cw, ch = w / z, h / z
        x0, y0 = (w - cw) * t, (h - ch) / 2
        frames.append(img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.LANCZOS))
    return frames


def video_to_frames(data):
    with tempfile.NamedTemporaryFile(suffix=".mp4", dir="/tmp", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        span = max(1, min(total, int(fps * CLIP_SECONDS)))
        frames = []
        for idx in np.linspace(0, span - 1, FRAMES).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                frames.append(fit(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
        cap.release()
        return frames
    finally:
        os.remove(path)


@app.post("/api/convert")
def convert():
    mode = request.form.get("mode", "photo")          # photo | video
    flash_on = request.form.get("flash", "off") == "on"
    file = request.files.get("file")
    if not file:
        return jsonify(error="File belum dipilih"), 400
    try:
        data = file.read()
        if mode == "video":
            frames = video_to_frames(data)
        else:
            frames = photo_to_frames(fit(Image.open(io.BytesIO(data))))
        if len(frames) < 2:
            return jsonify(error="Gagal membaca file"), 422
        n = len(frames)
        frames = [apply_flash(f, i, n, flash_on) for i, f in enumerate(frames)]
        buf = io.BytesIO()
        frames[0].save(buf, "WEBP", save_all=True, append_images=frames[1:],
                       duration=1000 // FPS, loop=0, quality=80)
        buf.seek(0)
        return send_file(buf, mimetype="image/webp")
    except Exception:
        return jsonify(error="Terjadi kesalahan saat memproses file"), 500


@app.errorhandler(413)
def too_big(_):
    return jsonify(error=f"File terlalu besar (maks {MAX_MB} MB)"), 413
