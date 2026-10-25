import io, os, tempfile
import numpy as np
import cv2
from flask import Flask, request, send_file, jsonify
from PIL import Image
from dotenv import load_dotenv

load_dotenv()  # baca .env saat lokal; di Vercel pakai Environment Variables

MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "4"))        # batas Vercel ~4.5MB
MAX_SIDE = int(os.getenv("MAX_SIDE", "720"))
FRAMES = int(os.getenv("FRAMES", "24"))               # jumlah frame gerakan
CLIP_SECONDS = float(os.getenv("CLIP_SECONDS", "3"))  # lama gerakan
HOLD_MS = int(os.getenv("HOLD_MS", "1500"))           # lama foto diam di akhir
QUALITY = int(os.getenv("QUALITY", "75"))

# Kilatan flash asli: naik instan, lalu meredup cepat (intensitas, durasi ms)
FLASH = [(1.0, 45), (0.7, 45), (0.42, 55), (0.24, 60), (0.12, 70), (0.05, 80)]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024


def fit(img):
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def flash_frame(img, level):
    """Overexposed + bloom, bukan sekadar blend putih."""
    a = np.asarray(img, dtype=np.float32)
    blur = cv2.GaussianBlur(a, (0, 0), max(img.size) / 40)
    base = a * 0.6 + blur * 0.4
    out = base * (1 + 3 * level) + 255 * 0.55 * level
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def read_video(data):
    """Ambil klip di sekitar TENGAH video; foto utama = frame tengah."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", dir="/tmp", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        if total < 2:
            return [], None
        mid = total // 2
        clip = max(2, int(fps * CLIP_SECONDS))
        if total <= clip:
            start, end = 0, total
        else:
            start = max(0, mid - clip // 2)
            end = min(total, start + clip)
        wanted = set(np.linspace(start, end - 1, FRAMES).astype(int).tolist())
        wanted.add(mid)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames, key = [], None
        for idx in range(start, end):
            ok, fr = cap.read()
            if not ok:
                break
            if idx in wanted:
                im = fit(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
                if idx == mid:
                    key = im
                if idx in set(np.linspace(start, end - 1, FRAMES).astype(int).tolist()):
                    frames.append(im)
        cap.release()
        return frames, key or (frames[len(frames) // 2] if frames else None)
    finally:
        os.remove(path)


def encode(frames, durations):
    q = QUALITY
    while True:
        buf = io.BytesIO()
        frames[0].save(buf, "WEBP", save_all=True, append_images=frames[1:],
                       duration=durations, loop=0, quality=q)
        if buf.tell() < 4_200_000 or q <= 40:
            buf.seek(0)
            return buf
        q -= 15


@app.post("/api/convert")
def convert():
    flash_on = request.form.get("flash", "off") == "on"
    file = request.files.get("file")
    if not file:
        return jsonify(error="File belum dipilih"), 400
    try:
        data = file.read()
        frames, key = read_video(data)
        if len(frames) < 2 or key is None:
            return jsonify(error="Gagal membaca file"), 422

        step = max(20, int(CLIP_SECONDS * 1000 / len(frames)))
        seq, dur = list(frames), [step] * len(frames)
        if flash_on:                                    # video/gerakan -> flash -> foto
            for level, ms in FLASH:
                seq.append(flash_frame(key, level))
                dur.append(ms)
        seq.append(key)                                 # berhenti di foto
        dur.append(HOLD_MS)
        return send_file(encode(seq, dur), mimetype="image/webp")
    except Exception:
        return jsonify(error="Terjadi kesalahan saat memproses file"), 500


@app.errorhandler(413)
def too_big(_):
    return jsonify(error=f"File terlalu besar (maks {MAX_MB} MB)"), 413
