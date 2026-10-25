import io, os
import numpy as np
from flask import Flask, request, send_file, jsonify
from PIL import Image, ImageFilter
from dotenv import load_dotenv

load_dotenv()  # baca .env saat lokal; di Vercel pakai Environment Variables

MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "4"))     # batas Vercel ~4.5MB
MAX_SIDE = int(os.getenv("MAX_SIDE", "640"))
FPS = int(os.getenv("FPS", "10"))
HOLD_MS = int(os.getenv("HOLD_MS", "1500"))       # lama foto diam di akhir
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
    blur = np.asarray(img.filter(ImageFilter.GaussianBlur(max(img.size) / 40)), dtype=np.float32)
    base = a * 0.6 + blur * 0.4
    out = base * (1 + 3 * level) + 255 * 0.55 * level
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


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


@app.get("/api/health")
def health():
    return jsonify(ok=True)


@app.post("/api/convert")
def convert():
    flash_on = request.form.get("flash", "off") == "on"
    files = request.files.getlist("frames")      # frame sudah diambil di browser
    if len(files) < 2:
        return jsonify(error="Frame video belum terkirim"), 400
    try:
        frames = [fit(Image.open(f.stream)) for f in files]
        key = frames[len(frames) // 2]           # foto utama = frame tengah
        seq, dur = list(frames), [1000 // FPS] * len(frames)
        if flash_on:                             # video -> flash -> foto
            for level, ms in FLASH:
                seq.append(flash_frame(key, level))
                dur.append(ms)
        seq.append(key)
        dur.append(HOLD_MS)
        return send_file(encode(seq, dur), mimetype="image/webp")
    except Exception as e:
        app.logger.exception("convert gagal")
        return jsonify(error=f"Server error: {type(e).__name__}: {e}"), 500


@app.errorhandler(413)
def too_big(_):
    return jsonify(error="Data terlalu besar, pilih durasi lebih pendek"), 413
