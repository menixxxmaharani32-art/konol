import io, os, base64
import numpy as np
from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from dotenv import load_dotenv

load_dotenv()  # baca .env saat lokal; di Vercel pakai Environment Variables

MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "4"))
MAX_SIDE = int(os.getenv("MAX_SIDE", "720"))
HOLD_MS = int(os.getenv("HOLD_MS", "1500"))        # lama foto diam di akhir
JPEG_Q = int(os.getenv("JPEG_QUALITY", "85"))
HD_MAX_SIDE = int(os.getenv("HD_MAX_SIDE", "3200"))   # sisi terpanjang hasil HD

# Kilatan flash asli: naik instan, lalu meredup cepat (intensitas, durasi ms)
FLASH = [(1.0, 45), (0.7, 45), (0.42, 55), (0.24, 60), (0.12, 70), (0.05, 80)]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024


def flash_frame(img, level):
    """Overexposed + bloom, bukan sekadar blend putih."""
    a = np.asarray(img, dtype=np.float32)
    blur = np.asarray(img.filter(ImageFilter.GaussianBlur(max(img.size) / 40)), dtype=np.float32)
    base = a * 0.6 + blur * 0.4
    out = base * (1 + 3 * level) + 255 * 0.55 * level
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


@app.get("/api/health")
def health():
    return jsonify(ok=True)


@app.post("/api/flash")
def flash():
    """Terima foto utama (frame tengah), kembalikan frame-frame kilatan flash."""
    f = request.files.get("key")
    if not f:
        return jsonify(error="Foto utama belum terkirim"), 400
    try:
        key = Image.open(f.stream).convert("RGB")
        key.thumbnail((MAX_SIDE, MAX_SIDE))
        frames = []
        for level, _ in FLASH:
            buf = io.BytesIO()
            flash_frame(key, level).save(buf, "JPEG", quality=JPEG_Q)
            frames.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
        return jsonify(frames=frames, durations=[ms for _, ms in FLASH], hold_ms=HOLD_MS)
    except Exception as e:
        app.logger.exception("flash gagal")
        return jsonify(error=f"Server error: {type(e).__name__}: {e}"), 500


@app.post("/api/hd")
def hd():
    """Perbesar 2x + pertajam foto (tanpa AI, murni Pillow)."""
    f = request.files.get("image")
    if not f:
        return jsonify(error="Foto belum terkirim"), 400
    strong = request.form.get("level") == "strong"
    try:
        img = ImageOps.exif_transpose(Image.open(f.stream)).convert("RGB")
        w, h = img.size
        factor = min(2.0, HD_MAX_SIDE / max(w, h))
        if factor > 1.05:
            img = img.resize((round(w * factor), round(h * factor)), Image.LANCZOS)
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=190 if strong else 120, threshold=2))
        img = ImageEnhance.Contrast(img).enhance(1.06 if strong else 1.03)
        img = ImageEnhance.Color(img).enhance(1.06 if strong else 1.03)
        q = 92
        while True:
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=q, optimize=True)
            if buf.tell() < 4_200_000 or q <= 60:
                break
            q -= 8
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")
    except Exception as e:
        app.logger.exception("hd gagal")
        return jsonify(error=f"Server error: {type(e).__name__}: {e}"), 500


@app.errorhandler(413)
def too_big(_):
    return jsonify(error="Data terlalu besar"), 413
