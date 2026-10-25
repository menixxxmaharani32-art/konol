import io, os, base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from dotenv import load_dotenv

load_dotenv()  # baca .env saat lokal; di Vercel pakai Environment Variables

MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "4"))
MAX_SIDE = int(os.getenv("MAX_SIDE", "720"))
HOLD_MS = int(os.getenv("HOLD_MS", "1500"))        # lama foto diam di akhir
JPEG_Q = int(os.getenv("JPEG_QUALITY", "85"))
HD_MAX_SIDE = int(os.getenv("HD_MAX_SIDE", "4096"))   # sisi terpanjang hasil HD

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


def hd_enhance(img, strong=False):
    """HD tanpa AI: rapikan noise/blok JPEG, perbesar 2x, lalu perjelas detail
    di kanal terang (multi-skala) + kontras lokal. Aman, tidak ada gambar pecah."""
    rgb = np.asarray(img)
    h, w = rgb.shape[:2]
    factor = min(2.0, HD_MAX_SIDE / max(w, h))
    den = cv2.bilateralFilter(rgb, 5, 22, 5)                       # kurangi noise & blok JPEG
    up = cv2.resize(den, None, fx=factor, fy=factor, interpolation=cv2.INTER_LANCZOS4) if factor > 1.05 else den
    lab = cv2.cvtColor(up, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]
    Lf = L.astype(np.float32)
    a1, a2 = (0.9, 0.4) if strong else (0.6, 0.25)
    e = Lf + a1 * (Lf - cv2.GaussianBlur(Lf, (0, 0), 1.0)) + a2 * (Lf - cv2.GaussianBlur(Lf, (0, 0), 2.5))
    k = np.ones((3, 3), np.uint8)                                   # batasi overshoot agar tanpa halo
    e = np.clip(e, cv2.erode(L, k).astype(np.float32) - 3, cv2.dilate(L, k).astype(np.float32) + 3)
    L2 = np.clip(e, 0, 255).astype(np.uint8)
    cl = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(L2)     # kontras lokal, dicampur halus
    mix = 0.45 if strong else 0.3
    L2 = cv2.addWeighted(L2, 1 - mix, cl, mix, 0)
    lab[:, :, 0] = L2
    out = Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
    return ImageEnhance.Color(out).enhance(1.08 if strong else 1.04)


def encode_jpeg(img, limit=4_200_000):
    while True:
        for q in (90, 82, 74, 66):
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=q, optimize=True)
            if buf.tell() < limit:
                buf.seek(0)
                return buf
        img = img.resize((int(img.width * .85), int(img.height * .85)), Image.LANCZOS)


@app.post("/api/hd")
def hd():
    """HD Foto: perbesar 2x + perjelas detail (Python + OpenCV)."""
    f = request.files.get("image")
    if not f:
        return jsonify(error="Foto belum terkirim"), 400
    try:
        img = ImageOps.exif_transpose(Image.open(f.stream)).convert("RGB")
        img = hd_enhance(img, request.form.get("level") == "strong")
        if max(img.size) > HD_MAX_SIDE:
            img.thumbnail((HD_MAX_SIDE, HD_MAX_SIDE), Image.LANCZOS)
        return send_file(encode_jpeg(img), mimetype="image/jpeg")
    except Exception as e:
        app.logger.exception("hd gagal")
        return jsonify(error=f"Server error: {type(e).__name__}: {e}"), 500


@app.errorhandler(413)
def too_big(_):
    return jsonify(error="Data terlalu besar"), 413
