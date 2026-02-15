import os
import uuid
from flask import Flask, render_template, request, send_file, url_for
import cv2
import numpy as np
from skimage import exposure, img_as_ubyte

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# -----------------------------
# Image preprocessing pipeline
# -----------------------------
def homomorphic_filter(gray_u8: np.ndarray, sigma=30.0, gamma_low=0.5, gamma_high=1.5) -> np.ndarray:
    gray = gray_u8.astype(np.float32) / 255.0
    gray = np.clip(gray, 1e-6, 1.0)

    log_img = np.log(gray)
    illum = cv2.GaussianBlur(log_img, (0, 0), sigmaX=sigma, sigmaY=sigma)

    reflect = log_img - illum
    out = gamma_low * illum + gamma_high * reflect

    exp_img = np.exp(out)
    exp_img = exp_img / (np.max(exp_img) + 1e-6)
    return img_as_ubyte(exp_img)


def clahe_normalize(gray_u8: np.ndarray, clip_limit=2.0, tile_grid_size=8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(gray_u8)


def bilateral_denoise(gray_u8: np.ndarray, d=7, sigma_color=50, sigma_space=50) -> np.ndarray:
    return cv2.bilateralFilter(gray_u8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def gaussian_denoise(gray_u8: np.ndarray, ksize=5, sigma=1.0) -> np.ndarray:
    return cv2.GaussianBlur(gray_u8, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def edge_mask(gray_u8: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray_u8, 60, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.GaussianBlur(edges, (5, 5), 0)
    return edges.astype(np.float32) / 255.0


def edge_preserving_sharpen(gray_u8: np.ndarray, amount=1.2) -> np.ndarray:
    base = gray_u8.astype(np.float32)
    blurred = cv2.GaussianBlur(gray_u8, (5, 5), 1.0).astype(np.float32)
    detail = base - blurred

    m = edge_mask(gray_u8)  # [0..1]
    sharpened = base + amount * (detail * m)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def preprocess_face(gray_u8: np.ndarray) -> np.ndarray:
    # 1) Illumination normalization
    x = homomorphic_filter(gray_u8, sigma=30.0, gamma_low=0.5, gamma_high=1.5)
    x = clahe_normalize(x, clip_limit=2.0, tile_grid_size=8)

    # 2) Denoising
    x = bilateral_denoise(x, d=7, sigma_color=50, sigma_space=50)
    x = gaussian_denoise(x, ksize=5, sigma=1.0)

    # 3) Edge enhancement without amplifying noise
    x = edge_preserving_sharpen(x, amount=1.2)

    # 4) Final normalization
    xf = x.astype(np.float32) / 255.0
    xf = exposure.rescale_intensity(xf, in_range=(0.02, 0.98))
    return img_as_ubyte(xf)


def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename.lower())[1]
    return ext in ALLOWED_EXT


# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        original_url=None,
        enhanced_url=None,
        download_url=None,
        error=None
    )


@app.route("/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return render_template("index.html", original_url=None, enhanced_url=None, download_url=None,
                               error="No file part found.")

    f = request.files["image"]
    if f.filename == "":
        return render_template("index.html", original_url=None, enhanced_url=None, download_url=None,
                               error="No file selected.")

    if not allowed_file(f.filename):
        return render_template("index.html", original_url=None, enhanced_url=None, download_url=None,
                               error="Unsupported file type. Use JPG/PNG/etc.")

    uid = str(uuid.uuid4())
    in_name = f"{uid}_input.png"
    out_name = f"{uid}_enhanced.png"

    in_path = os.path.join(UPLOAD_DIR, in_name)
    out_path = os.path.join(OUTPUT_DIR, out_name)

    f.save(in_path)

    img = cv2.imread(in_path)
    if img is None:
        return render_template("index.html", original_url=None, enhanced_url=None, download_url=None,
                               error="Could not read the uploaded image.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = preprocess_face(gray)
    cv2.imwrite(out_path, enhanced)

    original_url = url_for("serve_upload", filename=in_name)
    enhanced_url = url_for("serve_output", filename=out_name)
    download_url = url_for("download_output", filename=out_name)

    return render_template(
        "index.html",
        original_url=original_url,
        enhanced_url=enhanced_url,
        download_url=download_url,
        error=None
    )


@app.route("/uploads/<filename>")
def serve_upload(filename):
    return send_file(os.path.join(UPLOAD_DIR, filename), mimetype="image/png")


@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_file(os.path.join(OUTPUT_DIR, filename), mimetype="image/png")


@app.route("/download/<filename>")
def download_output(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    return send_file(path, as_attachment=True, download_name=filename)


if __name__ == "__main__":
    # Local run only. Render production run will use gunicorn.
    app.run(debug=True)
