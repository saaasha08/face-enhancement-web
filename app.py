#!/usr/bin/env python3
"""
Face Image Enhancement for Recognition Systems
Professional Flask + OpenCV web application
"""

import os
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
import traceback
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'face-enhancement-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG'}

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_image(filepath):
    """Load image from path and convert to RGB."""
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"Could not load image from {filepath}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image(img, filename):
    """Save image to outputs folder and return relative path."""
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, img_bgr)
    return f'/outputs/{filename}'

# Enhancement Functions
def illumination_normalization(img):
    """Output 1: Illumination Normalization"""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return result

def gaussian_filtering(img):
    """Output 2: Gaussian Filtering"""
    return cv2.GaussianBlur(img, (5, 5), 1.5)

def bilateral_filtering(img):
    """Output 3: Bilateral Filtering"""
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

def edge_enhancement(img):
    """Output 4: Edge Enhancement"""
    kernel_sharpen = np.array([[-1, -1, -1],
                                [-1,  9, -1],
                                [-1, -1, -1]]) / 1.0
    return cv2.filter2D(img, -1, kernel_sharpen)

def final_enhanced_image(img):
    """Output 5: Final Enhanced Image"""
    # CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    # Bilateral
    result = cv2.bilateralFilter(result, d=7, sigmaColor=50, sigmaSpace=50)
    
    # Sharpening
    kernel_sharpen = np.array([[-1, -1, -1],
                                [-1,  9, -1],
                                [-1, -1, -1]]) / 1.0
    result = cv2.filter2D(result, -1, kernel_sharpen)
    
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/outputs/<filename>')
def serve_output(filename):
    """Serve output images directly"""
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='image/jpeg')
    return 'File not found', 404

@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serve upload images directly"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='image/jpeg')
    return 'File not found', 404

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and processing"""
    print("=== Upload request received ===")
    
    if 'image' not in request.files:
        print("No image file in request")
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    print(f"File received: {file.filename}")
    
    if file.filename == '':
        print("Empty filename")
        return jsonify({'error': 'No image selected'}), 400
    
    if not allowed_file(file.filename):
        print(f"Invalid file type: {file.filename}")
        return jsonify({'error': 'Invalid file type. Use JPG, JPEG, or PNG'}), 400
    
    try:
        # Generate unique filename
        ext = file.filename.rsplit('.', 1)[1].lower()
        unique_id = str(uuid.uuid4())[:8]
        original_filename = f"original_{unique_id}.{ext}"
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        file.save(original_path)
        print(f"Image saved to: {original_path}")
        
        # Load and process image
        img = load_image(original_path)
        print(f"Image loaded: {img.shape}")
        
        # Apply all enhancements
        print("Processing illumination_normalization...")
        output1 = illumination_normalization(img)
        print("Processing gaussian_filtering...")
        output2 = gaussian_filtering(img)
        print("Processing bilateral_filtering...")
        output3 = bilateral_filtering(img)
        print("Processing edge_enhancement...")
        output4 = edge_enhancement(img)
        print("Processing final_enhanced_image...")
        output5 = final_enhanced_image(img)
        
        # Save all outputs
        output_paths = {}
        
        # Original
        output_paths['original'] = f'/uploads/{original_filename}'
        
        # Output 1
        out1_path = f"output1_{unique_id}.jpg"
        save_image(output1, out1_path)
        output_paths['output1'] = f'/outputs/{out1_path}'
        
        # Output 2
        out2_path = f"output2_{unique_id}.jpg"
        save_image(output2, out2_path)
        output_paths['output2'] = f'/outputs/{out2_path}'
        
        # Output 3
        out3_path = f"output3_{unique_id}.jpg"
        save_image(output3, out3_path)
        output_paths['output3'] = f'/outputs/{out3_path}'
        
        # Output 4
        out4_path = f"output4_{unique_id}.jpg"
        save_image(output4, out4_path)
        output_paths['output4'] = f'/outputs/{out4_path}'
        
        # Output 5
        out5_path = f"output5_{unique_id}.jpg"
        save_image(output5, out5_path)
        output_paths['output5'] = f'/outputs/{out5_path}'
        
        print("All images saved successfully!")
        print(f"Output paths: {output_paths}")
        
        return jsonify({
            'success': True,
            'outputs': output_paths
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download processed image"""
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True, download_name=f"enhanced_face_{filename}")
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    print("=" * 50)
    print("Starting Face Enhancement App...")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Output folder: {app.config['OUTPUT_FOLDER']}")
    print("Server running at: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, host='127.0.0.1', port=5000)