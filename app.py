from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import uuid
from PIL import Image
from ultralytics import YOLO
import numpy as np
import json

app = Flask(__name__)

# Configure upload and results folders
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Load YOLO model
try:
    model = YOLO("best.pt")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if file:
        # Generate unique filename
        original_filename = file.filename
        filename = str(uuid.uuid4()) + os.path.splitext(original_filename)[1]
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        return jsonify({
            'success': True, 
            'filename': filename,
            'original_filename': original_filename,
            'file_path': file_path
        })

@app.route('/detect', methods=['POST'])
def detect_defects():
    data = request.json
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], data['filename'])
    
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 404
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    try:
        # Run detection
        results = model(image_path)
        result = results[0]
        
        # Save result image
        result_filename = f"result_{data['filename']}"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        
        # Plot and save the image with detections
        result_img = result.plot()
        Image.fromarray(result_img).save(result_path)
        
        # Extract detection details
        boxes = result.boxes
        detection_results = []
        
        if len(boxes) > 0:
            classes = boxes.cls.tolist()
            confidences = boxes.conf.tolist()
            class_names = result.names
            
            for i, (cls_id, conf) in enumerate(zip(classes, confidences)):
                class_name = class_names[int(cls_id)]
                detection_results.append({
                    'id': i+1,
                    'class': class_name,
                    'confidence': round(conf, 2)
                })
        
        return jsonify({
            'success': True,
            'result_filename': result_filename,
            'result_path': f"/static/results/{result_filename}",
            'detections_count': len(boxes),
            'detections': detection_results
        })
        
    except Exception as e:
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
