import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from tensorflow.keras.models import load_model
import tensorflow as tf

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best_gender_model.keras')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model with enhanced error handling
try:
    model = load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from: {MODEL_PATH}")
    # Warm up model
    dummy_input = np.zeros((1, 128, 128, 3))
    model.predict(dummy_input, verbose=0)
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    raise SystemExit(1)

# Enhanced face detector with multiple cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_faces(img):
    """Detect faces using multiple cascade classifiers"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect frontal faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Detect profile faces if no frontal faces found
    if len(faces) == 0:
        faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    return faces

def preprocess_face(img, face_coords):
    """Crop and preprocess face region"""
    x, y, w, h = face_coords
    face_img = img[y:y+h, x:x+w]
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = cv2.resize(face_img, (128, 128))
    return np.expand_dims(face_img / 255.0, axis=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Allowed file types: png, jpg, jpeg'}), 400
    
    try:
        # Read and decode image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Detect faces
        faces = detect_faces(img)
        if len(faces) == 0:
            return jsonify({'error': 'No face detected'}), 400
        
        # Process all detected faces
        predictions = []
        for (x, y, w, h) in faces:
            processed_img = preprocess_face(img, (x, y, w, h))
            pred = model.predict(processed_img, verbose=0)[0]
            confidence = float(np.max(pred))
            gender = 'Male' if np.argmax(pred) == 0 else 'Female'
            predictions.append({
                'gender': gender,
                'confidence': confidence,
                'coords': (int(x), int(y), int(w), int(h))
            })
        
        # Save original file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.seek(0)
        file.save(filepath)
        
        return jsonify({
            'predictions': predictions,
            'image_url': f'/static/uploads/{filename}'
        })
        
    except Exception as e:
        print(f"⚠️ Error: {str(e)}")
        return jsonify({'error': 'Processing failed'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)