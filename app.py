"""
Flask API Backend for AI Image Detection Frontend.

This provides REST endpoints for the web interface to analyze images.
"""

import os
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from detector import FoodImageDetector

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)

# Configuration
UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif'}

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Initialize detector (lazy loading)
detector = None

def get_detector():
    """Lazy load the detector to avoid slow startup."""
    global detector
    if detector is None:
        print("🔄 Loading AI detector model...")
        detector = FoodImageDetector()
        print("✅ Detector ready!")
    return detector


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Serve the main frontend page."""
    return send_from_directory('static', 'index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """
    Analyze an uploaded image for AI generation.
    
    Returns:
        JSON with detection results
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: jpg, jpeg, png, webp, gif'}), 400
    
    try:
        # Save file temporarily
        filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))
        
        # Run detection
        det = get_detector()
        result = det.predict(str(filepath))
        
        # Clean up uploaded file
        os.remove(filepath)
        
        # Return results
        return jsonify({
            'success': True,
            'filename': file.filename,
            'ai_probability': round(result.ai_probability * 100, 2),
            'real_probability': round(result.real_probability * 100, 2),
            'decision': result.decision.value,
            'decision_emoji': result.decision.emoji,
            'decision_description': result.decision.description,
            'ai_label': result.ai_label,
            'real_label': result.real_label
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector is not None
    })


if __name__ == '__main__':
    print("=" * 60)
    print("🍕 Food Image AI Detection - Web Interface")
    print("=" * 60)
    print("📌 Open http://localhost:5000 in your browser")
    print("=" * 60)
    
    # Pre-load the model
    get_detector()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
