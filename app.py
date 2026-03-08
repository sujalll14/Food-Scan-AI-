import os
import io
import json
import base64
import numpy as np
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_from_directory, render_template

app = Flask(__name__, static_folder="static")

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB max payload
MODEL_NAME = "foodlens_model.h5"
CLASSES_FILE = "class_indices.json"
NUTRITION_FILE = "nutrition.json"

# Global lazy-loaded variables
model = None
class_indices = {}
nutrition_data = {}

def load_data():
    global model, class_indices, nutrition_data
    if model is None:
        if os.path.exists(MODEL_NAME):
            print("Loading ML model...")
            model = tf.keras.models.load_model(MODEL_NAME)
        else:
            print(f"Warning: Model {MODEL_NAME} not found. Predictions will fail until trained.")
            return False

    if not class_indices and os.path.exists(CLASSES_FILE):
        with open(CLASSES_FILE, 'r') as f:
            class_indices = json.load(f)

    if not nutrition_data and os.path.exists(NUTRITION_FILE):
        with open(NUTRITION_FILE, 'r') as f:
            nutrition_data = json.load(f)
            
    return True

def predict_single_image(img):
    """Expects a PIL Image object."""
    if not model or not class_indices:
        if not load_data():
            return {"error": "Model or classes not loaded. Please train the model first."}
            
    try:
        # Resize to match MobileNetV2 expected input
        img = img.resize((224, 224))
        
        # Ensure RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img_array = keras_image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create batch
        img_array /= 255.0 # Rescale 0-1
        
        # Predict
        predictions = model.predict(img_array)
        class_idx = str(np.argmax(predictions[0]))
        predicted_class = class_indices.get(class_idx, "Unknown")
        confidence = float(predictions[0][int(class_idx)])
        
        # Format for new UI
        result = {
            "predictions": [
                {
                    "name": predicted_class.replace('_', ' ').title(),
                    "raw": predicted_class,
                    "conf": round(confidence * 100, 1)
                }
            ]
        }
            
        return result
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"error": str(e)}

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    results = []
    
    # Preload model if not loaded
    if model is None:
        load_data()
        
    req_json = request.get_json()
    if not req_json or 'images' not in req_json:
        return jsonify({"error": "Invalid request format. Expected JSON with 'images' array."}), 400
        
    for img_data in req_json['images']:
        try:
            b64_str = img_data['b64']
            
            # remove header if it somehow got included
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
                
            img_bytes = base64.b64decode(b64_str)
            img = Image.open(io.BytesIO(img_bytes))
            
            res = predict_single_image(img)
            results.append(res)
        except Exception as e:
             results.append({"error": f"Failed evaluating: {e}"})

    return jsonify({"results": results})

if __name__ == '__main__':
    # Initial load attempt at startup
    load_data()
    print("Starting FoodLens web server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
