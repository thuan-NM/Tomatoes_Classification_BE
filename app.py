import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import time

model_path = 'optimized_residual_model.h5'
model = load_model(model_path)

class_labels = ['fully_ripened', 'green', 'half_ripened']

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

@app.route('/')
def home():
    return "Welcome to the Tomato Classification API!"

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        try:
            img = Image.open(file).convert('RGB')
            img = img.resize((224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = model.predict(img_array)
            predicted_class = class_labels[np.argmax(predictions)]
            
            end_time = time.time()
            print(f"Prediction took {end_time - start_time} seconds")
            
            return jsonify({
                'prediction': predicted_class,
                'confidence': float(np.max(predictions))
            })
        except Exception as e:
            return jsonify({'error': f'Error processing the image: {str(e)}'}), 500

    return jsonify({'error': 'Invalid request'}), 400

if __name__ == '__main__':
    app.run(debug=True)
