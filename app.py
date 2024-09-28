import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

model_path = 'optimized_residual_model.h5'  # Use relative path if the file is in the same folder
model = load_model(model_path)

# Class labels (update based on your dataset's classes)
class_labels = ['fully_ripened', 'green', 'half_ripened']

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
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
            
            return jsonify({
                'prediction': predicted_class,
                'confidence': float(np.max(predictions))
            })
        except Exception as e:
            return jsonify({'error': f'Error processing the image: {str(e)}'}), 500

    return jsonify({'error': 'Invalid request'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
