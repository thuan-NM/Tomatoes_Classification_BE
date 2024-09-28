import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image

model_path = r'C:\Users\nguye\OneDrive\Desktop\Deep Learning\project\Tomatoes_Classification_BE\optimized_residual_model.h5'  # Use relative path if the file is in the same folder
model = load_model(model_path)


# Class labels (change according to your dataset's classes)
class_labels = ['fully_ripened', 'green', 'half_ripened']  # Update based on your classes

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the home route
@app.route('/predict', methods=['POST'])
def predict():
    # Ensure a file is included in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        try:
            # Preprocess the image
            img = Image.open(file).convert('RGB')
            img = img.resize((224, 224))  # Resize image to match model input
            img_array = img_to_array(img) / 255.0  # Normalize the image
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
            # Make a prediction
            predictions = model.predict(img_array)
            predicted_class = class_labels[np.argmax(predictions)]  # Get the highest score
            
            # Return the result as JSON
            return jsonify({
                'prediction': predicted_class,
                'confidence': float(np.max(predictions))
            })
        except Exception as e:
            return jsonify({'error': f'Error processing the image: {str(e)}'}), 500

    return jsonify({'error': 'Invalid request'}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
