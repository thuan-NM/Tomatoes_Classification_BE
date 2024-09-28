import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image


model_path = r'C:\Users\nguye\OneDrive\Desktop\Deep Learning\project\Tomatoes_Classification_BE\optimized_residual_model.h5'  # Use relative path if the file is in the same folder
model = torch.load(model_path)  # Load the model
model.eval()  # Set the model to evaluation mode

# Class labels (change according to your dataset's classes)
class_labels = ['fully_ripened', 'green', 'half_ripened']  # Update based on your classes

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define image transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to match model input
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Define the prediction route
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
            img_tensor = preprocess(img)  # Preprocess the image
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
            
            # Make a prediction
            with torch.no_grad():  # Disable gradient calculation
                predictions = model(img_tensor)  # Get predictions
            
            predicted_class = class_labels[torch.argmax(predictions).item()]  # Get the highest score
            confidence = torch.max(torch.softmax(predictions, dim=1)).item()  # Get confidence score
            
            # Return the result as JSON
            return jsonify({
                'prediction': predicted_class,
                'confidence': confidence
            })
        except Exception as e:
            return jsonify({'error': f'Error processing the image: {str(e)}'}), 500

    return jsonify({'error': 'Invalid request'}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)