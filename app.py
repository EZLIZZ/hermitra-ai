import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model
MODEL_PATH = "model/bestmodel.h5"
model = load_model(MODEL_PATH)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return jsonify({"message": "PCOS Detection API is running"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Save the uploaded image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess the image
    img = load_img(filepath, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
       label = "Not Affected"
       confidence = prediction          # float, e.g. 0.87
    else:
       label = "Affected"
       confidence = 1 - prediction      # float, e.g. 0.93

# Now format confidence as a percentage string
    confidence_str = f"{confidence * 100:.2f}%"

    return jsonify({
        "prediction": label,
        "confidence": confidence_str,
        "image_path": filepath
    })

if __name__ == '__main__':
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)

