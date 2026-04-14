from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import cv2
from skimage.feature import hog

app = Flask(__name__)

with open('models.pkl', 'rb') as f:
    data = pickle.load(f)
    models = data['models']
    pca = data['pca']
    scaler = data['scaler']

IMG_SIZE = 80

def preprocess_image(file):
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_feat = hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    color_hist = []
    for ch in range(3):
        hist = cv2.calcHist([img], [ch], None, [64], [0, 256])
        color_hist.extend(hist.flatten())

    combined = np.concatenate([hog_feat, color_hist]).reshape(1, -1)
    combined_scaled = scaler.transform(combined)
    combined_pca = pca.transform(combined_scaled)
    return combined_pca

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    features = preprocess_image(file)

    label_map = {0: 'CAT', 1: 'DOG'}
    predictions = {}
    for name, result in models.items():
        pred = result['model'].predict(features)[0]
        predictions[name] = {
            'prediction': label_map[pred],
            'accuracy': f"{result['accuracy']*100:.2f}%"
        }

    best = max(models, key=lambda n: models[n]['accuracy'])
    return jsonify({'predictions': predictions, 'best_model': best})

if __name__ == '__main__':
    app.run(debug=True)
