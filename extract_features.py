import os
import cv2
import numpy as np
import pickle
from skimage.feature import hog

IMG_SIZE = 80
MAX_PER_CLASS = 5000  # use 5000 cats + 5000 dogs = 10000 total (balanced, faster)

def extract_hog_features(folder, label, max_samples):
    features = []
    labels = []
    count = 0
    for filename in os.listdir(folder):
        if count >= max_samples:
            break
        if not filename.endswith('.jpg'):
            continue
        path = os.path.join(folder, filename)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # HOG features - captures edges and shapes, much better than raw pixels
        hog_feat = hog(
            img_gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )

        # Also add color histogram (RGB channel stats)
        color_hist = []
        for ch in range(3):
            hist = cv2.calcHist([img], [ch], None, [32], [0, 256])
            color_hist.extend(hist.flatten())

        # Combine HOG + color histogram
        combined = np.concatenate([hog_feat, color_hist])
        features.append(combined)
        labels.append(label)
        count += 1

    print(f"  Extracted {count} samples from {folder}")
    return features, labels

print("Extracting HOG + Color features from Cat images...")
cat_features, cat_labels = extract_hog_features('Cat', 0, MAX_PER_CLASS)

print("Extracting HOG + Color features from Dog images...")
dog_features, dog_labels = extract_hog_features('Dog', 1, MAX_PER_CLASS)

X = np.array(cat_features + dog_features, dtype=np.float32)
y = np.array(cat_labels + dog_labels)

print(f"\nTotal samples: {len(X)}, Feature vector size: {X.shape[1]}")

with open('features.pkl', 'wb') as f:
    pickle.dump((X, y), f)

print("Features saved to features.pkl")
