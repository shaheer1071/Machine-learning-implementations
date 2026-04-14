# Cat vs Dog Classifier

A machine learning web app that classifies images as **Cat** or **Dog** using HOG + Color Histogram features with 3 classical ML algorithms.

## Models Used
- KNN (k=5, manhattan distance)
- Decision Tree (max_depth=15)
- Naive Bayes (GaussianNB)

## How It Works
1. Images are resized to 80x80
2. HOG features extracted from grayscale image
3. Color histogram (64 bins per channel) extracted from RGB
4. Features combined, scaled, and reduced with PCA (300 components)
5. All 3 models predict and the best accuracy model is highlighted

## Project Structure
```
PetImages/
├── extract_features.py   # Extracts HOG + color features from dataset
├── train_models.py       # Trains KNN, Decision Tree, Naive Bayes
├── app.py                # Flask web server
├── templates/
│   └── index.html        # Upload UI
├── accuracy_comparison.png
├── confusion_matrices.png
└── README.md
```

## Setup & Run

### 1. Install dependencies
```bash
pip install opencv-python scikit-image scikit-learn numpy flask matplotlib seaborn
```

### 2. Add dataset
Place your dataset in the following structure:
```
PetImages/
├── Cat/   (jpg images)
└── Dog/   (jpg images)
```

### 3. Extract features
```bash
python extract_features.py
```

### 4. Train models
```bash
python train_models.py
```

### 5. Run the app
```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser, upload any image, and get predictions from all 3 models.

## API Response Example
```json
{
  "predictions": {
    "KNN (k=5)":         { "prediction": "CAT", "accuracy": "80.12%" },
    "Decision Tree":     { "prediction": "CAT", "accuracy": "74.50%" },
    "Naive Bayes":       { "prediction": "DOG", "accuracy": "66.30%" }
  },
  "best_model": "KNN (k=5)"
}
```

## Dataset
Uses the [Microsoft Cats vs Dogs dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765) — 12,500 images per class.

> Note: Cat/ and Dog/ folders are excluded from this repo due to size.
