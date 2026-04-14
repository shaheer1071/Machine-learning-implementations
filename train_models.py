import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("Loading features...")
with open('features.pkl', 'rb') as f:
    X, y = pickle.load(f)

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA - keep 300 components (more variance retained)
print("Applying PCA (300 components)...")
pca = PCA(n_components=300, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

models = {
    'KNN (k=5)':        KNeighborsClassifier(n_neighbors=5, metric='manhattan', n_jobs=-1),
    'Decision Tree':    DecisionTreeClassifier(max_depth=15, min_samples_split=4, min_samples_leaf=2, random_state=42),
    'Naive Bayes':      GaussianNB(var_smoothing=1e-9)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {
        'model': model,
        'accuracy': acc,
        'y_pred': y_pred,
        'report': classification_report(y_test, y_pred, target_names=['Cat', 'Dog'])
    }
    print(f"  Accuracy: {acc * 100:.2f}%")
    print(results[name]['report'])

# Save everything
with open('models.pkl', 'wb') as f:
    pickle.dump({'models': results, 'pca': pca, 'scaler': scaler}, f)
print("\nModels saved to models.pkl")

# --- Accuracy Bar Chart ---
names = list(results.keys())
accuracies = [results[n]['accuracy'] * 100 for n in names]
colors = ['#4e79a7', '#59a14f', '#e15759']

plt.figure(figsize=(8, 5))
bars = plt.bar(names, accuracies, color=colors, edgecolor='white', linewidth=1.2)
plt.ylim(0, 100)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Model Accuracy Comparison (HOG + Color Features)', fontsize=13)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc:.2f}%', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=150)
plt.show()

# --- Confusion Matrices ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, name, color in zip(axes, names, ['Blues', 'Greens', 'Reds']):
    cm = confusion_matrix(y_test, results[name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                xticklabels=['Cat', 'Dog'],
                yticklabels=['Cat', 'Dog'],
                cmap=color)
    ax.set_title(f'{name}\nAcc: {results[name]["accuracy"]*100:.2f}%', fontsize=11)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150)
plt.show()

best = max(results, key=lambda n: results[n]['accuracy'])
print(f"\nBest Model: {best} — {results[best]['accuracy']*100:.2f}% accuracy")
