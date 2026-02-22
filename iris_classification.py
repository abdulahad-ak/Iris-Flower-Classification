"""
Iris Flower Classification
Author: Abdul Ahad Khan Kolachi
Description: Machine learning model to classify iris flowers into species
"""

# ============================================
# IMPORT LIBRARIES
# ============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CREATE OUTPUT DIRECTORIES
# ============================================
os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/models', exist_ok=True)

print("=" * 60)
print("       IRIS FLOWER CLASSIFICATION PROJECT")
print("=" * 60)

# ============================================
# LOAD DATASET
# ============================================
print("\n[1] Loading Iris Dataset...")
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(f"    Dataset Shape: {df.shape}")
print(f"    Features: {iris.feature_names}")
print(f"    Target Classes: {iris.target_names}")

# ============================================
# DATA EXPLORATION
# ============================================
print("\n[2] Exploring Data...")
print("\n    First 5 rows:")
print(df.head())

print("\n    Dataset Info:")
print(f"    - Total Samples: {len(df)}")
print(f"    - Features: {len(iris.feature_names)}")
print(f"    - Classes: {len(iris.target_names)}")

print("\n    Class Distribution:")
print(df['species_name'].value_counts())

print("\n    Statistical Summary:")
print(df.describe())

print("\n    Missing Values:")
print(df.isnull().sum().sum(), "missing values found")

# ============================================
# DATA VISUALIZATION
# ============================================
print("\n[3] Creating Visualizations...")

# Plot 1: Class Distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='species_name', palette='viridis')
plt.title('Iris Species Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Species')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('outputs/plots/01_species_distribution.png', dpi=300)
plt.close()
print("    ✓ Species distribution plot saved")

# Plot 2: Pairplot
plt.figure(figsize=(12, 10))
sns.pairplot(df, hue='species_name', palette='viridis', diag_kind='hist')
plt.suptitle('Pairplot of Iris Features', y=1.02, fontsize=14, fontweight='bold')
plt.savefig('outputs/plots/02_pairplot.png', dpi=300)
plt.close()
print("    ✓ Pairplot saved")

# Plot 3: Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation = df.drop(['species', 'species_name'], axis=1).corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, fmt='.2f')
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/plots/03_correlation_heatmap.png', dpi=300)
plt.close()
print("    ✓ Correlation heatmap saved")

# Plot 4: Box plots for each feature
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
features = iris.feature_names
for idx, ax in enumerate(axes.flat):
    sns.boxplot(data=df, x='species_name', y=features[idx], palette='viridis', ax=ax)
    ax.set_title(f'{features[idx]} by Species', fontweight='bold')
plt.suptitle('Feature Distribution by Species', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/plots/04_boxplots.png', dpi=300)
plt.close()
print("    ✓ Boxplots saved")

# Plot 5: Violin plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, ax in enumerate(axes.flat):
    sns.violinplot(data=df, x='species_name', y=features[idx], palette='viridis', ax=ax)
    ax.set_title(f'{features[idx]} Distribution', fontweight='bold')
plt.suptitle('Violin Plots of Features', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/plots/05_violin_plots.png', dpi=300)
plt.close()
print("    ✓ Violin plots saved")

# Plot 6: Scatter plot - Petal Length vs Petal Width
plt.figure(figsize=(10, 8))
for species in df['species_name'].unique():
    subset = df[df['species_name'] == species]
    plt.scatter(subset['petal length (cm)'], subset['petal width (cm)'], 
                label=species, alpha=0.7, s=100)
plt.xlabel('Petal Length (cm)', fontsize=12)
plt.ylabel('Petal Width (cm)', fontsize=12)
plt.title('Petal Length vs Petal Width', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/plots/06_petal_scatter.png', dpi=300)
plt.close()
print("    ✓ Petal scatter plot saved")

# Plot 7: Scatter plot - Sepal Length vs Sepal Width
plt.figure(figsize=(10, 8))
for species in df['species_name'].unique():
    subset = df[df['species_name'] == species]
    plt.scatter(subset['sepal length (cm)'], subset['sepal width (cm)'], 
                label=species, alpha=0.7, s=100)
plt.xlabel('Sepal Length (cm)', fontsize=12)
plt.ylabel('Sepal Width (cm)', fontsize=12)
plt.title('Sepal Length vs Sepal Width', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/plots/07_sepal_scatter.png', dpi=300)
plt.close()
print("    ✓ Sepal scatter plot saved")

# ============================================
# DATA PREPROCESSING
# ============================================
print("\n[4] Preprocessing Data...")

# Features and Target
X = df.drop(['species', 'species_name'], axis=1)
y = df['species']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                      random_state=42, stratify=y)
print(f"    Training samples: {len(X_train)}")
print(f"    Testing samples: {len(X_test)}")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("    ✓ Features scaled using StandardScaler")

# Save scaler
joblib.dump(scaler, 'outputs/models/scaler.pkl')
print("    ✓ Scaler saved")

# ============================================
# MODEL TRAINING & EVALUATION
# ============================================
print("\n[5] Training Multiple Models...")

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=200),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(kernel='rbf', random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

results = {}

for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred
    }
    print(f"    {name}: {accuracy * 100:.2f}%")

# ============================================
# FIND BEST MODEL
# ============================================
print("\n[6] Selecting Best Model...")

best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
best_accuracy = results[best_model_name]['accuracy']
best_predictions = results[best_model_name]['predictions']

print(f"    ★ Best Model: {best_model_name}")
print(f"    ★ Accuracy: {best_accuracy * 100:.2f}%")

# Save best model
joblib.dump(best_model, 'outputs/models/best_model.pkl')
print(f"    ✓ Best model saved")

# ============================================
# DETAILED EVALUATION
# ============================================
print("\n[7] Detailed Evaluation of Best Model...")
print(f"\n    Classification Report ({best_model_name}):")
print(classification_report(y_test, best_predictions, target_names=iris.target_names))

# Plot 8: Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, best_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('outputs/plots/08_confusion_matrix.png', dpi=300)
plt.close()
print("    ✓ Confusion matrix saved")

# Plot 9: Model Comparison
plt.figure(figsize=(10, 6))
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] * 100 for name in model_names]
colors = ['green' if name == best_model_name else 'steelblue' for name in model_names]
bars = plt.barh(model_names, accuracies, color=colors)
plt.xlabel('Accuracy (%)', fontsize=12)
plt.title('Model Comparison', fontsize=14, fontweight='bold')
plt.xlim(0, 105)
for bar, acc in zip(bars, accuracies):
    plt.text(acc + 1, bar.get_y() + bar.get_height()/2, f'{acc:.2f}%', 
             va='center', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/plots/09_model_comparison.png', dpi=300)
plt.close()
print("    ✓ Model comparison plot saved")

# ============================================
# FEATURE IMPORTANCE (for tree-based models)
# ============================================
print("\n[8] Feature Importance Analysis...")

rf_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'Feature': iris.feature_names,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n    Feature Importance (Random Forest):")
for idx, row in feature_importance.iterrows():
    print(f"    - {row['Feature']}: {row['Importance']:.4f}")

# Plot 10: Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('outputs/plots/10_feature_importance.png', dpi=300)
plt.close()
print("    ✓ Feature importance plot saved")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 60)
print("                    SUMMARY")
print("=" * 60)
print(f"""
    Dataset: Iris Flower Dataset
    Total Samples: 150
    Features: 4 (sepal length, sepal width, petal length, petal width)
    Classes: 3 (setosa, versicolor, virginica)
    
    Train/Test Split: 80/20
    
    Models Trained:
    ---------------
""")
for name in results:
    acc = results[name]['accuracy'] * 100
    star = "★" if name == best_model_name else " "
    print(f"    {star} {name}: {acc:.2f}%")

print(f"""
    ═══════════════════════════════════════
    BEST MODEL: {best_model_name}
    ACCURACY: {best_accuracy * 100:.2f}%
    ═══════════════════════════════════════
    
    Output Files:
    - outputs/models/best_model.pkl
    - outputs/models/scaler.pkl
    - outputs/plots/ (10 visualization plots)
""")
print("=" * 60)
print("    PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)