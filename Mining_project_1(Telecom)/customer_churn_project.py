"""
╔══════════════════════════════════════════════════════════════════════╗
║                      TELCO CUSTOMER CHURN PREDICTION                 ║
║                                                                      ║
║  This standalone script implements the full machine learning         ║
║  pipeline for predicting telecom customer churn, including:          ║
║                                                                      ║
║    1. Data Loading & Preprocessing                                   ║
║    2. Exploratory Data Analysis (EDA)                                ║
║    3. Feature Engineering (One-Hot Encoding + Scaling)               ║
║    4. Model Training: KNN, SVM, Neural Network                       ║
║    5. Evaluation: Accuracy, ROC-AUC, Confusion Matrix                ║
║    6. Model Comparison & Best Model Selection                        ║
║                                                                      ║
║  Dataset : IBM Telco Customer Churn (7043 customers, 21 features)    ║
║  Target  : Churn (Yes/No → 1/0) — Binary Classification              ║
║  Course  : intelligent IS                                            ║
╚══════════════════════════════════════════════════════════════════════╝

Dependencies:
    pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
"""

# ═══════════════════════════════════════════════════════════════
#  1. IMPORT LIBRARIES
# ═══════════════════════════════════════════════════════════════

import pandas as pd               
import numpy as np                
import matplotlib.pyplot as plt    
import seaborn as sns            

# Scikit-learn: ML pipeline components
from sklearn.model_selection import train_test_split   
from sklearn.preprocessing import StandardScaler       
from sklearn.neighbors import KNeighborsClassifier    
from sklearn.svm import SVC                           
from sklearn.metrics import (
    accuracy_score,     
    roc_auc_score,        
    confusion_matrix,   
    roc_curve,            
    classification_report 
)

# TensorFlow / Keras: Deep Learning
import tensorflow as tf
from tensorflow import keras


# ═══════════════════════════════════════════════════════════════
#  2. LOAD & PREPROCESS DATA
#     - Read the raw CSV dataset
#     - Drop non-predictive columns (customerID)
#     - Handle data type issues (TotalCharges stored as string)
#     - Fill missing values appropriately
#     - Encode the target variable (Churn: Yes → 1, No → 0)
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("  📂 STEP 1: Loading & Preprocessing Data")
print("=" * 60)

df = pd.read_csv("Telco-Customer-Churn.csv")
print(f"  ✓ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# Drop customerID — it's a unique identifier, not a feature
df.drop(columns=["customerID"], inplace=True)
print("  ✓ Dropped 'customerID' column")

# TotalCharges has blank strings for new customers (tenure = 0)
# Convert to numeric and fill missing values with the median
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan))
median_charges = df["TotalCharges"].median()
df["TotalCharges"].fillna(median_charges, inplace=True)
print(f"  ✓ TotalCharges cleaned — missing filled with median (${median_charges:.2f})")

# Fill any remaining missing values in categorical columns
for col in df.select_dtypes(include="object").columns:
    if df[col].isna().sum() > 0:
        df[col].fillna("Missing", inplace=True)

# Encode target variable: Yes → 1 (churned), No → 0 (retained)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
churn_rate = df["Churn"].mean() * 100
print(f"  ✓ Target encoded — Churn rate: {churn_rate:.1f}%")
print()


# ═══════════════════════════════════════════════════════════════
#  3. EXPLORATORY DATA ANALYSIS (EDA)
#     - Visualize the class distribution of the target variable
#     - Churned vs. retained customers
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("  📊 STEP 2: Churn Distribution Visualization")
print("=" * 60)

plt.figure(figsize=(7, 5))
ax = sns.countplot(x="Churn", data=df, palette=["#6366f1", "#ec4899"])
plt.title("Churn Distribution", fontsize=16, fontweight="bold")
plt.xlabel("Churn (0 = Retained, 1 = Churned)", fontsize=12)
plt.ylabel("Count", fontsize=12)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha="center", va="bottom", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()
print("  ✓ Churn distribution plot displayed\n")


# ═══════════════════════════════════════════════════════════════
#  4. FEATURE ENGINEERING
#     - One-Hot Encoding for categorical variables (drop_first
#       to avoid multicollinearity / dummy variable trap)
#     - Separate features (X) from target (y)
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("  🔧 STEP 3: Feature Engineering")
print("=" * 60)

# One-hot encode categorical features
X = pd.get_dummies(df.drop(columns=["Churn"]), drop_first=True)
y = df["Churn"]
print(f"  ✓ Features after encoding: {X.shape[1]} columns")
print(f"  ✓ Target: {y.value_counts().to_dict()}")
print()


# ═══════════════════════════════════════════════════════════════
#  5. TRAIN / TEST SPLIT
#     - 80% training, 20% testing
#     - Stratified split ensures both sets have the same churn ratio
#     - random_state=42 for reproducibility
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("  📐 STEP 4: Train / Test Split")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  ✓ Training set: {X_train.shape[0]} samples")
print(f"  ✓ Testing set:  {X_test.shape[0]} samples")
print()


# ═══════════════════════════════════════════════════════════════
#  6. FEATURE SCALING (STANDARDIZATION)
#     - Z-score normalization: (x - mean) / std
#     - Critical for KNN (distance-based) and SVM (margin-based)
#     - Fitted ONLY on training data to prevent data leakage
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("  ⚖️ STEP 5: Feature Scaling (StandardScaler)")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("  ✓ Features standardized (mean=0, std=1)")
print()


# ═══════════════════════════════════════════════════════════════
#  7. MODEL 1: K-NEAREST NEIGHBORS (KNN)
#     - Instance-based learner — classifies by majority vote
#       of the K nearest training samples
#     - k=5 is a common default that balances bias and variance
#     - Uses Euclidean distance on the scaled feature space
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("  🎯 STEP 6: Training KNN (k=5)")
print("=" * 60)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
knn_preds = knn.predict(X_test_scaled)
knn_proba = knn.predict_proba(X_test_scaled)[:, 1]

knn_acc = accuracy_score(y_test, knn_preds)
knn_auc = roc_auc_score(y_test, knn_proba)
print(f"  ✓ KNN Accuracy:  {knn_acc:.4f}")
print(f"  ✓ KNN ROC-AUC:   {knn_auc:.4f}")
print()


# ═══════════════════════════════════════════════════════════════
#  8. MODEL 2: SUPPORT VECTOR MACHINE (SVM)
#     - Finds the optimal hyperplane that maximizes the margin
#       between the two classes
#     - RBF kernel maps data to higher dimensions for
#       non-linear decision boundaries
#     - probability=True enables predict_proba() for ROC analysis
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("  🎯 STEP 7: Training SVM (RBF kernel)")
print("=" * 60)

svm = SVC(kernel="rbf", probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
svm_preds = svm.predict(X_test_scaled)
svm_proba = svm.predict_proba(X_test_scaled)[:, 1]

svm_acc = accuracy_score(y_test, svm_preds)
svm_auc = roc_auc_score(y_test, svm_proba)
print(f"  ✓ SVM Accuracy:  {svm_acc:.4f}")
print(f"  ✓ SVM ROC-AUC:   {svm_auc:.4f}")
print()


# ═══════════════════════════════════════════════════════════════
#  9. MODEL 3: DEEP NEURAL NETWORK (DNN)
#     Architecture:
#       Input(N) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)
#
#     - Binary cross-entropy loss for binary classification
#     - Adam optimizer with adaptive learning rates
#     - AUC tracked as an additional metric during training
#     - 10% validation split for monitoring overfitting
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("  🎯 STEP 8: Training Neural Network (20 epochs)")
print("=" * 60)

tf.random.set_seed(42)

nn = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(64, activation="relu"),    # Hidden layer 1: 64 neurons
    keras.layers.Dense(32, activation="relu"),    # Hidden layer 2: 32 neurons
    keras.layers.Dense(1, activation="sigmoid")   # Output: sigmoid for probability
])

nn.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

# Train with 10% validation split to monitor overfitting
history = nn.fit(
    X_train_scaled, y_train,
    validation_split=0.1,
    epochs=20,
    batch_size=64,
    verbose=1
)

nn_proba = nn.predict(X_test_scaled).ravel()
nn_preds = (nn_proba >= 0.5).astype(int)

nn_acc = accuracy_score(y_test, nn_preds)
nn_auc = roc_auc_score(y_test, nn_proba)
print(f"\n  ✓ NN Accuracy:   {nn_acc:.4f}")
print(f"  ✓ NN ROC-AUC:    {nn_auc:.4f}")
print()


# ═══════════════════════════════════════════════════════════════
#  10. NEURAL NETWORK TRAINING HISTORY
#      - Loss curve: should decrease and converge
#      - AUC curve: should increase and stabilize
#      - Gap between train/val indicates overfitting
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("  📈 STEP 9: NN Training History Visualization")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
axes[0].plot(history.history["loss"], label="Train Loss", color="#6366f1", linewidth=2)
axes[0].plot(history.history["val_loss"], label="Val Loss", color="#ec4899", linewidth=2)
axes[0].set_title("Neural Network — Loss Curve", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Binary Cross-Entropy Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

# AUC curve
axes[1].plot(history.history["auc"], label="Train AUC", color="#6366f1", linewidth=2)
axes[1].plot(history.history["val_auc"], label="Val AUC", color="#ec4899", linewidth=2)
axes[1].set_title("Neural Network — AUC Curve", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Area Under ROC Curve")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
print("  ✓ Training history plots displayed\n")


# ═══════════════════════════════════════════════════════════════
#  11. ROC CURVE COMPARISON
#      - Plots True Positive Rate vs False Positive Rate
#      - The closer the curve to the top-left corner, the better
#      - The diagonal dashed line represents random guessing (AUC=0.5)
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("  📉 STEP 10: ROC Curve Comparison")
print("=" * 60)

fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_proba)
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_proba)
fpr_nn, tpr_nn, _ = roc_curve(y_test, nn_proba)

plt.figure(figsize=(8, 7))
plt.plot(fpr_knn, tpr_knn, label=f"KNN (AUC={knn_auc:.3f})", color="#6366f1", linewidth=2)
plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC={svm_auc:.3f})", color="#8b5cf6", linewidth=2)
plt.plot(fpr_nn, tpr_nn, label=f"Neural Network (AUC={nn_auc:.3f})", color="#ec4899", linewidth=2)
plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Random Guess")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve Comparison — All Models", fontsize=16, fontweight="bold")
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
print("  ✓ ROC curves displayed\n")


# ═══════════════════════════════════════════════════════════════
#  12. BEST MODEL SELECTION & CONFUSION MATRIX
#      - Select the model with the highest ROC-AUC score
#      - Display confusion matrix: TP, TN, FP, FN
#      - FP = False alarm (predicted churn but stayed)
#      - FN = Missed churn (predicted retain but left) — most costly!
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("  🏆 STEP 11: Best Model & Confusion Matrix")
print("=" * 60)

roc_scores = {"KNN": knn_auc, "SVM": svm_auc, "Neural Network": nn_auc}
best_model_name = max(roc_scores, key=roc_scores.get)
pred_map = {"KNN": knn_preds, "SVM": svm_preds, "Neural Network": nn_preds}
best_preds = pred_map[best_model_name]

print(f"  🏆 Best Model: {best_model_name} (AUC = {roc_scores[best_model_name]:.4f})")
print()

cm = confusion_matrix(y_test, best_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu",
            xticklabels=["Retained", "Churned"],
            yticklabels=["Retained", "Churned"])
plt.title(f"Confusion Matrix — {best_model_name}", fontsize=14, fontweight="bold")
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════
#  13. FINAL RESULTS SUMMARY
# ═══════════════════════════════════════════════════════════════

print()
print("═" * 60)
print("  📊 FINAL RESULTS SUMMARY")
print("═" * 60)
print(f"  {'Model':<20} {'Accuracy':>10} {'ROC-AUC':>10}")
print(f"  {'─'*20} {'─'*10} {'─'*10}")
print(f"  {'KNN':<20} {knn_acc:>10.4f} {knn_auc:>10.4f}")
print(f"  {'SVM':<20} {svm_acc:>10.4f} {svm_auc:>10.4f}")
print(f"  {'Neural Network':<20} {nn_acc:>10.4f} {nn_auc:>10.4f}")
print(f"  {'─'*20} {'─'*10} {'─'*10}")
print(f"  🏆 Best: {best_model_name} (AUC = {roc_scores[best_model_name]:.4f})")
print("═" * 60)
