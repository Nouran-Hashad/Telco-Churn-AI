"""
╔══════════════════════════════════════════════════════════════╗
║                   MODEL UTILITIES MODULE                     ║
║  Provides training, evaluation, and comparison functions     ║
║  for multiple ML classifiers used in churn prediction.       ║
╚══════════════════════════════════════════════════════════════╝

Supported Models:
    • K-Nearest Neighbors  (KNN)
    • Support Vector Machine  (SVM)
    • Random Forest  (RF)
    • Gradient Boosting  (GB)
    • Deep Neural Network  (DNN — Keras/TensorFlow)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)


# ── Logger Setup ────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  MODEL TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def train_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int = 5,
) -> KNeighborsClassifier:
    """
    Train a K-Nearest Neighbors classifier.

    KNN classifies a sample based on the majority vote of its ``k``
    nearest neighbors in the feature space, using Euclidean distance
    by default.

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training features.
    y_train : np.ndarray
        Training labels (0 = No Churn, 1 = Churn).
    n_neighbors : int, default=5
        Number of neighbors to consider for majority voting.

    Returns
    -------
    KNeighborsClassifier
        The fitted KNN model.
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    logger.info("KNN trained — k=%d", n_neighbors)
    return knn


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel: str = "rbf",
    C: float = 1.0,
) -> SVC:
    """
    Train a Support Vector Machine classifier.

    SVM finds the optimal hyperplane that maximizes the margin between
    classes.  The RBF kernel maps data to a higher-dimensional space
    to handle non-linearly separable patterns.

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training features.
    y_train : np.ndarray
        Training labels.
    kernel : str, default='rbf'
        Kernel type: ``'linear'``, ``'rbf'``, ``'poly'``, or ``'sigmoid'``.
    C : float, default=1.0
        Regularization parameter — higher values reduce misclassification
        at the risk of overfitting.

    Returns
    -------
    SVC
        The fitted SVM model with probability estimation enabled.
    """
    svm = SVC(kernel=kernel, C=C, probability=True, random_state=42)
    svm.fit(X_train, y_train)
    logger.info("SVM trained — kernel=%s, C=%.2f", kernel, C)
    return svm


def train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Random Forest is an ensemble of Decision Trees trained on random
    subsets of data (bagging).  It reduces variance and overfitting
    compared to a single Decision Tree.

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training features.
    y_train : np.ndarray
        Training labels.
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int or None, default=None
        Maximum depth of each tree.  ``None`` means nodes expand until
        all leaves are pure or contain fewer than ``min_samples_split``.

    Returns
    -------
    RandomForestClassifier
        The fitted Random Forest model.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    logger.info("Random Forest trained — trees=%d, max_depth=%s", n_estimators, max_depth)
    return rf


def train_gb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float = 0.1,
    n_estimators: int = 100,
) -> GradientBoostingClassifier:
    """
    Train a Gradient Boosting classifier.

    Gradient Boosting builds trees sequentially, where each new tree
    corrects the errors of the previous ensemble.  It typically achieves
    higher accuracy than Random Forest but requires careful tuning.

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training features.
    y_train : np.ndarray
        Training labels.
    learning_rate : float, default=0.1
        Shrinkage factor — lower values need more estimators but
        generalize better.
    n_estimators : int, default=100
        Number of boosting stages.

    Returns
    -------
    GradientBoostingClassifier
        The fitted Gradient Boosting model.
    """
    gb = GradientBoostingClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=42,
    )
    gb.fit(X_train, y_train)
    logger.info("Gradient Boosting trained — lr=%.3f, estimators=%d", learning_rate, n_estimators)
    return gb


def train_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 20,
    batch_size: int = 64,
) -> Tuple[keras.Sequential, Any]:
    """
    Train a Deep Neural Network (DNN) for binary classification.

    Architecture:
        Input → Dense(128, ReLU) → Dropout(0.3) →
        Dense(64, ReLU) → Dropout(0.2) →
        Dense(32, ReLU) → Dense(1, Sigmoid)

    Uses Adam optimizer with binary cross-entropy loss and tracks
    accuracy + AUC during training.

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training features.
    y_train : np.ndarray
        Training labels.
    epochs : int, default=20
        Number of training epochs.
    batch_size : int, default=64
        Mini-batch size for gradient descent.

    Returns
    -------
    tuple[keras.Sequential, keras.callbacks.History]
        ``(model, training_history)``
    """
    tf.random.set_seed(42)

    nn = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])

    nn.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    history = nn.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    logger.info("Neural Network trained — epochs=%d, batch=%d", epochs, batch_size)
    return nn, history


# ═══════════════════════════════════════════════════════════════
#  MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str = "sklearn",
) -> Dict[str, Any]:
    """
    Evaluate a trained model and return a comprehensive metrics dictionary.

    Parameters
    ----------
    model : Any
        A trained scikit-learn estimator or Keras model.
    X_test : np.ndarray
        Scaled test features.
    y_test : np.ndarray
        True test labels.
    model_type : str, default='sklearn'
        Either ``'sklearn'`` or ``'keras'`` to determine prediction method.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - ``accuracy`` (float): Overall accuracy
        - ``precision`` (float): Precision for churn class
        - ``recall`` (float): Recall for churn class
        - ``f1`` (float): F1-score for churn class
        - ``auc`` (float): Area Under ROC Curve
        - ``cm`` (np.ndarray): Confusion matrix
        - ``fpr`` (np.ndarray): False positive rates for ROC
        - ``tpr`` (np.ndarray): True positive rates for ROC
        - ``predictions`` (np.ndarray): Predicted labels
        - ``probabilities`` (np.ndarray): Predicted probabilities
    """
    if model_type == "keras":
        y_proba = model.predict(X_test, verbose=0).ravel()
        y_pred = (y_proba >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    logger.info(
        "Evaluation → Acc=%.4f | AUC=%.4f | F1=%.4f",
        acc, auc, f1,
    )

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
        "predictions": y_pred,
        "probabilities": y_proba,
    }


# ═══════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE HELPER
# ═══════════════════════════════════════════════════════════════

def get_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 15,
) -> Optional[Dict[str, float]]:
    """
    Extract feature importances from tree-based models.

    Parameters
    ----------
    model : Any
        A fitted tree-based model (Random Forest or Gradient Boosting).
    feature_names : list[str]
        List of feature names matching the training columns.
    top_n : int, default=15
        Number of top features to return.

    Returns
    -------
    dict[str, float] or None
        Sorted dictionary of ``{feature_name: importance}`` or ``None``
        if the model doesn't support ``feature_importances_``.
    """
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model does not support feature_importances_.")
        return None

    importances = model.feature_importances_
    feat_imp = dict(zip(feature_names, importances))
    sorted_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:top_n])

    logger.info("Top %d features extracted.", top_n)
    return sorted_imp
