"""
╔══════════════════════════════════════════════════════════════╗
║                    DATA LOADER MODULE                        ║
║  Handles dataset loading, preprocessing, and feature         ║
║  engineering for the Telco Customer Churn pipeline.           ║
╚══════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Logger Setup ────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  LOAD & PREPROCESS
# ═══════════════════════════════════════════════════════════════

def load_data(
    uploaded_file,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load and preprocess the Telco Customer Churn dataset.

    Preprocessing steps:
        1. Drop ``customerID`` column (non-predictive identifier).
        2. Convert ``TotalCharges`` to numeric, filling blanks with median.
        3. Fill missing categorical values with ``'Missing'``.
        4. Map ``Churn`` column: ``{'Yes': 1, 'No': 0}``.

    Parameters
    ----------
    uploaded_file : UploadedFile | str
        A Streamlit ``UploadedFile`` object or a file path string.

    Returns
    -------
    tuple[pd.DataFrame | None, str | None]
        A tuple of ``(dataframe, error_message)``.
        On success, ``error_message`` is ``None``.
        On failure, ``dataframe`` is ``None``.

    Examples
    --------
    >>> df, err = load_data("Telco-Customer-Churn.csv")
    >>> if err:
    ...     print(f"Error: {err}")
    """
    if uploaded_file is None:
        return None, "No file uploaded."

    try:
        df = pd.read_csv(uploaded_file)
        logger.info("Dataset loaded — shape %s", df.shape)
    except Exception as exc:
        logger.error("Failed to read CSV: %s", exc)
        return None, str(exc)

    # ── Step 1: Drop non-predictive identifier ──────────────
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)
        logger.info("Dropped 'customerID' column.")

    # ── Step 2: Convert TotalCharges to numeric ─────────────
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        median_val = df["TotalCharges"].median()
        df["TotalCharges"].fillna(median_val, inplace=True)
        logger.info("TotalCharges converted — missing filled with median (%.2f).", median_val)

    # ── Step 3: Fill missing categorical values ─────────────
    cat_cols: List[str] = list(df.select_dtypes(include="object").columns)
    for col in cat_cols:
        if df[col].isna().sum() > 0:
            df[col].fillna("Missing", inplace=True)
            logger.info("Filled NaN in '%s' with 'Missing'.", col)

    # ── Step 4: Map target variable ─────────────────────────
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
        logger.info("Churn mapped → {Yes: 1, No: 0}.")

    return df, None


# ═══════════════════════════════════════════════════════════════
#  PREPARE FEATURES & TARGET
# ═══════════════════════════════════════════════════════════════

def prepare_data(
    df: pd.DataFrame,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[str]]:
    """
    Separate the DataFrame into feature matrix ``X`` and target vector ``y``.

    One-hot encodes all categorical features using ``pd.get_dummies``
    with ``drop_first=True`` to avoid multicollinearity.

    Parameters
    ----------
    df : pd.DataFrame
        The preprocessed DataFrame (output of :func:`load_data`).

    Returns
    -------
    tuple[pd.DataFrame | None, pd.Series | None, str | None]
        ``(X, y, error_message)`` — ``error_message`` is ``None`` on success.
    """
    if "Churn" not in df.columns:
        logger.error("Target column 'Churn' not found in DataFrame.")
        return None, None, "Churn column not found in dataset."

    X = pd.get_dummies(df.drop(columns=["Churn"]), drop_first=True)
    y = df["Churn"]

    logger.info("Feature matrix shape: %s | Target shape: %s", X.shape, y.shape)
    return X, y, None


# ═══════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING (SCALING)
# ═══════════════════════════════════════════════════════════════

def feature_engineering(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardize features by removing the mean and scaling to unit variance.

    Uses ``StandardScaler`` fitted **only** on the training set to prevent
    data leakage.  The same transformation is then applied to the test set.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    X_test : np.ndarray
        Testing feature matrix.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, StandardScaler]
        ``(X_train_scaled, X_test_scaled, fitted_scaler)``
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info("Features scaled — mean ≈ 0, std ≈ 1.")
    return X_train_scaled, X_test_scaled, scaler
