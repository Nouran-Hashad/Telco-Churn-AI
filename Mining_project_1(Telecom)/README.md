# ⚡ Telco Customer Churn Prediction AI

> **Advanced Data Mining Project** — Predict telecom customer churn using Machine Learning and Deep Neural Networks.

---

## 🚀 Overview

This project implements a complete **end-to-end machine learning pipeline** for predicting customer churn in the telecommunications industry. It includes:

- **Interactive Data Analysis** — Explore customer demographics, services, and churn patterns
- **Multi-Model Training** — Train and compare KNN, SVM, Random Forest, Gradient Boosting, and Neural Networks
- **Real-Time Predictions** — Enter customer details and get instant churn probability with retention recommendations

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.10+** | Core programming language |
| **Streamlit** | Interactive web application framework |
| **Scikit-learn** | Classical ML algorithms (KNN, SVM, RF, GB) |
| **TensorFlow / Keras** | Deep Neural Network implementation |
| **Plotly** | Interactive data visualizations |
| **Pandas / NumPy** | Data manipulation and computation |
| **Matplotlib / Seaborn** | Static visualizations (standalone script) |

## 📁 Project Structure

```
Mining_project_1(Telecom)/
├── app.py                        # 🏠 Home page — Hero section & dataset upload
├── customer_churn_project.py     # 📜 Standalone ML pipeline script
├── style.css                     # 🎨 Premium glassmorphism dark theme
├── requirements.txt              # 📦 Python dependencies
├── Telco-Customer-Churn.csv.csv  # 📊 Dataset (IBM Telco)
├── pages/
│   ├── 1_📊_Data_Analysis.py     # 📊 Interactive EDA with Plotly
│   ├── 2_🧠_Model_Training.py    # 🧠 Model training & comparison
│   └── 3_🔮_Prediction_System.py # 🔮 Real-time churn prediction
├── utils/
│   ├── __init__.py               # 📦 Package initialization
│   ├── data_loader.py            # 📂 Data loading & preprocessing
│   └── model_utils.py            # 🤖 Model training & evaluation
└── README.md                     # 📖 This file
```

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Web Application

```bash
streamlit run app.py
```

### 3. Run the Standalone Script

```bash
python customer_churn_project.py
```

## 📊 Models Implemented

| Model | Type | Key Strength |
|-------|------|-------------|
| **KNN** | Instance-based | Simple, no training phase |
| **SVM** | Margin-based | Excellent with high-dimensional data |
| **Random Forest** | Ensemble (Bagging) | Robust, handles non-linearity |
| **Gradient Boosting** | Ensemble (Boosting) | State-of-the-art accuracy |
| **Neural Network** | Deep Learning | Learns complex patterns |

## 📈 Features

- ✅ **Glassmorphism UI** — Premium dark theme with animated gradients
- ✅ **5 ML Models** — KNN, SVM, Random Forest, Gradient Boosting, DNN
- ✅ **Model Comparison** — Train all models at once with radar chart
- ✅ **Interactive EDA** — Histograms, box plots, scatter plots, heatmaps
- ✅ **Real-Time Prediction** — Gauge chart with risk assessment
- ✅ **Retention Recommendations** — AI-powered retention strategies
- ✅ **Professional Code** — Type hints, docstrings, logging

## 📂 Dataset

**IBM Telco Customer Churn** — 7,043 customers with 21 features including:
- Demographics (gender, age, partner, dependents)
- Services (phone, internet, streaming, security)
- Account info (tenure, contract, billing, charges)
- Target: **Churn** (Yes / No)

---

<p align="center">
  Built with ❤️ using Python · Streamlit · Scikit-learn · TensorFlow
</p>
