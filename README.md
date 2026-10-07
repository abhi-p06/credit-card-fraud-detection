# CreditGuard — Credit Card Fraud Detection using SVM

A machine learning project evaluating Support Vector Machine (SVM) classifiers (Linear vs. RBF kernels) for detecting credit card fraud, equipped with an interactive Streamlit web dashboard.

---

## 📌 Project Overview
Credit card fraud detection is a critical financial security challenge characterized by extreme class imbalance. This project implements and benchmarks:
- **Linear Support Vector Classifier (`LinearSVC`)**
- **Non-Linear Radial Basis Function Support Vector Classifier (`SVC(kernel='rbf')`)**
- **Preprocessing & Scaling:** Standard scaling on PCA features and Transaction Amount.
- **Evaluation Metrics:** Precision, Recall, F1-Score, ROC-AUC, and PR-AUC.
- **Interactive Web Interface:** Streamlit app with interactive live transaction testing, exploratory data analysis, and model comparison.

---

## 🗂️ Repository Structure

```
.
├── data/                       # Dataset files (managed with Git LFS)
│   ├── creditcard.csv
│   └── creditcard_2023.csv
├── models/                     # Serialized models and evaluation artifacts
│   ├── linear_svm.joblib
│   ├── rbf_svm.joblib
│   ├── scaler.joblib
│   ├── results.joblib
│   └── preprocessed_data.joblib
├── app.py                      # Interactive Streamlit application
├── train.py                    # Complete ML training & evaluation pipeline
├── requirements.txt            # Python dependencies
├── project_plan.md             # Project architecture plan
├── PROJECT_MASTER_GUIDE.md     # In-depth viva & technical documentation
└── README.md                   # Repository overview
```

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup
Clone the repository (make sure Git LFS is installed for dataset & model files):

```bash
git clone https://github.com/abhi-p06/credit-card-fraud-detection.git
cd credit-card-fraud-detection
git lfs pull
```

Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

### 2. Training the Models
To run the full preprocessing, training, and evaluation pipeline:
```bash
python train.py
```
This script will:
1. Load and clean the dataset.
2. Standardize features using `StandardScaler`.
3. Train the Linear SVM and RBF SVM models.
4. Compute classification metrics, confusion matrices, and ROC curves.
5. Export trained models and metrics to the `models/` directory.

---

### 3. Running the Streamlit Web Application
Launch the interactive dashboard:
```bash
streamlit run app.py
```

The application provides:
- **Project Overview:** High-level problem statement and ML methodology.
- **Exploratory Data Analysis (EDA):** Class distribution and feature correlation plots.
- **Model Comparison:** Head-to-head metrics (Accuracy, Precision, Recall, F1, ROC-AUC).
- **Confusion Matrix & ROC Curves:** Visual classification performance analysis.
- **Live Fraud Predictor:** Interactive transaction testing simulator.

---

## 📊 Key Results

| Metric | Linear SVM | RBF SVM |
|---|---|---|
| **Accuracy** | ~99.9% | ~99.9% |
| **Precision** | High | High |
| **Recall** | ~80%+ | ~82%+ |
| **F1-Score** | Robust | Optimal |

---

## 📄 License
This project is open-source and intended for academic and demonstration purposes.
