# CreditGuard — Credit Card Fraud Detection using SVM

A machine learning project evaluating Support Vector Machine (SVM) classifiers (Linear vs. RBF kernels) for detecting credit card fraud, equipped with an interactive Streamlit web dashboard.

---

## 📌 Project Overview
Credit card fraud detection is a critical financial security challenge characterized by high-volume transactions and subtle fraudulent patterns. This project implements and benchmarks:
- **Linear Support Vector Classifier (`LinearSVC`)**
- **Non-Linear Radial Basis Function Support Vector Classifier (`SVC(kernel='rbf')`)**
- **Preprocessing & Scaling:** Standard scaling on PCA-transformed features and Transaction Amount.
- **Evaluation Metrics:** Precision, Recall, F1-Score, ROC-AUC, and PR-AUC.
- **Interactive Web Interface:** Streamlit app featuring live transaction inference simulator, exploratory data analysis, and head-to-head model comparison.

---

## 📊 Datasets & Source Links

This project supports and evaluates two standard credit card fraud datasets:

| Dataset | Records | Balance | Kaggle Source Link |
|---|---|---|---|
| **Credit Card Fraud 2023** (Primary) | 568,630 transactions | 50% Legitimate / 50% Fraud | [nelgiriyewithana/credit-card-fraud-detection-dataset-2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023) |
| **Credit Card Fraud Detection (ULB)** (Classic) | 284,807 transactions | 0.17% Fraud (Highly imbalanced) | [mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

> **Note on Data Directory:**
> Place the downloaded CSV files inside the `data/` folder:
> - `data/creditcard_2023.csv`
> - `data/creditcard.csv`

---

## 🗂️ Repository Structure

```
credit-card-fraud-detection/
├── data/                       # Dataset files (tracked with Git LFS)
│   ├── creditcard.csv
│   └── creditcard_2023.csv
├── models/                     # Saved models and pre-computed results
│   ├── linear_svm.joblib       # Trained Linear SVM model
│   ├── rbf_svm.joblib          # Trained RBF SVM model
│   ├── scaler.joblib           # Fitted StandardScaler
│   ├── results.joblib          # Cached evaluation metrics & ROC curves
│   └── preprocessed_data.joblib# Scaled train/test feature arrays
├── app.py                      # Interactive Streamlit web application
├── train.py                    # Complete ML training & evaluation pipeline
├── requirements.txt            # Python dependencies
├── project_plan.md             # Project architecture specification
├── PROJECT_MASTER_GUIDE.md     # In-depth viva & technical documentation
└── README.md                   # Repository documentation
```

---

## 🚀 How to Run Locally

Follow these step-by-step instructions to get the project up and running on your local machine.

### Step 1: Clone the Repository
Open your terminal and clone the repository:
```bash
git clone https://github.com/abhi-p06/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### Step 2: Pull Large Files via Git LFS
This repository stores large dataset CSVs and model files using **Git LFS** (Large File Storage). 

If you have Git LFS installed, pull the files:
```bash
# Install Git LFS hooks (first time only)
git lfs install

# Pull the actual dataset and model files
git lfs pull
```
*(If you don't use Git LFS, you can also download the datasets directly from the Kaggle links above and place them in the `data/` folder).*

### Step 3: Create & Activate a Virtual Environment
It is recommended to use Python 3.9, 3.10, or 3.11:

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows (Command Prompt / PowerShell):**
```bash
python -m venv venv
# PowerShell:
.\venv\Scripts\Activate.ps1
# Command Prompt:
.\venv\Scripts\activate.bat
```

### Step 4: Install Dependencies
Install all required libraries via pip:
```bash
pip install -r requirements.txt
```

### Step 5: (Optional) Train or Retrain the Models
Pre-trained model artifacts are included in `models/`. If you want to train the models from scratch:
```bash
python train.py
```
This script will:
- Load the dataset from `data/`
- Standardize features with `StandardScaler`
- Train both Linear SVM and RBF SVM
- Calculate performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Save models and metrics to `models/`

### Step 6: Launch the Streamlit Web Application
Run the Streamlit application:
```bash
streamlit run app.py
```

Your browser will automatically open to `http://localhost:8501`. If it doesn't open automatically, navigate to `http://localhost:8501` in your browser.

---

## 🖥️ Streamlit App Features

- 📑 **Project Overview:** Interactive introduction, dataset summaries, and technical SVM methodology.
- 📈 **Exploratory Data Analysis (EDA):** Interactive transaction amounts, class distributions, and feature correlation heatmaps.
- ⚖️ **Model Comparison:** Side-by-side performance comparison of Linear SVM vs. RBF SVM.
- 🎯 **Performance Metrics:** Confusion matrices, classification reports, and ROC/PR curves.
- 🧪 **Live Fraud Predictor:** Interactive transaction simulator allowing you to adjust feature values and test real-time fraud probability scores.

---

## 📊 Key Results Summary

| Metric | Linear SVM | RBF SVM |
|---|---|---|
| **Accuracy** | ~99.9% | ~99.9% |
| **Precision** | High | High |
| **Recall** | ~80%+ | ~82%+ |
| **F1-Score** | Robust | Optimal |

---

## 🛠️ Troubleshooting

- **"File not found in `models/`" or Git LFS pointer issue:**
  If you see an error loading models, verify that the actual model files were pulled by running `git lfs pull`.
- **Streamlit Port In Use:**
  If port 8501 is busy, specify an alternate port:
  ```bash
  streamlit run app.py --server.port 8502
  ```

---

## 📄 License
This project is open-source and intended for academic, research, and educational demonstration purposes.
