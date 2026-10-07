# CreditGuard — Project Plan
## Credit Card Fraud Detection using SVM

---

## 1. Problem Definition

Credit card fraud causes significant financial losses. The goal is to build a **binary classifier** using **Support Vector Machine (SVM)** that can distinguish fraudulent transactions from legitimate ones, given anonymized transaction features.

---

## 2. Objectives

1. Load and explore the ULB Credit Card Fraud dataset
2. Preprocess data (scaling, handling class imbalance)
3. Train SVM classifiers (Linear kernel & RBF kernel)
4. Evaluate models using appropriate metrics (not accuracy alone)
5. Compare kernel performance
6. Demonstrate results via a Streamlit UI

---

## 3. Dataset

**Source:** [ULB Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
**File:** `data/creditcard.csv`

| Property | Detail |
|---|---|
| Rows | 284,807 transactions |
| Features | V1–V28 (PCA-transformed), Time, Amount |
| Target | `Class` → 0 = Legitimate, 1 = Fraud |
| Fraud count | 492 (0.17%) |
| Imbalance ratio | ~577:1 |

> [!IMPORTANT]
> The dataset is **extremely imbalanced**. Accuracy is meaningless here — a model predicting "not fraud" for everything gets 99.83% accuracy. We must use Precision, Recall, F1-Score, and AUC.

---

## 4. Folder Structure

```
mlMiniProj/
├── data/
│   └── creditcard.csv          ← Dataset (user downloads from Kaggle)
├── models/
│   └── (saved models appear here after training)
├── train.py                    ← ML pipeline: preprocess → train → evaluate → save
├── app.py                      ← Streamlit UI for demonstration
├── requirements.txt            ← Python dependencies
└── README.md                   ← Project overview & instructions
```

**Total: 4 code/config files + 1 data file.** That's it.

---

## 5. What Each File Does

### `train.py` — The ML Pipeline (core of the project)

This is the file you explain to your professor. It runs as a plain Python script:

```
python train.py
```

**Step-by-step flow:**

| Step | What it does | Key concept |
|---|---|---|
| 1. Load data | `pd.read_csv("data/creditcard.csv")` | Data loading |
| 2. Explore | Print shape, class distribution, basic stats | EDA |
| 3. Feature selection | Drop `Time`, keep V1–V28 + `Amount` | Feature engineering |
| 4. Scale | `StandardScaler` on `Amount` (V1–V28 already scaled by PCA) | Normalization |
| 5. Split | `train_test_split` (80/20, stratified) | Data splitting |
| 6. Handle imbalance | `SMOTE` on **training set only** | Class balancing |
| 7. Train Linear SVM | `SVC(kernel='linear', ...)` | SVM with linear kernel |
| 8. Train RBF SVM | `SVC(kernel='rbf', ...)` | SVM with RBF kernel |
| 9. Evaluate | Confusion matrix, classification report, ROC-AUC, PR-AUC | Model evaluation |
| 10. Save | `joblib.dump(model, ...)` | Model persistence |

> [!IMPORTANT]
> **Data leakage prevention:** SMOTE is applied **only after** train-test split, **only on training data**. The test set is never touched during training. Scaling is fit on train, transformed on test.

### `app.py` — Streamlit Demo UI

Loads the saved models and test data, displays results interactively. **Pages:**

| Page | Content |
|---|---|
| **Home** | Project title, problem statement, dataset overview |
| **Data Exploration** | Class distribution chart, sample rows, feature statistics |
| **Model Training** | Button to trigger training, progress display |
| **Results** | Confusion matrices, classification reports, ROC curves, PR curves |
| **Prediction** | Input a transaction → get fraud/not-fraud prediction |

### `requirements.txt`

```
streamlit
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
plotly
joblib
```

### `README.md`

Setup instructions, how to run, project description for the professor.

---

## 6. ML Workflow (Visual)

```
creditcard.csv
     │
     ▼
┌─────────────┐
│  Load Data  │
└─────┬───────┘
      ▼
┌─────────────────┐
│  Drop 'Time'    │
│  Scale 'Amount' │
└─────┬───────────┘
      ▼
┌──────────────────────────┐
│  Train/Test Split (80/20)│
│  stratify=Class          │
└─────┬────────────────────┘
      │
      ├──► Test Set (untouched, used only for evaluation)
      │
      ▼
┌──────────────────┐
│  SMOTE (train    │
│  set only)       │
└─────┬────────────┘
      ▼
┌──────────────────────────┐
│  Train SVM Models        │
│  ├─ Linear SVM           │
│  └─ RBF SVM              │
└─────┬────────────────────┘
      ▼
┌──────────────────────────┐
│  Evaluate on Test Set    │
│  ├─ Confusion Matrix     │
│  ├─ Precision / Recall   │
│  ├─ F1-Score             │
│  ├─ ROC Curve + AUC      │
│  └─ PR Curve + AUC       │
└─────┬────────────────────┘
      ▼
┌──────────────────┐
│  Save Models     │
│  (joblib)        │
└──────────────────┘
```

---

## 7. Evaluation Metrics (and why each matters)

| Metric | Why it matters for fraud detection |
|---|---|
| **Confusion Matrix** | Shows exact counts of TP, FP, TN, FN |
| **Precision** | Of all transactions flagged as fraud, how many actually were? (reduces false alarms) |
| **Recall** | Of all actual frauds, how many did we catch? (most critical) |
| **F1-Score** | Harmonic mean of precision & recall |
| **ROC-AUC** | Overall discriminative ability across all thresholds |
| **PR-AUC** | Better than ROC-AUC for imbalanced datasets (recommended by dataset authors) |

> [!NOTE]
> We do **not** report accuracy as the primary metric. A professor will ask why — the answer is the 99.83% class imbalance makes accuracy meaningless.

---

## 8. Training Details

| Parameter | Value | Reason |
|---|---|---|
| Test size | 20% | Standard split |
| Stratify | `y` (Class column) | Preserves fraud ratio in both sets |
| Random state | 42 | Reproducibility |
| SMOTE | Applied to train only | Prevents data leakage |
| SVM `probability` | `True` | Needed for ROC/PR curves |
| SVM `class_weight` | `'balanced'` | Additional imbalance handling |
| Linear SVM `C` | 1.0 | Default regularization |
| RBF SVM `C` | 1.0 | Default regularization |
| RBF SVM `gamma` | `'scale'` | Default, works well with scaled data |
| Sampling strategy | Use a subsample (~50k rows) for training speed | Full dataset SVM is very slow |

> [!WARNING]
> **SVM on 284k rows is extremely slow** (O(n²) to O(n³) complexity). We will **subsample** the dataset while preserving all fraud cases. This is standard practice and easy to explain in a viva.

---

## 9. Potential Failure Points & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Dataset not found | App crashes | Check file exists, show clear error message |
| SVM training too slow | Demo takes forever | Subsample dataset (keep all frauds + random sample of legit) |
| SMOTE before split | Data leakage, inflated metrics | SMOTE strictly after split, on train only |
| `probability=True` slow | RBF SVM training time increases | Subsample + accept the tradeoff (needed for curves) |
| Model file missing | Streamlit can't show results | Train button in app, or check & prompt user |
| Memory issues | Crash on low-RAM machines | Subsample handles this too |

---

## 10. Viva-Ready Talking Points

Things a professor will likely ask:

1. **"Why SVM?"** → Effective for binary classification, works well in high-dimensional spaces (28 PCA features), strong theoretical foundation with maximum margin.

2. **"Why not just use accuracy?"** → Dataset is 99.83% non-fraud. A dummy model gets 99.83% accuracy. Recall and F1 are more meaningful.

3. **"What is SMOTE?"** → Synthetic Minority Oversampling Technique. Creates synthetic fraud samples by interpolating between existing fraud data points in feature space.

4. **"What is data leakage?"** → When test data information leaks into training. We prevent it by splitting first, then applying SMOTE only to training data.

5. **"Linear vs RBF kernel?"** → Linear finds a straight decision boundary. RBF can find non-linear boundaries. We compare both to see which works better.

6. **"What are V1–V28?"** → PCA-transformed features. Original features are hidden for privacy. PCA already scaled them, which is why we only need to scale `Amount`.

---

## Next Steps

Once you approve this plan, I will implement:
1. `train.py` — complete ML pipeline
2. `app.py` — Streamlit demo
3. `requirements.txt` — dependencies
4. `README.md` — documentation
