# CREDIT CARD FRAUD DETECTION USING SUPPORT VECTOR MACHINES (SVM)
## Comprehensive Master Laboratory Project Guide & Viva Compendium

---

## Executive Summary & Metadata
- **Project Title:** Credit Card Fraud Detection using Support Vector Machine (SVM) Classification
- **Domain:** Supervised Machine Learning / Financial Security
- **Academic Context:** B.Tech Machine Learning Laboratory Mini Project
- **Primary Algorithm:** Support Vector Classifier (`sklearn.svm.SVC`)
- **Kernels Evaluated:** Linear Kernel (`kernel='linear'`) & Radial Basis Function Kernel (`kernel='rbf'`)
- **Feature Preprocessing:** `sklearn.preprocessing.StandardScaler` (Fitted strictly on training data)
- **Active Dataset:** Credit Card Fraud Detection Dataset 2023 (Kaggle: `nelgiriyewithana/credit-card-fraud-detection-dataset-2023`)
- **Total Transactions:** 568,630 records
- **Class Balance:** Inherent 50.0% Legitimate (`Class 0`: 284,315) vs 50.0% Fraud (`Class 1`: 284,315)
- **Train/Test Partition:** 80% Training (454,904 rows) / 20% Testing (113,726 rows), Stratified (`random_state=42`)
- **Interactive UI:** Streamlit Web Application (`app.py`, 5 dedicated pages)

---

# SECTION 1 — PROJECT OVERVIEW

### 1.1 Project Title
**Credit Card Fraud Detection using SVM Classification with Data Preprocessing**

### 1.2 Problem Being Solved
When credit card transactions occur, financial institutions must distinguish between **legitimate customer purchases** and **unauthorized fraudulent operations**. In modern automated economies, billions of card transactions occur daily. Criminals use stolen credentials, skimming devices, identity theft, and online data breaches to execute fraudulent charges. 

Manual inspection of every purchase is mathematically and operationally impossible. If an institution delays transaction clearance to inspect records manually, legitimate customer checkouts freeze, disrupting commerce. Conversely, if no automated defense exists, financial losses, chargebacks, and consumer distress skyrocket.

This project solves that challenge by constructing an automated, data-driven machine learning classifier that examines numerical attributes of each transaction and instantly categorizes it as either **Legitimate** or **Fraudulent**.

### 1.3 Why Credit Card Fraud Detection is a Machine Learning Problem
Fraud detection is fundamentally an inductive pattern recognition problem:
1. **High Dimensionality:** Transactions are described by many numerical variables simultaneously (28 principal components capturing transaction latent features, plus monetary amount). Humans cannot visualize relationships in 29-dimensional space.
2. **Subtle Non-Linear Boundaries:** Fraudulent patterns often masquerade as normal activity, differing only through subtle combinations of mathematical variables that define an anomalous manifold.
3. **Speed Requirements:** Decisions must be calculated within milliseconds using algebraic dot products and kernel transformations.
4. **Generalization:** Rule-based systems (e.g., "if amount > $5000 flag fraud") fail because fraudsters easily circumvent static thresholds. Machine learning models generalize from historical patterns to detect novel attacks.

### 1.4 What the Application Does
The application consists of two primary components:
1. **Offline Training & Evaluation Engine (`train.py`):** Loads the raw 2023 dataset, verifies integrity, performs stratified train-test splitting, standardizes features without data leakage, trains Linear and RBF Support Vector Machines, dynamically calculates all evaluation metrics on untouched test data, and serializes trained models to disk.
2. **Interactive Demonstration Interface (`app.py`):** A professional Streamlit web application providing a classroom demonstration interface. It allows professors and evaluators to select real test-set transactions (legitimate, fraud, or random), inspect transaction details, run real-time inference through the trained SVM models, observe the calculated decision scores, verify whether predictions match ground truth, compare model metrics, and explore dataset features.

### 1.5 System Inputs and Outputs
- **Input to the System:** A 29-element numerical feature vector corresponding to an individual transaction:
  $$\mathbf{x} = [V_1, V_2, V_3, \dots, V_{28}, \text{Amount}]$$
- **Output of the System:**
  1. **Discrete Class Prediction ($\hat{y}$):**
     $$\hat{y} \in \{0, 1\} \quad (0 = \text{Legitimate}, 1 = \text{Fraudulent})$$
  2. **Continuous Decision Score ($f(\mathbf{x})$):** The signed orthogonal Euclidean distance from the transaction's feature point to the SVM separating hyperplane:
     $$f(\mathbf{x}) = \mathbf{w}^T \phi(\mathbf{x}) + b$$
  3. **Verification Assessment:** Correct Classification (True Positive / True Negative) vs Misclassification (False Positive / False Negative).

### 1.6 Overall Project Workflow

```
┌────────────────────────────────────────────────────────┐
│                   1. DATASET ACQUISITION               │
│   Credit Card Fraud Detection Dataset 2023 (568,630 rows)│
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                  2. DATA PREPROCESSING                 │
│  - Verify 0 missing values & check duplicates          │
│  - Drop non-informative 'id' column                    │
│  - Separate Features (X: 29 cols) & Target (y: Class)  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                3. STRATIFIED TRAIN/TEST SPLIT          │
│  - 80% Training Set (454,904 rows: 50% legit, 50% fraud)│
│  - 20% Test Set     (113,726 rows: 50% legit, 50% fraud)│
│  - random_state = 42, stratify = y                     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                   4. FEATURE STANDARDIZATION           │
│  - StandardScaler fitted ONLY on training set (X_train)│
│  - Test set (X_test) transformed using fitted scaler   │
│  - Zero data leakage between partitions                │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                   5. SVM MODEL TRAINING                │
│  - Model 1: SVC(kernel='linear')                       │
│  - Model 2: SVC(kernel='rbf', C=1.0, gamma='scale')    │
│  - Serialized to disk in models/                       │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│               6. DYNAMIC TEST SET EVALUATION           │
│  - model.predict(X_test) & model.decision_function()   │
│  - Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC    │
│  - Confusion Matrices & Curve Coordinates saved        │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│           7. STREAMLIT DEMONSTRATION APPLICATION       │
│  - Live Transaction Analyzer with real test transactions│
│  - Random Demo Buttons (Fraud / Legit / Random)        │
│  - Real-time pipeline tracking & Decision Score visual │
│  - Model Comparison, Data Analysis & Methodology pages │
└────────────────────────────────────────────────────────┘
```

---

# SECTION 2 — PROBLEM DEFINITION

### 2.1 The Nature of Fraud Detection
Credit card fraud detection is formally defined as a **supervised binary classification problem**. Given a set of historical transactions with known outcomes, our goal is to estimate an unknown mapping function $f: \mathcal{X} \rightarrow \mathcal{Y}$ where:
- The input space $\mathcal{X} \subseteq \mathbb{R}^{29}$ represents the continuous feature space of transactions.
- The label space $\mathcal{Y} = \{0, 1\}$ represents the discrete binary states:
  - **Class 0 ($y = 0$): Legitimate Transaction** — Authorized purchase executed by the genuine account holder.
  - **Class 1 ($y = 1$): Fraudulent Transaction** — Unauthorized, deceptive charge executed by an illegitimate entity.

### 2.2 Why Manual Detection Fails
1. **Transaction Velocity:** Modern payment gateways process thousands of transactions per second. Human underwriters cannot review transactions in real time.
2. **Cognitive Limitations:** Fraudsters execute coordinated distributed attacks where individual transaction amounts appear benign (e.g., $15 to $50). The fraudulent signal is only apparent in high-dimensional PCA geometric projections.
3. **Economic Cost of Latency:** Delaying card authorization by even two seconds causes significant customer abandonment at point-of-sale terminals.

### 2.3 Why Machine Learning is the Industry Standard
Machine learning algorithms:
- Learn non-linear functional boundaries directly from empirical data without manual rule authoring.
- Operate deterministically at sub-millisecond evaluation latencies once trained.
- Enable quantitative tuning between conservative models (prioritizing high recall to catch fraud) and permissive models (prioritizing high precision to minimize customer friction).

---

# SECTION 3 — PROJECT OBJECTIVES

1. **Implement an End-to-End Machine Learning Pipeline:** Construct a clean, fully reproducible Python pipeline that handles data loading, quality auditing, preprocessing, scaling, training, evaluation, and artifact persistence.
2. **Eliminate Data Leakage:** Enforce strict experimental partitioning by fitting the `StandardScaler` strictly on training data and applying the learned parameters to untouched test data.
3. **Train and Benchmark Two Support Vector Machine Kernels:** Implement and compare:
   - **Linear SVM:** Hyperplane separation in native 29-dimensional Euclidean space.
   - **RBF SVM:** Non-linear separation using the Radial Basis Function (Gaussian) kernel.
4. **Perform Rigorous, Dynamic Model Evaluation:** Calculate all standard classification metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Confusion Matrix) strictly from dynamic calls to `model.predict()` and `model.decision_function()` on unseen test data.
5. **Build an Interactive Academic Demonstration Interface:** Create an intuitive Streamlit UI that allows examiners to interactively feed real test-set transactions into the trained models, examine decision scores, and observe the internal stages of the ML pipeline.

---

# SECTION 4 — THE DATASET

### 4.1 Dataset Overview
- **Name:** Credit Card Fraud Detection Dataset 2023
- **Source:** Kaggle (`nelgiriyewithana/credit-card-fraud-detection-dataset-2023`)
- **Total Transactions ($N$):** 568,630
- **Total Columns:** 31 (`id`, `V1` through `V28`, `Amount`, `Class`)
- **Storage Location:** `data/creditcard_2023.csv`
- **File Size on Disk:** ~154 MB

### 4.2 Column Schema

| Column Name | Data Type | Role | Description |
| :--- | :---: | :---: | :--- |
| `id` | `int64` | Identifier | Sequential integer transaction index ($0$ to $568,629$). Dropped during preprocessing. |
| `V1` to `V28` | `float64` | Numerical Features | 28 continuous numerical features derived from Principal Component Analysis (PCA). |
| `Amount` | `float64` | Numerical Feature | Monetary transaction amount in currency units (Range: 0.0 to 24,039.93). |
| `Class` | `int64` | Target Label | Binary ground truth label (`0` = Legitimate, `1` = Fraud). |

### 4.3 Data Quality Audit
When audited dynamically in `train.py`:
- **Missing Values:** Exactly **0 missing/null entries** across all 31 columns.
- **Duplicate Records:** Exactly **1 duplicate row** (excluding the unique `id` column).
- **Class Distribution:**
  - **Legitimate (`Class 0`):** 284,315 transactions (**50.00%**)
  - **Fraudulent (`Class 1`):** 284,315 transactions (**50.00%**)
  - Total: 568,630 transactions (**100.00%**)

### 4.4 The Anonymized PCA Features ($V_1$ to $V_{28}$)
In financial datasets, transaction features (such as cardholder name, account number, merchant category code, terminal ID, geographic IP location, and purchase history) contain sensitive Personal Identifiable Information (PII) and proprietary banking data.

To protect customer privacy and comply with privacy regulations (such as GDPR), the data publisher applied **Principal Component Analysis (PCA)**:
- PCA projects the high-dimensional correlated raw banking features onto orthogonal axes of maximum variance.
- Features $V_1$ through $V_{28}$ are the resulting principal components.
- **What can be said:** They are mathematically orthogonal, continuous numerical features that capture cardholder behavioral patterns and anomaly signatures.
- **What cannot be said:** We cannot attribute specific real-world meanings (e.g., we cannot state that $V_4$ is "merchant location" or $V_{12}$ is "cardholder age"). Doing so would be speculative and factually incorrect.

### 4.5 Absence of the `Time` Column
*Crucial Difference from Older Datasets:* Older datasets (such as the 2013 ULB dataset) included an elapsed `Time` column. The 2023 dataset contains **no `Time` column**. The feature matrix consists strictly of $V_1$ through $V_{28}$ plus `Amount` (totaling 29 features).

---

# SECTION 5 — WHY THIS DATASET WAS CHOSEN & LIMITATIONS

### 5.1 Why This Dataset is Suitable for an ML Laboratory
1. **Inherent Class Balance:** Unlike raw production datasets where fraud occurs at a rate of 0.17% (which can complicate standard classification for introductory lab demonstrations), this benchmark dataset is balanced (50% Class 0 and 50% Class 1).
2. **Methodological Simplicity:** Because the dataset is balanced, students do not need heuristic undersampling, SMOTE oversampling, or synthetic minority generation. The full mathematical rigor of Support Vector Machines and `StandardScaler` can be studied cleanly.
3. **Standard Feature Scale:** Features $V_1$ through $V_{28}$ have well-behaved variance distributions, making them ideal for demonstrating distance-based algorithms like SVM.
4. **Large Sample Size ($N = 568,630$):** Provides a realistic test of SVM computational scalability and demonstrates why kernel choice and optimization parameters matter.

### 5.2 Dataset Limitations
1. **Synthetic Balancing Artifacts:** To achieve a 50/50 balance across 568,630 records, the publishers synthetically balanced the distribution. Consequently, real-world base-rate fraud probabilities (~0.1% to 0.2%) are not directly mirrored here.
2. **Lack of Temporal Attributes:** Without transaction timestamps, temporal feature engineering (such as velocity checks, e.g., "number of transactions in the last 10 minutes") cannot be performed.
3. **Anonymized Interpretability:** Because $V_1$ to $V_{28}$ are PCA components, we cannot provide business-level feature explanations (e.g., "fraud occurred because transaction took place in a foreign country").

---

# SECTION 6 — DATA PREPROCESSING

The preprocessing pipeline in `train.py` follows strict academic and production best practices.

### 6.1 Step 1: Loading the Data
- **What it does:** Reads the raw CSV file into a pandas DataFrame.
- **Why we do it:** Enables structured tabular operations and schema inspection.
- **Implementation:**
  ```python
  df = pd.read_csv("data/creditcard_2023.csv")
  ```

### 6.2 Step 2: Removing the Identifier Column (`id`)
- **What it does:** Drops the `id` column from the DataFrame.
- **Why we do it:** `id` is an arbitrary sequential integer assigned to each row ($0, 1, 2, \dots$). It carries zero semantic relationship to whether a transaction is fraudulent. If retained, an algorithm might learn spurious correlations with transaction index numbers.
- **Implementation:**
  ```python
  df_clean = df.drop(columns=["id"])
  ```

### 6.3 Step 3: Feature and Target Separation
- **What it does:** Separates the 29 input predictors into feature matrix $X$ and the ground truth into target vector $y$.
- **Implementation:**
  ```python
  X = df_clean.drop(columns=["Class"])  # Shape: (568630, 29)
  y = df_clean["Class"]                 # Shape: (568630,)
  ```

### 6.4 Step 4: Stratified Train-Test Split (80/20)
- **What it does:** Divides the data into 80% training data ($454,904$ samples) and 20% test data ($113,726$ samples).
- **Why we do it:** Machine learning models must be validated on data they have never encountered during parameter estimation to verify generalizability and prevent overfitting.
- **Why `stratify=y`:** Stratification guarantees that the 50/50 ratio of legitimate to fraudulent transactions is preserved in both the training set and the test set:
  - Training Set: 227,452 Legit (50.0%) / 227,452 Fraud (50.0%)
  - Test Set: 56,863 Legit (50.0%) / 56,863 Fraud (50.0%)
- **Why `random_state=42`:** Seeds the pseudo-random number generator to ensure identical data splits across different runs.
- **Implementation:**
  ```python
  X_train, X_test, y_train, y_test = train_test_split(
      X, y,
      test_size=0.2,
      random_state=42,
      stratify=y
  )
  ```

### 6.5 Step 5: Feature Standardization with `StandardScaler`
- **What it does:** Normalizes every numerical feature so that it has a mean of zero ($\mu = 0$) and unit variance ($\sigma = 1$):
  $$z = \frac{x - \mu}{\sigma}$$
- **Why we do it for SVM:** Support Vector Machines find an optimal separating hyperplane by computing Euclidean distances ($||\mathbf{w}||$) and inner products ($\mathbf{x}_i^T \mathbf{x}_j$). The feature `Amount` ranges from $0$ to $24,039.93$, whereas PCA features $V_1$ to $V_{28}$ have much smaller ranges. If left unscaled, `Amount` would dominate the distance metric, making the SVM ignore subtle variations in $V_1$ through $V_{28}$.
- **Why fit ONLY on training data:** **To prevent data leakage.** The parameters $\mu$ (mean) and $\sigma$ (standard deviation) must be calculated exclusively from the training partition. If `fit_transform` were called on the entire dataset prior to splitting, information from the test set would influence the scaling parameters.
- **Implementation:**
  ```python
  scaler = StandardScaler()
  X_train_scaled = pd.DataFrame(
      scaler.fit_transform(X_train),
      columns=X_train.columns,
      index=X_train.index
  )
  X_test_scaled = pd.DataFrame(
      scaler.transform(X_test),
      columns=X_test.columns,
      index=X_test.index
  )
  ```

### 6.6 No Resampling Required
Because the 2023 dataset is inherently class-balanced, **no undersampling, SMOTE oversampling, or artificial class weighting (`class_weight='balanced'`) is used**. The natural data is used directly.

---

# SECTION 7 — MACHINE LEARNING ALGORITHM: SUPPORT VECTOR MACHINES (SVM)

### 7.1 Absolute Basics: Supervised Learning & Classification
- **Supervised Learning:** A branch of machine learning where an algorithm is provided with input-output pairs $(\mathbf{x}_i, y_i)$ during training. The algorithm learns a functional mapping that accurately predicts outputs for new, unseen inputs.
- **Classification:** A supervised learning task where the output label $y$ is categorical rather than continuous. In our project, it is **binary classification** ($y \in \{0, 1\}$).

### 7.2 What is a Support Vector Machine?
A Support Vector Machine (SVM) is a maximum-margin classifier. Given a set of points from two classes in an $n$-dimensional space, SVM finds a decision boundary that separates the classes while maximizing the distance between the boundary and the closest data points from either class.

```
       Class 1 (Fraud)
          ▲  ▲
          │   ▲    [Support Vector]
          │     ▲       │
──────────┼─────────────┼────────────── Margin Boundary (+1)
          │             ▼
          │      ══════════════════════ OPTIMAL HYPERPLANE: w^T x + b = 0
          │             ▲
──────────┼─────────────┼────────────── Margin Boundary (-1)
          │     ●       │
          │   ●    [Support Vector]
          ●  ●
       Class 0 (Legitimate)
          ◄─────────────►
            Total Margin
```

### 7.3 Core Geometric Concepts
1. **Hyperplane:** An $(n-1)$-dimensional flat affine subspace dividing an $n$-dimensional space into two halves. In our 29-dimensional space, the hyperplane is defined algebraically as:
   $$\mathbf{w}^T \mathbf{x} + b = 0$$
   where $\mathbf{w} \in \mathbb{R}^{29}$ is the normal weight vector (perpendicular to the hyperplane) and $b \in \mathbb{R}$ is the bias offset.
2. **Margin:** The geometric distance between the decision boundary ($\mathbf{w}^T \mathbf{x} + b = 0$) and the closest training points on either side:
   $$\text{Margin} = \frac{2}{||\mathbf{w}||}$$
   SVM minimizes $||\mathbf{w}||^2$ (which maximizes the margin $\frac{2}{||\mathbf{w}||}$). Maximizing the margin provides theoretical guarantees against overfitting (Vapnik-Chervonenkis dimension minimization).
3. **Support Vectors:** The critical training instances that lie exactly on the margin boundaries ($\mathbf{w}^T \mathbf{x}_i + b = \pm 1$). These points "support" the decision boundary. If any non-support-vector point is moved or removed, the hyperplane remains identical.
4. **Kernel Function ($K(\mathbf{x}, \mathbf{x}')$):** A mathematical function that computes the inner product of two vectors in a transformed feature space without explicitly calculating the transformation coordinates:
   $$K(\mathbf{x}, \mathbf{x}') = \langle \phi(\mathbf{x}), \phi(\mathbf{x}') \rangle$$
   This is known as the **Kernel Trick**.

### 7.4 The Linear SVM Kernel
- **Mathematical Form:**
  $$K(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T \mathbf{x}'$$
- **How it works:** Constructs a straight, planar decision boundary directly in the original 29-dimensional feature space.
- **When it is useful:** Computationally straightforward, produces an interpretable weight vector $\mathbf{w}$, and works well when features have high dimensionality and classes are linearly separable.

### 7.5 The RBF (Radial Basis Function / Gaussian) Kernel
- **Mathematical Form:**
  $$K(\mathbf{x}, \mathbf{x}') = \exp\left(-\gamma ||\mathbf{x} - \mathbf{x}'||^2\right)$$
- **How it works:** Implicitly maps input vectors into an infinite-dimensional Hilbert space. In this transformed space, complex, non-linear, island-like, or curved separation boundaries in the original 29 dimensions become linearly separable.
- **Parameters Used in Project:**
  1. **$C = 1.0$ (Regularization Parameter):** Controls the trade-off between maximizing the margin and minimizing training classification errors. A moderate $C=1.0$ provides balanced regularization without overfitting to noise.
  2. **$\gamma = \text{'scale'}$ (Kernel Coefficient):** Defines how far the influence of a single training point reaches:
     $$\gamma = \frac{1}{n_{\text{features}} \cdot \text{Var}(X)} = \frac{1}{29 \cdot 1.0} \approx 0.0345$$
     Setting $\gamma = \text{'scale'}$ prevents individual data points from exerting overly localized influence, ensuring smooth decision boundaries.

---

# SECTION 8 — MODEL IMPLEMENTATION

Both models are implemented in `train.py` using Scikit-Learn's `sklearn.svm.SVC`.

### 8.1 Actual Model Definitions in Code

```python
from sklearn.svm import SVC

# 1. Linear SVM
linear_model = SVC(kernel="linear", cache_size=1000)

# 2. RBF SVM
rbf_model = SVC(kernel="rbf", C=1.0, gamma="scale", cache_size=1000)
```

### 8.2 Parameter Explanation
- `kernel="linear"`: Specifies the linear dot-product kernel.
- `kernel="rbf"`: Specifies the Radial Basis Function Gaussian kernel.
- `C=1.0`: Standard soft-margin penalty.
- `gamma="scale"`: Uses $1 / (n_{\text{features}} \cdot X.\text{var}())$ for the RBF Gaussian spread.
- `cache_size=1000`: Allocates 1,000 MB (1 GB) of RAM to cache kernel matrix evaluations in the underlying LIBSVM C++ solver, speeding up convergence.

### 8.3 Core Methods Explained
1. **`model.fit(X_train, y_train)`:**
   - **What it does:** Solves the convex quadratic programming dual optimization problem:
     $$\max_{\alpha} \sum_{i=1}^N \alpha_i - \frac{1}{2}\sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$$
     subject to $0 \le \alpha_i \le C$ and $\sum_{i=1}^N \alpha_i y_i = 0$.
   - **What it learns:** Identifies the non-zero dual coefficients $\alpha_i$ (the support vectors) and the bias $b$.
2. **`model.predict(X_test)`:**
   - **What it does:** Generates discrete class labels ($0$ or $1$) for input samples by evaluating:
     $$\hat{y} = \text{sign}\left(\sum_{i \in \text{SV}} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b\right)$$
     If the sum is $\ge 0$, it returns $1$ (Fraud); if $< 0$, it returns $0$ (Legitimate).
3. **`model.decision_function(X_test)`:**
   - **What it does:** Returns the raw, unthresholded continuous distance score:
     $$f(\mathbf{x}) = \sum_{i \in \text{SV}} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b$$
   - **Important Note:** This output is a **geometric distance score**, NOT a probability. It is positive for fraud and negative for legitimate transactions.

---

# SECTION 9 — TRAINING PROCESS

The complete training execution executed via `python3 train.py`:

```
============================================================
  CreditGuard — Credit Card Fraud Detection using SVM
  Dataset 2023 Pipeline
============================================================

Step 1: Raw CSV loaded (568,630 rows x 31 columns)
Step 2: Quality audit completed (0 missing, 1 duplicate, 50/50 balance)
Step 3: 'id' dropped. Features (29 cols) and target separated.
Step 4: Stratified 80/20 split executed (454,904 train / 113,726 test).
Step 5: StandardScaler fitted strictly on X_train. X_train and X_test transformed.
Step 6: Linear SVM trained on 454,904 samples in 1224.2 seconds (~20.4 min).
Step 7: RBF SVM trained on 454,904 samples in 176.6 seconds (~2.94 min).
Step 8: Models evaluated dynamically on 113,726 test samples.
Step 9: Artifacts saved to models/ directory.
```

### What the Models Learned
- **Linear SVM:** Found an optimal 29-dimensional hyperplane parameterized by weight vector $\mathbf{w} \in \mathbb{R}^{29}$ and intercept $b \in \mathbb{R}$.
- **RBF SVM:** Identified a set of support vectors and dual weights $\alpha_i$ that create a non-linear Gaussian envelope separating fraudulent from legitimate transactions.

---

# SECTION 10 — PREDICTION PROCESS IN STREAMLIT

When a user opens the application, selects a transaction, and clicks **`[ ⚡ Analyze Transaction ]`**, the application executes the following end-to-end pipeline:

```
User clicks [ ⚡ Analyze Transaction ]
                     │
                     ▼
1. RETRIEVE TRANSACTION
   Selected index extracted from st.session_state["selected_txn_idx"]
   Row extracted from untouched test set: row_df = X_test.iloc[[curr_idx]]
                     │
                     ▼
2. RECOVER ORIGINAL VALUES
   Unscaled Amount recovered for display:
   orig_row = scaler.inverse_transform(row_df)[0]
   orig_amount = float(orig_row[-1])
                     │
                     ▼
3. PASS SCALED FEATURES TO MODEL
   Model selected (linear_model or rbf_model)
                     │
                     ▼
4. CALCULATE DECISION FUNCTION
   raw_score = float(model.decision_function(row_df)[0])
                     │
                     ▼
5. GENERATE PREDICTION LABEL
   pred_label = int(model.predict(row_df)[0])  # 0 or 1
                     │
                     ▼
6. COMPARE WITH GROUND TRUTH
   actual_label = int(y_test.iloc[curr_idx])
   is_correct = (pred_label == actual_label)
                     │
                     ▼
7. RENDER USER INTERFACE
   - Transaction Details Card (ID, Amount, Actual Class)
   - 6-Stage Visual Pipeline
   - Prominent Result Card (✓ LEGITIMATE or ⚠ FRAUD)
   - Decision Score Badge & Verification Assessment
```

---

# SECTION 11 — STREAMLIT APPLICATION ARCHITECTURE

The application (`app.py`) is organized into 5 primary pages accessible via the clean sidebar navigation menu:

```
CREDITGUARD SVM FRAUD DETECTION
├── 🏠 Home
├── 🔍 Fraud Detection  (Main Demonstration Page)
├── ⚖️ Model Comparison
├── 📊 Data Analysis
└── 📖 Methodology
```

### Page 1: `🏠 Home`
- **Purpose:** Academic landing page and executive summary.
- **Components:**
  - Academic top header and badge: `CREDIT CARD FRAUD DETECTION · SVM CLASSIFICATION`.
  - Hero panel explaining the project without commercial hype.
  - Call-to-action button: `[ 🔍 Analyze a Transaction ]` (redirects directly to Fraud Detection).
  - 4 Dynamic statistic cards calculated from the dataset: Total Transactions ($568,630$), Fraud Transactions ($284,315$), Fraud Rate ($50.0\%$), Models ($2$).
  - 5-Step visual Machine Learning Workflow diagram.
  - "How the Model Works" and "Models Used" architectural overview cards.

### Page 2: `🔍 Fraud Detection` (Main Demonstration Page)
- **Purpose:** Primary evaluation interface for classroom viva and live testing.
- **Components:**
  - Model selector radio button: `Linear SVM`, `RBF SVM`, `Compare Both Models`.
  - Quick Demonstration toolbar with 3 interactive sampling buttons:
    - `[ 🟢 Random Legitimate Transaction ]`
    - `[ 🔴 Random Fraud Transaction ]`
    - `[ 🎲 Random Transaction ]`
  - Manual transaction index input (`st.number_input` from $0$ to $113,725$).
  - Primary execution button: `[ ⚡ Analyze Transaction ]`.
  - Transaction Details panel: Transaction ID, Original Amount (€), Ground Truth Label.
  - 6-Stage ML Pipeline visualization.
  - Primary Prediction Result Card:
    - `✓ LEGITIMATE TRANSACTION` (Green theme)
    - `⚠ FRAUDULENT TRANSACTION DETECTED` (Red theme)
  - Decision Score indicator ($w^T \phi(x) + b$) with clarifying note explaining that the score is a signed geometric distance, not a probability.
  - Detailed model comparison table (when "Compare Both Models" is selected).
  - Expandable Technical Feature Inspector showing all 29 input values ($V_1$ through $V_{28}$, Amount).

### Page 3: `⚖️ Model Comparison`
- **Purpose:** Quantitative side-by-side performance evaluation.
- **Components:**
  - 2 Large model benchmark cards (Linear SVM vs RBF SVM) displaying Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and Training Duration.
  - Confusion Matrix heatmaps (Blues for Linear SVM, Oranges for RBF SVM).
  - ROC Curves comparing both models against a random classifier baseline.
  - Precision-Recall Curves comparing PR-AUC trajectories.
  - Key Observation panel analyzing performance trade-offs dynamically.

### Page 4: `📊 Data Analysis`
- **Purpose:** Exploratory data analysis supporting the ML model.
- **Components:**
  - 4 Summary cards: Total Rows, Fraud Transactions, Legitimate Transactions, Fraud Percentage.
  - Visualizations:
    1. Legitimate vs Fraud Count bar chart.
    2. Transaction Amount Distribution density histogram.
  - Interactive PCA Feature Explorer: Select any feature ($V_1$ through $V_{28}$) from a dropdown to inspect its distribution for legitimate vs fraudulent classes.

### Page 5: `📖 Methodology`
- **Purpose:** Academic documentation and technical details for viva review.
- **Components:**
  - Four cards covering Problem Formulation, Dataset & Scaling, Balanced Dataset & Split, and Support Vector Machines.

---

# SECTION 12 — FRAUD DETECTION DEMONSTRATION WORKFLOW

### 12.1 How Demonstration Buttons Work
To demonstrate that the model actually works during viva, the professor can click any of the 3 demonstration buttons:
1. **`[ 🟢 Random Legitimate Transaction ]`:**
   - **How it works:** Queries `y_test` for indices where `y_test == 0`:
     ```python
     legit_indices = np.where(y_test.values == 0)[0]
     st.session_state["selected_txn_idx"] = int(np.random.choice(legit_indices))
     ```
   - Automatically selects a genuine historical transaction from the test set.
2. **`[ 🔴 Random Fraud Transaction ]`:**
   - **How it works:** Queries `y_test` for indices where `y_test == 1`:
     ```python
     fraud_indices = np.where(y_test.values == 1)[0]
     st.session_state["selected_txn_idx"] = int(np.random.choice(fraud_indices))
     ```
   - Automatically selects an authentic fraud transaction from the test set.
3. **`[ 🎲 Random Transaction ]`:**
   - Selects a uniformly random integer between $0$ and $\text{len}(X_{\text{test}}) - 1$.

### 12.2 Actual Label vs Model Prediction
- **Actual Label (Ground Truth $y$):** The recorded outcome from the test dataset. Because this is a historical benchmark, we know whether the transaction was truly legitimate ($0$) or fraudulent ($1$).
- **Model Prediction ($\hat{y}$):** The output generated by the SVM when fed the standardized 29-feature vector. The model has no access to the ground truth column.
- **Verification Assessment:**
  - **True Positive:** Actual Fraud ($1$), Model Predicted Fraud ($1$) $\rightarrow$ **Correctly Intercepted!**
  - **True Negative:** Actual Legit ($0$), Model Predicted Legit ($0$) $\rightarrow$ **Correctly Cleared!**
  - **False Positive:** Actual Legit ($0$), Model Predicted Fraud ($1$) $\rightarrow$ **False Alarm.**
  - **False Negative:** Actual Fraud ($1$), Model Predicted Legit ($0$) $\rightarrow$ **Missed Fraud.**

---

# SECTION 13 — MODEL EVALUATION METRICS

### 13.1 Confusion Matrix Terminology
All evaluation metrics are derived from the four quadrants of the Confusion Matrix:
- **True Positive (TP):** Fraudulent transactions correctly predicted as fraud.
- **True Negative (TN):** Legitimate transactions correctly predicted as legitimate.
- **False Positive (FP):** Legitimate transactions incorrectly flagged as fraud (Type I error / False Alarm).
- **False Negative (FN):** Fraudulent transactions incorrectly predicted as legitimate (Type II error / Missed Fraud).

### 13.2 Accuracy
- **Definition:** The proportion of total predictions that were correct.
- **Formula:**
  $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
- **Significance:** Measures overall correctness. On balanced datasets (50/50), accuracy is a valid and informative high-level metric.

### 13.3 Precision
- **Definition:** The proportion of flagged transactions that were truly fraudulent.
- **Formula:**
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
- **Banking Significance:** Measures false alarm rate. High precision means when security flags an account, it is almost certainly fraudulent, minimizing unnecessary customer account freezes.

### 13.4 Recall (Sensitivity)
- **Definition:** The proportion of actual fraud transactions that the model successfully intercepted.
- **Formula:**
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
- **Banking Significance:** In financial security, **Recall is often the most critical metric**. A missed fraudulent transaction ($\text{FN}$) results in direct monetary theft and chargeback fees.

### 13.5 F1-Score
- **Definition:** The harmonic mean of Precision and Recall.
- **Formula:**
  $$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$$
- **Significance:** Balances the trade-off between false alarms ($\text{FP}$) and missed thefts ($\text{FN}$).

### 13.6 ROC-AUC (Receiver Operating Characteristic — Area Under Curve)
- **Definition:** Measures the ability of the model's decision function to rank positive instances higher than negative instances across all possible classification thresholds. Plots True Positive Rate ($\text{TPR}$) against False Positive Rate ($\text{FPR}$).
- **Formula:**
  $$\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$
- **Significance:** $1.0$ represents a perfect classifier; $0.5$ represents random guessing.

### 13.7 PR-AUC (Precision-Recall Area Under Curve / Average Precision)
- **Definition:** Summarizes the trade-off curve between Precision and Recall across various decision score thresholds.
- **Formula:**
  $$\text{PR-AUC} = \sum_{k} (R_k - R_{k-1}) P_k$$
- **Significance:** Evaluates how well high precision is maintained as recall increases.

---

# SECTION 14 — CONFUSION MATRIX ANALYSIS

### 14.1 Actual Confusion Matrices from Test Set Evaluation
Evaluated dynamically on the **113,726 untouched test transactions** (56,863 Legitimate, 56,863 Fraud):

#### Linear SVM Confusion Matrix:
```
                      Predicted Legit      Predicted Fraud
Actual Legit (0)       55,768 (TN)          1,095 (FP)
Actual Fraud (1)        2,775 (FN)         54,088 (TP)
```
- **True Negatives (TN):** $55,768$ (Legitimate transactions correctly cleared)
- **False Positives (FP):** $1,095$ (Legitimate transactions flagged as fraud)
- **False Negatives (FN):** $2,775$ (Fraud transactions missed)
- **True Positives (TP):** $54,088$ (Fraud transactions successfully intercepted)

#### RBF SVM Confusion Matrix:
```
                      Predicted Legit      Predicted Fraud
Actual Legit (0)       56,653 (TN)            210 (FP)
Actual Fraud (1)          122 (FN)         56,741 (TP)
```
- **True Negatives (TN):** $56,653$ (Legitimate transactions correctly cleared)
- **False Positives (FP):** $210$ (Legitimate transactions flagged as fraud)
- **False Negatives (FN):** $122$ (Fraud transactions missed)
- **True Positives (TP):** $56,741$ (Fraud transactions successfully intercepted)

### 14.2 Detailed Comparison:
- **Missed Fraud Comparison (FN):**
  - Linear SVM missed $2,775$ frauds.
  - RBF SVM missed only $122$ frauds.
  - **RBF SVM caught $99.79\%$ of all frauds**, reducing missed fraud by over $95\%$ compared to Linear SVM.
- **False Alarm Comparison (FP):**
  - Linear SVM produced $1,095$ false alarms.
  - RBF SVM produced only $210$ false alarms.
  - **RBF SVM reduced customer false alarms by more than $80\%$**.

---

# SECTION 15 — MODEL COMPARISON

### 15.1 Official Benchmark Results Table
All metrics calculated dynamically from `models/results.joblib`:

| Metric | Linear SVM (`kernel='linear'`) | RBF SVM (`kernel='rbf'`) | Superior Model | Difference / Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **Accuracy** | **96.60%** ($0.96597$) | **99.71%** ($0.99708$) | **RBF SVM** | $+3.11\%$ higher overall accuracy |
| **Precision** | **98.02%** ($0.98016$) | **99.63%** ($0.99631$) | **RBF SVM** | $+1.61\%$ higher precision (fewer false alarms) |
| **Recall** | **95.12%** ($0.95120$) | **99.79%** ($0.99785$) | **RBF SVM** | $+4.67\%$ higher recall ($56,741$ vs $54,088$ caught) |
| **F1-Score** | **0.9655** ($0.96546$) | **0.9971** ($0.99708$) | **RBF SVM** | $+0.0316$ higher harmonic mean |
| **ROC-AUC** | **0.9931** ($0.99309$) | **0.9998** ($0.99979$) | **RBF SVM** | Near-perfect threshold ranking ($0.9998$) |
| **PR-AUC** | **0.9942** ($0.99425$) | **0.9997** ($0.99971$) | **RBF SVM** | Superior precision-recall curve envelope |
| **Training Time** | $1224.2\text{ s}$ (~$20.4\text{ min}$) | **$176.6\text{ s}$ (~$2.94\text{ min}$)** | **RBF SVM** | **$6.9\times$ faster training convergence** |

### 15.2 Discussion of Results & Trade-Offs
- **Why RBF SVM Achieves Higher Metrics:** The underlying transaction distribution in 29-dimensional space is non-linear. The RBF kernel implicitly projects transactions into an infinite-dimensional space where complex decision boundaries can be formed around fraudulent instances, capturing non-linear interactions between $V_1$ to $V_{28}$ and Amount that a flat linear hyperplane cannot capture.
- **Why RBF SVM Trained Faster than Linear SVM:** In Scikit-Learn's LIBSVM solver on this specific dataset, the non-linear Gaussian transformation rapidly separates the balanced clusters. As a result, the Sequential Minimal Optimization (SMO) algorithm identifies the active support vectors with fewer constraint iterations. Conversely, the Linear kernel forced the solver to find a compromise hyperplane in 29 dimensions, requiring extensive iterative optimization.
- **Value of the Linear Model:** Linear SVM still achieves $96.60\%$ accuracy and provides direct mathematical interpretability through the weight coefficients $\mathbf{w}$, making it useful as a baseline model.

---

# SECTION 16 — WHY ACCURACY CAN BE MISLEADING (AND THE ROLE OF CLASS BALANCE)

### 16.1 The Classic Accuracy Paradox (Imbalanced Datasets)
In traditional fraud detection datasets (such as the 2013 ULB dataset), fraud occurs in only $0.17\%$ of transactions ($492$ out of $284,807$). On such datasets:
- A trivial dummy model that classifies **every transaction as legitimate** achieves:
  $$\text{Accuracy} = \frac{284,315}{284,807} = 99.83\%$$
- Despite a 99.83% accuracy, this model catches **zero frauds** ($\text{Recall} = 0\%$) and is completely useless.
- This is the **Accuracy Paradox**: accuracy appears high while the model fails at its core task.

### 16.2 The Situation in Our 2023 Balanced Dataset
In our active 2023 dataset:
- Exactly $50\%$ of transactions are Legitimate and $50\%$ are Fraudulent.
- A dummy model that predicts all transactions as legitimate achieves only **$50.00\%$ accuracy** (equivalent to a coin flip).
- Because the classes are balanced, **Accuracy (96.60% for Linear SVM, 99.71% for RBF SVM) is a valid, informative metric**.
- However, financial institutions still examine **Recall** ($99.79\%$) and **Precision** ($99.63\%$) separately because the operational costs of false positives (reviewing false alarms) and false negatives (unintercepted fraud) are distinct.

---

# SECTION 17 — ACTUAL PROJECT RESULTS SUMMARY

```
========================================================================
             FINAL VALIDATED MODEL BENCHMARKS (TEST SET: N=113,726)
========================================================================

Model 1: LINEAR SVM (kernel='linear')
------------------------------------------------------------------------
  • Accuracy:        96.60% (0.96597)
  • Precision:       98.02% (0.98016)
  • Recall:          95.12% (0.95120)
  • F1-Score:        0.9655 (0.96546)
  • ROC-AUC:         0.9931 (0.99309)
  • PR-AUC:          0.9942 (0.99425)
  • Train Time:      1,224.2 seconds (~20.4 minutes)
  • Confusion Matrix:
      TN = 55,768  |  FP = 1,095
      FN =  2,775  |  TP = 54,088

Model 2: RBF SVM (kernel='rbf', C=1.0, gamma='scale')
------------------------------------------------------------------------
  • Accuracy:        99.71% (0.99708)
  • Precision:       99.63% (0.99631)
  • Recall:          99.79% (0.99785)
  • F1-Score:        0.9971 (0.99708)
  • ROC-AUC:         0.9998 (0.99979)
  • PR-AUC:          0.9997 (0.99971)
  • Train Time:      176.6 seconds (~2.94 minutes)
  • Confusion Matrix:
      TN = 56,653  |  FP =   210
      FN =    122  |  TP = 56,741
========================================================================
```

---

# SECTION 18 — COMPLETE CODEBASE EXPLANATION

The project structure consists of the following core files:

```
mlMiniProj/
├── data/
│   └── creditcard_2023.csv          (Raw CSV: 568,630 rows x 31 cols)
├── models/
│   ├── linear_svm.joblib            (Serialized Linear SVM model)
│   ├── rbf_svm.joblib               (Serialized RBF SVM model)
│   ├── scaler.joblib                (Serialized StandardScaler)
│   ├── preprocessed_data.joblib     (Serialized train/test splits)
│   └── results.joblib               (Dictionary of all dynamic evaluation results)
├── train.py                         (Offline training & evaluation pipeline)
├── app.py                           (Streamlit web application)
└── requirements.txt                 (Python environment dependencies)
```

### 18.1 File: `train.py`
- **Purpose:** Standalone script that executes the complete machine learning training workflow from raw data to serialized model artifacts.
- **Key Functions:**
  - `load_data(path)`: Verifies dataset existence and loads CSV using `pd.read_csv`.
  - `explore_data(df)`: Computes and prints row count, column list, head preview, missing value tally, duplicate tally, and class balance.
  - `preprocess(df)`: Drops `id`, extracts $X$ and $y$, performs stratified 80/20 train-test split, fits `StandardScaler` on $X_{\text{train}}$, and standardizes $X_{\text{train}}$ and $X_{\text{test}}$.
  - `save_preprocessed(...)`: Dumps scaler and data splits using `joblib.dump`.
  - `train_svm(X_train, y_train, kernel)`: Instantiates `SVC` with requested kernel, records training duration, fits the model, and returns `(model, train_time)`.
  - `evaluate_model(model, X_test, y_test, model_name)`: Computes predictions via `predict`, decision scores via `decision_function`, calculates confusion matrix and metrics, and returns metric dictionary.
  - `save_models(...)`: Persists trained models, scaler, and evaluation dictionaries to `models/`.

### 18.2 File: `app.py`
- **Purpose:** Interactive multi-page Streamlit web application.
- **Key Architectural Sections:**
  - **Custom Styling & Layout:** Uses custom CSS styles for clean cards, badges, and metric displays.
  - **Caching Architecture:**
    - `@st.cache_data def load_dataset()`: Caches CSV reading to avoid disk I/O on re-renders.
    - `@st.cache_data def load_results()`: Caches evaluation dictionary loading.
    - `@st.cache_resource def load_models()`: Caches deserialized scikit-learn models in memory.
  - **Sidebar Navigation:** Renders a radio button menu controlling display of the 5 pages (`🏠 Home`, `🔍 Fraud Detection`, `⚖️ Model Comparison`, `📊 Data Analysis`, `📖 Methodology`).
  - **Transaction Demonstrator Engine:** Manages session state (`selected_txn_idx`, `has_run_analysis`), handles the 3 demo buttons, executes real-time inference, and renders the result cards.

### 18.3 File: `requirements.txt`
Specifies the exact Python libraries required:
```txt
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

---

# SECTION 19 — IMPORTANT PYTHON & SCIKIT-LEARN CONCEPTS

1. **`pandas.DataFrame`:** A 2-dimensional labeled tabular data structure with heterogeneous column types. Used to store, index, and manipulate the transaction dataset.
2. **`train_test_split`:** Utility from `sklearn.model_selection` that partitions arrays or matrices into random train and test subsets.
3. **`stratify=y`:** Parameter in `train_test_split` that ensures the train and test subsets contain the exact same class proportions as the original dataset.
4. **`StandardScaler`:** Transformer from `sklearn.preprocessing` that standardizes features by removing the mean and scaling to unit variance ($z = (x - \mu) / \sigma$).
5. **`SVC`:** Support Vector Classifier class from `sklearn.svm`, built on top of the LIBSVM library.
6. **`fit(X, y)`:** Fits the model by estimating the parameters (support vectors, weights, and biases) from training data.
7. **`predict(X)`:** Computes discrete class labels ($0$ or $1$) for input samples by applying a sign threshold to the decision function.
8. **`decision_function(X)`:** Computes the continuous signed distance of input samples to the separating hyperplane. Positive values indicate Class 1, negative values indicate Class 0.
9. **`confusion_matrix(y_true, y_pred)`:** Computes the $2 \times 2$ matrix of classification outcomes ($\text{TN}, \text{FP}, \text{FN}, \text{TP}$).
10. **`roc_auc_score(y_true, y_score)`:** Computes the Area Under the Receiver Operating Characteristic Curve from prediction scores.
11. **`average_precision_score(y_true, y_score)`:** Computes the Area Under the Precision-Recall Curve (PR-AUC).
12. **`joblib`:** Library for efficient persistence (serialization and deserialization) of Python objects containing large NumPy arrays.
13. **`streamlit`:** Python framework for building interactive data science web applications without writing frontend JavaScript.

---

# SECTION 20 — 50 COMMON VIVA QUESTIONS & ANSWERS

### Category A: General Machine Learning & Classification
1. **Q: What is Machine Learning?**
   - **Answer:** Machine Learning is a branch of artificial intelligence where algorithms learn patterns from empirical data to perform tasks without being explicitly programmed with static rules.
2. **Q: What is supervised learning?**
   - **Answer:** A machine learning paradigm where the algorithm is trained on labeled data pairs $(\mathbf{x}, y)$, learning a mapping from input features to known target outputs.
3. **Q: What is classification?**
   - **Answer:** A supervised learning task where the target output variable is categorical rather than continuous.
4. **Q: What is binary classification?**
   - **Answer:** A classification task with exactly two discrete classes (e.g., 0 = Legitimate, 1 = Fraud).
5. **Q: What is the difference between classification and regression?**
   - **Answer:** Classification predicts discrete class labels, whereas regression predicts continuous numerical values.
6. **Q: Why is fraud detection framed as classification rather than regression?**
   - **Answer:** Because the operational goal is to make a categorical decision: approve the transaction (legitimate) or block/flag it (fraud).

### Category B: The Dataset
7. **Q: What dataset did you use for this project?**
   - **Answer:** The Credit Card Fraud Detection Dataset 2023, sourced from Kaggle.
8. **Q: How many transactions are in the dataset?**
   - **Answer:** Exactly 568,630 transactions.
9. **Q: How many features are used by your models?**
   - **Answer:** 29 numerical features: $V_1$ through $V_{28}$, plus `Amount`.
10. **Q: What happened to the `id` column?**
    - **Answer:** It was removed during preprocessing because it is an arbitrary row index that carries no predictive information.
11. **Q: What does Class 0 represent?**
    - **Answer:** A legitimate, authorized credit card transaction.
12. **Q: What does Class 1 represent?**
    - **Answer:** A fraudulent, unauthorized transaction.
13. **Q: Is the dataset balanced or imbalanced?**
    - **Answer:** The 2023 dataset is balanced, containing exactly 284,315 legitimate transactions (50%) and 284,315 fraud transactions (50%).
14. **Q: What are the $V_1$ to $V_{28}$ features?**
    - **Answer:** They are principal component numerical features obtained via Principal Component Analysis (PCA) to protect cardholder confidentiality.
15. **Q: Can we determine the real-world meaning of $V_1$ or $V_{14}$?**
    - **Answer:** No, because they are anonymized mathematical projections of the original banking features.
16. **Q: Is there a `Time` column in the 2023 dataset?**
    - **Answer:** No, the 2023 dataset does not contain a `Time` column. Only $V_1$–$V_{28}$ and `Amount` are present.
17. **Q: Were there any missing values in the dataset?**
    - **Answer:** No, data auditing confirmed 0 missing values across all 568,630 rows.
18. **Q: Were there any duplicate rows?**
    - **Answer:** Exactly 1 duplicate row was identified and handled during exploration.

### Category C: Preprocessing & Data Hygiene
19. **Q: Why did you split the dataset into training and test sets?**
    - **Answer:** To evaluate the trained model on unseen data to test generalizability and ensure it has not simply memorized the training data.
20. **Q: What split ratio was used?**
    - **Answer:** An 80/20 train-test split (454,904 training samples and 113,726 test samples).
21. **Q: Why did you use `stratify=y`?**
    - **Answer:** To preserve the exact 50/50 proportion of legitimate and fraudulent transactions across both partitions.
22. **Q: Why did you set `random_state=42`?**
    - **Answer:** To seed the random number generator so that the train-test split is reproducible.
23. **Q: Why is feature scaling necessary for SVM?**
    - **Answer:** SVM relies on Euclidean distance metrics. Features with large scales (like `Amount`, up to $24,000$) would dominate distance calculations over features with smaller numerical ranges.
24. **Q: What scaler was used?**
    - **Answer:** Scikit-Learn's `StandardScaler`, which standardizes features to zero mean and unit variance ($z = (x - \mu) / \sigma$).
25. **Q: What is data leakage?**
    - **Answer:** Data leakage occurs when information from outside the training dataset (such as test set statistics) is used to train the model or fit preprocessing transformers.
26. **Q: How did you prevent data leakage during scaling?**
    - **Answer:** By fitting the `StandardScaler` strictly on the training set (`fit_transform`) and only using the learned mean and variance to transform the test set (`transform`).
27. **Q: Did you apply undersampling or SMOTE?**
    - **Answer:** No, because the 2023 dataset is already balanced (50% Class 0 and 50% Class 1).

### Category D: Support Vector Machines (SVM)
28. **Q: What is a Support Vector Machine?**
    - **Answer:** A supervised learning classifier that finds an optimal separating hyperplane that maximizes the margin between two classes.
29. **Q: What is a hyperplane?**
    - **Answer:** An $(n-1)$-dimensional flat affine decision boundary dividing an $n$-dimensional feature space into two classes ($\mathbf{w}^T \mathbf{x} + b = 0$).
30. **Q: What is the margin in SVM?**
    - **Answer:** The geometric distance between the decision boundary and the nearest data points of either class ($2 / ||\mathbf{w}||$).
31. **Q: Why does SVM maximize the margin?**
    - **Answer:** A wider margin provides theoretical guarantees of better generalization to unseen data and lower risk of overfitting.
32. **Q: What are support vectors?**
    - **Answer:** The critical training data points that lie directly on the margin boundaries and define the position and orientation of the separating hyperplane.
33. **Q: If non-support vectors are removed from the training data, what happens to the hyperplane?**
    - **Answer:** The hyperplane remains identical because its position depends solely on the support vectors.
34. **Q: What is a kernel in SVM?**
    - **Answer:** A function that computes the inner product between points in a higher-dimensional space without explicitly calculating their coordinates ($K(\mathbf{x}, \mathbf{x}') = \langle \phi(\mathbf{x}), \phi(\mathbf{x}') \rangle$).
35. **Q: What kernels did you implement?**
    - **Answer:** The Linear kernel (`kernel='linear'`) and the Radial Basis Function kernel (`kernel='rbf'`).
36. **Q: How does the Linear kernel work?**
    - **Answer:** It computes simple dot products $\mathbf{x}^T \mathbf{x}'$, creating a flat planar decision boundary in the original 29-dimensional space.
37. **Q: How does the RBF kernel work?**
    - **Answer:** It computes Gaussian radial distance $\exp(-\gamma ||\mathbf{x} - \mathbf{x}'||^2)$, mapping points into an infinite-dimensional space to form non-linear decision boundaries.
38. **Q: What does the parameter $C$ do in SVM?**
    - **Answer:** It is the regularization parameter that balances margin width against classification errors on training points.
39. **Q: What does the parameter $\gamma$ do in the RBF kernel?**
    - **Answer:** It defines the influence radius of individual support vectors. Small $\gamma$ produces smooth boundaries; large $\gamma$ produces tighter, localized boundaries around points.
40. **Q: What value of $\gamma$ was used?**
    - **Answer:** $\gamma = \text{'scale'}$, which sets $\gamma = 1 / (n_{\text{features}} \cdot \text{Var}(X))$.

### Category E: Evaluation & Inference
41. **Q: What does `model.predict(X)` return?**
    - **Answer:** Discrete predicted class labels ($0$ for Legitimate, $1$ for Fraud).
42. **Q: What does `model.decision_function(X)` return?**
    - **Answer:** The raw continuous signed distance from the input sample to the separating hyperplane.
43. **Q: Is the decision score a probability?**
    - **Answer:** No. It is an unbounded geometric Euclidean distance score ($w^T \phi(x) + b$), not a calibrated probability between 0 and 1.
44. **Q: What is a Confusion Matrix?**
    - **Answer:** A $2 \times 2$ table that displays True Negatives, False Positives, False Negatives, and True Positives.
45. **Q: What is a False Positive in fraud detection?**
    - **Answer:** A legitimate transaction incorrectly flagged as fraud (a false alarm).
46. **Q: What is a False Negative in fraud detection?**
    - **Answer:** A fraudulent transaction missed by the model and treated as legitimate.
47. **Q: Which error is more costly in banking: False Positive or False Negative?**
    - **Answer:** False Negatives are typically more costly because undetected fraud leads directly to financial theft and chargebacks.
48. **Q: What is Recall?**
    - **Answer:** The fraction of actual fraud transactions that the model successfully caught ($\text{TP} / (\text{TP} + \text{FN})$).
49. **Q: What is Precision?**
    - **Answer:** The fraction of flagged transactions that were truly fraudulent ($\text{TP} / (\text{TP} + \text{FP})$).
50. **Q: What is the F1-Score?**
    - **Answer:** The harmonic mean of Precision and Recall ($2 \cdot \frac{P \cdot R}{P + R}$).

---

# SECTION 21 — 20 TRICKY PROFESSOR QUESTIONS & DETAILED ANSWERS

1. **"Why did you choose SVM over Logistic Regression or Random Forest?"**
   - **Answer:** SVM is a core focus of our machine learning laboratory syllabus for studying convex optimization and maximum-margin theory. From a technical standpoint, SVM is effective in moderately high-dimensional spaces ($29$ continuous numerical features) where the margin-maximization principle provides strong generalization guarantees.
2. **"Why did the RBF SVM achieve higher accuracy than the Linear SVM?"**
   - **Answer:** Because the true distribution separating fraud from legitimate transactions in the 29-dimensional space is non-linear. The Linear SVM is constrained to a flat hyperplane, whereas the RBF Gaussian kernel can construct curved, non-linear decision boundaries around fraudulent patterns.
3. **"Why did RBF SVM train in 2.9 minutes while Linear SVM took 20.4 minutes on the same 454,904 training points?"**
   - **Answer:** In LIBSVM's Sequential Minimal Optimization (SMO) solver, the RBF kernel easily separates the balanced clusters in its transformed feature space, allowing the solver to identify support vectors with fewer iterations. In contrast, the linear kernel could not completely separate the classes with a flat hyperplane in 29 dimensions, forcing the SMO algorithm through many active-set constraint iterations.
4. **"What would happen if you scaled the data before the train-test split?"**
   - **Answer:** It would introduce data leakage. The global mean $\mu$ and standard deviation $\sigma$ would incorporate values from the test set, giving the training process indirect knowledge about the distribution of test instances.
5. **"Why didn't you scale only the `Amount` feature since $V_1$ to $V_{28}$ are already PCA components?"**
   - **Answer:** While $V_1$ to $V_{28}$ are PCA components, their individual empirical standard deviations vary across the dataset. Applying `StandardScaler` to all 29 features ensures that every feature contributes equally to the distance calculation in the SVM kernel.
6. **"Can you explain the mathematical difference between $C=0.01$ and $C=100$ in your SVM?"**
   - **Answer:** $C$ is the penalty weight for margin violations in the objective function $\frac{1}{2}||\mathbf{w}||^2 + C \sum \xi_i$. A small $C$ ($0.01$) produces a wider margin and tolerates more classification errors (underfitting risk). A large $C$ ($100$) penalizes margin violations heavily, forcing a narrower margin that fits training points closely (overfitting risk). We used $C=1.0$ as a balanced default.
7. **"What is the mathematical meaning of the decision score returned by `decision_function`?"**
   - **Answer:** It is the signed orthogonal Euclidean distance from the transaction point to the separating hyperplane: $f(\mathbf{x}) = \mathbf{w}^T \phi(\mathbf{x}) + b$. A score of $+2.5$ means the point lies $2.5$ normalized units inside the fraud half-space; a negative score indicates the legitimate half-space; $0$ lies directly on the decision boundary.
8. **"How does the Streamlit app know the transaction Amount if it was scaled during training?"**
   - **Answer:** The app uses the serialized `scaler.joblib` to call `scaler.inverse_transform(row_df)[0][-1]`, reversing the z-score standardization ($x = z \cdot \sigma + \mu$) to retrieve the original monetary amount for display.
9. **"Why is `cache_size=1000` specified in the code?"**
   - **Answer:** `cache_size=1000` allocates 1,000 MB of RAM to store recently evaluated kernel matrix entries. This reduces the need to recompute kernel inner products during the SMO optimization loop, speeding up training on large datasets.
10. **"How do you know your model didn't simply memorize the training data?"**
    - **Answer:** Because the evaluation metrics were computed on an untouched test partition of $113,726$ transactions that was held out before training. The RBF SVM achieved $99.71\%$ accuracy on this unseen test data, demonstrating genuine generalization.
11. **"Why did you use Stratified sampling instead of simple random sampling?"**
    - **Answer:** Simple random sampling can introduce small distribution shifts between partitions. Stratified sampling guarantees that the training and test sets have identical 50/50 class balance.
12. **"What is the geometric interpretation of a Support Vector?"**
    - **Answer:** A support vector is a training point $\mathbf{x}_i$ whose dual Lagrange multiplier $\alpha_i > 0$. Geometrically, it lies on the margin boundary ($\mathbf{w}^T \phi(\mathbf{x}_i) + b = \pm 1$) or violates the margin ($\xi_i > 0$).
13. **"Is Accuracy a misleading metric in your project?"**
    - **Answer:** In imbalanced datasets (e.g., 0.17% fraud), accuracy is misleading due to the Accuracy Paradox. However, in our dataset, exactly 50% of transactions are fraud and 50% are legitimate. Therefore, accuracy is a valid summary metric, though we still analyze Precision and Recall to evaluate false alarms and missed frauds separately.
14. **"Why didn't you use `class_weight='balanced'` in your current `train.py`?"**
    - **Answer:** `class_weight='balanced'` adjusts the misclassification penalty inversely proportional to class frequencies ($w_j = N / (2 \cdot N_j)$). Because our dataset already has an exact 50/50 balance ($N_0 = N_1$), the calculated class weights would both be $1.0$, making the parameter redundant.
15. **"What happens if a new transaction has an `Amount` larger than any amount seen in training?"**
    - **Answer:** `StandardScaler` standardizes it by subtracting the training mean and dividing by the training standard deviation. The resulting z-score will be large, and the SVM will project it along the normal vector $\mathbf{w}$. If the combination of features indicates fraud, it will be classified accordingly.
16. **"Why is the `id` column dropped instead of being used as a feature?"**
    - **Answer:** `id` is an arbitrary sequential database index. It has no causal or statistical relationship to fraudulent behavior. Including it would risk having the model learn spurious correlations with transaction sequence numbers.
17. **"Can an SVM handle missing values natively?"**
    - **Answer:** No. SVM computes dot products and Euclidean distances across all feature dimensions. If a single coordinate is missing (`NaN`), distance calculations fail. Our data audit verified that the dataset contains 0 missing values.
18. **"How does the Streamlit application demonstrate both correct predictions and misclassifications?"**
    - **Answer:** By sampling real transactions from the test set where the model's prediction matches ground truth (True Positives, True Negatives) or differs from ground truth (False Positives, False Negatives), the user can observe the model's behavior on both successful detections and edge cases.
19. **"What is the difference between ROC-AUC and PR-AUC?"**
    - **Answer:** ROC-AUC plots True Positive Rate against False Positive Rate ($\text{FP} / (\text{FP} + \text{TN})$). PR-AUC plots Precision against Recall. In datasets with many true negatives, False Positive Rate stays small, which can make ROC-AUC appear optimistic. PR-AUC focuses directly on the minority class trade-off.
20. **"Why serialize models using `joblib` instead of standard `pickle`?"**
    - **Answer:** `joblib` is optimized for serializing large NumPy arrays and Scikit-Learn estimators, resulting in faster disk I/O and lower memory overhead compared to standard `pickle`.

---

# SECTION 22 — LIMITATIONS & FUTURE WORK

### 22.1 Current Implementation Limitations
1. **No Temporal Features:** The 2023 dataset contains no transaction timestamps. The model cannot evaluate time-window metrics, such as spending velocity or transaction frequency per hour.
2. **Batch Training Architecture:** The models are trained offline in batch mode. The pipeline does not support real-time streaming updates (online learning).
3. **Anonymized Feature Interpretability:** Because features $V_1$ through $V_{28}$ are PCA components, the model cannot explain classifications in terms of human-understandable rules (e.g., "unusual merchant category").

### 22.2 Future Work
1. **Real-Time Streaming Pipeline:** Integrate a message broker (such as Apache Kafka) to evaluate incoming transaction streams in real time.
2. **Model Explainability (SHAP / LIME):** Apply SHAP (SHapley Additive exPlanations) to identify which PCA components contributed most to an individual transaction's decision score.
3. **Ensemble Methods Comparison:** Benchmark the SVM classifiers against gradient-boosted decision trees (such as XGBoost and LightGBM).

---

# SECTION 23 — ONE-MINUTE PROJECT EXPLANATION (VIVA SCRIPT)

> *"Good morning, Professor. My project is **Credit Card Fraud Detection using Support Vector Machines with Data Preprocessing**.*
>
> *We formulated fraud detection as a supervised binary classification problem to distinguish legitimate card transactions from unauthorized fraud. We used the **Credit Card Fraud Detection Dataset 2023**, which contains **568,630 transactions** with a balanced 50/50 class distribution across 29 numerical features: $V_1$ to $V_{28}$ from PCA, and transaction Amount.*
>
> *Our preprocessing pipeline drops the non-informative ID column, performs an 80/20 stratified split, and applies `StandardScaler` fitted strictly on the training set to prevent data leakage. We trained two SVM models: a **Linear SVM** and an **RBF SVM**.*
>
> *On our untouched test set of 113,726 transactions, the **RBF SVM achieved superior performance with 99.71% accuracy, 99.79% recall, and caught 56,741 out of 56,863 frauds** while producing only 210 false alarms. The Linear SVM achieved 96.60% accuracy.*
>
> *Finally, we developed an interactive **Streamlit web application** that demonstrates real-time inference on actual test transactions, displays continuous decision scores, and visualizes the internal stages of the ML pipeline."*

---

# SECTION 24 — FIVE-MINUTE PROJECT DEMONSTRATION SCRIPT

### [Minute 1: Introduction & Home Page]
*"Welcome, Professor. This is our Machine Learning Laboratory mini project: Credit Card Fraud Detection using Support Vector Machines.*

*(Navigate to 🏠 Home)*
*Here on the Home page, you can see our overall workflow. Our dataset consists of 568,630 transactions from the 2023 Kaggle benchmark. The data is balanced with 50% legitimate and 50% fraudulent records. The pipeline consists of five stages: data loading and verification, feature standardization, stratified train/test splitting, SVM training, and dynamic test evaluation. We trained two models: Linear SVM and RBF SVM."*

### [Minute 2: Fraud Detection Page & Demo Mode]
*(Navigate to 🔍 Fraud Detection)*
*"Now let us examine the core demonstration interface. This page allows us to feed real transactions from our held-out test set into the trained SVM models.*

*To demonstrate the model live, we have three test-set sampling buttons: 'Random Legitimate Transaction', 'Random Fraud Transaction', and 'Random Transaction'.*

*Let's click **`[ 🔴 Random Fraud Transaction ]`**. The system randomly selects an authentic fraudulent transaction from the test set. Notice that the transaction details show its index and original monetary amount. Now I click **`[ ⚡ Analyze Transaction ]`**."*

### [Minute 3: Analyzing the Prediction & Decision Score]
*"Here is the live inference result. The transaction passed through our 6-stage ML pipeline: Input $\rightarrow$ Feature Extraction $\rightarrow$ StandardScaler $\rightarrow$ SVM Kernel $\rightarrow$ Decision Score $\rightarrow$ Prediction.*

*The model outputs a prominent red alert: **`⚠ FRAUDULENT TRANSACTION DETECTED`**. The verification indicator confirms this is a **True Positive (Correct Prediction)**. Notice the Decision Score: it is positive, indicating that the feature vector lies inside the fraud half-space of the decision boundary. We explicitly note that this is a signed geometric distance, not a probability.*

*Now let's click **`[ 🟢 Random Legitimate Transaction ]`** and analyze it. The result switches to a green card: **`✓ LEGITIMATE TRANSACTION`**, with a negative decision score, confirming a **True Negative**."*

### [Minute 4: Model Comparison Page]
*(Navigate to ⚖️ Model Comparison)*
*"On the Model Comparison page, we benchmark the Linear SVM against the RBF SVM across all 113,726 untouched test transactions.*

*As shown in our metrics cards:
- **Linear SVM** achieved 96.60% accuracy, 98.02% precision, and 95.12% recall.
- **RBF SVM** achieved **99.71% accuracy, 99.63% precision, and 99.79% recall**.

Looking at the confusion matrices: Linear SVM missed 2,775 frauds, whereas RBF SVM missed only 122 frauds out of 56,863. Furthermore, RBF SVM reduced false alarms from 1,095 down to 210. The ROC-AUC of the RBF model is 0.9998, demonstrating strong discriminative ability."*

### [Minute 5: Data Analysis, Methodology & Conclusion]
*(Navigate to 📊 Data Analysis & 📖 Methodology)*
*"Finally, our Data Analysis page provides a look at the feature distributions. We can see the class balance bar chart and amount distributions. The PCA Feature Explorer allows us to select any component, such as $V_{14}$, and observe how its distribution separates fraudulent from legitimate transactions.*

*In conclusion, the laboratory project demonstrates that non-linear RBF Support Vector Machines can effectively identify credit card fraud when combined with proper feature standardization and experimental hygiene. Thank you, Professor. I am ready for questions."*

---

# SECTION 25 — "EXPLAIN THIS CODE TO ME" REFERENCE

### Code Block 1: Stratified Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```
- **What it does:** Splits the 568,630 transactions into 80% training (454,904) and 20% testing (113,726).
- **Why it is needed:** To validate the model on data it has not seen during training.
- **Viva Explanation:** *"We use `stratify=y` to ensure that both the training and test sets maintain the exact 50/50 ratio of legitimate to fraudulent transactions, and `random_state=42` ensures the split is reproducible."*

### Code Block 2: Data-Leakage-Free Feature Scaling
```python
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)
```
- **What it does:** Standardizes all 29 features so that each has zero mean and unit variance ($z = (x - \mu) / \sigma$).
- **Why it is needed:** To prevent high-magnitude features like `Amount` from dominating distance calculations in the SVM kernel.
- **Viva Explanation:** *"We fit the scaler strictly on `X_train` using `fit_transform` and only use `transform` on `X_test`. This prevents data leakage by ensuring that no test set statistics influence the scaling parameters."*

### Code Block 3: Model Fitting
```python
model = SVC(kernel="rbf", C=1.0, gamma="scale", cache_size=1000)
model.fit(X_train_scaled, y_train)
```
- **What it does:** Solves the convex quadratic optimization problem to determine the support vectors and decision boundary.
- **Why it is needed:** Learns the hyperplane that maximizes the separation margin between classes.
- **Viva Explanation:** *"We use `kernel='rbf'` to capture non-linear relationships, $C=1.0$ as a balanced regularization penalty, $\gamma=\text{'scale'}$ to set the Gaussian spread based on feature variance, and `cache_size=1000` to allocate 1 GB of memory to speed up solver convergence."*

### Code Block 4: Generating Predictions and Decision Scores
```python
y_pred = model.predict(X_test)
y_scores = model.decision_function(X_test)
```
- **What it does:** `predict` outputs binary predictions ($0$ or $1$); `decision_function` outputs continuous signed distance scores.
- **Why it is needed:** Discrete labels are used to compute accuracy, precision, and recall; continuous scores are used to compute ROC-AUC and PR-AUC curves.
- **Viva Explanation:** *"`predict` applies a threshold of zero to the signed distance returned by `decision_function`. The decision score represents the Euclidean distance from the data point to the separating hyperplane."*

---

# SECTION 26 — QUICK REVISION SHEET

| Dimension | Project Specification |
| :--- | :--- |
| **Project Title** | Credit Card Fraud Detection using SVM Classification |
| **Dataset Name** | Credit Card Fraud Detection Dataset 2023 |
| **Dataset Source** | Kaggle (`nelgiriyewithana/credit-card-fraud-detection-dataset-2023`) |
| **Dataset Size** | 568,630 rows $\times$ 31 columns |
| **Input Features** | 29 numerical features ($V_1$ through $V_{28}$, and `Amount`) |
| **Target Variable** | `Class` ($0$ = Legitimate, $1$ = Fraud) |
| **Class Proportions** | 50.0% Legitimate ($284,315$) vs 50.0% Fraud ($284,315$) |
| **Train/Test Split** | 80% Train ($454,904$) / 20% Test ($113,726$), Stratified, `random_state=42` |
| **Scaling Technique** | `StandardScaler` (Mean = 0, Std = 1), fitted strictly on training set |
| **Models Evaluated** | 1. Linear SVM (`SVC(kernel='linear')`)<br>2. RBF SVM (`SVC(kernel='rbf', C=1.0, gamma='scale')`) |
| **Evaluation Metrics** | Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Confusion Matrix |
| **Linear SVM Results** | Acc: 96.60% \| Prec: 98.02% \| Rec: 95.12% \| F1: 0.9655 \| ROC-AUC: 0.9931 |
| **RBF SVM Results** | Acc: 99.71% \| Prec: 99.63% \| Rec: 99.79% \| F1: 0.9971 \| ROC-AUC: 0.9998 |
| **UI Framework** | Streamlit (`app.py`, 5 pages: Home, Fraud Detection, Model Comparison, Data Analysis, Methodology) |
| **Model Persistence** | `joblib` (`models/linear_svm.joblib`, `rbf_svm.joblib`, `scaler.joblib`, `results.joblib`) |

### One-Line Definitions of Key Terms:
- **Supervised Learning:** Training an algorithm on labeled input-output pairs.
- **Binary Classification:** Categorizing data points into one of two discrete classes.
- **Hyperplane:** An $(n-1)$-dimensional flat decision boundary separating an $n$-dimensional feature space.
- **Margin:** The geometric distance between the decision boundary and the closest data points from either class.
- **Support Vectors:** The critical training data points lying on the margin boundaries that define the hyperplane.
- **Kernel Trick:** Computing inner products in a higher-dimensional space without explicitly evaluating the transformation.
- **RBF Kernel:** A Gaussian kernel function that maps inputs into an infinite-dimensional space to form non-linear decision boundaries.
- **Data Leakage:** Inadvertently sharing information from the test set with the training process.
- **Precision:** The proportion of flagged transactions that were truly fraudulent ($\text{TP} / (\text{TP} + \text{FP})$).
- **Recall:** The proportion of actual fraudulent transactions successfully caught ($\text{TP} / (\text{TP} + \text{FN})$).
- **Decision Score:** The signed orthogonal distance from a feature point to the SVM decision hyperplane.

---

# CURRENT PROJECT VERIFICATION

```
========================================================================
                      PROJECT INTEGRITY AUDIT
========================================================================
Dataset verified:                     YES (creditcard_2023.csv, 568,630 rows, balanced)
Preprocessing verified:               YES (Drop 'id', Stratified 80/20, StandardScaler)
Linear SVM verified:                  YES (SVC(kernel='linear'), trained & evaluated)
RBF SVM verified:                     YES (SVC(kernel='rbf', C=1.0, gamma='scale'))
Prediction flow verified:             YES (Live inference via model.predict & decision_function)
Evaluation verified:                  YES (Dynamically evaluated on 113,726 test set rows)
Streamlit pages verified:             YES (5 active pages: Home, Detection, Comparison, Analysis, Method)
Metrics generated dynamically:        YES (All metrics extracted from results.joblib)
Hard-coded ML results found:          NO  (Zero hardcoded predictions, scores, or metrics)
========================================================================
```
