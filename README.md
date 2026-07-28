# FraudGuard-AI: Credit Card Fraud Detection & Real-Time Scoring Platform

FraudGuard-AI is an end-to-end Machine Learning training, cost-optimization, real-time inference scoring API, and interactive merchant risk dashboard built on the ULB Credit Card Fraud dataset (284,807 transactions, ~0.17% fraud rate).

The project emphasizes **chronological data splitting**, **from-scratch SMOTE oversampling**, **business cost-sensitive decision threshold optimization**, a **FastAPI inference backend with SHAP explainability**, and a **React + Tailwind CSS merchant checkout simulator**.

---

## 🌟 Key Architectural & Design Decisions

1. **Chronological Train/Test Split (70% Train / 30% Test)**
   - Financial transactions are sequential time-series events. Standard random cross-validation or shuffled train/test splits introduce severe temporal data leakage.
   - The pipeline sorts transactions chronologically by `Time`, using the first 70% (~199k) for training and the final 30% (~85k) for testing.

2. **From-Scratch SMOTE Oversampling**
   - Implemented natively using scikit-learn's `NearestNeighbors` ($k=5$) without relying on third-party packages like `imbalanced-learn`.
   - Oversamples minority class to **10% of legitimate class count** in training set to provide strong decision boundary signals without synthetic noise.

3. **Native Histogram Gradient Boosting**
   - Employs `HistGradientBoostingClassifier` (native histogram-based gradient boosting).

4. **PR-AUC Metric Selection**
   - Evaluated using **Precision-Recall AUC (PR-AUC)** over misleading 99.83% accuracy.

5. **Business Cost-Sensitive Threshold Optimization & Model Selection**
   - Evaluated against realistic error costs ($10.00 FP/TP cost, transaction `Amount` for FN loss).
   - Sweeps decision thresholds (0.01 to 0.99) to select the model and threshold that **MINIMIZES total financial loss**.

6. **FastAPI Backend & SHAP Feature Explainability**
   - Serves low-latency real-time fraud predictions with risk level classification (`Low`, `Medium`, `High`) based on the cost-optimal decision threshold (**0.73**).
   - Computes SHAP values per prediction to return the top 5 feature contributions.

7. **React + Tailwind Merchant Risk Simulator**
   - Interactive merchant/admin dashboard with Quick-Test evaluators pre-populated from real dataset test records.
   - Live backend health indicator with auto-reconnection polling.
   - Visual risk payoff card with progress confidence meter and SHAP contribution horizontal bar chart.

---

## 📊 Benchmark & Model Comparison

| Candidate Model | PR-AUC | ROC-AUC | Optimal Threshold | Minimum Financial Cost ($) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | 0.7579 | 0.9798 | 0.99 | $4,563.95 |
| **HistGradientBoosting (Sample Weighted)** | 0.7417 | 0.9224 | 0.69 | $4,814.57 |
| **HistGradientBoosting (SMOTE)** 🏆 | **0.7749** | **0.9714** | **0.73** | **$3,371.66** |

---

## 📁 Repository Structure

```text
├── creditcard.csv                 # Raw dataset (284,807 rows)
├── requirements.txt               # Main dependencies
├── README.md                      # Documentation
├── model_training/
│   ├── train.py                   # Main pipeline script
│   ├── artifacts/                 # Saved model, scaler & metadata
│   └── reports/                   # Benchmark report & PR curve plot
├── backend/
│   ├── config.py                  # Pydantic BaseSettings config
│   ├── schemas.py                 # Pydantic data contracts
│   ├── inference.py               # Singleton FraudModel & SHAP explainer
│   ├── main.py                    # FastAPI application & endpoints
│   ├── requirements.txt           # Backend dependencies
│   └── model_artifacts/           # Local serving artifacts
└── frontend/
    ├── src/
    │   ├── api.js                 # API fetch wrappers & health check
    │   ├── presets.js             # Real transaction presets from test set
    │   ├── components/            # HealthIndicator, QuickTestButtons, Form, RiskResult
    │   └── App.jsx                # Main dashboard layout
    ├── package.json               # React + Tailwind dependencies
    └── vite.config.js             # Vite configuration
```

---

## ⚡ Real-Time API Endpoints

- **`GET /health`**: Health check (`{"status": "ok"}`).
- **`POST /predict`**: Accepts raw transaction (`Time`, `V1-V28`, `Amount`), performs server-side feature engineering (`hour_of_day`, `log_amount`), scores fraud risk using cost-optimal threshold (`0.73`), and returns top 5 SHAP feature contributions.

---

## 🚀 How to Run

1. **Train Model & Generate Artifacts**:
   ```bash
   python model_training/train.py
   ```

2. **Start FastAPI Backend**:
   ```bash
   pip install -r backend/requirements.txt
   uvicorn backend.main:app --reload --port 8000
   ```

3. **Start React Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   Open `http://localhost:3000` to interact with the merchant risk simulator dashboard.
