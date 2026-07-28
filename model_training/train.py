"""
Credit Card Fraud Detection Training Pipeline
=============================================

Key Architectural & Design Decisions:
-------------------------------------
1. Chronological Train/Test Split (70% / 30%):
   Transactions in fraud detection are time-series events. Standard random k-fold
   cross-validation or shuffled train/test splits introduce temporal data leakage,
   allowing models to learn patterns from future transactions to predict past ones.
   Sorting by Time ascending and taking the first 70% for training and the last 30%
   for testing mimics real-world production deployment (training on history, scoring future).

2. From-Scratch SMOTE Oversampling:
   Class imbalance in fraud dataset is extreme (~0.17% fraud). Rather than depending
   on heavy external packages like imbalanced-learn, SMOTE (Synthetic Minority
   Over-sampling Technique, Chawla et al. 2002) is implemented natively using
   scikit-learn's NearestNeighbors (k=5). Oversampling targets 10% fraud ratio rather
   than 50/50 balance to prevent introducing synthetic noise while providing sufficient
   gradient signal for non-linear decision boundaries. SMOTE is strictly fitted and
   applied ONLY on the training split to prevent test set data contamination.

3. Model Choice (HistGradientBoostingClassifier & LogisticRegression):
   HistGradientBoostingClassifier is scikit-learn's native implementation of histogram-based
   gradient boosted decision trees (inspired by LightGBM). It offers out-of-the-box support
   for high-throughput tabular modeling, missing value handling, fast binning, and tree
   depth control without requiring third-party C++ bindings (LightGBM/XGBoost).
   Logistic Regression with balanced class weights serves as an interpretable linear baseline.

4. Metric Selection (PR-AUC over Accuracy):
   In highly imbalanced datasets (99.83% legitimate), standard accuracy is a misleading metric;
   a naive model predicting all transactions as legitimate achieves 99.83% accuracy while
   failing to catch any fraud. Precision-Recall AUC (PR-AUC / Average Precision) is chosen as
   the primary ranking metric because it focuses directly on the minority class trade-off
   between false positives and false negatives without being inflated by true negatives.

5. Cost-Sensitive Threshold Optimization & Business-Centric Model Selection:
   Standard model evaluation uses a default classification probability threshold of 0.5.
   However, in production fraud detection, costs of classification errors are highly asymmetric:
   - False Positives (FP): Customer friction / blocking legitimate transaction (~$10 cost).
   - True Positives (TP): Investigation / review overhead (~$10 cost).
   - False Negatives (FN): Undetected fraud resulting in loss of full transaction amount.
   Instead of picking the model with the highest PR-AUC alone, this pipeline evaluates all
   trained models across a decision threshold sweep (0.01 to 0.99) against total expected financial loss.
   The final winning model is explicitly selected based on MINIMUM TOTAL COST, directly aligning
   machine learning model selection with business outcome metrics.
"""

import os
import json
import time
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
)

# Cost Constants (Illustrative/adjustable business parameters)
FP_COST = 10.0   # Cost of wrongly blocking a legitimate transaction ($)
TP_COST = 10.0   # Cost of reviewing/investigating a correctly identified fraud ($)
TN_COST = 0.0    # Cost of seamless legitimate transaction ($)
# FN_COST = Actual transaction Amount ($ lost if fraud is missed)

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["hour_of_day", "log_amount"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers domain-relevant features from raw columns:
    - hour_of_day: cyclic time of day feature derived from Time (in seconds)
    - log_amount: log1p transformed transaction amount to reduce right-skewness
    """
    df = df.copy()
    df["hour_of_day"] = (df["Time"] % 86400) // 3600
    df["log_amount"] = np.log1p(df["Amount"])
    return df


def load_and_split_data(data_path: str, train_ratio: float = 0.7):
    """
    Loads creditcard dataset, applies feature engineering, sorts chronologically by Time,
    splits into train/test (70/30), and scales features using StandardScaler fitted on train set only.
    """
    print(f"[1/6] Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"      Total dataset size: {len(df):,} rows, {len(df.columns)} columns")
    
    df = engineer_features(df)
    
    # Sort chronologically by Time
    df = df.sort_values("Time").reset_index(drop=True)
    
    n_train = int(len(df) * train_ratio)
    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()
    
    print(f"      Chronological Split -> Train: {len(train_df):,} rows ({train_ratio*100:.0f}%), Test: {len(test_df):,} rows ({(1-train_ratio)*100:.0f}%)")
    print(f"      Train Fraud count: {train_df['Class'].sum():,} ({train_df['Class'].mean()*100:.3f}%)")
    print(f"      Test Fraud count:  {test_df['Class'].sum():,} ({test_df['Class'].mean()*100:.3f}%)")
    
    X_train_raw = train_df[FEATURE_COLS].values
    y_train = train_df["Class"].values
    X_test_raw = test_df[FEATURE_COLS].values
    y_test = test_df["Class"].values
    test_amounts = test_df["Amount"].values
    
    print("      Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    return X_train, y_train, X_test, y_test, test_amounts, scaler, len(train_df), len(test_df)


def smote_oversample(X_minority: np.ndarray, n_synthetic: int, k_neighbors: int = 5, random_state: int = 42) -> np.ndarray:
    """
    Generates synthetic samples for the minority class matrix using NearestNeighbors.
    Reference: Chawla et al. (2002) SMOTE: Synthetic Minority Over-sampling Technique.
    """
    if n_synthetic <= 0:
        return np.empty((0, X_minority.shape[1]))
    
    rng = np.random.RandomState(random_state)
    
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm="auto")
    nn.fit(X_minority)
    
    # Query k nearest neighbors (col 0 is self, cols 1..k are neighbors)
    knn_indices = nn.kneighbors(X_minority, return_distance=False)[:, 1:]
    
    n_minority = len(X_minority)
    base_indices = rng.choice(n_minority, size=n_synthetic, replace=True)
    neighbor_offsets = rng.randint(0, k_neighbors, size=n_synthetic)
    gap = rng.uniform(0, 1, size=(n_synthetic, 1))
    
    selected_neighbors = knn_indices[base_indices, neighbor_offsets]
    diffs = X_minority[selected_neighbors] - X_minority[base_indices]
    synthetic_samples = X_minority[base_indices] + gap * diffs
    
    return synthetic_samples


def apply_smote(X_train: np.ndarray, y_train: np.ndarray, target_ratio: float = 0.10, random_state: int = 42):
    """
    Applies custom SMOTE oversampling to training set until minority class count equals target_ratio
    of majority class count. Shuffles combined output.
    """
    n_legit = np.sum(y_train == 0)
    n_fraud = np.sum(y_train == 1)
    target_fraud_count = int(n_legit * target_ratio)
    n_synthetic = target_fraud_count - n_fraud
    
    print(f"[2/6] Applying custom SMOTE from scratch...")
    print(f"      Original Train Class Balance -> Legit: {n_legit:,}, Fraud: {n_fraud:,} ({n_fraud/n_legit*100:.2f}% ratio)")
    
    if n_synthetic <= 0:
        print("      Fraud samples already exceed target ratio; skipping oversampling.")
        return X_train, y_train
    
    print(f"      Targeting fraud count: {target_fraud_count:,} ({target_ratio*100:.0f}% of legit class)")
    print(f"      Generating {n_synthetic:,} synthetic fraud samples...")
    
    X_minority = X_train[y_train == 1]
    synthetic_X = smote_oversample(X_minority, n_synthetic, k_neighbors=5, random_state=random_state)
    synthetic_y = np.ones(n_synthetic, dtype=y_train.dtype)
    
    X_resampled = np.vstack([X_train, synthetic_X])
    y_resampled = np.hstack([y_train, synthetic_y])
    
    rng = np.random.RandomState(random_state)
    shuffle_idx = rng.permutation(len(y_resampled))
    
    X_resampled = X_resampled[shuffle_idx]
    y_resampled = y_resampled[shuffle_idx]
    
    print(f"      Resampled Train Shape: {X_resampled.shape}, Fraud Ratio: {np.mean(y_resampled==1)*100:.2f}%")
    return X_resampled, y_resampled


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42):
    """Trains Logistic Regression baseline with balanced class weights."""
    print("      Training Model 1: LogisticRegression (class_weight='balanced')...")
    t0 = time.time()
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    print(f"      Completed in {time.time() - t0:.2f} seconds.")
    return model


def train_hgb_weighted(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42):
    """Trains HistGradientBoostingClassifier on original data with sample_weight balance."""
    print("      Training Model 2: HistGradientBoostingClassifier (sample_weight)...")
    t0 = time.time()
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    pos_weight = n_neg / n_pos
    sample_weights = np.where(y_train == 1, pos_weight, 1.0)
    
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_depth=6,
        random_state=random_state,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)
    print(f"      Completed in {time.time() - t0:.2f} seconds.")
    return model


def train_hgb_smote(X_train_smote: np.ndarray, y_train_smote: np.ndarray, random_state: int = 42):
    """Trains HistGradientBoostingClassifier on SMOTE-oversampled training data."""
    print("      Training Model 3: HistGradientBoostingClassifier (SMOTE resampled)...")
    t0 = time.time()
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_depth=6,
        random_state=random_state,
    )
    model.fit(X_train_smote, y_train_smote)
    print(f"      Completed in {time.time() - t0:.2f} seconds.")
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Computes evaluation metrics (PR-AUC, ROC-AUC, F1 @ 0.5, Confusion Matrix @ 0.5) on test set."""
    y_prob = model.predict_proba(X_test)[:, 1]
    pr_auc = float(average_precision_score(y_test, y_prob))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    
    y_pred_05 = (y_prob >= 0.5).astype(int)
    f1_05 = float(f1_score(y_test, y_pred_05))
    cm_05 = confusion_matrix(y_test, y_pred_05).tolist()
    
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1_at_0.5": f1_05,
        "confusion_matrix_at_0.5": cm_05,
        "y_prob": y_prob,
    }


def cost_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    fp_cost: float = FP_COST,
    tp_cost: float = TP_COST,
):
    """
    Sweeps probability thresholds from 0.01 to 0.99 in steps of 0.01.
    Calculates total cost = sum(FN transaction amounts) + FP_COST * num_FP + TP_COST * num_TP.
    Returns best threshold, minimum cost, default 0.5 threshold cost, and savings.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        fn_mask = (y_true == 1) & (y_pred == 0)
        fp_mask = (y_true == 0) & (y_pred == 1)
        tp_mask = (y_true == 1) & (y_pred == 1)
        
        fn_loss = np.sum(amounts[fn_mask])
        fp_loss = fp_cost * np.sum(fp_mask)
        tp_loss = tp_cost * np.sum(tp_mask)
        
        total_cost = fn_loss + fp_loss + tp_loss
        costs.append(total_cost)
        
    costs = np.array(costs)
    best_idx = np.argmin(costs)
    best_threshold = float(thresholds[best_idx])
    min_cost = float(costs[best_idx])
    
    # Calculate cost at default threshold 0.5
    idx_05 = np.argmin(np.abs(thresholds - 0.50))
    cost_at_05 = float(costs[idx_05])
    
    savings = float(cost_at_05 - min_cost)
    savings_pct = float((savings / cost_at_05) * 100) if cost_at_05 > 0 else 0.0
    
    return {
        "best_threshold": round(best_threshold, 2),
        "min_cost": round(min_cost, 2),
        "cost_at_0.5": round(cost_at_05, 2),
        "savings": round(savings, 2),
        "savings_pct": round(savings_pct, 2),
    }


def plot_pr_curve(models_eval: dict, y_test: np.ndarray, save_path: str):
    """Generates and saves PR-Curve comparison plot for all models."""
    plt.figure(figsize=(8, 6))
    
    for name, eval_data in models_eval.items():
        precision, recall, _ = precision_recall_curve(y_test, eval_data["y_prob"])
        pr_auc = eval_data["pr_auc"]
        plt.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.4f})", linewidth=2)
        
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve Comparison", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"      Saved PR Curve comparison plot to {save_path}")


def main():
    data_path = "./creditcard.csv"
    artifacts_dir = "model_training/artifacts"
    reports_dir = "model_training/reports"
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Step 1: Load and Split Data
    X_train, y_train, X_test, y_test, test_amounts, scaler, n_train, n_test = load_and_split_data(data_path)
    
    # Step 2: Apply SMOTE on Training Set
    X_train_smote, y_train_smote = apply_smote(X_train, y_train, target_ratio=0.10, random_state=42)
    
    # Step 3: Train 3 Models
    print("[3/6] Training Candidate Models...")
    models = {
        "Logistic Regression (Balanced)": train_logistic_regression(X_train, y_train, random_state=42),
        "HistGradientBoosting (Sample Weighted)": train_hgb_weighted(X_train, y_train, random_state=42),
        "HistGradientBoosting (SMOTE)": train_hgb_smote(X_train_smote, y_train_smote, random_state=42),
    }
    
    # Step 4: Evaluate Models & Run Cost Optimization
    print("[4/6] Evaluating Models & Optimizing Cost-Sensitive Thresholds...")
    results = {}
    
    for name, model in models.items():
        eval_metrics = evaluate_model(model, X_test, y_test)
        cost_metrics = cost_optimal_threshold(y_test, eval_metrics["y_prob"], test_amounts)
        
        results[name] = {
            "model": model,
            "pr_auc": eval_metrics["pr_auc"],
            "roc_auc": eval_metrics["roc_auc"],
            "f1_at_0.5": eval_metrics["f1_at_0.5"],
            "confusion_matrix_at_0.5": eval_metrics["confusion_matrix_at_0.5"],
            "best_threshold": cost_metrics["best_threshold"],
            "min_cost": cost_metrics["min_cost"],
            "cost_at_0.5": cost_metrics["cost_at_0.5"],
            "savings": cost_metrics["savings"],
            "savings_pct": cost_metrics["savings_pct"],
            "y_prob": eval_metrics["y_prob"],
        }
        
    # Step 5: Model Comparison & Winning Selection based on Minimum Financial Cost
    print("\n" + "=" * 90)
    print("MODEL COMPARISON TABLE")
    print("=" * 90)
    print(f"{'Model Name':<42} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'Opt Thresh':<10} | {'Min Cost ($)':<12}")
    print("-" * 90)
    
    winning_model_name = None
    lowest_cost = float("inf")
    
    for name, res in results.items():
        print(f"{name:<42} | {res['pr_auc']:<8.4f} | {res['roc_auc']:<8.4f} | {res['best_threshold']:<10.2f} | ${res['min_cost']:<11,.2f}")
        if res["min_cost"] < lowest_cost:
            lowest_cost = res["min_cost"]
            winning_model_name = name
            
    print("=" * 90)
    winning_res = results[winning_model_name]
    print(f"\nWINNING MODEL SELECTED: '{winning_model_name}'")
    print(f"Selection Reason: Achieves the LOWEST minimum financial cost (${winning_res['min_cost']:,.2f}) at optimal decision threshold {winning_res['best_threshold']:.2f}.")
    print(f"Cost Savings vs Default (0.5) Threshold: ${winning_res['savings']:,.2f} ({winning_res['savings_pct']:.1f}% reduction in expected loss)")
    
    # Step 6: Save Artifacts & Reports
    print("\n[5/6] Saving Artifacts & Evaluation Reports...")
    
    # Save winning model & scaler
    winning_model_obj = winning_res["model"]
    model_path = os.path.join(artifacts_dir, "model.joblib")
    scaler_path = os.path.join(artifacts_dir, "scaler.joblib")
    metadata_path = os.path.join(artifacts_dir, "metadata.json")
    
    joblib.dump(winning_model_obj, model_path)
    joblib.dump(scaler, scaler_path)
    
    metadata = {
        "model_type": winning_model_name,
        "feature_cols": FEATURE_COLS,
        "cost_optimal_threshold": winning_res["best_threshold"],
        "train_rows": n_train,
        "test_rows": n_test,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"      Saved {model_path}")
    print(f"      Saved {scaler_path}")
    print(f"      Saved {metadata_path}")
    
    # Save Report & PR Curve
    report_path = os.path.join(reports_dir, "evaluation_report.json")
    pr_plot_path = os.path.join(reports_dir, "pr_curve_comparison.png")
    
    report_data = {
        "cost_assumptions": {
            "FP_COST": FP_COST,
            "TP_COST": TP_COST,
            "FN_COST": "actual_transaction_amount",
            "TN_COST": TN_COST,
        },
        "all_models_metrics": {
            name: {
                "pr_auc": res["pr_auc"],
                "roc_auc": res["roc_auc"],
                "f1_at_0.5": res["f1_at_0.5"],
                "confusion_matrix_at_0.5": res["confusion_matrix_at_0.5"],
                "optimal_threshold": res["best_threshold"],
                "min_cost": res["min_cost"],
                "cost_at_0.5": res["cost_at_0.5"],
                "savings": res["savings"],
                "savings_pct": res["savings_pct"],
            }
            for name, res in results.items()
        },
        "winning_model": {
            "name": winning_model_name,
            "best_threshold": winning_res["best_threshold"],
            "cost_at_best_threshold": winning_res["min_cost"],
            "cost_at_default_threshold_0.5": winning_res["cost_at_0.5"],
            "estimated_savings": winning_res["savings"],
            "savings_pct": winning_res["savings_pct"],
        },
    }
    
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"      Saved {report_path}")
    
    plot_pr_curve(results, y_test, pr_plot_path)
    
    print("\n[6/6] Execution completed successfully!")


if __name__ == "__main__":
    main()
