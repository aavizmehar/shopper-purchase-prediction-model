"""
Train, compare, tune and save the Online Shopper Purchase Intention model.

Dataset : UCI "Online Shoppers Purchasing Intention Dataset" (Sakar et al., 2019)
          https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
Task    : Binary classification - will a browsing session end in a purchase (Revenue)?

Usage:
    python train_model.py                      # uses data/dataset.csv
    python train_model.py --data path/to/file  # custom path

Outputs:
    model.pkl            - the complete fitted Pipeline (preprocessing + estimator)
    reports/metrics.json - comparison, CV, tuning and final test results
    reports/*.png        - plots used in the report
"""
import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET = "Revenue"

NUMERIC = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]
# Integer-coded but NOT ordered quantities (e.g. Browser 2 is not "twice" Browser 1),
# so they are one-hot encoded together with the text columns.
CATEGORICAL = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType"]
BINARY = ["Weekend"]
# This is also the column order the Streamlit app must use.
FEATURES = NUMERIC + CATEGORICAL + BINARY


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"Dataset not found at {path}.\n"
            "Download online_shoppers_intention.csv from\n"
            "https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset\n"
            "and save it as data/dataset.csv. See README.md."
        )
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing expected columns {missing}. Found: {list(df.columns)}")
    df = df[FEATURES + [TARGET]].copy()
    df["Weekend"] = df["Weekend"].astype(int)
    df[TARGET] = df[TARGET].astype(int)
    return df


def metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def make_pipeline(model) -> Pipeline:
    """Scaling + one-hot encoding + estimator in one object, so the same steps run in
    training, cross-validation, testing and the app."""
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("bin", "passthrough", BINARY),
        ]
    )
    return Pipeline([("preprocess", pre), ("model", model)])


def main(data_path: str, out_model: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load and check the data -------------------------------------------------
    df = load_data(Path(data_path))
    checks = {
        "rows": int(len(df)),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "purchase_rate": float(df[TARGET].mean()),
    }
    # Exact duplicate rows are removed: copies of one session landing in both the training
    # and the test set would make the test score look better than it really is.
    df = df.drop_duplicates().dropna()
    checks["rows_after_cleaning"] = int(len(df))
    checks["purchase_rate_after_cleaning"] = float(df[TARGET].mean())
    print("Data checks:", checks)

    X, y = df[FEATURES], df[TARGET]

    # ---- 2. Stratified train / test split (test set locked away until the end) ------
    # Only ~15% of sessions end in a purchase, so stratify keeps that share in both parts.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    # ---- 3. Compare four algorithms with stratified 5-fold CV on the training data --
    # class_weight="balanced" makes mistakes on the rare "purchase" class cost more.
    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
    }
    grids = {
        "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10]},
        "KNN": {"model__n_neighbors": [5, 11, 21, 41], "model__weights": ["uniform", "distance"]},
        "Decision Tree": {
            "model__max_depth": [3, 5, 8, 12],
            "model__min_samples_leaf": [1, 10, 30],
        },
        "Random Forest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_leaf": [1, 3, 5],
        },
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    comparison = {}
    for name, model in candidates.items():
        pipe = make_pipeline(model)
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        pipe.fit(X_train, y_train)
        row = {f"cv_{m}_mean": float(res[f"test_{m}"].mean()) for m in scoring}
        row["cv_f1_std"] = float(res["test_f1"].std())
        row["train_f1"] = float(f1_score(y_train, pipe.predict(X_train)))  # vs cv_f1 -> overfitting
        comparison[name] = row
        print(f"{name:20s} CV F1 = {row['cv_f1_mean']:.3f} +/- {row['cv_f1_std']:.3f} | "
              f"CV AUC = {row['cv_roc_auc_mean']:.3f} | CV recall = {row['cv_recall_mean']:.3f} | "
              f"train F1 = {row['train_f1']:.3f}")

    # ---- 4. Tune the best model with GridSearchCV -----------------------------------
    # Accuracy would be misleading (always answering "no purchase" scores ~85%), so the
    # model is picked and tuned on F1 of the purchase class; ROC-AUC is reported alongside.
    # Read the table above before accepting the automatic choice: if two models are close,
    # prefer the simpler one and explain why in the report.
    chosen = max(comparison, key=lambda n: comparison[n]["cv_f1_mean"])
    print(f"Best CV F1: {chosen} -> tuning it with GridSearchCV")
    grid = GridSearchCV(
        make_pipeline(clone(candidates[chosen])),
        grids[chosen],
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print("Best params:", grid.best_params_)
    print(f"Best CV F1: {grid.best_score_:.3f}")

    # ---- 5. Final evaluation on the untouched test set ------------------------------
    best_pipeline = grid.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = metrics(y_test, y_pred, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    print("Final test metrics:", {k: round(v, 3) for k, v in test_metrics.items()})
    print("Confusion matrix [[TN FP] [FN TP]]:", cm.tolist())

    baseline_test = {}
    for name, model in candidates.items():
        p = make_pipeline(model).fit(X_train, y_train)
        baseline_test[name] = metrics(y_test, p.predict(X_test), p.predict_proba(X_test)[:, 1])

    # ---- 6. Save the complete fitted pipeline ---------------------------------------
    with open(out_model, "wb") as f:
        pickle.dump(best_pipeline, f)
    print(f"Saved pipeline to {out_model}")

    ranges = {c: [float(X[c].min()), float(X[c].max())] for c in NUMERIC}
    results = {
        "data_checks": checks,
        "features": FEATURES,
        "target": TARGET,
        "feature_ranges": ranges,
        "cv_comparison": comparison,
        "untuned_test_metrics": baseline_test,
        "majority_class_accuracy": float(1 - y_test.mean()),
        "chosen_model": chosen,
        "grid_search": {
            "param_grid": {k: [str(v) for v in vs] for k, vs in grids[chosen].items()},
            "best_params": {k: str(v) for k, v in grid.best_params_.items()},
            "best_cv_f1": float(grid.best_score_),
            "n_fits": int(len(grid.cv_results_["params"]) * 5),
        },
        "final_test_metrics": test_metrics,
        "confusion_matrix": {"TN": int(cm[0, 0]), "FP": int(cm[0, 1]),
                             "FN": int(cm[1, 0]), "TP": int(cm[1, 1])},
    }
    (out / "metrics.json").write_text(json.dumps(results, indent=2))

    # ---- 7. Plots for the report ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["No purchase", "Purchase"], ax=ax, colorbar=False
    )
    ax.set_title(f"{chosen}: confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name=chosen)
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.legend()
    ax.set_title("ROC curve (test set)")
    fig.tight_layout()
    fig.savefig(out / "roc_curve.png", dpi=150)
    plt.close(fig)

    names = list(comparison)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - 0.2, [comparison[n]["cv_f1_mean"] for n in names], 0.4, label="CV F1 (purchase)")
    ax.bar(x + 0.2, [comparison[n]["cv_roc_auc_mean"] for n in names], 0.4, label="CV ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison (5-fold CV on training data)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=150)
    plt.close(fig)

    # Permutation importance works for any model and reports per ORIGINAL column:
    # how much does the test F1 drop when one feature's values are shuffled?
    pi = permutation_importance(
        best_pipeline, X_test, y_test, scoring="f1", n_repeats=5,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    order = np.argsort(pi.importances_mean)[-10:]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(np.array(FEATURES)[order], pi.importances_mean[order])
    ax.set_xlabel("Drop in test F1 when shuffled")
    ax.set_title("Top 10 features (permutation importance)")
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=150)
    plt.close(fig)
    print("Saved metrics.json and plots to", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.csv")
    ap.add_argument("--model-out", default="model.pkl")
    ap.add_argument("--reports-dir", default="reports")
    a = ap.parse_args()
    main(a.data, a.model_out, a.reports_dir)
