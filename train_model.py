import json
import os
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split

RANDOM_STATE = 42
DATA_FILE = "Traffic_Collisions_Updated.csv"
ARTIFACTS_DIR = "artifacts"


def load_data(path: str):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"  {len(df)} rows, {df.shape[1]} columns")

    if "ACCIDENTDATE" in df.columns:
        df["ACCIDENTDATE"] = pd.to_datetime(df["ACCIDENTDATE"], errors="coerce").map(
            lambda d: d.toordinal() if pd.notna(d) else 0
        )

    if "zone_id" in df.columns:
        df = pd.get_dummies(df, columns=["zone_id"], prefix="zone")

    drop_cols = ["CRASH", "Unnamed: 0"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["CRASH"]

    print(f"  Features: {X.shape[1]}  |  Target distribution: {y.value_counts().to_dict()}")
    return X, y


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    X, y = load_data(DATA_FILE)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain: {len(X_train)} rows  |  Test: {len(X_test)} rows")

    # ------------------------------------------------------------------ #
    # GRADIENT BOOSTING — hyperparameter tuning via RandomizedSearchCV     #
    # ------------------------------------------------------------------ #
    print("\nTuning GradientBoosting (20 iterations, 3-fold CV)...")
    search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        param_distributions={
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 4, 5, 6],
            "subsample": [0.6, 0.8, 1.0],
            "min_samples_leaf": [1, 2, 4],
        },
        n_iter=20,
        scoring="roc_auc",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print(f"  Best params: {search.best_params_}")

    # ------------------------------------------------------------------ #
    # EVALUATION                                                           #
    # ------------------------------------------------------------------ #
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)

    print(f"\n  Accuracy            : {acc:.4f}")
    print(f"  ROC-AUC             : {auc:.4f}")
    print(f"  CV ROC-AUC (5-fold) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Crash", "Crash"]))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ------------------------------------------------------------------ #
    # SAVE                                                                 #
    # ------------------------------------------------------------------ #
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, "collision_model.pkl"))
    joblib.dump(list(X.columns), os.path.join(ARTIFACTS_DIR, "model_features.pkl"))

    summary = {
        "model_type": "GradientBoosting",
        "best_params": search.best_params_,
        "roc_auc": round(auc, 6),
        "accuracy": round(acc, 6),
        "cv_mean": round(cv_scores.mean(), 6),
        "cv_std": round(cv_scores.std(), 6),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X.shape[1],
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(ARTIFACTS_DIR, "model_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved model    -> artifacts/collision_model.pkl")
    print(f"Saved features -> artifacts/model_features.pkl")
    print(f"Saved summary  -> artifacts/model_summary.json")


if __name__ == "__main__":
    main()
