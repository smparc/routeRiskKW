import json
import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
DATA_FILE = "Traffic_Collisions_Updated.csv"
ARTIFACTS_DIR = "artifacts"


def load_data(path: str):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"  {len(df)} rows, {df.shape[1]} columns")

    drop_cols = ["CRASH", "ACCIDENTDATE", "zone_id"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["CRASH"]

    print(f"  Features: {X.shape[1]}  |  Target distribution: {y.value_counts().to_dict()}")
    return X, y


def evaluate(name: str, model, X_test, y_test):
    print(f"\n--- {name} ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Crash", "Crash"]))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return acc, auc


def print_feature_importance(model, feature_names, top_n=15):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=False).head(top_n)
    print(f"\nTop {top_n} Feature Importances (Random Forest):")
    for feat, score in top.items():
        print(f"  {feat:<45} {score:.4f}")


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    X, y = load_data(DATA_FILE)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain: {len(X_train)} rows  |  Test: {len(X_test)} rows")

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
    }

    results = {}
    trained = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        acc, auc = evaluate(name, model, X_test, y_test)
        results[name] = {"accuracy": acc, "roc_auc": auc}
        trained[name] = model

    print_feature_importance(trained["RandomForest"], list(X.columns))

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_model = trained[best_name]
    best_metrics = results[best_name]
    print(f"\nBest model: {best_name} (ROC-AUC = {best_metrics['roc_auc']:.4f})")

    model_path = os.path.join(ARTIFACTS_DIR, "collision_model.pkl")
    features_path = os.path.join(ARTIFACTS_DIR, "model_features.pkl")
    summary_path = os.path.join(ARTIFACTS_DIR, "model_summary.json")

    joblib.dump(best_model, model_path)
    joblib.dump(list(X.columns), features_path)

    summary = {
        "model_type": best_name,
        "roc_auc": round(best_metrics["roc_auc"], 6),
        "accuracy": round(best_metrics["accuracy"], 6),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X.shape[1],
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved model      -> {model_path}")
    print(f"Saved features   -> {features_path}")
    print(f"Saved summary    -> {summary_path}")


if __name__ == "__main__":
    main()
