import json
import os
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
DATA_FILE = "Traffic_Collisions_Updated.csv"
ARTIFACTS_DIR = "artifacts"


def load_data(path: str):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"  {len(df)} rows, {df.shape[1]} columns")

    drop_cols = ["CRASH", "ACCIDENTDATE", "zone_id", "Unnamed: 0"]
    # Convert ACCIDENTDATE to a numeric ordinal so the model can use it
    if "ACCIDENTDATE" in df.columns:
        df["ACCIDENTDATE"] = pd.to_datetime(df["ACCIDENTDATE"], errors="coerce").map(
            lambda d: d.toordinal() if pd.notna(d) else 0
        )

    # One-hot encode zone_id so the model can use zone identity as a feature
    if "zone_id" in df.columns:
        df = pd.get_dummies(df, columns=["zone_id"], prefix="zone")

    drop_cols = ["CRASH", "Unnamed: 0"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["CRASH"]

    print(f"  Features: {X.shape[1]}  |  Target distribution: {y.value_counts().to_dict()}")
    return X, y


def evaluate(name: str, model, X_test, y_test, X_train=None, y_train=None):
    print(f"\n--- {name} ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")

    cv_mean, cv_std = None, None
    if X_train is not None and y_train is not None:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
        print(f"  CV ROC-AUC (5-fold): {cv_mean:.4f} ± {cv_std:.4f}")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Crash", "Crash"]))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return acc, auc, cv_mean, cv_std


def tune_model(name: str, model, param_dist: dict, X_train, y_train, X_test, y_test, n_iter=20):
    print(f"\nTuning {name} ({n_iter} iterations, 3-fold CV)...")
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_

    print(f"  Best params: {search.best_params_}")
    y_prob = best.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, best.predict(X_test))
    print(f"  Tuned Accuracy : {acc:.4f}")
    print(f"  Tuned ROC-AUC  : {auc:.4f}")

    return best, acc, auc, search.best_params_


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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    all_results = {}

    # ------------------------------------------------------------------ #
    # 1. BASELINE MODELS                                                   #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("BASELINE MODELS")
    print("=" * 60)

    # Logistic Regression — tune C and l1_ratio (0=l2, 1=l1)
    lr_base = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver="saga")
    lr_best, lr_acc, lr_auc, lr_params = tune_model(
        "LogisticRegression", lr_base,
        {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "l1_ratio": [0.0, 0.5, 1.0],
        },
        X_train_scaled, y_train, X_test_scaled, y_test,
        n_iter=12,
    )
    cv_scores = cross_val_score(lr_best, X_train_scaled, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
    print(f"  CV ROC-AUC (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    all_results["LogisticRegression"] = {
        "accuracy": lr_acc, "roc_auc": lr_auc,
        "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
        "best_params": lr_params,
    }

    # Decision Tree — tune depth and criterion
    dt_base = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt_best, dt_acc, dt_auc, dt_params = tune_model(
        "DecisionTree", dt_base,
        {
            "max_depth": [3, 5, 7, 10, None],
            "criterion": ["gini", "entropy"],
            "min_samples_leaf": [1, 2, 4],
        },
        X_train, y_train, X_test, y_test,
        n_iter=15,
    )
    cv_scores = cross_val_score(dt_best, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
    print(f"  CV ROC-AUC (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    all_results["DecisionTree"] = {
        "accuracy": dt_acc, "roc_auc": dt_auc,
        "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
        "best_params": dt_params,
    }

    # ------------------------------------------------------------------ #
    # 2. ADVANCED MODELS                                                   #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("ADVANCED MODELS")
    print("=" * 60)

    # Random Forest — tune
    rf_base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    rf_best, rf_acc, rf_auc, rf_params = tune_model(
        "RandomForest", rf_base,
        {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5],
        },
        X_train, y_train, X_test, y_test,
    )
    cv_scores = cross_val_score(rf_best, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
    print(f"  CV ROC-AUC (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    all_results["RandomForest"] = {
        "accuracy": rf_acc, "roc_auc": rf_auc,
        "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
        "best_params": rf_params,
    }

    # Gradient Boosting — tune
    gb_base = GradientBoostingClassifier(random_state=RANDOM_STATE)
    gb_best, gb_acc, gb_auc, gb_params = tune_model(
        "GradientBoosting", gb_base,
        {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 4, 5, 6],
            "subsample": [0.6, 0.8, 1.0],
            "min_samples_leaf": [1, 2, 4],
        },
        X_train, y_train, X_test, y_test,
    )
    cv_scores = cross_val_score(gb_best, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
    print(f"  CV ROC-AUC (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    all_results["GradientBoosting"] = {
        "accuracy": gb_acc, "roc_auc": gb_auc,
        "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
        "best_params": gb_params,
    }

    # Hist Gradient Boosting (sklearn XGBoost equivalent)
    print("\nTraining HistGradientBoosting...")
    hgb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE
    )
    hgb.fit(X_train, y_train)
    hgb_acc, hgb_auc, hgb_cv_mean, hgb_cv_std = evaluate(
        "HistGradientBoosting", hgb, X_test, y_test, X_train, y_train
    )
    all_results["HistGradientBoosting"] = {
        "accuracy": hgb_acc, "roc_auc": hgb_auc,
        "cv_mean": hgb_cv_mean, "cv_std": hgb_cv_std,
        "best_params": {"max_iter": 300, "learning_rate": 0.05, "max_depth": 4},
    }

    print_feature_importance(rf_best, list(X.columns))

    # ------------------------------------------------------------------ #
    # 3. SUMMARY COMPARISON                                                #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n  {'Model':<28} {'Accuracy':>10} {'ROC-AUC':>10} {'CV Mean':>10} {'CV Std':>8}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for name, m in all_results.items():
        cv_m = f"{m['cv_mean']:.4f}" if m["cv_mean"] else "   N/A"
        cv_s = f"{m['cv_std']:.4f}" if m["cv_std"] else "   N/A"
        print(f"  {name:<28} {m['accuracy']:>10.4f} {m['roc_auc']:>10.4f} {cv_m:>10} {cv_s:>8}")

    # ------------------------------------------------------------------ #
    # 4. SAVE BEST OVERALL MODEL                                           #
    # ------------------------------------------------------------------ #
    trained_map = {
        "LogisticRegression": lr_best,
        "DecisionTree": dt_best,
        "RandomForest": rf_best,
        "GradientBoosting": gb_best,
        "HistGradientBoosting": hgb,
    }

    best_name = max(all_results, key=lambda n: all_results[n]["roc_auc"])
    best_model = trained_map[best_name]
    best_metrics = all_results[best_name]
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
        "all_results": {
            k: {
                "accuracy": round(v["accuracy"], 6),
                "roc_auc": round(v["roc_auc"], 6),
                "cv_mean": round(v["cv_mean"], 6) if v["cv_mean"] else None,
                "cv_std": round(v["cv_std"], 6) if v["cv_std"] else None,
                "best_params": v["best_params"],
            }
            for k, v in all_results.items()
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved model      -> {model_path}")
    print(f"Saved features   -> {features_path}")
    print(f"Saved summary    -> {summary_path}")


if __name__ == "__main__":
    main()
