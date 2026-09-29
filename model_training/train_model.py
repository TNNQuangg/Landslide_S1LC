import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance


FEATURE_COLUMNS = [
    "slope",
    "elevation",
    "rain_mean_year",
    "soil_type",
    "dist_river",
    "rain_forecast_24h",
]
NUMERIC_COLUMNS = [
    "slope",
    "elevation",
    "rain_mean_year",
    "dist_river",
    "rain_forecast_24h",
]
CATEGORICAL_COLUMNS = ["soil_type"]
TARGET_COLUMN = "label"
REQUIRED_COLUMNS = ["id", *FEATURE_COLUMNS, TARGET_COLUMN]


def build_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def file_sha256(path):
    sha256 = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_training_data(path):
    df = pd.read_csv(path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    df = df.copy()
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df.loc[df["rain_mean_year"] == 0, "rain_mean_year"] = np.nan
    df["soil_type"] = df["soil_type"].astype(str).str.strip()
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing or non-numeric values.")

    labels = set(df[TARGET_COLUMN].astype(int).unique())
    if not labels.issubset({0, 1}):
        raise ValueError("Target column must contain only labels 0 and 1.")

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def build_pipeline(random_state):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_COLUMNS),
            ("cat", build_one_hot_encoder(), CATEGORICAL_COLUMNS),
        ]
    )

    classifier = HistGradientBoostingClassifier(
        max_iter=300,
        random_state=random_state,
        class_weight="balanced"
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def summarize_feature_importance(pipeline, X_test, y_test):
    result = permutation_importance(pipeline, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = NUMERIC_COLUMNS + [
            f"soil_type_{value}"
            for value in preprocessor.named_transformers_["cat"].categories_[0]
        ]

    cleaned_names = [
        name.replace("num__", "").replace("cat__", "")
        for name in feature_names
    ]

    importances = result.importances_mean
    return [
        {"feature": feature, "importance": round(float(importance), 6)}
        for feature, importance in sorted(
            zip(cleaned_names, importances),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def train_and_evaluate(data_path, model_out, metrics_out, test_size, random_state):
    df = load_training_data(data_path)
    features = df[FEATURE_COLUMNS]
    target = df[TARGET_COLUMN]

    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    pipeline = build_pipeline(random_state)
    pipeline.fit(train_features, train_target)

    test_prediction = pipeline.predict(test_features)
    test_probability = pipeline.predict_proba(test_features)[:, 1]

    # Tính ngưỡng rủi ro 
    precisions, recalls, thresholds = precision_recall_curve(test_target, test_probability)
    fscores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(fscores)
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    computed_risk_thresholds = {
        "low": round(float(optimal_threshold * 0.5), 3),
        "high": round(float(optimal_threshold), 3)
    }

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        build_pipeline(random_state),
        features,
        target,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    final_pipeline = build_pipeline(random_state)
    final_pipeline.fit(features, target)

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "data_hash": file_sha256(data_path),
        "rows": int(len(df)),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "label_counts": {
            str(label): int(count)
            for label, count in target.value_counts().sort_index().items()
        },
        "risk_thresholds": computed_risk_thresholds,
        "test": {
            "test_size": test_size,
            "accuracy": round(float(accuracy_score(test_target, test_prediction)), 6),
            "precision": round(float(precision_score(test_target, test_prediction, zero_division=0)), 6),
            "recall": round(float(recall_score(test_target, test_prediction, zero_division=0)), 6),
            "f1": round(float(f1_score(test_target, test_prediction, zero_division=0)), 6),
            "roc_auc": round(float(roc_auc_score(test_target, test_probability)), 6),
            "confusion_matrix": confusion_matrix(test_target, test_prediction).tolist(),
        },
        "cross_validation": {
            metric.replace("test_", ""): {
                "scores": [round(float(score), 6) for score in values],
                "mean": round(float(np.mean(values)), 6),
                "std": round(float(np.std(values)), 6),
            }
            for metric, values in cv_results.items()
            if metric.startswith("test_")
        },
        "feature_importance": summarize_feature_importance(pipeline, test_features, test_target),
        "notes": [
            "rain_mean_year values equal to 0 are treated as missing and imputed.",
            "rain_forecast_24h values equal to 0 are kept as no-rain observations.",
            "Upgraded to HistGradientBoostingClassifier for higher speed and performance.",
        ],
    }

    artifact = {
        "pipeline": final_pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "target_column": TARGET_COLUMN,
        "risk_thresholds": computed_risk_thresholds,
        "soil_labels": sorted(df["soil_type"].dropna().unique().tolist()),
        "metrics": metrics,
        "trained_at": metrics["trained_at"],
        "data_hash": metrics["data_hash"],
    }

    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_out)
    metrics_out.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train the landslide risk model.")
    parser.add_argument("--data", default=Path("../data/Book1.csv"), type=Path)
    parser.add_argument("--model-out", default=Path("../models/landslide_pipeline.joblib"), type=Path)
    parser.add_argument("--metrics-out", default=Path("../models/model_metrics.json"), type=Path)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--random-state", default=42, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = train_and_evaluate(
        data_path=args.data,
        model_out=args.model_out,
        metrics_out=args.metrics_out,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(f"Saved model to {args.model_out}")
    print(f"Saved metrics to {args.metrics_out}")
    print(
        "Test metrics: "
        f"accuracy={metrics['test']['accuracy']}, "
        f"f1={metrics['test']['f1']}, "
        f"roc_auc={metrics['test']['roc_auc']}"
    )


if __name__ == "__main__":
    main()
