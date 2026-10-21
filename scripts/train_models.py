"""Placeholder module: scripts/train_models.py; implementation required."""

#!/usr/bin/env python3
"""
Kerala Flood Relief Coordination Agent
backend/scripts/train_models.py

Train reproducible DEMONSTRATION machine-learning models.

Models:
    1. Flood-risk classification
    2. Shelter occupancy regression
    3. Relief inventory shortage classification

Data source:
    JSON files from generate_demo_data.py

Safety:
    - Synthetic datasets only
    - No live weather or emergency feeds
    - No operational predictions or dispatch decisions
    - Model outputs must not be used for public warnings

Usage:
    python backend/scripts/train_models.py

    python backend/scripts/train_models.py \
        --input data/demo \
        --output models/trained \
        --seed 42

    python backend/scripts/train_models.py \
        --model flood

    python backend/scripts/train_models.py \
        --model shelter

    python backend/scripts/train_models.py \
        --model inventory

Dependencies:
    pip install numpy pandas scikit-learn joblib
"""

from __future__ import annotations

import argparse
import json
import math
import sys

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT = "Kerala Flood Relief Coordination Agent"
VERSION = "1.0.0"

SUPPORTED_MODELS = {
    "all",
    "flood",
    "shelter",
    "inventory",
}

FLOOD_FEATURES = [
    "rainfall_24h_mm",
    "river_pressure_index",
    "terrain_index",
    "exposure_index",
]

SHELTER_NUMERIC_FEATURES = [
    "capacity",
]

SHELTER_CATEGORICAL_FEATURES = [
    "district",
    "water_status",
    "food_status",
    "medical_status",
    "accessible",
]

INVENTORY_NUMERIC_FEATURES = [
    "quantity",
    "reserved",
]

INVENTORY_CATEGORICAL_FEATURES = [
    "district",
    "resource",
    "unit",
]

MIN_ROWS = 8


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )


def to_builtin(value: Any) -> Any:
    """Convert NumPy objects to JSON-compatible values."""
    if isinstance(value, dict):
        return {
            str(k): to_builtin(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            to_builtin(data),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_dataset(
    directory: Path,
    name: str,
) -> pd.DataFrame:
    path = directory / f"{name}.json"

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Run generate_demo_data.py first."
        )

    if path.stat().st_size > 50 * 1024 * 1024:
        raise ValueError(
            f"Dataset exceeds size limit: {name}"
        )

    with path.open("r", encoding="utf-8-sig") as file:
        records = json.load(file)

    if not isinstance(records, list) or not records:
        raise ValueError(
            f"{name}: expected a non-empty JSON array."
        )

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                f"{name}[{index}]: expected an object."
            )

        if (
            record.get("synthetic") is not True
            or record.get("operational_use") is not False
        ):
            raise ValueError(
                f"{name}[{index}]: "
                "only synthetic development data is accepted."
            )

    return pd.DataFrame(records)


def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    dataset: str,
) -> None:
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset}: missing columns {missing}"
        )


def numeric_frame(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    result = frame[columns].copy()

    for column in columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    return result


def make_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
        ),
    ])

    return ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ])


def split_classification(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
):
    counts = y.value_counts()

    stratify = (
        y
        if len(counts) > 1
        and counts.min() >= 2
        and math.ceil(len(y) * 0.25) >= len(counts)
        else None
    )

    return train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=stratify,
    )


def classification_metrics(
    y_true,
    y_pred,
) -> dict:
    return {
        "accuracy": round(
            accuracy_score(y_true, y_pred), 4
        ),
        "f1_macro": round(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            4,
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
        ).tolist(),
    }


def regression_metrics(
    y_true,
    y_pred,
) -> dict:
    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    return {
        "mae": round(
            mean_absolute_error(
                y_true,
                y_pred,
            ),
            4,
        ),
        "rmse": round(
            math.sqrt(mse),
            4,
        ),
        "r2": round(
            r2_score(
                y_true,
                y_pred,
            ),
            4,
        ),
    }


def classification_cv(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
) -> dict:
    counts = y.value_counts()

    if len(counts) < 2 or counts.min() < 2:
        return {
            "performed": False,
            "reason": "Insufficient samples per class.",
        }

    folds = min(
        5,
        int(counts.min()),
    )

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,
    )

    return {
        "performed": True,
        "folds": folds,
        "f1_macro_mean": round(
            float(scores.mean()), 4
        ),
        "f1_macro_std": round(
            float(scores.std()), 4
        ),
    }


def regression_cv(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
) -> dict:
    if len(X) < 10:
        return {
            "performed": False,
            "reason": "Insufficient rows for CV.",
        }

    folds = min(5, len(X))

    cv = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )

    return {
        "performed": True,
        "folds": folds,
        "mae_mean": round(
            float(-scores.mean()), 4
        ),
        "mae_std": round(
            float(scores.std()), 4
        ),
    }


def persist_model(
    model: Pipeline,
    output: Path,
    filename: str,
    metadata: dict,
) -> None:
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = output / f"{filename}.joblib"
    metadata_path = output / f"{filename}_metadata.json"

    joblib.dump(
        model,
        model_path,
        compress=3,
    )

    save_json(
        metadata_path,
        metadata,
    )

    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")


# ============================================================
# MODEL 1 — FLOOD RISK CLASSIFICATION
# ============================================================

def train_flood_model(
    input_dir: Path,
    output_dir: Path,
    seed: int,
) -> dict:
    print("\n[1] Training flood-risk classifier")

    frame = load_dataset(
        input_dir,
        "districts",
    )

    require_columns(
        frame,
        FLOOD_FEATURES + ["risk_level"],
        "districts",
    )

    X = numeric_frame(
        frame,
        FLOOD_FEATURES,
    )

    y = frame["risk_level"].astype(str)

    if len(frame) < MIN_ROWS:
        raise ValueError(
            "Not enough flood-risk samples."
        )

    if y.nunique() < 2:
        raise ValueError(
            "Flood-risk training needs at least two classes."
        )

    pipeline = Pipeline([
        (
            "preprocessor",
            make_preprocessor(
                FLOOD_FEATURES,
                [],
            ),
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
    ])

    X_train, X_test, y_train, y_test = (
        split_classification(X, y, seed)
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(X_test)

    metrics = classification_metrics(
        y_test,
        predictions,
    )

    # CV uses training rows only.
    metrics["cross_validation"] = classification_cv(
        pipeline,
        X_train,
        y_train,
        seed,
    )

    # Persist a model trained on all available demo rows.
    pipeline.fit(X, y)

    metadata = {
        "model_name": "flood_risk_classifier",
        "task": "classification",
        "algorithm": "RandomForestClassifier",
        "features": FLOOD_FEATURES,
        "target": "risk_level",
        "training_rows": len(frame),
        "holdout_rows": len(X_test),
        "metrics": metrics,
        "dataset": "districts.json",
        "label_source": (
            "Synthetic heuristic risk score from "
            "generate_demo_data.py."
        ),
        "evaluation_limitation": (
            "Only 14 district records; synthetic "
            "labels depend on input features. "
            "No real-world predictive validity."
        ),
        "created_at": timestamp(),
        "synthetic_training": True,
        "operational_use": False,
    }

    persist_model(
        pipeline,
        output_dir,
        "flood_risk_classifier",
        metadata,
    )

    print(
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"F1 macro: {metrics['f1_macro']:.4f}"
    )

    return metadata


# ============================================================
# MODEL 2 — SHELTER OCCUPANCY REGRESSION
# ============================================================

def train_shelter_model(
    input_dir: Path,
    output_dir: Path,
    seed: int,
) -> dict:
    print("\n[2] Training shelter-occupancy regressor")

    frame = load_dataset(
        input_dir,
        "shelters",
    )

    features = (
        SHELTER_NUMERIC_FEATURES
        + SHELTER_CATEGORICAL_FEATURES
    )

    require_columns(
        frame,
        features + ["occupancy_percent"],
        "shelters",
    )

    X = frame[features].copy()

    for column in SHELTER_NUMERIC_FEATURES:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    for column in SHELTER_CATEGORICAL_FEATURES:
        X[column] = X[column].astype(str)

    y = pd.to_numeric(
        frame["occupancy_percent"],
        errors="coerce",
    )

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    if len(X) < MIN_ROWS:
        raise ValueError(
            "Not enough shelter training records."
        )

    pipeline = Pipeline([
        (
            "preprocessor",
            make_preprocessor(
                SHELTER_NUMERIC_FEATURES,
                SHELTER_CATEGORICAL_FEATURES,
            ),
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=-1,
            ),
        ),
    ])

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=seed,
        )
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = np.clip(
        pipeline.predict(X_test),
        0,
        100,
    )

    metrics = regression_metrics(
        y_test,
        predictions,
    )

    metrics["cross_validation"] = regression_cv(
        pipeline,
        X_train,
        y_train,
        seed,
    )

    pipeline.fit(X, y)

    metadata = {
        "model_name": "shelter_occupancy_regressor",
        "task": "regression",
        "algorithm": "RandomForestRegressor",
        "features": features,
        "target": "occupancy_percent",
        "training_rows": len(X),
        "holdout_rows": len(X_test),
        "metrics": metrics,
        "dataset": "shelters.json",
        "evaluation_limitation": (
            "Synthetic occupancy is randomly generated. "
            "Features may have little genuine relationship "
            "with occupancy. This is not a time-series "
            "occupancy forecast."
        ),
        "created_at": timestamp(),
        "synthetic_training": True,
        "operational_use": False,
    }

    persist_model(
        pipeline,
        output_dir,
        "shelter_occupancy_regressor",
        metadata,
    )

    print(
        f"MAE: {metrics['mae']:.3f} | "
        f"RMSE: {metrics['rmse']:.3f} | "
        f"R²: {metrics['r2']:.3f}"
    )

    return metadata


# ============================================================
# MODEL 3 — INVENTORY SHORTAGE CLASSIFICATION
# ============================================================

def train_inventory_model(
    input_dir: Path,
    output_dir: Path,
    seed: int,
    shortage_threshold: int,
) -> dict:
    print("\n[3] Training inventory-shortage classifier")

    frame = load_dataset(
        input_dir,
        "relief_inventory",
    )

    features = (
        INVENTORY_NUMERIC_FEATURES
        + INVENTORY_CATEGORICAL_FEATURES
    )

    require_columns(
        frame,
        features,
        "relief_inventory",
    )

    X = frame[features].copy()

    for column in INVENTORY_NUMERIC_FEATURES:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    for column in INVENTORY_CATEGORICAL_FEATURES:
        X[column] = X[column].astype(str)

    available = (
        X["quantity"] - X["reserved"]
    )

    valid = (
        X["quantity"].notna()
        & X["reserved"].notna()
        & (X["quantity"] >= 0)
        & (X["reserved"] >= 0)
        & (available >= 0)
    )

    X = X.loc[valid]
    available = available.loc[valid]

    y = (
        available < shortage_threshold
    ).astype(int)

    if len(X) < MIN_ROWS:
        raise ValueError(
            "Not enough inventory training records."
        )

    if y.nunique() < 2:
        raise ValueError(
            "Only one shortage class is present. "
            "Try another --shortage-threshold."
        )

    pipeline = Pipeline([
        (
            "preprocessor",
            make_preprocessor(
                INVENTORY_NUMERIC_FEATURES,
                INVENTORY_CATEGORICAL_FEATURES,
            ),
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
    ])

    X_train, X_test, y_train, y_test = (
        split_classification(X, y, seed)
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(X_test)

    metrics = classification_metrics(
        y_test,
        predictions,
    )

    metrics["cross_validation"] = classification_cv(
        pipeline,
        X_train,
        y_train,
        seed,
    )

    pipeline.fit(X, y)

    metadata = {
        "model_name": "inventory_shortage_classifier",
        "task": "classification",
        "algorithm": "RandomForestClassifier",
        "features": features,
        "target": "shortage_label",
        "shortage_threshold": shortage_threshold,
        "class_labels": {
            "0": "adequate",
            "1": "shortage",
        },
        "training_rows": len(X),
        "holdout_rows": len(X_test),
        "metrics": metrics,
        "dataset": "relief_inventory.json",
        "evaluation_limitation": (
            "Shortage is directly determined by quantity "
            "minus reserved. A deterministic rule is "
            "more appropriate than ML for this task. "
            "This classifier is educational only."
        ),
        "created_at": timestamp(),
        "synthetic_training": True,
        "operational_use": False,
    }

    persist_model(
        pipeline,
        output_dir,
        "inventory_shortage_classifier",
        metadata,
    )

    print(
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"F1 macro: {metrics['f1_macro']:.4f}"
    )

    return metadata


# ============================================================
# MODEL TEST PREDICTIONS
# ============================================================

def generate_example_predictions(
    output_dir: Path,
) -> dict:
    examples = {}

    flood_path = (
        output_dir / "flood_risk_classifier.joblib"
    )

    if flood_path.exists():
        model = joblib.load(flood_path)

        sample = pd.DataFrame([{
            "rainfall_24h_mm": 145.0,
            "river_pressure_index": 82.0,
            "terrain_index": 65.0,
            "exposure_index": 72.0,
        }])

        examples["flood"] = {
            "input": sample.iloc[0].to_dict(),
            "predicted_risk": str(
                model.predict(sample)[0]
            ),
        }

    shelter_path = (
        output_dir / "shelter_occupancy_regressor.joblib"
    )

    if shelter_path.exists():
        model = joblib.load(shelter_path)

        sample = pd.DataFrame([{
            "capacity": 250,
            "district": "Thrissur",
            "water_status": "ready",
            "food_status": "ready",
            "medical_status": "limited",
            "accessible": True,
        }])

        predicted = float(
            model.predict(sample)[0]
        )

        examples["shelter"] = {
            "input": sample.iloc[0].to_dict(),
            "predicted_occupancy_percent": round(
                float(np.clip(predicted, 0, 100)),
                2,
            ),
        }

    inventory_path = (
        output_dir / "inventory_shortage_classifier.joblib"
    )

    if inventory_path.exists():
        model = joblib.load(inventory_path)

        sample = pd.DataFrame([{
            "quantity": 450,
            "reserved": 300,
            "district": "Alappuzha",
            "resource": "food_kits",
            "unit": "kits",
        }])

        predicted = int(
            model.predict(sample)[0]
        )

        examples["inventory"] = {
            "input": sample.iloc[0].to_dict(),
            "predicted_shortage": bool(predicted),
        }

    for item in examples.values():
        item["synthetic"] = True
        item["operational_use"] = False

    save_json(
        output_dir / "example_predictions.json",
        examples,
    )

    return examples


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train synthetic Kerala Flood AI ML models."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/demo"),
        help="Input JSON dataset directory",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/trained"),
        help="Trained-model output directory",
    )

    parser.add_argument(
        "--model",
        choices=sorted(SUPPORTED_MODELS),
        default="all",
        help="Model to train",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--shortage-threshold",
        type=int,
        default=400,
        help="Available-stock threshold",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.shortage_threshold <= 0:
        print(
            "ERROR: shortage threshold must be positive.",
            file=sys.stderr,
        )
        return 2

    print("=" * 65)
    print("KERALA FLOOD AI — ML MODEL TRAINING")
    print("=" * 65)
    print(f"Input directory : {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Selected model  : {args.model}")
    print(f"Random seed     : {args.seed}")
    print("-" * 65)

    reports = {}
    failures = {}

    training_tasks = {
        "flood": lambda: train_flood_model(
            args.input,
            args.output,
            args.seed,
        ),
        "shelter": lambda: train_shelter_model(
            args.input,
            args.output,
            args.seed,
        ),
        "inventory": lambda: train_inventory_model(
            args.input,
            args.output,
            args.seed,
            args.shortage_threshold,
        ),
    }

    selected = (
        list(training_tasks)
        if args.model == "all"
        else [args.model]
    )

    for name in selected:
        try:
            reports[name] = training_tasks[name]()
        except (
            ValueError,
            FileNotFoundError,
            KeyError,
            TypeError,
        ) as exc:
            failures[name] = str(exc)
            print(
                f"FAILED [{name}]: {exc}",
                file=sys.stderr,
            )

    if reports:
        try:
            examples = generate_example_predictions(
                args.output
            )
        except Exception as exc:
            examples = {}
            failures["examples"] = str(exc)

        summary = {
            "project": PROJECT,
            "version": VERSION,
            "created_at": timestamp(),
            "models": reports,
            "failures": failures,
            "example_predictions": examples,
            "synthetic_training": True,
            "operational_use": False,
        }

        save_json(
            args.output / "training_summary.json",
            summary,
        )

    print()
    print("=" * 65)
    print(f"Successful models: {len(reports)}")
    print(f"Failed tasks     : {len(failures)}")
    print("=" * 65)
    print(
        "DEMONSTRATION MODELS ONLY — "
        "NOT VALIDATED FOR EMERGENCY OPERATIONS"
    )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
