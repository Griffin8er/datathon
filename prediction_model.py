import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from threadpoolctl import threadpool_limits

BASE_DIR = Path(__file__).resolve().parent
# cleaner.py encodes missing values in these nominal fields as -1.
FACTORIZED_COLUMNS = {"app_fee", "acceptance_fee", "residency", "sex", "college"}


def read_table(path):
    """Read the same cleaned numeric representation used for feature selection."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        return pd.read_excel(path)
    raise ValueError(f"Expected a CSV or XLSX file, received: {path}")


def prepare_features(frame, features):
    """Enforce feature order and normalize missing values for training/inference."""
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing selected feature columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("The input contains no rows.")
    values = frame.loc[:, features].apply(pd.to_numeric, errors="raise").astype(float)
    values = values.replace([np.inf, -np.inf], np.nan)
    for name in FACTORIZED_COLUMNS.intersection(features):
        values[name] = values[name].mask(values[name].eq(-1))
    return values


def load_selection(path):
    summary = json.loads(path.read_text(encoding="utf-8"))
    features = summary.get("selected_features")
    target = summary.get("target")
    if (
        not isinstance(features, list)
        or not features
        or not all(isinstance(name, str) and name for name in features)
        or len(features) != len(set(features))
    ):
        raise ValueError("The selection summary must have unique selected_features.")
    if not isinstance(target, str) or not target:
        raise ValueError("The selection summary must specify a target column.")
    forbidden = set(summary.get("excluded_columns", {})) | {target}
    if forbidden.intersection(features):
        raise ValueError("Selected features contain the target or excluded columns.")
    return summary, features, target


def load_split(path, features, target):
    frame = read_table(path)
    if target not in frame:
        raise ValueError(f"Missing target column '{target}' in {path}.")
    labels = pd.to_numeric(frame[target], errors="raise")
    if labels.isna().any() or not labels.isin([0, 1]).all() or labels.nunique() != 2:
        raise ValueError(f"{path} must contain both classes, encoded as 0 and 1.")
    return frame, prepare_features(frame, features), labels.astype(int)


def build_model(features):
    """Small boosted trees capture nonlinear effects without scaling/imputation.

    Nominal codes are encoded as categorical features, without imposing an order.
    Unknown categories become missing, which the classifier handles natively.
    The encoder and classifier are saved together to preserve inference behavior.
    """
    categorical = [name for name in features if name in FACTORIZED_COLUMNS]
    numeric = [name for name in features if name not in categorical]
    transformers = [("numeric", "passthrough", numeric)]
    if categorical:
        transformers.append((
            "categorical",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
                encoded_missing_value=np.nan,
            ),
            categorical,
        ))
    return Pipeline([
        ("features", ColumnTransformer(transformers)),
        ("classifier", HistGradientBoostingClassifier(
            categorical_features=[False] * len(numeric) + [True] * len(categorical),
            learning_rate=0.1,
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )),
    ])


def evaluate(labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def save_predictions(frame, probabilities, threshold, output, target):
    # Row numbers are one-based data-row positions (excluding the header).
    result = pd.DataFrame({"row_number": np.arange(1, len(frame) + 1)})
    if "student_id" in frame:
        result["student_id"] = frame["student_id"].to_numpy()
    if target in frame:
        result[f"actual_{target}"] = frame[target].to_numpy()
    result["enrollment_probability"] = probabilities
    result["predicted_enrolled"] = (probabilities >= threshold).astype(int)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)


def train(summary_path, output_dir, model_path, threshold):
    summary, features, target = load_selection(summary_path)
    # Re-splitting would mix feature-selection training rows into the test set.
    train_path = summary_path.parent / "reduced_train.csv"
    test_path = summary_path.parent / "reduced_test.csv"
    _, training, train_labels = load_split(train_path, features, target)
    test_frame, testing, test_labels = load_split(test_path, features, target)
    for key, count in (("train_rows", len(training)), ("test_rows", len(testing))):
        if key in summary and summary[key] != count:
            raise ValueError(f"{key} does not match the selection summary; regenerate the split.")

    model = build_model(features)
    started = perf_counter()
    # A small thread pool avoids parallelism overhead on this modest dataset.
    with threadpool_limits(limits=2):
        model.fit(training, train_labels)
        fit_seconds = perf_counter() - started
        probabilities = model.predict_proba(testing)[:, 1]
    baseline = DummyClassifier(strategy="prior").fit(training, train_labels)
    baseline_probabilities = baseline.predict_proba(testing)[:, 1]
    metrics = evaluate(test_labels, probabilities, threshold)
    report = {
        "model": "HistGradientBoostingClassifier",
        "scikit_learn_version": sklearn.__version__,
        "selected_features": features,
        "target": target,
        "train_rows": len(training),
        "test_rows": len(testing),
        "train_enrollment_rate": float(train_labels.mean()),
        "test_enrollment_rate": float(test_labels.mean()),
        "threshold": threshold,
        "fit_seconds": fit_seconds,
        "boosting_iterations": int(model.named_steps["classifier"].n_iter_),
        "test_metrics": metrics,
        "prior_baseline_metrics": evaluate(test_labels, baseline_probabilities, threshold),
        "confusion_matrix_layout": "Rows = actual [0, 1]; columns = predicted [0, 1].",
        "selection_summary": str(summary_path.resolve()),
        "split": summary.get("split"),
        "prediction_time_assumption": summary.get("prediction_time_assumption"),
        "evaluation_note": (
            "Feature selection and preprocessing use training rows only. Early stopping "
            "uses an internal training-data validation split. The original test set is "
            "used only for final evaluation; these metrics do not measure future-cohort "
            "performance. All predictors must be available at prediction time."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "selected_features": features,
        "target": target,
        "threshold": threshold,
        "scikit_learn_version": sklearn.__version__,
    }, model_path)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    save_predictions(test_frame, probabilities, threshold, output_dir / "test_predictions.csv", target)
    print(f"Trained on {len(training):,} rows with {len(features)} features in {fit_seconds:.2f}s.")
    print(f"Held-out test rows: {len(testing):,}; enrollment rate: {test_labels.mean():.1%}")
    for name in ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"):
        print(f"  {name}: {metrics[name]:.4f}")
    print(f"Prior baseline accuracy: {report['prior_baseline_metrics']['accuracy']:.4f}")
    print(f"Saved model to {model_path}; metrics and test predictions to {output_dir}")


def predict(input_path, model_path, output, threshold=None):
    """Score cleaned rows with the stored encoder, feature order, and classifier."""
    artifact = joblib.load(model_path)
    cutoff = artifact["threshold"] if threshold is None else threshold
    frame = read_table(input_path)
    values = prepare_features(frame, artifact["selected_features"])
    with threadpool_limits(limits=2):
        probabilities = artifact["model"].predict_proba(values)[:, 1]
    save_predictions(frame, probabilities, cutoff, output, artifact["target"])
    print(f"Saved {len(frame):,} predictions to {output} (threshold {cutoff:g}).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=BASE_DIR / "feature_reduction_output" / "selection_summary.json")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "prediction_model_output")
    parser.add_argument("--model", type=Path, help="Saved model path; defaults to OUTPUT_DIR/enrollment_model.joblib.")
    parser.add_argument("--predict", type=Path, help="Predict on cleaned CSV/XLSX rows instead of training.")
    parser.add_argument("--predictions-output", type=Path, help="Prediction CSV path (with --predict).")
    parser.add_argument("--threshold", type=float, help="Probability cutoff in (0, 1); defaults to 0.5 or the saved cutoff.")
    args = parser.parse_args()
    if args.threshold is not None and not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")
    if args.predictions_output is not None and args.predict is None:
        parser.error("--predictions-output requires --predict")
    model_path = args.model if args.model is not None else args.output_dir / "enrollment_model.joblib"
    try:
        if args.predict is not None:
            output = args.predictions_output or args.output_dir / "predictions.csv"
            predict(args.predict, model_path, output, args.threshold)
        else:
            train(args.summary, args.output_dir, model_path, args.threshold if args.threshold is not None else 0.5)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
