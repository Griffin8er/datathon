import argparse
import json
from pathlib import Path

import pandas as pd

from feature_reduction import stratified_split
from prediction_model import load_selection, predict, read_table, train

BASE_DIR = Path(__file__).resolve().parent
DEPOSIT_FEATURES = ["deposit_dt", "deposit_mo", "deposit_yr", "deposited"]


def prepare_split(summary_path, input_path, output_dir):
    """Restore deposit columns and verify the original train/test row alignment."""
    summary, original_features, target = load_selection(summary_path)
    if output_dir.resolve() == summary_path.parent.resolve():
        raise ValueError("Choose an output directory separate from the original feature selection.")
    source = input_path if input_path is not None else Path(summary["input"])
    print(f"Reading {source} ...", flush=True)
    dataset = read_table(source)
    features = list(dict.fromkeys(original_features + DEPOSIT_FEATURES))
    missing = [name for name in features + [target] if name not in dataset]
    if missing:
        raise ValueError(f"Missing source columns: {', '.join(missing)}")
    if not dataset[target].dropna().isin([0, 1]).all():
        raise ValueError(f"'{target}' must contain only 0, 1, or missing values.")

    labeled = dataset.dropna(subset=[target]).reset_index(drop=True)
    train_positions, test_positions = stratified_split(labeled[target])
    splits = {
        "train": labeled.iloc[train_positions].reset_index(drop=True),
        "test": labeled.iloc[test_positions].reset_index(drop=True),
    }
    # The saved reduced CSVs omit identifiers, so verify all their values before
    # attaching deposit data from the same source row positions.
    for name, frame in splits.items():
        original = read_table(summary_path.parent / f"reduced_{name}.csv")
        try:
            pd.testing.assert_frame_equal(
                frame[original_features + [target]],
                original[original_features + [target]],
                check_dtype=False,
            )
        except AssertionError as error:
            raise ValueError(
                f"Reconstructed {name} rows do not match the original split. "
                "Use the same cleaned source data used for feature reduction."
            ) from error

    augmented = dict(summary)
    augmented.update({
        "method": "Original selected features plus four required deposit features",
        "input": str(source.resolve()),
        "selected_features": features,
        "additional_features": DEPOSIT_FEATURES,
        "excluded_columns": {
            name: reason for name, reason in summary.get("excluded_columns", {}).items()
            if name not in DEPOSIT_FEATURES
        },
        "original_selection_summary": str(summary_path.resolve()),
        "prediction_time_assumption": (
            "Deposit status and dates are available; includes all labeled applicants. "
            "Verify availability of all other fields at prediction time."
        ),
        "evaluation": "Same classifier settings and verified train/test rows as prediction_model.py.",
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame[features + [target]].to_csv(output_dir / f"reduced_{name}.csv", index=False)
    augmented_path = output_dir / "selection_summary.json"
    augmented_path.write_text(json.dumps(augmented, indent=2) + "\n", encoding="utf-8")
    print(f"Verified original split; using {len(features)} features including deposit data.", flush=True)
    return augmented_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Cleaned source CSV/XLSX; defaults to the original summary input.")
    parser.add_argument("--summary", type=Path, default=BASE_DIR / "feature_reduction_output" / "selection_summary.json")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "post_deposit_pred_output")
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
            augmented_path = prepare_split(args.summary, args.input, args.output_dir)
            train(augmented_path, args.output_dir, model_path, args.threshold if args.threshold is not None else 0.5)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
