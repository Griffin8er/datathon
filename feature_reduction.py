import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
TARGET = "enrolled"
FACTORIZED_COLUMNS = {"app_fee", "acceptance_fee", "residency", "sex", "college"}
EXCLUDED = {
    "student_id": "identifier",
    "credit_hrs": "likely post-enrollment information",
    "deposited": "deposit-stage information",
    "deposit_dt": "deposit-stage information",
    "deposit_mo": "deposit-stage information",
    "deposit_yr": "deposit-stage information",
    "acceptance_fee": "likely deposit-stage information",
}


def discretize(series, bins=10):
    """Return contiguous group codes, with missingness as its own group.

    Never impute or scale nominal category codes. Quantile bins keep outliers
    from dominating numeric scores. Bin edges are learned from training only.
    """
    values = series.copy()
    if series.name in FACTORIZED_COLUMNS:
        values = values.mask(values.eq(-1))
    if pd.api.types.is_numeric_dtype(values):
        values = values.replace([np.inf, -np.inf], np.nan)
        is_date_part = str(series.name).endswith(("_dt", "_mo", "_yr", "_year"))
        if values.nunique() > 20 and not is_date_part:
            # Include the extrema so tied quantiles never discard the lowest bin.
            edges = np.unique(values.dropna().quantile(np.linspace(0, 1, bins + 1)))
            if len(edges) > 1:
                values = pd.cut(values, bins=edges, include_lowest=True)
    return pd.factorize(values, use_na_sentinel=False)[0]


def mutual_information(left, right):
    """Empirical discrete mutual information in nats (natural logarithms)."""
    counts = np.zeros((int(left.max()) + 1, int(right.max()) + 1), dtype=float)
    np.add.at(counts, (left, right), 1)
    joint = counts / counts.sum()
    independent = joint.sum(axis=1)[:, None] * joint.sum(axis=0)[None, :]
    present = joint > 0
    return max(0.0, float(np.sum(joint[present] * np.log(joint[present] / independent[present]))))


def stratified_split(target, test_fraction=0.2, seed=42):
    """Create reproducible positional splits without another ML dependency."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for label in (0, 1):
        positions = np.flatnonzero(target.to_numpy() == label)
        if len(positions) < 2:
            raise ValueError("Each enrollment class needs at least two labeled rows.")
        rng.shuffle(positions)
        test_count = min(len(positions) - 1, max(1, round(len(positions) * test_fraction)))
        test.extend(positions[:test_count])
        train.extend(positions[test_count:])
    return np.sort(train), np.sort(test)


def select_features(training, top_k=15, redundancy_threshold=0.9):
    """Rank by target MI, skipping constants and highly redundant columns.

    Redundancy is symmetric uncertainty: 2 * MI(X, Z) / (H(X) + H(Z)).
    Unlike Pearson correlation, it does not impose an order on nominal codes.
    A threshold of 0.9 removes almost-equivalent information while allowing
    moderately related predictors to remain.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    target = training[TARGET].to_numpy(dtype=int)
    codes, entropies, rows = {}, {}, []
    for column in training.columns:
        if column == TARGET:
            continue
        row = {"feature": column, "mi_nats": np.nan, "selected": False, "reason": ""}
        if column in EXCLUDED:
            row["reason"] = EXCLUDED[column]
        else:
            groups = discretize(training[column])
            if np.unique(groups).size < 2:
                row["reason"] = "constant or entirely missing in training"
            else:
                codes[column] = groups
                entropies[column] = mutual_information(groups, groups)
                row["mi_nats"] = mutual_information(groups, target)
        rows.append(row)

    ranking = pd.DataFrame(rows, columns=["feature", "mi_nats", "selected", "reason"]).sort_values(
        ["mi_nats", "feature"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)
    selected = []
    for index, row in ranking.iterrows():
        column = row["feature"]
        if column not in codes:
            continue
        if row["mi_nats"] <= 1e-12:
            ranking.at[index, "reason"] = "no measured target information"
            continue
        redundant_with = next((
            previous for previous in selected
            if 2 * mutual_information(codes[column], codes[previous])
            / (entropies[column] + entropies[previous]) >= redundancy_threshold
        ), None)
        if redundant_with is not None:
            ranking.at[index, "reason"] = f"redundant with {redundant_with}"
        elif len(selected) >= top_k:
            ranking.at[index, "reason"] = "below top-k cutoff"
        else:
            selected.append(column)
            ranking.at[index, "selected"] = True
            ranking.at[index, "reason"] = "selected"
    if not selected:
        raise ValueError("No informative candidate features remain after filtering.")
    return selected, ranking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Cleaned CSV or Excel file.")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "feature_reduction_output")
    parser.add_argument("--top-k", type=int, default=15, help="Maximum predictors to retain (default: 15).")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    input_file = args.input
    if input_file is None:
        csv_file = BASE_DIR / "cleaned_dataset.csv"
        input_file = csv_file if csv_file.exists() else BASE_DIR / "cleaned_dataset.xlsx"
    print(f"Reading {input_file} ...", flush=True)
    dataset = pd.read_csv(input_file) if input_file.suffix.lower() == ".csv" else pd.read_excel(input_file)
    if TARGET not in dataset:
        raise ValueError(f"The dataset must contain '{TARGET}'.")
    if not dataset[TARGET].dropna().isin([0, 1]).all():
        raise ValueError(f"'{TARGET}' must contain only 0, 1, or missing values.")
    labeled = dataset.dropna(subset=[TARGET]).reset_index(drop=True)
    train_positions, test_positions = stratified_split(labeled[TARGET])
    training = labeled.iloc[train_positions]
    selected, ranking = select_features(training, top_k=args.top_k)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    # Export raw values, not bins: later models can choose their own preprocessing.
    columns = selected + [TARGET]
    dataset[columns].to_csv(output_dir / "reduced_dataset.csv", index=False)
    training[columns].to_csv(output_dir / "reduced_train.csv", index=False)
    labeled.iloc[test_positions][columns].to_csv(output_dir / "reduced_test.csv", index=False)
    ranking.to_csv(output_dir / "feature_rankings.csv", index=False)
    summary = {
        "method": "Quantile-binned mutual information + symmetric-uncertainty redundancy filter",
        "input": str(input_file.resolve()),
        "target": TARGET,
        "rows": len(dataset),
        "unlabeled_rows_excluded_from_split": int(dataset[TARGET].isna().sum()),
        "train_rows": len(training),
        "test_rows": len(test_positions),
        "split": "Stratified random 80/20; seed 42. Does not measure future-cohort performance.",
        "original_predictors": len(dataset.columns) - 1,
        "requested_top_k": args.top_k,
        "selected_features": selected,
        "excluded_columns": {key: value for key, value in EXCLUDED.items() if key in dataset},
        "prediction_time_assumption": "Before deposit; verify availability of all other fields.",
        "scoring": "Training data only; 10 quantile bins for dense numeric measurements; missingness is a group.",
        "redundancy_threshold": 0.9,
        "evaluation": "No predictive accuracy estimated; fit and evaluate a model using reduced_train/reduced_test.",
    }
    (output_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {len(selected)} of {len(dataset.columns) - 1} predictors using {len(training):,} training rows.")
    print(ranking.loc[ranking["selected"], ["feature", "mi_nats"]].to_string(index=False))
    print(f"Saved reduced data, train/test splits, rankings, and selection summary to {output_dir}")


if __name__ == "__main__":
    main()
