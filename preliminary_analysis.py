from pathlib import Path
import matplotlib

matplotlib.use("Agg")  # Save plots without opening interactive windows.

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "cleaned_dataset.csv"
OUTPUT_DIR = BASE_DIR / "preliminary_plots"
# cleaner.py uses factorize(), whose missing-value code is -1.
FACTORIZED_COLUMNS = {"app_fee", "acceptance_fee", "residency", "sex", "college"}
MISSING_LABEL = "Missing / not recorded"


def category_labels(series):
    """Keep discrete codes intact and bin dense numeric measurements."""
    values = series.copy()
    if series.name in FACTORIZED_COLUMNS:
        values = values.mask(values.eq(-1))

    binned = False
    # Day/month/year fields have meaningful discrete values, including day 31.
    if (
        pd.api.types.is_numeric_dtype(values)
        and values.nunique() > 30
        and not str(series.name).endswith(("_dt", "_mo", "_yr", "_year"))
    ):
        # Equal-width ranges preserve differences in population density.
        values = pd.cut(values, bins=10, precision=2, include_lowest=True)
        binned = True

    categories = list(values.dropna().unique())
    categories.sort()
    order = [str(value) for value in categories]
    labels = values.astype("string").fillna(MISSING_LABEL)
    if values.isna().any():
        order.append(MISSING_LABEL)
    return labels, order, binned


def summarize_column(dataset, column):
    labels, order, binned = category_labels(dataset[column])
    outcomes = pd.DataFrame(
        {
            "category": labels,
            "enrolled": dataset["enrolled"].eq(1).fillna(False).astype(int),
            "not_enrolled": dataset["enrolled"].eq(0).fillna(False).astype(int),
            "unknown": dataset["enrolled"].isna().astype(int),
        }
    )
    summary = outcomes.groupby("category", sort=False).sum().reindex(order)
    summary["total"] = summary[["enrolled", "not_enrolled", "unknown"]].sum(axis=1)
    known = summary["enrolled"] + summary["not_enrolled"]
    summary["rate"] = summary["enrolled"].div(known.where(known.ne(0))).mul(100)
    return summary, binned


def main():
    input_file = INPUT_FILE if INPUT_FILE.exists() else INPUT_FILE.with_suffix(".xlsx")
    dataset = pd.read_csv(input_file) if input_file.suffix == ".csv" else pd.read_excel(input_file)
    if "enrolled" not in dataset.columns:
        raise ValueError("The dataset must contain an 'enrolled' column.")
    if not dataset["enrolled"].dropna().isin([0, 1]).all():
        raise ValueError("The 'enrolled' column must contain only 0, 1, or missing values.")

    if dataset.empty:
        raise ValueError("The dataset contains no rows.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_count = 0
    for column in dataset.columns:
        if column in {"student_id", "enrolled"}:
            continue

        summary, binned = summarize_column(dataset, column)
        positions = range(len(summary))
        figure, (count_axis, rate_axis) = plt.subplots(
            ncols=2,
            sharey=True,
            figsize=(15, max(5, len(summary) * 0.35 + 2)),
            gridspec_kw={"width_ratios": [2.3, 1]},
        )
        left = pd.Series(0, index=summary.index)
        for key, label, color in (
            ("enrolled", "Enrolled", "#16877d"),
            ("not_enrolled", "Not enrolled", "#b7c9dc"),
            ("unknown", "Enrollment unknown", "#d6d6d6"),
        ):
            if key == "unknown" and not summary[key].any():
                continue
            count_axis.barh(positions, summary[key], left=left, label=label, color=color)
            left = left + summary[key]

        # Every bar uses the same count scale: its length reflects group size.
        max_count = summary["total"].max()
        for position, total in enumerate(summary["total"]):
            count_axis.text(
                total + max_count * 0.015, position,
                f"{total:,} ({total / len(dataset):.1%})", va="center", fontsize=8,
            )
        count_axis.set_xlim(0, max_count * 1.32)
        count_axis.set_yticks(positions, summary.index)
        count_axis.invert_yaxis()
        count_axis.set_xlabel("Number of students (label: count and share of all students)")
        count_axis.set_title("Group size and enrollment outcomes", loc="left", fontsize=11)
        count_axis.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncols=3, frameon=False)

        rate_axis.scatter(summary["rate"], positions, color="#16877d", s=35, zorder=3)
        for position, (_, row) in enumerate(summary.iterrows()):
            annotation = (
                f"{row['rate']:.1f}% ({int(row['enrolled']):,}/{int(row['total'] - row['unknown']):,})"
                if pd.notna(row["rate"]) else "No known outcomes"
            )
            rate_axis.text(103, position, annotation, va="center", fontsize=8)
        overall_rate = dataset["enrolled"].mean() * 100
        if pd.notna(overall_rate):
            rate_axis.axvline(overall_rate, color="#666666", linestyle="--", linewidth=1,
                              label=f"Overall: {overall_rate:.1f}%")
            rate_axis.legend(loc="lower left", bbox_to_anchor=(0, 1.04), frameon=False)
        rate_axis.set_xlim(0, 155)
        rate_axis.set_xticks([0, 25, 50, 75, 100])
        rate_axis.xaxis.set_major_formatter(PercentFormatter(xmax=100))
        rate_axis.set_xlabel("Enrolled / known enrollment outcomes")
        rate_axis.set_title("Within-group enrollment rate", loc="left", fontsize=11)
        rate_axis.tick_params(axis="y", left=False)
        for axis in (count_axis, rate_axis):
            axis.set_axisbelow(True)
            axis.grid(axis="x", alpha=0.2)
            axis.spines[["top", "right"]].set_visible(False)

        missing_count = int(summary.loc[MISSING_LABEL, "total"]) if MISSING_LABEL in summary.index else 0
        figure.suptitle(f"Enrollment by {column}", fontsize=16, x=0.02, ha="left")
        note = (
            f"All {len(dataset):,} students included | Missing {column}: "
            f"{missing_count:,} ({missing_count / len(dataset):.1%})"
        )
        if binned:
            note += " | Numeric values grouped into 10 equal-width ranges"
        figure.text(0.02, 0.02, note, fontsize=9, color="#555555")
        figure.tight_layout(rect=(0, 0.05, 1, 0.93))

        # Keep output filenames valid even if future column names contain slashes.
        safe_name = "".join(character if character.isalnum() or character in "_-" else "_" for character in column)
        output_file = OUTPUT_DIR / f"{safe_name}.png"
        figure.savefig(output_file, dpi=180)
        plt.close(figure)
        plot_count += 1

    print(f"Saved {plot_count} plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
