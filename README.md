# datathon

For a guided walkthrough with saved cell outputs, open these notebooks in order and run each from top to bottom:

**Temporal-validation update:** train on entry years 2022-2024, validate on 2025, and reserve 2026 for final testing. Feature selection uses training rows only. Run All in notebook 04 evaluates all three variants on 2025, then scores the unchanged fitted models on 2026 in one final-evaluation section (`final_test_comparison.csv` in `results/`). Do not tune against the final-test results.

**Calendar-field update:** feature selection now excludes absolute calendar years and days of the month (`entry_year`, `app_yr`, `app_dt`, `visit_yr`, `visit_dt`, `first_contact_yr`, `first_contact_dt`), which do not transfer to a new cohort. Visit timing is kept as `visit_yrs_before_entry`, the entry year minus the most recent visit year. This change was made after a first 2026 evaluation, so the current 2026 results are a revised comparison, not an untouched test.

1. [01_cleaner.ipynb](./01_cleaner.ipynb) — clean the spreadsheet, inspect missingness, and save the data and category-code lookup.
2. [02_preliminary_analysis.ipynb](./02_preliminary_analysis.ipynb) — display every predictor's enrollment chart and summary table, while saving PNGs in [preliminary_plots/](./preliminary_plots/).
3. [03_feature_reduction.ipynb](./03_feature_reduction.ipynb) — explain the split and selection, inspect rankings, and save reduced data in [feature_reduction_output/](./feature_reduction_output/).
4. [04_prediction_model.ipynb](./04_prediction_model.ipynb) — fit and compare three models, display metrics and prediction previews, and save models, reports, predictions, and comparison charts in [results/](./results/).
5. [05_predict_enrollment.ipynb](./05_predict_enrollment.ipynb) — forecast total enrollment at pre-acceptance, post-acceptance, and post-deposit checkpoints using aggregate Poisson regression. Save evaluation and production models, count reports, and comparison charts in [results/total_enrollment/](./results/total_enrollment/).
6. [06_final_prediction.ipynb](./06_final_prediction.ipynb) — reload saved models and display the three 2026 enrollment forecasts: Poisson models for pre-acceptance and post-acceptance, and the gradient boosted classifier with acceptance-fee and deposit features for post-deposit. This notebook writes no prediction files.

Notebook 05 requires only notebook 01's [cleaned_dataset.xlsx](./cleaned_dataset.xlsx) and [cleaned_category_lookup.csv](./cleaned_category_lookup.csv); it does not depend on notebooks 03–04. It models enrolled counts by entry year, college, and residency, using applicant volume as exposure, then sums group forecasts into cohort totals. Regularization is selected through expanding chronological validation (2024 and 2025 with the current data), with 2026 reserved for final testing. Evaluation models train on 2022–2025; separate production models are subsequently refitted on all labeled years for future cohorts. Outputs include checkpoint test totals, group counts, validation results, metrics, and a source-column audit.

Run notebooks 04 and 05 before notebook 06 to generate its saved models. Notebook 06 prefers notebook 04's chronological validation model for post-deposit predictions and falls back to the older random-split model if needed; that fallback may include 2026 applicants in training and is not an independent 2026 evaluation. Its post-deposit forecast uses the classifier, rather than notebook 05's post-deposit Poisson model.

The checkpoint forecasts assume the same complete applicant pool and fields available at each stated checkpoint. The source contains final records rather than historical dated snapshots, so these comparisons do not establish accuracy at an actual calendar cutoff. Only five annual cohorts are available, and notebook 05 does not provide calibrated forecast intervals.

Choose the project's virtual environment as the notebook kernel. Install the dependencies below, plus `ipykernel`, to run the notebooks in VS Code (`python -m pip install -r requirements.txt ipykernel`). Keep the project folder as the working directory. **Run All** refreshes both the visible results and the files on disk. The notebooks explicitly read [cleaned_dataset.xlsx](./cleaned_dataset.xlsx) to stay consistent with the cleaning step. The command-line examples below are from the earlier script workflow; this checkout currently includes notebooks only.

If using PowerShell, run these commands from the project folder. Setup is followed by cleaning, preliminary analysis, feature reduction, and model training.

```powershell
# Create a virtual environment and install dependencies (first-time setup).
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 1. Encode categories and dates, remove missing/duplicate student IDs,
#    and save cleaned_dataset.xlsx.
.\.venv\Scripts\python.exe cleaner.py

# 2. Plot feature distributions and enrollment rates in preliminary_plots/.
.\.venv\Scripts\python.exe preliminary_analysis.py

# 3. Select informative, nonredundant predictors using training rows only;
#    save rankings and an 80/20 train/test split in feature_reduction_output/.
.\.venv\Scripts\python.exe feature_reduction.py

# 4. Train and evaluate all three enrollment model versions in results/.
.\.venv\Scripts\python.exe prediction_model.py
```

One training run saves three versions in [results/](./results/), each retaining all 15 selected features from [03_feature_reduction.ipynb](./03_feature_reduction.ipynb):

| File prefix | Features |
| --- | --- |
| `no_acceptance_fee_no_deposits` | Original 15 only |
| `with_acceptance_fee_no_deposits` | Original 15 + acceptance_fee (16 total) |
| `with_acceptance_fee_and_deposits` | Original 15 + acceptance_fee + deposit_dt, deposit_mo, deposit_yr, deposited (20 total) |

Each prefix now has a `_validation_model.joblib`, `_validation_metrics.json`, and `_validation_predictions.csv` file. All versions use the same verified training/validation rows and unchanged classifier settings. Reports include classification metrics and class-size forecasts (sum of probabilities), with model and baseline forecast errors. Combined results are saved as `validation_model_comparison.csv` and `validation_model_comparison.png` in [results/](./results/). Generic legacy outputs are not final 2026 test results.

To score new cleaned rows with a saved version:

```powershell
.\.venv\Scripts\python.exe prediction_model.py --predict cleaned_dataset.xlsx --variant with_acceptance_fee_and_deposits
```

This saves `with_acceptance_fee_and_deposits_predictions.csv` in [results/](./results/). Use `--predictions-output` to choose another path, or `--model` to load an explicit saved model. The default scoring version is `no_acceptance_fee_no_deposits`.

Preliminary analysis and feature reduction prefer `cleaned_dataset.csv` if present; otherwise they use the Excel file produced by cleaning. Remove any stale CSV before running the pipeline so these steps use the newly cleaned data.
