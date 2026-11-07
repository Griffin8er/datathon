# datathon

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

One training run saves three versions in [results/](./results/), each retaining all 15 original features from [feature_reduction.py](./feature_reduction.py):

| File prefix | Features |
| --- | --- |
| `no_acceptance_fee_no_deposits` | Original 15 only |
| `with_acceptance_fee_no_deposits` | Original 15 + acceptance_fee (16 total) |
| `with_acceptance_fee_and_deposits` | Original 15 + acceptance_fee + deposit_dt, deposit_mo, deposit_yr, deposited (20 total) |

Each prefix has a `_model.joblib`, `_metrics.json`, and `_test_predictions.csv` file. All versions use the same verified train/test rows and classifier settings. Training reads the cleaned source specified by the feature selection summary; use `--input` to override its location. The previous output folders are historical; new runs write to [results/](./results/).

To score new cleaned rows with a saved version:

```powershell
.\.venv\Scripts\python.exe prediction_model.py --predict cleaned_dataset.xlsx --variant with_acceptance_fee_and_deposits
```

This saves `with_acceptance_fee_and_deposits_predictions.csv` in [results/](./results/). Use `--predictions-output` to choose another path, or `--model` to load an explicit saved model. The default scoring version is `no_acceptance_fee_no_deposits`.

Preliminary analysis and feature reduction prefer `cleaned_dataset.csv` if present; otherwise they use the Excel file produced by cleaning. Remove any stale CSV before running the pipeline so these steps use the newly cleaned data.
