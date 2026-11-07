# datathon

If using PowerShell, run these commands from the project folder. Setup is followed by cleaning, preliminary analysis, feature reduction, and model training; the three model commands can run in any order after feature reduction.

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

# 4. Train and evaluate the enrollment model using the selected features.
.\.venv\Scripts\python.exe prediction_model.py

# 5. Train and evaluate a variant that adds deposit status and dates.
.\.venv\Scripts\python.exe post_deposit_pred.py

# 6. Train and evaluate a variant that adds acceptance fee information
#    while excluding deposit status and dates.
.\.venv\Scripts\python.exe post_acc_fee_pred.py
```

Each model saves its trained model, evaluation metrics, and test predictions in its own output folder: `prediction_model_output/`, `post_deposit_pred_output/`, or `post_acc_fee_pred_output/`. The variants use the same train/test rows for comparison.

Preliminary analysis and feature reduction prefer `cleaned_dataset.csv` if present; otherwise they use the Excel file produced by cleaning. Remove any stale CSV before running the pipeline so these steps use the newly cleaned data.
