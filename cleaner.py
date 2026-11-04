import pandas as pd

dataset = pd.read_excel("datathon-dataset-2026.xlsx")

months = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12
}

dataset["app_mo"] = dataset["app_mo"].str.lower().map(months)
dataset["first_contact_mo"] = dataset["first_contact_mo"].str.lower().map(months)
dataset["visit_mo"] = dataset["visit_mo"].str.lower().map(months)
dataset['app_fee'] = pd.factorize(dataset['app_fee'])[0]
dataset['acceptance_fee'] = pd.factorize(dataset['acceptance_fee'])[0]
dataset['residency'] = pd.factorize(dataset['residency'])[0]
dataset['sex'] = pd.factorize(dataset['sex'])[0]
dataset['college'] = pd.factorize(dataset['college'])[0]

dataset = dataset.dropna(subset=["student_id"]).drop_duplicates(subset="student_id", keep="first")

cols_to_convert = [col for col in dataset.columns if col not in ['hs_gpa', 'credit_hrs']]
dataset[cols_to_convert] = dataset[cols_to_convert].astype('Int64')

# Could change to original file name but chose not to overwrite cuz safer
# cleaned_dataset.xlsx is in the .gitignore

with pd.ExcelWriter("cleaned_dataset.xlsx") as writer:
    dataset.to_excel(writer, index=False)