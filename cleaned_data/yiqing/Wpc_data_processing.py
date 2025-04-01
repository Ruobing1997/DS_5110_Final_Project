import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

# === Step 1: Load files ===
file_path1 = 'prepare/Copy of WPV Data Collection MHA 2024.xlsm'
file_path2 = 'prepare/MHA WPV Data Export 09.01.24_11.30.24(1).csv'
file_path3 = 'prepare/Reports MHA WPV January 2025 Data.csv'

# Read all sheets from the Excel file
xls = pd.ExcelFile(file_path1)
data1_sheets = [pd.read_excel(xls, sheet_name=sheet, header=1) for sheet in xls.sheet_names]
data1 = pd.concat(data1_sheets, ignore_index=True)

# Read the CSV files
data2 = pd.read_csv(file_path2, skiprows=4)
data3 = pd.read_csv(file_path3)

# === Step 2: Standardize column names ===
def standardize_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_')
    return df

data1 = standardize_columns(data1)
data2 = standardize_columns(data2)
data3 = standardize_columns(data3)

# === Step 3: Merge data ===
unified_data = pd.concat([data1, data2, data3], ignore_index=True)

# === Step 4: Remove Unnamed columns ===
unified_data = unified_data.loc[:, ~unified_data.columns.str.contains('^unnamed', case=False)]

# === Step 5: Missing Value Imputation ===
imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
data_imputed = pd.DataFrame(imputer.fit_transform(unified_data), columns=unified_data.columns)

# === Step 6: One-Hot Encoding for categorical features ===
categorical_cols = [
    'facility_type', 'job_role', 'aggressor', 'type_of_violence',
    'primary_contributing_factors', 'severity_of_assault',
    'emotional_and_or_psychological_impact', 'level_of_care_needed',
    'response_action_taken'
]
categorical_cols = [col for col in categorical_cols if col in data_imputed.columns]

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(data_imputed[categorical_cols])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

# === Step 7: PCA ===
pca = PCA(n_components=2)
principal_components = pca.fit_transform(encoded_df)
pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])

# === Step 8: Export cleaned merged data ===
unified_data.dropna(how='all', inplace=True)
unified_data.to_csv('cleaned_data/yiqing/merged_wpv_cleaned.csv', index=False)

# === Step 9: Output ===
print(pca_df.head())
