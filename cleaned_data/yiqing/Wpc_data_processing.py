import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

# === Step 1: Load files ===
file_path1 = 'prepare/Copy of WPV Data Collection MHA 2024.xlsm'
file_path2 = 'prepare/MHA WPV Data Export 09.01.24_11.30.24(1).csv'
file_path3 = 'prepare/Reports MHA WPV January 2025 Data.csv'

# Load all sheets from Excel
xls = pd.ExcelFile(file_path1)
data1_sheets = [pd.read_excel(xls, sheet_name=sheet, header=1) for sheet in xls.sheet_names]
data1 = pd.concat(data1_sheets, ignore_index=True)

# Load CSV files
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

# === Step 4: Deep cleaning function ===
def deep_cleaning(df):
    # Drop fully empty columns and those with too many missing values
    df.dropna(axis=1, how='all', inplace=True)
    df = df.loc[:, df.isna().mean() < 0.9]

    # Replace N/S, n/s, empty, short text, etc.
    df = df.applymap(lambda x: 'unknown' if str(x).strip().lower() in ['n/s', 'n\\s', 'ns', '', ' ', 'na'] or len(str(x).strip()) < 2 else x)

    # Standardize all object/text fields
    str_cols = df.select_dtypes(include='object').columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip().str.lower())

    # Aggressor standardization
    if 'aggressor' in df.columns:
        df['aggressor'] = df['aggressor'].replace({
            'family_member': 'family',
            'visitor': 'visitor',
            'coworker': 'coworker',
            'patient': 'patient',
            'self': 'self',
            'unknown': 'unknown'
        })

    # Department field normalization
    if 'department_office_incident_took_place' in df.columns:
        df['department_office_incident_took_place'] = df['department_office_incident_took_place'].replace({
            'ed': 'emergency_department',
            'er': 'emergency_department',
            'icu': 'emergency_department',
            'ed room': 'emergency_department',
            'ach': 'emergency_department',
            'hallway': 'hallway',
            'halleay': 'hallway'
        })

    # Time parsing and new columns
    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
        df = df[df['event_date'].notna()]
        df['event_month'] = df['event_date'].dt.to_period('M')
        df['event_year'] = df['event_date'].dt.year

    return df

# === Step 5: Apply deep cleaning ===
unified_data = deep_cleaning(unified_data)

# === Step 6: Remove unnamed columns again if needed ===
unified_data = unified_data.loc[:, ~unified_data.columns.str.contains('^unnamed', case=False)]

# === Step 7: Impute remaining missing values ===
imputer = SimpleImputer(strategy='constant', fill_value='unknown')
data_imputed = pd.DataFrame(imputer.fit_transform(unified_data), columns=unified_data.columns)

# === Step 8: One-Hot Encoding ===
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

# === Step 9: PCA ===
pca = PCA(n_components=2)
principal_components = pca.fit_transform(encoded_df)
pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])

# === Step 10: Export cleaned data ===
unified_data.to_csv('cleaned_data/yiqing/merged_wpv_cleaned.csv', index=False)

# === Step 11: Preview PCA output ===
print(pca_df.head())
