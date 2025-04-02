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

# === Step 4: Deep Cleaning v2 (extended for <n/s>) ===
def deep_cleaning_v2(df):
    # Drop fully empty columns and those with too much missing
    df.dropna(axis=1, how='all', inplace=True)
    df = df.loc[:, df.isna().mean() < 0.9]

    # Replace low-quality text values with 'unknown'
    ns_pattern = [
        'n/s', 'n\\s', 'ns', 'na', 'n.a.', 'n.a', '<n/s>', '<ns>', '<na>', '<n/a>', 'n\\a',
        '', ' ', 'none', 'null', '-', '--', 'n-a'
    ]
    df = df.applymap(lambda x: 'unknown' if str(x).strip().lower() in ns_pattern or len(str(x).strip()) < 2 else x)

    # Standardize all text fields
    str_cols = df.select_dtypes(include='object').columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip().str.lower())

    # Normalize specific fields
    if 'perpetrator_type' in df.columns:
        df['perpetrator_type'] = df['perpetrator_type'].replace({
            'family_member': 'family', 'relative': 'family',
            'visitor': 'visitor', 'staff': 'coworker',
            'employee': 'coworker', 'coworker': 'coworker',
            'patient': 'patient', 'self': 'self',
            'unknown': 'unknown'
        })

    if 'aggressor' in df.columns:
        df['aggressor'] = df['aggressor'].replace({
            'family_member': 'family', 'relative': 'family',
            'visitor': 'visitor', 'coworker': 'coworker',
            'patient': 'patient', 'self': 'self',
            'unknown': 'unknown'
        })

    if 'violence_type' in df.columns:
        df['violence_type'] = df['violence_type'].replace({
            'verbal abuse': 'verbal', 'verbal': 'verbal',
            'physical': 'physical', 'physical assault': 'physical',
            'sexual': 'sexual', 'sexual harassment': 'sexual',
            'harassment': 'verbal', 'property damage': 'property'
        })

    if 'victim_profession' in df.columns:
        df['victim_profession'] = df['victim_profession'].replace({
            'rn': 'nurse', 'registered nurse': 'nurse',
            'lpn': 'nurse', 'doctor': 'physician',
            'physician': 'physician', 'tech': 'technician',
            'housekeeping': 'support', 'environmental services': 'support',
            'security': 'security', 'unknown': 'unknown'
        })

    if 'department' in df.columns:
        df['department'] = df['department'].replace({
            'ed': 'emergency', 'er': 'emergency', 'icu': 'emergency',
            'ed room': 'emergency', 'ach': 'emergency',
            'hallway': 'hallway', 'halleay': 'hallway'
        })

    # Parse and extract time
    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
        df = df[df['event_date'].notna()]
        df['event_month'] = df['event_date'].dt.to_period('M')
        df['event_year'] = df['event_date'].dt.year

    if 'event_time' in df.columns:
        df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')
        df = df[df['event_time'].notna()]
        df['event_month'] = df['event_time'].dt.to_period('M')
        df['event_year'] = df['event_time'].dt.year

    return df

unified_data = deep_cleaning_v2(unified_data)

# === Step 5: Smart rule-based imputation ===
if 'perpetrator_type' in unified_data.columns and 'department' in unified_data.columns:
    mask = (unified_data['perpetrator_type'].isin(['unknown', 'nan'])) & (unified_data['department'].str.contains('emergency', na=False))
    unified_data.loc[mask, 'perpetrator_type'] = 'patient'

if 'violence_type' in unified_data.columns and 'response_action_taken' in unified_data.columns:
    mask = (unified_data['violence_type'].isin(['unknown', 'nan'])) & (unified_data['response_action_taken'].str.contains('security', na=False))
    unified_data.loc[mask, 'violence_type'] = 'physical'

# === Step 6: Final fill for any leftover missing values ===
unified_data.fillna('unknown', inplace=True)

# === Step 7: One-Hot Encoding ===
categorical_cols = [
    'facility_type', 'job_role', 'aggressor', 'perpetrator_type',
    'type_of_violence', 'violence_type', 'primary_contributing_factors',
    'severity_of_assault', 'emotional_and_or_psychological_impact',
    'level_of_care_needed', 'response_action_taken'
]
categorical_cols = [col for col in categorical_cols if col in unified_data.columns]

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(unified_data[categorical_cols])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

# === Step 8: PCA ===
pca = PCA(n_components=2)
principal_components = pca.fit_transform(encoded_df)
pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])

# === Step 9: Export cleaned result ===
unified_data.to_csv('cleaned_data/yiqing/merged_wpv_cleaned.csv', index=False)

# === Step 10: Preview ===
print(pca_df.head())
