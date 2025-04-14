import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import os

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

# === Step 4: Enhanced deep cleaning ===
def deep_cleaning_v2(df):
    df.dropna(axis=1, how='all', inplace=True)
    df = df.loc[:, df.isna().mean() < 0.9]

    ns_pattern = [
        'n/s', 'n\\s', 'ns', 'na', 'n.a.', 'n.a', '<n/s>', '<ns>', '<na>', '<n/a>', 'n\\a',
        '', ' ', 'none', 'null', '-', '--', 'n-a'
    ]
    df = df.applymap(lambda x: 'unknown' if str(x).strip().lower() in ns_pattern or len(str(x).strip()) < 2 else x)

    str_cols = df.select_dtypes(include='object').columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip().str.lower())

    if 'aggressor' in df.columns:
        df['aggressor'] = df['aggressor'].replace({
            'family_member': 'family', 'relative': 'family',
            'visitor': 'visitor', 'coworker': 'coworker',
            'patient': 'patient', 'self': 'self', 'unknown': 'unknown'
        })

    if 'type_of_violence' in df.columns:
        df['type_of_violence'] = df['type_of_violence'].replace({
            'verbal abuse': 'verbal', 'physical assault': 'physical',
            'sexual harassment': 'sexual', 'harassment': 'verbal',
            'property damage': 'property'
        })

    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
        df = df[df['event_date'].notna()]
        df['event_month'] = df['event_date'].dt.to_period('M')
        df['event_year'] = df['event_date'].dt.year

    return df

unified_data = deep_cleaning_v2(unified_data)

# === Step 5: Expanded rule-based fill ===
# violence_type -> type_of_violence
if 'type_of_violence' in unified_data.columns and 'response_action_taken' in unified_data.columns:
    mask = (unified_data['type_of_violence'] == 'unknown') & (
        unified_data['response_action_taken'].str.contains('security|restrain|law enforcement|code gray|escort', na=False)
    )
    unified_data.loc[mask, 'type_of_violence'] = 'physical'

# aggressor fill
if 'aggressor' in unified_data.columns and 'occupational_category_of_person_affected' in unified_data.columns:
    mask = (unified_data['aggressor'] == 'unknown') & (
        unified_data['occupational_category_of_person_affected'].str.contains('ed|er|icu|triage|emergency', na=False)
    )
    unified_data.loc[mask, 'aggressor'] = 'patient'

# === Step 6: Mode-based fill for remaining 'unknown'
violence_mode = unified_data[unified_data['type_of_violence'] != 'unknown']['type_of_violence'].mode()[0]
aggressor_mode = unified_data[unified_data['aggressor'] != 'unknown']['aggressor'].mode()[0]

unified_data['type_of_violence'] = unified_data['type_of_violence'].replace('unknown', violence_mode)
unified_data['aggressor'] = unified_data['aggressor'].replace('unknown', aggressor_mode)

# === Step 7: Save filled result ===
output_path = 'cleaned_data/yiqing/merged_wpv_filled.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
unified_data.to_csv(output_path, index=False)

# === Step 8: Print success ===
print("Final cleaned and filled file saved to:", output_path)
