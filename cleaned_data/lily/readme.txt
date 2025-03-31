# WPV Data Processing - Lily
Author: Lily Song  
Last Updated: March 31, 2025

## 🔧 Files Covered
- WVP Data Collection - Phase I - M.xlsm
- MH December 2024 Data.xlsx
- MHA WPV events August through October 2024.xlsx

## 📌 Tasks Completed

### 1. Data Cleaning & Standardization
- Loaded raw WPV (Workplace Violence) incident data from three separate Excel sources.
- Mapped and standardized key fields:
  - Date → `event_time`
  - Job title → `victim_profession` and `victim_primary_job`
  - Department → `department`
  - Type of violence, severity, physical/emotional impact, etc.
- Cleaned and normalized values:
  - Replaced `<N/S>` and blanks with `NaN`
  - Standardized `severity_level` into {None, Mild, Moderate, Severe, Unknown}
  - Trimmed and unified inconsistent labels (e.g. job titles with slashes or extra spaces)
- Assigned dataset source labels (`phase1`, `december`, `aug_oct`) for traceability
- Merged all three datasets into one final cleaned CSV:  
  `../../cleaned_data/lily/merged_wpv_cleaned.csv`

### 2. Exploratory Data Analysis (EDA)
- Loaded the cleaned merged dataset.
- Performed data overview: shape, column list, sample rows, and missing values.
- Generated visual summaries:
  - **Severity Level Distribution**: Bar chart of violence severity levels.
  - **Violence Type Distribution**: Most common types of violence.
  - **Top Departments**: Departments with highest WPV incident counts.
  - **Top Professions Affected**: Most impacted job roles (filtered and cleaned).
- Addressed formatting issues:
  - Rotated long x-axis labels to avoid overlap
  - Filtered invalid professions like `<N/S>`
  - Applied layout adjustments for cleaner visual output

## 📁 Output Files
- `merged_wpv_cleaned.csv`: Cleaned + merged dataset
- `eda.ipynb`: Exploratory Data Analysis notebook (visual insights)
- `data_clean.ipynb`: Data preprocessing and cleaning pipeline

## ✅ Next Steps
- Join with additional datasets if needed
- Begin feature engineering & modeling (e.g. severity prediction, clustering)
