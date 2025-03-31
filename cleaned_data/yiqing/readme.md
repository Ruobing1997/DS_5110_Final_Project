In my part of this project, I conducted comprehensive data preprocessing and analysis for the Workplace Violence (WPV) data. Specifically, I performed the following steps:

Data Integration and Cleaning:

Merged three separate WPV datasets into one unified dataset.

Standardized column names, removed inconsistencies, and handled missing values (e.g., filling missing severity data with "Unknown").

Exploratory Data Analysis (EDA):

Explored the distribution of violence types, severity levels, occupational categories, and facility types through visualizations.

Identified nurses and emergency departments as significantly impacted groups and locations.

Predictive Modeling:

Built classification models (Random Forest and Logistic Regression) to predict incident severity based on factors like aggressor type, violence type, facility type, and occupational roles.

Evaluated model performance (achieving approximately 48% accuracy with Logistic Regression and 42% accuracy with Random Forest) and visualized important factors influencing WPV severity.

Advanced Analysis:

Performed interaction analysis between key factors (e.g., facility types and aggressor types).

Conducted K-Means clustering analysis to identify high-risk incident patterns.

Analyzed the time series trends to reveal monthly patterns of WPV occurrences.

Current Results:
Identified significant factors such as "Patient" aggressors, "Physical" violence, and nursing roles being critical in predicting violence severity.

Revealed clear monthly incident trends and specific risk clusters.

Next Steps:
Further optimize predictive models (e.g., tuning parameters or using advanced algorithms).

Investigate textual descriptions of WPV incidents for richer insights.

Summarize results clearly and provide practical recommendations for WPV prevention.