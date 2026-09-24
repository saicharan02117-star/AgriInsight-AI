# AgriInsight AI
## AI-Powered Crop Yield & Farm Decision Analytics Platform

**Internship:** AICTE | IBM SkillsBuild - Data Analytics with AI Internship 2026  
**Submitted by:** Thodupunuri Sai Charan  
**Program:** B.Tech - Computer Science and Engineering (Data Science)  
**Institution:** CMR College of Engineering & Technology

## 1. Project Overview
AgriInsight AI is a data analytics and machine-learning project that converts historical agricultural records into clear decision-support information. Instead of only showing charts, the project follows a business-intelligence style workflow: identify the important KPIs, study trends and drivers, compare machine-learning models, validate the model, and convert the result into risk/opportunity notes.

The project is intentionally kept simple and readable. The objective is not to create the maximum number of charts or the most complex code. The objective is to make the data useful for a decision.

## 2. Problem Statement
Historical agricultural datasets contain crop yield, rainfall, temperature and input-use information, but raw rows do not directly answer practical questions such as:

- Which crops and areas show stronger historical yield performance?
- How has yield changed over time?
- Which environmental/input variables are associated with yield variation?
- Can a machine-learning model estimate yield from known conditions?
- Is the model reliable enough to use as decision support?
- What should a user check next when the predicted yield is unusually low or high?

AgriInsight AI addresses these questions through a reproducible data analytics pipeline and interactive dashboard.

## 3. Dataset
**Primary dataset:** Crop Yield Prediction Dataset  
**Kaggle source:** https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset

The dataset contains worldwide crop-yield observations and combines yield information with rainfall, pesticide-use and average-temperature variables. The public dataset contains approximately 28,242 raw observations, 101 countries/areas and 10 crop types. The dataset page acknowledges publicly available data from FAO and the World Bank.

The submitted code first looks for a local `yield_df.csv`. If it is not available, it attempts to load a public raw mirror of the same file. For offline evaluation, download `yield_df.csv` from the Kaggle link and place it beside the Python file.

### Main fields used
| Field | Use in the project |
|---|---|
| `Area` | Country/area dimension |
| `Item` | Crop type |
| `Year` | Historical time dimension |
| `average_rain_fall_mm_per_year` | Climate input |
| `pesticides_tonnes` | Agricultural input variable |
| `avg_temp` | Climate input |
| `hg/ha_yield` | Prediction target and main yield KPI |

## 4. Data Analytics Workflow
The project follows this sequence:

**Data collection -> Data cleaning -> Exploratory Data Analysis -> KPI and trend analysis -> Feature preprocessing -> Model comparison -> Validation -> Prediction -> Risk/opportunity interpretation -> Recommended next checks**

### Data cleaning
- Remove unnamed index columns.
- Convert numeric columns safely.
- Remove rows with missing required values.
- Remove duplicate rows.
- Remove non-physical negative values for yield, rainfall and pesticide-use fields.
- Keep the cleaning logic visible in code rather than hiding it in a pre-processed file.

### Exploratory Data Analysis
The dashboard provides:
- clean-record count;
- number of countries/areas;
- number of crop types;
- average yield;
- crop-wise average yield;
- historical yield trend;
- top areas for a selected crop;
- rainfall vs yield;
- temperature vs yield;
- crop-level Pearson correlation table.

The correlation outputs are treated as exploratory signals only. The project does not claim that correlation proves causation.

## 5. AI / Machine Learning
Two models are trained so that an ML result is compared against a simple baseline:

1. **Linear Regression** - baseline model.
2. **Random Forest Regressor** - non-linear ensemble model.

Categorical fields (`Area`, `Item`) are one-hot encoded. Numeric variables are standardized. The data is split into training and test sets using a fixed random seed for reproducibility.

### Model evaluation
The dashboard calculates:
- **MAE** - Mean Absolute Error;
- **RMSE** - Root Mean Squared Error;
- **R²** - coefficient of determination.

The model with the higher test-set R² is selected. Metric values are calculated by the submitted code when the project runs; they are not hard-coded into the report.

### Model Reliability Gate
A prediction is not automatically treated as trustworthy. AgriInsight AI converts test-set R² into a visible reliability status:

- Strong validation signal
- Moderate validation signal
- Weak validation signal
- Low reliability

This prevents a visually attractive dashboard from presenting a weak statistical model as certain.

## 6. Decision Intelligence
After a prediction, the system compares the predicted yield with the historical distribution for the selected crop.

It produces simple evidence-linked flags such as:
- lower historical yield quartile -> yield-risk review;
- upper historical yield quartile -> opportunity review;
- rainfall outside the crop's central historical range -> water/drainage check;
- temperature outside the central range -> higher uncertainty note;
- pesticide use above the historical upper quartile -> review IPM/local guidance.

The project does **not** prescribe fertilizer or pesticide dose. Final field decisions require local soil tests, crop stage, weather information and agronomic guidance.

## 7. Visualization and Dashboard
**Visualization tools:** Matplotlib + Streamlit.

The application is divided into four clear sections:

1. **Executive Overview** - important KPIs and historical performance.
2. **Data Analytics** - crop/area trends and exploratory relationships.
3. **AI Prediction** - user inputs, yield estimate, reliability gate and decision-support notes.
4. **Method & Reliability** - model comparison, workflow, IBM SkillsBuild alignment and limitations.

The final project report includes dashboard-output views and the main analytical charts generated from the working dataset: executive KPIs, crop-wise average yield, historical yield trend, top-area comparison and exploratory correlation results. The interface uses larger headings and readable text and avoids unnecessary decorative elements.

### Main dataset-level findings shown in the report
- 28,242 historical records, 101 countries/areas and 10 crop types.
- Overall average yield: about 7.71 t/ha.
- Highest crop-level historical average in the dataset: Potatoes, about 19.98 t/ha.
- Aggregate average yield rises from about 6.64 t/ha in 1990 to about 9.04 t/ha in 2013.
- Full-dataset linear correlations with yield are weak for rainfall, pesticide use, year and average temperature; therefore they are presented as exploratory signals, not causal conclusions.

## 8. IBM SkillsBuild / IBM Bob Integration
IBM Bob is integrated at the **repository/workflow level** for AI-assisted planning, review and project-file quality checks. It is not used as a hidden prediction API. The reproducible Python pipeline remains the source of truth for cleaning, EDA, validation metrics and yield prediction.

The Streamlit application includes a dedicated **IBM Bob Integration** tab. It automatically creates a Bob-ready project brief from the live cleaned dataset and model-validation results. The brief contains:

- project problem statement and objective;
- current dataset profile and KPIs;
- input features and prediction target;
- selected model and current MAE, RMSE and R²;
- the exact project files Bob should review;
- review tasks for cleaning, EDA, ML, dependencies, README and report;
- responsible-use constraints so missing soil/agronomic variables are not invented;
- an approval workflow requiring Bob to propose changes before material edits are accepted.

The same handoff is included in the repository as `IBM_Bob_Project_Brief.txt`. A project-level Bob Skill is included at `.bob/skills/agriinsight-review/SKILL.md`; in Bob Advanced mode it can be invoked with `use_skill agriinsight-review`. The project follows the internship sequence: **define problem -> attach/tag dataset -> plan -> generate/review files -> approve changes -> verify required files -> submit GitHub repository**.

### IBM Bob checkpoints used by the project
1. Confirm the problem statement, KPIs, dataset and scope before code changes.
2. Review data cleaning, EDA and visualization logic against the attached dataset.
3. Review Linear Regression and Random Forest modelling and held-out evaluation.
4. Verify `requirements.txt`, README and project report are consistent with the code.
5. Ask for approval before material revisions.
6. Perform a final GitHub-ready repository check.

## 9. Project Files
```text
AgriInsight-AI/
|-- AgriInsight_AI.py
|-- AgriInsight_AI.ipynb
|-- requirements.txt
|-- README.md
|-- AgriInsight_AI_Project_Report.docx
|-- AgriInsight_AI_Project_Report.pdf
|-- IBM_Bob_Project_Brief.txt
|-- IBM_Bob_Integration_Guide.md
|-- .bob/skills/agriinsight-review/SKILL.md
|-- .bob/skills/agriinsight-review/CHECKLIST.md
|-- visuals/  # dashboard and chart outputs used in the report
`-- yield_df.csv  # optional local copy; can be downloaded from the dataset source
```

## 10. Installation
Use Python 3.9+.

```bash
pip install -r requirements.txt
```

## 11. Run the Dashboard
```bash
streamlit run AgriInsight_AI.py
```

If the automatic dataset download is blocked, download `yield_df.csv` from the Kaggle source and put it in the same folder as `AgriInsight_AI.py`.

## 12. Run the Notebook
```bash
jupyter notebook AgriInsight_AI.ipynb
```

The notebook contains the full analytics workflow in a submission-friendly sequence: loading, cleaning, EDA, model training, evaluation, prediction and decision-support logic.

## 13. Limitations
- Historical associations do not prove agronomic causation.
- The dataset is country-level rather than plant-level or plot-level.
- Soil-test variables are not available in this dataset and are therefore not invented.
- A yield estimate is statistical decision support, not a guaranteed farm outcome.
- Recommendations are checks/actions to consider, not chemical-dose prescriptions.

## 14. References
- Kaggle - Crop Yield Prediction Dataset: https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset
- Food and Agriculture Organization (FAO): https://www.fao.org/
- World Bank Data: https://data.worldbank.org/
- IBM SkillsBuild - IBM Bob learning resources: https://skillsbuild.org/learn-with-ibm-bob
- Scikit-learn documentation: https://scikit-learn.org/
- Streamlit documentation: https://docs.streamlit.io/

---
**Project:** AgriInsight AI - AI-Powered Crop Yield & Farm Decision Analytics Platform


## 14. Submission Form Mapping
- **Code File:** `AgriInsight_AI.ipynb`
- **Requirements File:** `requirements.txt`
- **Project Report:** `AgriInsight_AI_Project_Report.docx` (PDF copy also included)
- **README File:** `README.md`
- **GitHub Repository:** upload this repository to GitHub and submit its public repository URL

The dataset source link is included above as requested in the project discussion session.

## 15. IBM Bob References
- IBM Bob Skills documentation: https://bob.ibm.com/docs/ide/features/skills
- IBM SkillsBuild - Learn with IBM Bob: https://skillsbuild.org/learn-with-ibm-bob
