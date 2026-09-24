---
name: agriinsight-review
description: Review the AgriInsight AI data analytics project for dataset fidelity, EDA quality, ML validation, reproducibility, documentation consistency, and responsible decision-support claims.
---

You are reviewing the AgriInsight AI internship project.

Before changing any file:
1. Read README.md, IBM_Bob_Project_Brief.txt, requirements.txt, AgriInsight_AI.ipynb, and AgriInsight_AI.py.
2. Confirm the dataset fields actually used by the code: Area, Item, Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, and hg/ha_yield.
3. Return a short review plan and list proposed changes. Do not edit files until the student approves the material changes.

During review:
- Check data cleaning for missing values, duplicates, numeric conversion, and invalid negative values.
- Check that EDA covers useful KPIs, crop/area comparisons, historical trends, and exploratory relationships.
- Treat correlation only as an exploratory association; do not claim causation.
- Check that Linear Regression is used as a baseline and Random Forest Regressor as the non-linear comparison model.
- Verify MAE, RMSE, and R2 are calculated on the held-out test split rather than hard-coded.
- Preserve the model-reliability gate and the distinction between prediction and agronomic advice.
- Do not invent soil variables, fertilizer doses, pesticide doses, or field measurements that are absent from the dataset.
- Check requirements.txt against actual imports.
- Check README.md and the project report for consistency with the submitted code and dataset scope.
- Keep the repository simple, readable, reproducible, and suitable for a student internship submission.

At the end, report:
- files checked,
- issues found,
- changes made after approval,
- remaining limitations,
- whether the required submission files are present: code file, requirements.txt, project report, README, and GitHub-ready repository.
