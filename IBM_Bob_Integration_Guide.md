# IBM Bob Integration Guide - AgriInsight AI

## How IBM Bob is integrated
AgriInsight AI includes a repository-level IBM Bob integration, not a hidden prediction API. The project contains a Bob-ready project brief and a project-level Bob Skill at `.bob/skills/agriinsight-review/SKILL.md`.

The Python notebook and Streamlit application remain the reproducible source of truth for data cleaning, EDA, model evaluation and yield predictions.

## Bob workflow for this project
1. Open the AgriInsight AI project folder in IBM Bob.
2. Make sure the project files and dataset (or dataset source) are available in the workspace.
3. In Bob Advanced mode, invoke the project skill with `use_skill agriinsight-review` if it is not activated automatically.
4. Give Bob the context in `IBM_Bob_Project_Brief.txt`.
5. Ask Bob to return a plan and proposed changes before editing files.
6. Review and approve only relevant changes.
7. Re-check code, `requirements.txt`, README and the report for consistency.
8. Keep the final repository reproducible and GitHub-ready.

## Files Bob should review
- `AgriInsight_AI.ipynb`
- `AgriInsight_AI.py`
- `requirements.txt`
- `README.md`
- `AgriInsight_AI_Project_Report.docx`
- `IBM_Bob_Project_Brief.txt`
- `yield_df.csv` if stored locally, or the dataset source linked in README

## Constraints
- Do not create variables that are not present in the dataset.
- Do not turn correlation into causal claims.
- Do not generate fertilizer or pesticide dose prescriptions.
- Keep MAE, RMSE and R2 tied to the actual held-out evaluation.
- Preserve reliability and responsible-use notes.

## In-app handoff
The Streamlit application includes an **IBM Bob Integration** tab. It builds a Bob-ready brief from the cleaned dataset profile and current model-validation table and allows the brief to be downloaded for review in IBM Bob.
