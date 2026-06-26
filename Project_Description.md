**Project_Description**

CardioSense — Interactive Heart Disease Risk Prediction

CardioSense is an end-to-end data science project that analyzes a real-world heart disease dataset, summarizes key probability and statistical insights, trains an interpretable logistic regression model, and exposes an interactive Streamlit dashboard for exploration and individual risk estimation. The project demonstrates reproducible EDA, statistical inference (confidence intervals & probability summaries), model training/evaluation, visualizations, and automated report generation.

- **Role:** Data scientist & full-stack developer (EDA, modeling, dashboard)
- **Key features:** Exploratory data analysis, descriptive statistics, confidence intervals, probability summaries, logistic regression with preprocessing, ROC/confusion metrics, interactive Streamlit UI, and exportable report (DOCX)
- **Dataset:** Heart Disease dataset (Kaggle) — target variable is `target` (1 = disease, 0 = no disease)
- **Tech stack:** Python, pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, Streamlit, python-docx
- **Why it matters:** Conveys work across the full ML lifecycle — data cleaning, statistical reasoning, model building, interpretability, and user-facing productization — suitable for portfolio demonstration of applied statistics and ML engineering
- **What to look for in the demo:** interactive risk sliders/forms, model metrics (accuracy/precision/recall/F1/ROC-AUC), correlation heatmap, and saved visual artifacts in the `outputs/` folder

Quick run instructions:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/app.py
```

Optional short description for portfolio listing:

CardioSense is an interactive Streamlit dashboard and statistical analysis pipeline for predicting heart disease risk using logistic regression and probability-driven insights. It highlights reproducible EDA, model evaluation, and user-facing risk estimation.
