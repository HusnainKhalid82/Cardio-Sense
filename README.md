# Heart Disease Detection (CardioSense)

Exploratory data analysis, statistical modeling, and an interactive Streamlit demo for the heart disease dataset with reproducible report generation.

CardioSense is a probability-and-statistics driven project that analyzes the heart disease dataset and delivers an interactive Streamlit dashboard for exploration and risk prediction. The project includes EDA, confidence intervals, probability summaries, a logistic regression model, and a report generator.

## Highlights

- Descriptive statistics, confidence intervals, and probability summaries on key features
- Logistic regression model with evaluation metrics
- Streamlit app for visual exploration and prediction
- Automated report generation with charts

## Dataset

- Source: Heart Disease Dataset (Kaggle)
- Link: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset?select=heart.csv
- File: heart.csv
- Target: target (1 = presence of heart disease, 0 = absence)

## Repository Structure

```
heart.csv
requirements.txt
src/
  analysis.py
  app.py
outputs/        # generated charts
report/
  build_report.py
  report_template.md
```

## Quickstart

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run analysis

```bash
python src/analysis.py
```

### 3) Launch the Streamlit app

```bash
streamlit run src/app.py
```

### 4) Build the report (DOCX)

```bash
python report/build_report.py
```

## What the Project Does

- Loads and summarizes the dataset
- Computes descriptive statistics and confidence intervals
- Calculates probability summaries (e.g., conditional probabilities, binomial sample probability)
- Trains a logistic regression model with preprocessing (scaling + one-hot encoding)
- Generates visualizations (count plot, histograms, correlation heatmap, boxplots)
- Provides a Streamlit dashboard to explore the data and run predictions
- Produces a formatted report with charts and metrics

## Outputs

- Plots are saved to outputs/ (e.g., target_count.png, correlation_heatmap.png)
- The report is saved to report/report_submission.docx

## Tech Stack

- Python
- pandas, numpy, scipy
- scikit-learn
- matplotlib, seaborn
- streamlit
- python-docx

## Notes

This project is for educational purposes and should not be used as a medical diagnostic tool.
