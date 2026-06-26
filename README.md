# CardioSense

CardioSense is an interactive heart disease risk analysis and prediction project built with Python and Streamlit. It combines reproducible exploratory data analysis, probability-driven statistical inference, and an interpretable logistic regression model to demonstrate the full machine learning workflow for a healthcare dataset.

## Project Overview

This repository includes:
- A clean data analysis pipeline in `src/analysis.py`
- A polished Streamlit dashboard in `src/app.py`
- The heart disease dataset in `heart.csv`
- Generated visual artifacts in `outputs/`
- Dependencies defined in `requirements.txt`

The application delivers:
- dataset profiling and missing-value checking
- descriptive statistics and 95% confidence intervals
- conditional probability summaries for target prevalence
- probability calculations using binomial and normal models
- a logistic regression prediction pipeline with preprocessing
- model evaluation via accuracy, precision, recall, F1 score, ROC AUC, and confusion matrix
- an interactive dashboard with user-driven risk estimation

## Key Features

- Dataset summary with total observations, variable counts, and target prevalence
- Descriptive statistics for age, blood pressure, cholesterol, heart rate, and ST depression
- 95% confidence intervals for numeric predictors
- Probability summaries for heart disease by sex and cholesterol risk threshold
- Visual analytics including distribution histograms, count plots, boxplots, and a correlation heatmap
- Logistic regression model training with scaling and one-hot encoding
- A Streamlit interface for exploring data and estimating patient risk

## Dataset

- File: `heart.csv`
- Target column: `target`
  - `1` = presence of heart disease
  - `0` = absence of heart disease
- Key predictors include:
  - `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`

## Repository Structure

```
heart.csv
requirements.txt
README.md
DEMO_SCRIPT.md
Project_Description.md
Project_Description.txt
outputs/
src/
  analysis.py
  app.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Run the analysis pipeline

```bash
python src/analysis.py
```

This command:
- loads the heart disease dataset
- computes dataset summary and descriptive statistics
- calculates confidence intervals and probability summaries
- trains a logistic regression model
- saves visual artifacts to `outputs/`

### Launch the interactive dashboard

```bash
streamlit run src/app.py
```

The dashboard provides:
- an overview of dataset characteristics
- detailed descriptive statistics and confidence intervals
- probability summaries for risk factors
- visualizations of distributions and correlations
- model performance metrics and ROC curve
- an interactive form for predicting risk from patient inputs

## Outputs

Generated plot files are saved in `outputs/`, including:
- `target_count.png`
- `hist_age.png`
- `hist_chol.png`
- `hist_trestbps.png`
- `hist_thalach.png`
- `hist_oldpeak.png`
- `correlation_heatmap.png`
- `chol_by_target.png`

## Tech Stack

- Python
- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- seaborn
- Streamlit
- python-docx

## Notes

- This repository is intended for educational and portfolio use.
- It is not a medical diagnostic application.
- The analysis is based on an open-source heart disease dataset and should be used for demonstration only.
