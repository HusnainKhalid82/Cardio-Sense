from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "target"

NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion: List[List[int]]
    roc_curve_points: Tuple[List[float], List[float], List[float]]


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def dataset_summary(df: pd.DataFrame) -> Dict[str, object]:
    summary = {
        "rows": int(df.shape[0]),
        "columns": df.columns.tolist(),
        "missing_values": df.isna().sum().to_dict(),
        "target_rate": float(df[TARGET_COLUMN].mean()) if TARGET_COLUMN in df else None,
    }
    return summary


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe().T


def confidence_intervals(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    if columns is None:
        columns = NUMERIC_COLUMNS

    results = []
    for col in columns:
        series = df[col].dropna().astype(float)
        n = len(series)
        if n == 0:
            continue
        mean = series.mean()
        std = series.std(ddof=1)
        alpha = 1 - confidence
        t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        margin = t_crit * std / np.sqrt(n)
        results.append(
            {
                "column": col,
                "mean": mean,
                "std": std,
                "n": n,
                "ci_low": mean - margin,
                "ci_high": mean + margin,
            }
        )

    return pd.DataFrame(results)


def probability_summary(df: pd.DataFrame) -> Dict[str, object]:
    summary: Dict[str, object] = {}

    if TARGET_COLUMN in df:
        target_rate = df[TARGET_COLUMN].mean()
        summary["target_rate"] = float(target_rate)

        male_rate = df.loc[df["sex"] == 1, TARGET_COLUMN].mean()
        female_rate = df.loc[df["sex"] == 0, TARGET_COLUMN].mean()
        summary["prob_target_given_male"] = float(male_rate)
        summary["prob_target_given_female"] = float(female_rate)

        sample_prob = stats.binom.pmf(7, 10, target_rate)
        summary["binomial_p_7_of_10"] = float(sample_prob)

    chol = df["chol"].dropna().astype(float)
    if len(chol) > 1:
        mu = float(chol.mean())
        sigma = float(chol.std(ddof=1))
        summary["cholesterol_normal_mu"] = mu
        summary["cholesterol_normal_sigma"] = sigma
        summary["cholesterol_gt_240"] = float(1 - stats.norm.cdf(240, loc=mu, scale=sigma))

    return summary


def _build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )
    return preprocessor


def train_logistic_regression(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, ModelMetrics]:
    features = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    X = df[features]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = _build_preprocessor()
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = ModelMetrics(
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred)),
        recall=float(recall_score(y_test, y_pred)),
        f1=float(f1_score(y_test, y_pred)),
        roc_auc=float(roc_auc_score(y_test, y_prob)),
        confusion=confusion_matrix(y_test, y_pred).tolist(),
        roc_curve_points=tuple([list(x) for x in roc_curve(y_test, y_prob)]),
    )

    return model, metrics


def generate_plots(df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots: Dict[str, Path] = {}

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    sns.countplot(x=TARGET_COLUMN, data=df)
    plt.title("Heart Disease Target Count")
    target_path = output_dir / "target_count.png"
    plt.tight_layout()
    plt.savefig(target_path, dpi=150)
    plt.close()
    plots["target_count"] = target_path

    for col in ["age", "chol", "trestbps", "thalach", "oldpeak"]:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        path = output_dir / f"hist_{col}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        plots[f"hist_{col}"] = path

    plt.figure(figsize=(7, 5))
    corr = df[NUMERIC_COLUMNS + [TARGET_COLUMN]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    heatmap_path = output_dir / "correlation_heatmap.png"
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    plots["correlation_heatmap"] = heatmap_path

    plt.figure(figsize=(6, 4))
    sns.boxplot(x=TARGET_COLUMN, y="chol", data=df)
    plt.title("Cholesterol by Heart Disease Target")
    box_path = output_dir / "chol_by_target.png"
    plt.tight_layout()
    plt.savefig(box_path, dpi=150)
    plt.close()
    plots["chol_by_target"] = box_path

    return plots


def run_full_analysis(csv_path: Path, output_dir: Path) -> Dict[str, object]:
    df = load_data(csv_path)
    summary = dataset_summary(df)
    stats_table = descriptive_stats(df)
    ci_table = confidence_intervals(df)
    prob = probability_summary(df)
    _, metrics = train_logistic_regression(df)
    plots = generate_plots(df, output_dir)

    return {
        "summary": summary,
        "descriptive_stats": stats_table,
        "confidence_intervals": ci_table,
        "probability": prob,
        "model_metrics": metrics,
        "plots": plots,
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    results = run_full_analysis(
        project_root / "heart.csv",
        project_root / "outputs",
    )

    print("Dataset summary:")
    print(results["summary"])
    print("\nConfidence intervals:")
    print(results["confidence_intervals"])
    print("\nModel metrics:")
    print(results["model_metrics"])
