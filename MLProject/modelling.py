from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.utils import estimator_html_repr


RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "datasetco_preprocessing"
RUN_ID_PATH = PROJECT_ROOT / "run_id_tuning.txt"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "mlflow_tuning"
MODEL_ARTIFACT_DIR = ARTIFACT_DIR / "model"
MODEL_PATH = MODEL_ARTIFACT_DIR / "model.pkl"

PARAM_GRID: Dict[str, list] = {
    "n_estimators": [100, 200],
    "max_depth": [10, 12],
    "min_samples_split": [10],
    "min_samples_leaf": [4],
}


def ensure_directories_exist(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def configure_mlflow() -> None:
    dagshub_repo = os.getenv("DAGSHUB_REPO")
    dagshub_username = os.getenv("DAGSHUB_USERNAME")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")

    if dagshub_repo and dagshub_username and dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        try:
            import dagshub

            owner, repo = dagshub_repo.split("/", maxsplit=1)
            dagshub.init(repo_owner=owner, repo_name=repo, mlflow=True)
        except Exception:
            mlflow.set_tracking_uri(f"https://dagshub.com/{dagshub_repo}.mlflow")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif not dagshub_repo:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "shipment-delay-risk-ci")
    mlflow.set_experiment(experiment_name)


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    required = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    missing = [name for name in required if not (PROCESSED_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing processed data files: {missing}. Expected them in {PROCESSED_DIR}."
        )

    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test


def compute_metrics(model: RandomForestClassifier, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_proba),
    }


def save_confusion_matrix_artifact(
    y_true: pd.Series,
    y_pred: np.ndarray,
    filename: str,
    title: str,
) -> Path:
    path = ARTIFACT_DIR / filename
    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["On Time", "Late"])
    display.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def save_shap_summary_artifact(model: RandomForestClassifier, X_test: pd.DataFrame) -> Path:
    path = ARTIFACT_DIR / "shap_summary_tuning.png"
    sample_size = int(os.getenv("SHAP_SAMPLE_SIZE", "500"))
    sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    class_values = shap_values[1] if isinstance(shap_values, list) else shap_values
    if hasattr(class_values, "ndim") and class_values.ndim == 3:
        class_values = class_values[:, :, 1]
    plt.figure()
    shap.summary_plot(class_values, sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def save_estimator_html_artifact(model: RandomForestClassifier) -> Path:
    path = ARTIFACT_DIR / "estimator.html"
    path.write_text(estimator_html_repr(model), encoding="utf-8")
    return path


def save_metric_info_artifact(
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    best_params: Dict[str, object],
    best_cv_f1: float,
) -> Path:
    path = ARTIFACT_DIR / "metric_info.json"
    metric_info = {
        "best_cv_f1": best_cv_f1,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "best_params": best_params,
        "scoring": "f1",
    }
    path.write_text(json.dumps(metric_info, indent=2), encoding="utf-8")
    return path


def run_tuning() -> GridSearchCV:
    configure_mlflow()
    ensure_directories_exist([ARTIFACT_DIR, MODEL_ARTIFACT_DIR])

    X_train, X_test, y_train, y_test = load_processed_data()
    with mlflow.start_run(run_name=os.getenv("MLFLOW_RUN_NAME", "ci-random-forest-tuning")) as run:
        search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            param_grid=PARAM_GRID,
            scoring=make_scorer(f1_score, zero_division=0),
            cv=3,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        train_metrics = compute_metrics(best_model, X_train, y_train)
        test_metrics = compute_metrics(best_model, X_test, y_test)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("tuning_method", "GridSearchCV")
        mlflow.log_param("cv", search.cv)
        mlflow.log_param("scoring", "f1")
        mlflow.log_param("feature_count", X_train.shape[1])
        mlflow.log_param("train_rows", X_train.shape[0])
        mlflow.log_param("test_rows", X_test.shape[0])
        mlflow.log_param("param_grid", str(PARAM_GRID))
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_cv_f1", float(search.best_score_))
        mlflow.log_metrics({f"train_{key}": value for key, value in train_metrics.items()})
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
        mlflow.log_metric("test_f1_from_scorer", float(f1_score(y_test, y_test_pred, zero_division=0)))

        artifact_paths = [
            save_confusion_matrix_artifact(
                y_train,
                y_train_pred,
                "training_confusion_matrix.png",
                "Tuned Random Forest - Training Confusion Matrix",
            ),
            save_confusion_matrix_artifact(
                y_test,
                y_test_pred,
                "test_confusion_matrix.png",
                "Tuned Random Forest - Test Confusion Matrix",
            ),
            save_shap_summary_artifact(best_model, X_test),
            save_estimator_html_artifact(best_model),
            save_metric_info_artifact(
                train_metrics,
                test_metrics,
                search.best_params_,
                float(search.best_score_),
            ),
        ]
        for path in artifact_paths:
            mlflow.log_artifact(str(path), artifact_path="tuning")

        if MODEL_ARTIFACT_DIR.exists():
            shutil.rmtree(MODEL_ARTIFACT_DIR)
        MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.sklearn.save_model(
            sk_model=best_model,
            path=str(MODEL_ARTIFACT_DIR),
            input_example=X_test.head(3),
        )
        joblib.dump(best_model, MODEL_PATH)
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=X_test.head(3),
            registered_model_name=os.getenv("MLFLOW_REGISTERED_MODEL_NAME") or None,
        )
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model_pickle")
        RUN_ID_PATH.write_text(run.info.run_id, encoding="utf-8")

    return search


if __name__ == "__main__":
    run_tuning()
