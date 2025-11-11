import os
import argparse
import tempfile
import time

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from mlflow.models.signature import infer_signature


# Carga de datos
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)

    # Variable objetivo y predictores
    y = df["loan_status"].astype(int)  # 0=solvente, 1=riesgo
    X = df.drop(columns=["loan_status"])

    # Definición de variables categóricas y numéricas
    cat_cols = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return X, y, cat_cols, num_cols


# Pipeline con imputación + OneHot + RF
def build_pipeline(cat_cols, num_cols):
    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, num_cols),
            ("cat", categorical_tf, cat_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model = Pipeline(steps=[("pre", pre), ("rf", rf)])
    return model


# Matriz de confusión como PNG
def log_confusion_matrix(y_true, y_pred, run_dir, name="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Matriz de confusión")
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.tight_layout()
    path = os.path.join(run_dir, name)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# Main
def main(args):
    # MLflow config (variables de entorno tienen prioridad)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", args.tracking_uri or "")
    experiment   = os.getenv("MLFLOW_EXPERIMENT_NAME", args.experiment or "")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if experiment:
        mlflow.set_experiment(experiment)

    # Datos
    X, y, cat_cols, num_cols = load_data(args.csv)

    # Split estratificado
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # Modelo
    model = build_pipeline(cat_cols, num_cols)

    # Nombre dinámico de corrida (prefijo args.run_name + timestamp)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.run_name}{timestamp}" 

    with mlflow.start_run(run_name=run_name):
        # Hiperparámetros (override por CLI)
        # Nota: max_depth=0 -> None
        rf = model.named_steps["rf"]
        rf.set_params(
            n_estimators=args.n_estimators,
            max_depth=None if args.max_depth == 0 else args.max_depth,
            max_features=args.max_features,
            class_weight=None if args.class_weight == "none" else args.class_weight
        )

        # Log de parámetros
        mlflow.log_params({
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "max_features": args.max_features,
            "class_weight": args.class_weight,
            "test_size": args.test_size,
            "threshold": args.threshold
        })

        # Entrenamietno
        model.fit(Xtr, ytr)

        # Predicción
        proba = model.predict_proba(Xte)[:, 1]
        yhat  = (proba >= args.threshold).astype(int)

        # Métricas
        metrics = {
            "roc_auc": float(roc_auc_score(yte, proba)),
            "accuracy": float(accuracy_score(yte, yhat)),
            "precision": float(precision_score(yte, yhat, zero_division=0)),
            "recall": float(recall_score(yte, yhat, zero_division=0)),
        }
        mlflow.log_metrics(metrics)

        # Artefactos: reporte + matriz de confusión
        report = classification_report(yte, yhat, digits=4)
        with tempfile.TemporaryDirectory() as td:
            rpt_path = os.path.join(td, "classification_report.txt")
            with open(rpt_path, "w", encoding="utf-8") as f:
                f.write(report)
            mlflow.log_artifact(rpt_path, artifact_path="reports")

            cm_path = log_confusion_matrix(yte, yhat, td)
            mlflow.log_artifact(cm_path, artifact_path="figures")

        # Firma e input_example para evitar warning
        # (El pipeline espera features crudos; usamos una fila de Xte)
        X_example = Xte.iloc[[0]].copy()
        yhat_example = model.predict_proba(X_example)[:, 1]
        signature = infer_signature(X_example, yhat_example)

        # Log del modelo completo (pipeline) con firma y ejemplo
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=X_example,
            signature=signature
        )

        print("== Métricas ==")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        if tracking_uri:
            print("Tracking URI:", tracking_uri)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="Ruta a credit_risk_dataset.csv")

    # Hiperparámetros (editables por CLI)
    p.add_argument("--n_estimators", type=int, default=300)
    p.add_argument("--max_depth", type=int, default=0, help="0 = None")
    p.add_argument("--max_features", type=str, default="sqrt", choices=["sqrt", "log2", "auto"])
    p.add_argument("--class_weight", type=str, default="balanced", choices=["none", "balanced"])

    # Otross
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--run_name", type=str, default="rf_")  # prefijo; se concatena con timestamp
    p.add_argument("--tracking_uri", type=str, default="")
    p.add_argument("--experiment", type=str, default="")

    args = p.parse_args()
    main(args)
