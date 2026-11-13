"""Métricas homogéneas para los experimentos de clasificación ASL.

Estas utilidades no entrenan ni seleccionan modelos. Reciben las etiquetas
reales y predichas de una partición ya definida para que todos los modelos se
reporten con el mismo criterio.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


COMPARISON_COLUMNS = [
    "modelo",
    "configuración",
    "aumentación",
    "accuracy_test",
    "precision_macro_test",
    "recall_macro_test",
    "f1_macro_test",
    "tiempo_entrenamiento_s",
    "observaciones",
]


def evaluate_classification(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> dict[str, pd.DataFrame | dict[str, float]]:
    """Calcula métricas globales, por clase y la matriz de confusión.

    ``labels`` debe ser el vocabulario fijo de las particiones oficiales; así
    las filas y columnas de cada matriz quedan comparables incluso si un modelo
    no predice alguna clase. ``zero_division=0`` evita valores indefinidos sin
    ocultar el caso: precisión, recall o F1 quedan en cero para esa clase.
    """
    if len(labels) == 0:
        raise ValueError("labels debe contener al menos una clase.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred deben tener la misma longitud.")
    if len(y_true) == 0:
        raise ValueError("Se requiere al menos una predicción para evaluar.")

    ordered_labels = list(labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=ordered_labels,
        output_dict=True,
        zero_division=0,
    )
    per_class = (
        pd.DataFrame(report).T.loc[ordered_labels, ["precision", "recall", "f1-score", "support"]]
        .rename(columns={"f1-score": "f1"})
        .rename_axis("clase")
        .reset_index()
    )
    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
    }
    matrix = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=ordered_labels),
        index=pd.Index(ordered_labels, name="real"),
        columns=pd.Index(ordered_labels, name="predicción"),
    )
    return {"summary": summary, "per_class": per_class, "confusion_matrix": matrix}


def comparison_template() -> pd.DataFrame:
    """Devuelve la tabla vacía que se llena al concluir cada experimento."""
    return pd.DataFrame(columns=COMPARISON_COLUMNS)
