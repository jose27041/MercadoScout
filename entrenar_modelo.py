"""
MercadoScout - Entrenamiento binario, experimentos de overfitting y serialización.
Target: 1 = Élite (Top 25% del valor de mercado); 0 = Estándar.
Genera:
- Comparación Accuracy Train/Test: árbol sin poda vs. árbol con pre-poda.
- Curvas ROC comparativas: árbol sin poda, árbol podado y Random Forest.
- Matriz de confusión y Feature Importance Gini del Random Forest.
- Modelo serializado y métricas JSON.
"""
import json
import os
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score,
    RocCurveDisplay
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.database.conector import RepositorioTransfermarkt
from src.pipeline.features import (
    CATEGORIAS, COLUMNAS_CATEGORICAS, COLUMNAS_NUMERICAS,
    MedianaAgrupadaPorPosicion, asignar_categoria, construir_variables
)

FECHA_REFERENCIA = "2025-09-01"
TEMPORADA_MINIMA = 2024
PERCENTIL_OBJETIVO = 0.75
SEMILLA = 7
RUTA_MODELO = "models/modelo_scouting_rf.joblib"
CARPETA_FIGURAS = "reports/figures"
RUTA_METRICAS = "reports/metricas.json"


def crear_preprocesador_y_pipeline(clasificador):
    """El imputador agrupado se ajusta dentro del pipeline solo con datos de entrenamiento."""
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), COLUMNAS_NUMERICAS),
        ("cat", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), COLUMNAS_CATEGORICAS),
    ])
    return Pipeline([
        ("mediana_agrupada", MedianaAgrupadaPorPosicion()),
        ("prep", pre),
        ("clf", clasificador),
    ])


def crear_pipeline_rf():
    clf = RandomForestClassifier(
        class_weight="balanced", random_state=SEMILLA, n_jobs=-1, oob_score=False
    )
    return crear_preprocesador_y_pipeline(clf)


def guardar_grafico_accuracy(resultados, ruta):
    etiquetas = ["Árbol sin poda", "Árbol con pre-poda"]
    train = [resultados["arbol_sin_poda"]["accuracy_train"],
             resultados["arbol_con_poda"]["accuracy_train"]]
    test = [resultados["arbol_sin_poda"]["accuracy_test"],
            resultados["arbol_con_poda"]["accuracy_test"]]

    x = np.arange(len(etiquetas))
    ancho = 0.34
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - ancho / 2, train, ancho, label="Train")
    b2 = ax.bar(x + ancho / 2, test, ancho, label="Test")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas)
    ax.set_title("Diagnóstico de Overfitting: Accuracy Train vs. Test")
    ax.legend()
    ax.bar_label(b1, fmt="%.3f", padding=3)
    ax.bar_label(b2, fmt="%.3f", padding=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ruta, dpi=180)
    plt.close(fig)


def main():
    print("=" * 72)
    print("MERCADOSCOUT | Clasificación binaria: Estándar vs Élite")
    print("=" * 72)

    repo = RepositorioTransfermarkt()
    crudo = repo.jugadores_activos(FECHA_REFERENCIA, TEMPORADA_MINIMA)
    crudo["valor_mercado"] = pd.to_numeric(crudo["valor_mercado"], errors="coerce")
    crudo = crudo[
        crudo["valor_mercado"].notna() & (crudo["valor_mercado"] > 0)
    ].reset_index(drop=True)

    if crudo.empty:
        raise ValueError("La consulta no devolvió jugadores con valor de mercado válido.")

    umbral_p75 = float(crudo["valor_mercado"].quantile(PERCENTIL_OBJETIVO))
    y = asignar_categoria(crudo["valor_mercado"], umbral_p75)
    X = construir_variables(crudo)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEMILLA, stratify=y
    )

    # 1) Experimento solicitado: árbol de decisión sin restricciones.
    arbol_sin_poda = crear_preprocesador_y_pipeline(
        DecisionTreeClassifier(
            random_state=SEMILLA, class_weight="balanced",
            max_depth=None, min_samples_leaf=1, min_samples_split=2
        )
    )
    arbol_sin_poda.fit(X_train, y_train)
    pred_arbol_libre = arbol_sin_poda.predict(X_test)
    proba_arbol_libre = arbol_sin_poda.predict_proba(X_test)[:, 1]

    # 2) Árbol con pre-poda: límites fijados antes del entrenamiento.
    arbol_con_poda = crear_preprocesador_y_pipeline(
        DecisionTreeClassifier(
            random_state=SEMILLA, class_weight="balanced",
            max_depth=8, min_samples_leaf=5, min_samples_split=10
        )
    )
    arbol_con_poda.fit(X_train, y_train)
    pred_arbol_podado = arbol_con_poda.predict(X_test)
    proba_arbol_podado = arbol_con_poda.predict_proba(X_test)[:, 1]

    metricas_arboles = {
        "arbol_sin_poda": {
            "accuracy_train": float(accuracy_score(y_train, arbol_sin_poda.predict(X_train))),
            "accuracy_test": float(accuracy_score(y_test, pred_arbol_libre)),
            "auc_test": float(roc_auc_score(y_test, proba_arbol_libre)),
            "parametros": {"max_depth": None, "min_samples_leaf": 1, "min_samples_split": 2}
        },
        "arbol_con_poda": {
            "accuracy_train": float(accuracy_score(y_train, arbol_con_poda.predict(X_train))),
            "accuracy_test": float(accuracy_score(y_test, pred_arbol_podado)),
            "auc_test": float(roc_auc_score(y_test, proba_arbol_podado)),
            "parametros": {"max_depth": 8, "min_samples_leaf": 5, "min_samples_split": 10}
        }
    }

    # 3) Optimización del Random Forest.
    parametros = {
        "clf__n_estimators": [200, 350],
        "clf__max_depth": [None, 8, 14],
        "clf__min_samples_leaf": [1, 3, 5],
        "clf__max_features": ["sqrt", 0.8],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    busqueda = GridSearchCV(
        crear_pipeline_rf(), parametros, scoring="f1", cv=cv, n_jobs=-1, verbose=1
    )
    busqueda.fit(X_train, y_train)
    modelo = busqueda.best_estimator_

    pred = modelo.predict(X_test)
    proba = modelo.predict_proba(X_test)[:, 1]
    metricas = {
        "accuracy_train": float(accuracy_score(y_train, modelo.predict(X_train))),
        "accuracy_test": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "f1_elite": float(f1_score(y_test, pred, pos_label=1)),
        "auc": float(roc_auc_score(y_test, proba)),
        "mejores_parametros": busqueda.best_params_,
        "mejor_f1_cv": float(busqueda.best_score_),
        "umbral_p75_eur": umbral_p75,
        "distribucion": {"Estándar": int((y == 0).sum()), "Élite": int((y == 1).sum())},
        "reporte": classification_report(
            y_test, pred, target_names=CATEGORIAS, output_dict=True
        ),
        "diagnostico_arboles": metricas_arboles,
    }

    os.makedirs(CARPETA_FIGURAS, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Matriz de confusión del Random Forest.
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, pred), display_labels=CATEGORIAS
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de confusión | Random Forest")
    fig.tight_layout()
    fig.savefig(f"{CARPETA_FIGURAS}/matriz_confusion.png", dpi=180)
    plt.close(fig)

    # Curvas ROC comparativas calculadas sobre el mismo conjunto de prueba.
    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(
        y_test, proba_arbol_libre,
        name=f"Árbol sin poda (AUC={metricas_arboles['arbol_sin_poda']['auc_test']:.3f})",
        ax=ax
    )
    RocCurveDisplay.from_predictions(
        y_test, proba_arbol_podado,
        name=f"Árbol con poda (AUC={metricas_arboles['arbol_con_poda']['auc_test']:.3f})",
        ax=ax
    )
    RocCurveDisplay.from_predictions(
        y_test, proba,
        name=f"Random Forest (AUC={metricas['auc']:.3f})",
        ax=ax
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar")
    ax.set_title("Curvas ROC comparativas | Árboles vs Random Forest")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{CARPETA_FIGURAS}/curva_roc_comparativa.png", dpi=180)
    # Mantener también el nombre anterior usado en informes previos.
    fig.savefig(f"{CARPETA_FIGURAS}/curva_roc.png", dpi=180)
    plt.close(fig)

    # Comparación de Accuracy Train/Test de los dos árboles.
    guardar_grafico_accuracy(
        metricas_arboles, f"{CARPETA_FIGURAS}/overfitting_arboles.png"
    )

    # Importancia Gini del Random Forest.
    nombres = modelo.named_steps["prep"].get_feature_names_out()
    importancias = modelo.named_steps["clf"].feature_importances_
    imp = pd.Series(importancias, index=nombres).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    imp.tail(15).plot(kind="barh", ax=ax)
    ax.set_title("Importancia de variables | Gini")
    ax.set_xlabel("Importancia")
    fig.tight_layout()
    fig.savefig(f"{CARPETA_FIGURAS}/feature_importance.png", dpi=180)
    plt.close(fig)

    paquete = {
        "pipeline": modelo,
        "categorias": CATEGORIAS,
        "umbral_p75_eur": umbral_p75,
        "posiciones": sorted(X["posicion"].dropna().unique().tolist()),
        "medianas": {c: float(X_train[c].median()) for c in COLUMNAS_NUMERICAS},
        "metricas_test": {
            k: metricas[k] for k in
            ("accuracy_train", "accuracy_test", "balanced_accuracy", "f1_elite", "auc")
        },
        "entrenado_en": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(paquete, RUTA_MODELO, compress=3)

    with open(RUTA_METRICAS, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2, default=float)

    print("\\nResultados Random Forest:")
    print(classification_report(y_test, pred, target_names=CATEGORIAS, digits=3))
    print(f"Umbral Top 25%: EUR {umbral_p75:,.0f}")
    print(f"Accuracy RF train/test: {metricas['accuracy_train']:.3f} / {metricas['accuracy_test']:.3f}")
    print(f"F1 Élite: {metricas['f1_elite']:.3f} | AUC RF: {metricas['auc']:.3f}")
    print("\\nDiagnóstico de overfitting (árboles):")
    for nombre, valores in metricas_arboles.items():
        print(
            f"{nombre}: Accuracy train/test = "
            f"{valores['accuracy_train']:.3f} / {valores['accuracy_test']:.3f} | "
            f"AUC = {valores['auc_test']:.3f}"
        )
    print(f"Modelo guardado en {RUTA_MODELO}")
    print(f"Gráficos guardados en {CARPETA_FIGURAS}/")


if __name__ == "__main__":
    main()
