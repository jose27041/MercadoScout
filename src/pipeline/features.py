"""Transformaciones compartidas entre entrenamiento y API."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

COLUMNAS_NUMERICAS = ["edad", "estatura_cm"]
COLUMNAS_CATEGORICAS = ["posicion"]
COLUMNAS_MODELO = COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS
CATEGORIAS = ["Estándar", "Élite"]


def construir_variables(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in COLUMNAS_NUMERICAS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["posicion"] = out["posicion"].astype("object").replace("", np.nan).fillna("Missing")
    return out[COLUMNAS_MODELO]


def asignar_categoria(valores: pd.Series, umbral_p75: float) -> pd.Series:
    """0 = Estándar; 1 = Élite (valor de mercado >= percentil 75)."""
    return pd.Series(np.where(valores >= umbral_p75, 1, 0), index=valores.index, dtype="int64")


class MedianaAgrupadaPorPosicion(BaseEstimator, TransformerMixin):
    """Imputa edad y estatura con medianas de la posición; usa mediana global como respaldo."""

    def __init__(self, columnas_numericas=None, columna_grupo="posicion"):
        self.columnas_numericas = columnas_numericas
        self.columna_grupo = columna_grupo

    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy()
        cols = self.columnas_numericas or COLUMNAS_NUMERICAS
        self.columnas_ = list(df.columns)
        self.mediana_global_ = {
            col: pd.to_numeric(df[col], errors="coerce").median()
            for col in cols
        }
        self.mediana_grupo_ = {}
        for col in cols:
            valores = pd.to_numeric(df[col], errors="coerce")
            tabla = pd.DataFrame({
                self.columna_grupo: df[self.columna_grupo].astype("object").fillna("Missing"),
                col: valores
            })
            self.mediana_grupo_[col] = tabla.groupby(self.columna_grupo)[col].median().to_dict()
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=self.columnas_).copy()
        grupos = df[self.columna_grupo].astype("object").fillna("Missing")
        for col in self.mediana_global_:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            imputada = grupos.map(self.mediana_grupo_[col])
            imputada = imputada.fillna(self.mediana_global_[col])
            df[col] = df[col].fillna(imputada)
        return df
