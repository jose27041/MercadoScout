"""Clasifica prospectos de un CSV con el modelo binario MercadoScout."""
import sys
import joblib
import numpy as np
import pandas as pd
from src.pipeline.features import construir_variables

RUTA_MODELO = "models/modelo_scouting_rf.joblib"

def clasificar(df, paquete):
    X = construir_variables(df)
    clases = list(paquete["pipeline"].named_steps["clf"].classes_)
    probabilidades = paquete["pipeline"].predict_proba(X)
    p_elite = probabilidades[:, clases.index(1)] * 100
    salida = df.copy()
    salida["probabilidad_estandar"] = (100 - p_elite).round(1)
    salida["probabilidad_elite"] = p_elite.round(1)
    salida["prediccion"] = np.where(paquete["pipeline"].predict(X) == 1, "Élite", "Estándar")
    salida["clase"] = paquete["pipeline"].predict(X)
    return salida.sort_values("probabilidad_elite", ascending=False).reset_index(drop=True)

def main():
    entrada = sys.argv[1] if len(sys.argv) > 1 else "datos/prospectos_ejemplo.csv"
    destino = sys.argv[2] if len(sys.argv) > 2 else "reports/ranking_prospectos.csv"
    paquete = joblib.load(RUTA_MODELO)
    ranking = clasificar(pd.read_csv(entrada), paquete)
    print(ranking[["nombre","posicion","edad","estatura_cm","prediccion",
                   "probabilidad_elite","probabilidad_estandar"]].to_string(index=False))
    ranking.to_csv(destino, index=False)
    print(f"\n[OK] Ranking guardado en {destino}")

if __name__ == "__main__":
    main()
