"""API REST de MercadoScout: clasificación binaria Estándar / Élite."""
from functools import lru_cache
from typing import List, Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.pipeline.features import construir_variables

RUTA_MODELO = "models/modelo_scouting_rf.joblib"

@lru_cache(maxsize=1)
def cargar_paquete():
    try:
        return joblib.load(RUTA_MODELO)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Modelo no encontrado. Ejecuta primero: python entrenar_modelo.py") from exc

app = FastAPI(
    title="MercadoScout API",
    description="Clasificación binaria de futbolistas: Élite (Top 25% del valor de mercado) o Estándar.",
    version="3.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Perfil(BaseModel):
    nombre: str = Field("Jugador sin nombre", max_length=80)
    posicion: Literal["Attack", "Midfield", "Defender", "Goalkeeper"]
    edad: Optional[int] = Field(None, ge=15, le=45)
    estatura_cm: Optional[float] = Field(None, ge=150, le=210)

class Resultado(BaseModel):
    nombre: str
    prediccion: Literal["Estándar", "Élite"]
    clase: int
    probabilidad_elite: float
    confianza: float
    probabilidades: dict
    valores_imputados: List[str]
    umbral_referencia_eur: float

def predecir(perfiles: List[Perfil]):
    pk = cargar_paquete()
    df = pd.DataFrame([p.model_dump() for p in perfiles])
    X = construir_variables(df)
    predicciones = pk["pipeline"].predict(X)
    probabilidades = pk["pipeline"].predict_proba(X)
    clases = list(pk["pipeline"].named_steps["clf"].classes_)
    resultados = []
    for perfil, clase, probs in zip(perfiles, predicciones, probabilidades):
        p_elite = float(probs[clases.index(1)])
        categoria = "Élite" if int(clase) == 1 else "Estándar"
        resultados.append(Resultado(
            nombre=perfil.nombre,
            prediccion=categoria,
            clase=int(clase),
            probabilidad_elite=round(p_elite * 100, 1),
            confianza=round(max(p_elite, 1-p_elite) * 100, 1),
            probabilidades={
                "Estándar": round(float(probs[clases.index(0)]) * 100, 1),
                "Élite": round(p_elite * 100, 1),
            },
            valores_imputados=[c for c in ("edad", "estatura_cm") if getattr(perfil, c) is None],
            umbral_referencia_eur=float(pk["umbral_p75_eur"]),
        ))
    return resultados

@app.get("/", include_in_schema=False)
def inicio():
    return FileResponse("web/index.html")

@app.get("/salud")
def salud():
    pk = cargar_paquete()
    return {
        "estado": "ok",
        "modelo": type(pk["pipeline"].named_steps["clf"]).__name__,
        "entrenado_en": pk["entrenado_en"],
        "metricas_test": pk["metricas_test"],
    }

@app.get("/opciones")
def opciones():
    pk = cargar_paquete()
    return {"posiciones": pk["posiciones"], "medianas": pk["medianas"],
            "umbral_p75_eur": pk["umbral_p75_eur"]}

@app.post("/predict", response_model=Resultado)
def predict(perfil: Perfil):
    return predecir([perfil])[0]

# Alias para conservar compatibilidad con versiones previas del frontend.
@app.post("/clasificar", response_model=Resultado)
def clasificar(perfil: Perfil):
    return predecir([perfil])[0]

@app.post("/predict/lote", response_model=List[Resultado])
def predict_lote(perfiles: List[Perfil]):
    if not 1 <= len(perfiles) <= 50:
        raise HTTPException(status_code=422, detail="El lote debe tener entre 1 y 50 perfiles")
    return sorted(predecir(perfiles), key=lambda r: r.probabilidad_elite, reverse=True)

@app.post("/clasificar/lote", response_model=List[Resultado])
def clasificar_lote(perfiles: List[Perfil]):
    return predict_lote(perfiles)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8001, reload=False)
