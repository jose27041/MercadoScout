"""Acceso a la base de datos transfermarkt (MySQL / MariaDB)."""
import os
from contextlib import contextmanager

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

load_dotenv()


def _url_desde_entorno() -> URL:
    """Construye la URL de conexión escapando correctamente la contraseña."""
    password = os.getenv("DB_PASSWORD")
    if not password:
        raise RuntimeError("Falta configurar DB_PASSWORD en las variables de entorno o en .env")
    return URL.create(
        drivername="mysql+pymysql",
        username=os.getenv("DB_USER", "root"),
        password=password,
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        database=os.getenv("DB_NAME", "transferfutbol"),
        query={"charset": "utf8mb4"},
    )


class RepositorioTransfermarkt:
    """Repositorio de solo lectura sobre la tabla players."""

    def __init__(self):
        self.engine = create_engine(_url_desde_entorno(), pool_pre_ping=True)

    @contextmanager
    def _conexion(self):
        with self.engine.connect() as con:
            yield con

    def consultar(self, sql: str, **parametros) -> pd.DataFrame:
        with self._conexion() as con:
            df = pd.read_sql_query(text(sql), con, params=parametros)
        print(f"[BD] {len(df):,} filas recuperadas")
        return df

    def jugadores_activos(self, fecha_referencia: str, temporada_minima: int) -> pd.DataFrame:
        """Jugadores activos con valor de mercado: posición, edad y estatura."""
        sql = """
            SELECT
                p.player_id,
                p.name                                   AS jugador,
                p.position                               AS posicion,
                p.height_in_cm                           AS estatura_cm,
                TIMESTAMPDIFF(YEAR, p.date_of_birth, :ref)  AS edad,
                p.market_value_in_eur                    AS valor_mercado
            FROM players p
            WHERE p.market_value_in_eur IS NOT NULL
              AND p.date_of_birth IS NOT NULL
              AND p.last_season >= :temporada
        """
        return self.consultar(sql, ref=fecha_referencia, temporada=temporada_minima)
