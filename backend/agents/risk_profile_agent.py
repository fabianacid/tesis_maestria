"""
Agente de Perfil de Riesgo del Inversor

Cuantifica la tolerancia al riesgo mediante el instrumento validado de
Grable & Lytton (1999) y genera recomendaciones de asignación sectorial
alineadas con ese perfil.

Fuente del instrumento:
  Grable, J., & Lytton, R. H. (1999). Financial Risk Tolerance Revisited:
  The Development of a Risk Assessment Instrument. Financial Services
  Review, 8(3), 163-181.
  Ampliamente reproducido (con su clave de puntuación) por servicios de
  extensión universitaria en EEUU (ej. Kansas State University Research
  and Extension, Rutgers Cooperative Extension) bajo el nombre "Risk
  Tolerance Quiz".

Metodología de scoring (reemplaza la tabla ad-hoc de versiones previas):
- 13 ítems, cada uno con 2 a 4 opciones de respuesta y su propio puntaje
  publicado (no inventado por el autor de esta tesis).
- Score crudo = suma de puntos de los 13 ítems (rango teórico 13-47/48
  según la edición del instrumento reproducida; el límite superior real
  de esta implementación se registra en `_MAX_SCORE_RAW`).
- Clasificación oficial de 5 niveles (Grable & Lytton, 1999):
    ≤ 18            → muy_conservador  ("low risk tolerance")
    19 - 22         → conservador      ("below-average risk tolerance")
    23 - 28         → moderado         ("average/moderate risk tolerance")
    29 - 32         → agresivo         ("above-average risk tolerance")
    ≥ 33            → muy_agresivo     ("high risk tolerance")
  La clasificación se aplica siempre sobre el score CRUDO (no el
  normalizado a 0-100); el score 0-100 es solo un reescalado lineal para
  el gauge visual del dashboard.

Salida:
  - Perfil de riesgo con score y descripción
  - Sectores recomendados con ETFs representativos y pesos target
  - Restricciones de cartera para integrar con PortfolioAgent
  - Presupuesto de VaR máximo tolerado
  - Coeficiente de aversión al riesgo (δ) por perfil, para Black-Litterman
    en PortfolioAgent (rango 2-10 citado en Bodie/Kane/Marcus, "Investments")
"""
import concurrent.futures
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Ítems del instrumento Grable & Lytton (1999)
# ──────────────────────────────────────────────

class Q01Autopercepcion(str, Enum):
    EVASOR = "evasor"
    CAUTELOSO = "cauteloso"
    CALCULADOR = "calculador"
    JUGADOR = "jugador"


class Q02ConcursoTV(str, Enum):
    EFECTIVO_1000 = "efectivo_1000"
    CHANCE_50_5000 = "chance_50_5000"
    CHANCE_25_10000 = "chance_25_10000"
    CHANCE_5_100000 = "chance_5_100000"


class Q03VacacionPerdidaEmpleo(str, Enum):
    CANCELAR = "cancelar"
    REDUCIR = "reducir"
    MANTENER_PLAN = "mantener_plan"
    EXTENDER = "extender"


class Q04Inversion20k(str, Enum):
    DEPOSITO_SEGURO = "deposito_seguro"
    BONOS_CALIDAD = "bonos_calidad"
    ACCIONES = "acciones"


class Q05ComodidadAcciones(str, Enum):
    NADA_COMODO = "nada_comodo"
    ALGO_COMODO = "algo_comodo"
    MUY_COMODO = "muy_comodo"


class Q06PalabraRiesgo(str, Enum):
    PERDIDA = "perdida"
    INCERTIDUMBRE = "incertidumbre"
    OPORTUNIDAD = "oportunidad"
    ADRENALINA = "adrenalina"


class Q07BonosVsActivosDuros(str, Enum):
    MANTENER_BONOS = "mantener_bonos"
    MITAD_ACTIVOS_DUROS = "mitad_activos_duros"
    TODO_ACTIVOS_DUROS = "todo_activos_duros"
    TODO_MAS_APALANCADO = "todo_mas_apalancado"


class Q08GananciaPerdidaPotencial(str, Enum):
    BAJO_RIESGO = "bajo_riesgo"          # +200 / 0
    MODERADO_BAJO = "moderado_bajo"      # +800 / -200
    MODERADO_ALTO = "moderado_alto"      # +2600 / -800
    ALTO_RIESGO = "alto_riesgo"          # +4800 / -2400


class Q09GananciaSegura(str, Enum):
    GANANCIA_SEGURA_500 = "ganancia_segura_500"
    APUESTA_50_1000 = "apuesta_50_1000"


class Q10PerdidaSegura(str, Enum):
    PERDIDA_SEGURA_500 = "perdida_segura_500"
    APUESTA_50_1000 = "apuesta_50_1000"


class Q11Herencia100k(str, Enum):
    AHORRO = "ahorro"
    FONDO_MIXTO = "fondo_mixto"
    ACCIONES_INDIVIDUALES = "acciones_individuales"
    COMMODITIES = "commodities"


class Q12Asignacion20k(str, Enum):
    MUY_CONSERVADORA = "muy_conservadora"     # 60/30/10 bajo/medio/alto riesgo
    CONSERVADORA = "conservadora"              # 40/40/20
    EQUILIBRADA = "equilibrada"                # 20/40/40
    AGRESIVA = "agresiva"                      # 10/30/60


class Q13MinaOro(str, Enum):
    NADA = "nada"
    UN_MES_SALARIO = "un_mes_salario"
    TRES_MESES_SALARIO = "tres_meses_salario"
    SEIS_MESES_SALARIO = "seis_meses_salario"


class PerfilRiesgo(str, Enum):
    MUY_CONSERVADOR = "muy_conservador"
    CONSERVADOR     = "conservador"
    MODERADO        = "moderado"
    AGRESIVO        = "agresivo"
    MUY_AGRESIVO    = "muy_agresivo"


# ──────────────────────────────────────────────
# Tablas de puntuación oficiales (Grable & Lytton, 1999)
# ──────────────────────────────────────────────

_SCORE_Q01: Dict[str, int] = {
    Q01Autopercepcion.EVASOR:     1,
    Q01Autopercepcion.CAUTELOSO:  2,
    Q01Autopercepcion.CALCULADOR: 3,
    Q01Autopercepcion.JUGADOR:    4,
}

_SCORE_Q02: Dict[str, int] = {
    Q02ConcursoTV.EFECTIVO_1000:    1,
    Q02ConcursoTV.CHANCE_50_5000:   2,
    Q02ConcursoTV.CHANCE_25_10000:  3,
    Q02ConcursoTV.CHANCE_5_100000:  4,
}

_SCORE_Q03: Dict[str, int] = {
    Q03VacacionPerdidaEmpleo.CANCELAR:      1,
    Q03VacacionPerdidaEmpleo.REDUCIR:       2,
    Q03VacacionPerdidaEmpleo.MANTENER_PLAN: 3,
    Q03VacacionPerdidaEmpleo.EXTENDER:      4,
}

_SCORE_Q04: Dict[str, int] = {
    Q04Inversion20k.DEPOSITO_SEGURO: 1,
    Q04Inversion20k.BONOS_CALIDAD:   2,
    Q04Inversion20k.ACCIONES:        3,
}

_SCORE_Q05: Dict[str, int] = {
    Q05ComodidadAcciones.NADA_COMODO: 1,
    Q05ComodidadAcciones.ALGO_COMODO: 2,
    Q05ComodidadAcciones.MUY_COMODO:  3,
}

_SCORE_Q06: Dict[str, int] = {
    Q06PalabraRiesgo.PERDIDA:        1,
    Q06PalabraRiesgo.INCERTIDUMBRE:  2,
    Q06PalabraRiesgo.OPORTUNIDAD:    3,
    Q06PalabraRiesgo.ADRENALINA:     4,
}

_SCORE_Q07: Dict[str, int] = {
    Q07BonosVsActivosDuros.MANTENER_BONOS:       1,
    Q07BonosVsActivosDuros.MITAD_ACTIVOS_DUROS:  2,
    Q07BonosVsActivosDuros.TODO_ACTIVOS_DUROS:   3,
    Q07BonosVsActivosDuros.TODO_MAS_APALANCADO:  4,
}

_SCORE_Q08: Dict[str, int] = {
    Q08GananciaPerdidaPotencial.BAJO_RIESGO:     1,
    Q08GananciaPerdidaPotencial.MODERADO_BAJO:   2,
    Q08GananciaPerdidaPotencial.MODERADO_ALTO:   3,
    Q08GananciaPerdidaPotencial.ALTO_RIESGO:     4,
}

_SCORE_Q09: Dict[str, int] = {
    Q09GananciaSegura.GANANCIA_SEGURA_500: 1,
    Q09GananciaSegura.APUESTA_50_1000:     3,
}

_SCORE_Q10: Dict[str, int] = {
    Q10PerdidaSegura.PERDIDA_SEGURA_500: 3,
    Q10PerdidaSegura.APUESTA_50_1000:    1,
}

_SCORE_Q11: Dict[str, int] = {
    Q11Herencia100k.AHORRO:                  1,
    Q11Herencia100k.FONDO_MIXTO:             2,
    Q11Herencia100k.ACCIONES_INDIVIDUALES:   3,
    Q11Herencia100k.COMMODITIES:             4,
}

_SCORE_Q12: Dict[str, int] = {
    Q12Asignacion20k.MUY_CONSERVADORA: 1,
    Q12Asignacion20k.CONSERVADORA:     2,
    Q12Asignacion20k.EQUILIBRADA:      3,
    Q12Asignacion20k.AGRESIVA:         4,
}

_SCORE_Q13: Dict[str, int] = {
    Q13MinaOro.NADA:                 1,
    Q13MinaOro.UN_MES_SALARIO:       2,
    Q13MinaOro.TRES_MESES_SALARIO:   3,
    Q13MinaOro.SEIS_MESES_SALARIO:   4,
}

# Rango teórico del score crudo: mínimo = suma de puntos mínimos (13 ítems × 1pt).
# Máximo = suma de puntos máximos por ítem (algunas ediciones publicadas citan 47,
# otras 48 según cuántas opciones tenga el ítem 12; se usa el máximo real de las
# tablas anteriores para el reescalado 0-100, la clasificación por cortes oficiales
# no depende de este valor porque el tramo superior es abierto ("33 y más")).
_MIN_SCORE_RAW = 13
_MAX_SCORE_RAW = (
    max(_SCORE_Q01.values()) + max(_SCORE_Q02.values()) + max(_SCORE_Q03.values())
    + max(_SCORE_Q04.values()) + max(_SCORE_Q05.values()) + max(_SCORE_Q06.values())
    + max(_SCORE_Q07.values()) + max(_SCORE_Q08.values()) + max(_SCORE_Q09.values())
    + max(_SCORE_Q10.values()) + max(_SCORE_Q11.values()) + max(_SCORE_Q12.values())
    + max(_SCORE_Q13.values())
)


# ──────────────────────────────────────────────
# Recomendaciones sectoriales por perfil
# Cada sector incluye: nombre, ETF representativo, peso target (suma=1)
# ──────────────────────────────────────────────

@dataclass
class SectorRecomendado:
    sector: str
    descripcion: str
    etf: str                              # ticker del activo representativo
    peso_target: float                    # 0-1, suma = 1 para el perfil
    # Métricas históricas (pobladas solo en selección dinámica)
    retorno_hist: Optional[float] = None  # % anualizado (lookback)
    volatilidad_hist: Optional[float] = None
    sharpe_hist: Optional[float] = None
    ranking_score: Optional[float] = None
    seleccion: str = "predefinida"        # "dinamica" | "predefinida"
    tipo: str = "ETF"                     # "ETF" | "Acción"
    señal_sentimiento: Optional[float] = None  # 0-1 (SentimentAgent)
    señal_prediccion:  Optional[float] = None  # 0-1 (momentum proxy)
    señal_sec:         Optional[float] = None  # 0-1 solo acciones (SECAgent)


_SECTORES_POR_PERFIL: Dict[str, List[SectorRecomendado]] = {
    PerfilRiesgo.MUY_CONSERVADOR: [
        SectorRecomendado("Bonos EEUU largo plazo",  "Renta fija de alta calidad, baja volatilidad",    "TLT",  0.35),
        SectorRecomendado("Bonos diversificados",    "Mercado total de renta fija EEUU",                "BND",  0.25),
        SectorRecomendado("Utilities",               "Electricidad y servicios regulados, muy estables","XLU",  0.20),
        SectorRecomendado("Consumer Staples",        "Bienes de consumo básico, demanda inelástica",    "XLP",  0.12),
        SectorRecomendado("Healthcare",              "Salud y farmacéuticas, sector defensivo",         "XLV",  0.08),
    ],
    PerfilRiesgo.CONSERVADOR: [
        SectorRecomendado("Bonos diversificados",    "Mercado total de renta fija EEUU",                "BND",  0.25),
        SectorRecomendado("Healthcare",              "Salud y farmacéuticas, sector defensivo",         "XLV",  0.18),
        SectorRecomendado("Consumer Staples",        "Bienes de consumo básico, demanda inelástica",    "XLP",  0.15),
        SectorRecomendado("Utilities",               "Electricidad y servicios regulados",              "XLU",  0.15),
        SectorRecomendado("Real Estate (REITs)",     "Inmuebles diversificados, generan dividendos",    "VNQ",  0.12),
        SectorRecomendado("Financials",              "Bancos y seguros, orientados a dividendos",       "XLF",  0.10),
        SectorRecomendado("S&P 500 (blend)",         "Exposición amplia al mercado EEUU",               "SPY",  0.05),
    ],
    PerfilRiesgo.MODERADO: [
        SectorRecomendado("Technology",              "Tecnología e innovación, crecimiento estructural", "XLK",  0.20),
        SectorRecomendado("Healthcare",              "Salud, mix de crecimiento y defensivo",           "XLV",  0.20),
        SectorRecomendado("Bonos diversificados",    "Colchón de renta fija EEUU",                      "BND",  0.20),
        SectorRecomendado("Financials",              "Bancos y seguros",                                "XLF",  0.15),
        SectorRecomendado("International (DM)",      "Mercados desarrollados fuera de EEUU (EFA)",      "EFA",  0.15),
        SectorRecomendado("Industrials",             "Manufactura e infraestructura",                   "XLI",  0.10),
    ],
    PerfilRiesgo.AGRESIVO: [
        SectorRecomendado("Technology",              "Tecnología e innovación, alto crecimiento",        "XLK",  0.30),
        SectorRecomendado("Small Cap EEUU",          "Empresas pequeñas, mayor potencial y riesgo",     "IWM",  0.20),
        SectorRecomendado("Consumer Discretionary",  "Consumo discrecional, beta alto",                 "XLY",  0.15),
        SectorRecomendado("Energy",                  "Petróleo, gas y energías, alta ciclicidad",       "XLE",  0.15),
        SectorRecomendado("Materials",               "Minería y materiales básicos",                    "XLB",  0.10),
        SectorRecomendado("International (DM)",      "Mercados desarrollados internacionales",          "EFA",  0.10),
    ],
    PerfilRiesgo.MUY_AGRESIVO: [
        SectorRecomendado("Technology",              "Tecnología e innovación, máximo crecimiento",     "XLK",  0.35),
        SectorRecomendado("Small Cap EEUU",          "Empresas pequeñas con alta beta",                 "IWM",  0.20),
        SectorRecomendado("Consumer Discretionary",  "Consumo discrecional, beta muy alto",             "XLY",  0.15),
        SectorRecomendado("Emerging Markets",        "Mercados emergentes, alta volatilidad y retorno", "EEM",  0.15),
        SectorRecomendado("Energy",                  "Petróleo y gas, alta ciclicidad",                 "XLE",  0.15),
    ],
}


# Presupuesto de VaR máximo diario tolerable (%) y volatilidad anual objetivo (%)
_RISK_BUDGET: Dict[str, Dict] = {
    PerfilRiesgo.MUY_CONSERVADOR: {"var_95_max": 5.0,  "vol_anual_max": 10.0, "max_peso_activo": 0.40},
    PerfilRiesgo.CONSERVADOR:     {"var_95_max": 10.0, "vol_anual_max": 15.0, "max_peso_activo": 0.40},
    PerfilRiesgo.MODERADO:        {"var_95_max": 15.0, "vol_anual_max": 20.0, "max_peso_activo": 0.50},
    PerfilRiesgo.AGRESIVO:        {"var_95_max": 25.0, "vol_anual_max": 30.0, "max_peso_activo": 0.60},
    PerfilRiesgo.MUY_AGRESIVO:    {"var_95_max": 40.0, "vol_anual_max": 45.0, "max_peso_activo": 0.65},
}

# Coeficiente de aversión al riesgo (δ) por perfil, para Black-Litterman en
# PortfolioAgent. Rango 2-10 citado como típico en Bodie, Z., Kane, A., &
# Marcus, A. (Investments) para el coeficiente de aversión al riesgo de un
# inversor individual; espaciado linealmente entre los 5 perfiles.
_DELTA_AVERSION_POR_PERFIL: Dict[str, float] = {
    PerfilRiesgo.MUY_CONSERVADOR: 10.0,
    PerfilRiesgo.CONSERVADOR:      8.0,
    PerfilRiesgo.MODERADO:         6.0,
    PerfilRiesgo.AGRESIVO:         4.0,
    PerfilRiesgo.MUY_AGRESIVO:     2.0,
}

# Universo de activos candidatos para selección dinámica
# (ticker, nombre_sector, descripcion, tipo)
UNIVERSO_ACTIVOS: List[Tuple[str, str, str, str]] = [
    # ── ETFs de renta fija (defensivos) ─────────────────────────────────
    ("TLT", "Bonos EEUU largo plazo",       "Bonos del Tesoro EEUU +20 años, alta duración",          "ETF"),
    ("BND", "Bonos diversificados",          "Mercado total de renta fija EEUU",                        "ETF"),
    ("IEF", "Bonos EEUU mediano plazo",      "Bonos del Tesoro EEUU 7-10 años",                         "ETF"),
    ("LQD", "Bonos corporativos IG",         "Bonos corporativos grado inversión EEUU",                 "ETF"),
    # ── ETFs sectoriales defensivos ─────────────────────────────────────
    ("GLD", "Oro",                           "Commodity refugio, baja correlación con equity",          "ETF"),
    ("XLU", "Utilities",                     "Electricidad y servicios regulados, defensivo",           "ETF"),
    ("XLP", "Consumer Staples",              "Bienes de consumo básico, demanda inelástica",            "ETF"),
    ("XLV", "Healthcare",                    "Salud y farmacéuticas, sector defensivo",                 "ETF"),
    ("VNQ", "Real Estate (REITs)",           "Inmuebles diversificados, generan dividendos",            "ETF"),
    # ── ETFs sectoriales moderados/agresivos ────────────────────────────
    ("XLF", "Financials",                    "Bancos y seguros",                                        "ETF"),
    ("XLI", "Industrials",                   "Manufactura e infraestructura",                           "ETF"),
    ("XLC", "Communications",                "Telecomunicaciones y medios digitales",                   "ETF"),
    ("XLK", "Technology",                    "Tecnología e innovación, alto crecimiento",               "ETF"),
    ("XLY", "Consumer Discretionary",        "Consumo discrecional, cíclico",                           "ETF"),
    ("XLE", "Energy",                        "Petróleo, gas y energías, alta ciclicidad",               "ETF"),
    ("XLB", "Materials",                     "Minería y materiales básicos",                            "ETF"),
    ("EFA", "International DM",              "Mercados desarrollados fuera de EEUU",                    "ETF"),
    ("VGK", "Europa",                        "Acciones europeas desarrolladas",                         "ETF"),
    ("EEM", "Emerging Markets",              "Mercados emergentes, alta volatilidad",                   "ETF"),
    ("IWM", "Small Cap EEUU",                "Empresas pequeñas EEUU, mayor potencial y riesgo",        "ETF"),
    ("MDY", "Mid Cap EEUU",                  "Empresas medianas EEUU",                                  "ETF"),
    ("QQQ", "Nasdaq 100",                    "Las 100 mayores empresas no financieras del Nasdaq",      "ETF"),
    # ── Acciones individuales defensivas (baja volatilidad) ─────────────
    ("KO",   "Coca-Cola",                    "Bebidas globales, dividendo estable, consumo defensivo",  "Acción"),
    ("PG",   "Procter & Gamble",             "Consumo básico global, márgenes estables, dividendo",     "Acción"),
    ("JNJ",  "Johnson & Johnson",            "Salud diversificada, farmacéutica y dispositivos",        "Acción"),
    ("WMT",  "Walmart",                      "Retail defensivo, ingresos estables en ciclos recesivos", "Acción"),
    ("MCD",  "McDonald's",                   "Franquicias globales, flujo de caja predecible",          "Acción"),
    ("VZ",   "Verizon",                      "Telecomunicaciones EEUU, alto dividendo, defensivo",      "Acción"),
    # ── Acciones individuales moderadas ─────────────────────────────────
    ("MSFT", "Microsoft",                    "Software cloud líder, crecimiento estructural sólido",    "Acción"),
    ("AAPL", "Apple",                        "Ecosistema tech y servicios, flujo de caja robusto",      "Acción"),
    ("JPM",  "JPMorgan Chase",               "Banco diversificado líder EEUU, dividendo creciente",     "Acción"),
    ("V",    "Visa",                         "Pagos digitales globales, altos márgenes operativos",     "Acción"),
    ("UNH",  "UnitedHealth",                 "Salud gestionada EEUU, crecimiento defensivo",            "Acción"),
    # ── Acciones individuales de alto crecimiento (agresivas) ───────────
    ("GOOGL","Alphabet (Google)",            "Publicidad digital y cloud, negocio altamente rentable",  "Acción"),
    ("AMZN", "Amazon",                       "E-commerce y AWS cloud, liderazgo en múltiples mercados", "Acción"),
    ("NVDA", "NVIDIA",                       "Chips para IA y data centers, crecimiento excepcional",   "Acción"),
    ("META", "Meta Platforms",               "Redes sociales y realidad aumentada, alta generación FCF","Acción"),
    ("TSLA", "Tesla",                        "Vehículos eléctricos y energía, alta beta y volatilidad", "Acción"),
]

# Alias para compatibilidad
UNIVERSO_ETF = UNIVERSO_ACTIVOS


# ──────────────────────────────────────────────
# Universo dinámico de acciones: componentes actuales del S&P 500
# filtrados por liquidez, en vez de una lista fija de 16 acciones
# elegidas a mano. Los 22 ETFs sectoriales de UNIVERSO_ACTIVOS se
# mantienen (son la taxonomía estándar GICS vía SPDR/iShares, no una
# opinión del autor) y se combinan con estas acciones dinámicas.
# ──────────────────────────────────────────────

_PRECIO_MINIMO_FIJO = 5.0            # SEC: umbral de "penny stock" (no configurable)
_PRECIO_MAXIMO_DEFAULT = 1000.0      # techo por defecto si el inversor no lo define
_VOLUMEN_MINIMO_USD_DEFAULT = 10_000_000.0  # ADV mínimo típico de liquidez institucional
_TOP_N_ACCIONES_POR_SECTOR = 5        # tope por sector GICS para acotar el fan-out de señales

_SP500_CACHE_TTL_HORAS = 24
_sp500_cache: Dict[str, object] = {"data": None, "timestamp": None}

_UNIVERSO_DINAMICO_CACHE_TTL_HORAS = 6
_universo_dinamico_cache: Dict[Tuple[float, float], Dict[str, object]] = {}


def _obtener_constituyentes_sp500() -> Optional[List[Tuple[str, str]]]:
    """
    (ticker, sector_gics) de los componentes actuales del S&P 500, vía la
    tabla de Wikipedia (fuente pública, se actualiza con cada cambio de
    índice). Cacheado 24h para no depender de Wikipedia en cada corrida.
    """
    now = datetime.now()
    cached = _sp500_cache.get("data")
    ts = _sp500_cache.get("timestamp")
    if cached is not None and ts is not None and (now - ts) < timedelta(hours=_SP500_CACHE_TTL_HORAS):
        return cached

    try:
        import io
        import requests
        import pandas as pd

        headers = {"User-Agent": "Mozilla/5.0 (compatible; TesisRiskProfileAgent/1.0)"}
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers, timeout=15,
        )
        resp.raise_for_status()
        tabla = pd.read_html(io.StringIO(resp.text))[0]

        resultado = [
            (str(row["Symbol"]).strip().replace(".", "-"), str(row["GICS Sector"]).strip())
            for _, row in tabla.iterrows()
        ]
        _sp500_cache["data"] = resultado
        _sp500_cache["timestamp"] = now
        logger.info(f"Constituyentes S&P 500 actualizados desde Wikipedia: {len(resultado)} tickers")
        return resultado
    except Exception as exc:
        logger.warning(f"No se pudo obtener constituyentes S&P 500 de Wikipedia: {exc}")
        return cached  # puede ser None si nunca se cacheó con éxito


def _filtrar_universo_acciones_por_liquidez(
    candidatos: List[Tuple[str, str]],
    precio_maximo: float,
    volumen_minimo_usd: float,
    periodo: str = "3mo",
) -> List[Tuple[str, str, str, str]]:
    """
    Filtra (ticker, sector_gics) por precio y volumen en dólares, y devuelve
    el top N por sector según volumen (formato compatible con UNIVERSO_ACTIVOS).

    - Precio: piso fijo $5 (definición SEC de penny stock) + techo definido
      por el inversor (accesibilidad para portafolios chicos).
    - Volumen: piso en dólares (precio × volumen) definido por el inversor.
    """
    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np
    except ImportError:
        return []

    tickers = [t for t, _ in candidatos]
    sector_map = dict(candidatos)

    try:
        raw = yf.download(tickers, period=periodo, auto_adjust=True, progress=False, threads=True)
        if raw is None or raw.empty or not isinstance(raw.columns, pd.MultiIndex):
            raise ValueError("Descarga de precio/volumen vacía o con formato inesperado")
        closes = raw["Close"]
        volumes = raw["Volume"]
    except Exception as exc:
        logger.warning(f"Descarga de precio/volumen del universo S&P 500 falló: {exc}")
        return []

    metricas = []
    for t in tickers:
        if t not in closes.columns or t not in volumes.columns:
            continue
        precio_serie = closes[t].dropna()
        vol_serie = volumes[t].dropna()
        if len(precio_serie) < 20 or len(vol_serie) < 20:
            continue
        precio_prom = float(precio_serie.mean())
        dollar_vol_prom = float((precio_serie * vol_serie.reindex(precio_serie.index)).mean())
        if not np.isfinite(precio_prom) or not np.isfinite(dollar_vol_prom):
            continue
        if precio_prom < _PRECIO_MINIMO_FIJO or precio_prom > precio_maximo:
            continue
        if dollar_vol_prom < volumen_minimo_usd:
            continue
        metricas.append({
            "ticker": t, "sector": sector_map.get(t, "Otro"),
            "precio_prom": precio_prom, "dollar_vol_prom": dollar_vol_prom,
        })

    por_sector: Dict[str, list] = {}
    for m in metricas:
        por_sector.setdefault(m["sector"], []).append(m)

    seleccion = []
    for _, items in por_sector.items():
        items.sort(key=lambda x: x["dollar_vol_prom"], reverse=True)
        seleccion.extend(items[:_TOP_N_ACCIONES_POR_SECTOR])

    return [
        (m["ticker"], m["sector"], f"Componente S&P 500 — sector {m['sector']}", "Acción")
        for m in seleccion
    ]


def _construir_universo_dinamico(
    precio_maximo: Optional[float],
    volumen_minimo_usd: Optional[float],
) -> List[Tuple[str, str, str, str]]:
    """
    Universo de activos para la selección dinámica: los 22 ETFs sectoriales
    fijos + acciones del S&P 500 filtradas por liquidez (precio/volumen).
    Si la construcción dinámica falla o deja muy pocas acciones, cae a la
    lista predefinida de 16 acciones de UNIVERSO_ACTIVOS (mismo patrón de
    fallback que el resto del agente).
    """
    etfs_fijos = [e for e in UNIVERSO_ACTIVOS if e[3] == "ETF"]
    precio_max = precio_maximo if precio_maximo is not None else _PRECIO_MAXIMO_DEFAULT
    vol_min = volumen_minimo_usd if volumen_minimo_usd is not None else _VOLUMEN_MINIMO_USD_DEFAULT

    cache_key = (round(precio_max, 2), round(vol_min, -3) if vol_min > 0 else 0.0)
    now = datetime.now()
    cached = _universo_dinamico_cache.get(cache_key)
    if cached is not None and (now - cached["timestamp"]) < timedelta(hours=_UNIVERSO_DINAMICO_CACHE_TTL_HORAS):
        return cached["data"]

    acciones_predefinidas = [e for e in UNIVERSO_ACTIVOS if e[3] == "Acción"]

    constituyentes = _obtener_constituyentes_sp500()
    if not constituyentes:
        logger.warning("Sin constituyentes S&P 500 disponibles — usando universo de acciones predefinido")
        resultado = etfs_fijos + acciones_predefinidas
    else:
        acciones_dinamicas = _filtrar_universo_acciones_por_liquidez(
            constituyentes, precio_max, vol_min
        )
        if len(acciones_dinamicas) < 5:
            logger.warning(
                f"Filtro de liquidez (precio<=${precio_max:.0f}, vol>=${vol_min:,.0f}) dejó solo "
                f"{len(acciones_dinamicas)} acciones — usando universo de acciones predefinido"
            )
            resultado = etfs_fijos + acciones_predefinidas
        else:
            logger.info(
                f"Universo dinámico S&P 500: {len(acciones_dinamicas)} acciones pasaron el filtro "
                f"de liquidez (precio<=${precio_max:.0f}, vol>=${vol_min:,.0f})"
            )
            resultado = etfs_fijos + acciones_dinamicas

    _universo_dinamico_cache[cache_key] = {"data": resultado, "timestamp": now}
    return resultado


# Factor de agresividad por perfil: 0 = puramente Sharpe, 1 = puramente retorno
_AGRESIVIDAD: Dict[str, float] = {
    PerfilRiesgo.MUY_CONSERVADOR: 0.0,
    PerfilRiesgo.CONSERVADOR:     0.20,
    PerfilRiesgo.MODERADO:        0.45,
    PerfilRiesgo.AGRESIVO:        0.70,
    PerfilRiesgo.MUY_AGRESIVO:    0.90,
}

_PERFIL_DESCRIPCION: Dict[str, str] = {
    PerfilRiesgo.MUY_CONSERVADOR: (
        "Prioriza preservar el capital sobre obtener altos retornos. "
        "Acepta rendimientos moderados a cambio de máxima estabilidad. "
        "Cartera dominada por renta fija y sectores defensivos."
    ),
    PerfilRiesgo.CONSERVADOR: (
        "Tolera pequeñas fluctuaciones para obtener retornos algo superiores a la inflación. "
        "Combina renta fija con renta variable defensiva y dividendos."
    ),
    PerfilRiesgo.MODERADO: (
        "Busca equilibrio entre crecimiento y protección. "
        "Acepta volatilidad moderada con horizonte de medio-largo plazo. "
        "Mezcla equitativa entre sectores cíclicos y defensivos."
    ),
    PerfilRiesgo.AGRESIVO: (
        "Orientado al crecimiento del capital. Acepta caídas significativas "
        "de corto plazo a cambio de mayor potencial de retorno. "
        "Exposición importante a tecnología, small caps y sectores cíclicos."
    ),
    PerfilRiesgo.MUY_AGRESIVO: (
        "Maximizar el retorno es el objetivo principal. "
        "Tolerancia alta a pérdidas temporales elevadas. "
        "Concentración en activos de alto crecimiento y mercados emergentes."
    ),
}


# ──────────────────────────────────────────────
# Dataclasses de resultado
# ──────────────────────────────────────────────

@dataclass
class DimensionScore:
    dimension: str
    respuesta: str
    puntos: int
    max_puntos: int
    interpretacion: str


@dataclass
class RiskProfileResult:
    perfil: PerfilRiesgo
    score: float                          # 0-100 (reescalado lineal, solo visual)
    score_raw: int                        # score crudo del instrumento (13-47/48)
    descripcion_perfil: str
    dimensiones: List[DimensionScore]
    sectores_recomendados: List[SectorRecomendado]
    tickers_recomendados: List[str]       # solo los ETFs, para pasar a PortfolioAgent
    pesos_sugeridos: List[float]          # alineados con tickers_recomendados
    risk_budget: Dict                     # var_95_max, vol_anual_max, max_peso_activo
    delta_aversion: float = 6.0           # δ para Black-Litterman (PortfolioAgent)
    advertencia: str = ""
    seleccion_dinamica: bool = False      # True si se usaron datos históricos reales
    periodo_analisis: str = ""            # e.g. "1y" o "predefinido"
    universo_evaluado: int = 0            # cuántos ETFs del universo se analizaron
    metodologia: str = "Grable & Lytton (1999) — Financial Risk Tolerance Scale, 13 ítems"
    fecha_analisis: datetime = field(default_factory=datetime.now)  # timestamp de la corrida


# ──────────────────────────────────────────────
# Agente principal
# ──────────────────────────────────────────────

class RiskProfileAgent:
    """
    Evalúa el perfil de riesgo del inversor (instrumento Grable & Lytton, 1999)
    y genera recomendaciones de asignación sectorial alineadas con su
    tolerancia al riesgo.

    Uso básico:
        agent = RiskProfileAgent()
        result = agent.evaluar(
            q01=Q01Autopercepcion.CALCULADOR,
            q02=Q02ConcursoTV.CHANCE_25_10000,
            q03=Q03VacacionPerdidaEmpleo.MANTENER_PLAN,
            q04=Q04Inversion20k.ACCIONES,
            q05=Q05ComodidadAcciones.MUY_COMODO,
            q06=Q06PalabraRiesgo.OPORTUNIDAD,
            q07=Q07BonosVsActivosDuros.MITAD_ACTIVOS_DUROS,
            q08=Q08GananciaPerdidaPotencial.MODERADO_ALTO,
            q09=Q09GananciaSegura.APUESTA_50_1000,
            q10=Q10PerdidaSegura.APUESTA_50_1000,
            q11=Q11Herencia100k.ACCIONES_INDIVIDUALES,
            q12=Q12Asignacion20k.EQUILIBRADA,
            q13=Q13MinaOro.TRES_MESES_SALARIO,
        )
    """

    def evaluar(
        self,
        q01: Q01Autopercepcion,
        q02: Q02ConcursoTV,
        q03: Q03VacacionPerdidaEmpleo,
        q04: Q04Inversion20k,
        q05: Q05ComodidadAcciones,
        q06: Q06PalabraRiesgo,
        q07: Q07BonosVsActivosDuros,
        q08: Q08GananciaPerdidaPotencial,
        q09: Q09GananciaSegura,
        q10: Q10PerdidaSegura,
        q11: Q11Herencia100k,
        q12: Q12Asignacion20k,
        q13: Q13MinaOro,
        usar_seleccion_dinamica: bool = True,
        lookback: str = "1y",
        precio_maximo: Optional[float] = None,
        volumen_minimo_usd: Optional[float] = None,
    ) -> RiskProfileResult:
        raw, dimensiones = self._calcular_score(
            q01, q02, q03, q04, q05, q06, q07, q08, q09, q10, q11, q12, q13
        )
        # Clasificación oficial Grable & Lytton (1999) sobre el score CRUDO.
        perfil = self._clasificar(raw)
        # Reescalado lineal 0-100 SOLO para el gauge visual del dashboard;
        # la clasificación ya se hizo arriba sobre el score crudo.
        score_norm = round(
            (raw - _MIN_SCORE_RAW) / (_MAX_SCORE_RAW - _MIN_SCORE_RAW) * 100, 1
        )

        # Selección de sectores: dinámica (datos reales) o predefinida (fallback)
        seleccion_dinamica = False
        universo_evaluado = 0
        if usar_seleccion_dinamica:
            sectores, seleccion_dinamica, universo_evaluado = self._seleccionar_sectores_dinamico(
                perfil, lookback=lookback,
                precio_maximo=precio_maximo, volumen_minimo_usd=volumen_minimo_usd,
            )
        else:
            sectores = _SECTORES_POR_PERFIL[perfil]

        tickers = [s.etf for s in sectores]
        pesos   = [s.peso_target for s in sectores]

        advertencia = ""
        if perfil in (PerfilRiesgo.MUY_AGRESIVO, PerfilRiesgo.AGRESIVO):
            advertencia = (
                "Este perfil implica alta volatilidad. "
                "Solo invierta capital que pueda mantener inmovilizado y cuya pérdida total pueda asumir."
            )

        periodo = lookback if seleccion_dinamica else "predefinido"
        logger.info(
            f"RiskProfileAgent: score_raw={raw} ({_MIN_SCORE_RAW}-{_MAX_SCORE_RAW}) → perfil={perfil.value} "
            f"| selección={'dinámica (' + lookback + ')' if seleccion_dinamica else 'predefinida'}"
        )

        return RiskProfileResult(
            perfil=perfil,
            score=score_norm,
            score_raw=raw,
            descripcion_perfil=_PERFIL_DESCRIPCION[perfil],
            dimensiones=dimensiones,
            sectores_recomendados=sectores,
            tickers_recomendados=tickers,
            pesos_sugeridos=pesos,
            risk_budget=_RISK_BUDGET[perfil],
            delta_aversion=_DELTA_AVERSION_POR_PERFIL[perfil],
            advertencia=advertencia,
            seleccion_dinamica=seleccion_dinamica,
            periodo_analisis=periodo,
            universo_evaluado=universo_evaluado,
        )

    # ------------------------------------------------------------------
    # Selección dinámica de sectores basada en datos históricos
    # ------------------------------------------------------------------

    def _seleccionar_sectores_dinamico(
        self,
        perfil: PerfilRiesgo,
        n_etfs: int = 6,
        lookback: str = "1y",
        precio_maximo: Optional[float] = None,
        volumen_minimo_usd: Optional[float] = None,
    ) -> Tuple[List[SectorRecomendado], bool, int]:
        """
        Selecciona los N mejores activos del universo para el perfil dado
        usando datos históricos reales de yfinance.

        El universo evaluado combina los 22 ETFs sectoriales fijos con
        acciones del S&P 500 filtradas dinámicamente por precio y volumen
        (ver `_construir_universo_dinamico`), en vez de una lista fija de
        acciones elegidas a mano.

        Algoritmo:
          1. Descarga `lookback` de retornos diarios para todo el universo.
          2. Calcula retorno anualizado, volatilidad y Sharpe por activo.
          3. Filtra activos cuya volatilidad supere el límite del perfil (con tolerancia 30%).
          4. Rankea por score ajustado al perfil:
               score = (1 - agresividad) * sharpe_normalizado
                     + agresividad * retorno_normalizado
             donde agresividad ∈ [0, 1] según el perfil.
          5. Toma los top N y asigna pesos por volatilidad inversa.

        Returns:
            (sectores, fue_dinamico, n_activos_evaluados)
        """
        try:
            import yfinance as yf
            import pandas as pd
        except ImportError:
            logger.warning("yfinance/pandas no disponibles — usando selección predefinida")
            return _SECTORES_POR_PERFIL[perfil], False, 0

        budget    = _RISK_BUDGET[perfil]
        vol_max   = budget["vol_anual_max"]
        agr       = _AGRESIVIDAD[perfil]
        rf        = 4.5   # tasa libre de riesgo %
        TRADING_DAYS = 252

        universo_actual = _construir_universo_dinamico(precio_maximo, volumen_minimo_usd)
        tickers_universo = [e[0] for e in universo_actual]
        etf_meta = {e[0]: {"sector": e[1], "descripcion": e[2], "tipo": e[3]} for e in universo_actual}

        # ── Descargar datos ──────────────────────────────────────────────
        try:
            raw = yf.download(
                tickers_universo, period=lookback,
                auto_adjust=True, progress=False, threads=True,
            )
            if raw is None or raw.empty:
                raise ValueError("Sin datos de yfinance")

            closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
            rets = closes.pct_change().dropna()
            if len(rets) < 60:
                raise ValueError("Historial insuficiente")
        except Exception as exc:
            logger.warning(f"Selección dinámica falló en descarga: {exc} — usando predefinida")
            return _SECTORES_POR_PERFIL[perfil], False, 0

        # ── Calcular métricas por ETF ────────────────────────────────────
        metricas = []
        for ticker in tickers_universo:
            if ticker not in rets.columns:
                continue
            serie = rets[ticker].dropna()
            if len(serie) < 30:
                continue
            ret_a  = float(serie.mean() * TRADING_DAYS * 100)
            vol_a  = float(serie.std()  * np.sqrt(TRADING_DAYS) * 100)
            sharpe = (ret_a - rf) / vol_a if vol_a > 0 else -999.0
            metricas.append({
                "ticker": ticker, "ret_a": ret_a,
                "vol_a": vol_a,   "sharpe": sharpe,
            })

        if len(metricas) < 3:
            logger.warning("Pocos ETFs con datos — usando selección predefinida")
            return _SECTORES_POR_PERFIL[perfil], False, len(metricas)

        universo_evaluado = len(metricas)

        # ── Filtrar por restricción de volatilidad del perfil ────────────
        # Multiplicador estricto para perfiles conservadores, permisivo para agresivos
        _VOL_MULT = {
            PerfilRiesgo.MUY_CONSERVADOR: 1.00,
            PerfilRiesgo.CONSERVADOR:     1.05,
            PerfilRiesgo.MODERADO:        1.10,
            PerfilRiesgo.AGRESIVO:        1.20,
            PerfilRiesgo.MUY_AGRESIVO:    1.30,
        }
        vol_mult   = _VOL_MULT.get(perfil, 1.10)
        candidatos = [m for m in metricas if m["vol_a"] <= vol_max * vol_mult]
        if len(candidatos) < n_etfs:
            candidatos = [m for m in metricas if m["vol_a"] <= vol_max * 1.50]
        if len(candidatos) < 2:
            logger.warning("Muy pocos candidatos tras filtro de vol — usando predefinida")
            return _SECTORES_POR_PERFIL[perfil], False, universo_evaluado

        # ── Señales de agentes (sentimiento, predicción momentum, SEC) ──
        try:
            agente_scores = self._obtener_señales_agentes(candidatos, closes, etf_meta)
        except Exception as e:
            logger.warning(f"_obtener_señales_agentes falló: {e}")
            agente_scores = {}

        # ── Score integrado: 60% histórico + 40% agentes ────────────────
        W_HIST  = 0.60
        W_AGENT = 0.40

        sharpes  = [m["sharpe"] for m in candidatos]
        retornos = [m["ret_a"]  for m in candidatos]
        s_min, s_rng = min(sharpes),  max(sharpes)  - min(sharpes)  or 1.0
        r_min, r_rng = min(retornos), max(retornos) - min(retornos) or 1.0

        for m in candidatos:
            s_norm      = (m["sharpe"] - s_min) / s_rng
            r_norm      = (m["ret_a"]  - r_min) / r_rng
            hist_score  = (1 - agr) * s_norm + agr * r_norm
            agent_data  = agente_scores.get(m["ticker"], {})
            agent_score = agent_data.get("combined", hist_score)
            m["score"]          = W_HIST * hist_score + W_AGENT * agent_score
            m["sentim_score"]   = agent_data.get("sentimiento")
            m["pred_score"]     = agent_data.get("prediccion")
            m["sec_score"]      = agent_data.get("sec")

        candidatos.sort(key=lambda x: x["score"], reverse=True)

        # ── Filtro greedy de correlación ─────────────────────────────────
        # Agrega ETFs en orden de score; descarta si correlación histórica
        # con alguno ya seleccionado supera el umbral del perfil.
        _CORR_UMBRAL = {
            PerfilRiesgo.MUY_CONSERVADOR: 0.75,
            PerfilRiesgo.CONSERVADOR:     0.80,
            PerfilRiesgo.MODERADO:        0.85,
            PerfilRiesgo.AGRESIVO:        0.90,
            PerfilRiesgo.MUY_AGRESIVO:    0.95,
        }
        umbral_corr = _CORR_UMBRAL.get(perfil, 0.85)

        # Precalcular matriz de correlación entre candidatos disponibles
        tickers_cand = [m["ticker"] for m in candidatos if m["ticker"] in rets.columns]
        corr_matrix  = rets[tickers_cand].corr() if len(tickers_cand) >= 2 else pd.DataFrame()

        seleccionados = []
        for m in candidatos:
            if len(seleccionados) >= n_etfs:
                break
            t = m["ticker"]
            # Verificar correlación con todos los ya seleccionados
            demasiado_correlacionado = False
            for sel in seleccionados:
                s = sel["ticker"]
                if (not corr_matrix.empty
                        and t in corr_matrix.index
                        and s in corr_matrix.columns):
                    corr_val = abs(corr_matrix.loc[t, s])
                    if corr_val > umbral_corr:
                        demasiado_correlacionado = True
                        logger.debug(
                            f"Descartado {t} por correlación {corr_val:.2f} con {s} "
                            f"(umbral {umbral_corr})"
                        )
                        break
            if not demasiado_correlacionado:
                seleccionados.append(m)

        # Si el filtro fue demasiado estricto y quedaron pocos, completar sin restricción
        if len(seleccionados) < max(2, n_etfs // 2):
            logger.warning(
                f"Filtro de correlación dejó solo {len(seleccionados)} ETFs — "
                "completando sin restricción"
            )
            ya_sel = {m["ticker"] for m in seleccionados}
            for m in candidatos:
                if len(seleccionados) >= n_etfs:
                    break
                if m["ticker"] not in ya_sel:
                    seleccionados.append(m)

        # ── Piso defensivo para perfiles no agresivos ────────────────────
        # Si ningún activo defensivo quedó en la selección, reemplazar el
        # de menor score con el mejor defensivo disponible.
        _DEFENSIVOS = {"TLT", "BND", "IEF", "LQD", "GLD", "XLU", "XLP"}
        _PERFILES_NO_AGRESIVOS = {
            PerfilRiesgo.MUY_CONSERVADOR, PerfilRiesgo.CONSERVADOR, PerfilRiesgo.MODERADO
        }
        if perfil in _PERFILES_NO_AGRESIVOS:
            sel_tickers = {m["ticker"] for m in seleccionados}
            tiene_defensivo = bool(sel_tickers & _DEFENSIVOS)
            if not tiene_defensivo:
                # Buscar el mejor defensivo entre los candidatos (por score)
                defensivos_cand = [
                    m for m in candidatos
                    if m["ticker"] in _DEFENSIVOS and m["ticker"] not in sel_tickers
                ]
                if defensivos_cand:
                    mejor_def = max(defensivos_cand, key=lambda x: x["score"])
                    # Reemplazar el peor de los seleccionados
                    seleccionados[-1] = mejor_def
                    logger.info(
                        f"Piso defensivo: añadido {mejor_def['ticker']} "
                        f"(reemplaza al de menor score en perfil {perfil.value})"
                    )

        # ── Pesos por volatilidad inversa ────────────────────────────────
        inv_vols = [1.0 / max(m["vol_a"], 0.5) for m in seleccionados]
        total_iv = sum(inv_vols)
        pesos    = [iv / total_iv for iv in inv_vols]

        # ── Construir lista de sectores ──────────────────────────────────
        resultado = []
        for m, peso in zip(seleccionados, pesos):
            info = etf_meta.get(m["ticker"], {"sector": m["ticker"], "descripcion": "", "tipo": "ETF"})
            resultado.append(SectorRecomendado(
                sector=info["sector"],
                descripcion=info["descripcion"],
                etf=m["ticker"],
                peso_target=round(peso, 4),
                retorno_hist=round(m["ret_a"],  2),
                volatilidad_hist=round(m["vol_a"], 2),
                sharpe_hist=round(m["sharpe"],  3),
                ranking_score=round(m["score"], 3),
                seleccion="dinamica",
                tipo=info.get("tipo", "ETF"),
                señal_sentimiento=round(m["sentim_score"], 3) if m.get("sentim_score") is not None else None,
                señal_prediccion =round(m["pred_score"],  3) if m.get("pred_score")  is not None else None,
                señal_sec        =round(m["sec_score"],   3) if m.get("sec_score")   is not None else None,
            ))

        logger.info(
            f"Selección dinámica ({lookback}): {[s.etf for s in resultado]} "
            f"— {universo_evaluado} ETFs evaluados, {len(candidatos)} candidatos"
        )
        return resultado, True, universo_evaluado

    # ------------------------------------------------------------------
    # Señales de agentes para la selección dinámica
    # ------------------------------------------------------------------

    def _obtener_señales_agentes(
        self,
        candidatos: list,
        closes: "pd.DataFrame",
        etf_meta: dict,
        timeout: float = 25.0,
    ) -> dict:
        """
        Obtiene señales de tres fuentes para cada candidato:
          - Sentimiento (SentimentAgent): noticias recientes del activo
          - Predicción (momentum proxy): señal de dirección sobre precios ya
            descargados, sin I/O adicional (rápida)
          - SEC (SECAgent): solo para acciones individuales; señal fundamental

        Retorna {ticker: {"sentimiento": 0-1, "prediccion": 0-1,
                           "sec": 0-1 | None, "combined": 0-1}}
        """
        try:
            from .sentiment_agent import SentimentAgent
            from .sec_agent import SECAgent
        except ImportError as e:
            logger.warning(f"Agentes no importables — scoring sin señales multiagente: {e}")
            return {}

        logger.info(f"_obtener_señales_agentes: {len(candidatos)} candidatos, cols closes={list(closes.columns[:5])}...")
        if not hasattr(self, "_sentiment_agent"):
            try:
                self._sentiment_agent = SentimentAgent()
            except Exception as e:
                logger.warning(f"SentimentAgent no disponible: {e}")
                self._sentiment_agent = None
        if not hasattr(self, "_sec_agent"):
            try:
                self._sec_agent = SECAgent()
            except Exception as e:
                logger.warning(f"SECAgent no disponible: {e}")
                self._sec_agent = None

        def _momentum_signal(ticker: str) -> float:
            """Proxy de predicción: momentum de precio + posición respecto a MA20."""
            if ticker not in closes.columns:
                return 0.5
            serie = closes[ticker].dropna()
            if len(serie) < 21:
                return 0.5
            mom_5d   = float(serie.iloc[-1] / serie.iloc[-5] - 1) if len(serie) >= 5 else 0.0
            ma20     = float(serie.rolling(20).mean().iloc[-1])
            above_ma = 1.0 if float(serie.iloc[-1]) > ma20 else 0.0
            mom_norm = (math.tanh(mom_5d * 10) + 1.0) / 2.0
            return round(0.5 * mom_norm + 0.5 * above_ma, 4)

        def _sentimiento_ticker(ticker: str) -> float:
            try:
                if self._sentiment_agent is None:
                    return 0.5
                result = self._sentiment_agent.analizar(ticker)
                raw = float(getattr(result, "score", 0.0) or 0.0)
                return (raw + 1.0) / 2.0
            except Exception:
                return 0.5

        def _sec_ticker(ticker: str) -> float:
            try:
                if self._sec_agent is None:
                    return 0.5
                result = self._sec_agent.analizar(ticker)
                raw = float(getattr(result, "fundamental_score", 0.0) or 0.0)
                return (raw + 1.0) / 2.0
            except Exception:
                return 0.5

        tickers = [m["ticker"] for m in candidatos]

        # Señal de momentum — síncrona, sin I/O extra
        pred_scores = {t: _momentum_signal(t) for t in tickers}

        # Señal de sentimiento — paralela con timeout
        sentim_scores = {t: 0.5 for t in tickers}
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            fut_map = {ex.submit(_sentimiento_ticker, t): t for t in tickers}
            done, _ = concurrent.futures.wait(fut_map, timeout=timeout)
            for fut in done:
                t = fut_map[fut]
                try:
                    sentim_scores[t] = fut.result()
                except Exception:
                    pass

        # Señal SEC — paralela, solo acciones individuales
        sec_scores: dict = {}
        stocks = [t for t in tickers if etf_meta.get(t, {}).get("tipo") == "Acción"]
        if stocks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                fut_map = {ex.submit(_sec_ticker, t): t for t in stocks}
                done, _ = concurrent.futures.wait(fut_map, timeout=timeout)
                for fut in done:
                    t = fut_map[fut]
                    try:
                        sec_scores[t] = fut.result()
                    except Exception:
                        pass

        # Combinar señales por activo
        result = {}
        for t in tickers:
            tipo    = etf_meta.get(t, {}).get("tipo", "ETF")
            sentim  = sentim_scores.get(t, 0.5)
            pred    = pred_scores.get(t, 0.5)
            sec     = sec_scores.get(t)
            if tipo == "Acción" and sec is not None:
                combined = 0.35 * sentim + 0.35 * pred + 0.30 * sec
            else:
                combined = 0.50 * sentim + 0.50 * pred
            result[t] = {
                "sentimiento": round(sentim,   3),
                "prediccion":  round(pred,     3),
                "sec":         round(sec, 3) if sec is not None else None,
                "combined":    round(combined, 4),
            }
            logger.debug(
                f"Señal agentes {t}: sent={sentim:.2f} pred={pred:.2f} "
                f"sec={f'{sec:.2f}' if sec is not None else 'N/A'} → {combined:.3f}"
            )
        logger.info(f"_obtener_señales_agentes completado: {len(result)} señales")
        return result

    # ------------------------------------------------------------------

    def _calcular_score(
        self,
        q01: Q01Autopercepcion,
        q02: Q02ConcursoTV,
        q03: Q03VacacionPerdidaEmpleo,
        q04: Q04Inversion20k,
        q05: Q05ComodidadAcciones,
        q06: Q06PalabraRiesgo,
        q07: Q07BonosVsActivosDuros,
        q08: Q08GananciaPerdidaPotencial,
        q09: Q09GananciaSegura,
        q10: Q10PerdidaSegura,
        q11: Q11Herencia100k,
        q12: Q12Asignacion20k,
        q13: Q13MinaOro,
    ) -> Tuple[int, List[DimensionScore]]:

        def _dim(nombre, respuesta, tabla, interps):
            pts = tabla.get(respuesta, 0)
            max_pts = max(tabla.values())
            return DimensionScore(
                dimension=nombre,
                respuesta=str(respuesta),
                puntos=pts,
                max_puntos=max_pts,
                interpretacion=interps.get(pts, ""),
            )

        interp_q01 = {1: "Se percibe como evasor del riesgo.", 2: "Se percibe cauteloso.",
                      3: "Se percibe como un tomador de riesgo calculado.", 4: "Se percibe como un verdadero apostador."}
        interp_q02 = {1: "Prefiere el efectivo seguro, sin exposición al azar.",
                      2: "Acepta una apuesta moderada por una ganancia mayor.",
                      3: "Acepta más incertidumbre por un premio mayor.",
                      4: "Busca el premio más alto aun con probabilidad mínima."}
        interp_q03 = {1: "Prioriza la seguridad financiera inmediata ante la pérdida de ingresos.",
                      2: "Reduce el gasto pero mantiene el plan parcialmente.",
                      3: "Mantiene el plan pese a la incertidumbre laboral.",
                      4: "Prioriza la experiencia por sobre la prudencia financiera."}
        interp_q04 = {1: "Prefiere liquidez y capital garantizado.",
                      2: "Acepta renta fija de calidad para buscar algo más de retorno.",
                      3: "Acepta la volatilidad de la renta variable por mayor retorno esperado."}
        interp_q05 = {1: "No se siente cómodo con la volatilidad de las acciones.",
                      2: "Tolera cierta volatilidad en renta variable.",
                      3: "Se siente cómodo invirtiendo en acciones."}
        interp_q06 = {1: "Asocia el riesgo primero con pérdida.", 2: "Asocia el riesgo con incertidumbre.",
                      3: "Asocia el riesgo con oportunidad.", 4: "Asocia el riesgo con adrenalina/emoción."}
        interp_q07 = {1: "Mantiene la posición defensiva pese a la señal de mercado.",
                      2: "Diversifica parcialmente hacia activos duros.",
                      3: "Rota completamente hacia activos duros.",
                      4: "Rota y además apalanca la posición — tolerancia muy alta."}
        interp_q08 = {1: "Prefiere el perfil de menor ganancia/pérdida potencial.",
                      2: "Acepta un perfil moderado-bajo de ganancia/pérdida.",
                      3: "Acepta un perfil moderado-alto de ganancia/pérdida.",
                      4: "Prefiere el perfil de mayor ganancia/pérdida potencial."}
        interp_q09 = {1: "Prefiere la ganancia segura antes que apostar por más.",
                      3: "Prefiere apostar por una ganancia mayor con riesgo de no ganar nada."}
        interp_q10 = {1: "Prefiere arriesgarse a perder más con tal de tener chance de no perder nada.",
                      3: "Prefiere asegurar la pérdida menor antes que arriesgarse a una mayor."}
        interp_q11 = {1: "Invertiría toda la herencia en instrumentos de máxima seguridad.",
                      2: "Invertiría en un fondo mixto de acciones y bonos.",
                      3: "Invertiría en un portafolio de acciones individuales.",
                      4: "Invertiría en commodities de alta volatilidad."}
        interp_q12 = {1: "Prefiere una asignación muy conservadora (mayoría bajo riesgo).",
                      2: "Prefiere una asignación conservadora.",
                      3: "Prefiere una asignación equilibrada.",
                      4: "Prefiere una asignación agresiva (mayoría alto riesgo)."}
        interp_q13 = {1: "No arriesgaría capital en un proyecto especulativo de alto riesgo.",
                      2: "Arriesgaría una porción menor de sus ingresos.",
                      3: "Arriesgaría una porción moderada de sus ingresos.",
                      4: "Arriesgaría una porción alta de sus ingresos en un proyecto especulativo."}

        dims = [
            _dim("1. Autopercepción como tomador de riesgo", q01, _SCORE_Q01, interp_q01),
            _dim("2. Elección en concurso de TV",             q02, _SCORE_Q02, interp_q02),
            _dim("3. Vacación tras pérdida de empleo",        q03, _SCORE_Q03, interp_q03),
            _dim("4. Inversión de $20.000 inesperados",       q04, _SCORE_Q04, interp_q04),
            _dim("5. Comodidad invirtiendo en acciones",      q05, _SCORE_Q05, interp_q05),
            _dim("6. Primera palabra asociada a 'riesgo'",    q06, _SCORE_Q06, interp_q06),
            _dim("7. Rotación bonos vs. activos duros",       q07, _SCORE_Q07, interp_q07),
            _dim("8. Perfil de ganancia/pérdida potencial",   q08, _SCORE_Q08, interp_q08),
            _dim("9. Ganancia segura vs. apuesta",            q09, _SCORE_Q09, interp_q09),
            _dim("10. Pérdida segura vs. apuesta",            q10, _SCORE_Q10, interp_q10),
            _dim("11. Herencia de $100.000 en una opción",    q11, _SCORE_Q11, interp_q11),
            _dim("12. Asignación de $20.000 por nivel de riesgo", q12, _SCORE_Q12, interp_q12),
            _dim("13. Inversión especulativa (mina de oro)",  q13, _SCORE_Q13, interp_q13),
        ]

        return sum(d.puntos for d in dims), dims

    @staticmethod
    def _clasificar(score_raw: int) -> PerfilRiesgo:
        """Clasificación oficial Grable & Lytton (1999) sobre el score crudo."""
        if score_raw <= 18:
            return PerfilRiesgo.MUY_CONSERVADOR
        elif score_raw <= 22:
            return PerfilRiesgo.CONSERVADOR
        elif score_raw <= 28:
            return PerfilRiesgo.MODERADO
        elif score_raw <= 32:
            return PerfilRiesgo.AGRESIVO
        else:
            return PerfilRiesgo.MUY_AGRESIVO
