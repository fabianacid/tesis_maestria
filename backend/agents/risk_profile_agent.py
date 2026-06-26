"""
Agente de Perfil de Riesgo del Inversor

Cuantifica la tolerancia al riesgo mediante un cuestionario de 6 dimensiones
y genera recomendaciones de asignación sectorial alineadas con ese perfil.

Metodología de scoring:
- Cada dimensión aporta 0-4 puntos → total máximo 24
- Score normalizado a 0-100
- 5 perfiles: muy_conservador (0-20), conservador (21-40), moderado (41-60),
  agresivo (61-80), muy_agresivo (81-100)

Dimensiones evaluadas:
  1. Edad (proxy de horizonte vital)
  2. Horizonte de inversión declarado
  3. Estabilidad de ingresos
  4. Tolerancia a pérdidas (reacción ante caída del 20%)
  5. Experiencia inversora
  6. Objetivo financiero principal

Salida:
  - Perfil de riesgo con score y descripción
  - Sectores recomendados con ETFs representativos y pesos target
  - Restricciones de cartera para integrar con PortfolioAgent
  - Presupuesto de VaR máximo tolerado
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Enums del cuestionario
# ──────────────────────────────────────────────

class GrupoEdad(str, Enum):
    MENOR_35  = "menor_35"
    ENTRE_36_45 = "36_45"
    ENTRE_46_55 = "46_55"
    ENTRE_56_65 = "56_65"
    MAYOR_65  = "mayor_65"


class HorizonteInversion(str, Enum):
    MENOS_1_ANIO  = "menos_1_anio"
    UNO_3_ANIOS   = "1_3_anios"
    TRES_5_ANIOS  = "3_5_anios"
    CINCO_10_ANIOS = "5_10_anios"
    MAS_10_ANIOS  = "mas_10_anios"


class EstabilidadIngresos(str, Enum):
    MUY_INESTABLE = "muy_inestable"
    INESTABLE     = "inestable"
    MODERADA      = "moderada"
    ESTABLE       = "estable"
    MUY_ESTABLE   = "muy_estable"


class ToleranciaPerdidas(str, Enum):
    VENDER_TODO    = "vender_todo"
    VENDER_MAYORIA = "vender_mayoria"
    MANTENER       = "mantener"
    COMPRAR_POCO   = "comprar_poco"
    COMPRAR_MUCHO  = "comprar_mucho"


class ExperienciaInversora(str, Enum):
    NINGUNA    = "ninguna"
    BASICA     = "basica"
    INTERMEDIA = "intermedia"
    AVANZADA   = "avanzada"
    EXPERTO    = "experto"


class ObjetivoFinanciero(str, Enum):
    PRESERVACION       = "preservacion"
    INGRESO            = "ingreso"
    CRECIMIENTO_INGRESO = "crecimiento_ingreso"
    CRECIMIENTO        = "crecimiento"
    ESPECULACION       = "especulacion"


class PerfilRiesgo(str, Enum):
    MUY_CONSERVADOR = "muy_conservador"
    CONSERVADOR     = "conservador"
    MODERADO        = "moderado"
    AGRESIVO        = "agresivo"
    MUY_AGRESIVO    = "muy_agresivo"


# ──────────────────────────────────────────────
# Tablas de puntuación (0-4 por dimensión, total máx 24)
# ──────────────────────────────────────────────

_SCORE_EDAD: Dict[str, int] = {
    GrupoEdad.MENOR_35:    4,
    GrupoEdad.ENTRE_36_45: 3,
    GrupoEdad.ENTRE_46_55: 2,
    GrupoEdad.ENTRE_56_65: 1,
    GrupoEdad.MAYOR_65:    0,
}

_SCORE_HORIZONTE: Dict[str, int] = {
    HorizonteInversion.MENOS_1_ANIO:   0,
    HorizonteInversion.UNO_3_ANIOS:    1,
    HorizonteInversion.TRES_5_ANIOS:   2,
    HorizonteInversion.CINCO_10_ANIOS: 3,
    HorizonteInversion.MAS_10_ANIOS:   4,
}

_SCORE_INGRESOS: Dict[str, int] = {
    EstabilidadIngresos.MUY_INESTABLE: 0,
    EstabilidadIngresos.INESTABLE:     1,
    EstabilidadIngresos.MODERADA:      2,
    EstabilidadIngresos.ESTABLE:       3,
    EstabilidadIngresos.MUY_ESTABLE:   4,
}

_SCORE_PERDIDAS: Dict[str, int] = {
    ToleranciaPerdidas.VENDER_TODO:    0,
    ToleranciaPerdidas.VENDER_MAYORIA: 1,
    ToleranciaPerdidas.MANTENER:       2,
    ToleranciaPerdidas.COMPRAR_POCO:   3,
    ToleranciaPerdidas.COMPRAR_MUCHO:  4,
}

_SCORE_EXPERIENCIA: Dict[str, int] = {
    ExperienciaInversora.NINGUNA:    0,
    ExperienciaInversora.BASICA:     1,
    ExperienciaInversora.INTERMEDIA: 2,
    ExperienciaInversora.AVANZADA:   3,
    ExperienciaInversora.EXPERTO:    4,
}

_SCORE_OBJETIVO: Dict[str, int] = {
    ObjetivoFinanciero.PRESERVACION:        0,
    ObjetivoFinanciero.INGRESO:             1,
    ObjetivoFinanciero.CRECIMIENTO_INGRESO: 2,
    ObjetivoFinanciero.CRECIMIENTO:         3,
    ObjetivoFinanciero.ESPECULACION:        4,
}

_MAX_SCORE_RAW = 24  # 6 dimensiones × 4 pts


# ──────────────────────────────────────────────
# Recomendaciones sectoriales por perfil
# Cada sector incluye: nombre, ETF representativo, peso target (suma=1)
# ──────────────────────────────────────────────

@dataclass
class SectorRecomendado:
    sector: str
    descripcion: str
    etf: str                              # ticker del ETF representativo
    peso_target: float                    # 0-1, suma = 1 para el perfil
    # Métricas históricas (pobladas solo en selección dinámica)
    retorno_hist: Optional[float] = None  # % anualizado (lookback)
    volatilidad_hist: Optional[float] = None
    sharpe_hist: Optional[float] = None
    ranking_score: Optional[float] = None
    seleccion: str = "predefinida"        # "dinamica" | "predefinida"


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

# Universo de ETFs candidatos para selección dinámica
# (ticker, nombre_sector, descripcion)
UNIVERSO_ETF: List[Tuple[str, str, str]] = [
    ("TLT", "Bonos EEUU largo plazo",       "Bonos del Tesoro EEUU +20 años, alta duración"),
    ("BND", "Bonos diversificados",          "Mercado total de renta fija EEUU"),
    ("IEF", "Bonos EEUU mediano plazo",      "Bonos del Tesoro EEUU 7-10 años"),
    ("LQD", "Bonos corporativos IG",         "Bonos corporativos grado inversión EEUU"),
    ("GLD", "Oro",                           "Commodity refugio, baja correlación con equity"),
    ("XLU", "Utilities",                     "Electricidad y servicios regulados, defensivo"),
    ("XLP", "Consumer Staples",              "Bienes de consumo básico, demanda inelástica"),
    ("XLV", "Healthcare",                    "Salud y farmacéuticas, sector defensivo"),
    ("VNQ", "Real Estate (REITs)",           "Inmuebles diversificados, generan dividendos"),
    ("XLF", "Financials",                    "Bancos y seguros"),
    ("XLI", "Industrials",                   "Manufactura e infraestructura"),
    ("XLC", "Communications",               "Telecomunicaciones y medios digitales"),
    ("XLK", "Technology",                    "Tecnología e innovación, alto crecimiento"),
    ("XLY", "Consumer Discretionary",        "Consumo discrecional, cíclico"),
    ("XLE", "Energy",                        "Petróleo, gas y energías, alta ciclicidad"),
    ("XLB", "Materials",                     "Minería y materiales básicos"),
    ("EFA", "International DM",              "Mercados desarrollados fuera de EEUU"),
    ("VGK", "Europa",                        "Acciones europeas desarrolladas"),
    ("EEM", "Emerging Markets",              "Mercados emergentes, alta volatilidad"),
    ("IWM", "Small Cap EEUU",               "Empresas pequeñas EEUU, mayor potencial y riesgo"),
    ("MDY", "Mid Cap EEUU",                 "Empresas medianas EEUU"),
    ("QQQ", "Nasdaq 100",                   "Las 100 mayores empresas no financieras del Nasdaq"),
]

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
    score: float                          # 0-100
    score_raw: int                        # 0-24
    descripcion_perfil: str
    dimensiones: List[DimensionScore]
    sectores_recomendados: List[SectorRecomendado]
    tickers_recomendados: List[str]       # solo los ETFs, para pasar a PortfolioAgent
    pesos_sugeridos: List[float]          # alineados con tickers_recomendados
    risk_budget: Dict                     # var_95_max, vol_anual_max, max_peso_activo
    advertencia: str = ""
    seleccion_dinamica: bool = False      # True si se usaron datos históricos reales
    periodo_analisis: str = ""            # e.g. "1y" o "predefinido"
    universo_evaluado: int = 0            # cuántos ETFs del universo se analizaron


# ──────────────────────────────────────────────
# Agente principal
# ──────────────────────────────────────────────

class RiskProfileAgent:
    """
    Evalúa el perfil de riesgo del inversor y genera recomendaciones
    de asignación sectorial alineadas con su tolerancia al riesgo.

    Uso básico:
        agent = RiskProfileAgent()
        result = agent.evaluar(
            edad=GrupoEdad.MENOR_35,
            horizonte=HorizonteInversion.MAS_10_ANIOS,
            ingresos=EstabilidadIngresos.MUY_ESTABLE,
            perdidas=ToleranciaPerdidas.COMPRAR_POCO,
            experiencia=ExperienciaInversora.INTERMEDIA,
            objetivo=ObjetivoFinanciero.CRECIMIENTO,
        )
    """

    def evaluar(
        self,
        edad: GrupoEdad,
        horizonte: HorizonteInversion,
        ingresos: EstabilidadIngresos,
        perdidas: ToleranciaPerdidas,
        experiencia: ExperienciaInversora,
        objetivo: ObjetivoFinanciero,
        usar_seleccion_dinamica: bool = True,
        lookback: str = "1y",
    ) -> RiskProfileResult:
        raw, dimensiones = self._calcular_score(
            edad, horizonte, ingresos, perdidas, experiencia, objetivo
        )
        score_norm = round((raw / _MAX_SCORE_RAW) * 100, 1)
        perfil = self._clasificar(score_norm)

        # Selección de sectores: dinámica (datos reales) o predefinida (fallback)
        seleccion_dinamica = False
        universo_evaluado = 0
        if usar_seleccion_dinamica:
            sectores, seleccion_dinamica, universo_evaluado = self._seleccionar_sectores_dinamico(
                perfil, lookback=lookback
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
            f"RiskProfileAgent: score={score_norm}/100 → perfil={perfil.value} "
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
    ) -> Tuple[List[SectorRecomendado], bool, int]:
        """
        Selecciona los N mejores ETFs del universo para el perfil dado
        usando datos históricos reales de yfinance.

        Algoritmo:
          1. Descarga `lookback` de retornos diarios para todos los ETFs del universo.
          2. Calcula retorno anualizado, volatilidad y Sharpe por ETF.
          3. Filtra ETFs cuya volatilidad supere el límite del perfil (con tolerancia 30%).
          4. Rankea por score ajustado al perfil:
               score = (1 - agresividad) * sharpe_normalizado
                     + agresividad * retorno_normalizado
             donde agresividad ∈ [0, 1] según el perfil.
          5. Toma los top N y asigna pesos por volatilidad inversa.

        Returns:
            (sectores, fue_dinamico, n_etfs_evaluados)
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

        tickers_universo = [e[0] for e in UNIVERSO_ETF]
        etf_meta = {e[0]: {"sector": e[1], "descripcion": e[2]} for e in UNIVERSO_ETF}

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

        # ── Score ajustado al perfil ─────────────────────────────────────
        sharpes  = [m["sharpe"] for m in candidatos]
        retornos = [m["ret_a"]  for m in candidatos]
        s_min, s_rng = min(sharpes),  max(sharpes)  - min(sharpes)  or 1.0
        r_min, r_rng = min(retornos), max(retornos) - min(retornos) or 1.0

        for m in candidatos:
            s_norm = (m["sharpe"] - s_min) / s_rng
            r_norm = (m["ret_a"]  - r_min) / r_rng
            m["score"] = (1 - agr) * s_norm + agr * r_norm

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
            info = etf_meta.get(m["ticker"], {"sector": m["ticker"], "descripcion": ""})
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
            ))

        logger.info(
            f"Selección dinámica ({lookback}): {[s.etf for s in resultado]} "
            f"— {universo_evaluado} ETFs evaluados, {len(candidatos)} candidatos"
        )
        return resultado, True, universo_evaluado

    # ------------------------------------------------------------------

    def _calcular_score(
        self,
        edad: GrupoEdad,
        horizonte: HorizonteInversion,
        ingresos: EstabilidadIngresos,
        perdidas: ToleranciaPerdidas,
        experiencia: ExperienciaInversora,
        objetivo: ObjetivoFinanciero,
    ) -> Tuple[int, List[DimensionScore]]:

        def _dim(nombre, respuesta, tabla, interps):
            pts = tabla.get(respuesta, 0)
            return DimensionScore(
                dimension=nombre,
                respuesta=str(respuesta),
                puntos=pts,
                max_puntos=4,
                interpretacion=interps[pts],
            )

        interp_edad = [
            "Horizonte vital reducido; menor capacidad de recuperar pérdidas.",
            "Horizonte moderado; cautela recomendada.",
            "Horizonte medio; equilibrio riesgo-retorno factible.",
            "Horizonte amplio; puede absorber ciclos bajistas.",
            "Horizonte muy amplio; tiempo suficiente para recuperar caídas.",
        ]
        interp_horizonte = [
            "Inversión a muy corto plazo; máxima liquidez necesaria.",
            "Corto plazo; volatilidad difícil de absorber.",
            "Mediano plazo; tolerable algo de riesgo.",
            "Largo plazo; capacidad alta de esperar recuperación.",
            "Muy largo plazo; máxima capacidad de asumir riesgo temporal.",
        ]
        interp_ingresos = [
            "Ingresos muy inestables; prioridad a liquidez.",
            "Ingresos variables; mantener colchón de seguridad.",
            "Ingresos moderados; margen limitado para asumir pérdidas.",
            "Ingresos estables; puede comprometer capital a largo plazo.",
            "Ingresos muy estables; puede asumir alta volatilidad.",
        ]
        interp_perdidas = [
            "Tolera pérdidas mínimas; perfil muy defensivo.",
            "Vende ante pérdidas; evita activos volátiles.",
            "Mantiene calma; espera recuperación sin actuar.",
            "Ve caídas como oportunidad moderada.",
            "Compra agresivamente en caídas; alto temple inversor.",
        ]
        interp_experiencia = [
            "Sin experiencia; desconoce volatilidad real de mercados.",
            "Experiencia básica; puede sorprenderse ante caídas.",
            "Experiencia intermedia; comprende ciclos de mercado.",
            "Experiencia avanzada; gestiona posiciones complejas.",
            "Experto; conocimiento profundo de riesgo y derivados.",
        ]
        interp_objetivo = [
            "Objetivo: no perder. Retorno secundario.",
            "Objetivo: flujo de dividendos estable.",
            "Objetivo: crecimiento moderado con ingresos.",
            "Objetivo: apreciación del capital a largo plazo.",
            "Objetivo: máximo retorno; acepta alta incertidumbre.",
        ]

        dims = [
            _dim("Edad / Horizonte vital",         edad,        _SCORE_EDAD,        interp_edad),
            _dim("Horizonte de inversión",          horizonte,   _SCORE_HORIZONTE,   interp_horizonte),
            _dim("Estabilidad de ingresos",         ingresos,    _SCORE_INGRESOS,    interp_ingresos),
            _dim("Tolerancia a pérdidas",           perdidas,    _SCORE_PERDIDAS,    interp_perdidas),
            _dim("Experiencia inversora",           experiencia, _SCORE_EXPERIENCIA, interp_experiencia),
            _dim("Objetivo financiero",             objetivo,    _SCORE_OBJETIVO,    interp_objetivo),
        ]

        return sum(d.puntos for d in dims), dims

    @staticmethod
    def _clasificar(score: float) -> PerfilRiesgo:
        if score <= 20:
            return PerfilRiesgo.MUY_CONSERVADOR
        elif score <= 40:
            return PerfilRiesgo.CONSERVADOR
        elif score <= 60:
            return PerfilRiesgo.MODERADO
        elif score <= 80:
            return PerfilRiesgo.AGRESIVO
        else:
            return PerfilRiesgo.MUY_AGRESIVO
