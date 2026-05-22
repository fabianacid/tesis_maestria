"""
Agente de Datos Fundamentales y Filings SEC

Obtiene información financiera fundamental de dos fuentes:
- yfinance: ratios financieros y balances en tiempo real
- SEC EDGAR API (gratuita): filings recientes (8-K, 10-Q, 10-K)

Calcula un score fundamental para integrar al pipeline multiagente.
"""
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_cache: Dict[str, Any] = {}
_cache_ts: Dict[str, datetime] = {}
CACHE_TTL = timedelta(hours=2)

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    logger.warning("yfinance no disponible para SECAgent")

try:
    import requests as _http
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False
    logger.warning("requests no disponible para SECAgent")

EDGAR_HEADERS = {
    "User-Agent": "FinancialAnalysisSystem academic@finanalysis.edu",
    "Accept-Encoding": "gzip, deflate",
}
EDGAR_BASE = "https://data.sec.gov"
TICKERS_JSON_URL = "https://www.sec.gov/files/company_tickers.json"


@dataclass
class SECFiling:
    form_type: str
    filing_date: str
    description: str
    accession_number: str


@dataclass
class FinancialRatios:
    # Valuación
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    # Rentabilidad
    roe: Optional[float] = None
    roa: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    profit_margin: Optional[float] = None
    # Crecimiento
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    # Deuda y liquidez
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    # Mercado
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    market_cap: Optional[float] = None
    # Evaluación
    health_score: float = 5.0
    health_label: str = "neutral"


@dataclass
class BalanceSummary:
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    total_equity: Optional[float] = None
    total_debt: Optional[float] = None
    cash_and_equivalents: Optional[float] = None
    revenue_ttm: Optional[float] = None
    net_income_ttm: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    free_cash_flow: Optional[float] = None


@dataclass
class SECData:
    ticker: str
    company_name: str
    ratios: FinancialRatios
    balance: BalanceSummary
    recent_filings: List[SECFiling]
    fundamental_signal: str       # "positivo", "neutral", "negativo"
    fundamental_score: float      # -1.0 a +1.0
    resumen: str
    fecha_actualizacion: datetime
    disponible: bool = True
    error_msg: str = ""


class SECAgent:
    """
    Agente de análisis fundamental y SEC.

    Combina yfinance para ratios y SEC EDGAR para filings,
    produciendo un score fundamental que se integra al
    pipeline de recomendación.
    """

    def __init__(self):
        self._cik_cache: Dict[str, Optional[str]] = {}
        logger.info("SECAgent inicializado")

    def analizar(self, ticker: str) -> SECData:
        """Obtiene datos fundamentales y filings para el ticker."""
        cache_key = f"sec_{ticker.upper()}"
        if cache_key in _cache and datetime.now() - _cache_ts.get(cache_key, datetime.min) < CACHE_TTL:
            return _cache[cache_key]

        try:
            ratios, balance, company_name = self._get_yfinance_data(ticker)
            filings = self._get_sec_filings(ticker)
            score, signal, resumen = self._evaluar_fundamentales(ratios, ticker)

            result = SECData(
                ticker=ticker.upper(),
                company_name=company_name,
                ratios=ratios,
                balance=balance,
                recent_filings=filings,
                fundamental_signal=signal,
                fundamental_score=score,
                resumen=resumen,
                fecha_actualizacion=datetime.now(),
                disponible=True,
            )
            _cache[cache_key] = result
            _cache_ts[cache_key] = datetime.now()
            logger.info(f"[{ticker}] SECAgent: señal={signal}, score={score:.3f}, filings={len(filings)}")
            return result

        except Exception as exc:
            logger.error(f"[{ticker}] SECAgent error: {exc}")
            return SECData(
                ticker=ticker.upper(),
                company_name=ticker.upper(),
                ratios=FinancialRatios(),
                balance=BalanceSummary(),
                recent_filings=[],
                fundamental_signal="neutral",
                fundamental_score=0.0,
                resumen="No se pudieron obtener datos fundamentales.",
                fecha_actualizacion=datetime.now(),
                disponible=False,
                error_msg=str(exc),
            )

    # ------------------------------------------------------------------
    # Datos yfinance
    # ------------------------------------------------------------------

    def _get_yfinance_data(self, ticker: str):
        if not YF_AVAILABLE:
            return FinancialRatios(), BalanceSummary(), ticker

        t = yf.Ticker(ticker)
        info = t.info or {}
        company_name = info.get("longName") or info.get("shortName") or ticker

        def _f(key, divisor=1.0):
            v = info.get(key)
            return self._safe_float(v, divisor)

        ratios = FinancialRatios(
            pe_ratio=_f("trailingPE") or _f("forwardPE"),
            pb_ratio=_f("priceToBook"),
            ps_ratio=_f("priceToSalesTrailing12Months"),
            ev_ebitda=_f("enterpriseToEbitda"),
            roe=_f("returnOnEquity"),
            roa=_f("returnOnAssets"),
            gross_margin=_f("grossMargins"),
            operating_margin=_f("operatingMargins"),
            profit_margin=_f("profitMargins"),
            revenue_growth=_f("revenueGrowth"),
            earnings_growth=_f("earningsGrowth"),
            debt_to_equity=_f("debtToEquity", 100.0),  # yfinance devuelve en %; dividimos entre 100
            current_ratio=_f("currentRatio"),
            quick_ratio=_f("quickRatio"),
            dividend_yield=_f("dividendYield"),
            beta=_f("beta"),
            market_cap=_f("marketCap"),
        )

        balance = BalanceSummary(
            total_debt=_f("totalDebt"),
            cash_and_equivalents=_f("totalCash"),
            total_equity=None,
            revenue_ttm=_f("totalRevenue"),
            net_income_ttm=_f("netIncomeToCommon"),
            operating_cash_flow=_f("operatingCashflow"),
            free_cash_flow=_f("freeCashflow"),
        )

        # Intentar obtener activos y pasivos totales del balance sheet
        try:
            bs = t.balance_sheet
            if bs is not None and not bs.empty:
                col = bs.columns[0]
                def _bs(label):
                    for key in bs.index:
                        if label.lower() in str(key).lower():
                            return self._safe_float(bs.loc[key, col])
                    return None

                balance.total_assets = _bs("total assets") or _bs("totalassets")
                balance.total_liabilities = _bs("total liabilities") or _bs("totalliabilitiesnetminorityinterest")
                balance.total_equity = _bs("stockholders equity") or _bs("totalequitygrossminorityinterest")
        except Exception:
            pass

        return ratios, balance, company_name

    # ------------------------------------------------------------------
    # Filings SEC EDGAR
    # ------------------------------------------------------------------

    def _get_sec_filings(self, ticker: str, n: int = 6) -> List[SECFiling]:
        if not HTTP_AVAILABLE:
            return []
        try:
            cik = self._get_cik(ticker)
            if not cik:
                return []

            cik_padded = str(int(cik)).zfill(10)
            url = f"{EDGAR_BASE}/submissions/CIK{cik_padded}.json"
            resp = _http.get(url, headers=EDGAR_HEADERS, timeout=12)
            if resp.status_code != 200:
                return []

            data = resp.json()
            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            descriptions = recent.get("primaryDocDescription", [])

            target_forms = {"10-K", "10-Q", "8-K", "DEF 14A"}
            filings: List[SECFiling] = []
            for i, form in enumerate(forms):
                if form in target_forms and len(filings) < n:
                    filings.append(SECFiling(
                        form_type=form,
                        filing_date=dates[i] if i < len(dates) else "N/A",
                        description=descriptions[i] if i < len(descriptions) and descriptions[i] else form,
                        accession_number=accessions[i] if i < len(accessions) else "",
                    ))
            return filings

        except Exception as exc:
            logger.warning(f"[{ticker}] Error filings SEC: {exc}")
            return []

    def _get_cik(self, ticker: str) -> Optional[str]:
        ticker_up = ticker.upper()
        if ticker_up in self._cik_cache:
            return self._cik_cache[ticker_up]
        try:
            resp = _http.get(TICKERS_JSON_URL, headers=EDGAR_HEADERS, timeout=12)
            if resp.status_code != 200:
                self._cik_cache[ticker_up] = None
                return None
            for _, company in resp.json().items():
                if company.get("ticker", "").upper() == ticker_up:
                    cik = str(company["cik_str"])
                    self._cik_cache[ticker_up] = cik
                    return cik
            self._cik_cache[ticker_up] = None
            return None
        except Exception as exc:
            logger.warning(f"Error buscando CIK para {ticker}: {exc}")
            self._cik_cache[ticker_up] = None
            return None

    # ------------------------------------------------------------------
    # Evaluación fundamental
    # ------------------------------------------------------------------

    def _evaluar_fundamentales(self, ratios: FinancialRatios, ticker: str):
        """
        Produce score fundamental entre -1 y +1 y señal cualitativa.
        Pesos calibrados para empresas que cotizan en EE.UU.
        """
        score = 0.0
        n_factors = 0
        factors_desc = []

        def add(contribution, weight, desc=None):
            nonlocal score, n_factors
            score += contribution * weight
            n_factors += 1
            if desc:
                factors_desc.append(desc)

        # --- Rentabilidad ---
        if ratios.roe is not None:
            s = min(max(ratios.roe / 0.20, -1.5), 1.5)
            add(s, 0.20, f"ROE={ratios.roe*100:.1f}%")

        if ratios.profit_margin is not None:
            s = min(max(ratios.profit_margin / 0.15, -1.5), 1.5)
            add(s, 0.12)

        if ratios.operating_margin is not None:
            s = min(max(ratios.operating_margin / 0.20, -1.5), 1.5)
            add(s, 0.08)

        # --- Valuación P/E ---
        if ratios.pe_ratio is not None and ratios.pe_ratio > 0:
            if ratios.pe_ratio < 10:
                s = 0.9
            elif ratios.pe_ratio < 18:
                s = 0.5
            elif ratios.pe_ratio < 30:
                s = 0.0
            elif ratios.pe_ratio < 50:
                s = -0.4
            else:
                s = -0.8
            add(s, 0.15, f"P/E={ratios.pe_ratio:.1f}x")

        # --- Crecimiento ---
        if ratios.revenue_growth is not None:
            s = min(max(ratios.revenue_growth / 0.12, -1.5), 1.5)
            add(s, 0.15, f"Rev.Growth={ratios.revenue_growth*100:.1f}%")

        if ratios.earnings_growth is not None:
            s = min(max(ratios.earnings_growth / 0.15, -1.5), 1.5)
            add(s, 0.10)

        # --- Liquidez y deuda ---
        if ratios.current_ratio is not None:
            if ratios.current_ratio > 2.0:
                s = 0.5
            elif ratios.current_ratio > 1.2:
                s = 0.2
            elif ratios.current_ratio > 0.8:
                s = -0.2
            else:
                s = -0.7
            add(s, 0.10)

        if ratios.debt_to_equity is not None:
            if ratios.debt_to_equity < 0.3:
                s = 0.6
            elif ratios.debt_to_equity < 0.8:
                s = 0.2
            elif ratios.debt_to_equity < 1.5:
                s = -0.2
            else:
                s = -0.7
            add(s, 0.10)

        # Score final normalizado
        if n_factors == 0:
            score = 0.0
        else:
            score = max(-1.0, min(1.0, score))

        # Señal
        if score >= 0.20:
            signal = "positivo"
        elif score <= -0.20:
            signal = "negativo"
        else:
            signal = "neutral"

        # Health score 0-10
        ratios.health_score = round((score + 1.0) / 2.0 * 10.0, 1)
        ratios.health_label = signal

        # Resumen
        if factors_desc:
            resumen = f"Fundamentales {signal}s: {', '.join(factors_desc[:3])}."
        else:
            resumen = f"Datos fundamentales disponibles limitados para {ticker}."

        return round(score, 3), signal, resumen

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(val, divisor: float = 1.0) -> Optional[float]:
        if val is None:
            return None
        try:
            f = float(val) / divisor
            if math.isnan(f) or math.isinf(f):
                return None
            return round(f, 6)
        except (TypeError, ValueError):
            return None
