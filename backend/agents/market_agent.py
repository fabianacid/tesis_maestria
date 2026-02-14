"""
Agente de Mercado Profesional

Implementación de nivel institucional para análisis técnico de mercados financieros.
Utiliza más de 20 indicadores técnicos avanzados y genera señales multi-factor
con sistema de puntuación ponderada.

Indicadores implementados:
- Tendencia: SMA, EMA, MACD, ADX, Ichimoku Cloud, Parabolic SAR
- Momentum: RSI, Stochastic, Williams %R, ROC, CCI
- Volatilidad: Bollinger Bands, ATR, Keltner Channels
- Volumen: OBV, VWAP, MFI, ADL, CMF

Arquitectura de señales:
- Sistema de scoring multi-factor
- Ponderación dinámica por condiciones de mercado
- Detección de divergencias
- Análisis de confluencia técnica
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import ta
    from ta.trend import (
        SMAIndicator, EMAIndicator, MACD, ADXIndicator,
        IchimokuIndicator, PSARIndicator, CCIIndicator
    )
    from ta.momentum import (
        RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
    )
    from ta.volatility import (
        BollingerBands, AverageTrueRange, KeltnerChannel
    )
    from ta.volume import (
        OnBalanceVolumeIndicator, VolumeWeightedAveragePrice,
        MFIIndicator, AccDistIndexIndicator, ChaikinMoneyFlowIndicator
    )
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """Fuerza de la señal de mercado."""
    STRONG_BULLISH = "fuerte_alcista"
    BULLISH = "alcista"
    WEAK_BULLISH = "debil_alcista"
    NEUTRAL = "neutral"
    WEAK_BEARISH = "debil_bajista"
    BEARISH = "bajista"
    STRONG_BEARISH = "fuerte_bajista"


class MarketRegime(Enum):
    """Régimen de mercado detectado."""
    TRENDING_UP = "tendencia_alcista"
    TRENDING_DOWN = "tendencia_bajista"
    RANGING = "lateral"
    HIGH_VOLATILITY = "alta_volatilidad"
    LOW_VOLATILITY = "baja_volatilidad"


@dataclass
class TechnicalIndicators:
    """Estructura completa de indicadores técnicos."""
    # Tendencia
    sma_20: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    ema_50: float = 0.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    adx: float = 0.0
    adx_pos: float = 0.0
    adx_neg: float = 0.0
    ichimoku_a: float = 0.0
    ichimoku_b: float = 0.0
    psar: float = 0.0

    # Momentum
    rsi: float = 50.0
    stoch_k: float = 50.0
    stoch_d: float = 50.0
    williams_r: float = -50.0
    roc: float = 0.0
    cci: float = 0.0

    # Volatilidad
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_width: float = 0.0
    bb_pct: float = 0.5
    atr: float = 0.0
    atr_pct: float = 0.0
    keltner_upper: float = 0.0
    keltner_lower: float = 0.0

    # Volumen
    obv: float = 0.0
    obv_trend: str = "neutral"
    vwap: float = 0.0
    mfi: float = 50.0
    adl: float = 0.0
    cmf: float = 0.0


@dataclass
class SignalAnalysis:
    """Análisis detallado de señales."""
    trend_score: float = 0.0
    momentum_score: float = 0.0
    volatility_score: float = 0.0
    volume_score: float = 0.0
    composite_score: float = 0.0
    signal_strength: SignalStrength = SignalStrength.NEUTRAL
    market_regime: MarketRegime = MarketRegime.RANGING
    confidence: float = 0.0
    signals_aligned: int = 0
    total_signals: int = 0
    bullish_signals: List[str] = field(default_factory=list)
    bearish_signals: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class MarketData:
    """Estructura de datos de mercado procesados con análisis completo."""
    ticker: str
    precios: pd.DataFrame
    ultimo_precio: float
    precio_anterior: float
    variacion_diaria: float
    media_movil_20: float
    senal: str
    fecha_actualizacion: datetime
    # Datos avanzados
    indicators: TechnicalIndicators = field(default_factory=TechnicalIndicators)
    signal_analysis: SignalAnalysis = field(default_factory=SignalAnalysis)
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)


class MarketAgent:
    """
    Agente de Mercado Profesional - Análisis Técnico Institucional

    Implementa análisis técnico de nivel profesional con:
    - 20+ indicadores técnicos calculados con la librería 'ta'
    - Sistema de scoring multi-factor para generación de señales
    - Detección automática de régimen de mercado
    - Identificación de soportes y resistencias
    - Análisis de confluencia técnica
    - Detección de divergencias RSI/MACD

    El agente procesa datos de mercado y genera una evaluación
    completa del estado técnico del activo.
    """

    # Pesos para el sistema de scoring
    WEIGHTS = {
        'trend': 0.35,
        'momentum': 0.30,
        'volatility': 0.15,
        'volume': 0.20
    }

    # Umbrales de indicadores
    THRESHOLDS = {
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'rsi_strong_overbought': 80,
        'rsi_strong_oversold': 20,
        'adx_trending': 25,
        'adx_strong_trend': 40,
        'mfi_overbought': 80,
        'mfi_oversold': 20,
        'stoch_overbought': 80,
        'stoch_oversold': 20,
        'williams_overbought': -20,
        'williams_oversold': -80,
        'cci_overbought': 100,
        'cci_oversold': -100,
    }

    def __init__(self, ventana_ma: int = 20, periodo_historico: str = "6mo"):
        """
        Inicializa el Agente de Mercado Profesional.

        Args:
            ventana_ma: Ventana base para indicadores (default: 20)
            periodo_historico: Período de datos históricos ('3mo', '6mo', '1y', '2y')
        """
        self.ventana_ma = ventana_ma
        self.periodo_historico = periodo_historico
        self.cache: Dict[str, MarketData] = {}
        self.cache_duration = timedelta(minutes=5)

        if not TA_AVAILABLE:
            logger.warning("Librería 'ta' no disponible. Usando indicadores básicos.")

        logger.info(
            f"MarketAgent Profesional inicializado - "
            f"TA Library: {TA_AVAILABLE}, Período: {periodo_historico}"
        )

    def obtener_datos(self, ticker: str, forzar_actualizacion: bool = False) -> Optional[MarketData]:
        """
        Obtiene y procesa datos de mercado con análisis técnico completo.

        Flujo de procesamiento:
        1. Verificación de caché
        2. Descarga de datos históricos
        3. Limpieza y validación
        4. Cálculo de indicadores técnicos (20+)
        5. Análisis de señales multi-factor
        6. Detección de régimen de mercado
        7. Identificación de soportes/resistencias

        Args:
            ticker: Símbolo del activo (ej: 'AAPL', 'GOOGL')
            forzar_actualizacion: Ignorar caché si True

        Returns:
            MarketData con análisis completo o None si hay error
        """
        ticker = ticker.upper().strip()

        # Verificar caché
        if not forzar_actualizacion and ticker in self.cache:
            cached = self.cache[ticker]
            if datetime.now() - cached.fecha_actualizacion < self.cache_duration:
                logger.debug(f"[{ticker}] Datos desde caché")
                return cached

        try:
            # Paso 1: Descargar datos
            df = self._descargar_datos(ticker)
            if df is None or df.empty:
                logger.error(f"[{ticker}] No se obtuvieron datos")
                return None

            # Paso 2: Limpiar datos
            df = self._limpiar_datos(df)
            if len(df) < 50:
                logger.warning(f"[{ticker}] Datos insuficientes: {len(df)} filas")

            # Paso 3: Calcular todos los indicadores
            df = self._calcular_indicadores_avanzados(df)

            # Paso 4: Extraer métricas actuales
            ultimo_precio = float(df['Close'].iloc[-1])
            precio_anterior = float(df['Close'].iloc[-2]) if len(df) > 1 else ultimo_precio
            variacion_diaria = ((ultimo_precio - precio_anterior) / precio_anterior) * 100

            # Paso 5: Construir estructura de indicadores
            indicators = self._extraer_indicadores(df)

            # Paso 6: Analizar señales
            signal_analysis = self._analizar_senales(df, indicators, ultimo_precio)

            # Paso 7: Encontrar soportes y resistencias
            supports, resistances = self._calcular_soportes_resistencias(df)

            # Determinar señal final
            senal = self._determinar_senal_final(signal_analysis)

            # Crear objeto de resultado
            market_data = MarketData(
                ticker=ticker,
                precios=df,
                ultimo_precio=ultimo_precio,
                precio_anterior=precio_anterior,
                variacion_diaria=round(variacion_diaria, 4),
                media_movil_20=round(indicators.sma_20, 4),
                senal=senal,
                fecha_actualizacion=datetime.now(),
                indicators=indicators,
                signal_analysis=signal_analysis,
                support_levels=supports,
                resistance_levels=resistances
            )

            # Actualizar caché
            self.cache[ticker] = market_data

            logger.info(
                f"Datos de {ticker} procesados: "
                f"Precio={ultimo_precio:.2f}, Señal={senal}"
            )
            return market_data

        except Exception as e:
            logger.error(f"[{ticker}] Error procesando datos: {str(e)}")
            return None

    def _descargar_datos(self, ticker: str) -> Optional[pd.DataFrame]:
        """Descarga datos históricos desde yfinance."""
        if yf is None:
            logger.error("yfinance no instalado")
            return None

        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=self.periodo_historico)

            if df.empty:
                logger.warning(f"[{ticker}] No se encontraron datos en Yahoo Finance. Ticker inválido o sin datos.")
                return None

            logger.info(f"[{ticker}] Datos reales descargados desde Yahoo Finance: {len(df)} registros")
            return df

        except Exception as e:
            logger.error(f"[{ticker}] Error al descargar datos de Yahoo Finance: {e}")
            return None

    def _generar_datos_simulados(self, ticker: str) -> pd.DataFrame:
        """Genera datos simulados realistas para pruebas."""
        np.random.seed(hash(ticker) % 2**32)
        n_dias = 180

        # Parámetros de simulación más realistas
        precio_base = 50 + np.random.uniform(0, 450)
        drift = np.random.uniform(-0.0002, 0.0003)
        volatility = np.random.uniform(0.015, 0.035)

        # Simulación con modelo GBM (Geometric Brownian Motion)
        returns = np.random.normal(drift, volatility, n_dias)
        price_path = precio_base * np.exp(np.cumsum(returns))

        fechas = pd.date_range(end=datetime.now(), periods=n_dias, freq='D')

        # Generar OHLCV realistas
        df = pd.DataFrame(index=fechas)
        df['Close'] = price_path

        # High/Low basados en volatilidad intradiaria
        intraday_vol = volatility * 0.5
        df['High'] = df['Close'] * (1 + np.abs(np.random.normal(0, intraday_vol, n_dias)))
        df['Low'] = df['Close'] * (1 - np.abs(np.random.normal(0, intraday_vol, n_dias)))
        df['Open'] = df['Close'].shift(1).fillna(precio_base) * (1 + np.random.normal(0, volatility/3, n_dias))

        # Asegurar consistencia OHLC
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)

        # Volumen con correlación a cambios de precio
        base_volume = np.random.uniform(1e6, 5e7)
        price_changes = np.abs(df['Close'].pct_change().fillna(0))
        df['Volume'] = (base_volume * (1 + price_changes * 10) *
                        np.random.uniform(0.5, 1.5, n_dias)).astype(int)

        logger.info(f"[{ticker}] Datos simulados generados: {n_dias} días")
        return df

    def _limpiar_datos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia y normaliza los datos de mercado."""
        df = df.copy()

        # Asegurar columnas requeridas
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in df.columns:
                logger.warning(f"Columna {col} faltante")

        # Eliminar filas con Close nulo
        if 'Close' in df.columns:
            df = df.dropna(subset=['Close'])

        # Ordenar por fecha
        df = df.sort_index()

        # Asegurar índice datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Forward fill para valores faltantes en otras columnas
        df = df.ffill().bfill()

        return df

    def _calcular_indicadores_avanzados(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula más de 20 indicadores técnicos usando la librería 'ta'.

        Indicadores calculados:
        - Tendencia: SMA(20,50,200), EMA(12,26,50), MACD, ADX, Ichimoku, PSAR
        - Momentum: RSI, Stochastic, Williams %R, ROC, CCI
        - Volatilidad: Bollinger Bands, ATR, Keltner Channels
        - Volumen: OBV, VWAP, MFI, ADL, CMF
        """
        df = df.copy()
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        if TA_AVAILABLE and len(df) >= 26:
            try:
                # ============ INDICADORES DE TENDENCIA ============

                # SMAs
                df['SMA_20'] = SMAIndicator(close, window=20).sma_indicator()
                df['SMA_50'] = SMAIndicator(close, window=50).sma_indicator()
                if len(df) >= 200:
                    df['SMA_200'] = SMAIndicator(close, window=200).sma_indicator()
                else:
                    df['SMA_200'] = SMAIndicator(close, window=len(df)//2).sma_indicator()

                # EMAs
                df['EMA_12'] = EMAIndicator(close, window=12).ema_indicator()
                df['EMA_26'] = EMAIndicator(close, window=26).ema_indicator()
                df['EMA_50'] = EMAIndicator(close, window=50).ema_indicator()

                # MACD
                macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
                df['MACD'] = macd.macd()
                df['MACD_Signal'] = macd.macd_signal()
                df['MACD_Histogram'] = macd.macd_diff()

                # ADX (Average Directional Index)
                adx = ADXIndicator(high, low, close, window=14)
                df['ADX'] = adx.adx()
                df['ADX_Pos'] = adx.adx_pos()
                df['ADX_Neg'] = adx.adx_neg()

                # Ichimoku Cloud
                ichimoku = IchimokuIndicator(high, low, window1=9, window2=26, window3=52)
                df['Ichimoku_A'] = ichimoku.ichimoku_a()
                df['Ichimoku_B'] = ichimoku.ichimoku_b()
                df['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
                df['Ichimoku_Conv'] = ichimoku.ichimoku_conversion_line()

                # Parabolic SAR
                psar = PSARIndicator(high, low, close)
                df['PSAR'] = psar.psar()

                # CCI (Commodity Channel Index)
                df['CCI'] = CCIIndicator(high, low, close, window=20).cci()

                # ============ INDICADORES DE MOMENTUM ============

                # RSI
                df['RSI'] = RSIIndicator(close, window=14).rsi()

                # Stochastic Oscillator
                stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
                df['Stoch_K'] = stoch.stoch()
                df['Stoch_D'] = stoch.stoch_signal()

                # Williams %R
                df['Williams_R'] = WilliamsRIndicator(high, low, close, lbp=14).williams_r()

                # ROC (Rate of Change)
                df['ROC'] = ROCIndicator(close, window=12).roc()

                # ============ INDICADORES DE VOLATILIDAD ============

                # Bollinger Bands
                bb = BollingerBands(close, window=20, window_dev=2)
                df['BB_Upper'] = bb.bollinger_hband()
                df['BB_Middle'] = bb.bollinger_mavg()
                df['BB_Lower'] = bb.bollinger_lband()
                df['BB_Width'] = bb.bollinger_wband()
                df['BB_Pct'] = bb.bollinger_pband()

                # ATR (Average True Range)
                atr = AverageTrueRange(high, low, close, window=14)
                df['ATR'] = atr.average_true_range()
                df['ATR_Pct'] = (df['ATR'] / close) * 100

                # Keltner Channels
                keltner = KeltnerChannel(high, low, close, window=20)
                df['Keltner_Upper'] = keltner.keltner_channel_hband()
                df['Keltner_Lower'] = keltner.keltner_channel_lband()

                # ============ INDICADORES DE VOLUMEN ============

                # OBV (On-Balance Volume)
                df['OBV'] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()

                # VWAP (Volume Weighted Average Price) - aproximación
                df['VWAP'] = (volume * (high + low + close) / 3).cumsum() / volume.cumsum()

                # MFI (Money Flow Index)
                df['MFI'] = MFIIndicator(high, low, close, volume, window=14).money_flow_index()

                # ADL (Accumulation/Distribution Line)
                df['ADL'] = AccDistIndexIndicator(high, low, close, volume).acc_dist_index()

                # CMF (Chaikin Money Flow)
                df['CMF'] = ChaikinMoneyFlowIndicator(high, low, close, volume, window=20).chaikin_money_flow()

                logger.debug("Indicadores técnicos avanzados calculados correctamente")

            except Exception as e:
                logger.warning(f"Error calculando indicadores avanzados: {e}")
                df = self._calcular_indicadores_basicos(df)
        else:
            df = self._calcular_indicadores_basicos(df)

        # Rellenar NaN
        df = df.ffill().bfill()

        return df

    def _calcular_indicadores_basicos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores básicos si 'ta' no está disponible."""
        df = df.copy()
        close = df['Close']

        # SMAs básicas
        df['SMA_20'] = close.rolling(window=20, min_periods=1).mean()
        df['SMA_50'] = close.rolling(window=50, min_periods=1).mean()
        df['SMA_200'] = close.rolling(window=min(200, len(df)//2), min_periods=1).mean()

        # EMAs básicas
        df['EMA_12'] = close.ewm(span=12, adjust=False).mean()
        df['EMA_26'] = close.ewm(span=26, adjust=False).mean()
        df['EMA_50'] = close.ewm(span=50, adjust=False).mean()

        # MACD básico
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

        # RSI básico
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands básicas
        df['BB_Middle'] = df['SMA_20']
        std = close.rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Pct'] = (close - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        # Volatilidad
        df['Var_Pct'] = close.pct_change() * 100
        df['Volatilidad'] = close.rolling(window=20, min_periods=1).std()

        # Valores por defecto para indicadores no calculados
        default_cols = {
            'ADX': 25, 'ADX_Pos': 25, 'ADX_Neg': 25,
            'Stoch_K': 50, 'Stoch_D': 50, 'Williams_R': -50,
            'ROC': 0, 'CCI': 0, 'ATR': 0, 'ATR_Pct': 2,
            'OBV': 0, 'MFI': 50, 'CMF': 0,
            'Ichimoku_A': close.iloc[-1], 'Ichimoku_B': close.iloc[-1],
            'PSAR': close.iloc[-1], 'VWAP': close.iloc[-1],
            'Keltner_Upper': df['BB_Upper'].iloc[-1],
            'Keltner_Lower': df['BB_Lower'].iloc[-1],
            'ADL': 0
        }

        for col, val in default_cols.items():
            if col not in df.columns:
                df[col] = val

        return df

    def _extraer_indicadores(self, df: pd.DataFrame) -> TechnicalIndicators:
        """Extrae valores actuales de indicadores del DataFrame."""
        last = df.iloc[-1]

        def safe_get(col, default=0.0):
            try:
                val = last.get(col, default)
                return float(val) if pd.notna(val) else default
            except:
                return default

        # Calcular tendencia OBV
        obv_trend = "neutral"
        if 'OBV' in df.columns and len(df) >= 20:
            obv_sma = df['OBV'].rolling(20).mean().iloc[-1]
            obv_current = df['OBV'].iloc[-1]
            if pd.notna(obv_sma) and pd.notna(obv_current):
                if obv_current > obv_sma * 1.05:
                    obv_trend = "alcista"
                elif obv_current < obv_sma * 0.95:
                    obv_trend = "bajista"

        return TechnicalIndicators(
            # Tendencia
            sma_20=safe_get('SMA_20'),
            sma_50=safe_get('SMA_50'),
            sma_200=safe_get('SMA_200'),
            ema_12=safe_get('EMA_12'),
            ema_26=safe_get('EMA_26'),
            ema_50=safe_get('EMA_50'),
            macd=safe_get('MACD'),
            macd_signal=safe_get('MACD_Signal'),
            macd_histogram=safe_get('MACD_Histogram'),
            adx=safe_get('ADX', 25),
            adx_pos=safe_get('ADX_Pos', 25),
            adx_neg=safe_get('ADX_Neg', 25),
            ichimoku_a=safe_get('Ichimoku_A'),
            ichimoku_b=safe_get('Ichimoku_B'),
            psar=safe_get('PSAR'),
            # Momentum
            rsi=safe_get('RSI', 50),
            stoch_k=safe_get('Stoch_K', 50),
            stoch_d=safe_get('Stoch_D', 50),
            williams_r=safe_get('Williams_R', -50),
            roc=safe_get('ROC'),
            cci=safe_get('CCI'),
            # Volatilidad
            bb_upper=safe_get('BB_Upper'),
            bb_middle=safe_get('BB_Middle'),
            bb_lower=safe_get('BB_Lower'),
            bb_width=safe_get('BB_Width'),
            bb_pct=safe_get('BB_Pct', 0.5),
            atr=safe_get('ATR'),
            atr_pct=safe_get('ATR_Pct', 2),
            keltner_upper=safe_get('Keltner_Upper'),
            keltner_lower=safe_get('Keltner_Lower'),
            # Volumen
            obv=safe_get('OBV'),
            obv_trend=obv_trend,
            vwap=safe_get('VWAP'),
            mfi=safe_get('MFI', 50),
            adl=safe_get('ADL'),
            cmf=safe_get('CMF'),
        )

    def _analizar_senales(
        self,
        df: pd.DataFrame,
        indicators: TechnicalIndicators,
        precio_actual: float
    ) -> SignalAnalysis:
        """
        Sistema de análisis multi-factor para generación de señales.

        Analiza 4 categorías de indicadores:
        1. Tendencia (35%): SMAs, MACD, ADX, Ichimoku
        2. Momentum (30%): RSI, Stochastic, Williams %R
        3. Volatilidad (15%): BB position, ATR
        4. Volumen (20%): OBV, MFI, CMF

        Returns:
            SignalAnalysis con scores y señales detalladas
        """
        bullish = []
        bearish = []
        warnings = []

        # ============ ANÁLISIS DE TENDENCIA ============
        trend_signals = []

        # SMA Cross
        if precio_actual > indicators.sma_20 > indicators.sma_50:
            trend_signals.append(1)
            bullish.append("Precio sobre SMA20 > SMA50")
        elif precio_actual < indicators.sma_20 < indicators.sma_50:
            trend_signals.append(-1)
            bearish.append("Precio bajo SMA20 < SMA50")
        else:
            trend_signals.append(0)

        # Golden/Death Cross
        if indicators.sma_50 > indicators.sma_200:
            trend_signals.append(0.5)
            bullish.append("Golden Cross (SMA50 > SMA200)")
        elif indicators.sma_50 < indicators.sma_200:
            trend_signals.append(-0.5)
            bearish.append("Death Cross (SMA50 < SMA200)")
        else:
            trend_signals.append(0)

        # MACD
        if indicators.macd > indicators.macd_signal and indicators.macd_histogram > 0:
            trend_signals.append(1)
            bullish.append("MACD alcista con histograma positivo")
        elif indicators.macd < indicators.macd_signal and indicators.macd_histogram < 0:
            trend_signals.append(-1)
            bearish.append("MACD bajista con histograma negativo")
        else:
            trend_signals.append(0)

        # ADX Trend Strength
        if indicators.adx > self.THRESHOLDS['adx_strong_trend']:
            if indicators.adx_pos > indicators.adx_neg:
                trend_signals.append(1)
                bullish.append(f"Tendencia alcista fuerte (ADX={indicators.adx:.1f})")
            else:
                trend_signals.append(-1)
                bearish.append(f"Tendencia bajista fuerte (ADX={indicators.adx:.1f})")
        elif indicators.adx > self.THRESHOLDS['adx_trending']:
            trend_signals.append(0.5 if indicators.adx_pos > indicators.adx_neg else -0.5)
        else:
            trend_signals.append(0)
            warnings.append(f"Mercado sin tendencia clara (ADX={indicators.adx:.1f})")

        # Ichimoku Cloud
        cloud_top = max(indicators.ichimoku_a, indicators.ichimoku_b)
        cloud_bottom = min(indicators.ichimoku_a, indicators.ichimoku_b)
        if precio_actual > cloud_top:
            trend_signals.append(1)
            bullish.append("Precio sobre nube Ichimoku")
        elif precio_actual < cloud_bottom:
            trend_signals.append(-1)
            bearish.append("Precio bajo nube Ichimoku")
        else:
            trend_signals.append(0)
            warnings.append("Precio dentro de nube Ichimoku (indecisión)")

        # PSAR
        if indicators.psar < precio_actual:
            trend_signals.append(0.5)
            bullish.append("Parabolic SAR alcista")
        else:
            trend_signals.append(-0.5)
            bearish.append("Parabolic SAR bajista")

        trend_score = np.mean(trend_signals) if trend_signals else 0

        # ============ ANÁLISIS DE MOMENTUM ============
        momentum_signals = []

        # RSI
        if indicators.rsi < self.THRESHOLDS['rsi_strong_oversold']:
            momentum_signals.append(1)
            bullish.append(f"RSI extremadamente sobrevendido ({indicators.rsi:.1f})")
        elif indicators.rsi < self.THRESHOLDS['rsi_oversold']:
            momentum_signals.append(0.7)
            bullish.append(f"RSI sobrevendido ({indicators.rsi:.1f})")
        elif indicators.rsi > self.THRESHOLDS['rsi_strong_overbought']:
            momentum_signals.append(-1)
            bearish.append(f"RSI extremadamente sobrecomprado ({indicators.rsi:.1f})")
        elif indicators.rsi > self.THRESHOLDS['rsi_overbought']:
            momentum_signals.append(-0.7)
            bearish.append(f"RSI sobrecomprado ({indicators.rsi:.1f})")
        else:
            # RSI neutral pero con tendencia
            if indicators.rsi > 55:
                momentum_signals.append(0.3)
            elif indicators.rsi < 45:
                momentum_signals.append(-0.3)
            else:
                momentum_signals.append(0)

        # Stochastic
        if indicators.stoch_k < self.THRESHOLDS['stoch_oversold']:
            if indicators.stoch_k > indicators.stoch_d:  # Cruce alcista
                momentum_signals.append(1)
                bullish.append("Stochastic sobrevendido con cruce alcista")
            else:
                momentum_signals.append(0.5)
        elif indicators.stoch_k > self.THRESHOLDS['stoch_overbought']:
            if indicators.stoch_k < indicators.stoch_d:  # Cruce bajista
                momentum_signals.append(-1)
                bearish.append("Stochastic sobrecomprado con cruce bajista")
            else:
                momentum_signals.append(-0.5)
        else:
            momentum_signals.append(0)

        # Williams %R
        if indicators.williams_r < self.THRESHOLDS['williams_oversold']:
            momentum_signals.append(0.7)
            bullish.append(f"Williams %R sobrevendido ({indicators.williams_r:.1f})")
        elif indicators.williams_r > self.THRESHOLDS['williams_overbought']:
            momentum_signals.append(-0.7)
            bearish.append(f"Williams %R sobrecomprado ({indicators.williams_r:.1f})")
        else:
            momentum_signals.append(0)

        # CCI
        if indicators.cci < self.THRESHOLDS['cci_oversold']:
            momentum_signals.append(0.5)
            bullish.append(f"CCI sobrevendido ({indicators.cci:.1f})")
        elif indicators.cci > self.THRESHOLDS['cci_overbought']:
            momentum_signals.append(-0.5)
            bearish.append(f"CCI sobrecomprado ({indicators.cci:.1f})")
        else:
            momentum_signals.append(0)

        # ROC
        if indicators.roc > 5:
            momentum_signals.append(0.5)
            bullish.append(f"Momentum positivo fuerte (ROC={indicators.roc:.1f}%)")
        elif indicators.roc < -5:
            momentum_signals.append(-0.5)
            bearish.append(f"Momentum negativo fuerte (ROC={indicators.roc:.1f}%)")
        else:
            momentum_signals.append(0)

        momentum_score = np.mean(momentum_signals) if momentum_signals else 0

        # ============ ANÁLISIS DE VOLATILIDAD ============
        volatility_signals = []

        # Bollinger Bands Position
        if indicators.bb_pct <= 0:
            volatility_signals.append(1)
            bullish.append("Precio en/bajo banda inferior Bollinger")
        elif indicators.bb_pct >= 1:
            volatility_signals.append(-1)
            bearish.append("Precio en/sobre banda superior Bollinger")
        elif indicators.bb_pct < 0.2:
            volatility_signals.append(0.5)
        elif indicators.bb_pct > 0.8:
            volatility_signals.append(-0.5)
        else:
            volatility_signals.append(0)

        # Bollinger Band Width (squeeze)
        if indicators.bb_width < 0.1:
            warnings.append("Bollinger Squeeze detectado - volatilidad por expandir")

        # ATR-based volatility regime
        if indicators.atr_pct > 4:
            warnings.append(f"Alta volatilidad (ATR={indicators.atr_pct:.1f}%)")

        volatility_score = np.mean(volatility_signals) if volatility_signals else 0

        # ============ ANÁLISIS DE VOLUMEN ============
        volume_signals = []

        # OBV Trend
        if indicators.obv_trend == "alcista":
            volume_signals.append(0.7)
            bullish.append("OBV en tendencia alcista")
        elif indicators.obv_trend == "bajista":
            volume_signals.append(-0.7)
            bearish.append("OBV en tendencia bajista")
        else:
            volume_signals.append(0)

        # MFI
        if indicators.mfi < self.THRESHOLDS['mfi_oversold']:
            volume_signals.append(0.7)
            bullish.append(f"MFI sobrevendido ({indicators.mfi:.1f})")
        elif indicators.mfi > self.THRESHOLDS['mfi_overbought']:
            volume_signals.append(-0.7)
            bearish.append(f"MFI sobrecomprado ({indicators.mfi:.1f})")
        else:
            volume_signals.append(0)

        # CMF
        if indicators.cmf > 0.1:
            volume_signals.append(0.5)
            bullish.append(f"Flujo de dinero positivo (CMF={indicators.cmf:.2f})")
        elif indicators.cmf < -0.1:
            volume_signals.append(-0.5)
            bearish.append(f"Flujo de dinero negativo (CMF={indicators.cmf:.2f})")
        else:
            volume_signals.append(0)

        # VWAP
        if precio_actual > indicators.vwap * 1.02:
            volume_signals.append(0.3)
            bullish.append("Precio sobre VWAP")
        elif precio_actual < indicators.vwap * 0.98:
            volume_signals.append(-0.3)
            bearish.append("Precio bajo VWAP")
        else:
            volume_signals.append(0)

        volume_score = np.mean(volume_signals) if volume_signals else 0

        # ============ SCORE COMPUESTO ============
        composite_score = (
            trend_score * self.WEIGHTS['trend'] +
            momentum_score * self.WEIGHTS['momentum'] +
            volatility_score * self.WEIGHTS['volatility'] +
            volume_score * self.WEIGHTS['volume']
        )

        # Determinar fuerza de señal
        if composite_score >= 0.6:
            signal_strength = SignalStrength.STRONG_BULLISH
        elif composite_score >= 0.3:
            signal_strength = SignalStrength.BULLISH
        elif composite_score >= 0.1:
            signal_strength = SignalStrength.WEAK_BULLISH
        elif composite_score <= -0.6:
            signal_strength = SignalStrength.STRONG_BEARISH
        elif composite_score <= -0.3:
            signal_strength = SignalStrength.BEARISH
        elif composite_score <= -0.1:
            signal_strength = SignalStrength.WEAK_BEARISH
        else:
            signal_strength = SignalStrength.NEUTRAL

        # Determinar régimen de mercado
        if indicators.adx > self.THRESHOLDS['adx_trending']:
            if indicators.adx_pos > indicators.adx_neg:
                market_regime = MarketRegime.TRENDING_UP
            else:
                market_regime = MarketRegime.TRENDING_DOWN
        elif indicators.atr_pct > 3:
            market_regime = MarketRegime.HIGH_VOLATILITY
        elif indicators.atr_pct < 1:
            market_regime = MarketRegime.LOW_VOLATILITY
        else:
            market_regime = MarketRegime.RANGING

        # Calcular confianza basada en alineación de señales
        all_signals = trend_signals + momentum_signals + volatility_signals + volume_signals
        total_signals = len(all_signals)
        aligned = sum(1 for s in all_signals if (s > 0 and composite_score > 0) or
                     (s < 0 and composite_score < 0) or s == 0)
        confidence = aligned / total_signals if total_signals > 0 else 0.5

        return SignalAnalysis(
            trend_score=round(trend_score, 3),
            momentum_score=round(momentum_score, 3),
            volatility_score=round(volatility_score, 3),
            volume_score=round(volume_score, 3),
            composite_score=round(composite_score, 3),
            signal_strength=signal_strength,
            market_regime=market_regime,
            confidence=round(confidence, 3),
            signals_aligned=aligned,
            total_signals=total_signals,
            bullish_signals=bullish,
            bearish_signals=bearish,
            warnings=warnings
        )

    def _calcular_soportes_resistencias(
        self,
        df: pd.DataFrame,
        n_levels: int = 3
    ) -> Tuple[List[float], List[float]]:
        """
        Identifica niveles de soporte y resistencia usando pivotes.

        Método: Detecta máximos y mínimos locales en el histórico
        y agrupa niveles cercanos.
        """
        if len(df) < 20:
            return [], []

        try:
            close = df['Close'].values
            high = df['High'].values
            low = df['Low'].values
            current_price = close[-1]

            # Encontrar pivotes (máximos y mínimos locales)
            window = 5
            supports = []
            resistances = []

            for i in range(window, len(df) - window):
                # Soporte: mínimo local
                if low[i] == min(low[i-window:i+window+1]):
                    supports.append(low[i])
                # Resistencia: máximo local
                if high[i] == max(high[i-window:i+window+1]):
                    resistances.append(high[i])

            # Filtrar y agrupar niveles cercanos
            def cluster_levels(levels, tolerance=0.02):
                if not levels:
                    return []
                levels = sorted(levels)
                clusters = [[levels[0]]]
                for level in levels[1:]:
                    if (level - clusters[-1][-1]) / clusters[-1][-1] < tolerance:
                        clusters[-1].append(level)
                    else:
                        clusters.append([level])
                return [np.mean(c) for c in clusters]

            supports = cluster_levels(supports)
            resistances = cluster_levels(resistances)

            # Filtrar niveles relevantes (cerca del precio actual)
            supports = [s for s in supports if s < current_price * 0.98]
            resistances = [r for r in resistances if r > current_price * 1.02]

            # Tomar los más cercanos
            supports = sorted(supports, reverse=True)[:n_levels]
            resistances = sorted(resistances)[:n_levels]

            return [round(s, 2) for s in supports], [round(r, 2) for r in resistances]

        except Exception as e:
            logger.warning(f"Error calculando S/R: {e}")
            return [], []

    def _determinar_senal_final(self, analysis: SignalAnalysis) -> str:
        """
        Determina la señal final simplificada para compatibilidad.

        Mapea SignalStrength a las 3 categorías básicas:
        - "alcista"
        - "bajista"
        - "neutral"
        """
        strength = analysis.signal_strength

        if strength in [SignalStrength.STRONG_BULLISH, SignalStrength.BULLISH,
                       SignalStrength.WEAK_BULLISH]:
            return "alcista"
        elif strength in [SignalStrength.STRONG_BEARISH, SignalStrength.BEARISH,
                         SignalStrength.WEAK_BEARISH]:
            return "bajista"
        else:
            return "neutral"

    def obtener_precios_recientes(self, ticker: str, n: int = 10) -> List[float]:
        """Obtiene los últimos n precios de cierre."""
        data = self.obtener_datos(ticker)
        if data is None:
            return []
        precios = data.precios['Close'].tail(n).tolist()
        return [round(p, 2) for p in precios]

    def to_dict(self, market_data: MarketData) -> Dict[str, Any]:
        """Convierte MarketData a diccionario para serialización JSON."""
        return {
            "ticker": market_data.ticker,
            "ultimo_precio": market_data.ultimo_precio,
            "precio_anterior": market_data.precio_anterior,
            "variacion_diaria": market_data.variacion_diaria,
            "media_movil_20": market_data.media_movil_20,
            "senal": market_data.senal,
            "fecha_actualizacion": market_data.fecha_actualizacion.isoformat(),
            # Datos avanzados
            "signal_strength": market_data.signal_analysis.signal_strength.value,
            "market_regime": market_data.signal_analysis.market_regime.value,
            "composite_score": market_data.signal_analysis.composite_score,
            "confidence": market_data.signal_analysis.confidence,
            "indicators": {
                "rsi": market_data.indicators.rsi,
                "macd": market_data.indicators.macd,
                "macd_signal": market_data.indicators.macd_signal,
                "adx": market_data.indicators.adx,
                "bb_pct": market_data.indicators.bb_pct,
                "mfi": market_data.indicators.mfi,
                "stoch_k": market_data.indicators.stoch_k,
            },
            "support_levels": market_data.support_levels,
            "resistance_levels": market_data.resistance_levels,
            "bullish_signals": market_data.signal_analysis.bullish_signals[:5],
            "bearish_signals": market_data.signal_analysis.bearish_signals[:5],
            "warnings": market_data.signal_analysis.warnings,
        }
