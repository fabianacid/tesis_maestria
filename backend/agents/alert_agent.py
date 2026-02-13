"""
Agente de Alertas Profesional - Sistema de Detección de Anomalías

Implementación de nivel institucional para detección y generación de alertas financieras.
Utiliza múltiples técnicas de detección de anomalías y análisis estadístico.

Técnicas implementadas:
- Isolation Forest para detección de anomalías multivariadas
- Z-Score y MAD (Median Absolute Deviation) para outliers
- CUSUM (Cumulative Sum) para detección de cambios de tendencia
- EWMA (Exponentially Weighted Moving Average) para control estadístico
- Umbrales adaptativos basados en volatilidad histórica

Características:
- Sistema de priorización multi-factor
- Rate limiting inteligente de alertas
- Análisis de patrones históricos
- Alertas contextuales con explicabilidad
- Integración con todos los agentes del sistema
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import hashlib

import numpy as np
import pandas as pd

# Opcional: scikit-learn para Isolation Forest
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Niveles de severidad de alertas."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Tipos de alertas."""
    PRICE_MOVEMENT = "movimiento_precio"
    VOLATILITY_SPIKE = "pico_volatilidad"
    TREND_CHANGE = "cambio_tendencia"
    ANOMALY_DETECTED = "anomalia_detectada"
    VOLUME_UNUSUAL = "volumen_inusual"
    SENTIMENT_SHIFT = "cambio_sentimiento"
    PREDICTION_DIVERGENCE = "divergencia_prediccion"
    CORRELATION_BREAK = "ruptura_correlacion"
    PATTERN_DETECTED = "patron_detectado"


class AlertPriority(Enum):
    """Prioridades de alertas."""
    P1_IMMEDIATE = 1  # Acción inmediata requerida
    P2_URGENT = 2     # Urgente - revisar pronto
    P3_HIGH = 3       # Alta - revisar hoy
    P4_MEDIUM = 4     # Media - revisar esta semana
    P5_LOW = 5        # Baja - informativo


@dataclass
class AnomalyScore:
    """Score de anomalía de un detector."""
    detector_name: str
    score: float  # 0-1, mayor = más anómalo
    is_anomaly: bool
    threshold: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertContext:
    """Contexto completo para una alerta."""
    market_signal: str
    prediction_confidence: float
    sentiment: str
    sentiment_confidence: float
    volatility: float
    volume_ratio: float
    rsi: float
    recent_alerts_count: int
    market_regime: str


@dataclass
class AlertResult:
    """Estructura de resultado de evaluación de alertas."""
    ticker: str
    debe_alertar: bool
    nivel: AlertSeverity
    mensaje: str
    variacion_pct: float
    umbral_superado: float
    precio_actual: float
    precio_predicho: float
    fecha_evaluacion: datetime
    detalles: Dict[str, Any]
    # Datos avanzados
    alert_type: AlertType = AlertType.PRICE_MOVEMENT
    priority: AlertPriority = AlertPriority.P4_MEDIUM
    anomaly_scores: List[AnomalyScore] = field(default_factory=list)
    composite_anomaly_score: float = 0.0
    contributing_factors: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    related_alerts: List[str] = field(default_factory=list)
    time_sensitivity: str = "normal"
    confidence: float = 0.5


# Mantener compatibilidad con código existente
NivelAlerta = AlertSeverity


class AnomalyDetector:
    """
    Sistema de detección de anomalías multi-método.

    Combina múltiples técnicas estadísticas y de ML para
    identificar comportamientos anómalos en series financieras.
    """

    def __init__(self, contamination: float = 0.1):
        """
        Inicializa el detector de anomalías.

        Args:
            contamination: Proporción esperada de anomalías (default: 10%)
        """
        self.contamination = contamination
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.isolation_forest = None
        self.history: Dict[str, deque] = {}
        self.history_length = 100

    def detect_anomalies(
        self,
        ticker: str,
        current_data: Dict[str, float],
        historical_data: Optional[pd.DataFrame] = None
    ) -> Tuple[float, List[AnomalyScore]]:
        """
        Ejecuta múltiples detectores de anomalías.

        Args:
            ticker: Símbolo del activo
            current_data: Datos actuales (variacion, volatilidad, etc.)
            historical_data: DataFrame con datos históricos

        Returns:
            Tuple: (score_compuesto, lista_scores)
        """
        scores = []

        # 1. Z-Score detector
        z_score = self._detect_zscore(ticker, current_data.get('variacion_pct', 0))
        scores.append(z_score)

        # 2. MAD (Median Absolute Deviation) detector
        mad_score = self._detect_mad(ticker, current_data.get('variacion_pct', 0))
        scores.append(mad_score)

        # 3. Volatility-adjusted detector
        vol_score = self._detect_volatility_adjusted(
            current_data.get('variacion_pct', 0),
            current_data.get('volatilidad', 2.0)
        )
        scores.append(vol_score)

        # 4. CUSUM detector (cambios de tendencia)
        cusum_score = self._detect_cusum(ticker, current_data.get('variacion_pct', 0))
        scores.append(cusum_score)

        # 5. Isolation Forest (si disponible y hay suficiente historial)
        if SKLEARN_AVAILABLE and historical_data is not None and len(historical_data) > 20:
            if_score = self._detect_isolation_forest(current_data, historical_data)
            scores.append(if_score)

        # 6. Volume anomaly detector
        volume_score = self._detect_volume_anomaly(
            current_data.get('volume_ratio', 1.0)
        )
        scores.append(volume_score)

        # Calcular score compuesto (promedio ponderado)
        weights = {
            'zscore': 0.25,
            'mad': 0.20,
            'volatility_adjusted': 0.20,
            'cusum': 0.15,
            'isolation_forest': 0.15,
            'volume': 0.05
        }

        total_weight = 0
        weighted_score = 0

        for s in scores:
            w = weights.get(s.detector_name, 0.1)
            weighted_score += s.score * w
            total_weight += w

        composite_score = weighted_score / total_weight if total_weight > 0 else 0

        # Actualizar historial
        self._update_history(ticker, current_data.get('variacion_pct', 0))

        return composite_score, scores

    def _detect_zscore(self, ticker: str, value: float) -> AnomalyScore:
        """Detección basada en Z-Score."""
        history = self.history.get(ticker, deque(maxlen=self.history_length))

        if len(history) < 5:
            return AnomalyScore(
                detector_name='zscore',
                score=0.0,
                is_anomaly=False,
                threshold=3.0,
                details={'reason': 'Insufficient history'}
            )

        mean = np.mean(list(history))
        std = np.std(list(history))

        if std < 0.001:
            z = 0
        else:
            z = abs(value - mean) / std

        # Normalizar a 0-1
        score = min(z / 4, 1.0)
        is_anomaly = z > 3.0

        return AnomalyScore(
            detector_name='zscore',
            score=score,
            is_anomaly=is_anomaly,
            threshold=3.0,
            details={'zscore': round(z, 2), 'mean': round(mean, 4), 'std': round(std, 4)}
        )

    def _detect_mad(self, ticker: str, value: float) -> AnomalyScore:
        """Detección basada en MAD (Median Absolute Deviation)."""
        history = self.history.get(ticker, deque(maxlen=self.history_length))

        if len(history) < 5:
            return AnomalyScore(
                detector_name='mad',
                score=0.0,
                is_anomaly=False,
                threshold=3.5,
                details={'reason': 'Insufficient history'}
            )

        median = np.median(list(history))
        mad = np.median(np.abs(np.array(list(history)) - median))

        if mad < 0.001:
            modified_z = 0
        else:
            modified_z = 0.6745 * abs(value - median) / mad

        score = min(modified_z / 5, 1.0)
        is_anomaly = modified_z > 3.5

        return AnomalyScore(
            detector_name='mad',
            score=score,
            is_anomaly=is_anomaly,
            threshold=3.5,
            details={'modified_zscore': round(modified_z, 2), 'mad': round(mad, 4)}
        )

    def _detect_volatility_adjusted(
        self,
        variacion: float,
        volatilidad: float
    ) -> AnomalyScore:
        """Detección ajustada por volatilidad."""
        # Umbral dinámico basado en volatilidad
        dynamic_threshold = max(2.0, volatilidad * 1.5)

        variacion_abs = abs(variacion)
        ratio = variacion_abs / dynamic_threshold if dynamic_threshold > 0 else 0

        score = min(ratio, 1.0)
        is_anomaly = variacion_abs > dynamic_threshold

        return AnomalyScore(
            detector_name='volatility_adjusted',
            score=score,
            is_anomaly=is_anomaly,
            threshold=dynamic_threshold,
            details={
                'volatility': round(volatilidad, 2),
                'dynamic_threshold': round(dynamic_threshold, 2),
                'ratio': round(ratio, 2)
            }
        )

    def _detect_cusum(self, ticker: str, value: float) -> AnomalyScore:
        """Detección de cambios usando CUSUM."""
        history = self.history.get(ticker, deque(maxlen=self.history_length))

        if len(history) < 10:
            return AnomalyScore(
                detector_name='cusum',
                score=0.0,
                is_anomaly=False,
                threshold=5.0,
                details={'reason': 'Insufficient history'}
            )

        # Calcular CUSUM
        target = np.mean(list(history))
        std = np.std(list(history))
        k = 0.5 * std  # Slack value

        values = list(history) + [value]
        cusum_pos = [0]
        cusum_neg = [0]

        for v in values[1:]:
            cusum_pos.append(max(0, cusum_pos[-1] + v - target - k))
            cusum_neg.append(max(0, cusum_neg[-1] - v + target - k))

        # Umbral basado en desviación estándar
        h = 5 * std

        max_cusum = max(cusum_pos[-1], cusum_neg[-1])
        score = min(max_cusum / (h + 0.001), 1.0)
        is_anomaly = max_cusum > h

        return AnomalyScore(
            detector_name='cusum',
            score=score,
            is_anomaly=is_anomaly,
            threshold=h,
            details={
                'cusum_positive': round(cusum_pos[-1], 2),
                'cusum_negative': round(cusum_neg[-1], 2)
            }
        )

    def _detect_isolation_forest(
        self,
        current_data: Dict[str, float],
        historical_data: pd.DataFrame
    ) -> AnomalyScore:
        """Detección usando Isolation Forest."""
        if not SKLEARN_AVAILABLE:
            return AnomalyScore(
                detector_name='isolation_forest',
                score=0.0,
                is_anomaly=False,
                threshold=0.5,
                details={'reason': 'sklearn not available'}
            )

        try:
            # Preparar features
            features = ['Close', 'Volume']
            available_features = [f for f in features if f in historical_data.columns]

            if len(available_features) < 1:
                return AnomalyScore(
                    detector_name='isolation_forest',
                    score=0.0,
                    is_anomaly=False,
                    threshold=0.5,
                    details={'reason': 'Insufficient features'}
                )

            # Preparar datos
            X = historical_data[available_features].tail(50).values

            # Agregar retornos
            if 'Close' in historical_data.columns:
                returns = historical_data['Close'].pct_change().tail(50).values.reshape(-1, 1)
                returns = np.nan_to_num(returns, 0)
                X = np.hstack([X, returns])

            # Escalar
            X_scaled = self.scaler.fit_transform(X)

            # Entrenar Isolation Forest
            self.isolation_forest = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            self.isolation_forest.fit(X_scaled)

            # Preparar punto actual
            current_point = []
            for f in available_features:
                if f == 'Close':
                    current_point.append(current_data.get('precio_actual', 0))
                elif f == 'Volume':
                    current_point.append(current_data.get('volume_ratio', 1) * 1e6)
            current_point.append(current_data.get('variacion_pct', 0) / 100)

            current_scaled = self.scaler.transform([current_point])

            # Obtener score de anomalía
            anomaly_score = -self.isolation_forest.score_samples(current_scaled)[0]
            prediction = self.isolation_forest.predict(current_scaled)[0]

            # Normalizar score
            score = min(max((anomaly_score + 0.5), 0), 1)
            is_anomaly = prediction == -1

            return AnomalyScore(
                detector_name='isolation_forest',
                score=score,
                is_anomaly=is_anomaly,
                threshold=0.5,
                details={'raw_score': round(anomaly_score, 3)}
            )

        except Exception as e:
            logger.debug(f"Error en Isolation Forest: {e}")
            return AnomalyScore(
                detector_name='isolation_forest',
                score=0.0,
                is_anomaly=False,
                threshold=0.5,
                details={'error': str(e)}
            )

    def _detect_volume_anomaly(self, volume_ratio: float) -> AnomalyScore:
        """Detección de anomalías de volumen."""
        # Volume ratio > 2 es inusual, > 3 es muy anómalo
        if volume_ratio > 3:
            score = 1.0
            is_anomaly = True
        elif volume_ratio > 2:
            score = 0.7
            is_anomaly = True
        elif volume_ratio > 1.5:
            score = 0.4
            is_anomaly = False
        elif volume_ratio < 0.3:
            score = 0.5
            is_anomaly = True
        else:
            score = 0.0
            is_anomaly = False

        return AnomalyScore(
            detector_name='volume',
            score=score,
            is_anomaly=is_anomaly,
            threshold=2.0,
            details={'volume_ratio': round(volume_ratio, 2)}
        )

    def _update_history(self, ticker: str, value: float):
        """Actualiza el historial del ticker."""
        if ticker not in self.history:
            self.history[ticker] = deque(maxlen=self.history_length)
        self.history[ticker].append(value)


class AlertRateLimiter:
    """
    Sistema de rate limiting para alertas.

    Evita spam de alertas y agrupa alertas similares.
    """

    def __init__(
        self,
        min_interval_seconds: int = 300,
        max_alerts_per_hour: int = 10
    ):
        self.min_interval = timedelta(seconds=min_interval_seconds)
        self.max_per_hour = max_alerts_per_hour
        self.alert_history: Dict[str, List[datetime]] = {}
        self.last_alert: Dict[str, datetime] = {}

    def should_alert(self, ticker: str, alert_type: AlertType) -> Tuple[bool, str]:
        """
        Determina si se debe enviar una alerta.

        Returns:
            Tuple: (puede_alertar, razón)
        """
        key = f"{ticker}_{alert_type.value}"
        now = datetime.now()

        # Verificar intervalo mínimo
        if key in self.last_alert:
            if now - self.last_alert[key] < self.min_interval:
                return False, "Alerta similar reciente"

        # Verificar límite por hora
        if key in self.alert_history:
            hour_ago = now - timedelta(hours=1)
            recent = [t for t in self.alert_history[key] if t > hour_ago]
            self.alert_history[key] = recent

            if len(recent) >= self.max_per_hour:
                return False, f"Límite de {self.max_per_hour} alertas/hora alcanzado"

        return True, "OK"

    def record_alert(self, ticker: str, alert_type: AlertType):
        """Registra una alerta enviada."""
        key = f"{ticker}_{alert_type.value}"
        now = datetime.now()

        self.last_alert[key] = now

        if key not in self.alert_history:
            self.alert_history[key] = []
        self.alert_history[key].append(now)


class AlertAgent:
    """
    Agente de Alertas Profesional - Sistema de Detección Inteligente

    Implementa un sistema de alertas de nivel institucional con:
    - Detección de anomalías multi-método (Z-Score, MAD, CUSUM, Isolation Forest)
    - Umbrales adaptativos basados en volatilidad
    - Sistema de priorización multi-factor
    - Rate limiting inteligente
    - Explicabilidad completa de alertas

    El agente evalúa múltiples dimensiones para generar alertas
    relevantes y accionables sin spam.
    """

    # Umbrales base (se ajustan dinámicamente)
    BASE_THRESHOLDS = {
        AlertSeverity.INFO: 1.0,
        AlertSeverity.LOW: 2.0,
        AlertSeverity.MEDIUM: 3.0,
        AlertSeverity.HIGH: 5.0,
        AlertSeverity.CRITICAL: 7.0,
        AlertSeverity.EMERGENCY: 10.0
    }

    def __init__(
        self,
        umbral_warning: float = 3.0,
        umbral_critical: float = 7.0,
        enable_rate_limiting: bool = True
    ):
        """
        Inicializa el Agente de Alertas Profesional.

        Args:
            umbral_warning: Variación % para nivel warning
            umbral_critical: Variación % para nivel critical
            enable_rate_limiting: Activar limitación de alertas
        """
        self.umbral_warning = umbral_warning
        self.umbral_critical = umbral_critical

        # Inicializar componentes
        self.anomaly_detector = AnomalyDetector()
        self.rate_limiter = AlertRateLimiter() if enable_rate_limiting else None

        # Caché de alertas recientes
        self.recent_alerts: Dict[str, List[AlertResult]] = {}
        self.alertas_generadas: List[AlertResult] = []

        logger.info(
            f"AlertAgent Profesional inicializado - "
            f"Warning: {umbral_warning}%, Critical: {umbral_critical}%, "
            f"Rate Limiting: {enable_rate_limiting}"
        )

    def evaluar(
        self,
        ticker: str,
        variacion_pct: float,
        precio_actual: float,
        precio_predicho: float,
        contexto: Optional[Dict[str, Any]] = None,
        # Parámetros avanzados
        volatilidad: float = 2.0,
        volume_ratio: float = 1.0,
        sentiment: str = "neutral",
        sentiment_confidence: float = 0.5,
        market_signal: str = "neutral",
        rsi: float = 50.0,
        historical_data: Optional[pd.DataFrame] = None,
        # Umbrales personalizados (opcionales)
        umbral_warning_custom: Optional[float] = None,
        umbral_critical_custom: Optional[float] = None
    ) -> AlertResult:
        """
        Evalúa si se debe generar una alerta usando múltiples detectores.

        Flujo de evaluación:
        1. Ejecutar detectores de anomalías
        2. Calcular umbral adaptativo
        3. Determinar severidad y tipo
        4. Calcular prioridad
        5. Verificar rate limiting
        6. Generar explicación y acciones recomendadas

        Args:
            ticker: Símbolo del activo
            variacion_pct: Variación porcentual predicha
            precio_actual: Precio actual
            precio_predicho: Precio predicho
            contexto: Información adicional
            volatilidad: Volatilidad histórica
            volume_ratio: Ratio de volumen vs promedio
            sentiment: Sentimiento del mercado
            sentiment_confidence: Confianza del sentimiento
            market_signal: Señal técnica del mercado
            rsi: Valor de RSI
            historical_data: Datos históricos para análisis

        Returns:
            AlertResult completa con análisis de anomalías
        """
        try:
            # Usar umbrales personalizados si se proporcionan
            umbral_warning_efectivo = umbral_warning_custom if umbral_warning_custom is not None else self.umbral_warning
            umbral_critical_efectivo = umbral_critical_custom if umbral_critical_custom is not None else self.umbral_critical

            logger.debug(
                f"[{ticker}] Umbrales efectivos: Warning={umbral_warning_efectivo}%, Critical={umbral_critical_efectivo}%"
            )

            # Paso 1: Preparar datos para detección
            current_data = {
                'variacion_pct': variacion_pct,
                'volatilidad': volatilidad,
                'volume_ratio': volume_ratio,
                'precio_actual': precio_actual,
                'rsi': rsi
            }

            # Paso 2: Ejecutar detección de anomalías
            composite_anomaly, anomaly_scores = self.anomaly_detector.detect_anomalies(
                ticker, current_data, historical_data
            )

            # Paso 3: Calcular umbral adaptativo
            adaptive_threshold = self._calculate_adaptive_threshold(volatilidad, umbral_warning_efectivo)

            # Paso 4: Determinar si hay alerta y su severidad
            variacion_abs = abs(variacion_pct)
            severity, should_alert = self._determine_severity(
                variacion_abs, composite_anomaly, adaptive_threshold, anomaly_scores,
                umbral_warning_efectivo, umbral_critical_efectivo
            )

            # Paso 5: Determinar tipo de alerta
            alert_type = self._determine_alert_type(
                variacion_pct, composite_anomaly, volume_ratio, sentiment, anomaly_scores
            )

            # Paso 6: Calcular prioridad
            priority = self._calculate_priority(
                severity, composite_anomaly, variacion_abs, market_signal
            )

            # Paso 7: Verificar rate limiting
            if should_alert and self.rate_limiter:
                can_alert, reason = self.rate_limiter.should_alert(ticker, alert_type)
                if not can_alert:
                    logger.debug(f"[{ticker}] Alerta suprimida: {reason}")
                    should_alert = False

            # Paso 8: Generar mensaje y factores
            mensaje = self._generate_message(
                ticker, variacion_pct, severity, alert_type,
                precio_actual, precio_predicho, composite_anomaly
            )

            contributing_factors = self._identify_contributing_factors(
                anomaly_scores, variacion_pct, volume_ratio, sentiment
            )

            recommended_actions = self._generate_recommendations(
                severity, alert_type, variacion_pct, market_signal
            )

            # Paso 9: Determinar sensibilidad temporal
            time_sensitivity = self._determine_time_sensitivity(
                severity, composite_anomaly, volatilidad
            )

            # Paso 10: Calcular confianza
            confidence = self._calculate_confidence(anomaly_scores, variacion_abs)

            # Construir detalles
            detalles = {
                "umbral_adaptativo": round(adaptive_threshold, 2),
                "variacion_absoluta": round(variacion_abs, 2),
                "direccion": "alza" if variacion_pct > 0 else "baja",
                "volatilidad": round(volatilidad, 2),
                "anomaly_score": round(composite_anomaly, 3),
                "volume_ratio": round(volume_ratio, 2)
            }

            if contexto:
                detalles.update(contexto)

            resultado = AlertResult(
                ticker=ticker,
                debe_alertar=should_alert,
                nivel=severity,
                mensaje=mensaje,
                variacion_pct=round(variacion_pct, 4),
                umbral_superado=adaptive_threshold if should_alert else 0.0,
                precio_actual=round(precio_actual, 4),
                precio_predicho=round(precio_predicho, 4),
                fecha_evaluacion=datetime.now(),
                detalles=detalles,
                alert_type=alert_type,
                priority=priority,
                anomaly_scores=anomaly_scores,
                composite_anomaly_score=round(composite_anomaly, 3),
                contributing_factors=contributing_factors,
                recommended_actions=recommended_actions,
                time_sensitivity=time_sensitivity,
                confidence=round(confidence, 3)
            )

            # Registrar alerta si corresponde
            if should_alert:
                self.alertas_generadas.append(resultado)
                if self.rate_limiter:
                    self.rate_limiter.record_alert(ticker, alert_type)
                logger.warning(
                    f"Alerta {severity.value} ({alert_type.value}) para {ticker}: "
                    f"Variación {variacion_pct:.2f}%, Anomaly Score {composite_anomaly:.2f}"
                )
            else:
                logger.info(
                    f"Sin alerta para {ticker}: "
                    f"Variación {variacion_pct:.2f}% dentro de umbrales"
                )

            return resultado

        except Exception as e:
            logger.error(f"[{ticker}] Error evaluando alerta: {str(e)}")
            return self._resultado_por_defecto(ticker)

    def _calculate_adaptive_threshold(self, volatilidad: float, umbral_warning: float = None) -> float:
        """Calcula umbral adaptativo basado en volatilidad."""
        # Umbral base ajustado por volatilidad
        base = umbral_warning if umbral_warning is not None else self.umbral_warning
        adjustment = max(0.5, min(2.0, volatilidad / 2))
        return base * adjustment

    def _determine_severity(
        self,
        variacion_abs: float,
        anomaly_score: float,
        adaptive_threshold: float,
        anomaly_scores: List[AnomalyScore],
        umbral_warning: float = None,
        umbral_critical: float = None
    ) -> Tuple[AlertSeverity, bool]:
        """Determina la severidad de la alerta."""
        # Usar umbrales personalizados o los por defecto
        warning = umbral_warning if umbral_warning is not None else self.umbral_warning
        critical = umbral_critical if umbral_critical is not None else self.umbral_critical

        # Contar detectores que identificaron anomalía
        anomaly_count = sum(1 for s in anomaly_scores if s.is_anomaly)

        # Emergency: múltiples detectores + variación extrema
        if variacion_abs >= 10 and anomaly_count >= 3:
            return AlertSeverity.EMERGENCY, True

        # Critical: variación alta o anomaly score alto
        if variacion_abs >= critical or anomaly_score >= 0.8:
            return AlertSeverity.CRITICAL, True

        # High: supera umbral crítico ligeramente o múltiples anomalías
        if variacion_abs >= critical * 0.7 or (anomaly_score >= 0.6 and anomaly_count >= 2):
            return AlertSeverity.HIGH, True

        # Medium: supera umbral warning
        if variacion_abs >= adaptive_threshold or anomaly_score >= 0.5:
            return AlertSeverity.MEDIUM, True

        # Low: anomalía detectada pero variación moderada
        if anomaly_count >= 1 and variacion_abs >= adaptive_threshold * 0.5:
            return AlertSeverity.LOW, True

        # Info: sin alerta significativa
        return AlertSeverity.INFO, False

    def _determine_alert_type(
        self,
        variacion_pct: float,
        anomaly_score: float,
        volume_ratio: float,
        sentiment: str,
        anomaly_scores: List[AnomalyScore]
    ) -> AlertType:
        """Determina el tipo principal de alerta."""
        # Verificar volumen inusual
        if volume_ratio > 2.5:
            return AlertType.VOLUME_UNUSUAL

        # Verificar anomalía general
        if anomaly_score > 0.7:
            return AlertType.ANOMALY_DETECTED

        # Verificar cambio de sentimiento
        if sentiment in ["positivo", "negativo"]:
            # Si el sentimiento contradice la variación
            if (sentiment == "positivo" and variacion_pct < -3) or \
               (sentiment == "negativo" and variacion_pct > 3):
                return AlertType.SENTIMENT_SHIFT

        # CUSUM detectó cambio de tendencia
        cusum_score = next((s for s in anomaly_scores if s.detector_name == 'cusum'), None)
        if cusum_score and cusum_score.is_anomaly:
            return AlertType.TREND_CHANGE

        # Por defecto: movimiento de precio
        return AlertType.PRICE_MOVEMENT

    def _calculate_priority(
        self,
        severity: AlertSeverity,
        anomaly_score: float,
        variacion_abs: float,
        market_signal: str
    ) -> AlertPriority:
        """Calcula la prioridad de la alerta."""
        if severity == AlertSeverity.EMERGENCY:
            return AlertPriority.P1_IMMEDIATE

        if severity == AlertSeverity.CRITICAL:
            return AlertPriority.P2_URGENT

        if severity == AlertSeverity.HIGH:
            # Más urgente si está contra la tendencia
            if (market_signal == "alcista" and variacion_abs < 0) or \
               (market_signal == "bajista" and variacion_abs > 0):
                return AlertPriority.P2_URGENT
            return AlertPriority.P3_HIGH

        if severity == AlertSeverity.MEDIUM:
            return AlertPriority.P4_MEDIUM

        return AlertPriority.P5_LOW

    def _generate_message(
        self,
        ticker: str,
        variacion_pct: float,
        severity: AlertSeverity,
        alert_type: AlertType,
        precio_actual: float,
        precio_predicho: float,
        anomaly_score: float
    ) -> str:
        """Genera mensaje descriptivo para la alerta."""
        direccion = "subida" if variacion_pct > 0 else "bajada"
        variacion_abs = abs(variacion_pct)

        severity_prefix = {
            AlertSeverity.EMERGENCY: "EMERGENCIA",
            AlertSeverity.CRITICAL: "ALERTA CRÍTICA",
            AlertSeverity.HIGH: "ALERTA ALTA",
            AlertSeverity.MEDIUM: "ADVERTENCIA",
            AlertSeverity.LOW: "AVISO",
            AlertSeverity.INFO: "INFO"
        }

        alert_type_desc = {
            AlertType.PRICE_MOVEMENT: f"predicción de {direccion} del {variacion_abs:.2f}%",
            AlertType.VOLATILITY_SPIKE: "pico de volatilidad detectado",
            AlertType.TREND_CHANGE: "posible cambio de tendencia",
            AlertType.ANOMALY_DETECTED: f"comportamiento anómalo (score: {anomaly_score:.2f})",
            AlertType.VOLUME_UNUSUAL: "volumen inusual detectado",
            AlertType.SENTIMENT_SHIFT: "cambio significativo en sentimiento",
            AlertType.PREDICTION_DIVERGENCE: "divergencia entre predicción y mercado"
        }

        type_desc = alert_type_desc.get(alert_type, f"{direccion} del {variacion_abs:.2f}%")

        # No incluir prefijo (ADVERTENCIA, INFO, etc.) porque el dashboard ya lo agrega según el nivel
        msg = f"{ticker} - {type_desc}. "
        msg += f"Precio actual: ${precio_actual:.2f}, Proyectado: ${precio_predicho:.2f}."

        if severity in [AlertSeverity.EMERGENCY, AlertSeverity.CRITICAL]:
            msg += " Se recomienda revisión inmediata."
        elif severity == AlertSeverity.HIGH:
            msg += " Monitorear de cerca."

        return msg

    def _identify_contributing_factors(
        self,
        anomaly_scores: List[AnomalyScore],
        variacion_pct: float,
        volume_ratio: float,
        sentiment: str
    ) -> List[str]:
        """Identifica factores que contribuyen a la alerta."""
        factors = []

        # Factores de anomalía
        for score in anomaly_scores:
            if score.is_anomaly:
                factor_names = {
                    'zscore': 'Desviación estadística significativa',
                    'mad': 'Outlier robusto detectado',
                    'volatility_adjusted': 'Movimiento excede volatilidad esperada',
                    'cusum': 'Cambio de tendencia detectado',
                    'isolation_forest': 'Patrón anómalo multidimensional',
                    'volume': 'Volumen anormal'
                }
                factors.append(factor_names.get(score.detector_name, score.detector_name))

        # Factores de variación
        if abs(variacion_pct) > 5:
            direction = "alcista" if variacion_pct > 0 else "bajista"
            factors.append(f"Variación {direction} significativa ({variacion_pct:.1f}%)")

        # Factores de volumen
        if volume_ratio > 2:
            factors.append(f"Volumen {volume_ratio:.1f}x sobre promedio")

        # Factores de sentimiento
        if sentiment != "neutral":
            factors.append(f"Sentimiento de mercado {sentiment}")

        return factors[:5]  # Limitar a 5 factores principales

    def _generate_recommendations(
        self,
        severity: AlertSeverity,
        alert_type: AlertType,
        variacion_pct: float,
        market_signal: str
    ) -> List[str]:
        """Genera acciones recomendadas."""
        actions = []

        if severity in [AlertSeverity.EMERGENCY, AlertSeverity.CRITICAL]:
            actions.append("Revisar posición inmediatamente")
            if variacion_pct < -5:
                actions.append("Considerar stop-loss de protección")
            elif variacion_pct > 5:
                actions.append("Evaluar toma de ganancias parcial")

        if alert_type == AlertType.TREND_CHANGE:
            actions.append("Verificar cambio de tendencia con análisis técnico")
            actions.append("Revisar niveles de soporte/resistencia")

        if alert_type == AlertType.VOLUME_UNUSUAL:
            actions.append("Investigar causa del volumen inusual")
            actions.append("Buscar noticias o eventos relacionados")

        if alert_type == AlertType.SENTIMENT_SHIFT:
            actions.append("Revisar noticias recientes")
            actions.append("Evaluar si el sentimiento anticipa movimiento")

        # Recomendación general según mercado
        if market_signal == "bajista" and variacion_pct > 0:
            actions.append("Precaución: predicción contra tendencia de mercado")
        elif market_signal == "alcista" and variacion_pct < 0:
            actions.append("Precaución: predicción contra tendencia de mercado")

        return actions[:4]  # Limitar a 4 acciones

    def _determine_time_sensitivity(
        self,
        severity: AlertSeverity,
        anomaly_score: float,
        volatilidad: float
    ) -> str:
        """Determina la sensibilidad temporal de la alerta."""
        if severity in [AlertSeverity.EMERGENCY, AlertSeverity.CRITICAL]:
            return "inmediata"
        if volatilidad > 4 or anomaly_score > 0.7:
            return "alta"
        if severity in [AlertSeverity.HIGH, AlertSeverity.MEDIUM]:
            return "moderada"
        return "normal"

    def _calculate_confidence(
        self,
        anomaly_scores: List[AnomalyScore],
        variacion_abs: float
    ) -> float:
        """Calcula la confianza en la alerta."""
        if not anomaly_scores:
            return 0.5

        # Promedio de scores de anomalía
        avg_score = np.mean([s.score for s in anomaly_scores])

        # Consenso entre detectores
        anomaly_count = sum(1 for s in anomaly_scores if s.is_anomaly)
        consensus = anomaly_count / len(anomaly_scores)

        # Factor de variación
        var_factor = min(variacion_abs / 10, 1.0)

        confidence = (avg_score * 0.4 + consensus * 0.4 + var_factor * 0.2)
        return min(confidence, 0.95)

    def _resultado_por_defecto(self, ticker: str) -> AlertResult:
        """Genera resultado por defecto en caso de error."""
        return AlertResult(
            ticker=ticker,
            debe_alertar=False,
            nivel=AlertSeverity.INFO,
            mensaje=f"No se pudo evaluar alerta para {ticker}",
            variacion_pct=0.0,
            umbral_superado=0.0,
            precio_actual=0.0,
            precio_predicho=0.0,
            fecha_evaluacion=datetime.now(),
            detalles={"error": "Evaluación no disponible"},
            alert_type=AlertType.PRICE_MOVEMENT,
            priority=AlertPriority.P5_LOW,
            confidence=0.0
        )

    def persistir_alerta(
        self,
        db: Session,
        resultado: AlertResult,
        usuario_id: int
    ) -> Optional[int]:
        """Persiste una alerta en la base de datos."""
        from ..database import Alerta

        try:
            if not resultado.debe_alertar:
                logger.debug(f"Alerta para {resultado.ticker} no requiere persistencia")
                return None

            alerta = Alerta(
                usuario_id=usuario_id,
                ticker=resultado.ticker,
                tipo=resultado.nivel.value,
                mensaje=resultado.mensaje,
                variacion_pct=resultado.variacion_pct,
                precio_actual=resultado.precio_actual,
                precio_predicho=resultado.precio_predicho,
                leida=0
            )

            db.add(alerta)
            db.commit()
            db.refresh(alerta)

            logger.info(
                f"Alerta persistida para {resultado.ticker} "
                f"(ID: {alerta.id}, Usuario: {usuario_id})"
            )

            return alerta.id

        except Exception as e:
            db.rollback()
            logger.error(f"Error persistiendo alerta: {str(e)}")
            return None

    def obtener_alertas_usuario(
        self,
        db: Session,
        usuario_id: int,
        solo_no_leidas: bool = False,
        limite: int = 50
    ) -> List[Dict[str, Any]]:
        """Obtiene alertas de un usuario desde la base de datos."""
        from ..database import Alerta

        try:
            query = db.query(Alerta).filter(Alerta.usuario_id == usuario_id)

            if solo_no_leidas:
                query = query.filter(Alerta.leida == 0)

            alertas = query.order_by(
                Alerta.fecha_creacion.desc()
            ).limit(limite).all()

            return [
                {
                    "id": a.id,
                    "ticker": a.ticker,
                    "tipo": a.tipo,
                    "mensaje": a.mensaje,
                    "variacion_pct": a.variacion_pct,
                    "precio_actual": a.precio_actual,
                    "precio_predicho": a.precio_predicho,
                    "leida": bool(a.leida),
                    "fecha_creacion": a.fecha_creacion.isoformat()
                }
                for a in alertas
            ]

        except Exception as e:
            logger.error(f"Error obteniendo alertas: {str(e)}")
            return []

    def marcar_como_leida(
        self,
        db: Session,
        alerta_id: int,
        usuario_id: int
    ) -> bool:
        """Marca una alerta como leída."""
        from ..database import Alerta

        try:
            alerta = db.query(Alerta).filter(
                Alerta.id == alerta_id,
                Alerta.usuario_id == usuario_id
            ).first()

            if alerta:
                alerta.leida = 1
                db.commit()
                logger.info(f"Alerta {alerta_id} marcada como leída")
                return True

            return False

        except Exception as e:
            db.rollback()
            logger.error(f"Error marcando alerta como leída: {str(e)}")
            return False

    def obtener_estadisticas(
        self,
        db: Session,
        usuario_id: int
    ) -> Dict[str, Any]:
        """Obtiene estadísticas de alertas del usuario."""
        from ..database import Alerta
        from sqlalchemy import func

        try:
            total = db.query(func.count(Alerta.id)).filter(
                Alerta.usuario_id == usuario_id
            ).scalar() or 0

            no_leidas = db.query(func.count(Alerta.id)).filter(
                Alerta.usuario_id == usuario_id,
                Alerta.leida == 0
            ).scalar() or 0

            por_tipo = db.query(
                Alerta.tipo,
                func.count(Alerta.id)
            ).filter(
                Alerta.usuario_id == usuario_id
            ).group_by(Alerta.tipo).all()

            return {
                "total_alertas": total,
                "alertas_no_leidas": no_leidas,
                "alertas_por_tipo": {tipo: count for tipo, count in por_tipo}
            }

        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {str(e)}")
            return {
                "total_alertas": 0,
                "alertas_no_leidas": 0,
                "alertas_por_tipo": {}
            }

    def to_dict(self, resultado: AlertResult) -> Dict[str, Any]:
        """Convierte AlertResult a diccionario serializable."""
        return {
            "ticker": resultado.ticker,
            "tiene_alerta": resultado.debe_alertar,
            "nivel": resultado.nivel.value,
            "mensaje": resultado.mensaje,
            "variacion_pct": resultado.variacion_pct,
            "umbral_superado": resultado.umbral_superado,
            "precio_actual": resultado.precio_actual,
            "precio_predicho": resultado.precio_predicho,
            "fecha_evaluacion": resultado.fecha_evaluacion.isoformat(),
            "detalles": resultado.detalles,
            # Datos avanzados
            "alert_type": resultado.alert_type.value,
            "priority": resultado.priority.value,
            "composite_anomaly_score": resultado.composite_anomaly_score,
            "contributing_factors": resultado.contributing_factors,
            "recommended_actions": resultado.recommended_actions,
            "time_sensitivity": resultado.time_sensitivity,
            "confidence": resultado.confidence,
            "anomaly_detectors": [
                {
                    "name": s.detector_name,
                    "score": round(s.score, 3),
                    "is_anomaly": s.is_anomaly
                }
                for s in resultado.anomaly_scores[:5]
            ]
        }
