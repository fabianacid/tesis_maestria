"""
Agente de Recomendación Profesional - Sistema de Decisión Multi-Factor

Implementación de nivel institucional para generación de recomendaciones de inversión.
Utiliza un sistema de decisión basado en múltiples factores con calibración probabilística.

Características:
- Sistema de scoring multi-factor con 15+ variables
- Análisis de riesgo integrado (VaR, volatilidad, correlaciones)
- Recomendaciones de position sizing
- Explicabilidad completa de decisiones
- Calibración dinámica basada en condiciones de mercado
- Integración de señales técnicas, fundamentales y de sentimiento

Arquitectura:
- Factor Model para combinación de señales
- Risk Management Module
- Position Sizing Calculator
- Confidence Calibration System
- Explainability Engine
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

import numpy as np

logger = logging.getLogger(__name__)


class RecommendationType(Enum):
    """Tipos de recomendación."""
    STRONG_BUY = "compra_fuerte"
    BUY = "compra"
    WEAK_BUY = "compra_debil"
    HOLD = "mantener"
    WEAK_SELL = "venta_debil"
    SELL = "venta"
    STRONG_SELL = "venta_fuerte"


class RiskLevel(Enum):
    """Niveles de riesgo."""
    VERY_LOW = "muy_bajo"
    LOW = "bajo"
    MODERATE = "moderado"
    HIGH = "alto"
    VERY_HIGH = "muy_alto"


class TimeHorizon(Enum):
    """Horizontes temporales."""
    INTRADAY = "intraday"
    SHORT_TERM = "corto_plazo"  # 1-5 días
    MEDIUM_TERM = "mediano_plazo"  # 1-4 semanas
    LONG_TERM = "largo_plazo"  # 1+ meses


@dataclass
class RiskAssessment:
    """Evaluación de riesgo detallada."""
    risk_level: RiskLevel
    volatility_score: float
    var_95: float  # Value at Risk 95%
    max_drawdown_expected: float
    correlation_risk: float
    liquidity_risk: float
    event_risk: float
    overall_risk_score: float
    risk_factors: List[str] = field(default_factory=list)


@dataclass
class PositionSizing:
    """Recomendación de tamaño de posición."""
    suggested_allocation_pct: float  # % del portfolio
    max_allocation_pct: float
    entry_strategy: str
    exit_strategy: str
    stop_loss_pct: float
    take_profit_pct: float
    risk_reward_ratio: float


@dataclass
class FactorContribution:
    """Contribución de cada factor a la decisión."""
    factor_name: str
    raw_value: float
    normalized_score: float
    weight: float
    contribution: float
    direction: str  # "bullish", "bearish", "neutral"
    confidence: float


@dataclass
class RecommendationResult:
    """Estructura de resultado de recomendación completa."""
    ticker: str
    recomendacion: str
    tipo: str
    confianza: float
    razon: str
    factores: Dict[str, Any]
    fecha: datetime
    # Datos avanzados
    recommendation_type: RecommendationType = RecommendationType.HOLD
    risk_assessment: RiskAssessment = None
    position_sizing: PositionSizing = None
    factor_contributions: List[FactorContribution] = field(default_factory=list)
    time_horizon: TimeHorizon = TimeHorizon.SHORT_TERM
    price_target: float = 0.0
    expected_return: float = 0.0
    probability_profit: float = 0.5
    key_catalysts: List[str] = field(default_factory=list)
    key_risks: List[str] = field(default_factory=list)
    # Explicaciones para inversor minorista
    accion_sugerida: str = ""
    explicacion_simple: str = ""
    nivel_riesgo_simple: str = ""
    porque_esta_recomendacion: str = ""
    icono: str = ""


class FactorModel:
    """
    Modelo de factores para combinación de señales.

    Implementa un sistema de scoring multi-factor que pondera
    diferentes tipos de señales según su relevancia y confiabilidad.
    """

    # Definición de factores y pesos
    FACTOR_WEIGHTS = {
        # Factores técnicos (40%)
        'trend_signal': 0.12,
        'momentum_signal': 0.10,
        'volatility_signal': 0.08,
        'volume_signal': 0.06,
        'support_resistance': 0.04,

        # Factores de predicción (35%)
        'model_prediction': 0.15,
        'prediction_confidence': 0.10,
        'ensemble_agreement': 0.10,

        # Factores de sentimiento (15%)
        'sentiment_score': 0.08,
        'sentiment_trend': 0.04,
        'news_impact': 0.03,

        # Factores de riesgo (10%)
        'risk_adjusted_return': 0.05,
        'market_regime': 0.03,
        'correlation_factor': 0.02,
    }

    @classmethod
    def calculate_composite_score(
        cls,
        factors: Dict[str, float]
    ) -> Tuple[float, List[FactorContribution]]:
        """
        Calcula score compuesto ponderando todos los factores.

        Args:
            factors: Diccionario con valores de cada factor

        Returns:
            Tuple: (score_compuesto, lista_contribuciones)
        """
        contributions = []
        total_score = 0.0
        total_weight = 0.0

        for factor_name, weight in cls.FACTOR_WEIGHTS.items():
            if factor_name in factors:
                raw_value = factors[factor_name]

                # Normalizar a [-1, 1]
                normalized = np.clip(raw_value, -1, 1)

                # Calcular contribución
                contribution = normalized * weight
                total_score += contribution
                total_weight += weight

                # Determinar dirección
                if normalized > 0.1:
                    direction = "bullish"
                elif normalized < -0.1:
                    direction = "bearish"
                else:
                    direction = "neutral"

                contributions.append(FactorContribution(
                    factor_name=factor_name,
                    raw_value=raw_value,
                    normalized_score=normalized,
                    weight=weight,
                    contribution=contribution,
                    direction=direction,
                    confidence=abs(normalized)
                ))

        # Normalizar score final
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.0

        # Ordenar contribuciones por impacto absoluto
        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)

        return final_score, contributions


class RiskManager:
    """
    Módulo de gestión de riesgo.

    Evalúa el riesgo de una posición considerando múltiples factores.
    """

    @staticmethod
    def assess_risk(
        volatility: float,
        variacion_pct: float,
        sentiment_confidence: float,
        market_regime: str = "normal"
    ) -> RiskAssessment:
        """
        Evalúa el nivel de riesgo de la posición.

        Args:
            volatility: Volatilidad histórica (%)
            variacion_pct: Variación esperada (%)
            sentiment_confidence: Confianza del sentimiento
            market_regime: Régimen de mercado actual

        Returns:
            RiskAssessment con evaluación completa
        """
        risk_factors = []

        # Volatility score (0-1, mayor = más riesgo)
        volatility_score = min(1.0, volatility / 5.0)
        if volatility_score > 0.6:
            risk_factors.append(f"Alta volatilidad ({volatility:.1f}%)")

        # VaR 95% estimado (asumiendo distribución normal)
        var_95 = abs(variacion_pct) * 1.65  # 95% confidence
        if var_95 > 5:
            risk_factors.append(f"VaR 95% elevado ({var_95:.1f}%)")

        # Max drawdown esperado
        max_dd = volatility * 2.5
        if max_dd > 10:
            risk_factors.append(f"Drawdown potencial significativo ({max_dd:.1f}%)")

        # Correlation risk (simplificado)
        correlation_risk = 0.3 if market_regime in ["alta_volatilidad", "tendencia_bajista"] else 0.15

        # Liquidity risk (placeholder - en producción usar datos reales)
        liquidity_risk = 0.1

        # Event risk basado en incertidumbre del sentimiento
        event_risk = 1 - sentiment_confidence
        if event_risk > 0.5:
            risk_factors.append("Alta incertidumbre en sentimiento")

        # Overall risk score
        overall_risk = (
            volatility_score * 0.35 +
            (var_95 / 10) * 0.25 +
            correlation_risk * 0.15 +
            liquidity_risk * 0.10 +
            event_risk * 0.15
        )
        overall_risk = min(1.0, overall_risk)

        # Determinar nivel de riesgo
        if overall_risk < 0.2:
            risk_level = RiskLevel.VERY_LOW
        elif overall_risk < 0.4:
            risk_level = RiskLevel.LOW
        elif overall_risk < 0.6:
            risk_level = RiskLevel.MODERATE
        elif overall_risk < 0.8:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.VERY_HIGH
            risk_factors.append("Nivel de riesgo general muy elevado")

        return RiskAssessment(
            risk_level=risk_level,
            volatility_score=round(volatility_score, 3),
            var_95=round(var_95, 2),
            max_drawdown_expected=round(max_dd, 2),
            correlation_risk=round(correlation_risk, 3),
            liquidity_risk=round(liquidity_risk, 3),
            event_risk=round(event_risk, 3),
            overall_risk_score=round(overall_risk, 3),
            risk_factors=risk_factors
        )


class PositionSizer:
    """
    Calculador de tamaño de posición.

    Determina el tamaño óptimo de posición basado en el riesgo
    y las características de la oportunidad.
    """

    # Kelly Criterion modificado para mayor conservadurismo
    KELLY_FRACTION = 0.25  # Usar solo 25% del Kelly óptimo

    @classmethod
    def calculate_position(
        cls,
        expected_return: float,
        win_probability: float,
        risk_score: float,
        volatility: float
    ) -> PositionSizing:
        """
        Calcula el tamaño de posición recomendado.

        Args:
            expected_return: Retorno esperado (%)
            win_probability: Probabilidad de ganancia
            risk_score: Score de riesgo (0-1)
            volatility: Volatilidad (%)

        Returns:
            PositionSizing con recomendaciones
        """
        # Kelly Criterion: f* = (bp - q) / b
        # donde b = odds, p = prob win, q = prob loss
        if expected_return > 0:
            b = expected_return / volatility if volatility > 0 else 1
            p = win_probability
            q = 1 - p
            kelly = (b * p - q) / b if b > 0 else 0
            kelly = max(0, kelly) * cls.KELLY_FRACTION
        else:
            kelly = 0

        # Ajustar por riesgo
        risk_adjustment = 1 - risk_score
        base_allocation = kelly * risk_adjustment * 100  # Convertir a %

        # Límites
        suggested_allocation = min(max(base_allocation, 1), 10)  # 1-10%
        max_allocation = min(suggested_allocation * 1.5, 15)  # Max 15%

        # Stop loss y take profit basados en volatilidad
        stop_loss = min(max(volatility * 1.5, 2), 10)  # 2-10%
        take_profit = abs(expected_return) * 1.5 if expected_return > 0 else volatility * 2

        # Risk-reward ratio
        risk_reward = take_profit / stop_loss if stop_loss > 0 else 1

        # Estrategias de entrada/salida
        if risk_score < 0.3:
            entry_strategy = "Entrada completa - Riesgo bajo permite posición full"
            exit_strategy = "Trailing stop al alcanzar 50% del objetivo"
        elif risk_score < 0.6:
            entry_strategy = "Entrada escalonada - 50% inicial, 50% en confirmación"
            exit_strategy = "Stop loss estricto, toma parcial en +50% objetivo"
        else:
            entry_strategy = "Entrada mínima - Solo 30% inicialmente"
            exit_strategy = "Stop loss ajustado, salir ante primera señal adversa"

        return PositionSizing(
            suggested_allocation_pct=round(suggested_allocation, 2),
            max_allocation_pct=round(max_allocation, 2),
            entry_strategy=entry_strategy,
            exit_strategy=exit_strategy,
            stop_loss_pct=round(stop_loss, 2),
            take_profit_pct=round(take_profit, 2),
            risk_reward_ratio=round(risk_reward, 2)
        )


class RecommendationAgent:
    """
    Agente de Recomendación Profesional - Sistema de Decisión Integrado

    Implementa un sistema de recomendaciones de nivel institucional con:
    - Modelo de factores con 15+ variables
    - Gestión de riesgo integrada
    - Position sizing basado en Kelly Criterion
    - Explicabilidad completa de decisiones
    - Calibración de confianza

    El agente produce recomendaciones accionables con niveles de
    riesgo, tamaños de posición y estrategias de entrada/salida.
    """

    # Umbrales de decisión
    THRESHOLDS = {
        'strong_buy': 0.6,
        'buy': 0.3,
        'weak_buy': 0.1,
        'neutral_high': 0.1,
        'neutral_low': -0.1,
        'weak_sell': -0.1,
        'sell': -0.3,
        'strong_sell': -0.6
    }

    def __init__(
        self,
        umbral_compra: float = 2.0,
        umbral_venta: float = -2.0,
        peso_mercado: float = 0.35,
        peso_prediccion: float = 0.40,
        peso_sentimiento: float = 0.25
    ):
        """
        Inicializa el Agente de Recomendación Profesional.

        Args:
            umbral_compra: Variación % mínima para considerar compra
            umbral_venta: Variación % mínima para considerar venta
            peso_mercado: Peso de señales de mercado
            peso_prediccion: Peso de predicción del modelo
            peso_sentimiento: Peso del sentimiento
        """
        self.umbral_compra = umbral_compra
        self.umbral_venta = umbral_venta
        self.peso_mercado = peso_mercado
        self.peso_prediccion = peso_prediccion
        self.peso_sentimiento = peso_sentimiento

        logger.info(
            f"RecommendationAgent Profesional inicializado - "
            f"Pesos: Mercado={peso_mercado}, Pred={peso_prediccion}, Sent={peso_sentimiento}"
        )

    def generar_recomendacion(
        self,
        ticker: str,
        senal_mercado: str,
        variacion_pct: float,
        sentimiento: str,
        confianza_sentimiento: float = 0.5,
        # Parámetros avanzados opcionales
        volatilidad: float = 2.0,
        market_regime: str = "normal",
        rsi: float = 50.0,
        macd_signal: float = 0.0,
        volume_ratio: float = 1.0,
        prediction_confidence: float = 0.5,
        sentiment_score: float = 0.0,
        ultimo_precio: float = 0.0
    ) -> RecommendationResult:
        """
        Genera recomendación profesional integrando todas las señales.

        Flujo de decisión:
        1. Construir vector de factores
        2. Calcular score compuesto con modelo de factores
        3. Evaluar riesgo
        4. Calcular position sizing
        5. Determinar tipo de recomendación
        6. Generar explicación y catalystas

        Args:
            ticker: Símbolo del activo
            senal_mercado: "alcista", "bajista" o "neutral"
            variacion_pct: Cambio porcentual esperado
            sentimiento: "positivo", "negativo" o "neutral"
            confianza_sentimiento: Nivel de confianza del sentimiento
            volatilidad: Volatilidad histórica
            market_regime: Régimen de mercado
            rsi: Valor de RSI
            macd_signal: Señal MACD normalizada
            volume_ratio: Ratio de volumen vs promedio
            prediction_confidence: Confianza de la predicción
            sentiment_score: Score numérico de sentimiento
            ultimo_precio: Último precio conocido

        Returns:
            RecommendationResult completa con análisis de riesgo y posición
        """
        try:
            # Paso 1: Construir vector de factores
            factors = self._build_factor_vector(
                senal_mercado=senal_mercado,
                variacion_pct=variacion_pct,
                sentimiento=sentimiento,
                confianza_sentimiento=confianza_sentimiento,
                volatilidad=volatilidad,
                market_regime=market_regime,
                rsi=rsi,
                macd_signal=macd_signal,
                volume_ratio=volume_ratio,
                prediction_confidence=prediction_confidence,
                sentiment_score=sentiment_score
            )

            # Paso 2: Calcular score compuesto
            composite_score, contributions = FactorModel.calculate_composite_score(factors)

            # Paso 3: Evaluar riesgo
            risk_assessment = RiskManager.assess_risk(
                volatility=volatilidad,
                variacion_pct=variacion_pct,
                sentiment_confidence=confianza_sentimiento,
                market_regime=market_regime
            )

            # Paso 4: Calcular probabilidad de ganancia
            prob_profit = self._calculate_profit_probability(
                composite_score, prediction_confidence, risk_assessment.overall_risk_score
            )

            # Paso 5: Calcular position sizing
            position_sizing = PositionSizer.calculate_position(
                expected_return=variacion_pct,
                win_probability=prob_profit,
                risk_score=risk_assessment.overall_risk_score,
                volatility=volatilidad
            )

            # Paso 6: Determinar tipo de recomendación
            rec_type = self._determine_recommendation_type(composite_score)
            tipo_simple = self._type_to_simple(rec_type)

            # Paso 7: Generar texto de recomendación
            recomendacion_texto = self._generate_recommendation_text(
                rec_type, composite_score, risk_assessment.risk_level
            )

            # Paso 8: Calcular price target
            price_target = ultimo_precio * (1 + variacion_pct / 100) if ultimo_precio > 0 else 0

            # Paso 9: Generar explicación
            razon = self._generate_explanation(contributions[:5], risk_assessment)

            # Paso 10: Identificar catalizadores y riesgos
            catalysts, risks = self._identify_catalysts_and_risks(
                senal_mercado, sentimiento, variacion_pct, risk_assessment
            )

            # Paso 11: Calcular confianza calibrada
            confianza = self._calculate_calibrated_confidence(
                composite_score, contributions, risk_assessment
            )

            # Paso 12: Determinar horizonte temporal
            time_horizon = self._determine_time_horizon(volatilidad, variacion_pct)

            # Generar explicaciones para inversor minorista
            explicacion = self._generar_explicacion_simple(
                ticker=ticker,
                tipo=tipo_simple,
                variacion_pct=variacion_pct,
                risk_level=risk_assessment.risk_level.value,
                prob_profit=prob_profit,
                price_target=price_target,
                ultimo_precio=ultimo_precio,
                senal_mercado=senal_mercado,
                sentimiento=sentimiento,
                composite_score=composite_score
            )

            resultado = RecommendationResult(
                ticker=ticker,
                recomendacion=recomendacion_texto,
                tipo=tipo_simple,
                confianza=round(confianza, 3),
                razon=razon,
                factores={
                    "senal_mercado": senal_mercado,
                    "variacion_pct": round(variacion_pct, 2),
                    "sentimiento": sentimiento,
                    "composite_score": round(composite_score, 3),
                    "risk_level": risk_assessment.risk_level.value,
                    "prob_profit": round(prob_profit, 3)
                },
                fecha=datetime.now(),
                recommendation_type=rec_type,
                risk_assessment=risk_assessment,
                position_sizing=position_sizing,
                factor_contributions=contributions,
                time_horizon=time_horizon,
                price_target=round(price_target, 2),
                expected_return=round(variacion_pct, 2),
                probability_profit=round(prob_profit, 3),
                key_catalysts=catalysts,
                key_risks=risks,
                accion_sugerida=explicacion["accion_sugerida"],
                explicacion_simple=explicacion["explicacion_simple"],
                nivel_riesgo_simple=explicacion["nivel_riesgo_simple"],
                porque_esta_recomendacion=explicacion["porque_esta_recomendacion"],
                icono=explicacion["icono"]
            )

            logger.info(
                f"Recomendación para {ticker}: {rec_type.value} - "
                f"{recomendacion_texto} (confianza: {confianza:.2f})"
            )

            return resultado

        except Exception as e:
            logger.error(f"[{ticker}] Error generando recomendación: {str(e)}")
            return self._recomendacion_por_defecto(ticker)

    def _build_factor_vector(
        self,
        senal_mercado: str,
        variacion_pct: float,
        sentimiento: str,
        confianza_sentimiento: float,
        volatilidad: float,
        market_regime: str,
        rsi: float,
        macd_signal: float,
        volume_ratio: float,
        prediction_confidence: float,
        sentiment_score: float
    ) -> Dict[str, float]:
        """Construye el vector de factores para el modelo."""

        # Factor: Trend signal
        trend_map = {"alcista": 0.8, "neutral": 0.0, "bajista": -0.8}
        trend_signal = trend_map.get(senal_mercado.lower(), 0.0)

        # Factor: Momentum signal (basado en RSI)
        if rsi < 30:
            momentum_signal = 0.7  # Sobreventa = oportunidad
        elif rsi > 70:
            momentum_signal = -0.7  # Sobrecompra = cautela
        else:
            momentum_signal = (50 - rsi) / 50 * 0.5

        # Factor: Volatility signal (inverso - alta vol = cautela)
        volatility_signal = -min(volatilidad / 5, 1) + 0.5

        # Factor: Volume signal
        if volume_ratio > 1.5:
            volume_signal = 0.5 * np.sign(variacion_pct)  # Alto volumen confirma dirección
        elif volume_ratio < 0.5:
            volume_signal = -0.3  # Bajo volumen = cautela
        else:
            volume_signal = 0.0

        # Factor: Support/Resistance (simplificado)
        support_resistance = 0.0  # Placeholder

        # Factor: Model prediction
        model_prediction = np.tanh(variacion_pct / 5)

        # Signo de la predicción: los factores de confianza deben reforzar la dirección
        direction_sign = np.sign(variacion_pct) if variacion_pct != 0 else 1.0

        # Factor: Prediction confidence (direccional: positivo si sube, negativo si baja)
        pred_conf_factor = (prediction_confidence * 2 - 1) * direction_sign

        # Factor: Ensemble agreement (direccional)
        ensemble_agreement = prediction_confidence * 0.8 * direction_sign

        # Factor: Sentiment score
        sent_map = {"positivo": 0.7, "neutral": 0.0, "negativo": -0.7}
        sentiment_factor = sent_map.get(sentimiento.lower(), 0.0) * confianza_sentimiento

        # Factor: Sentiment trend (usar score directamente si disponible)
        sentiment_trend = sentiment_score if sentiment_score != 0 else sentiment_factor * 0.5

        # Factor: News impact (simplificado)
        news_impact = sentiment_factor * 0.5

        # Factor: Risk-adjusted return
        risk_adj = variacion_pct / (volatilidad + 0.5) if volatilidad > 0 else variacion_pct
        risk_adjusted_return = np.tanh(risk_adj / 2)

        # Factor: Market regime
        regime_map = {
            "tendencia_alcista": 0.5,
            "tendencia_bajista": -0.5,
            "alta_volatilidad": -0.3,
            "baja_volatilidad": 0.2,
            "lateral": 0.0,
            "normal": 0.0
        }
        market_regime_factor = regime_map.get(market_regime, 0.0)

        # Factor: Correlation
        correlation_factor = 0.0  # Placeholder

        return {
            'trend_signal': trend_signal,
            'momentum_signal': momentum_signal,
            'volatility_signal': volatility_signal,
            'volume_signal': volume_signal,
            'support_resistance': support_resistance,
            'model_prediction': model_prediction,
            'prediction_confidence': pred_conf_factor,
            'ensemble_agreement': ensemble_agreement,
            'sentiment_score': sentiment_factor,
            'sentiment_trend': sentiment_trend,
            'news_impact': news_impact,
            'risk_adjusted_return': risk_adjusted_return,
            'market_regime': market_regime_factor,
            'correlation_factor': correlation_factor
        }

    def _calculate_profit_probability(
        self,
        composite_score: float,
        prediction_confidence: float,
        risk_score: float
    ) -> float:
        """Calcula la probabilidad estimada de ganancia."""
        # Base: transformación del composite score
        base_prob = 0.5 + (composite_score * 0.3)

        # Ajuste por confianza de predicción
        confidence_adj = prediction_confidence * 0.1

        # Ajuste por riesgo (mayor riesgo = menor certeza)
        risk_adj = -risk_score * 0.1

        prob = base_prob + confidence_adj + risk_adj
        return np.clip(prob, 0.2, 0.8)

    def _determine_recommendation_type(self, score: float) -> RecommendationType:
        """Determina el tipo de recomendación basado en el score."""
        if score >= self.THRESHOLDS['strong_buy']:
            return RecommendationType.STRONG_BUY
        elif score >= self.THRESHOLDS['buy']:
            return RecommendationType.BUY
        elif score >= self.THRESHOLDS['weak_buy']:
            return RecommendationType.WEAK_BUY
        elif score >= self.THRESHOLDS['neutral_low']:
            return RecommendationType.HOLD
        elif score >= self.THRESHOLDS['sell']:
            return RecommendationType.WEAK_SELL
        elif score >= self.THRESHOLDS['strong_sell']:
            return RecommendationType.SELL
        else:
            return RecommendationType.STRONG_SELL

    def _type_to_simple(self, rec_type: RecommendationType) -> str:
        """Convierte tipo detallado a simple."""
        buy_types = [RecommendationType.STRONG_BUY, RecommendationType.BUY,
                    RecommendationType.WEAK_BUY]
        sell_types = [RecommendationType.STRONG_SELL, RecommendationType.SELL,
                     RecommendationType.WEAK_SELL]

        if rec_type in buy_types:
            return "compra"
        elif rec_type in sell_types:
            return "venta"
        else:
            return "mantener"

    def _generate_recommendation_text(
        self,
        rec_type: RecommendationType,
        score: float,
        risk_level: RiskLevel
    ) -> str:
        """Genera el texto de la recomendación."""
        texts = {
            RecommendationType.STRONG_BUY:
                "Compra fuerte recomendada - Múltiples factores alineados positivamente",
            RecommendationType.BUY:
                "Considerar posición de compra - Señales favorables predominan",
            RecommendationType.WEAK_BUY:
                "Oportunidad de compra moderada - Considerar entrada parcial",
            RecommendationType.HOLD:
                "Mantener posición actual - Señales mixtas, esperar confirmación",
            RecommendationType.WEAK_SELL:
                "Considerar reducción parcial - Señales de debilidad emergentes",
            RecommendationType.SELL:
                "Reducir exposición - Factores negativos predominan",
            RecommendationType.STRONG_SELL:
                "Venta recomendada - Múltiples señales de alerta activas"
        }

        base_text = texts.get(rec_type, "Mantener y monitorear")

        # Agregar nota de riesgo si aplica
        if risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
            base_text += " (NOTA: Riesgo elevado detectado)"

        return base_text

    def _generate_explanation(
        self,
        top_contributions: List[FactorContribution],
        risk_assessment: RiskAssessment
    ) -> str:
        """Genera explicación detallada de la recomendación."""
        parts = []

        # Explicar factores principales
        for contrib in top_contributions[:3]:
            factor_names = {
                'trend_signal': 'tendencia de mercado',
                'momentum_signal': 'momentum (RSI)',
                'model_prediction': 'predicción del modelo',
                'sentiment_score': 'sentimiento de mercado',
                'volatility_signal': 'nivel de volatilidad',
                'volume_signal': 'comportamiento del volumen'
            }
            name = factor_names.get(contrib.factor_name, contrib.factor_name)

            if contrib.direction == "bullish":
                parts.append(f"{name} positivo")
            elif contrib.direction == "bearish":
                parts.append(f"{name} negativo")

        # Agregar nivel de riesgo
        parts.append(f"riesgo {risk_assessment.risk_level.value}")

        return "Basado en: " + ", ".join(parts) + "."

    def _identify_catalysts_and_risks(
        self,
        senal_mercado: str,
        sentimiento: str,
        variacion_pct: float,
        risk_assessment: RiskAssessment
    ) -> Tuple[List[str], List[str]]:
        """Identifica catalizadores positivos y riesgos."""
        catalysts = []
        risks = []

        # Catalizadores
        if senal_mercado == "alcista":
            catalysts.append("Tendencia técnica alcista")
        if sentimiento == "positivo":
            catalysts.append("Sentimiento de mercado favorable")
        if variacion_pct > 3:
            catalysts.append("Predicción de apreciación significativa")

        # Riesgos
        if risk_assessment.volatility_score > 0.6:
            risks.append("Alta volatilidad")
        if risk_assessment.var_95 > 5:
            risks.append("Riesgo de pérdida elevado (VaR)")
        if senal_mercado == "bajista":
            risks.append("Tendencia técnica desfavorable")
        if sentimiento == "negativo":
            risks.append("Sentimiento negativo en el mercado")

        risks.extend(risk_assessment.risk_factors[:2])

        return catalysts, risks

    def _calculate_calibrated_confidence(
        self,
        composite_score: float,
        contributions: List[FactorContribution],
        risk_assessment: RiskAssessment
    ) -> float:
        """Calcula confianza calibrada de la recomendación."""
        # Base: magnitud del score
        score_confidence = abs(composite_score)

        # Factor de acuerdo entre contribuciones
        if contributions:
            directions = [c.direction for c in contributions[:5]]
            bullish = directions.count("bullish")
            bearish = directions.count("bearish")
            agreement = abs(bullish - bearish) / len(directions)
        else:
            agreement = 0

        # Penalización por riesgo
        risk_penalty = risk_assessment.overall_risk_score * 0.2

        confidence = (score_confidence * 0.4 + agreement * 0.4 + 0.2) - risk_penalty
        return np.clip(confidence, 0.3, 0.95)

    def _determine_time_horizon(
        self,
        volatility: float,
        variacion_pct: float
    ) -> TimeHorizon:
        """Determina el horizonte temporal apropiado."""
        if volatility > 4:
            return TimeHorizon.INTRADAY
        elif abs(variacion_pct) > 5:
            return TimeHorizon.SHORT_TERM
        elif abs(variacion_pct) > 2:
            return TimeHorizon.MEDIUM_TERM
        else:
            return TimeHorizon.LONG_TERM

    def _generar_explicacion_simple(
        self,
        ticker: str,
        tipo: str,
        variacion_pct: float,
        risk_level: str,
        prob_profit: float,
        price_target: float,
        ultimo_precio: float,
        senal_mercado: str = "neutral",
        sentimiento: str = "neutral",
        composite_score: float = 0.0
    ) -> Dict[str, str]:
        """
        Genera explicaciones claras para inversores minoristas.

        Returns:
            Dict con accion_sugerida, explicacion_simple, nivel_riesgo_simple,
            porque_esta_recomendacion, icono
        """
        # Determinar icono y acción según tipo
        if tipo in ["compra_fuerte", "compra"]:
            icono = "🟢"
            if tipo == "compra_fuerte":
                accion = f"COMPRAR {ticker}"
                explicacion = f"Este es un BUEN MOMENTO para considerar comprar {ticker}. "
            else:
                accion = f"Considerar comprar {ticker}"
                explicacion = f"Podría ser un momento favorable para comprar {ticker}. "
        elif tipo in ["venta_fuerte", "venta"]:
            icono = "🔴"
            if tipo == "venta_fuerte":
                accion = f"VENDER {ticker}"
                explicacion = f"Considera VENDER {ticker} pronto. "
            else:
                accion = f"Considerar vender {ticker}"
                explicacion = f"Podría ser momento de reducir tu posición en {ticker}. "
        else:  # mantener
            icono = "🟡"
            accion = f"MANTENER {ticker}"
            explicacion = f"Por ahora, mantén tu posición en {ticker} sin cambios. "

        # Agregar contexto de precio
        if variacion_pct > 0:
            explicacion += f"El modelo predice que el precio podría SUBIR un {abs(variacion_pct):.1f}% "
            explicacion += f"(de ${ultimo_precio:.2f} a ${price_target:.2f}). "
        elif variacion_pct < 0:
            explicacion += f"El modelo predice que el precio podría BAJAR un {abs(variacion_pct):.1f}% "
            explicacion += f"(de ${ultimo_precio:.2f} a ${price_target:.2f}). "
        else:
            explicacion += "El modelo predice que el precio se mantendrá estable. "

        # Probabilidad de ganancia
        prob_pct = prob_profit * 100
        if prob_pct >= 60:
            explicacion += f"Hay un {prob_pct:.0f}% de probabilidad de ganancia, lo cual es favorable."
        elif prob_pct >= 50:
            explicacion += f"Hay un {prob_pct:.0f}% de probabilidad de ganancia, lo cual es neutral."
        else:
            explicacion += f"Hay solo un {prob_pct:.0f}% de probabilidad de ganancia, lo cual es desfavorable."

        # Explicar nivel de riesgo de forma simple
        risk_explicaciones = {
            "muy_bajo": ("MUY BAJO", "Esta inversión tiene muy poco riesgo. Ideal si eres conservador."),
            "bajo": ("BAJO", "El riesgo es bajo. Adecuado para la mayoría de inversores."),
            "moderado": ("MODERADO", "Riesgo normal de mercado. Podrías ganar o perder."),
            "alto": ("ALTO", "Esta inversión es arriesgada. Solo invierte dinero que puedas perder."),
            "muy_alto": ("MUY ALTO", "Inversión muy arriesgada. Ten mucha precaución.")
        }

        nivel_simple, riesgo_explicacion = risk_explicaciones.get(
            risk_level, ("MODERADO", "Riesgo normal de mercado.")
        )
        nivel_riesgo_simple = f"Riesgo {nivel_simple}: {riesgo_explicacion}"

        # Agregar explicación de QUÉ HACER según el tipo de recomendación
        que_hacer_explicaciones = {
            "compra_fuerte": f"""
📋 ¿QUÉ HACER AHORA?
  • ✅ Si YA TIENES acciones de {ticker} → MANTENERLAS y considerar comprar más
  • ✅ Si NO TIENES acciones de {ticker} → Considerar comprar (buen momento)
  • 💰 Momento favorable para iniciar o aumentar posición
""",
            "compra": f"""
📋 ¿QUÉ HACER AHORA?
  • ✅ Si YA TIENES acciones de {ticker} → CONSERVARLAS
  • ✅ Si NO TIENES acciones de {ticker} → Considerar comprar si tienes confianza
  • 💡 Podría ser buen momento para entrar con cautela
""",
            "mantener": f"""
📋 ¿QUÉ HACER AHORA?
  • ✅ Si YA TIENES acciones de {ticker} → CONSERVARLAS (no vender)
  • ✅ Si NO TIENES acciones de {ticker} → NO comprar ahora, ESPERAR
  • ⏸️ Esperar señales más claras antes de actuar
""",
            "venta": f"""
📋 ¿QUÉ HACER AHORA?
  • ⚠️ Si YA TIENES acciones de {ticker} → Considerar VENDER o reducir posición
  • ✅ Si NO TIENES acciones de {ticker} → NO comprar
  • 🚫 No es buen momento para iniciar posiciones
""",
            "venta_fuerte": f"""
📋 ¿QUÉ HACER AHORA?
  • 🚨 Si YA TIENES acciones de {ticker} → VENDER o reducir significativamente
  • ✅ Si NO TIENES acciones de {ticker} → DEFINITIVAMENTE NO comprar
  • ⛔ Momento desfavorable, proteger capital
"""
        }

        que_hacer = que_hacer_explicaciones.get(tipo, que_hacer_explicaciones["mantener"])
        explicacion += f"\n{que_hacer}"

        # Explicar POR QUÉ se llegó a esta recomendación
        factores_a_favor = []
        factores_en_contra = []

        # Analizar señal de mercado (tendencia técnica)
        if senal_mercado == "alcista":
            factores_a_favor.append("📈 La TENDENCIA TÉCNICA es alcista (el precio está subiendo)")
        elif senal_mercado == "bajista":
            factores_en_contra.append("📉 La TENDENCIA TÉCNICA es bajista (el precio está bajando)")
        else:
            factores_a_favor.append("➡️ La tendencia técnica es neutral (precio estable)")

        # Analizar predicción del modelo
        if variacion_pct > 1:
            factores_a_favor.append(f"🤖 El MODELO DE IA predice subida de {variacion_pct:.1f}%")
        elif variacion_pct < -1:
            factores_en_contra.append(f"🤖 El MODELO DE IA predice bajada de {abs(variacion_pct):.1f}%")
        else:
            factores_a_favor.append("🤖 El modelo de IA predice precio estable")

        # Analizar sentimiento
        if sentimiento == "positivo":
            factores_a_favor.append("😊 El SENTIMIENTO del mercado es positivo (noticias favorables)")
        elif sentimiento == "negativo":
            factores_en_contra.append("😟 El SENTIMIENTO del mercado es negativo (noticias desfavorables)")
        else:
            factores_a_favor.append("😐 El sentimiento del mercado es neutral")

        # Analizar riesgo
        if risk_level in ["muy_bajo", "bajo"]:
            factores_a_favor.append(f"🛡️ El RIESGO es {risk_level.replace('_', ' ')} (inversión segura)")
        elif risk_level in ["alto", "muy_alto"]:
            factores_en_contra.append(f"⚠️ El RIESGO es {risk_level.replace('_', ' ')} (inversión arriesgada)")

        # Construir explicación de por qué
        porque = "📊 ¿CÓMO SE CALCULÓ ESTA RECOMENDACIÓN?\n\n"
        porque += "Analizamos 3 factores principales:\n"
        porque += "1️⃣ TENDENCIA TÉCNICA: ¿El precio está subiendo o bajando?\n"
        porque += "2️⃣ PREDICCIÓN DE IA: ¿Qué dice nuestro modelo de machine learning?\n"
        porque += "3️⃣ SENTIMIENTO: ¿Las noticias son positivas o negativas?\n\n"

        if factores_a_favor:
            porque += "✅ FACTORES A FAVOR:\n"
            for f in factores_a_favor:
                porque += f"   • {f}\n"

        if factores_en_contra:
            porque += "\n❌ FACTORES EN CONTRA:\n"
            for f in factores_en_contra:
                porque += f"   • {f}\n"

        # Score final
        porque += f"\n🎯 PUNTUACIÓN FINAL: {composite_score:.2f} "
        if composite_score > 0.3:
            porque += "(muy favorable para comprar)"
        elif composite_score > 0.1:
            porque += "(ligeramente favorable)"
        elif composite_score > -0.1:
            porque += "(neutral - sin señal clara)"
        elif composite_score > -0.3:
            porque += "(ligeramente desfavorable)"
        else:
            porque += "(desfavorable para comprar)"

        return {
            "accion_sugerida": accion,
            "explicacion_simple": explicacion,
            "nivel_riesgo_simple": nivel_riesgo_simple,
            "porque_esta_recomendacion": porque,
            "icono": icono
        }

    def _recomendacion_por_defecto(self, ticker: str) -> RecommendationResult:
        """Genera recomendación conservadora por defecto."""
        explicacion = self._generar_explicacion_simple(
            ticker, "mantener", 0.0, "moderado", 0.5, 0.0, 0.0
        )
        return RecommendationResult(
            ticker=ticker,
            recomendacion="Mantener posición y monitorear - Información insuficiente",
            tipo="mantener",
            confianza=0.3,
            razon="No se pudo completar el análisis. Se recomienda cautela.",
            factores={},
            fecha=datetime.now(),
            recommendation_type=RecommendationType.HOLD,
            risk_assessment=RiskAssessment(
                risk_level=RiskLevel.MODERATE,
                volatility_score=0.5,
                var_95=3.0,
                max_drawdown_expected=5.0,
                correlation_risk=0.3,
                liquidity_risk=0.2,
                event_risk=0.5,
                overall_risk_score=0.5,
                risk_factors=["Información insuficiente"]
            ),
            probability_profit=0.5,
            accion_sugerida=explicacion["accion_sugerida"],
            explicacion_simple=explicacion["explicacion_simple"],
            nivel_riesgo_simple=explicacion["nivel_riesgo_simple"],
            porque_esta_recomendacion=explicacion.get("porque_esta_recomendacion", "No hay datos suficientes para explicar la recomendación."),
            icono=explicacion["icono"]
        )

    def to_dict(self, resultado: RecommendationResult) -> Dict[str, Any]:
        """Convierte RecommendationResult a diccionario serializable."""
        return {
            "ticker": resultado.ticker,
            "recomendacion": resultado.recomendacion,
            "tipo": resultado.tipo,
            "confianza": resultado.confianza,
            "razon": resultado.razon,
            "factores": resultado.factores,
            "fecha": resultado.fecha.isoformat(),
            # Datos avanzados
            "recommendation_type": resultado.recommendation_type.value,
            "time_horizon": resultado.time_horizon.value,
            "price_target": resultado.price_target,
            "expected_return": resultado.expected_return,
            "probability_profit": resultado.probability_profit,
            "risk": {
                "level": resultado.risk_assessment.risk_level.value if resultado.risk_assessment else "moderate",
                "volatility_score": resultado.risk_assessment.volatility_score if resultado.risk_assessment else 0.5,
                "var_95": resultado.risk_assessment.var_95 if resultado.risk_assessment else 3.0,
                "factors": resultado.risk_assessment.risk_factors[:3] if resultado.risk_assessment else []
            },
            "position_sizing": {
                "suggested_allocation": resultado.position_sizing.suggested_allocation_pct if resultado.position_sizing else 5,
                "stop_loss": resultado.position_sizing.stop_loss_pct if resultado.position_sizing else 5,
                "take_profit": resultado.position_sizing.take_profit_pct if resultado.position_sizing else 10,
                "risk_reward": resultado.position_sizing.risk_reward_ratio if resultado.position_sizing else 2
            } if resultado.position_sizing else {},
            "key_catalysts": resultado.key_catalysts,
            "key_risks": resultado.key_risks,
            "top_factors": [
                {
                    "name": f.factor_name,
                    "direction": f.direction,
                    "contribution": round(f.contribution, 3)
                }
                for f in resultado.factor_contributions[:5]
            ]
        }
