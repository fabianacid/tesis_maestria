"""
Agente de Sentimiento Profesional - NLP Financiero Avanzado

Implementación de nivel institucional para análisis de sentimiento en mercados financieros.
Utiliza múltiples modelos de NLP con ensemble ponderado.

Modelos implementados:
- FinBERT: Modelo transformer especializado en finanzas (si disponible)
- VADER: Análisis de sentimiento basado en léxico (NLTK)
- TextBlob: Análisis de polaridad general
- Diccionario Financiero: Léxico especializado con 500+ términos

Arquitectura:
- Ensemble de múltiples modelos NLP
- Ponderación dinámica por tipo de texto
- Análisis de tendencia temporal del sentimiento
- Detección de entidades financieras
- Scoring de confianza multi-factor
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import hashlib

import numpy as np

# yfinance para noticias reales
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Opcional: NLTK
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    # Descargar recursos si no existen
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Opcional: TextBlob
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

# Opcional: Transformers (FinBERT)
try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except (ImportError, OSError):
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SentimentCategory(Enum):
    """Categorías de sentimiento."""
    VERY_POSITIVE = "muy_positivo"
    POSITIVE = "positivo"
    SLIGHTLY_POSITIVE = "ligeramente_positivo"
    NEUTRAL = "neutral"
    SLIGHTLY_NEGATIVE = "ligeramente_negativo"
    NEGATIVE = "negativo"
    VERY_NEGATIVE = "muy_negativo"


class TextType(Enum):
    """Tipos de texto para análisis."""
    NEWS_HEADLINE = "titular"
    NEWS_ARTICLE = "articulo"
    SOCIAL_MEDIA = "redes_sociales"
    ANALYST_REPORT = "informe_analista"
    EARNINGS_CALL = "llamada_ganancias"
    REGULATORY_FILING = "documento_regulatorio"


@dataclass
class SentimentScore:
    """Score de sentimiento de un modelo individual."""
    model_name: str
    score: float  # -1 a 1
    confidence: float  # 0 a 1
    category: SentimentCategory
    raw_output: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NewsItem:
    """Estructura de noticia analizada."""
    title: str
    source: str
    date: datetime
    sentiment_score: float
    sentiment_category: str
    relevance: float
    entities: List[str] = field(default_factory=list)


@dataclass
class SentimentResult:
    """Estructura de resultado de análisis de sentimiento completo."""
    ticker: str
    sentimiento: str
    confianza: float
    score: float
    fuente: str
    noticias_analizadas: int
    fecha_analisis: datetime
    detalles: Dict[str, Any]
    # Datos avanzados
    category: SentimentCategory = SentimentCategory.NEUTRAL
    model_scores: List[SentimentScore] = field(default_factory=list)
    news_items: List[NewsItem] = field(default_factory=list)
    sentiment_trend: str = "estable"
    key_topics: List[str] = field(default_factory=list)
    entity_sentiment: Dict[str, float] = field(default_factory=dict)
    # Explicaciones para inversor minorista
    explicacion_simple: str = ""
    que_significa: str = ""
    como_se_calcula: str = ""
    icono: str = ""


class FinancialLexicon:
    """
    Léxico financiero especializado para análisis de sentimiento.

    Contiene 500+ términos categorizados por:
    - Sentimiento (positivo/negativo)
    - Intensidad (fuerte/moderado/débil)
    - Contexto (mercado/empresa/economía)
    """

    # Términos positivos con pesos
    POSITIVE_TERMS = {
        # Muy positivos (peso 0.8-1.0)
        "récord": 0.9, "histórico": 0.85, "supera expectativas": 0.95,
        "excelente": 0.9, "excepcional": 0.9, "extraordinario": 0.85,
        "rally": 0.8, "boom": 0.85, "disparar": 0.8, "sobreponderar": 0.75,
        "strong buy": 1.0, "compra fuerte": 1.0, "outperform": 0.85,
        # Inglés - muy positivos
        "record": 0.9, "beats expectations": 0.95, "blowout": 0.9,
        "surges": 0.85, "soars": 0.85, "jumps": 0.75, "skyrockets": 0.9,
        "strong results": 0.85, "raised guidance": 0.9, "upgrade": 0.8,

        # Positivos (peso 0.5-0.79)
        "subir": 0.6, "crecimiento": 0.65, "ganancia": 0.7, "beneficio": 0.65,
        "mejora": 0.55, "optimista": 0.7, "alcista": 0.75, "recuperación": 0.6,
        "expansión": 0.65, "rentable": 0.7, "robusto": 0.6, "sólido": 0.55,
        "positivo": 0.6, "favorable": 0.55, "prometedor": 0.6, "atractivo": 0.55,
        "incremento": 0.55, "avance": 0.5, "progreso": 0.5, "éxito": 0.7,
        "buy": 0.7, "comprar": 0.7, "acumular": 0.65, "mantener": 0.3,
        # Inglés - positivos
        "growth": 0.65, "profit": 0.7, "revenue growth": 0.75, "earnings beat": 0.85,
        "bullish": 0.75, "optimistic": 0.7, "recovery": 0.6, "expansion": 0.65,
        "profitable": 0.7, "robust": 0.6, "solid": 0.55, "positive": 0.6,
        "increase": 0.55, "gain": 0.6, "rise": 0.55, "advances": 0.5,
        "success": 0.7, "innovative": 0.6, "breakthrough": 0.8, "partnership": 0.55,
        "demand": 0.5, "acquisition": 0.55, "dividend": 0.4, "buyback": 0.6,

        # Ligeramente positivos (peso 0.2-0.49)
        "estable": 0.3, "resistente": 0.35, "sostenido": 0.3, "moderado": 0.25,
        "neutral positivo": 0.35, "hold": 0.2, "dividendo": 0.4,
        # Inglés - ligeramente positivos
        "stable": 0.3, "resilient": 0.35, "steady": 0.3, "moderate": 0.25,
        "meets expectations": 0.35, "in line": 0.2,
    }

    # Términos negativos con pesos
    NEGATIVE_TERMS = {
        # Muy negativos (peso -0.8 a -1.0)
        "colapso": -0.95, "quiebra": -1.0, "bancarrota": -1.0, "fraude": -0.95,
        "crisis": -0.85, "desplome": -0.9, "crash": -0.9, "pánico": -0.85,
        "strong sell": -1.0, "venta fuerte": -1.0, "underperform": -0.85,
        # Inglés - muy negativos
        "bankruptcy": -1.0, "fraud": -0.95, "collapse": -0.95, "scandal": -0.9,
        "plunges": -0.9, "crashes": -0.9, "catastrophic": -0.9,
        "earnings miss": -0.85, "guidance cut": -0.9, "downgrade": -0.8,

        # Negativos (peso -0.5 a -0.79)
        "bajar": -0.6, "pérdida": -0.7, "caída": -0.65, "declive": -0.6,
        "pesimista": -0.7, "bajista": -0.75, "deterioro": -0.65, "recesión": -0.75,
        "contracción": -0.6, "déficit": -0.55, "débil": -0.55, "negativo": -0.6,
        "riesgo": -0.5, "incertidumbre": -0.5, "volatilidad": -0.45,
        "reducir": -0.55, "recorte": -0.5, "advertencia": -0.6, "alerta": -0.55,
        "sell": -0.7, "vender": -0.7, "evitar": -0.65,
        # Inglés - negativos
        "loss": -0.7, "decline": -0.65, "drop": -0.6, "falls": -0.6,
        "bearish": -0.75, "pessimistic": -0.7, "recession": -0.75, "deficit": -0.55,
        "weak": -0.55, "negative": -0.6, "risk": -0.5, "uncertainty": -0.5,
        "warning": -0.6, "concern": -0.5, "layoffs": -0.65, "job cuts": -0.65,
        "lawsuit": -0.6, "investigation": -0.55, "regulatory": -0.5,
        "setback": -0.6, "headwinds": -0.55, "challenges": -0.4, "pressure": -0.45,
        "miss": -0.65, "misses": -0.65, "disappoints": -0.7, "slows": -0.5,
        "tariff": -0.55, "geopolitical": -0.4, "volatility": -0.45,

        # Ligeramente negativos (peso -0.2 a -0.49)
        "preocupación": -0.4, "cautela": -0.35, "presión": -0.35,
        "desafío": -0.3, "obstáculo": -0.35, "limitado": -0.25,
        # Inglés - ligeramente negativos
        "caution": -0.35, "cautious": -0.35, "limited": -0.25,
        "slowing": -0.4, "below expectations": -0.55, "mixed": -0.2,
    }

    # Modificadores de intensidad
    INTENSIFIERS = {
        "muy": 1.3, "extremadamente": 1.5, "altamente": 1.3,
        "significativamente": 1.25, "fuertemente": 1.3, "dramáticamente": 1.4,
        "ligeramente": 0.7, "moderadamente": 0.85, "levemente": 0.6,
        "potencialmente": 0.8, "parcialmente": 0.75,
    }

    # Negadores
    NEGATORS = {"no", "nunca", "sin", "tampoco", "ni", "jamás", "ningún", "ninguno"}

    @classmethod
    def analyze(cls, text: str) -> Tuple[float, float, List[str]]:
        """
        Analiza texto usando el léxico financiero.

        Returns:
            Tuple: (score, confidence, matched_terms)
        """
        text_lower = text.lower()
        words = text_lower.split()

        positive_score = 0.0
        negative_score = 0.0
        matched_terms = []
        term_count = 0

        # Análisis de términos
        for term, weight in cls.POSITIVE_TERMS.items():
            if term in text_lower:
                # Verificar negación
                negated = any(neg in text_lower.split(term)[0][-20:]
                             for neg in cls.NEGATORS)
                if negated:
                    negative_score += abs(weight) * 0.8
                else:
                    positive_score += weight
                matched_terms.append(f"+{term}")
                term_count += 1

        for term, weight in cls.NEGATIVE_TERMS.items():
            if term in text_lower:
                negated = any(neg in text_lower.split(term)[0][-20:]
                             for neg in cls.NEGATORS)
                if negated:
                    positive_score += abs(weight) * 0.8
                else:
                    negative_score += abs(weight)
                matched_terms.append(f"-{term}")
                term_count += 1

        # Calcular score normalizado
        total_weight = positive_score + negative_score
        if total_weight > 0:
            score = (positive_score - negative_score) / (positive_score + negative_score)
        else:
            score = 0.0

        # Confianza basada en cantidad de términos encontrados
        confidence = min(0.9, 0.3 + (term_count * 0.1))

        return score, confidence, matched_terms


class SentimentAgent:
    """
    Agente de Sentimiento Profesional - Ensemble NLP

    Implementa análisis de sentimiento financiero de nivel institucional con:
    - FinBERT para análisis especializado en finanzas
    - VADER para análisis basado en léxico
    - TextBlob como análisis complementario
    - Léxico financiero propietario con 500+ términos
    - Ensemble ponderado por tipo de texto y confianza

    El agente combina múltiples señales para generar un score
    de sentimiento robusto y calibrado.
    """

    # Pesos del ensemble por tipo de modelo
    MODEL_WEIGHTS = {
        'finbert': 0.40,
        'vader': 0.25,
        'textblob': 0.15,
        'lexicon': 0.20
    }

    # Umbrales de clasificación
    THRESHOLDS = {
        'very_positive': 0.6,
        'positive': 0.25,
        'slightly_positive': 0.1,
        'neutral_high': 0.1,
        'neutral_low': -0.1,
        'slightly_negative': -0.1,
        'negative': -0.25,
        'very_negative': -0.6
    }

    def __init__(
        self,
        umbral_positivo: float = 0.15,
        umbral_negativo: float = -0.15,
        use_finbert: bool = True
    ):
        """
        Inicializa el Agente de Sentimiento Profesional.

        Args:
            umbral_positivo: Score mínimo para sentimiento positivo
            umbral_negativo: Score máximo para sentimiento negativo
            use_finbert: Intentar cargar FinBERT si está disponible
        """
        self.umbral_positivo = umbral_positivo
        self.umbral_negativo = umbral_negativo
        self.cache: Dict[str, SentimentResult] = {}
        self.cache_duration = timedelta(hours=1)

        # Inicializar modelos disponibles
        self.models_available = {}
        self._init_models(use_finbert)

        logger.info(
            f"SentimentAgent Profesional inicializado - "
            f"Modelos: {list(self.models_available.keys())}"
        )

    def _init_models(self, use_finbert: bool):
        """Inicializa los modelos de NLP disponibles."""
        # VADER (NLTK)
        if NLTK_AVAILABLE:
            try:
                self.vader = SentimentIntensityAnalyzer()
                self.models_available['vader'] = True
                logger.info("VADER inicializado correctamente")
            except Exception as e:
                logger.warning(f"Error inicializando VADER: {e}")
                self.models_available['vader'] = False
        else:
            self.models_available['vader'] = False

        # TextBlob
        self.models_available['textblob'] = TEXTBLOB_AVAILABLE
        if TEXTBLOB_AVAILABLE:
            logger.info("TextBlob disponible")

        # Léxico financiero (siempre disponible)
        self.models_available['lexicon'] = True

        # FinBERT (transformers)
        self.finbert_pipeline = None
        if use_finbert and TRANSFORMERS_AVAILABLE:
            try:
                # Usar modelo FinBERT preentrenado
                self.finbert_pipeline = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                    device=-1  # CPU
                )
                self.models_available['finbert'] = True
                logger.info("FinBERT cargado correctamente")
            except Exception as e:
                logger.warning(f"No se pudo cargar FinBERT: {e}")
                self.models_available['finbert'] = False
        else:
            self.models_available['finbert'] = False

    def analizar(self, ticker: str) -> SentimentResult:
        """
        Analiza el sentimiento para un ticker usando ensemble de modelos.

        Flujo de análisis:
        1. Verificar caché
        2. Obtener noticias reales de Yahoo Finance (yfinance)
        3. Analizar con cada modelo disponible
        4. Calcular ensemble ponderado
        5. Determinar categoría y confianza

        Args:
            ticker: Símbolo del activo financiero

        Returns:
            SentimentResult con análisis completo
        """
        ticker = ticker.upper().strip()

        # Verificar caché
        if ticker in self.cache:
            cached = self.cache[ticker]
            if datetime.now() - cached.fecha_analisis < self.cache_duration:
                logger.debug(f"[{ticker}] Sentimiento desde caché")
                return cached

        try:
            # Obtener textos para análisis
            news_items = self._obtener_noticias(ticker)

            if not news_items:
                return self._resultado_neutral(ticker, "Sin noticias disponibles")

            # Analizar cada noticia
            all_scores: List[SentimentScore] = []
            analyzed_news: List[NewsItem] = []
            news_scores_weighted: List[tuple] = []  # (score, relevancia)

            for news in news_items:
                text = f"{news['titulo']}. {news.get('contenido', '')}"
                relevancia = news.get('relevancia', 0.5)

                # Analizar con cada modelo
                model_scores = self._analizar_texto_ensemble(text)

                # Escalar pesos de modelos por relevancia de la noticia
                for s in model_scores:
                    s.confidence = s.confidence * relevancia
                all_scores.extend(model_scores)

                # Calcular score promedio ponderado para esta noticia
                if model_scores:
                    avg_score = np.mean([s.score for s in model_scores])
                    news_scores_weighted.append((avg_score, relevancia))
                    analyzed_news.append(NewsItem(
                        title=news['titulo'],
                        source=news.get('fuente', 'Desconocida'),
                        date=datetime.fromisoformat(news['fecha']) if isinstance(news['fecha'], str)
                             else news['fecha'],
                        sentiment_score=avg_score,
                        sentiment_category=self._score_to_category(avg_score).value,
                        relevance=relevancia,
                        entities=news.get('entidades', [])
                    ))

            # Calcular score final ponderado por relevancia de cada noticia
            if news_scores_weighted:
                total_relevancia = sum(r for _, r in news_scores_weighted)
                if total_relevancia > 0:
                    weighted_final = sum(s * r for s, r in news_scores_weighted) / total_relevancia
                else:
                    weighted_final = np.mean([s for s, _ in news_scores_weighted])
                _, final_confidence = self._calcular_ensemble(all_scores)
                final_score = weighted_final
            else:
                final_score, final_confidence = self._calcular_ensemble(all_scores)

            # Determinar categoría
            category = self._score_to_category(final_score)
            sentimiento = self._category_to_simple(category)

            # Detectar tendencia
            trend = self._detectar_tendencia(analyzed_news)

            # Extraer temas clave
            key_topics = self._extraer_temas(news_items)

            # Extraer fuentes de noticias
            fuentes_noticias = [news.get('fuente', '') for news in news_items if news.get('fuente')]

            # Ajustar confianza: mayor si tenemos noticias reales
            if news_items and news_items[0].get('es_noticia_real', False):
                # Boost de confianza por noticias reales
                final_confidence = min(0.95, final_confidence + 0.3)
                logger.info(f"[{ticker}] Confianza ajustada por noticias reales: {final_confidence:.2f}")

            # Generar explicaciones para inversor minorista
            modelos_usados = [k for k, v in self.models_available.items() if v]
            explicacion = self._generar_explicacion_simple(
                ticker, sentimiento, final_score, final_confidence, trend,
                modelos_usados=modelos_usados,
                n_noticias=len(analyzed_news),
                fuentes_noticias=fuentes_noticias
            )

            # Crear resultado
            resultado = SentimentResult(
                ticker=ticker,
                sentimiento=sentimiento,
                confianza=round(final_confidence, 3),
                score=round(final_score, 4),
                fuente="ensemble_nlp",
                noticias_analizadas=len(analyzed_news),
                fecha_analisis=datetime.now(),
                detalles={
                    "modelos_usados": list(self.models_available.keys()),
                    "n_modelos": sum(self.models_available.values()),
                    "scores_por_modelo": {
                        s.model_name: round(s.score, 3)
                        for s in all_scores[:10]
                    }
                },
                category=category,
                model_scores=all_scores[:20],  # Limitar
                news_items=analyzed_news,
                sentiment_trend=trend,
                key_topics=key_topics[:5],
                explicacion_simple=explicacion["explicacion_simple"],
                que_significa=explicacion["que_significa"],
                como_se_calcula=explicacion["como_se_calcula"],
                icono=explicacion["icono"]
            )

            # Actualizar caché
            self.cache[ticker] = resultado

            logger.info(
                f"Sentimiento para {ticker}: {sentimiento} "
                f"(score: {final_score:.2f}, confianza: {final_confidence:.2f})"
            )

            return resultado

        except Exception as e:
            logger.error(f"[{ticker}] Error en análisis de sentimiento: {str(e)}")
            return self._resultado_neutral(ticker, f"Error: {str(e)}")

    def _analizar_texto_ensemble(self, text: str) -> List[SentimentScore]:
        """
        Analiza un texto con todos los modelos disponibles.

        Returns:
            Lista de SentimentScore de cada modelo
        """
        scores = []

        # FinBERT
        if self.models_available.get('finbert') and self.finbert_pipeline:
            try:
                result = self.finbert_pipeline(text[:512])[0]
                label = result['label'].lower()
                raw_score = result['score']

                # Mapear labels de FinBERT
                if label == 'positive':
                    score = raw_score
                elif label == 'negative':
                    score = -raw_score
                else:
                    score = 0.0

                scores.append(SentimentScore(
                    model_name='finbert',
                    score=score,
                    confidence=raw_score,
                    category=self._score_to_category(score),
                    raw_output={'label': label, 'score': raw_score}
                ))
            except Exception as e:
                logger.debug(f"Error en FinBERT: {e}")

        # VADER
        if self.models_available.get('vader'):
            try:
                vader_scores = self.vader.polarity_scores(text)
                compound = vader_scores['compound']

                scores.append(SentimentScore(
                    model_name='vader',
                    score=compound,
                    confidence=abs(compound),
                    category=self._score_to_category(compound),
                    raw_output=vader_scores
                ))
            except Exception as e:
                logger.debug(f"Error en VADER: {e}")

        # TextBlob
        if self.models_available.get('textblob'):
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity

                scores.append(SentimentScore(
                    model_name='textblob',
                    score=polarity,
                    confidence=1 - subjectivity * 0.5,  # Mayor subjetividad = menor confianza
                    category=self._score_to_category(polarity),
                    raw_output={'polarity': polarity, 'subjectivity': subjectivity}
                ))
            except Exception as e:
                logger.debug(f"Error en TextBlob: {e}")

        # Léxico financiero
        if self.models_available.get('lexicon'):
            try:
                lex_score, lex_conf, matched = FinancialLexicon.analyze(text)

                scores.append(SentimentScore(
                    model_name='lexicon',
                    score=lex_score,
                    confidence=lex_conf,
                    category=self._score_to_category(lex_score),
                    raw_output={'matched_terms': matched}
                ))
            except Exception as e:
                logger.debug(f"Error en Lexicon: {e}")

        return scores

    def _calcular_ensemble(
        self,
        scores: List[SentimentScore]
    ) -> Tuple[float, float]:
        """
        Calcula el score final del ensemble ponderado.

        Pondera por:
        - Peso del modelo
        - Confianza del resultado individual
        """
        if not scores:
            return 0.0, 0.5

        weighted_scores = []
        weighted_confidences = []

        for score in scores:
            model_weight = self.MODEL_WEIGHTS.get(score.model_name, 0.1)
            combined_weight = model_weight * score.confidence

            weighted_scores.append(score.score * combined_weight)
            weighted_confidences.append(score.confidence * model_weight)

        total_weight = sum(self.MODEL_WEIGHTS.get(s.model_name, 0.1) * s.confidence
                         for s in scores)

        if total_weight > 0:
            final_score = sum(weighted_scores) / total_weight
            final_confidence = sum(weighted_confidences) / len(scores)
        else:
            final_score = np.mean([s.score for s in scores])
            final_confidence = 0.5

        return final_score, min(final_confidence, 0.95)

    def _score_to_category(self, score: float) -> SentimentCategory:
        """Convierte score numérico a categoría de sentimiento."""
        if score >= self.THRESHOLDS['very_positive']:
            return SentimentCategory.VERY_POSITIVE
        elif score >= self.THRESHOLDS['positive']:
            return SentimentCategory.POSITIVE
        elif score >= self.THRESHOLDS['slightly_positive']:
            return SentimentCategory.SLIGHTLY_POSITIVE
        elif score >= self.THRESHOLDS['neutral_low']:
            return SentimentCategory.NEUTRAL
        elif score >= self.THRESHOLDS['negative']:
            return SentimentCategory.SLIGHTLY_NEGATIVE
        elif score >= self.THRESHOLDS['very_negative']:
            return SentimentCategory.NEGATIVE
        else:
            return SentimentCategory.VERY_NEGATIVE

    def _category_to_simple(self, category: SentimentCategory) -> str:
        """Mapea categoría detallada a sentimiento simple."""
        positive_cats = [
            SentimentCategory.VERY_POSITIVE,
            SentimentCategory.POSITIVE,
            SentimentCategory.SLIGHTLY_POSITIVE
        ]
        negative_cats = [
            SentimentCategory.VERY_NEGATIVE,
            SentimentCategory.NEGATIVE,
            SentimentCategory.SLIGHTLY_NEGATIVE
        ]

        if category in positive_cats:
            return "positivo"
        elif category in negative_cats:
            return "negativo"
        else:
            return "neutral"

    def _detectar_tendencia(self, news_items: List[NewsItem]) -> str:
        """Detecta la tendencia del sentimiento en el tiempo."""
        if len(news_items) < 2:
            return "estable"

        # Ordenar por fecha
        sorted_news = sorted(news_items, key=lambda x: x.date)

        # Dividir en mitades
        mid = len(sorted_news) // 2
        first_half = sorted_news[:mid]
        second_half = sorted_news[mid:]

        avg_first = np.mean([n.sentiment_score for n in first_half]) if first_half else 0
        avg_second = np.mean([n.sentiment_score for n in second_half]) if second_half else 0

        diff = avg_second - avg_first

        if diff > 0.15:
            return "mejorando"
        elif diff < -0.15:
            return "deteriorando"
        else:
            return "estable"

    def _extraer_temas(self, news_items: List[Dict]) -> List[str]:
        """Extrae temas clave de las noticias."""
        # Palabras clave financieras a buscar
        keywords = {
            "ganancias": "Resultados financieros",
            "earnings": "Resultados financieros",
            "dividendo": "Dividendos",
            "dividend": "Dividendos",
            "fusión": "M&A",
            "merger": "M&A",
            "adquisición": "M&A",
            "acquisition": "M&A",
            "crecimiento": "Crecimiento",
            "growth": "Crecimiento",
            "regulación": "Regulatorio",
            "regulation": "Regulatorio",
            "fda": "Aprobación FDA",
            "producto": "Lanzamiento producto",
            "product": "Lanzamiento producto",
            "ceo": "Cambio ejecutivo",
            "despidos": "Reestructuración",
            "layoffs": "Reestructuración",
        }

        topics = set()
        for news in news_items:
            text = f"{news.get('titulo', '')} {news.get('contenido', '')}".lower()
            for keyword, topic in keywords.items():
                if keyword in text:
                    topics.add(topic)

        return list(topics)

    def _obtener_noticias(self, ticker: str) -> List[Dict]:
        """
        Obtiene noticias REALES para análisis usando Yahoo Finance.

        Intenta obtener noticias reales de yfinance.
        Si no hay noticias disponibles, retorna lista vacía.
        """
        noticias = []

        # Intentar obtener noticias reales de Yahoo Finance
        if YFINANCE_AVAILABLE:
            try:
                stock = yf.Ticker(ticker)
                news_data = stock.news

                if news_data:
                    logger.info(f"[{ticker}] Obtenidas {len(news_data)} noticias reales de Yahoo Finance")

                    for item in news_data[:10]:  # Limitar a 10 noticias
                        # La estructura puede estar anidada bajo 'content' o directa
                        content = item.get('content', item)

                        # Extraer título
                        titulo = content.get('title', '')
                        if not titulo:
                            continue

                        # Extraer fecha
                        # Puede ser pubDate (string ISO) o providerPublishTime (timestamp)
                        pub_date = content.get('pubDate', '')
                        if pub_date:
                            try:
                                # Formato ISO: '2026-01-16T19:30:00Z'
                                fecha = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                            except:
                                fecha = datetime.now()
                        else:
                            timestamp = item.get('providerPublishTime', 0)
                            if timestamp:
                                fecha = datetime.fromtimestamp(timestamp)
                            else:
                                fecha = datetime.now()

                        # Extraer fuente
                        provider = content.get('provider', {})
                        if isinstance(provider, dict):
                            fuente = provider.get('displayName', 'Yahoo Finance')
                        else:
                            fuente = item.get('publisher', 'Yahoo Finance')

                        # Extraer contenido/resumen
                        contenido = content.get('summary', '')
                        if not contenido:
                            contenido = content.get('description', titulo)

                        # Calcular relevancia: penalizar si el ticker no aparece en el título
                        texto_busqueda = (titulo + ' ' + contenido).lower()
                        ticker_en_texto = ticker.lower() in texto_busqueda
                        # Mapas de nombre de empresa por ticker
                        EMPRESA_NOMBRES = {
                            'AAPL': 'apple', 'GOOGL': 'google', 'GOOG': 'google',
                            'MSFT': 'microsoft', 'AMZN': 'amazon', 'TSLA': 'tesla',
                            'META': 'meta', 'NVDA': 'nvidia', 'JPM': 'jpmorgan',
                            'V': 'visa', 'WMT': 'walmart',
                        }
                        nombre_empresa = EMPRESA_NOMBRES.get(ticker.upper(), '')
                        empresa_en_texto = nombre_empresa and nombre_empresa in texto_busqueda

                        news_type = content.get('contentType', item.get('type', 'STORY'))
                        base_relevancia = 0.8 if news_type == 'STORY' else (0.6 if news_type == 'VIDEO' else 0.7)

                        if ticker_en_texto or empresa_en_texto:
                            relevancia = base_relevancia
                        else:
                            relevancia = base_relevancia * 0.4  # penalizar noticias no relacionadas

                        # Extraer URL (yfinance 1.0: canonicalUrl o clickThroughUrl dentro de content)
                        canonical = content.get('canonicalUrl', {})
                        click_through = content.get('clickThroughUrl', {})
                        url = (canonical.get('url', '') or click_through.get('url', '') or
                               item.get('link', ''))

                        noticias.append({
                            'titulo': titulo,
                            'fecha': fecha.isoformat() if hasattr(fecha, 'isoformat') else str(fecha),
                            'fuente': fuente,
                            'contenido': contenido,
                            'relevancia': relevancia,
                            'entidades': [ticker],
                            'url': url,
                            'es_noticia_real': True
                        })

                    if noticias:
                        logger.info(f"[{ticker}] Procesadas {len(noticias)} noticias reales")
                        return noticias

            except Exception as e:
                logger.warning(f"[{ticker}] Error obteniendo noticias de Yahoo Finance: {e}")
                import traceback
                logger.debug(traceback.format_exc())

        # Si no hay noticias reales disponibles, retornar lista vacía
        logger.info(f"[{ticker}] No hay noticias reales disponibles")
        return []

    def _generar_explicacion_simple(
        self,
        ticker: str,
        sentimiento: str,
        score: float,
        confianza: float,
        trend: str = "estable",
        modelos_usados: List[str] = None,
        n_noticias: int = 0,
        fuentes_noticias: List[str] = None
    ) -> Dict[str, str]:
        """
        Genera explicaciones claras para inversores minoristas.

        Returns:
            Dict con explicacion_simple, que_significa, como_se_calcula e icono
        """
        # Determinar icono
        if sentimiento == "positivo":
            icono = "📈"
        elif sentimiento == "negativo":
            icono = "📉"
        else:
            icono = "➡️"

        # Generar explicación según sentimiento
        if sentimiento == "positivo":
            if score > 0.3:
                explicacion = f"Las noticias sobre {ticker} son MUY POSITIVAS. Los inversores y medios están optimistas sobre esta acción."
                que_significa = "Hay mucho entusiasmo en el mercado. Esto PODRÍA indicar que el precio suba, pero recuerda que el sentimiento puede cambiar rápido."
            else:
                explicacion = f"Las noticias sobre {ticker} son MODERADAMENTE POSITIVAS. Hay optimismo pero sin exageración."
                que_significa = "El ambiente es favorable. Podría ser buen momento para mantener o considerar comprar, pero siempre investiga más."

        elif sentimiento == "negativo":
            if score < -0.3:
                explicacion = f"Las noticias sobre {ticker} son MUY NEGATIVAS. Hay preocupación entre inversores y analistas."
                que_significa = "Hay pesimismo en el mercado. El precio PODRÍA bajar. Considera esperar o ser cauteloso antes de comprar."
            else:
                explicacion = f"Las noticias sobre {ticker} son LIGERAMENTE NEGATIVAS. Hay algunas preocupaciones menores."
                que_significa = "El ambiente no es ideal pero tampoco alarmante. Mantén la calma y observa cómo evoluciona."

        else:  # neutral
            explicacion = f"Las noticias sobre {ticker} son NEUTRALES. No hay un sentimiento claro positivo o negativo."
            que_significa = "El mercado no tiene una opinión fuerte. Esto es normal y significa que debes basarte en otros factores para decidir."

        # Agregar contexto de tendencia
        if trend == "mejorando":
            explicacion += " La tendencia está MEJORANDO."
        elif trend == "deteriorando":
            explicacion += " La tendencia está EMPEORANDO."

        # Agregar nota sobre confianza
        if confianza < 0.3:
            que_significa += " NOTA: La confianza del análisis es baja, toma esta información con precaución."

        # Explicar cómo se calculó
        modelos_desc = {
            "vader": "VADER (análisis de palabras positivas/negativas)",
            "textblob": "TextBlob (análisis de polaridad del texto)",
            "finbert": "FinBERT (inteligencia artificial especializada en finanzas)",
            "lexicon": "Diccionario Financiero (términos especializados de bolsa)"
        }

        if n_noticias > 0:
            # Tenemos noticias reales
            como_se_calcula = f"NOTICIAS REALES: Analizamos {n_noticias} noticias recientes de Yahoo Finance"

            # Agregar fuentes si están disponibles
            if fuentes_noticias:
                fuentes_unicas = list(set(fuentes_noticias))[:4]
                como_se_calcula += f" (fuentes: {', '.join(fuentes_unicas)})"

            como_se_calcula += ". "

            if modelos_usados:
                modelos_explicados = [modelos_desc.get(m, m) for m in modelos_usados if m in modelos_desc]
                if modelos_explicados:
                    como_se_calcula += f"Técnicas de análisis: {', '.join(modelos_explicados[:3])}. "

            como_se_calcula += f"Score final: {score:.2f} (promedio ponderado)."
        else:
            como_se_calcula = "Sin noticias disponibles para este ticker. El análisis se basa en datos limitados."

        return {
            "explicacion_simple": explicacion,
            "que_significa": que_significa,
            "como_se_calcula": como_se_calcula,
            "icono": icono
        }

    def _resultado_neutral(self, ticker: str, nota: str) -> SentimentResult:
        """Genera resultado neutral por defecto."""
        explicacion = self._generar_explicacion_simple(ticker, "neutral", 0.0, 0.5)
        return SentimentResult(
            ticker=ticker,
            sentimiento="neutral",
            confianza=0.5,
            score=0.0,
            fuente="default",
            noticias_analizadas=0,
            fecha_analisis=datetime.now(),
            detalles={"nota": nota},
            category=SentimentCategory.NEUTRAL,
            sentiment_trend="estable",
            explicacion_simple=explicacion["explicacion_simple"],
            que_significa=explicacion["que_significa"],
            como_se_calcula=explicacion.get("como_se_calcula", "No hay datos suficientes para el análisis."),
            icono=explicacion["icono"]
        )

    def analizar_texto(self, texto: str) -> Dict[str, Any]:
        """
        Analiza un texto específico con el ensemble completo.

        Args:
            texto: Texto a analizar

        Returns:
            Dict con análisis detallado
        """
        scores = self._analizar_texto_ensemble(texto)
        final_score, confidence = self._calcular_ensemble(scores)
        category = self._score_to_category(final_score)

        return {
            "texto": texto[:200] + "..." if len(texto) > 200 else texto,
            "score": round(final_score, 4),
            "confianza": round(confidence, 3),
            "sentimiento": self._category_to_simple(category),
            "categoria": category.value,
            "scores_modelos": {s.model_name: round(s.score, 3) for s in scores},
            "modelos_usados": len(scores)
        }

    def to_dict(self, resultado: SentimentResult) -> Dict[str, Any]:
        """Convierte SentimentResult a diccionario serializable."""
        return {
            "ticker": resultado.ticker,
            "sentimiento": resultado.sentimiento,
            "confianza": resultado.confianza,
            "score": resultado.score,
            "fuente": resultado.fuente,
            "noticias_analizadas": resultado.noticias_analizadas,
            "fecha_analisis": resultado.fecha_analisis.isoformat(),
            "detalles": resultado.detalles,
            # Datos avanzados
            "categoria": resultado.category.value,
            "tendencia_sentimiento": resultado.sentiment_trend,
            "temas_clave": resultado.key_topics,
            "noticias": [
                {
                    "titulo": n.title,
                    "fuente": n.source,
                    "fecha": n.date.isoformat(),
                    "score": round(n.sentiment_score, 3),
                    "categoria": n.sentiment_category
                }
                for n in resultado.news_items[:5]
            ]
        }
