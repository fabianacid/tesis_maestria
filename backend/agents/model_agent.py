"""
Agente de Modelo Profesional - Ensemble de ML para Clasificación de Dirección de Precios

Implementación de nivel institucional para clasificación binaria de movimientos de precios.
Utiliza un ensemble de modelos de machine learning con validación temporal rigurosa.

Modelos implementados (Clasificación Binaria):
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier (con objective='binary:logistic')
- LightGBM Classifier (con objective='binary')

Arquitectura:
- Feature engineering avanzado con 52 características técnicas
- Walk-forward validation temporal con 3 splits
- Ensemble con ponderación dinámica por performance (F1-Score)
- Predicción de dirección (SUBIDA/BAJADA) a 3 días
- Métricas de clasificación: Accuracy, Precision, Recall, F1-Score, AUC-ROC
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

# Opcional: Modelos avanzados
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Tipos de modelos disponibles."""
    LINEAR = "linear_regression"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    LSTM = "lstm"
    ENSEMBLE = "ensemble"


@dataclass
class ModelMetrics:
    """Métricas de evaluación para clasificación binaria de dirección de precios."""
    # Métricas de clasificación binaria (SUBIDA vs BAJADA)
    accuracy: float = 0.0        # Precisión general del clasificador
    precision: float = 0.0       # Precisión de clase positiva (SUBIDA)
    recall: float = 0.0          # Recall de clase positiva (SUBIDA)
    f1: float = 0.0              # F1-score (balance precision-recall)
    auc: float = 0.5             # Area Under ROC Curve (capacidad discriminativa)
    # Métricas adicionales
    direction_accuracy: float = 0.0  # Exactitud de dirección (porcentaje de aciertos)
    rmse: float = 0.0                # Proxy de error (1 - accuracy)
    mape: float = 0.0                # Proxy de error porcentual ((1 - accuracy) * 100)
    mae: float = 0.0                 # Proxy de error absoluto (1 - accuracy)
    r2: float = 0.0                  # Proxy de R² (accuracy como referencia)


@dataclass
class PredictionResult:
    """Estructura de resultado de predicción con análisis completo."""
    ticker: str
    modelo: str
    precio_predicho: float
    variacion_pct: float
    ultimo_precio: float
    rmse: float
    mape: float
    mae: float
    fecha_prediccion: datetime
    parametros: Dict[str, Any]
    # Datos avanzados
    intervalo_confianza: Tuple[float, float] = (0.0, 0.0)
    metricas_completas: ModelMetrics = field(default_factory=ModelMetrics)
    predicciones_modelos: Dict[str, float] = field(default_factory=dict)
    pesos_ensemble: Dict[str, float] = field(default_factory=dict)
    features_importance: Dict[str, float] = field(default_factory=dict)
    tendencia: str = "lateral"
    confianza: float = 0.5


# ============================================================
# LSTM Model (si PyTorch disponible)
# ============================================================
if TORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        """Red LSTM para predicción de series temporales."""

        def __init__(self, input_size: int = 1, hidden_size: int = 64,
                     num_layers: int = 2, dropout: float = 0.2):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )

            self.fc_layers = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            # x shape: (batch, seq_len, input_size)
            lstm_out, _ = self.lstm(x)
            # Usar solo la última salida
            last_output = lstm_out[:, -1, :]
            prediction = self.fc_layers(last_output)
            return prediction


class ModelAgent:
    """
    Agente de Modelo Profesional - Ensemble de ML

    Implementa predicción de series temporales financieras con:
    - Ensemble de 5+ modelos de ML
    - Feature engineering con 30+ indicadores técnicos
    - Walk-forward cross-validation
    - Selección automática del mejor modelo
    - Predicciones con intervalos de confianza
    - LSTM neural network (opcional)

    El agente combina múltiples modelos usando ponderación
    basada en performance histórica para maximizar precisión.
    """

    # Configuración de modelos
    MODEL_CONFIGS = {
        ModelType.RIDGE: {'alpha': 1.0},
        ModelType.LASSO: {'alpha': 0.1},
        ModelType.ELASTIC_NET: {'alpha': 0.5, 'l1_ratio': 0.5},
        ModelType.RANDOM_FOREST: {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        },
        ModelType.GRADIENT_BOOSTING: {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42
        },
    }

    XGBOOST_CONFIG = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': 0
    }

    LIGHTGBM_CONFIG = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': -1,
        'force_col_wise': True
    }

    def __init__(self, ventana_entrenamiento: int = 504):
        """
        Inicializa el Agente de Modelo Profesional.

        Args:
            ventana_entrenamiento: Períodos para entrenamiento (default: 252, 1 año)
        """
        self.ventana_entrenamiento = ventana_entrenamiento
        self.modelo_activo = "ensemble"
        self.scaler = RobustScaler()
        self.modelos_entrenados: Dict[str, Any] = {}
        self.pesos_modelos: Dict[str, float] = {}
        self.metricas_historicas: Dict[str, List[float]] = {}

        # Verificar disponibilidad de modelos
        self.modelos_disponibles = [
            ModelType.LINEAR, ModelType.RIDGE, ModelType.RANDOM_FOREST,
            ModelType.GRADIENT_BOOSTING
        ]

        if XGB_AVAILABLE:
            self.modelos_disponibles.append(ModelType.XGBOOST)
        if LGB_AVAILABLE:
            self.modelos_disponibles.append(ModelType.LIGHTGBM)
        if TORCH_AVAILABLE:
            self.modelos_disponibles.append(ModelType.LSTM)

        logger.info(
            f"ModelAgent Profesional inicializado - "
            f"Ventana: {ventana_entrenamiento}, "
            f"Modelos: {len(self.modelos_disponibles)}, "
            f"XGBoost: {XGB_AVAILABLE}, LightGBM: {LGB_AVAILABLE}, "
            f"LSTM: {TORCH_AVAILABLE}"
        )

    def predecir(
        self,
        precios: pd.DataFrame,
        ticker: str,
        modelo: str = "ensemble"
    ) -> Optional[PredictionResult]:
        """
        Genera predicción usando ensemble de modelos.

        Flujo de predicción:
        1. Validación de datos
        2. Feature engineering (30+ features)
        3. Entrenamiento de múltiples modelos
        4. Walk-forward validation
        5. Cálculo de pesos del ensemble
        6. Predicción con intervalo de confianza

        Args:
            precios: DataFrame con OHLCV
            ticker: Símbolo del activo
            modelo: Tipo de modelo ('ensemble', 'xgboost', etc.)

        Returns:
            PredictionResult con predicción y métricas completas
        """
        try:
            # Validación de datos
            if precios is None or 'Close' not in precios.columns:
                logger.error(f"[{ticker}] Datos de precios inválidos")
                return None

            if len(precios) < 30:
                logger.warning(f"[{ticker}] Datos insuficientes: {len(precios)}")
                return self._prediccion_fallback(precios, ticker)

            # Feature engineering
            X, y, feature_names = self._crear_features(precios)

            if X is None or len(X) < 20:
                logger.warning(f"[{ticker}] Features insuficientes")
                return self._prediccion_fallback(precios, ticker)

            ultimo_precio = float(precios['Close'].iloc[-1])

            # Escalar features
            X_scaled = self.scaler.fit_transform(X)

            # Entrenar y evaluar modelos
            predicciones = {}
            metricas_modelos = {}
            importancias = {}

            for model_type in self.modelos_disponibles:
                try:
                    pred, metrics, imp = self._entrenar_modelo(
                        X_scaled, y, model_type, feature_names
                    )
                    if pred is not None:
                        predicciones[model_type.value] = pred
                        metricas_modelos[model_type.value] = metrics
                        if imp:
                            importancias[model_type.value] = imp
                except Exception as e:
                    logger.debug(f"[{ticker}] Error en {model_type.value}: {e}")
                    continue

            if not predicciones:
                return self._prediccion_fallback(precios, ticker)

            # Calcular pesos del ensemble basado en métricas (usando accuracy)
            pesos = self._calcular_pesos_ensemble(metricas_modelos)

            # Predicción del ensemble: PROBABILIDAD de subida (promedio ponderado)
            prob_subida = sum(
                pred * pesos.get(name, 0)
                for name, pred in predicciones.items()
            )

            # Convertir probabilidad a predicción de precio
            # Estimar variación basada en volatilidad histórica
            volatilidad = precios['Close'].pct_change().std()
            if prob_subida > 0.5:
                # Predicción de subida
                variacion_esperada = volatilidad * (prob_subida - 0.5) * 2  # Escalar de 0-0.5 a 0-1
                precio_predicho = ultimo_precio * (1 + variacion_esperada)
            else:
                # Predicción de bajada
                variacion_esperada = volatilidad * (0.5 - prob_subida) * 2
                precio_predicho = ultimo_precio * (1 - variacion_esperada)

            # Calcular intervalo de confianza basado en volatilidad
            intervalo = (
                ultimo_precio * (1 - volatilidad * 1.96),
                ultimo_precio * (1 + volatilidad * 1.96)
            )

            # Métricas del mejor modelo (ahora basado en accuracy, no rmse)
            mejor_modelo = max(metricas_modelos.items(), key=lambda x: x[1].accuracy)
            mejor_metrics = mejor_modelo[1]

            # Calcular variación
            variacion_pct = ((precio_predicho - ultimo_precio) / ultimo_precio) * 100

            # Determinar tendencia basada en probabilidad
            if prob_subida > 0.55:
                tendencia = "alcista"
            elif prob_subida < 0.45:
                tendencia = "bajista"
            else:
                tendencia = "lateral"

            # Calcular confianza basada en accuracy y consenso
            confianza = (mejor_metrics.accuracy + abs(prob_subida - 0.5) * 2) / 2

            # Agregar feature importance promediada
            feature_importance_avg = self._promediar_importancias(importancias)

            # Crear resultado
            resultado = PredictionResult(
                ticker=ticker,
                modelo="ensemble" if modelo == "ensemble" else mejor_modelo[0],
                precio_predicho=round(precio_predicho, 4),
                variacion_pct=round(variacion_pct, 4),
                ultimo_precio=round(ultimo_precio, 4),
                rmse=round(mejor_metrics.rmse, 4),
                mape=round(mejor_metrics.mape, 4),
                mae=round(mejor_metrics.mae, 4),
                fecha_prediccion=datetime.now(),
                parametros={
                    "ventana": self.ventana_entrenamiento,
                    "n_features": len(feature_names),
                    "n_modelos": len(predicciones),
                    "mejor_modelo": mejor_modelo[0]
                },
                intervalo_confianza=(round(intervalo[0], 2), round(intervalo[1], 2)),
                metricas_completas=mejor_metrics,
                predicciones_modelos={k: round(v, 2) for k, v in predicciones.items()},
                pesos_ensemble={k: round(v, 4) for k, v in pesos.items()},
                features_importance=feature_importance_avg,
                tendencia=tendencia,
                confianza=round(confianza, 3)
            )

            logger.info(
                f"Predicción para {ticker}: {precio_predicho:.2f} "
                f"(Var: {variacion_pct:.2f}%, Accuracy: {mejor_metrics.accuracy:.2%}, "
                f"Prob subida: {prob_subida:.2%})"
            )

            return resultado

        except Exception as e:
            logger.error(f"[{ticker}] Error en predicción: {str(e)}")
            return self._prediccion_fallback(precios, ticker) if precios is not None else None

    def _crear_features(
        self,
        precios: pd.DataFrame
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        """
        Crea 30+ features técnicas para el modelo.

        Features incluidas:
        - Retornos (1, 2, 3, 5, 10, 20 días)
        - Medias móviles y cruces
        - Volatilidad (múltiples ventanas)
        - Momentum indicators
        - Price patterns
        - Volume features (si disponible)
        """
        df = precios.tail(self.ventana_entrenamiento + 50).copy()

        if len(df) < 30:
            return None, None, []

        close = df['Close']
        feature_names = []

        # ============ RETORNOS ============
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f'return_{lag}d'] = close.pct_change(lag)
            feature_names.append(f'return_{lag}d')

        # ============ MEDIAS MÓVILES ============
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = close.rolling(window).mean()
            df[f'sma_ratio_{window}'] = close / df[f'sma_{window}']
            feature_names.extend([f'sma_{window}', f'sma_ratio_{window}'])

        # EMAs
        for span in [12, 26]:
            df[f'ema_{span}'] = close.ewm(span=span).mean()
            df[f'ema_ratio_{span}'] = close / df[f'ema_{span}']
            feature_names.extend([f'ema_{span}', f'ema_ratio_{span}'])

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        feature_names.extend(['macd', 'macd_signal', 'macd_hist'])

        # ============ VOLATILIDAD ============
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = close.rolling(window).std()
            df[f'volatility_pct_{window}'] = df[f'volatility_{window}'] / close
            feature_names.extend([f'volatility_{window}', f'volatility_pct_{window}'])

        # ATR aproximado
        if 'High' in df.columns and 'Low' in df.columns:
            df['range'] = df['High'] - df['Low']
            df['atr_14'] = df['range'].rolling(14).mean()
            df['atr_pct'] = df['atr_14'] / close
            feature_names.extend(['range', 'atr_14', 'atr_pct'])

        # ============ MOMENTUM ============
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        feature_names.append('rsi')

        # ROC
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = close.pct_change(period) * 100
            feature_names.append(f'roc_{period}')

        # ============ BOLLINGER BANDS ============
        bb_sma = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df['bb_upper'] = bb_sma + 2 * bb_std
        df['bb_lower'] = bb_sma - 2 * bb_std
        df['bb_pct'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_sma
        feature_names.extend(['bb_pct', 'bb_width'])

        # ============ PRICE PATTERNS ============
        # Higher highs / Lower lows
        df['higher_high'] = (close > close.shift(1)).astype(int)
        df['lower_low'] = (close < close.shift(1)).astype(int)
        df['consecutive_up'] = df['higher_high'].rolling(5).sum()
        df['consecutive_down'] = df['lower_low'].rolling(5).sum()
        feature_names.extend(['higher_high', 'lower_low', 'consecutive_up', 'consecutive_down'])

        # Distance from high/low
        df['dist_from_high_20'] = (close - close.rolling(20).max()) / close
        df['dist_from_low_20'] = (close - close.rolling(20).min()) / close
        feature_names.extend(['dist_from_high_20', 'dist_from_low_20'])

        # ============ VOLUME FEATURES ============
        if 'Volume' in df.columns:
            volume = df['Volume']
            df['volume_sma_20'] = volume.rolling(20).mean()
            df['volume_ratio'] = volume / df['volume_sma_20']
            df['volume_change'] = volume.pct_change()
            feature_names.extend(['volume_sma_20', 'volume_ratio', 'volume_change'])

            # OBV approximation
            df['obv_change'] = np.where(close > close.shift(1), volume, -volume)
            df['obv_change'] = df['obv_change'].fillna(0)
            feature_names.append('obv_change')

        # ============ LAG FEATURES ============
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = close.shift(lag)
            feature_names.append(f'close_lag_{lag}')

        # ============ TEMPORAL FEATURES ============
        if isinstance(df.index, pd.DatetimeIndex):
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            feature_names.extend(['day_of_week', 'month'])

        # Limpiar NaN
        df = df.dropna()

        if len(df) < 20:
            return None, None, []

        # Usar solo la ventana de entrenamiento
        df = df.tail(self.ventana_entrenamiento)

        # Preparar X e y
        # Target: DIRECCIÓN del precio en 3 días (1=sube, 0=baja) - Clasificación
        future_price = df['Close'].shift(-3)
        current_price = df['Close']

        # Crear target binario: 1 si sube más de 0.5%, 0 si baja más de 0.5%
        # Ignorar movimientos menores (ruido de mercado)
        cambio_pct = (future_price - current_price) / current_price
        y_direction = (cambio_pct > 0.005).astype(int)

        # Eliminar NaN y alinear
        valid_idx = ~(future_price.isna() | current_price.isna())
        y = y_direction[valid_idx].values
        X = df[feature_names][valid_idx].iloc[:-3].values

        if len(X) != len(y):
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]

        return X, y, feature_names

    def _entrenar_modelo(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: ModelType,
        feature_names: List[str]
    ) -> Tuple[Optional[float], ModelMetrics, Dict[str, float]]:
        """
        Entrena un modelo específico con walk-forward validation.

        Returns:
            Tuple: (predicción, métricas, feature_importance)
        """
        if len(X) < 15:
            return None, ModelMetrics(), {}

        # Time series split para validación (3 splits para mejor precisión)
        tscv = TimeSeriesSplit(n_splits=5)
        metrics_list = []
        direction_correct = 0
        total_predictions = 0

        # Crear modelo según tipo
        modelo = self._crear_modelo(model_type)
        if modelo is None:
            return None, ModelMetrics(), {}

        # Walk-forward validation
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if len(X_train) < 10 or len(X_val) < 2:
                continue

            try:
                if model_type == ModelType.LSTM and TORCH_AVAILABLE:
                    y_pred = self._entrenar_lstm(X_train, y_train, X_val)
                else:
                    modelo.fit(X_train, y_train)
                    y_pred = modelo.predict(X_val)

                # Calcular métricas de CLASIFICACIÓN
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred, zero_division=0)
                recall = recall_score(y_val, y_pred, zero_division=0)
                f1 = f1_score(y_val, y_pred, zero_division=0)

                # Intentar calcular AUC si el modelo tiene predict_proba
                try:
                    if hasattr(modelo, 'predict_proba'):
                        y_proba = modelo.predict_proba(X_val)[:, 1]
                        auc = roc_auc_score(y_val, y_proba)
                    else:
                        auc = 0.5
                except:
                    auc = 0.5

                # Direction accuracy (ahora es lo mismo que accuracy)
                direction_correct += np.sum(y_val == y_pred)
                total_predictions += len(y_val)

                metrics_list.append({
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc
                })

            except Exception as e:
                logger.debug(f"Error en fold de {model_type.value}: {e}")
                continue

        if not metrics_list:
            return None, ModelMetrics(), {}

        # Entrenar modelo final con todos los datos y predecir PROBABILIDAD
        try:
            if model_type == ModelType.LSTM and TORCH_AVAILABLE:
                # Para LSTM, predecir directamente
                prediccion = self._entrenar_lstm(X[:-1], y[:-1], X[-1:])
                if isinstance(prediccion, np.ndarray):
                    prediccion = float(prediccion[0])
            else:
                modelo.fit(X, y)
                # Predecir PROBABILIDAD de subida (clase 1)
                if hasattr(modelo, 'predict_proba'):
                    prediccion = float(modelo.predict_proba(X[-1:].reshape(1, -1))[0][1])
                else:
                    # Si no tiene predict_proba, usar predicción de clase
                    prediccion = float(modelo.predict(X[-1:].reshape(1, -1))[0])
        except Exception as e:
            logger.debug(f"Error en predicción final de {model_type.value}: {e}")
            return None, ModelMetrics(), {}

        # Promediar métricas de CLASIFICACIÓN
        avg_metrics = ModelMetrics(
            accuracy=np.mean([m['accuracy'] for m in metrics_list]),
            precision=np.mean([m['precision'] for m in metrics_list]),
            recall=np.mean([m['recall'] for m in metrics_list]),
            f1=np.mean([m['f1'] for m in metrics_list]),
            auc=np.mean([m['auc'] for m in metrics_list]),
            direction_accuracy=direction_correct / total_predictions if total_predictions > 0 else 0.5,
            # Legacy para compatibilidad (usar accuracy como proxy de rmse/mape)
            rmse=1.0 - np.mean([m['accuracy'] for m in metrics_list]),
            mape=(1.0 - np.mean([m['accuracy'] for m in metrics_list])) * 100,
            mae=1.0 - np.mean([m['accuracy'] for m in metrics_list]),
            r2=np.mean([m['accuracy'] for m in metrics_list])
        )

        # Feature importance (si disponible)
        importance = {}
        if hasattr(modelo, 'feature_importances_'):
            for name, imp in zip(feature_names, modelo.feature_importances_):
                importance[name] = float(imp)
        elif hasattr(modelo, 'coef_'):
            for name, coef in zip(feature_names, np.abs(modelo.coef_)):
                importance[name] = float(coef)

        return prediccion, avg_metrics, importance

    def _crear_modelo(self, model_type: ModelType):
        """Crea instancia del modelo según el tipo (CLASIFICADORES)."""
        try:
            if model_type == ModelType.LINEAR:
                return LogisticRegression(max_iter=1000, random_state=42)
            elif model_type == ModelType.RIDGE:
                return RidgeClassifier(**self.MODEL_CONFIGS[ModelType.RIDGE])
            elif model_type == ModelType.LASSO:
                # Lasso no tiene versión clasificador, usar Logistic con L1
                return LogisticRegression(penalty='l1', solver='liblinear', C=2.0, random_state=42)
            elif model_type == ModelType.ELASTIC_NET:
                # ElasticNet no tiene versión clasificador, usar Logistic con elasticnet
                return LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=2.0, random_state=42, max_iter=1000)
            elif model_type == ModelType.RANDOM_FOREST:
                return RandomForestClassifier(**self.MODEL_CONFIGS[ModelType.RANDOM_FOREST])
            elif model_type == ModelType.GRADIENT_BOOSTING:
                return GradientBoostingClassifier(**self.MODEL_CONFIGS[ModelType.GRADIENT_BOOSTING])
            elif model_type == ModelType.XGBOOST and XGB_AVAILABLE:
                config = self.XGBOOST_CONFIG.copy()
                config['objective'] = 'binary:logistic'
                config['eval_metric'] = 'logloss'
                return xgb.XGBClassifier(**config)
            elif model_type == ModelType.LIGHTGBM and LGB_AVAILABLE:
                config = self.LIGHTGBM_CONFIG.copy()
                config['objective'] = 'binary'
                config['metric'] = 'binary_logloss'
                return lgb.LGBMClassifier(**config)
            elif model_type == ModelType.LSTM:
                return None  # LSTM se maneja aparte (clasificación binaria)
            else:
                return None
        except Exception as e:
            logger.debug(f"Error creando modelo {model_type.value}: {e}")
            return None

    def _entrenar_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        epochs: int = 50,
        seq_length: int = 10
    ) -> np.ndarray:
        """Entrena y predice con modelo LSTM."""
        if not TORCH_AVAILABLE:
            return np.array([y_train[-1]])

        try:
            # Preparar secuencias
            def create_sequences(X, y, seq_length):
                Xs, ys = [], []
                for i in range(len(X) - seq_length):
                    Xs.append(X[i:i+seq_length])
                    ys.append(y[i+seq_length])
                return np.array(Xs), np.array(ys)

            if len(X_train) < seq_length + 5:
                return np.array([y_train[-1]])

            X_seq, y_seq = create_sequences(X_train, y_train, seq_length)

            if len(X_seq) < 5:
                return np.array([y_train[-1]])

            # Convertir a tensores
            X_tensor = torch.FloatTensor(X_seq)
            y_tensor = torch.FloatTensor(y_seq).unsqueeze(1)

            # Crear modelo
            input_size = X_train.shape[1]
            model = LSTMModel(input_size=input_size, hidden_size=32, num_layers=1)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Entrenar
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()

            # Predecir
            model.eval()
            with torch.no_grad():
                # Usar las últimas seq_length observaciones
                last_seq = X_train[-seq_length:] if len(X_train) >= seq_length else X_train
                if len(last_seq) < seq_length:
                    # Padding
                    pad = np.zeros((seq_length - len(last_seq), X_train.shape[1]))
                    last_seq = np.vstack([pad, last_seq])

                X_pred = torch.FloatTensor(last_seq).unsqueeze(0)
                prediction = model(X_pred).numpy()[0, 0]

            return np.array([prediction])

        except Exception as e:
            logger.debug(f"Error en LSTM: {e}")
            return np.array([y_train[-1]])

    def _calcular_pesos_ensemble(
        self,
        metricas: Dict[str, ModelMetrics]
    ) -> Dict[str, float]:
        """
        Calcula pesos del ensemble basados en performance (CLASIFICACIÓN).

        Usa accuracy para ponderar modelos (mejor accuracy = mayor peso).
        """
        if not metricas:
            return {}

        # Usar accuracy directamente (mejor accuracy = mayor peso)
        accuracy_scores = {}
        for name, metrics in metricas.items():
            accuracy_scores[name] = max(metrics.accuracy, 0.01)  # Evitar división por cero

        # Normalizar pesos
        total = sum(accuracy_scores.values())
        if total > 0:
            return {name: val / total for name, val in accuracy_scores.items()}
        else:
            # Pesos iguales si no hay información
            n = len(metricas)
            return {name: 1.0 / n for name in metricas}

    def _determinar_tendencia(
        self,
        precios: pd.DataFrame,
        prediccion: float,
        ultimo_precio: float
    ) -> str:
        """Determina la tendencia basada en predicción y precios recientes."""
        var_pct = ((prediccion - ultimo_precio) / ultimo_precio) * 100

        # También considerar tendencia histórica
        if len(precios) >= 5:
            sma_5 = precios['Close'].tail(5).mean()
            sma_20 = precios['Close'].tail(20).mean() if len(precios) >= 20 else sma_5

            tendencia_historica = "alcista" if sma_5 > sma_20 else "bajista"
        else:
            tendencia_historica = "neutral"

        if var_pct > 1:
            return "alcista"
        elif var_pct < -1:
            return "bajista"
        else:
            return tendencia_historica if tendencia_historica != "neutral" else "lateral"

    def _calcular_confianza(
        self,
        metrics: ModelMetrics,
        std_predicciones: float,
        ultimo_precio: float
    ) -> float:
        """
        Calcula nivel de confianza de la predicción.

        Factores:
        - MAPE bajo = mayor confianza
        - Std de predicciones bajo = mayor confianza
        - Direction accuracy alto = mayor confianza
        """
        # Factor MAPE (normalizado, MAPE < 5% es bueno)
        mape_factor = max(0, 1 - (metrics.mape / 10))

        # Factor de consenso entre modelos
        cv = std_predicciones / ultimo_precio if ultimo_precio > 0 else 0
        consensus_factor = max(0, 1 - cv * 10)

        # Factor de dirección
        direction_factor = metrics.direction_accuracy

        # Promedio ponderado
        confianza = (
            0.4 * mape_factor +
            0.3 * consensus_factor +
            0.3 * direction_factor
        )

        return np.clip(confianza, 0, 1)

    def _promediar_importancias(
        self,
        importancias: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Promedia feature importance de todos los modelos."""
        if not importancias:
            return {}

        all_features = set()
        for imp in importancias.values():
            all_features.update(imp.keys())

        avg_importance = {}
        for feature in all_features:
            values = [
                imp.get(feature, 0)
                for imp in importancias.values()
            ]
            avg_importance[feature] = np.mean(values)

        # Normalizar y ordenar
        total = sum(avg_importance.values())
        if total > 0:
            avg_importance = {k: v / total for k, v in avg_importance.items()}

        # Top 10 features
        sorted_imp = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_imp[:10])

    def _prediccion_fallback(
        self,
        precios: pd.DataFrame,
        ticker: str
    ) -> Optional[PredictionResult]:
        """
        Predicción de respaldo usando regresión lineal simple.

        Se usa cuando no hay suficientes datos para el ensemble.
        """
        if precios is None or len(precios) < 5:
            return None

        try:
            close = precios['Close'].values
            n = len(close)

            X = np.arange(n).reshape(-1, 1)
            y = close

            modelo = LinearRegression()
            modelo.fit(X, y)

            prediccion = float(modelo.predict([[n]])[0])
            ultimo = float(close[-1])
            variacion = ((prediccion - ultimo) / ultimo) * 100

            return PredictionResult(
                ticker=ticker,
                modelo="linear_fallback",
                precio_predicho=round(prediccion, 4),
                variacion_pct=round(variacion, 4),
                ultimo_precio=round(ultimo, 4),
                rmse=0.0,
                mape=0.0,
                mae=0.0,
                fecha_prediccion=datetime.now(),
                parametros={"tipo": "fallback", "n_datos": n},
                tendencia="lateral",
                confianza=0.3
            )
        except Exception as e:
            logger.error(f"[{ticker}] Error en fallback: {e}")
            return None

    def obtener_tendencia(self, precios: pd.DataFrame) -> str:
        """Determina la tendencia general usando múltiples indicadores."""
        if precios is None or len(precios) < 10:
            return "lateral"

        close = precios['Close']

        # SMA cross
        sma_10 = close.tail(10).mean()
        sma_20 = close.tail(20).mean() if len(close) >= 20 else sma_10

        # Momentum
        momentum = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100

        # Regresión
        n = min(20, len(close))
        X = np.arange(n).reshape(-1, 1)
        y = close.tail(n).values

        modelo = LinearRegression()
        modelo.fit(X, y)
        pendiente = modelo.coef_[0] / np.mean(y) * 100

        # Combinar señales
        signals = []
        if sma_10 > sma_20:
            signals.append(1)
        else:
            signals.append(-1)

        if momentum > 2:
            signals.append(1)
        elif momentum < -2:
            signals.append(-1)
        else:
            signals.append(0)

        if pendiente > 0.5:
            signals.append(1)
        elif pendiente < -0.5:
            signals.append(-1)
        else:
            signals.append(0)

        avg_signal = np.mean(signals)

        if avg_signal > 0.3:
            return "alcista"
        elif avg_signal < -0.3:
            return "bajista"
        else:
            return "lateral"

    def to_dict(self, resultado: PredictionResult) -> Dict[str, Any]:
        """Convierte PredictionResult a diccionario serializable."""
        return {
            "ticker": resultado.ticker,
            "modelo": resultado.modelo,
            "precio_predicho": resultado.precio_predicho,
            "variacion_pct": resultado.variacion_pct,
            "ultimo_precio": resultado.ultimo_precio,
            "rmse": resultado.rmse,
            "mape": resultado.mape,
            "mae": resultado.mae,
            "fecha_prediccion": resultado.fecha_prediccion.isoformat(),
            "parametros": resultado.parametros,
            # Datos avanzados
            "intervalo_confianza": resultado.intervalo_confianza,
            "predicciones_modelos": resultado.predicciones_modelos,
            "pesos_ensemble": resultado.pesos_ensemble,
            "features_importance": resultado.features_importance,
            "tendencia": resultado.tendencia,
            "confianza": resultado.confianza,
            "metricas": {
                # Métricas de clasificación
                "accuracy": resultado.metricas_completas.accuracy,
                "precision": resultado.metricas_completas.precision,
                "recall": resultado.metricas_completas.recall,
                "f1": resultado.metricas_completas.f1,
                "auc": resultado.metricas_completas.auc,
                # Legacy para compatibilidad
                "rmse": resultado.metricas_completas.rmse,
                "mae": resultado.metricas_completas.mae,
                "mape": resultado.metricas_completas.mape,
                "r2": resultado.metricas_completas.r2,
                "direction_accuracy": resultado.metricas_completas.direction_accuracy
            }
        }
