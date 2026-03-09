# ANEXO TÉCNICO: CÁLCULOS Y FÓRMULAS DEL SISTEMA

**Sistema Multiagente de Seguimiento Financiero**
**Análisis Matemático y Metodológico Completo**

---

## ÍNDICE

1. [Agente de Mercado (MarketAgent)](#1-agente-de-mercado-marketagent)
2. [Agente de Modelo Predictivo (ModelAgent)](#2-agente-de-modelo-predictivo-modelagent)
3. [Agente de Sentimiento (SentimentAgent)](#3-agente-de-sentimiento-sentimentagent)
4. [Agente de Recomendación (RecommendationAgent)](#4-agente-de-recomendación-recommendationagent)
5. [Agente de Alertas (AlertAgent)](#5-agente-de-alertas-alertagent)
6. [Flujo de Integración](#6-flujo-de-integración)

---

## 1. AGENTE DE MERCADO (MarketAgent)

### 1.1 Obtención de Datos

**Fuente**: Yahoo Finance API (yfinance)

**Datos descargados**:
```python
periodo = 6 meses
intervalo = 1 día
datos = {precio_apertura, precio_cierre, precio_máximo, precio_mínimo, volumen}
```

### 1.2 Indicadores Técnicos Implementados

El sistema calcula **más de 35 indicadores técnicos** organizados en 4 categorías:

#### 1.2.1 Indicadores de Tendencia

##### RSI (Relative Strength Index)

**Concepto**: Mide la velocidad y magnitud de cambios de precio para identificar condiciones de sobrecompra/sobreventa.

**Fórmula**:
```
RSI = 100 - (100 / (1 + RS))

donde:
RS = Ganancia Promedio / Pérdida Promedio

Ganancia Promedio = Σ(ganancias en período) / n
Pérdida Promedio = Σ(pérdidas en período) / n

período = 14 días (estándar)
```

**Cálculo paso a paso**:
1. Calcular cambios diarios: Δ = Precio_cierre(día_i) - Precio_cierre(día_i-1)
2. Separar ganancias (Δ > 0) y pérdidas (Δ < 0)
3. Calcular promedio móvil exponencial (EMA) de 14 días para ganancias y pérdidas
4. RS = EMA(ganancias) / EMA(pérdidas)
5. RSI = 100 - (100 / (1 + RS))

**Interpretación**:
- RSI > 70: Sobrecompra (posible corrección a la baja)
- RSI < 30: Sobreventa (posible rebote al alza)
- RSI ≈ 50: Neutral

**Ejemplo práctico**:
```
Ganancias últimos 14 días: [0.5, 1.2, 0.8, 0, 0, 0.3, 1.1, 0.9, 0.6, 0, 0.4, 0.7, 0.5, 0.8]
Pérdidas últimos 14 días:  [0, 0, 0, 0.4, 0.6, 0, 0, 0, 0, 0.5, 0, 0, 0, 0]

Ganancia_promedio = 0.56
Pérdida_promedio = 0.11

RS = 0.56 / 0.11 = 5.09
RSI = 100 - (100 / (1 + 5.09)) = 83.6

Resultado: RSI = 83.6 → SOBRECOMPRA
```

##### MACD (Moving Average Convergence Divergence)

**Concepto**: Indicador de momentum que muestra la relación entre dos medias móviles exponenciales.

**Fórmulas**:
```
MACD_line = EMA(12) - EMA(26)
Signal_line = EMA(9) del MACD_line
MACD_histogram = MACD_line - Signal_line

donde:
EMA(n) = Media Móvil Exponencial de n períodos
```

**Cálculo de EMA**:
```
EMA_hoy = (Precio_hoy × multiplicador) + (EMA_ayer × (1 - multiplicador))

multiplicador = 2 / (n + 1)

Para EMA(12): multiplicador = 2 / 13 = 0.1538
Para EMA(26): multiplicador = 2 / 27 = 0.0741
```

**Señales de trading**:
- MACD_line cruza por encima de Signal_line → Señal ALCISTA
- MACD_line cruza por debajo de Signal_line → Señal BAJISTA
- MACD_histogram > 0 → Momentum positivo
- MACD_histogram < 0 → Momentum negativo

##### Medias Móviles

**Fórmulas implementadas**:
```
SMA(n) = (P₁ + P₂ + ... + Pₙ) / n

EMA(n): Como se describió arriba
```

**Sistema implementa**:
- SMA(20), SMA(50), SMA(200)
- EMA(12), EMA(26), EMA(50)

##### ADX (Average Directional Index)

**Concepto**: Mide la fuerza de la tendencia (no su dirección).

**Fórmulas**:
```
+DM = Alto_hoy - Alto_ayer (si > 0, sino 0)
-DM = Bajo_ayer - Bajo_hoy (si > 0, sino 0)

+DI = (EMA(+DM, 14) / ATR) × 100
-DI = (EMA(-DM, 14) / ATR) × 100

DX = (|+DI - -DI| / |+DI + -DI|) × 100
ADX = EMA(DX, 14)
```

**Interpretación**:
- ADX > 25: Tendencia fuerte
- ADX < 20: Tendencia débil o mercado lateral
- ADX > 50: Tendencia muy fuerte

##### Indicadores Adicionales de Tendencia

El sistema también calcula:

- **Ichimoku Cloud**: Señales de Tenkan-sen y Kijun-sen
- **Parabolic SAR**: Detección de reversiones de tendencia
- **CCI** (Commodity Channel Index): Identificación de ciclos

#### 1.2.2 Indicadores de Momentum

##### Stochastic Oscillator

**Fórmulas**:
```
%K = (Precio_cierre - Mínimo_14días) / (Máximo_14días - Mínimo_14días) × 100
%D = SMA(%K, 3)
```

**Interpretación**:
- %K > 80: Sobrecompra
- %K < 20: Sobreventa

##### Otros Momentum

- **Williams %R**: Similar a Stochastic pero invertido
- **ROC** (Rate of Change): Velocidad del cambio de precio

#### 1.2.3 Indicadores de Volatilidad

##### Bandas de Bollinger

**Concepto**: Miden la volatilidad y niveles de sobrecompra/sobreventa basándose en desviación estándar.

**Fórmulas**:
```
Banda_Media = SMA(20)
Banda_Superior = SMA(20) + (2 × σ)
Banda_Inferior = SMA(20) - (2 × σ)
Banda_Width = Banda_Superior - Banda_Inferior
Bollinger_%B = (Precio - Banda_Inferior) / (Banda_Superior - Banda_Inferior)

donde:
σ = Desviación Estándar de 20 períodos
```

**Interpretación**:
- Precio cerca de Banda_Superior → Sobrecompra
- Precio cerca de Banda_Inferior → Sobreventa
- %B > 1: Por encima de banda superior
- %B < 0: Por debajo de banda inferior

##### ATR (Average True Range)

**Concepto**: Mide la volatilidad del mercado.

**Fórmula**:
```
True_Range = max(
    Alto - Bajo,
    |Alto - Cierre_anterior|,
    |Bajo - Cierre_anterior|
)

ATR = EMA(True_Range, 14)
ATR_% = (ATR / Precio_cierre) × 100
```

##### Keltner Channels

**Fórmulas**:
```
Keltner_Middle = EMA(20)
Keltner_Upper = EMA(20) + (2 × ATR)
Keltner_Lower = EMA(20) - (2 × ATR)
```

#### 1.2.4 Indicadores de Volumen

##### OBV (On-Balance Volume)

**Concepto**: Indicador de momentum que relaciona volumen con cambio de precio.

**Fórmula**:
```
Si Cierre_hoy > Cierre_ayer:
    OBV = OBV_ayer + Volumen_hoy

Si Cierre_hoy < Cierre_ayer:
    OBV = OBV_ayer - Volumen_hoy

Si Cierre_hoy = Cierre_ayer:
    OBV = OBV_ayer
```

**Tendencia del OBV**:
```
OBV_trend = "alcista" si OBV > EMA(OBV, 20)
          = "bajista" si OBV < EMA(OBV, 20)
```

##### VWAP (Volume Weighted Average Price)

**Fórmula**:
```
VWAP = Σ(Precio_típico × Volumen) / Σ(Volumen)

donde:
Precio_típico = (Alto + Bajo + Cierre) / 3
```

##### MFI (Money Flow Index)

**Concepto**: RSI ponderado por volumen, mide presión compradora/vendedora.

**Fórmulas**:
```
Precio_típico = (Alto + Bajo + Cierre) / 3
Flujo_monetario = Precio_típico × Volumen

Flujo_positivo = Σ(Flujo cuando Precio_típico ↑)
Flujo_negativo = Σ(Flujo cuando Precio_típico ↓)

Ratio_flujo = Flujo_positivo / Flujo_negativo

MFI = 100 - (100 / (1 + Ratio_flujo))
```

**Interpretación**:
- MFI > 80: Sobrecompra
- MFI < 20: Sobreventa

##### Indicadores Adicionales de Volumen

- **ADL** (Accumulation/Distribution Line)
- **CMF** (Chaikin Money Flow)

### 1.3 Detección de Régimen de Mercado

**Algoritmo**:
```python
if ADX > 25:
    if +DI > -DI:
        régimen = "tendencia_alcista"
    else:
        régimen = "tendencia_bajista"
elif ATR > percentil_75(ATR_histórico):
    régimen = "alta_volatilidad"
elif ATR < percentil_25(ATR_histórico):
    régimen = "baja_volatilidad"
else:
    régimen = "lateral"
```

### 1.4 Señal de Mercado Unificada

**Algoritmo de votación ponderada multi-factor**:

El sistema evalúa múltiples señales de diferentes categorías y las combina con pesos específicos:

```python
# SEÑALES TÉCNICAS
señales_técnicas = {
    'RSI': señal_rsi,           # -1, 0, +1
    'MACD': señal_macd,         # -1, 0, +1
    'Bollinger': señal_bb,      # -1, 0, +1
    'Stochastic': señal_stoch,  # -1, 0, +1
}

# SEÑALES DE TENDENCIA
señales_tendencia = {
    'SMA_cross': señal_sma,     # -1, 0, +1
    'EMA_cross': señal_ema,     # -1, 0, +1
    'ADX': señal_adx,           # -1, 0, +1
}

# SEÑALES DE VOLUMEN
señales_volumen = {
    'OBV': señal_obv,           # -1, 0, +1
    'MFI': señal_mfi,           # -1, 0, +1
}

# PESOS POR CATEGORÍA
pesos = {
    'técnicas': 0.40,
    'tendencia': 0.35,
    'volumen': 0.25
}

# CÁLCULO DEL SCORE FINAL
score_técnicas = promedio(señales_técnicas)
score_tendencia = promedio(señales_tendencia)
score_volumen = promedio(señales_volumen)

score_final = (score_técnicas × 0.40) +
              (score_tendencia × 0.35) +
              (score_volumen × 0.25)

# CLASIFICACIÓN
if score_final > 0.3:
    señal = "alcista"
elif score_final < -0.3:
    señal = "bajista"
else:
    señal = "neutral"
```

**Ejemplo completo**:
```
Señales técnicas:
- RSI = 65 → neutral (0)
- MACD > Signal → alcista (+1)
- Precio en medio de Bollinger → neutral (0)
- Stochastic = 75 → sobrecompra (-0.5)
Score técnicas = (0 + 1 + 0 - 0.5) / 4 = 0.125

Señales tendencia:
- SMA(20) > SMA(50) → alcista (+1)
- EMA(12) > EMA(26) → alcista (+1)
- ADX = 30, +DI > -DI → alcista (+1)
Score tendencia = (1 + 1 + 1) / 3 = 1.0

Señales volumen:
- OBV tendencia alcista → alcista (+1)
- MFI = 55 → neutral (0)
Score volumen = (1 + 0) / 2 = 0.5

Score final = (0.125 × 0.40) + (1.0 × 0.35) + (0.5 × 0.25)
            = 0.050 + 0.350 + 0.125 = 0.525

0.525 > 0.3 → Señal = "ALCISTA"
```

---

## 2. AGENTE DE MODELO PREDICTIVO (ModelAgent)

### 2.1 Arquitectura del Ensemble - Clasificación Binaria

**Objetivo**: Predecir la dirección del precio (SUBIDA o BAJADA) a 3 días, no el precio exacto.

**Modelos utilizados** (4 modelos de clasificación):
1. Random Forest Classifier (ensemble de árboles de decisión)
2. Gradient Boosting Classifier (boosting gradual para clasificación)
3. XGBoost Classifier (gradient boosting optimizado)
4. LightGBM Classifier (gradient boosting eficiente)

**Ventana de entrenamiento**: 504 días (2 años)
**Horizonte de predicción**: 3 días
**Validación**: Time Series Split con 5 folds

### 2.2 Features (Variables de Entrada)

**El sistema utiliza más de 50 features** organizadas en categorías:

#### 2.2.1 Features de Precio

```python
# Precios base
features_precio = [
    'Close',        # Precio de cierre
    'High',         # Precio máximo
    'Low',          # Precio mínimo
    'Open'          # Precio de apertura
]

# Lags de precio (valores pasados)
features_lags = [
    'Close_lag_1',  # Precio hace 1 día
    'Close_lag_2',  # Precio hace 2 días
    'Close_lag_3'   # Precio hace 3 días
]

# Retornos (cambios porcentuales)
features_retornos = [
    'Return_1d',    # Retorno 1 día
    'Return_5d',    # Retorno 5 días
    'Return_10d',   # Retorno 10 días
    'Return_20d'    # Retorno 20 días
]
```

#### 2.2.2 Features de Tendencia

```python
features_tendencia = [
    'SMA_20',       # Media móvil simple 20 días
    'SMA_50',       # Media móvil simple 50 días
    'EMA_12',       # Media móvil exponencial 12 días
    'EMA_26',       # Media móvil exponencial 26 días
    'MACD',         # MACD line
    'MACD_signal'   # MACD signal line
]
```

#### 2.2.3 Features de Momentum

```python
features_momentum = [
    'RSI',              # Relative Strength Index
    'Stochastic_K',     # Stochastic %K
    'Williams_R'        # Williams %R
]
```

#### 2.2.4 Features de Volatilidad

```python
features_volatilidad = [
    'ATR',              # Average True Range
    'Bollinger_upper',  # Banda de Bollinger superior
    'Bollinger_lower',  # Banda de Bollinger inferior
    'Bollinger_width',  # Ancho de Bollinger
    'Bollinger_pct',    # %B de Bollinger
    'Volatility_20d',   # Volatilidad histórica 20 días
    'Volatility_5d',    # Volatilidad histórica 5 días
    'Volatility_10d'    # Volatilidad histórica 10 días
]
```

#### 2.2.5 Features de Volumen

```python
features_volumen = [
    'Volume',           # Volumen
    'OBV',             # On-Balance Volume
    'Volume_ratio'     # Volumen vs promedio
]
```

#### 2.2.6 Features Temporales

```python
features_temporales = [
    'day_of_week',     # Día de la semana (0-6)
    'month',           # Mes del año (1-12)
    'quarter'          # Trimestre (1-4)
]
```

#### 2.2.7 Features de Patrones

```python
features_patrones = [
    'higher_high',     # ¿Máximo más alto que anterior?
    'lower_low',       # ¿Mínimo más bajo que anterior?
    'distance_from_high_20d',  # Distancia del máximo de 20 días
    'distance_from_low_20d'    # Distancia del mínimo de 20 días
]
```

**Total**: 50+ features

### 2.3 Target (Variable Objetivo) - Clasificación Binaria

**Fórmula del target**:
```
precio_futuro = Close[t+3]   # Precio dentro de 3 días
precio_actual = Close[t]      # Precio hoy

# Clasificación binaria
y = 1  si precio_futuro > precio_actual  (SUBIDA)
y = 0  si precio_futuro ≤ precio_actual  (BAJADA)
```

**Ejemplo práctico**:
```
Día t: Close = $150.00
Día t+3: Close = $152.50

precio_futuro ($152.50) > precio_actual ($150.00)
→ y = 1 (SUBIDA)
```

### 2.4 División de Datos

```
Total datos: 504 días de trading (2 años)

Validación: Time Series Split con 5 folds

Fold 1: Train[0:202]  → Test[202:252]  (202 días train, 50 días test)
Fold 2: Train[0:252]  → Test[252:302]  (252 días train, 50 días test)
Fold 3: Train[0:302]  → Test[302:352]  (302 días train, 50 días test)
Fold 4: Train[0:352]  → Test[352:402]  (352 días train, 50 días test)
Fold 5: Train[0:402]  → Test[402:452]  (402 días train, 52 días test)

Promedio final: métricas de los 5 folds
```

### 2.5 Configuración de Modelos de Clasificación

#### 2.5.1 Random Forest Classifier

```python
Hiperparámetros:
- n_estimators = 100
- max_depth = 10
- min_samples_split = 5
- min_samples_leaf = 2
- max_features = 'sqrt'
- random_state = 42
```

**Funcionamiento**:
- Crea 100 árboles de decisión
- Cada árbol vota por una clase (0 o 1)
- Predicción final = clase con más votos
- Output: probabilidad P(SUBIDA)

#### 2.5.2 XGBoost Classifier

```python
Hiperparámetros:
- n_estimators = 100
- max_depth = 6
- learning_rate = 0.1
- subsample = 0.8
- colsample_bytree = 0.8
- objective = 'binary:logistic'  # Clasificación binaria
- eval_metric = 'logloss'
- random_state = 42
```

**Algoritmo Gradient Boosting para Clasificación**:
```
F₀(x) = log(p/(1-p))  donde p = proporción de clase positiva

Para m = 1 hasta M:
    1. Calcular probabilidades: pᵢ = 1/(1 + e^(-Fₘ₋₁(xᵢ)))
    2. Calcular pseudo-residuales: rᵢ = yᵢ - pᵢ
    3. Entrenar árbol hₘ(x) para predecir residuales
    4. Actualizar: Fₘ(x) = Fₘ₋₁(x) + η × hₘ(x)

Predicción final: P(y=1|x) = 1/(1 + e^(-F_M(x)))
```

#### 2.5.3 LightGBM Classifier

```python
Hiperparámetros:
- n_estimators = 100
- max_depth = 6
- learning_rate = 0.1
- num_leaves = 31
- min_child_samples = 20
- objective = 'binary'
- metric = 'binary_logloss'
- random_state = 42
```

**Diferencia con XGBoost**: Construye árboles leaf-wise (más profundo) en vez de level-wise (más balanceado). Suele ser más rápido y eficiente en memoria.

#### 2.5.4 Gradient Boosting Classifier

```python
Hiperparámetros:
- n_estimators = 100
- max_depth = 5
- learning_rate = 0.1
- subsample = 0.8
- loss = 'log_loss'  # Para clasificación binaria
- min_samples_split = 5
- random_state = 42
```

### 2.6 Predicción del Ensemble

**Método: Promedio Ponderado por Accuracy**

```python
Paso 1: Entrenar cada modelo con validación cruzada
for modelo in [RF, GBM, XGB, LGBM]:
    for fold in time_series_cv(n_splits=3):
        modelo.fit(X_train_fold, y_train_fold)
        pred_proba = modelo.predict_proba(X_val_fold)[:, 1]
        pred_class = (pred_proba > 0.5).astype(int)
        calcular métricas (Accuracy, Precision, Recall, F1, AUC)

Paso 2: Calcular Accuracy promedio de cada modelo
accuracy_rf = promedio(accuracy_folds_rf)
accuracy_gbm = promedio(accuracy_folds_gbm)
accuracy_xgb = promedio(accuracy_folds_xgb)
accuracy_lgbm = promedio(accuracy_folds_lgbm)

Paso 3: Calcular pesos basados en Accuracy
peso_i = accuracy_i / Σ(accuracy_j)

Paso 4: Entrenar modelos con todos los datos
for modelo in [RF, GBM, XGB, LGBM]:
    modelo.fit(X_completo, y_completo)

Paso 5: Predicción final ponderada (probabilidades)
prob_rf = modelo_rf.predict_proba(X_nuevo)[:, 1]
prob_gbm = modelo_gbm.predict_proba(X_nuevo)[:, 1]
prob_xgb = modelo_xgb.predict_proba(X_nuevo)[:, 1]
prob_lgbm = modelo_lgbm.predict_proba(X_nuevo)[:, 1]

prob_ensemble = Σ(peso_i × prob_i)
```

**Ejemplo numérico**:
```
VALIDACIÓN CRUZADA:
Modelo           Fold1   Fold2   Fold3   Accuracy_avg
Random Forest    0.64    0.66    0.65    0.650
Gradient Boost   0.63    0.64    0.64    0.637
XGBoost          0.66    0.67    0.66    0.663
LightGBM         0.67    0.68    0.68    0.677  ← Mejor

CÁLCULO DE PESOS:
Modelo           Accuracy  Peso (normalizado)
Random Forest    0.650     0.2423 (24.23%)
Gradient Boost   0.637     0.2338 (23.38%)
XGBoost          0.663     0.2535 (25.35%)
LightGBM         0.677     0.2704 (27.04%) ← Mayor peso
                           ------
Total            2.627     1.0000

PREDICCIÓN PARA DÍA t+3:
Probabilidades de SUBIDA:
RF:    0.88  (88%)
GBM:   1.00  (100%)
XGB:   0.92  (92%)
LGBM:  0.97  (97%)  ← Mayor peso

Prob_ensemble = (0.2423×0.88) + (0.2338×1.00) + (0.2535×0.92) + (0.2704×0.97)
              = 0.2132 + 0.2338 + 0.2332 + 0.2623
              = 0.9425 = 94.25%

Interpretación: El ensemble predice SUBIDA con 94.25% de probabilidad
```

### 2.7 Conversión de Probabilidad a Precio

**Método basado en volatilidad histórica**:

```
Paso 1: Obtener probabilidad del ensemble
prob_subida = ensemble_predict_proba()

Paso 2: Calcular volatilidad histórica (20 días)
returns = (Close[t] - Close[t-1]) / Close[t-1]
volatilidad = std(returns_20d)

Paso 3: Estimar variación porcentual
Si prob_subida > 0.5:  # Predice SUBIDA
    variacion_pct = (prob_subida - 0.5) × 2 × volatilidad × 100
Sino:  # Predice BAJADA
    variacion_pct = (prob_subida - 0.5) × 2 × volatilidad × 100

Paso 4: Calcular precio predicho
precio_predicho = precio_actual × (1 + variacion_pct/100)
```

**Ejemplo con SPY**:
```
Precio actual: $682.15
Probabilidad ensemble: 0.9425 (94.25% SUBIDA)
Volatilidad histórica (20d): 0.015 (1.5%)

Paso 1: prob_subida = 0.9425 > 0.5 → Predice SUBIDA

Paso 2: variacion_pct = (0.9425 - 0.5) × 2 × 1.5
                       = 0.4425 × 3.0
                       = +1.328%

Paso 3: precio_predicho = 682.15 × (1 + 0.01328)
                        = 682.15 × 1.01328
                        = $691.21

Interpretación:
- El modelo predice SUBIDA con 94% de confianza
- Precio proyectado: $691.21 en 3 días
- Ganancia esperada: +$9.06 (+1.33%)
```

**Ejemplo con predicción de BAJADA**:
```
Precio actual: $150.00
Probabilidad ensemble: 0.30 (30% SUBIDA = 70% BAJADA)
Volatilidad histórica: 0.02 (2%)

Paso 1: prob_subida = 0.30 < 0.5 → Predice BAJADA

Paso 2: variacion_pct = (0.30 - 0.5) × 2 × 2.0
                       = -0.20 × 4.0
                       = -0.80%

Paso 3: precio_predicho = 150.00 × (1 - 0.008)
                        = 150.00 × 0.992
                        = $148.80

Interpretación:
- El modelo predice BAJADA con 70% de confianza
- Precio proyectado: $148.80 en 3 días
- Pérdida esperada: -$1.20 (-0.80%)
```

### 2.8 Métricas de Evaluación - Clasificación Binaria

#### 2.8.1 Matriz de Confusión

**Concepto**: Tabla que compara predicciones vs realidad.

```
                  Predicho
               BAJADA  SUBIDA
Real BAJADA      TN      FP
     SUBIDA      FN      TP

donde:
TP = True Positive (predijo SUBIDA, fue SUBIDA)
TN = True Negative (predijo BAJADA, fue BAJADA)
FP = False Positive (predijo SUBIDA, fue BAJADA)
FN = False Negative (predijo BAJADA, fue SUBIDA)
```

**Ejemplo**:
```
             Predicho
          BAJADA  SUBIDA
Real
BAJADA      40      10
SUBIDA      15      35

TP = 35, TN = 40, FP = 10, FN = 15
Total = 100 predicciones
```

#### 2.8.2 Accuracy (Exactitud)

**Fórmula**:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Ejemplo**:
```
Accuracy = (35 + 40) / (35 + 40 + 10 + 15)
         = 75 / 100
         = 0.75 = 75%

Interpretación: El modelo acierta la dirección en el 75% de los casos
```

**Interpretación**:
- Accuracy > 70%: Excelente para predicción de mercados
- Accuracy 60-70%: Bueno
- Accuracy 50-60%: Moderado
- Accuracy ≈ 50%: Equivalente a lanzar una moneda

#### 2.8.3 Precision (Precisión)

**Fórmula**:
```
Precision = TP / (TP + FP)
```

**Ejemplo**:
```
Precision = 35 / (35 + 10)
          = 35 / 45
          = 0.778 = 77.8%

Interpretación: Cuando predice SUBIDA, acierta el 77.8% de las veces
```

**¿Cuándo importa?**: Cuando el costo de un falso positivo es alto (comprar y que baje).

#### 2.8.4 Recall / Sensitivity (Sensibilidad)

**Fórmula**:
```
Recall = TP / (TP + FN)
```

**Ejemplo**:
```
Recall = 35 / (35 + 15)
       = 35 / 50
       = 0.70 = 70%

Interpretación: Detecta el 70% de las subidas reales
```

**¿Cuándo importa?**: Cuando no quieres perder oportunidades (capturar todas las subidas).

#### 2.8.5 F1-Score

**Fórmula**:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Ejemplo**:
```
F1 = 2 × (0.778 × 0.70) / (0.778 + 0.70)
   = 2 × 0.5446 / 1.478
   = 1.0892 / 1.478
   = 0.737 = 73.7%

Interpretación: Balance óptimo entre precision y recall
```

**¿Cuándo usarlo?**: Métrica general cuando precision y recall son igualmente importantes.

#### 2.8.6 AUC-ROC (Area Under Curve)

**Concepto**: Área bajo la curva ROC (True Positive Rate vs False Positive Rate).

**Fórmula**:
```
TPR = TP / (TP + FN)  (Recall)
FPR = FP / (FP + TN)

AUC ∈ [0, 1]
```

**Interpretación**:
- AUC = 1.0: Clasificador perfecto
- AUC = 0.9-1.0: Excelente
- AUC = 0.8-0.9: Muy bueno
- AUC = 0.7-0.8: Bueno
- AUC = 0.6-0.7: Moderado
- AUC = 0.5: Aleatorio (sin valor predictivo)
- AUC < 0.5: Peor que aleatorio

**Ejemplo con SPY**:
```
Resultados del modelo (promedio 10 tickers, sesión 08/03/2026, configuración final):
- Accuracy: 57.0%
- Precision: 60.7%
- Recall: 66.2%
- F1-Score: 58.9%
- AUC: 0.586

Interpretación:
El modelo supera el umbral aleatorio del 50% en los 10 tickers evaluados.
Cuando predice subida, acierta el 60.7% de las veces en promedio.

**Importancia**: En trading, predecir la dirección correcta puede ser más valioso que predecir el precio exacto.

---

## 3. AGENTE DE SENTIMIENTO (SentimentAgent)

### 3.1 Fuentes de Noticias

**API utilizada**: Yahoo Finance News
```
Endpoint: /v1/finance/search
Parámetros:
- q = ticker
- newsCount = 10
```

### 3.2 Métodos de Análisis Implementados

El sistema utiliza **4 métodos independientes** de análisis de sentimiento:

#### 3.2.1 VADER (Valence Aware Dictionary and sEntiment Reasoner)

**Características**:
- Diseñado específicamente para redes sociales
- Considera emoticonos, mayúsculas, puntuación
- Retorna 4 scores: positivo, negativo, neutral, compuesto

**Fórmula del Score Compuesto**:
```
compound = Normalizar(suma_valencias)

Normalización:
compound = valencia / √(valencia² + α)

donde α = 15 (constante de normalización)

Rango: [-1, +1]
```

**Interpretación**:
```
compound ≥ 0.05:  Sentimiento POSITIVO
compound ≤ -0.05: Sentimiento NEGATIVO
-0.05 < compound < 0.05: NEUTRAL
```

**Ejemplo**:
```
Noticia: "Tesla stock surges 10% on strong earnings!"

VADER analiza:
- "surges" → valencia positiva (+3.5)
- "strong" → valencia positiva (+2.5)
- "!" → intensificador (×1.2)

Valencia_total = (3.5 + 2.5) × 1.2 = 7.2
compound = 7.2 / √(7.2² + 15) = 7.2 / √66.84 = 7.2 / 8.17 = 0.88

Resultado: POSITIVO (score = 0.88)
```

#### 3.2.2 TextBlob

**Funcionamiento**:
- Análisis léxico basado en diccionario
- Retorna: polaridad [-1, +1] y subjetividad [0, 1]

**Fórmula**:
```
Polaridad = Σ(valencia_palabra_i) / n_palabras

Subjetividad = n_palabras_subjetivas / n_palabras_totales
```

**Ejemplo**:
```
Noticia: "Good performance but uncertain outlook"

Polaridades:
- "good" → +0.7
- "performance" → 0.0
- "but" → 0.0
- "uncertain" → -0.3
- "outlook" → 0.0

Polaridad = (0.7 + 0 + 0 - 0.3 + 0) / 5 = 0.4 / 5 = 0.08

Subjetividad:
Palabras subjetivas: "good", "uncertain" = 2
Total palabras: 5
Subjetividad = 2/5 = 0.4
```

#### 3.2.3 FinBERT (Transformer para finanzas)

**Arquitectura**: BERT fine-tuned en textos financieros

**Proceso**:
1. Tokenización del texto
2. Encoding con BERT (768 dimensiones)
3. Clasificación en 3 clases: {positivo, negativo, neutral}

**Output**:
```
Probabilidades:
P(positivo) = 0.75
P(negativo) = 0.10
P(neutral) = 0.15

Sentimiento = argmax(P) = "positivo"
Score = P(positivo) - P(negativo) = 0.75 - 0.10 = 0.65
```

#### 3.2.4 Financial Lexicon (Léxico Financiero)

**Concepto**: Diccionario especializado de términos financieros con valencias.

**Método**:
```
Para cada palabra en la noticia:
    Si palabra en lexicon_positivo:
        score += valencia_positiva
    Si palabra en lexicon_negativo:
        score -= valencia_negativa

score_final = score / total_palabras
```

**Ejemplos de términos**:
```
Positivos: "profit", "growth", "surge", "beat", "strong"
Negativos: "loss", "decline", "miss", "weak", "concern"
```

### 3.3 Ensemble de Sentimiento

**IMPORTANTE**: Los pesos implementados son:

```python
pesos = {
    'finbert': 0.40,      # 40% - Transformer especializado
    'vader': 0.25,        # 25% - Redes sociales
    'lexicon': 0.20,      # 20% - Léxico financiero
    'textblob': 0.15      # 15% - Análisis léxico general
}

score_final = (score_finbert × 0.40) +
              (score_vader × 0.25) +
              (score_lexicon × 0.20) +
              (score_textblob × 0.15)
```

**Razón de los pesos**:
- FinBERT tiene mayor peso porque está especializado en finanzas
- VADER segundo porque maneja bien intensificadores
- Lexicon tercero porque usa términos financieros específicos
- TextBlob menor peso porque es más general

**Ejemplo completo**:
```
Para 10 noticias de AAPL:

Noticia 1: "Apple beats earnings expectations"
- VADER: 0.72
- TextBlob: 0.65
- FinBERT: 0.85
- Lexicon: 0.75

Score_1 = (0.85×0.40) + (0.72×0.25) + (0.75×0.20) + (0.65×0.15)
        = 0.340 + 0.180 + 0.150 + 0.098
        = 0.768

Noticia 2: "iPhone sales decline in China"
- VADER: -0.45
- TextBlob: -0.30
- FinBERT: -0.60
- Lexicon: -0.50

Score_2 = (-0.60×0.40) + (-0.45×0.25) + (-0.50×0.20) + (-0.30×0.15)
        = -0.240 - 0.113 - 0.100 - 0.045
        = -0.498

... (continuar con las 10 noticias)

Scores: [0.768, -0.498, 0.320, 0.150, -0.200, 0.410, 0.550, -0.100, 0.280, 0.350]

Score_promedio = Σ scores / 10 = 2.030 / 10 = 0.203
```

### 3.4 Clasificación de Sentimiento

```python
if score_final > 0.05:
    sentimiento = "positivo"
elif score_final < -0.05:
    sentimiento = "negativo"
else:
    sentimiento = "neutral"
```

**Umbral de ±0.05**: Evita clasificar como positivo/negativo sentimientos muy débiles.

### 3.5 Cálculo de Confianza

**Fórmula implementada**:

```python
# Base de confianza
base = 0.5

# Ajuste por número de noticias (máx +0.2)
ajuste_cantidad = min(n_noticias / 10, 1.0) × 0.2

# Ajuste por acuerdo entre métodos (máx +0.3)
scores_métodos = [score_finbert, score_vader, score_lexicon, score_textblob]
desviación = std(scores_métodos)
acuerdo = 1 - min(desviación, 1.0)
ajuste_acuerdo = acuerdo × 0.3

confianza = base + ajuste_cantidad + ajuste_acuerdo

# Limitar rango
confianza_final ∈ [0.3, 0.9]
```

**Ejemplo**:
```
n_noticias = 10
Scores para una noticia:
- FinBERT: 0.85
- VADER: 0.72
- Lexicon: 0.75
- TextBlob: 0.65

Desviación = std([0.85, 0.72, 0.75, 0.65]) = 0.08

Base = 0.5
Ajuste_cantidad = min(10/10, 1.0) × 0.2 = 0.2
Acuerdo = 1 - 0.08 = 0.92
Ajuste_acuerdo = 0.92 × 0.3 = 0.276

Confianza = 0.5 + 0.2 + 0.276 = 0.976

Aplicar límite: min(0.976, 0.9) = 0.9

Confianza_final = 90%
```

Si solo hay 3 noticias:
```
Ajuste_cantidad = min(3/10, 1.0) × 0.2 = 0.06
Confianza = 0.5 + 0.06 + 0.276 = 0.836

Confianza_final = 83.6%  # Valor ilustrativo del cálculo; las confianzas reales varían según ticker y sesión
```

---

## 4. AGENTE DE RECOMENDACIÓN (RecommendationAgent)

### 4.1 Modelo de Factores Multi-Señal

#### 4.1.1 Los 15 Factores del Modelo

**Tabla de factores y pesos**:

| Factor | Peso | Categoría | Descripción |
|--------|------|-----------|-------------|
| **trend_signal** | 0.12 | Técnico | Tendencia alcista/bajista/neutral |
| **momentum_signal** | 0.10 | Técnico | RSI (sobreventa/sobrecompra) |
| **volatility_signal** | 0.08 | Técnico | Nivel de volatilidad |
| **volume_signal** | 0.06 | Técnico | Volumen vs promedio |
| **support_resistance** | 0.04 | Técnico | Niveles de soporte/resistencia* |
| **model_prediction** | 0.15 | Predicción | Predicción del modelo IA |
| **prediction_confidence** | 0.10 | Predicción | Confianza del modelo |
| **ensemble_agreement** | 0.10 | Predicción | Acuerdo entre modelos |
| **sentiment_score** | 0.08 | Sentimiento | Sentimiento positivo/negativo |
| **sentiment_trend** | 0.04 | Sentimiento | Tendencia del sentimiento |
| **news_impact** | 0.03 | Sentimiento | Impacto de noticias |
| **risk_adjusted_return** | 0.05 | Riesgo | Retorno ajustado por riesgo |
| **market_regime** | 0.03 | Riesgo | Régimen de mercado actual |
| **correlation_factor** | 0.02 | Riesgo | Correlación con mercado* |

**Total peso**: 1.00 (100%)

*Nota: support_resistance y correlation_factor están implementados como placeholders (valor 0.0).

#### 4.1.2 Cálculo de Cada Factor

**1. Trend Signal** (peso: 0.12):
```
trend = {
    "alcista": +0.8,
    "neutral": 0.0,
    "bajista": -0.8
}
```

**2. Momentum Signal** (peso: 0.10):
```
if RSI < 30:
    momentum = +0.7    # Sobreventa = oportunidad de compra
elif RSI > 70:
    momentum = -0.7    # Sobrecompra = cautela
else:
    momentum = (50 - RSI) / 50 × 0.5
```

Ejemplo:
```
RSI = 35 → momentum = (50-35)/50 × 0.5 = 0.15
RSI = 65 → momentum = (50-65)/50 × 0.5 = -0.15
```

**3. Volatility Signal** (peso: 0.08):
```
volatility_signal = -min(ATR / 5, 1.0) + 0.5

# Alta volatilidad genera cautela (señal negativa)
```

Ejemplo:
```
ATR = 2.0 → signal = -min(2/5, 1) + 0.5 = -0.4 + 0.5 = 0.1
ATR = 4.0 → signal = -min(4/5, 1) + 0.5 = -0.8 + 0.5 = -0.3
ATR = 6.0 → signal = -min(6/5, 1) + 0.5 = -1.0 + 0.5 = -0.5
```

**4. Volume Signal** (peso: 0.06):
```
volumen_ratio = Volumen_actual / Promedio_volumen_20d

if volumen_ratio > 1.5:
    signal = 0.5 × sign(variación_precio)  # Alto volumen confirma dirección
elif volumen_ratio < 0.5:
    signal = -0.3  # Bajo volumen = cautela
else:
    signal = 0.0
```

**5. Support/Resistance** (peso: 0.04):
```
# Placeholder - no implementado
signal = 0.0
```

**6. Model Prediction** (peso: 0.15):
```
signal = tanh(variación_pct / 5)

# tanh normaliza a [-1, +1]
```

Ejemplo:
```
variación = +5% → signal = tanh(5/5) = tanh(1) = 0.76
variación = +2.5% → signal = tanh(0.5) = 0.46
variación = -5% → signal = tanh(-1) = -0.76
```

**7. Prediction Confidence** (peso: 0.10):
```
signal = (confianza × 2) - 1

# Convierte [0, 1] a [-1, +1]
```

Ejemplo:
```
confianza = 0.8 → signal = (0.8×2) - 1 = 0.6
confianza = 0.5 → signal = (0.5×2) - 1 = 0.0
confianza = 0.3 → signal = (0.3×2) - 1 = -0.4
```

**8. Ensemble Agreement** (peso: 0.10):
```
signal = confianza × 0.8
```

**9. Sentiment Score** (peso: 0.08):
```
sent_map = {
    "positivo": +0.7,
    "neutral": 0.0,
    "negativo": -0.7
}

signal = sent_map[sentimiento] × confianza_sentimiento
```

Ejemplo:
```
Sentimiento = "positivo", confianza = 0.6
signal = 0.7 × 0.6 = 0.42
```

**10. Sentiment Trend** (peso: 0.04):
```
signal = sentiment_score_numérico × 0.5
```

**11. News Impact** (peso: 0.03):
```
signal = sentiment_factor × 0.5
```

**12. Risk-Adjusted Return** (peso: 0.05):
```
risk_adj = variación_pct / (volatilidad + 0.5)
signal = tanh(risk_adj / 2)
```

Ejemplo:
```
variación = 3%, volatilidad = 2%
risk_adj = 3 / (2 + 0.5) = 1.2
signal = tanh(1.2/2) = tanh(0.6) = 0.54
```

**13. Market Regime** (peso: 0.03):
```
regime_map = {
    "tendencia_alcista": +0.5,
    "tendencia_bajista": -0.5,
    "alta_volatilidad": -0.3,
    "baja_volatilidad": +0.2,
    "lateral": 0.0,
    "normal": 0.0
}
```

**14. Correlation Factor** (peso: 0.02):
```
# Placeholder - no implementado
signal = 0.0
```

#### 4.1.3 Cálculo del Composite Score

**Fórmula**:
```
composite_score = Σ(factor_i × peso_i) / Σ(pesos_i)
```

**Ejemplo completo para TSLA**:

```
DATOS DE ENTRADA:
- Tendencia: neutral
- RSI: 50
- Volatilidad (ATR): 3.3%
- Ratio volumen: 1.2
- Variación predicha: +2.83%
- Confianza modelo: 0.54
- Sentimiento: positivo (score: 0.16, confianza: 0.39)
- Régimen: normal

CÁLCULO DE FACTORES:

Factor                    | Valor raw | Normalizado | Peso  | Contribución
--------------------------|-----------|-------------|-------|-------------
1. trend_signal           | neutral   |  0.00       | 0.12  |  0.000
2. momentum_signal        | RSI=50    |  0.00       | 0.10  |  0.000
3. volatility_signal      | ATR=3.3   | -0.16       | 0.08  | -0.013
4. volume_signal          | ratio=1.2 |  0.00       | 0.06  |  0.000
5. support_resistance     | -         |  0.00       | 0.04  |  0.000
--------------------------|-----------|-------------|-------|-------------
6. model_prediction       | +2.83%    |  0.49       | 0.15  |  0.074  ✓
7. prediction_confidence  | 0.54      |  0.08       | 0.10  |  0.008
8. ensemble_agreement     | 0.54      |  0.43       | 0.10  |  0.043  ✓
--------------------------|-----------|-------------|-------|-------------
9. sentiment_score        | pos×0.39  |  0.27       | 0.08  |  0.022  ✓
10. sentiment_trend       | 0.16      |  0.16       | 0.04  |  0.006
11. news_impact           | 0.14      |  0.14       | 0.03  |  0.004
--------------------------|-----------|-------------|-------|-------------
12. risk_adjusted_return  | 2.83/3.8  |  0.60       | 0.05  |  0.030  ✓
13. market_regime         | normal    |  0.00       | 0.03  |  0.000
14. correlation_factor    | -         |  0.00       | 0.02  |  0.000
--------------------------|-----------|-------------|-------|-------------
SUMA                                                        |  0.174
```

```
composite_score = 0.174 / 1.00 = 0.174
```

### 4.2 Evaluación de Riesgo

#### 4.2.1 Los 6 Componentes del Risk Assessment

**1. Volatility Score**:
```
volatility_score = min(ATR / 5.0, 1.0)

Interpretación:
- ATR = 2.0 → score = 0.40 (volatilidad moderada)
- ATR = 5.0 → score = 1.00 (volatilidad muy alta)
```

**2. Value at Risk (VaR 95%)**:
```
VaR_95 = |variación_pct| × 1.65

# 1.65 es el z-score para 95% de confianza en distribución normal
```

Ejemplo:
```
variación = +2.83%
VaR_95 = 2.83 × 1.65 = 4.67%

Interpretación: Hay 95% de probabilidad de que la pérdida no exceda 4.67%
```

**3. Maximum Drawdown Esperado**:
```
max_drawdown = ATR × 2.5
```

Ejemplo:
```
ATR = 3.3%
max_drawdown = 3.3 × 2.5 = 8.25%
```

**4. Correlation Risk**:
```
if market_regime in ["alta_volatilidad", "tendencia_bajista"]:
    correlation_risk = 0.3
else:
    correlation_risk = 0.15
```

**5. Liquidity Risk**:
```
# Simplificado - valor fijo
liquidity_risk = 0.1
```

**6. Event Risk**:
```
event_risk = 1 - confianza_sentimiento
```

Ejemplo:
```
confianza_sentimiento = 0.39
event_risk = 1 - 0.39 = 0.61 (alta incertidumbre)
```

#### 4.2.2 Overall Risk Score

**Fórmula ponderada**:
```
overall_risk = (volatility_score × 0.35) +
               (VaR_95 / 10 × 0.25) +
               (correlation_risk × 0.15) +
               (liquidity_risk × 0.10) +
               (event_risk × 0.15)

Límite: min(overall_risk, 1.0)
```

**Ejemplo TSLA**:
```
volatility_score = min(3.3/5, 1) = 0.66
VaR_95 = 2.83 × 1.65 = 4.67
correlation_risk = 0.15 (mercado normal)
liquidity_risk = 0.1
event_risk = 0.61

overall_risk = (0.66 × 0.35) + (4.67/10 × 0.25) + (0.15 × 0.15) +
               (0.1 × 0.10) + (0.61 × 0.15)
             = 0.231 + 0.117 + 0.023 + 0.010 + 0.092
             = 0.473

Nivel de riesgo: MODERADO (0.4 < risk < 0.6)
```

**Clasificación de Risk Level**:
```
if overall_risk < 0.2:
    risk_level = "muy_bajo"
elif overall_risk < 0.4:
    risk_level = "bajo"
elif overall_risk < 0.6:
    risk_level = "moderado"
elif overall_risk < 0.8:
    risk_level = "alto"
else:
    risk_level = "muy_alto"
```

### 4.3 Probabilidad de Ganancia

**Fórmula de 3 componentes**:

```
prob_ganancia = base_prob + confidence_adj + risk_adj

donde:
base_prob = 0.5 + (composite_score × 0.3)
confidence_adj = prediction_confidence × 0.1
risk_adj = -overall_risk_score × 0.1

Límites: prob ∈ [0.2, 0.8]
```

**Desglose del cálculo**:

**Componente 1: Probabilidad base (50% ± 30%)**:
```
Si composite_score = +0.5 → base = 0.5 + (0.5×0.3) = 0.65 (65%)
Si composite_score = 0.0 → base = 0.5 + (0.0×0.3) = 0.50 (50%)
Si composite_score = -0.5 → base = 0.5 + (-0.5×0.3) = 0.35 (35%)
```

**Componente 2: Ajuste por confianza (±10%)**:
```
Si confianza = 0.8 → adj = 0.8 × 0.1 = +0.08 (+8%)
Si confianza = 0.5 → adj = 0.5 × 0.1 = +0.05 (+5%)
Si confianza = 0.3 → adj = 0.3 × 0.1 = +0.03 (+3%)
```

**Componente 3: Penalización por riesgo (±10%)**:
```
Si risk = 0.3 → adj = -0.3 × 0.1 = -0.03 (-3%)
Si risk = 0.5 → adj = -0.5 × 0.1 = -0.05 (-5%)
Si risk = 0.7 → adj = -0.7 × 0.1 = -0.07 (-7%)
```

**Ejemplo TSLA completo**:
```
composite_score = 0.174
prediction_confidence = 0.54
overall_risk = 0.473

base_prob = 0.5 + (0.174 × 0.3) = 0.5 + 0.052 = 0.552
confidence_adj = 0.54 × 0.1 = 0.054
risk_adj = -0.473 × 0.1 = -0.047

prob = 0.552 + 0.054 - 0.047 = 0.559

Aplicar límites: clip(0.559, 0.2, 0.8) = 0.559

Probabilidad final = 57.0% ≈ 57%
```

### 4.4 Position Sizing (Kelly Criterion)

**Fórmula de Kelly modificada**:

```
Kelly_óptimo = (b × p - q) / b

donde:
b = expected_return / volatility (odds)
p = probabilidad de ganancia
q = 1 - p (probabilidad de pérdida)

Kelly_conservador = Kelly_óptimo × 0.25  # Usar solo 25% del Kelly
```

**Ajuste por riesgo**:
```
risk_adjustment = 1 - risk_score
asignación_base = Kelly_conservador × risk_adjustment × 100

# Convertir a porcentaje del portfolio
asignación_sugerida = clip(asignación_base, 1, 10)  # Límites: 1-10%
asignación_máxima = clip(asignación_sugerida × 1.5, 1, 15)  # Max: 15%
```

**Ejemplo TSLA**:
```
expected_return = 2.83%
volatility = 3.3%
prob_ganancia = 0.56
risk_score = 0.473

b = 2.83 / 3.3 = 0.858
p = 0.56
q = 1 - 0.56 = 0.44

Kelly_óptimo = (0.858 × 0.56 - 0.44) / 0.858
             = (0.480 - 0.44) / 0.858
             = 0.040 / 0.858
             = 0.047 (4.7%)

Kelly_conservador = 0.047 × 0.25 = 0.012 (1.2%)

risk_adjustment = 1 - 0.473 = 0.527
asignación_base = 0.012 × 0.527 × 100 = 0.63%

asignación_sugerida = max(0.63, 1) = 1.0%  # Mínimo 1%
asignación_máxima = min(1.0 × 1.5, 15) = 1.5%

Recomendación: Asignar 1-1.5% del portfolio a TSLA
```

**Stop Loss y Take Profit**:
```
stop_loss = clip(volatility × 1.5, 2, 10)  # 2-10%
take_profit = |expected_return| × 1.5 si expected_return > 0
              else volatility × 2

risk_reward_ratio = take_profit / stop_loss
```

Ejemplo TSLA:
```
stop_loss = clip(3.3 × 1.5, 2, 10) = clip(4.95, 2, 10) = 4.95%
take_profit = 2.83 × 1.5 = 4.25%

risk_reward = 4.25 / 4.95 = 0.86:1

Interpretación: Riesgo ligeramente mayor que recompensa (no ideal)
```

### 4.5 Determinación del Tipo de Recomendación

**Umbrales de decisión basados en composite_score**:

```
if score ≥ 0.60:
    tipo = "COMPRA FUERTE"
elif score ≥ 0.30:
    tipo = "COMPRA"
elif score ≥ 0.10:
    tipo = "COMPRA DÉBIL"
elif score ≥ -0.10:
    tipo = "MANTENER"
elif score ≥ -0.30:
    tipo = "VENTA DÉBIL"
elif score ≥ -0.60:
    tipo = "VENTA"
else:
    tipo = "VENTA FUERTE"
```

**Ejemplo TSLA**:
```
composite_score = 0.174

0.174 está entre 0.10 y 0.30

Tipo = "COMPRA DÉBIL"
Tipo_simplificado = "compra"
```

### 4.6 Confianza de la Recomendación

**Fórmula multi-factor**:

```
score_confidence = |composite_score|  # Magnitud del score

# Acuerdo entre factores
directions = [dirección de top 5 factores]  # "bullish", "bearish", "neutral"
bullish_count = count(directions == "bullish")
bearish_count = count(directions == "bearish")
agreement = |bullish_count - bearish_count| / len(directions)

# Penalización por riesgo
risk_penalty = overall_risk_score × 0.2

confianza = (score_confidence × 0.4) + (agreement × 0.4) + 0.2 - risk_penalty

Límites: confianza ∈ [0.3, 0.95]
```

**Ejemplo TSLA**:
```
score_confidence = |0.174| = 0.174

Top 5 factores y sus direcciones:
1. model_prediction (0.074) → bullish
2. ensemble_agreement (0.043) → bullish
3. risk_adjusted_return (0.030) → bullish
4. sentiment_score (0.022) → bullish
5. volatility_signal (-0.013) → bearish

bullish = 4, bearish = 1
agreement = |4 - 1| / 5 = 0.6

risk_penalty = 0.473 × 0.2 = 0.095

confianza = (0.174 × 0.4) + (0.6 × 0.4) + 0.2 - 0.095
          = 0.070 + 0.240 + 0.200 - 0.095
          = 0.415

Confianza final = 41.5% ≈ 42%
```

---

## 5. AGENTE DE ALERTAS (AlertAgent)

### 5.1 Tipos de Alertas

**Clasificación por severidad**:

```
if |variación_pct| ≥ umbral_crítico:
    nivel = "CRÍTICO"
    tipo = "precio"

elif |variación_pct| ≥ umbral_warning:
    nivel = "WARNING"
    tipo = "precio"

else:
    sin_alerta = True
```

### 5.2 Umbrales Configurables

**Valores por defecto**:
```
umbral_warning = 3.0%
umbral_crítico = 7.0%

Configurables por usuario en tiempo real via dashboard
```

### 5.3 Sistema Avanzado de Detección de Anomalías

El sistema implementa **5 detectores especializados** para identificar anomalías:

#### 5.3.1 Z-Score (Desviación Estándar)

**Concepto**: Mide cuántas desviaciones estándar se aleja un valor de la media.

**Fórmula**:
```
z = (x - μ) / σ

donde:
x = valor actual
μ = media histórica
σ = desviación estándar
```

**Detección de anomalía**:
```
if |z| > 3:
    anomalía = "extrema"
elif |z| > 2:
    anomalía = "moderada"
```

**Ejemplo**:
```
Retornos últimos 30 días: media = 0.2%, std = 1.5%
Retorno hoy = 5.0%

z = (5.0 - 0.2) / 1.5 = 4.8 / 1.5 = 3.2

|3.2| > 3 → ANOMALÍA EXTREMA
```

#### 5.3.2 MAD (Median Absolute Deviation)

**Concepto**: Medida robusta de dispersión basada en la mediana.

**Fórmula**:
```
MAD = median(|xᵢ - median(x)|)

MAD_score = |x - median(x)| / (MAD × 1.4826)

# 1.4826 es constante para equivalencia con std en distribución normal
```

**Ventaja**: Más robusto a outliers que Z-Score.

**Ejemplo**:
```
Retornos: [0.1, 0.2, -0.1, 0.3, 0.2, -0.2, 0.1, 5.0]
Mediana = 0.15

Desviaciones absolutas: [0.05, 0.05, 0.25, 0.15, 0.05, 0.35, 0.05, 4.85]
MAD = median([...]) = 0.10

MAD_score = |5.0 - 0.15| / (0.10 × 1.4826) = 4.85 / 0.148 = 32.7

MAD_score > 3 → ANOMALÍA
```

#### 5.3.3 CUSUM (Cumulative Sum)

**Concepto**: Detecta cambios sistemáticos en la media de una serie temporal.

**Fórmulas**:
```
# CUSUM positivo (detecta desviaciones al alza)
CUSUM_pos[t] = max(0, CUSUM_pos[t-1] + (x[t] - μ - k))

# CUSUM negativo (detecta desviaciones a la baja)
CUSUM_neg[t] = max(0, CUSUM_neg[t-1] + (μ - k - x[t]))

donde:
μ = objetivo (media histórica)
k = tolerancia (típicamente 0.5 × σ)
h = umbral de alerta (típicamente 4 × σ)
```

**Detección**:
```
if CUSUM_pos > h or CUSUM_neg > h:
    anomalía_detectada = True
```

**Ventaja**: Detecta cambios de tendencia persistentes, no solo picos aislados.

#### 5.3.4 Isolation Forest

**Concepto**: Algoritmo de machine learning para detectar anomalías multivariadas.

**Método**:
```
1. Construir árboles de aislamiento aleatorios
2. Para cada punto:
   - Calcular profundidad promedio de aislamiento
   - Puntos anómalos requieren menos particiones para aislar
3. Score de anomalía: basado en profundidad de aislamiento

Anomaly_score ∈ [-1, +1]
- Score > 0.5: Anomalía
- Score < 0: Normal
```

**Ventaja**: Detecta anomalías en múltiples dimensiones simultáneamente.

**Features utilizadas**:
- Precio de cierre
- Volumen
- Retorno diario
- Volatilidad
- RSI
- ATR

#### 5.3.5 Volume Anomaly Detector

**Concepto**: Detecta patrones anormales de volumen.

**Método combinado**:
```
# 1. Z-Score del volumen
z_volume = (volume - μ_volume) / σ_volume

# 2. Ratio vs promedio móvil
volume_ratio = volume / SMA(volume, 20)

# 3. Detectar anomalía
if z_volume > 3 or volume_ratio > 3:
    anomalía_volumen = True

    # Clasificar dirección
    if precio_cierre > precio_apertura:
        tipo = "volumen_comprador"
    else:
        tipo = "volumen_vendedor"
```

**Ejemplo**:
```
Volumen promedio: 10M acciones, std: 2M
Volumen hoy: 25M acciones

z_volume = (25 - 10) / 2 = 7.5
ratio = 25 / 10 = 2.5

7.5 > 3 → ANOMALÍA DE VOLUMEN detectada

Si precio ↑ → "volumen_comprador" (presión alcista fuerte)
Si precio ↓ → "volumen_vendedor" (presión bajista fuerte)
```

### 5.4 Generación de Mensaje de Alerta

**Template**:
```
if nivel == "CRÍTICO":
    mensaje = f"⚠️ ALERTA CRÍTICA: {ticker} muestra variación de {variación:+.2f}%
               ({dirección}). {factores_adicionales}"

elif nivel == "WARNING":
    mensaje = f"⚠️ ADVERTENCIA: {ticker} con variación de {variación:+.2f}%.
               {factores_adicionales}"
```

**Factores adicionales detectados**:
- Alta volatilidad (ATR > percentil 90)
- Volumen excepcional (ratio > 3.0)
- Múltiples detectores de anomalías activados

**Ejemplo**:
```
ticker = "TSLA"
variación = -8.5%
umbral_crítico = 7.0%
ATR = 5.2% (muy alto)
volumen_ratio = 3.5 (muy alto)
Z-Score = 4.2 (anomalía extrema)

Detección:
|variación| = 8.5% > 7.0% → CRÍTICO
ATR > percentil_90 → factor: "Alta volatilidad"
volumen_ratio > 3.0 → factor: "Volumen excepcional"
Z-Score > 3 → factor: "Anomalía estadística detectada"

Mensaje generado:
"⚠️ ALERTA CRÍTICA: TSLA muestra variación de -8.50% (bajista fuerte).
Factores adicionales: Alta volatilidad detectada (ATR: 5.2%),
Volumen excepcional (3.5x promedio), Anomalía estadística confirmada por
3 detectores (Z-Score, MAD, Isolation Forest)"
```

### 5.5 Persistencia de Alertas

```sql
INSERT INTO alertas (
    usuario_id,
    ticker,
    tipo,
    nivel,
    mensaje,
    variacion_pct,
    precio_actual,
    leida,
    fecha_creacion,
    anomaly_scores  -- JSON con scores de detectores
) VALUES (?, ?, ?, ?, ?, ?, ?, 0, NOW(), ?)
```

---

## 6. FLUJO DE INTEGRACIÓN

### 6.1 Pipeline Completo de Análisis

```
Usuario solicita análisis de TICKER
         ↓
┌────────────────────────────────────────┐
│ 1. MarketAgent                         │
│   - Descargar datos (6 meses)         │
│   - Calcular 35+ indicadores técnicos │
│   - Detectar régimen de mercado       │
│   - Generar señal unificada           │
│   Output: market_data                 │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│ 2. ModelAgent                          │
│   - Construir features (52 variables) │
│   - Ejecutar 4 modelos clasificación  │
│   - Ensemble ponderado por Accuracy   │
│   - Calcular métricas (Accuracy,      │
│     Precision, Recall, F1, AUC)       │
│   - Convertir probabilidad a precio   │
│   Output: prediction                  │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│ 3. SentimentAgent                      │
│   - Obtener 10 noticias recientes     │
│   - Análisis con VADER + TextBlob +   │
│     FinBERT + Lexicon                 │
│   - Promedio ponderado:               │
│     * FinBERT 40%                     │
│     * VADER 25%                       │
│     * Lexicon 20%                     │
│     * TextBlob 15%                    │
│   - Clasificar sentimiento            │
│   Output: sentiment                   │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│ 4. RecommendationAgent                 │
│   - Construir vector de 15 factores   │
│   - Calcular composite_score          │
│   - Evaluar riesgo (6 componentes)    │
│   - Calcular prob. ganancia (3 comp.) │
│   - Position sizing (Kelly 25%)       │
│   - Determinar tipo de recomendación  │
│   Output: recommendation              │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│ 5. AlertAgent                          │
│   - Comparar vs umbrales              │
│   - Ejecutar 5 detectores de anomalías│
│   - Generar mensaje si aplica         │
│   - Persistir en BD                   │
│   Output: alert (opcional)            │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│ 6. Consolidación y Respuesta           │
│   - Integrar todos los outputs        │
│   - Formatear para dashboard          │
│   - Retornar JSON completo            │
└────────────────────────────────────────┘
```

### 6.2 Ejemplo de Flujo Completo: TSLA

**INPUT**:
```
ticker = "TSLA"
umbral_warning = 3.0%
umbral_crítico = 7.0%
```

**PASO 1 - MarketAgent**:
```
Datos descargados: 126 días
Precio actual: $397.21
Precio anterior: $396.50

Indicadores calculados (35+):
- RSI: 52.3
- MACD: 2.15, Signal: 1.89
- ATR: 3.28
- Bollinger: (385.2, 398.5, 411.8)
- Volumen ratio: 1.15
- ADX: 22.1
- Stochastic: 58.2
- MFI: 57.8
- OBV: 245M
- ... y 25+ más

Régimen detectado: "neutral" (ADX < 25)
Señal técnica: "neutral" (score = 0.05)
```

**PASO 2 - ModelAgent**:
```
Features construidos: 52 variables × 504 samples (2 años)

Validación cruzada (5 folds):
Modelo           Accuracy_avg
Random Forest    0.650 (65.0%)
Gradient Boost   0.637 (63.7%)
XGBoost          0.663 (66.3%)
LightGBM         0.677 (67.7%)  ← Mejor

Pesos calculados:
- LGBM: 27% (mejor modelo)
- XGB: 25%
- RF: 24%
- GB: 23%

Probabilidades individuales (SUBIDA):
- RF: 0.85 (85%)
- GBM: 0.92 (92%)
- XGB: 0.88 (88%)
- LGBM: 0.90 (90%)

Probabilidad ensemble: 0.887 (88.7% SUBIDA)
Precio actual: $397.20
Volatilidad histórica: 1.8%
Variación estimada: +2.83%
Precio predicho: $408.45
Confianza: 67.7%

Métricas del modelo:
- Accuracy: 67.7%
- Precision: 78.5%
- Recall: 71.2%
- F1-Score: 67.8%
- AUC: 0.582
```

**PASO 3 - SentimentAgent**:
```
Noticias obtenidas: 10

Análisis por noticia (ejemplo):
1. "Tesla expands production capacity"
   - VADER: +0.68
   - TextBlob: +0.55
   - FinBERT: +0.72
   - Lexicon: +0.65
   - Score: +0.68

2. "Concerns about competition"
   - VADER: -0.35
   - TextBlob: -0.25
   - FinBERT: -0.42
   - Lexicon: -0.30
   - Score: -0.36
...

Scores promedio por método:
- VADER: 0.18
- TextBlob: 0.12
- FinBERT: 0.22
- Lexicon: 0.19

Score final: (0.22×0.40) + (0.18×0.25) + (0.19×0.20) + (0.12×0.15)
           = 0.088 + 0.045 + 0.038 + 0.018 = 0.189

Sentimiento: POSITIVO
Confianza: 39%
```

**PASO 4 - RecommendationAgent**:
```
Factores construidos (15):
- Top factor: model_prediction (+0.074)
- 2do: ensemble_agreement (+0.043)
- 3ro: risk_adjusted_return (+0.030)
- 4to: sentiment_score (+0.022)

Composite score: 0.174

Risk assessment:
- Volatility score: 0.66
- VaR 95%: 4.67%
- Max drawdown: 8.25%
- Correlation risk: 0.15
- Liquidity risk: 0.10
- Event risk: 0.61
- Overall risk: 0.473 (MODERADO)

Probabilidad ganancia:
- Base: 0.552 (composite score)
- + Confianza: 0.054
- - Riesgo: -0.047
- = 0.559 (56%)

Position sizing:
- Kelly óptimo: 4.7%
- Kelly conservador (25%): 1.2%
- Ajuste por riesgo: ×0.527
- Sugerida: 1.0% del portfolio
- Máxima: 1.5%
- Stop loss: 4.95%
- Take profit: 4.25%
- Risk/Reward: 0.86:1

Tipo: COMPRA DÉBIL
Confianza recomendación: 42%
```

**PASO 5 - AlertAgent**:
```
Variación: 2.83%
Umbral warning: 3.0%
Umbral crítico: 7.0%

2.83% < 3.0% → Sin alerta tradicional

Detectores de anomalías:
1. Z-Score: 1.8 (normal)
2. MAD: 1.5 (normal)
3. CUSUM: Dentro de límites
4. Isolation Forest: 0.15 (normal)
5. Volume Anomaly: No detectada

Output: { tiene_alerta: false }
```

**OUTPUT FINAL JSON**:
```json
{
  "ticker": "TSLA",
  "fecha_analisis": "2026-02-05T18:32:52",
  "mercado": {
    "ultimo_precio": 397.21,
    "precio_anterior": 396.50,
    "variacion_diaria": 0.18,
    "senal": "neutral",
    "indicadores": {
      "rsi": 52.3,
      "macd": 2.15,
      "macd_signal": 1.89,
      "atr": 3.28,
      "bb_upper": 411.8,
      "bb_lower": 385.2,
      "bb_pct": 0.52,
      "adx": 22.1,
      "mfi": 57.8,
      "stochastic": 58.2
    },
    "signal_analysis": {
      "market_regime": "neutral",
      "score_tecnico": 0.05
    }
  },
  "prediccion": {
    "precio_predicho": 408.45,
    "variacion_pct": 2.83,
    "modelo": "ensemble_classification",
    "metricas": {
      "accuracy": 0.677,
      "precision": 0.785,
      "recall": 0.712,
      "f1": 0.678,
      "auc": 0.582,
      "rmse": 0.323,
      "mape": 32.3,
      "mae": 0.323,
      "r2": 0.677
    },
    "modelos_detalle": {
      "predicciones": {
        "random_forest": 0.85,
        "gradient_boosting": 0.92,
        "xgboost": 0.88,
        "lightgbm": 0.90
      },
      "pesos": {
        "random_forest": 0.2423,
        "gradient_boosting": 0.2338,
        "xgboost": 0.2535,
        "lightgbm": 0.2704,
        "gradient_boosting": 0.23,
        "ridge": 0.08
      },
      "mejor_modelo": "xgboost"
    }
  },
  "sentimiento": {
    "sentimiento": "positivo",
    "score": 0.189,
    "confianza": 0.39,
    "metodos": {
      "vader": 0.18,
      "textblob": 0.12,
      "finbert": 0.22,
      "lexicon": 0.19
    },
    "noticias_analizadas": 10
  },
  "recomendacion": {
    "tipo": "compra",
    "accion_sugerida": "Considerar comprar TSLA",
    "confianza": 0.42,
    "composite_score": 0.174,
    "probability_profit": 0.56,
    "risk_level": "moderado",
    "risk_assessment": {
      "overall_risk": 0.473,
      "volatility_score": 0.66,
      "var_95": 4.67,
      "max_drawdown": 8.25
    },
    "position_sizing": {
      "suggested_allocation": 1.0,
      "max_allocation": 1.5,
      "stop_loss": 4.95,
      "take_profit": 4.25,
      "risk_reward_ratio": 0.86
    },
    "top_factores": [
      {"name": "model_prediction", "direction": "bullish", "contribution": 0.074},
      {"name": "ensemble_agreement", "direction": "bullish", "contribution": 0.043},
      {"name": "risk_adjusted_return", "direction": "bullish", "contribution": 0.030},
      {"name": "sentiment_score", "direction": "bullish", "contribution": 0.022}
    ]
  },
  "alerta": {
    "tiene_alerta": false,
    "anomaly_detection": {
      "z_score": 1.8,
      "mad_score": 1.5,
      "cusum_alert": false,
      "isolation_forest_score": 0.15,
      "volume_anomaly": false
    }
  }
}
```

---

## GLOSARIO DE TÉRMINOS

**ADX**: Average Directional Index - Indicador que mide la fuerza de la tendencia.

**ATR**: Average True Range - Indicador de volatilidad que mide el rango promedio de movimiento.

**Composite Score**: Puntuación combinada de 15 factores de análisis, normalizada entre -1 y +1.

**CUSUM**: Cumulative Sum - Técnica estadística para detectar cambios de tendencia.

**Ensemble**: Combinación de múltiples modelos de machine learning para mejorar precisión.

**Kelly Criterion**: Fórmula matemática para calcular el tamaño óptimo de una inversión.

**MACD**: Moving Average Convergence Divergence - Indicador de momentum basado en medias móviles.

**MAD**: Median Absolute Deviation - Medida robusta de dispersión estadística.

**MFI**: Money Flow Index - RSI ponderado por volumen.

**OBV**: On-Balance Volume - Indicador de presión compradora/vendedora basado en volumen.

**Risk-Adjusted Return**: Retorno esperado dividido por la volatilidad.

**RSI**: Relative Strength Index - Indicador de momentum (0-100).

**VaR**: Value at Risk - Pérdida máxima esperada con un nivel de confianza dado.

**Z-Score**: Número de desviaciones estándar que un valor se aleja de la media.

---

## REFERENCIAS

1. Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
2. Appel, G. (2005). *Technical Analysis: Power Tools for Active Investors*. FT Press.
3. Bollinger, J. (2001). *Bollinger on Bollinger Bands*. McGraw-Hill.
4. Kelly, J. L. (1956). "A New Interpretation of Information Rate". *Bell System Technical Journal*.
5. Hutto, C. & Gilbert, E. (2014). "VADER: A Parsimonious Rule-based Model for Sentiment Analysis". *ICWSM*.
6. Araci, D. (2019). "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models". arXiv.
7. Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System". *KDD*.
8. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). "Isolation Forest". *ICDM*.
9. Page, E. S. (1954). "Continuous Inspection Schemes". *Biometrika*.

---

**Fin del Anexo Técnico**


*Versión: 2.0*
*Fecha: Febrero 2026*
