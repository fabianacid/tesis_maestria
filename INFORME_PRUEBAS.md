# Informe Detallado de Pruebas del Sistema Multiagente

**Fecha de ejecución:** 8 de marzo de 2026 (configuración final)
**Sistema evaluado:** Sistema Multiagente de Seguimiento Financiero
**Tipo de pruebas:** Funcionales con validación de métricas de clasificación
**Tickers evaluados:** 10 (AAPL, MSFT, TSLA, GOOGL, AMZN, META, NVDA, JPM, V, WMT)
**Iteraciones por ticker:** 3
**Total de pruebas:** 30

**Nota:** Este informe documenta los resultados de la configuración final del ModelAgent (ventana 504 días, 5 folds, umbral target 0.5%). Los resultados de la sesión inicial del 13 de febrero de 2026 (configuración base: 252 días, 3 folds) arrojaron Accuracy 55.92% y Precision 58.64%.

---

## Resumen Ejecutivo

### Resultados Generales

| Métrica | Valor | Estado |
|---------|-------|--------|
| **Tasa de éxito** | 100% (30/30) | ✅ Excelente |
| **Latencia promedio** | 3.20s | ✅ Cumple objetivo (<5s) |
| **Latencia mínima** | 2.73s | ✅ Consistente |
| **Latencia máxima** | 7.99s | ⚠️ Solo primera iteración |
| **Mejora con caché** | 31.7% (4.04s → 2.76s) | ✅ Muy efectivo |

### Rendimiento de Agentes

| Agente | Tests | Éxitos | Tasa |
|--------|-------|--------|------|
| MarketAgent | 30 | 30 | 100% |
| ModelAgent | 30 | 30 | 100% |
| SentimentAgent | 30 | 30 | 100% |
| RecommendationAgent | 30 | 30 | 100% |
| AlertAgent | 30 | 30 | 100% |

**Conclusión:** Todos los agentes operan al 100% de efectividad. No se detectaron fallos ni inconsistencias.

---

## Métricas de Clasificación del Modelo

### Configuración final aplicada

| Parámetro | Valor |
|-----------|-------|
| Ventana de entrenamiento | 504 días (2 años) |
| Folds validación cruzada | 5 (TimeSeriesSplit) |
| Umbral target | 0.5% (SUBIDA si cambio > 0.5%) |
| Modelos del ensemble | RF, GB, XGBoost, LightGBM |
| Features técnicos | 52 |

### Resultados Agregados

| Métrica | Promedio | Mínimo | Máximo |
|---------|----------|--------|--------|
| **Accuracy** | **57.0%** | 53.1% (TSLA, META) | 63.3% (GOOGL) |
| **Precision** | **60.7%** | 49.2% (MSFT) | 76.5% (WMT) |
| **Recall** | **66.2%** | 50.1% (META) | 74.6% (GOOGL) |
| **F1-Score** | **58.9%** | 49.1% (WMT) | 74.6% (GOOGL) |
| **AUC** | **0.586** | 0.499 (META) | 0.679 (AAPL) |

### Mejora respecto a configuración inicial

| Métrica | Config. inicial (13 feb) | Config. final (8 mar) | Mejora |
|---------|--------------------------|----------------------|--------|
| Accuracy | 55.92% | **57.0%** | +1.1 pp |
| Precision | 58.64% | **60.7%** | +2.1 pp |
| Recall | 69.66% | **66.2%** | -3.5 pp* |
| F1-Score | 58.06% | **58.9%** | +0.8 pp |
| AUC | 0.595 | **0.586** | -0.9 pp* |

*El umbral de 0.5% reduce el recall al filtrar movimientos pequeños, pero mejora la calidad de las señales (mayor precision).

### Interpretación

**Accuracy (57.0%):**
- Supera el baseline de clasificación aleatoria (50%)
- Predice correctamente la dirección en ~57% de los casos
- Mejora de 1.1 puntos porcentuales respecto a configuración inicial

**Precision (60.7%):**
- Cuando predice SUBIDA, acierta en ~6 de cada 10 casos
- Mejora significativa de 2.1 pp: más confiable al emitir señales alcistas
- Variabilidad entre tickers: 49.2% (MSFT) – 76.5% (WMT)

**Recall (66.2%):**
- Detecta el 66% de las subidas reales
- Reducción esperada por el umbral de 0.5%: el modelo es más selectivo

**F1-Score (58.9%):**
- Balance mejorado entre precision y recall
- Confirma que el modelo aporta valor predictivo moderado

**AUC (0.586):**
- Capacidad de discriminación moderada
- Valores > 0.5 confirman poder predictivo

---

## Análisis Detallado por Ticker

### Mejores Tickers (Accuracy ≥ 59%)

#### 1. **GOOGL (Alphabet Inc.)** - Accuracy: 63.3%
```
Métricas:
├── Accuracy: 63.3% (Mejor rendimiento)
├── Precision: 65.3%
├── Recall: 74.6%
├── F1-Score: 74.6%
└── AUC: 0.594

Latencia promedio: 2.93s
Sentimiento: Positivo (+0.329)
```

**Análisis:**
- Mejor ticker del portafolio para clasificación
- F1 más alto del conjunto (74.6%)
- Tendencias técnicas consistentes y señales claras

#### 2. **AAPL (Apple Inc.)** - Accuracy: 59.9%
```
Métricas:
├── Accuracy: 59.9%
├── Precision: 66.5%
├── Recall: 61.0%
├── F1-Score: 61.0%
└── AUC: 0.679 (Mejor AUC del portafolio)

Latencia promedio: 2.97s
Sentimiento: Negativo (-0.124)
```

**Análisis:**
- Mejor AUC del portafolio (0.679): mayor capacidad discriminativa
- Alta liquidez y volumen facilitan señales técnicas claras

#### 3. **AMZN (Amazon.com Inc.)** - Accuracy: 59.9%
```
Métricas:
├── Accuracy: 59.9%
├── Precision: 61.6%
├── Recall: 61.7%
├── F1-Score: 61.7%
└── AUC: 0.629

Latencia promedio: 3.10s
Sentimiento: Neutral (-0.059)
```

**Análisis:**
- Rendimiento equilibrado y consistente
- Balance preciso entre precision y recall

---

### Tickers en Rango Medio (53% – 59%)

#### JPM (JPMorgan Chase) - Accuracy: 58.5%
- Precision: 64.3%, Recall: 62.7%, F1: 62.7%, AUC: 0.618
- Sector financiero con patrones predecibles
- Latencia baja y consistente (3.42s)

#### NVDA (NVIDIA Corp.) - Accuracy: 56.5%
- Precision: 61.8%, Recall: 55.8%, F1: 55.8%, AUC: 0.573
- Alta volatilidad del sector semiconductores
- Latencia: 3.18s

#### V (Visa Inc.) - Accuracy: 56.5%
- Precision: 54.7%, Recall: 57.8%, F1: 57.8%, AUC: 0.607
- AUC sólido (0.607)
- Latencia: 3.31s

#### WMT (Walmart Inc.) - Accuracy: 55.1%
- Precision: 76.5% (mayor del portafolio), Recall: 49.1%, F1: 49.1%, AUC: 0.580
- Alta precisión pero recall bajo: el modelo es muy selectivo con WMT
- El umbral 0.5% beneficia especialmente a este ticker (era 48.98% en configuración inicial)

#### MSFT (Microsoft Corp.) - Accuracy: 54.4%
- Precision: 49.2%, Recall: 59.7%, F1: 59.7%, AUC: 0.562
- Recall aceptable pero precision más baja

---

### ⚠️ Tickers con Mayor Desafío (Accuracy ≈ 53%)

#### TSLA (Tesla Inc.) - Accuracy: 53.1%
```
Métricas:
├── Accuracy: 53.1%
├── Precision: 54.2%
├── Recall: 56.1%
├── F1-Score: 56.1%
└── AUC: 0.516

Latencia promedio: 3.01s
Sentimiento: Positivo (+0.215)
```

**Análisis:**
- Alta volatilidad y sensibilidad a eventos externos impredecibles
- AUC cercano a 0.5: capacidad discriminativa limitada

#### META (Meta Platforms) - Accuracy: 53.1%
```
Métricas:
├── Accuracy: 53.1%
├── Precision: 52.7%
├── Recall: 50.1%
├── F1-Score: 50.1%
└── AUC: 0.499 (Menor AUC del portafolio)

Latencia promedio: 3.25s
Sentimiento: Neutral (+0.093)
```

**Análisis:**
- AUC 0.499: prácticamente sin capacidad discriminativa
- Alta sensibilidad a eventos regulatorios impredecibles
- F1 de 50.1% indica rendimiento equivalente al azar para este ticker

---

## Análisis de Rendimiento y Latencia

### Distribución de Latencia por Iteración

| Iteración | Latencia Promedio | Variación |
|-----------|-------------------|-----------|
| 1ª (sin caché) | 4.04s | — |
| 2ª (con caché) | 2.76s | **-31.7%** |
| 3ª (con caché) | 2.79s | +1.1% |

### Latencia por Ticker

| Ticker | Latencia promedio | Estado |
|--------|-------------------|--------|
| AAPL | 2.97s | ✅ |
| MSFT | 2.84s | ✅ |
| TSLA | 3.01s | ✅ |
| GOOGL | 2.93s | ✅ |
| AMZN | 3.10s | ✅ |
| META | 3.25s | ✅ |
| NVDA | 3.18s | ✅ |
| JPM | 3.42s | ✅ |
| V | 3.31s | ✅ |
| WMT | 4.19s | ✅ |
| **Promedio** | **3.20s** | ✅ |

**Conclusión:**
- Latencia muy consistente en todos los tickers
- WMT tiene la latencia más alta (4.19s) por características del dataset
- Todos dentro del objetivo < 5s

---

## Análisis de Sentimiento

### Scores de Sentimiento (13 de febrero de 2026)

| Ticker | Score | Categoría |
|--------|-------|-----------|
| AAPL | -0.124 | Negativo |
| MSFT | +0.086 | Neutral |
| TSLA | +0.215 | Positivo |
| GOOGL | +0.329 | Positivo |
| AMZN | -0.059 | Neutral |
| META | +0.093 | Neutral |
| NVDA | +0.293 | Positivo |
| JPM | +0.295 | Positivo |
| V | +0.354 | Positivo |
| WMT | -0.052 | Neutral |

**Distribución:** 5 positivos / 4 neutrales / 1 negativo

**Nota metodológica:** El SentimentAgent no fue evaluado con un corpus etiquetado. La evaluación es cualitativa, verificando consistencia de scores con el tono de las noticias disponibles.

---

## Comparativa con Baseline

| Métrica | Baseline (Azar) | Config. inicial | Config. final | Mejora total |
|---------|-----------------|-----------------|---------------|--------------|
| Accuracy | 50.00% | 55.92% | **57.0%** | +14.0% |
| Precision | 50.00% | 58.64% | **60.7%** | +21.4% |
| Recall | 50.00% | 69.66% | **66.2%** | +32.4% |
| F1-Score | 50.00% | 58.06% | **58.9%** | +17.8% |
| AUC | 0.500 | 0.595 | **0.586** | +17.2% |

**Conclusión:** El sistema supera consistentemente al baseline en todas las métricas. La configuración final mejora la Accuracy y Precision respecto a la configuración inicial, con una ligera reducción en Recall esperada por el umbral de 0.5%.

---

## Fortalezas y Limitaciones

### ✅ Fortalezas

1. **Estabilidad Operacional** — 100% de éxito en todas las pruebas, todos los agentes funcionan correctamente
2. **Latencia cumple SLA** — 3.20s promedio < 5s objetivo
3. **Mejora efectiva** — Las 3 mejoras aplicadas elevaron Accuracy de 55.9% a 57.0% y Precision de 58.6% a 60.7%
4. **GOOGL destaca** — 63.3% accuracy y mejor F1 (74.6%)
5. **WMT mejoró significativamente** — de 48.98% (peor que azar) a 55.1%

### ⚠️ Limitaciones

1. **Accuracy modesta** — 57.0% supera el azar pero tiene margen de mejora
2. **Variabilidad entre tickers** — rango 53.1% (TSLA, META) a 63.3% (GOOGL)
3. **META con AUC 0.499** — prácticamente sin capacidad discriminativa
4. **Escalabilidad** — soporta hasta 25 usuarios concurrentes (16% éxito con 50)

### 📋 Siguientes Pasos

1. **Corto plazo:** Features macroeconómicos (VIX, tasa de interés) — se estima Accuracy 62–65%
2. **Mediano plazo:** Temporal Fusion Transformer, backtesting histórico
3. **Largo plazo:** Migración a PostgreSQL, Docker, pipeline CI/CD

---

## Archivos Generados

- `test_results/functional_test_20260308_195912.json` — Datos completos (configuración final)
- `test_results/functional_test_20260308_195912.csv` — Formato tabular
- `test_results/summary_20260308_195912.json` — Resumen estadístico
- `test_results/functional_test_20260213_233918.json` — Datos configuración inicial (referencia)
- `test_results/summary_20260213_233918.json` — Resumen configuración inicial (referencia)

---

**Informe generado:** 8 de marzo de 2026
**Autor:** Ing. María Fabiana Cid
**Versión:** 2.0 (configuración final)
