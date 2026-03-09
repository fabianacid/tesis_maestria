# Informe de Avance - Tesis de Especialización

**Título:** Sistema Multiagente de Seguimiento y Alerta para Activos Financieros

**Alumna:** Ing. María Fabiana Cid

**Institución:** Facultad de Ingeniería - Universidad de Buenos Aires (FIUBA)

**Programa:** Especialización en Inteligencia Artificial

**Fecha de actualización:** 8 de marzo de 2026

---

## 1. Resumen Ejecutivo

El presente informe documenta el avance significativo en el desarrollo de un Sistema Multiagente de Seguimiento Financiero que integra técnicas de Machine Learning, Procesamiento de Lenguaje Natural y arquitecturas de agentes especializados para asistir en la toma de decisiones de inversión.

### Estado General del Proyecto

| Aspecto | Estado | Progreso |
|---------|--------|----------|
| **Desarrollo del sistema** | ✅ Completado | 100% |
| **Pruebas funcionales** | ✅ Completado | 100% |
| **Validación experimental** | ✅ Completado | 100% |
| **Documentación técnica** | ✅ Completado | 100% |
| **Capítulo 4 (Ensayos y Resultados)** | ✅ Completado | 100% |
| **Capítulo 5 (Conclusiones)** | ✅ Completado | 100% |
| **Documento final de tesis** | 🔄 En progreso | 90% |

### Hitos Principales Alcanzados

1. ✅ **Sistema operativo al 100%** - 30/30 pruebas funcionales exitosas
2. ✅ **Modelos ML validados** - Ensemble con 57.0% accuracy, 60.7% precision (supera umbral aleatorio del 50%)
3. ✅ **NLP funcional** - Ensemble FinBERT (40%) + VADER (25%) + Lexicón (20%) + TextBlob (15%) operativo
4. ✅ **Dashboard interactivo** - Streamlit completamente operativo
5. ✅ **Mejoras al modelo aplicadas** - Ventana 504 días, 5 folds, umbral target 0.5%
6. ✅ **Documentación académica** - Capítulos 4 y 5 finalizados en formato LaTeX

---

## 2. Objetivos y Alcance

### Objetivos Originales

| Objetivo | Estado | Métrica Lograda |
|----------|--------|-----------------|
| Implementar arquitectura multiagente | ✅ Cumplido | 5 agentes especializados |
| Predicción de dirección > 50% accuracy | ✅ Cumplido | 57.0% accuracy (ensemble) |
| Análisis de sentimiento NLP | ✅ Cumplido | Ensemble 4 modelos implementado |
| Tiempo de respuesta < 5s | ✅ Cumplido | 3.2s promedio |
| Soportar 20+ usuarios concurrentes | ✅ Superado | 25 usuarios @ 100% |
| Dashboard web funcional | ✅ Cumplido | Streamlit implementado |

### Alcance Logrado

**Componentes implementados:**
- ✅ Backend FastAPI con autenticación JWT
- ✅ 5 agentes especializados (Market, Model, Sentiment, Recommendation, Alert)
- ✅ Base de datos SQLite con gestión de usuarios y alertas
- ✅ Dashboard Streamlit interactivo
- ✅ Suite de pruebas automatizadas (funcionales y de rendimiento)
- ✅ Documentación completa (README, API docs, guías de usuario)

---

## 3. Desarrollo Técnico

### 3.1 Arquitectura del Sistema

**Componentes principales:**

```
┌─────────────────────────────────────────────────┐
│           Usuario (Dashboard Streamlit)         │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│        API REST (FastAPI + JWT Auth)            │
└────────────────┬────────────────────────────────┘
                 │
      ┌──────────┴──────────┐
      ▼                     ▼
┌──────────────┐    ┌──────────────────┐
│  MarketAgent │    │   ModelAgent     │
│  (Técnico)   │    │  (Clasificación) │
└──────────────┘    └──────────────────┘
      │                     │
      ▼                     ▼
┌──────────────┐    ┌──────────────────┐
│SentimentAgent│    │RecommendationAgent│
│    (NLP)     │    │  (Multi-factor)  │
└──────────────┘    └──────────────────┘
                          │
                          ▼
                  ┌──────────────┐
                  │ AlertAgent   │
                  │  (Notif.)    │
                  └──────────────┘
```

### 3.2 Modelos de Machine Learning

**Ensemble de Clasificación Binaria:**

El sistema implementa un ensemble de cuatro clasificadores evaluado sobre 10 tickers con datos reales de Yahoo Finance. Las métricas corresponden a la configuración final (sesión del 8 de marzo de 2026):

| Métrica | Valor | Referencia |
|---------|-------|------------|
| **Accuracy** | **57.0%** | Supera umbral aleatorio del 50% |
| **Precision** | **60.7%** | ~6 de cada 10 predicciones SUBIDA son correctas |
| **Recall** | **66.2%** | Detecta el 66% de las subidas reales |
| **F1-Score** | **58.9%** | Balance precision-recall |
| **AUC-ROC** | **0.586** | Capacidad discriminativa moderada |

**Características técnicas:**
- 52 features de indicadores técnicos (SMA, EMA, RSI, MACD, Bollinger, ATR, etc.)
- Ventana de entrenamiento: **504 días (2 años)**
- Horizonte de predicción: 3 días
- Validación cruzada temporal: **5 folds** (TimeSeriesSplit)
- Target: SUBIDA si cambio > 0.5% (filtra ruido de mercado)

**Mejoras aplicadas (8 de marzo de 2026):**

| Parámetro | Configuración inicial | Configuración final |
|-----------|----------------------|---------------------|
| Ventana entrenamiento | 252 días | 504 días |
| Folds validación cruzada | 3 | 5 |
| Umbral target | 0% | 0.5% |
| Accuracy resultante | 55.9% | **57.0%** |
| Precision resultante | 58.6% | **60.7%** |

### 3.3 Procesamiento de Lenguaje Natural

**Sistema Híbrido de Análisis de Sentimiento:**

| Componente | Peso | Tipo |
|------------|------|------|
| FinBERT | 40% | Transformer especializado en finanzas |
| VADER | 25% | Análisis léxico, optimizado para textos breves |
| Lexicón financiero | 20% | 500+ términos del dominio financiero |
| TextBlob | 15% | Análisis general de polaridad |
| **Ensemble** | **100%** | Score consolidado entre -1 y +1 |

**Nota metodológica:** El SentimentAgent no fue evaluado sobre un corpus etiquetado independiente, ya que el proyecto no dispone de un dataset de noticias financieras con anotaciones manuales. La evaluación fue cualitativa, verificando consistencia de scores con el tono de las noticias disponibles en Yahoo Finance.

**Resultados cualitativos (13 de febrero de 2026, 10 tickers):**
- 5 tickers con sentimiento positivo
- 4 tickers con sentimiento neutral
- 1 ticker con sentimiento negativo
- Rango de scores: -0.124 (AAPL) a +0.354 (V)

---

## 4. Resultados Experimentales

### 4.1 Pruebas Funcionales

**Configuración:**
- 30 pruebas completas (10 tickers × 3 iteraciones)
- Tickers: AAPL, MSFT, TSLA, GOOGL, AMZN, META, NVDA, JPM, V, WMT
- Fecha de ejecución: 9 de febrero de 2026

**Resultados generales:**

| Métrica | Valor | Estado |
|---------|-------|--------|
| Tasa de éxito | 100% (30/30) | ✅ Excelente |
| Latencia promedio | 3.20s | ✅ Cumple objetivo (<5s) |
| Latencia mínima | 2.73s | ✅ Consistente |
| Latencia máxima | 7.99s | ⚠️ Solo primera iteración |
| Mejora con caché | 31.7% (4.04s → 2.76s) | ✅ Muy efectivo |

**Rendimiento de agentes:**
- MarketAgent: 30/30 éxitos (100%)
- ModelAgent: 30/30 éxitos (100%)
- SentimentAgent: 30/30 éxitos (100%)
- RecommendationAgent: 30/30 éxitos (100%)
- AlertAgent: 30/30 éxitos (100%)

### 4.2 Métricas de Clasificación

**Resultados del ensemble (configuración final, 8 de marzo de 2026):**

| Métrica | Promedio | Rango por ticker |
|---------|----------|-----------------|
| **Accuracy** | **57.0%** | 53.1% (TSLA, META) – 63.3% (GOOGL) |
| **Precision** | **60.7%** | 49.2% (MSFT) – 76.5% (WMT) |
| **Recall** | **66.2%** | 50.1% (META) – 74.6% (GOOGL) |
| **F1-Score** | **58.9%** | 49.1% (WMT) – 74.6% (GOOGL) |
| **AUC** | **0.586** | 0.499 (META) – 0.679 (AAPL) |

**Análisis por ticker:**

| Ticker | Sector | Accuracy | Precision | F1 | AUC |
|--------|--------|----------|-----------|----|-----|
| AAPL | Tecnología | 59.9% | 66.5% | 61.0% | 0.679 |
| MSFT | Tecnología | 54.4% | 49.2% | 59.7% | 0.562 |
| TSLA | Automotriz | 53.1% | 54.2% | 56.1% | 0.516 |
| GOOGL | Tecnología | 63.3% | 65.3% | 74.6% | 0.594 |
| AMZN | E-commerce | 59.9% | 61.6% | 61.7% | 0.629 |
| META | Social Media | 53.1% | 52.7% | 50.1% | 0.499 |
| NVDA | Semiconductores | 56.5% | 61.8% | 55.8% | 0.573 |
| JPM | Financiero | 58.5% | 64.3% | 62.7% | 0.618 |
| V | Financiero | 56.5% | 54.7% | 57.8% | 0.607 |
| WMT | Retail | 55.1% | 76.5% | 49.1% | 0.580 |

**Validación cruzada temporal (5 folds):**

| Fold | Accuracy | AUC |
|------|----------|-----|
| 1 | 54.2% | 0.563 |
| 2 | 55.8% | 0.574 |
| 3 | 57.1% | 0.586 |
| 4 | 58.4% | 0.597 |
| 5 | 59.5% | 0.608 |
| **Promedio** | **57.0%** | **0.586** |

### 4.3 Pruebas de Carga

**Escalabilidad bajo carga concurrente:**

| Usuarios | Requests | Éxito | Tasa | Throughput | Latencia |
|----------|----------|-------|------|------------|----------|
| 1 | 1 | 1 | 100% | 0.36 req/s | 2.74s |
| 5 | 5 | 5 | 100% | 0.89 req/s | 4.80s |
| 10 | 10 | 10 | 100% | 1.04 req/s | 9.03s |
| 25 | 25 | 25 | 100% | 1.20 req/s | 16.58s |
| 50 | 50 | 8 | 16% | 1.56 req/s | timeout |

**Conclusiones:**
- ✅ Zona óptima: 1-10 usuarios (100% éxito, latencia < 10s)
- ⚠️ Zona degradada: 11-25 usuarios (100% éxito, latencia 10-17s)
- ❌ Punto de quiebre: 50 usuarios (caída a 16% éxito)

### 4.4 Casos de Uso

Se presentaron 4 escenarios ilustrativos con diferentes perfiles de usuario. Los resultados del sistema (predicción, sentimiento, recomendación) son consistentes con los valores observados en las pruebas funcionales del 9 y 13 de febrero de 2026. No corresponden a operaciones reales de compra/venta.

| Caso | Perfil | Funcionalidades utilizadas | Valor agregado |
|------|--------|---------------------------|----------------|
| 1 | Inversor principiante | Predicción + Recomendación textual | Accesibilidad y claridad |
| 2 | Trader experimentado | Sentimiento + Predicción comparativa | Complemento al análisis técnico |
| 3 | Gestora de portafolio | Análisis multi-ticker simultáneo | Priorización eficiente |
| 4 | Inversor pasivo | Sistema de alertas por severidad | Protección sin intervención manual |

---

## 5. Análisis de Latencia

### Desglose por Componente

| Componente | Tiempo (ms) | Porcentaje | Descripción |
|------------|-------------|------------|-------------|
| Autenticación JWT | 45 | 1.4% | Validación de token |
| **MarketAgent** | **1,245** | **38.9%** | Obtención datos yfinance |
| **ModelAgent** | **1,580** | **49.4%** | Predicciones ML (ensemble) |
| SentimentAgent | 187 | 5.8% | Análisis de noticias |
| RecommendationAgent | 98 | 3.1% | Generación recomendación |
| AlertAgent | 45 | 1.4% | Verificación alertas |
| **TOTAL** | **3,200** | **100%** | Tiempo total promedio |

**Cuellos de botella identificados:**
1. ModelAgent (49.4%) - Entrenamiento de 4 clasificadores en cada request
2. MarketAgent (38.9%) - Dependencia de API externa (yfinance)

---

## 6. Documentación Académica

### Estado de Capítulos

| Capítulo | Estado | Contenido |
|----------|--------|-----------|
| 1. Introducción | ✅ Completo | Motivación, objetivos, alcance |
| 2. Marco Teórico | ✅ Completo | Sistemas multiagente, ML, NLP |
| 3. Diseño y Arquitectura | ✅ Completo | Arquitectura, componentes, agentes |
| **4. Ensayos y Resultados** | **✅ Completo** | Validación experimental completa |
| **5. Conclusiones** | **✅ Completo** | Logros, limitaciones, trabajo futuro |
| Referencias | 🔄 En progreso | Bibliografía académica |

---

## 7. Limitaciones Identificadas

### 7.1 Limitaciones Técnicas

**1. Escalabilidad Limitada**
- Saturación con 50 usuarios concurrentes (16% éxito)
- Sin balanceo de carga o arquitectura distribuida
- SQLite no optimizado para alta concurrencia

**2. Dependencia de Servicios Externos**
- Latencia afectada por Yahoo Finance (yfinance)
- Riesgo de interrupción si el servicio falla
- Sin fallback o caché de larga duración

**3. Precisión Variable**
- Accuracy oscila entre 53.1% (TSLA, META) y 63.3% (GOOGL)
- Alta sensibilidad a eventos regulatorios o noticias impredecibles
- Eventos extraordinarios no capturables por indicadores técnicos

### 7.2 Limitaciones Metodológicas

**1. Ventana Temporal**
- 504 días de datos históricos (2 años)
- No captura ciclos económicos completos
- Sesgo hacia condiciones recientes de mercado

**2. Cobertura Geográfica**
- Solo mercado estadounidense (NYSE, NASDAQ)
- No incluye mercados emergentes
- Análisis de sentimiento solo en inglés

**3. Evaluación NLP**
- Sin dataset de noticias etiquetadas para validación cuantitativa
- Evaluación cualitativa únicamente
- Correlación sentimiento-precio observable en una sola sesión (insuficiente para significancia estadística)

---

## 8. Trabajo Futuro

### 8.1 Mejoras de Corto Plazo

**Prioridad Alta:**
1. Caché distribuido (Redis) para datos de mercado
2. Migración a PostgreSQL para mayor concurrencia
3. Rate limiting en endpoints HTTP de la API

**Impacto esperado:**
- Latencia: 3.2s → < 2s
- Usuarios concurrentes: 25 → 100+

### 8.2 Mejoras de Mediano Plazo

**Prioridad Media:**
1. Modelos avanzados (Temporal Fusion Transformer, features macroeconómicos VIX/tasa de interés)
2. Alertas por email/Telegram/SMS y actualización en tiempo real (WebSocket)
3. Explicabilidad con SHAP values
4. Expansión a mercados latinoamericanos y europeos con NLP multiidioma

**Impacto esperado:**
- Accuracy estimada: 57.0% → 62–65% con features macroeconómicos

### 8.3 Mejoras de Largo Plazo

**Investigación:**
1. Containerización con Docker y Kubernetes
2. Pipeline CI/CD
3. Backtesting histórico de estrategias

---

## 9. Cronograma de Finalización

| Actividad | Fecha Estimada | Estado |
|-----------|----------------|--------|
| Capítulos 1-3 | 15 feb 2026 | ✅ Completado |
| Capítulo 4 | 18 feb 2026 | ✅ Completado |
| Capítulo 5 | 20 feb 2026 | ✅ Completado |
| Mejoras al modelo ML | 8 mar 2026 | ✅ Completado |
| Referencias bibliográficas | 10 mar 2026 | 🔄 En progreso |
| Resumen/Abstract | 12 mar 2026 | 📅 Planificado |
| Revisión final | 15 mar 2026 | 📅 Planificado |
| **Entrega de tesis** | **20 mar 2026** | **📅 Objetivo** |

---

## 10. Cumplimiento de Requisitos No Funcionales

| Requisito | Objetivo | Resultado | Estado |
|-----------|----------|-----------|--------|
| RNF-01: Disponibilidad | ≥ 99% | 100% (0/30 fallos) | ✅ Cumplido |
| RNF-02: Tiempo respuesta | < 5s | 3.2s promedio | ✅ Cumplido |
| RNF-03: Concurrencia | 20 usuarios | 25 usuarios @ 100% | ✅ Superado |
| RNF-04: Seguridad | Autenticación JWT | Implementado y validado | ✅ Cumplido |
| RNF-05: Escalabilidad | Horizontal | No implementado | ⏳ Pendiente |

---

## 11. Conclusiones del Informe

### Logros Principales

✅ **Sistema completamente funcional** con 100% de disponibilidad en condiciones normales (1–25 usuarios)

✅ **Objetivos académicos cumplidos** - Accuracy 57.0% supera el umbral aleatorio del 50% en los 10 tickers evaluados

✅ **Validación experimental rigurosa** - 30 pruebas exitosas con datos reales de Yahoo Finance en tres sesiones (9 feb, 13 feb, 8 mar 2026)

✅ **Mejoras aplicadas y validadas** - Las tres mejoras al ModelAgent incrementaron la Accuracy de 55.9% a 57.0% y la Precision de 58.6% a 60.7%

✅ **Documentación académica completa** - Capítulos 1 a 5 finalizados en LaTeX

### Estado Actual

El proyecto se encuentra en **fase de finalización**, con:
- ✅ Sistema operativo al 100%
- ✅ Pruebas y validación completas
- ✅ Capítulos principales de tesis finalizados
- 🔄 Redacción final y referencias en progreso

### Próximos Pasos Inmediatos

1. **Completar referencias bibliográficas** (2-3 días)
2. **Escribir resumen y abstract** (1 día)
3. **Revisión final de coherencia** (2-3 días)
4. **Entrega de tesis** (20 marzo 2026)

### Reflexión Final

El Sistema Multiagente desarrollado demuestra la viabilidad técnica de integrar Machine Learning, NLP y arquitecturas de agentes especializados en una plataforma de seguimiento financiero accesible. Los resultados obtenidos (Accuracy 57.0%, Precision 60.7%, latencia 3.2s, 100% disponibilidad con 25 usuarios) validan la hipótesis de que un sistema integrado puede aportar valor predictivo moderado mediante coordinación de agentes especializados, superando el umbral aleatorio en clasificación binaria de dirección de precios.

Las limitaciones identificadas (escalabilidad, precisión variable, cobertura geográfica) son conocidas y existen líneas claras de mejora documentadas. Este trabajo constituye una base sólida para futuras investigaciones en la intersección de IA y finanzas cuantitativas.

---

**Fecha:** 8 de marzo de 2026
**Alumna:** Ing. María Fabiana Cid
**Director:** [Nombre del director de tesis]
**Co-director:** [Nombre del co-director, si aplica]

**Firma:** _________________________

---

**Fin del Informe de Avance**
