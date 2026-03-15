# Sistema Multiagente de Seguimiento y Alerta para Activos Financieros

Prototipo funcional de un sistema inteligente basado en agentes para el seguimiento y generación de alertas sobre activos financieros.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-Academic-yellow.svg)](LICENSE)



## Tabla de Contenidos

- [Descripción](#descripción)
- [Arquitectura](#arquitectura)
  - [Diagrama del Sistema](#diagrama-del-sistema)
  - [Flujo de Análisis Completo](#flujo-de-análisis-completo)
  - [Agentes Especializados](#agentes-especializados)
- [Tecnologías](#tecnologías)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Ejecución](#ejecución)
- [Dashboard Interactivo](#dashboard-interactivo)
- [Uso](#uso)
- [API Endpoints](#api-endpoints)
  - [Autenticación](#autenticación)
  - [Predicción y Análisis](#predicción-y-análisis)
  - [Alertas](#alertas)
  - [Estado del Sistema](#estado-del-sistema)
- [Características de Seguridad](#características-de-seguridad)
- [Troubleshooting](#troubleshooting)
- [Extensiones Futuras](#extensiones-futuras)
- [Autor](#autor)
- [Licencia](#licencia)



## Descripción

Este proyecto implementa un sistema multiagente que integra:

- **Obtención de datos de mercado** mediante yfinance (datos reales, sin simulaciones)
- **Análisis técnico** con 35+ indicadores y detección de anomalías
- **Predicción de dirección de precios** con ensemble de 4 modelos base + 3 opcionales de clasificación ML
- **Análisis de sentimiento** con 4 métodos NLP
- **Generación de recomendaciones** explicables multi-factor
- **Sistema de alertas** con umbrales configurables
- **Validación estricta de tickers** - rechaza símbolos inválidos con error HTTP 404

## Arquitectura

### Diagrama del Sistema

```mermaid
graph TB
    User[ Usuario] --> Dashboard[ Dashboard Streamlit<br/>:8501]
    Dashboard --> API[ API REST FastAPI<br/>:8000]
    API --> Auth[ Auth JWT]
    API --> Router1[ Routers]

    Router1 --> MA[ MarketAgent<br/>Análisis Técnico]
    Router1 --> MOA[ ModelAgent<br/>Predicción ML]
    Router1 --> SA[ SentimentAgent<br/>Análisis NLP]
    Router1 --> RA[ RecommendationAgent<br/>Decisión Multi-Factor]
    Router1 --> AA[ AlertAgent<br/>Alertas]

    MA --> YF[ Yahoo Finance]
    SA --> YF
    MOA --> Cache[( Cache)]
    SA --> Cache

    AA --> DB[( SQLite DB<br/>Usuarios/Alertas)]
    Auth --> DB

    style MA fill:#e3f2fd
    style MOA fill:#e8f5e9
    style SA fill:#fff3e0
    style RA fill:#f3e5f5
    style AA fill:#ffebee
```

### Flujo de Análisis Completo

```
┌─────────────────────────────────────────────────────────────────┐
│ Usuario solicita análisis de ticker (ej: AAPL)                 │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
         ┌────────────────────────────────────┐
         │  1. MarketAgent                    │
         │  • Descarga datos históricos       │
         │  • Calcula 35+ indicadores técnicos│
         │  • SMA, EMA, RSI, MACD, Bollinger  │
         │  • Score técnico ponderado         │
         └────────────┬───────────────────────┘
                      ▼
         ┌────────────────────────────────────┐
         │  2. ModelAgent                     │
         │  • Clasificación de dirección (3d) │
         │  • Ensemble de 4 base + 3 opcionales│
         │  • Linear, Ridge, RF, GBM          │
         │  • + XGBoost, LightGBM, LSTM (opt) │
         │  • Ventana: 504 días (2 años)      │
         │  • Métricas: Accuracy, Precision,  │
         │    Recall, F1, AUC                 │
         └────────────┬───────────────────────┘
                      ▼
         ┌────────────────────────────────────┐
         │  3. SentimentAgent                 │
         │  • Análisis de 7 noticias recientes│
         │  • FinBERT + VADER + TextBlob      │
         │  • Score de sentimiento ponderado  │
         │  • Detección de tendencia          │
         └────────────┬───────────────────────┘
                      ▼
         ┌────────────────────────────────────┐
         │  4. RecommendationAgent            │
         │  • Integra señales anteriores      │
         │  • Análisis de riesgo (VaR 95%)    │
         │  • Position sizing (Kelly)         │
         │  • Recomendación: Compra/Venta     │
         └────────────┬───────────────────────┘
                      ▼
         ┌────────────────────────────────────┐
         │  5. AlertAgent                     │
         │  • Evalúa umbrales (3% / 7%)       │
         │  • Genera alertas si necesario     │
         │  • Persiste en base de datos       │
         └────────────┬───────────────────────┘
                      ▼
         ┌────────────────────────────────────┐
         │  Respuesta JSON completa           │
         │  + Dashboard visualiza resultados  │
         └────────────────────────────────────┘
```

### Agentes Especializados

1. **MarketAgent** 
   - Descarga datos históricos de Yahoo Finance
   - Calcula 35+ indicadores técnicos avanzados:
     - **Tendencia**: SMA, EMA, MACD, ADX, Ichimoku, Parabolic SAR
     - **Momentum**: RSI, Stochastic, Williams %R, ROC, CCI
     - **Volatilidad**: Bollinger Bands, ATR, Keltner Channels
     - **Volumen**: OBV, VWAP, MFI, ADL, CMF
   - **Detección de anomalías** con 5 algoritmos:
     - Z-Score, MAD, CUSUM, Isolation Forest, Volume Anomaly
   - Genera señales de mercado con scoring ponderado
   - Identifica régimen de mercado (trending/ranging)
   - Detecta soportes y resistencias

2. **ModelAgent**
   - **Modelo de clasificación** que predice dirección del precio (subida/bajada)
   - Ensemble con modelos base (siempre presentes):
     - LogisticRegression (Linear) — clasificador probabilístico
     - RidgeClassifier — produce clase directa (0/1), no probabilidad
     - Random Forest Classifier
     - Gradient Boosting Classifier
   - Modelos opcionales (según librerías instaladas):
     - XGBoost Classifier (si xgboost disponible)
     - LightGBM Classifier (si lightgbm disponible)
     - LSTM (si pytorch disponible)
   - **Ensemble probabilístico efectivo**: RF, GB, XGB, LGB (los que aportan predict_proba)
   - Feature engineering con 52 características técnicas
   - Ventana de entrenamiento: **504 días** (2 años)
   - Horizonte de predicción: **3 días**
   - Walk-forward validation temporal con 5 splits
   - Métricas de clasificación: **Accuracy**, **Precision**, **Recall**, **F1-Score**, **AUC-ROC**
   - Conversión de probabilidad a precio estimado usando volatilidad histórica

3. **SentimentAgent** 
   - Ensemble de modelos NLP:
     - **FinBERT** (40% peso) - Transformer especializado en finanzas
     - **VADER** (25% peso) - Análisis léxico
     - **TextBlob** (15% peso) - Polaridad general
     - **Léxico Financiero** (20% peso) - 500+ términos propietarios
   - Análisis de 7 noticias recientes de Yahoo Finance
   - Detección de tendencia de sentimiento
   - Extracción de temas clave (earnings, M&A, regulatory)
   - Caché de 1 hora para optimizar

4. **RecommendationAgent** 
   - Sistema de decisión multi-factor con 15+ variables:
     - **Señales Técnicas** (40%): Tendencia, momentum, volatilidad, volumen
     - **Predicción** (35%): Score del modelo, confianza, acuerdo ensemble
     - **Sentimiento** (15%): Score NLP, tendencia, impacto noticias
     - **Riesgo** (10%): Sharpe ratio, régimen mercado, correlación
   - Gestión de riesgo:
     - Value at Risk (VaR) al 95%
     - Estimación de max drawdown
     - Evaluación de riesgos específicos
   - Position sizing con Kelly Criterion
   - 7 niveles de recomendación (Compra Fuerte → Venta Fuerte)
   - Explicabilidad completa de cada factor

5. **AlertAgent** 
   - Evaluación de umbrales configurables:
     - **Warning**: 3% de variación
     - **Critical**: 7% de variación
   - Persistencia en base de datos con timestamps
   - 6 niveles de severidad: Info / Low / Medium / High / Critical / Emergency
   - Gestión de alertas leídas/no leídas
   - Integrable con notificaciones push/email

## Tecnologías

- **Backend**: FastAPI, SQLAlchemy, Pydantic
- **ML**: scikit-learn, pandas, numpy
- **Datos**: yfinance
- **Frontend**: Streamlit
- **Seguridad**: JWT, bcrypt

## Estructura del Proyecto

```
proyecto_final/
├── backend/
│   ├── __init__.py
│   ├── main.py              # Aplicación FastAPI
│   ├── config.py            # Configuración
│   ├── database.py          # Modelos SQLAlchemy
│   ├── schemas.py           # Schemas Pydantic
│   ├── auth.py              # Autenticación JWT
│   ├── email_service.py     # Servicio de email (recuperación de contraseña)
│   ├── routers/
│   │   ├── auth_router.py   # Endpoints de autenticación
│   │   ├── predict_router.py # Endpoints de predicción
│   │   └── alerts_router.py  # Endpoints de alertas
│   └── agents/
│       ├── market_agent.py
│       ├── model_agent.py
│       ├── sentiment_agent.py
│       ├── recommendation_agent.py
│       └── alert_agent.py
├── dashboard/
│   └── app.py               # Dashboard Streamlit
├── docs/
│   └── ANEXO_TECNICO_CALCULOS.md  # Fórmulas y cálculos técnicos del sistema
├── tests/                    # Suite de pruebas
│   ├── __init__.py
│   ├── test_functional.py   # Pruebas funcionales (30 tests)
│   └── test_performance.py  # Pruebas de carga
├── test_results/             # Resultados de pruebas
│   ├── graficos/            # Gráficos (PDF/PNG)
│   └── *.json               # Resultados en JSON
├── Chapter1.tex             # Cap. 1: Introducción
├── Chapter2.tex             # Cap. 2: Marco teórico
├── Chapter3.tex             # Cap. 3: Diseño e implementación
├── Chapter4.tex             # Cap. 4: Ensayos y resultados
├── Chapter5.tex             # Cap. 5: Conclusiones y trabajo futuro
├── requirements.txt
├── requirements-test.txt     # Dependencias de testing
├── run_all_tests.bat        # Script Windows para ejecutar tests
├── run_all_tests.sh         # Script Linux/Mac para ejecutar tests
├── manage_services.sh       # Script de gestión de servicios (start/stop/restart)
├── check_setup.py           # Verificación de instalación
├── reset_password_helper.py # Utilidad para reset de contraseña vía consola
├── test_reset_flow.py       # Prueba del flujo de recuperación de contraseña
├── VALIDACION_ACADEMICA.md  # Validación académica: fundamentos, estado del arte y métricas
├── INFORME_PRUEBAS.md       # Informe detallado de resultados de pruebas
├── CONFIGURATION.md         # Configuración avanzada y deployment
├── EXAMPLES.md              # Ejemplos prácticos de uso
├── .env.example
└── README.md
```

## Instalación

### 1. Clonar el repositorio

```bash
cd proyecto_final
```

### 2. Crear entorno virtual

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

```bash
# Copiar el archivo de ejemplo
cp .env.example .env

# Editar .env con tu editor favorito
nano .env  # o vim, code, notepad++, etc.
```

** IMPORTANTE:** El archivo `.env` contiene información sensible y **NO DEBE** ser commiteado a git (ya está en `.gitignore`).

#### Variables Críticas (DEBES configurar):

**`SECRET_KEY`** - Clave secreta para firmar tokens JWT
```bash
# Generar una clave segura ejecutando en terminal:
python -c "import secrets; print(secrets.token_urlsafe(32))"

# O con openssl:
openssl rand -hex 32

# Copiar el resultado a .env:
SECRET_KEY=tu_clave_generada_aqui
```

#### Variables Opcionales (pueden usar valores por defecto):

| Variable | Descripción | Default | Ejemplo |
|----------|-------------|---------|---------|
| `ALGORITHM` | Algoritmo de firma JWT | `HS256` | `HS256` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Expiración de token | `30` | `30` |
| `DATABASE_URL` | Conexión a base de datos | SQLite local | `sqlite:///./financial_tracker.db` |
| `ALERT_THRESHOLD_WARNING` | Umbral alerta warning (%) | `3.0` | `3.0` |
| `ALERT_THRESHOLD_CRITICAL` | Umbral alerta crítica (%) | `7.0` | `7.0` |
| `SMTP_SERVER` | Servidor email (opcional) | - | `smtp.gmail.com` |
| `SMTP_PORT` | Puerto SMTP (opcional) | - | `587` |
| `SMTP_USER` | Usuario email (opcional) | - | `tu_email@gmail.com` |
| `SMTP_PASSWORD` | Contraseña de aplicación (opcional) | - | `tu_app_password` |

**Nota sobre SMTP:**
- **Sin SMTP configurado**: El sistema funciona normalmente. Los tokens de recuperación de contraseña aparecen en los logs del backend.
- **Con SMTP configurado**: Los tokens se envían automáticamente por email para mejor experiencia de usuario.

### 5. Verificar la instalación (Recomendado)

Ejecuta el script de verificación para asegurarte de que todo está configurado correctamente:

```bash
python check_setup.py
```

El script verificará:
-  Versión de Python (3.8+)
-  Dependencias instaladas
-  Estructura del proyecto
-  Configuración del archivo .env
-  Puertos disponibles (8000, 8501)

**Salida esperada:**
```
 VERIFICACIÓN DE INSTALACIÓN
Sistema Multiagente de Seguimiento Financiero

============================================================
  Verificando Python
============================================================
 Versión de Python: 3.10.0 (OK)

============================================================
  Verificando Dependencias Principales
============================================================
 FastAPI: Instalado
 Uvicorn: Instalado
...

============================================================
  Resumen
============================================================
 ¡Todo está configurado correctamente!
   El sistema está listo para ejecutarse.
```

Si encuentras errores, el script te proporcionará instrucciones específicas para corregirlos.

---

## Ejecución

### Iniciar el Backend

```bash
uvicorn backend.main:app --reload --port 8000
```

El API estará disponible en:
- API: http://localhost:8000
- Documentación Swagger: http://localhost:8000/docs
- Documentación ReDoc: http://localhost:8000/redoc

### Iniciar el Dashboard

En otra terminal:

```bash
streamlit run dashboard/app.py
```

El dashboard estará disponible en:
- http://localhost:8501

### Script de Gestión de Servicios (Linux/Mac)

Para facilitar la gestión de ambos servicios, puedes usar el script `manage_services.sh`:

```bash
# Iniciar backend + dashboard
bash manage_services.sh start

# Detener ambos servicios
bash manage_services.sh stop

# Reiniciar servicios
bash manage_services.sh restart

# Ver estado de los servicios
bash manage_services.sh status
```

**Nota:** Este script es para sistemas Unix/Linux/Mac. En Windows, inicia cada servicio en terminales separadas como se describe arriba.

---

## Dashboard Interactivo

El dashboard de Streamlit proporciona una interfaz visual completa para interactuar con el sistema:

###  Pantalla de Autenticación
- Login con usuario y contraseña
- Registro de nuevos usuarios
- Validación en tiempo real
- Sesión persistente con JWT

###  Análisis de Activos

**Búsqueda de Ticker:**
```
┌────────────────────────────────┐
│  Buscar Activo Financiero    │
│ ┌────────────────────────────┐ │
│ │ AAPL                    []││
│ └────────────────────────────┘ │
│                                │
│ Ejemplos: AAPL, TSLA, MSFT,   │
│          GOOGL, AMZN, META     │
└────────────────────────────────┘
```

**Panel de Análisis Técnico:**
- Gráfico de precio histórico con candlesticks (Plotly interactivo)
- Indicadores técnicos superpuestos:
  - Medias móviles (SMA 20, EMA 12)
  - Bandas de Bollinger
  - MACD
  - RSI
-  Score técnico visual (0-10)
-  Régimen de mercado actual

**Panel de Predicción:**
-  **Predicción de dirección** a 3 días (SUBIDA ⬆️ / BAJADA ⬇️)
-  Gráfico con proyección de 3 días (línea punteada)
-  Precio estimado basado en probabilidad y volatilidad
-  **Métricas de clasificación**:
  ```
  ✓ Accuracy: 57.0%    ✓ Precision: 60.7%
  ✓ Recall: 66.2%      ✓ F1-Score: 58.9%
  ```
-  Probabilidad de cada dirección por modelo individual
-  Contribución de cada modelo en el ensemble (pesos)

**Panel de Sentimiento:**
-    Indicador visual de sentimiento
-  Lista de noticias analizadas con scores individuales
- Tendencia de sentimiento (mejorando/deteriorando/estable)
-  Temas clave identificados (earnings, M&A, regulatory)

**Panel de Recomendación:**
-    Indicador visual de acción (Compra/Mantener/Venta)
-  Nivel de confianza (barra de progreso)
- Explicación detallada de la recomendación
-  Análisis de riesgo:
  - VaR 95%
  - Max Drawdown estimado
  - Sharpe Ratio
-  Position Sizing sugerido:
  - Allocación recomendada (%)
  - Stop Loss
  - Take Profit
  - Risk/Reward Ratio

###  Centro de Alertas
```
╔════════════════════════════════════════╗
║   Alertas Activas                    ║
╠════════════════════════════════════════╣
║    TSLA - Variación crítica (-7.8%)  ║
║      $245.30 vs $265.50 predicho       ║
║      Hace 2 horas                  [✓] ║
╟────────────────────────────────────────╢
║    AAPL - Variación warning (+3.5%)  ║
║      $185.42 vs $179.20 predicho       ║
║      Hace 5 horas                  [✓] ║
╚════════════════════════════════════════╝
```

**Funcionalidades:**
- Lista de alertas con filtros (leídas/no leídas)
- Códigos de color por tipo (info/warning/critical)
- Marcar como leída con un click
- Estadísticas de alertas
- Historial completo

###  Estadísticas del Usuario
- Total de consultas realizadas
- Activos más consultados
- Historial de recomendaciones
- Precisión de predicciones pasadas

## Uso

### 1. Registro de Usuario

Desde el dashboard o via API:

```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "usuario", "email": "usuario@email.com", "password": "password123"}'
```

### 2. Login

```bash
curl -X POST http://localhost:8000/auth/login \
  -d "username=usuario&password=password123"
```

### 3. Recuperar Contraseña (Sin SMTP Configurado)

Si olvidaste tu contraseña y no tienes SMTP configurado, sigue estos pasos:

#### Paso 1: Solicitar Token de Reseteo

Desde el dashboard o via API:

```bash
curl -X POST http://localhost:8000/auth/forgot-password \
  -H "Content-Type: application/json" \
  -d '{"email": "tu_email@example.com"}'
```

**Respuesta:**
```json
{
  "message": "Si el email está registrado, recibirás instrucciones para resetear tu contraseña.",
  "detail": "Por seguridad, no indicamos si el email existe o no."
}
```

#### Paso 2: Obtener el Token de los Logs

Como SMTP no está configurado, el token se imprime en los logs del backend. Busca en la consola donde ejecutaste `uvicorn`:

```bash
# Buscar en logs del backend
grep "Token de reseteo" <archivo_logs>

# Verás algo como:
# INFO - Token de reseteo para tu_usuario: ABC123XYZ456...
```

**Ejemplo del log:**
```
2026-02-06 14:12:11,167 - backend.email_service - INFO - Token de reseteo para demo_test: Id6SNF3G-Ua7ev4jvvHU0tb4WvkcdPoEAHujZzPob__3quId16e5e2kGFh0tEapA
```

**Copia el token completo** (la cadena después de los dos puntos).

#### Paso 3: Resetear la Contraseña con el Token

Usa el token para establecer una nueva contraseña:

```bash
curl -X POST http://localhost:8000/auth/reset-password \
  -H "Content-Type: application/json" \
  -d '{
    "token": "ABC123XYZ456...",
    "new_password": "NuevaPassword123!"
  }'
```

**Respuesta exitosa:**
```json
{
  "message": "Contraseña actualizada exitosamente",
  "detail": "Ya puedes iniciar sesión con tu nueva contraseña"
}
```

** Notas Importantes:**
- El token expira en 1 hora
- El token es de un solo uso
- Si SMTP está configurado, recibirás el token por email automáticamente

### 4. Análisis de Activo

```bash
curl -X GET http://localhost:8000/predict/AAPL \
  -H "Authorization: Bearer <token>"
```

### 5. Ver Alertas

```bash
curl -X GET http://localhost:8000/alerts \
  -H "Authorization: Bearer <token>"
```

## API Endpoints

### Autenticación

#### `POST /auth/register` - Registrar usuario
**Request:**
```json
{
  "username": "investor123",
  "email": "investor@example.com",
  "password": "SecurePass123!"
}
```

**Response:** `201 Created`
```json
{
  "id": 1,
  "username": "investor123",
  "email": "investor@example.com",
  "rol": "user",
  "fecha_registro": "2026-02-05T10:30:00"
}
```

#### `POST /auth/login` - Iniciar sesión
**Request:** (Form data)
```
username=investor123
password=SecurePass123!
```

**Response:** `200 OK`
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

#### `GET /auth/me` - Obtener usuario actual
**Headers:** `Authorization: Bearer {token}`

**Response:** `200 OK`
```json
{
  "id": 1,
  "username": "investor123",
  "email": "investor@example.com",
  "rol": "user"
}
```

#### `POST /auth/refresh` - Refrescar token
**Headers:** `Authorization: Bearer {token}`

**Response:** `200 OK`
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```



### Predicción y Análisis

#### `GET /predict/{ticker}` - Análisis completo
**Ejemplo:** `GET /predict/AAPL`

**Headers:** `Authorization: Bearer {token}`

**Response:** `200 OK`
```json
{
  "ticker": "AAPL",
  "market_data": {
    "precio_actual": 185.42,
    "precio_max": 189.50,
    "precio_min": 182.30,
    "volumen": 52847300,
    "variacion_pct": 2.15,
    "sma_20": 183.45,
    "ema_12": 184.20,
    "rsi": 58.32,
    "macd": 1.45,
    "signal": 1.20,
    "bollinger_upper": 190.30,
    "bollinger_lower": 176.60,
    "volatilidad": 0.0245,
    "score_tecnico": 6.8,
    "señales": ["RSI neutral", "MACD bullish", "Precio sobre SMA20"],
    "regimen_mercado": "trending_up",
    "timestamp": "2026-02-05T14:30:00Z"
  },
  "prediction": {
    "precio_predicho": 188.75,
    "variacion_pct": 1.80,
    "horizonte_dias": 3,
    "modelo_usado": "ensemble_classification",
    "metricas": {
      "accuracy": 0.570,
      "precision": 0.607,
      "recall": 0.662,
      "f1": 0.589,
      "auc": 0.586
    },
    "modelos_detalle": {
      "predicciones": {
        "linear": 0.55,
        "ridge": 0.57,
        "random_forest": 0.88,
        "gradient_boosting": 0.85,
        "xgboost": 0.92,
        "lightgbm": 0.90
      },
      "pesos": {
        "linear": 0.12,
        "ridge": 0.13,
        "random_forest": 0.22,
        "gradient_boosting": 0.21,
        "xgboost": 0.16,
        "lightgbm": 0.16
      }
    },
    "parametros": {
      "ventana": 504,
      "n_features": 52,
      "n_modelos": "4-7 (según librerías instaladas)",
      "mejor_modelo": "varía según ticker y entrenamiento"
    }
  },
  "sentiment": {
    "score": 0.42,
    "categoria": "positivo",
    "tendencia": "mejorando",
    "noticias_analizadas": 7,
    "noticias": [
      {
        "titulo": "Apple Reports Record Q1 Earnings",
        "score": 0.78,
        "fuente": "Yahoo Finance",
        "fecha": "2026-02-04"
      }
    ],
    "temas_clave": ["earnings", "innovation", "market_expansion"],
    "confianza": 0.85
  },
  "recommendation": {
    "accion": "compra",
    "nivel": "moderado",
    "confianza": 0.82,
    "razon": "Señales técnicas alcistas combinadas con sentimiento positivo y predicción favorable",
    "factores": {
      "tecnico": {"score": 6.8, "peso": 0.40, "contribucion": 2.72},
      "prediccion": {"score": 7.5, "peso": 0.35, "contribucion": 2.63},
      "sentimiento": {"score": 6.5, "peso": 0.15, "contribucion": 0.98},
      "riesgo": {"score": 5.2, "peso": 0.10, "contribucion": 0.52}
    },
    "score_total": 6.85,
    "gestion_riesgo": {
      "var_95": 8750.50,
      "max_drawdown_estimado": 0.12,
      "sharpe_ratio": 1.85,
      "riesgos": ["volatilidad_moderada", "exposicion_sector_tech"]
    },
    "position_sizing": {
      "kelly_criterion": 0.08,
      "allocacion_sugerida_pct": 5.0,
      "stop_loss": 176.15,
      "take_profit": 195.80,
      "risk_reward_ratio": 2.5
    }
  },
  "alertas_generadas": [
    {
      "tipo": "info",
      "mensaje": "Precio actual dentro del rango esperado",
      "variacion_pct": 2.15
    }
  ],
  "timestamp": "2026-02-05T14:30:00Z"
}
```

#### `GET /predict/{ticker}/market` - Solo datos de mercado
**Response:** Retorna únicamente el objeto `market_data` del ejemplo anterior

#### `GET /predict/{ticker}/sentiment` - Solo sentimiento
**Response:** Retorna únicamente el objeto `sentiment` del ejemplo anterior



### Alertas

#### `GET /alerts` - Listar alertas del usuario
**Headers:** `Authorization: Bearer {token}`

**Query params:**
- `skip`: Número de registros a omitir (paginación)
- `limit`: Máximo de registros a retornar (default: 100)

**Response:** `200 OK`
```json
{
  "total": 23,
  "alertas": [
    {
      "id": 45,
      "ticker": "TSLA",
      "tipo": "critical",
      "mensaje": " Variación crítica detectada",
      "variacion_pct": -7.8,
      "precio_actual": 245.30,
      "precio_predicho": 265.50,
      "leida": false,
      "fecha_creacion": "2026-02-05T09:15:00Z"
    },
    {
      "id": 44,
      "ticker": "AAPL",
      "tipo": "warning",
      "mensaje": " Variación significativa detectada",
      "variacion_pct": 3.5,
      "precio_actual": 185.42,
      "precio_predicho": 179.20,
      "leida": true,
      "fecha_creacion": "2026-02-05T08:30:00Z"
    }
  ]
}
```

#### `GET /alerts/stats` - Estadísticas de alertas
**Headers:** `Authorization: Bearer {token}`

**Response:** `200 OK`
```json
{
  "total_alertas": 156,
  "alertas_no_leidas": 8,
  "por_tipo": {
    "info": 98,
    "warning": 42,
    "critical": 16
  },
  "ultimas_24h": 12,
  "ticker_mas_alertas": "TSLA"
}
```

#### `GET /alerts/{id}` - Detalle de una alerta
**Headers:** `Authorization: Bearer {token}`

**Response:** `200 OK`
```json
{
  "id": 45,
  "usuario_id": 1,
  "ticker": "TSLA",
  "tipo": "critical",
  "mensaje": " Variación crítica detectada: El precio actual ($245.30) difiere significativamente del predicho ($265.50)",
  "variacion_pct": -7.8,
  "precio_actual": 245.30,
  "precio_predicho": 265.50,
  "leida": false,
  "fecha_creacion": "2026-02-05T09:15:00Z"
}
```

#### `PUT /alerts/{id}/read` - Marcar alerta como leída
**Headers:** `Authorization: Bearer {token}`

**Response:** `200 OK`
```json
{
  "id": 45,
  "leida": true,
  "mensaje": "Alerta marcada como leída"
}
```

#### `DELETE /alerts/{id}` - Eliminar alerta
**Headers:** `Authorization: Bearer {token}`

**Response:** `204 No Content`



### Estado del Sistema

#### `GET /` - Información de la API
**Response:** `200 OK`
```json
{
  "nombre": "Sistema Multiagente de Seguimiento Financiero",
  "version": "1.0.0",
  "descripcion": "API para análisis y predicción de activos financieros",
  "endpoints": {
    "docs": "/docs",
    "redoc": "/redoc",
    "health": "/health"
  }
}
```

#### `GET /health` - Estado del sistema
**Response:** `200 OK`
```json
{
  "status": "healthy",
  "database": "connected",
  "timestamp": "2026-02-05T14:30:00Z",
  "uptime_seconds": 345678
}
```

#### `GET /config` - Configuración pública
**Response:** `200 OK`
```json
{
  "alert_threshold_warning": 3.0,
  "alert_threshold_critical": 7.0,
  "token_expire_minutes": 30,
  "cors_origins": ["http://localhost:8501"]
}
```

## Características de Seguridad

- Autenticación JWT con tokens firmados
- Hash de contraseñas con bcrypt
- Validación de datos con Pydantic
- CORS configurado para frontend
- Sin exposición de secretos en configuración pública



## Testing y Resultados

### Suite de Pruebas Automatizadas

El proyecto incluye una suite completa de pruebas funcionales y de rendimiento validadas con datos reales.

####  Ejecutar Todas las Pruebas

**Windows:**
```bash
run_all_tests.bat
```

**Linux/Mac:**
```bash
bash run_all_tests.sh
```

**Manualmente:**
```bash
# Instalar dependencias de testing
pip install -r requirements-test.txt

# Asegúrate que el backend esté corriendo
uvicorn backend.main:app --reload

# Ejecutar pruebas funcionales (30 tests)
python tests/test_functional.py

# Ejecutar pruebas de rendimiento (carga concurrente)
python tests/test_performance.py
```

####  Resultados Obtenidos

**Pruebas Funcionales** (Fecha: 9 febrero 2026)
-  **30/30 pruebas exitosas (100% tasa de éxito)**
-  **Latencia promedio: 3.20 segundos**
-  **Mejora del 31.7%** después de primera iteración (caché: 4.04s → 2.76s)
-  **10 tickers evaluados** × 3 iteraciones cada uno
-  **Todos los agentes operativos al 100%** (MarketAgent, ModelAgent, SentimentAgent, RecommendationAgent, AlertAgent)

**Modelos de Machine Learning (Clasificación Binaria)** — configuración final (8 marzo 2026)
- **Ensemble Accuracy: 57.0%** - Predice correctamente la dirección en ~57% de los casos (mejor que azar del 50%)
- **Precision: 60.7%** - Cuando predice subida, acierta en ~6 de cada 10 casos
- **Recall: 66.2%** - Detecta el 66% de las subidas reales
- **F1-Score: 58.9%** - Balance entre precision y recall
- **AUC: 58.6%** - Capacidad de discriminación del modelo
- **Variabilidad**: Accuracy varía entre 53.1% (TSLA, META) y 63.3% (GOOGL) según el ticker
- **Mejor rendimiento**: GOOGL (63.3%), AAPL (59.9%), JPM (58.5%)
- **Nota**: Clasificación de dirección a 3 días (SUBIDA/BAJADA), umbral target 0.5%, ventana 504 días, 5 folds

**Análisis de Sentimiento (NLP)**
-  Ensemble de 4 modelos: FinBERT (40%), VADER (25%), Lexicón financiero (20%), TextBlob (15%)
-  Fuente de noticias: Yahoo Finance (yfinance)
-  Score de sentimiento consolidado entre -1 (negativo) y +1 (positivo)
-  Sin dataset de validación etiquetado — evaluación cualitativa en sesiones de prueba

**Pruebas de Rendimiento**
-  **Zona óptima**: 1-25 usuarios concurrentes (100% éxito)
-  **Zona degradada**: 25-50 usuarios (latencia aumenta)
-  **Punto de colapso**: 50 usuarios (16% tasa de éxito)
-  **Throughput máximo**: 1.56 req/s

**Cuellos de Botella Identificados**
1. **ModelAgent**: Entrenamiento del ensemble de clasificadores (Linear, Ridge, RF, GB + XGB/LGB/LSTM opcionales)
2. **MarketAgent**: Descarga de datos históricos desde Yahoo Finance (dependencia de API externa)

####  Gráficos y Visualizaciones

```bash
test_results/graficos/
├── grafico_latencia_componentes.pdf   # Distribución de latencia por nivel de carga
└── grafico_pruebas_carga.pdf          # Escalabilidad del sistema (1-50 usuarios)
```

####  Cumplimiento de Objetivos

| Objetivo | Meta | Resultado | Estado |
|----------|------|-----------|--------|
| Arquitectura multiagente | 5 agentes | 5 agentes @ 100% |  |
| Predicción de dirección | > 50% Accuracy | 57.0% Accuracy |  |
| Precision en clasificación | > 55% | 60.7% Precision |  |
| Recall en detección | > 65% | 66.2% Recall |  |
| Análisis sentimiento NLP | Ensemble 4 modelos | FinBERT+VADER+Lexicón+TextBlob |  |
| Tiempo respuesta | < 5s | 3.20s promedio |  |
| 20+ usuarios concurrentes | ≥ 20 | 25 usuarios @ 100% |  |
| Dashboard funcional | Implementado | Streamlit |  |

####  Archivos de Resultados

Todos los resultados están guardados en formato JSON/CSV para análisis posterior:

```bash
test_results/
├── functional_test_20260213_233918.json    # Resultados detallados con métricas de clasificación
├── functional_test_20260213_233918.csv     # Formato tabular
├── performance_test_20260209_160016.json   # Pruebas de carga
└── summary_20260213_233918.json            # Resumen ejecutivo con estadísticas de clasificación
```

**Contenido del resumen incluye:**
- Métricas de clasificación agregadas (accuracy, precision, recall, f1, auc)
- Rendimiento por agente (100% todos los agentes)
- Estadísticas de latencia (promedio, mín, máx)
- Distribución de errores (si los hay)



## Troubleshooting

### Problemas Comunes y Soluciones

####  Error: "ModuleNotFoundError: No module named 'backend'"

**Causa:** Ejecutando el comando desde el directorio incorrecto

**Solución:**
```bash
# Asegúrate de estar en el directorio raíz del proyecto
cd proyecto_final

# Verifica que veas la carpeta backend
ls backend/

# Ejecuta desde aquí
uvicorn backend.main:app --reload
```



####  Error: "SECRET_KEY not configured"

**Causa:** Archivo `.env` no existe o no tiene SECRET_KEY

**Solución:**
```bash
# 1. Verifica que .env existe
ls -la .env

# 2. Si no existe, créalo desde el ejemplo
cp .env.example .env

# 3. Genera una clave segura
python -c "import secrets; print(secrets.token_urlsafe(32))"

# 4. Copia la salida y pégala en .env
# SECRET_KEY=la_clave_generada_aqui
```



####  Error: "Could not connect to database"

**Causa:** Problemas con SQLite o permisos

**Solución:**
```bash
# 1. Verifica permisos del directorio
ls -la financial_tracker.db

# 2. Si el archivo no existe, se creará automáticamente
# Asegúrate de tener permisos de escritura

# 3. Para resetear la base de datos:
rm financial_tracker.db
# Se recreará al iniciar el backend
```



####  Error: "Address already in use" (Puerto 8000 o 8501 ocupado)

**Causa:** Ya hay un proceso usando el puerto

**Solución en Linux/Mac:**
```bash
# Ver qué proceso usa el puerto 8000
lsof -i :8000

# Matar el proceso (reemplaza PID con el número mostrado)
kill -9 PID

# O usar otro puerto
uvicorn backend.main:app --reload --port 8001
```

**Solución en Windows:**
```bash
# Ver qué proceso usa el puerto 8000
netstat -ano | findstr :8000

# Matar el proceso (reemplaza PID con el número mostrado)
taskkill /PID PID /F

# O usar otro puerto
uvicorn backend.main:app --reload --port 8001
```



####  Error: "401 Unauthorized" en el dashboard

**Causa:** Token JWT expirado o inválido

**Solución:**
1. Cierra sesión en el dashboard
2. Vuelve a iniciar sesión
3. Si persiste, verifica que el SECRET_KEY sea el mismo en backend y que no haya cambiado



####  Error: "Failed to fetch data from Yahoo Finance"

**Causa:** Problemas de conexión o ticker inválido

**Solución:**
```bash
# 1. Verifica tu conexión a internet
ping yahoo.com

# 2. Verifica que el ticker sea válido
# Usa tickers reales: AAPL, TSLA, MSFT, GOOGL, etc.
# Consulta https://finance.yahoo.com/ para verificar el símbolo correcto

# 3. El sistema NO genera datos simulados
# Si el ticker no existe, recibirás un error HTTP 404
# El sistema trabaja exclusivamente con datos reales de Yahoo Finance
```



####  Error: "ImportError: cannot import name 'FinBERT'"

**Causa:** Dependencias opcionales de NLP no instaladas

**Solución:**
```bash
# Instalar todas las dependencias incluyendo opcionales
pip install torch transformers

# Si tienes problemas con PyTorch en Windows:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Nota:** El sistema funcionará sin estas dependencias, usando modelos de NLP más ligeros (VADER, TextBlob)



####  Dashboard muestra "Connection Error"

**Causa:** Backend no está ejecutándose

**Solución:**
```bash
# 1. Verifica que el backend esté corriendo
curl http://localhost:8000/health

# 2. Si no responde, inicia el backend en otra terminal
uvicorn backend.main:app --reload

# 3. Espera a ver el mensaje "Application startup complete"

# 4. Recarga el dashboard
```



####  Advertencia: "UserWarning: Matplotlib is building the font cache"

**Causa:** Primera ejecución de matplotlib

**Solución:** Es normal, solo ocurre la primera vez. Espera unos segundos y se completará automáticamente.



####  Las predicciones parecen muy inexactas

**Causa:** Los modelos de predicción necesitan más datos históricos o el mercado es muy volátil

**Solución:**
- Los modelos de clasificación funcionan mejor con tickers estables y líquidos
- Usa tickers de grandes empresas (AAPL, MSFT) en lugar de penny stocks
- El modelo predice dirección (SUBIDA/BAJADA), no precio exacto
- Accuracy del 57.0% supera el umbral aleatorio del 50% en los 10 tickers evaluados
- Evalúa las métricas (Accuracy, Precision, Recall) para entender el rendimiento
- Una precision del 60.7% significa que cuando predice subida, acierta en ~6 de cada 10 casos



####  No recibo el email de recuperación de contraseña

**Causa:** SMTP no está configurado (comportamiento normal)

**Solución:**

El sistema está diseñado para funcionar sin SMTP. El token aparece en los logs del backend:

```bash
# 1. Busca en la consola donde ejecutaste uvicorn
# Verás una línea como esta:
# INFO - Token de reseteo para tu_usuario: ABC123XYZ456...

# 2. O busca en logs si los guardaste
grep "Token de reseteo" backend.log

# 3. Ejemplo de salida:
# 2026-02-06 14:12:11 - backend.email_service - INFO - Token de reseteo para demo_test: Id6SNF3G-Ua7ev4jvvHU0tb4WvkcdPoEAHujZzPob__3quId16e5e2kGFh0tEapA
```

**Copia el token completo** y úsalo en el endpoint `/auth/reset-password` o en el dashboard (botón "Tengo el token").

**Para configurar SMTP (opcional):**
```bash
# Agrega a tu archivo .env:
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=tu_email@gmail.com
SMTP_PASSWORD=tu_app_password  # No uses tu contraseña regular
```

** Nota:** Con Gmail necesitas una "Contraseña de Aplicación", no tu contraseña normal.



### Logs y Debugging

#### Ver logs del backend:
El backend imprime logs en la terminal. Busca:
-  `INFO: Application startup complete` - Todo OK
-  `ERROR:` - Indica un problema específico

#### Modo debug detallado:
```bash
# Agregar en .env
DEBUG=true

# Reiniciar backend
uvicorn backend.main:app --reload --log-level debug
```

#### Verificar configuración:
```bash
# Ver configuración pública actual
curl http://localhost:8000/config

# Verificar salud del sistema
curl http://localhost:8000/health
```



### Obtener Ayuda

Si encuentras un problema no listado aquí:

1. **Revisa los logs** del backend en la terminal
2. **Verifica tu configuración** en `.env`
3. **Prueba con curl** para aislar si es problema del backend o dashboard
4. **Busca en Issues** del repositorio si es código público
5. **Crea un Issue** con:
   - Descripción del problema
   - Logs relevantes
   - Pasos para reproducir
   - Sistema operativo y versión de Python

## Extensiones Futuras

1. **Notificaciones en tiempo real**: Agregar WebSocket para actualización de alertas sin recargar página
2. **Alertas por email/mensajería**: Envío de alertas por correo electrónico, Telegram o SMS
3. **Escalabilidad**: Migrar a PostgreSQL y agregar caché Redis para mayor concurrencia
4. **Backtesting**: Sistema de pruebas históricas de estrategias
5. **Explicabilidad**: SHAP values para explicar predicciones del ensemble
6. **Despliegue**: Containerización con Docker y pipeline CI/CD
7. **Expansión geográfica**: Soporte para mercados internacionales y análisis en español



## Documentación Adicional

-  **[docs/ANEXO_TECNICO_CALCULOS.md](docs/ANEXO_TECNICO_CALCULOS.md)**
  
-  **[VALIDACION_ACADEMICA.md](VALIDACION_ACADEMICA.md)** 

-  **[INFORME_PRUEBAS.md](INFORME_PRUEBAS.md)**
   
- **[CONFIGURATION.md](CONFIGURATION.md)** 

-  **[EXAMPLES.md](EXAMPLES.md)** 
    
-  **[check_setup.py](check_setup.py)** 
    

## Alumna:

Ing. María Fabiana Cid

## Licencia

Proyecto académico - Especialización en Inteligencia Artificial, FIUBA
