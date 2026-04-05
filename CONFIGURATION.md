# Guía de Configuración Avanzada

Este documento proporciona información detallada sobre la configuración del sistema para usuarios avanzados y escenarios de producción.

##  Contenidos

- [Variables de Entorno](#variables-de-entorno)
- [Configuración de Base de Datos](#configuración-de-base-de-datos)
- [Configuración de Seguridad](#configuración-de-seguridad)
- [Configuración de Agentes](#configuración-de-agentes)
- [Configuración de Cache](#configuración-de-cache)
- [Deployment en Producción](#deployment-en-producción)
- [Optimización de Performance](#optimización-de-performance)

---

## Variables de Entorno

### Ubicación del Archivo

El archivo `.env` debe estar en la raíz del proyecto:
```
proyecto_final/
├── .env              ← Aquí
├── .env.example
├── backend/
└── dashboard/
```

### Variables Completas

#### Seguridad y Autenticación

```bash
# Clave secreta para firmar tokens JWT
# CRÍTICO: Debe ser único y aleatorio en producción
SECRET_KEY=your_secret_key_here

# Algoritmo de firma JWT
# Opciones: HS256 (recomendado), HS384, HS512
ALGORITHM=HS256

# Tiempo de expiración del token en minutos
# Desarrollo: 30-60, Producción: 15-30
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Tiempo de expiración del refresh token en días (futuro)
REFRESH_TOKEN_EXPIRE_DAYS=7
```

**Generación de SECRET_KEY segura:**

```bash
# Método 1: Python secrets (recomendado)
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Método 2: OpenSSL
openssl rand -hex 32

# Método 3: UUID
python -c "import uuid; print(str(uuid.uuid4()).replace('-', ''))"
```

---

#### Base de Datos

```bash
# SQLite (Desarrollo)
DATABASE_URL=sqlite:///./financial_tracker.db

# PostgreSQL (Producción)
DATABASE_URL=postgresql://user:password@localhost:5432/financial_tracker
# Con conexión SSL:
DATABASE_URL=postgresql://user:password@localhost:5432/financial_tracker?sslmode=require

# MySQL (Producción)
DATABASE_URL=mysql+pymysql://user:password@localhost:3306/financial_tracker

# Pool de conexiones (PostgreSQL/MySQL)
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10
DATABASE_POOL_TIMEOUT=30
```

---

#### Configuración de Alertas

```bash
# Umbrales de variación porcentual
ALERT_THRESHOLD_WARNING=3.0    # Alerta warning al ±3%
ALERT_THRESHOLD_CRITICAL=7.0   # Alerta crítica al ±7%

# Límite de alertas por usuario por día
ALERT_MAX_PER_USER_PER_DAY=50

# Tiempo mínimo entre alertas del mismo ticker (minutos)
ALERT_COOLDOWN_MINUTES=15
```

---

#### Configuración de Cache

```bash
# Tiempo de vida del caché de datos de mercado (segundos)
CACHE_MARKET_DATA_TTL=300       # 5 minutos (default)

# Tiempo de vida del caché de sentimiento (segundos)
CACHE_SENTIMENT_TTL=3600        # 1 hora (default)

# Tiempo de vida del caché de predicciones (segundos)
CACHE_PREDICTION_TTL=1800       # 30 minutos

# Backend de caché: memory, redis
CACHE_BACKEND=memory

# Configuración de Redis (si CACHE_BACKEND=redis)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
```

---

#### CORS y Networking

```bash
# Orígenes permitidos para CORS (separados por coma)
CORS_ORIGINS=http://localhost:8501,http://localhost:3000,https://yourdomain.com

# Permitir credenciales en CORS
CORS_ALLOW_CREDENTIALS=true

# Métodos HTTP permitidos
CORS_ALLOW_METHODS=GET,POST,PUT,DELETE

# Headers permitidos
CORS_ALLOW_HEADERS=*

# Puerto del servidor backend
PORT=8000

# Host del servidor (0.0.0.0 para acceso externo)
HOST=127.0.0.1
```

---

#### Logging y Debugging

```bash
# Nivel de log: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=INFO

# Archivo de log (opcional, si no se especifica solo stdout)
LOG_FILE=logs/app.log

# Formato de log: json, text
LOG_FORMAT=text

# Modo debug (activa logs detallados)
DEBUG=false

# Mostrar SQL queries en logs
SQL_ECHO=false
```

---

#### Configuración de Agentes ML

```bash
# ModelAgent: Número de días históricos para entrenamiento
MODEL_TRAINING_DAYS=504        # ~2 años de trading

# ModelAgent: Horizonte de predicción (días)
MODEL_PREDICTION_HORIZON=3

# ModelAgent: Validación temporal (porcentaje para test)
MODEL_TEST_SIZE=0.2

# SentimentAgent: Número de noticias a analizar
SENTIMENT_NEWS_COUNT=7

# SentimentAgent: Ponderación de modelos (suma debe ser 1.0)
SENTIMENT_FINBERT_WEIGHT=0.40
SENTIMENT_VADER_WEIGHT=0.25
SENTIMENT_TEXTBLOB_WEIGHT=0.15
SENTIMENT_LEXICON_WEIGHT=0.20

# RecommendationAgent: Ponderación de factores
RECOMMENDATION_TECHNICAL_WEIGHT=0.40
RECOMMENDATION_PREDICTION_WEIGHT=0.35
RECOMMENDATION_SENTIMENT_WEIGHT=0.15
RECOMMENDATION_RISK_WEIGHT=0.10
```

---

#### Notificaciones Email (Futuro)

```bash
# Servidor SMTP
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USE_TLS=true

# Credenciales
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Remitente
SMTP_FROM_NAME=Financial Tracker
SMTP_FROM_EMAIL=noreply@financialtracker.com

# Enviar alertas críticas por email
EMAIL_ALERTS_ENABLED=false
```

---

#### Rate Limiting

```bash
# Límite de requests por minuto por usuario
RATE_LIMIT_PER_MINUTE=60

# Límite de análisis de predicción por hora
RATE_LIMIT_PREDICT_PER_HOUR=20

# Límite de registros por IP por día
RATE_LIMIT_REGISTER_PER_DAY=5
```

---

## Configuración de Base de Datos

### SQLite (Desarrollo)

**Ventajas:**
- Sin configuración adicional
- Archivo único portable
- Perfecto para desarrollo y pruebas

**Limitaciones:**
- No recomendado para producción con múltiples usuarios
- Sin soporte para conexiones concurrentes pesadas

**Configuración:**
```bash
DATABASE_URL=sqlite:///./financial_tracker.db
```

---

### PostgreSQL (Producción Recomendada)

**Instalación (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
```

**Crear base de datos:**
```sql
CREATE DATABASE financial_tracker;
CREATE USER fintrack_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE financial_tracker TO fintrack_user;
```

**Configuración en .env:**
```bash
DATABASE_URL=postgresql://fintrack_user:secure_password@localhost:5432/financial_tracker
```

**Instalar driver Python:**
```bash
pip install psycopg2-binary
```

---

### MySQL (Alternativa)

**Instalación (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install mysql-server
sudo mysql_secure_installation
```

**Crear base de datos:**
```sql
CREATE DATABASE financial_tracker CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'fintrack_user'@'localhost' IDENTIFIED BY 'secure_password';
GRANT ALL PRIVILEGES ON financial_tracker.* TO 'fintrack_user'@'localhost';
FLUSH PRIVILEGES;
```

**Configuración en .env:**
```bash
DATABASE_URL=mysql+pymysql://fintrack_user:secure_password@localhost:3306/financial_tracker
```

**Instalar driver Python:**
```bash
pip install pymysql
```

---

## Configuración de Seguridad

### Generación de Claves Seguras

```python
# Script para generar múltiples claves
import secrets

print("SECRET_KEY:", secrets.token_urlsafe(32))
print("API_KEY:", secrets.token_hex(32))
print("ENCRYPTION_KEY:", secrets.token_urlsafe(32))
```

### Configuración de HTTPS (Producción)

**Con Nginx como reverse proxy:**

```nginx
# /etc/nginx/sites-available/financial-tracker
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**Obtener certificado SSL gratuito (Let's Encrypt):**
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

---

## Configuración de Agentes

### Personalización de ModelAgent

Editar `backend/agents/model_agent.py`:

```python
# Línea ~50: Cambiar modelos en el ensemble
self.models = {
    'linear': LinearRegression(),
    'ridge': Ridge(alpha=1.0),
    'rf': RandomForestRegressor(n_estimators=200, max_depth=10),
    'gb': GradientBoostingRegressor(n_estimators=150, learning_rate=0.05),
    # Agregar/quitar modelos según necesidad
}

# Línea ~120: Ajustar feature engineering
def create_features(self, df):
    # Agregar features personalizadas
    df['custom_indicator'] = df['Close'].rolling(window=10).mean()
    return df
```

### Personalización de SentimentAgent

Editar `backend/agents/sentiment_agent.py`:

```python
# Línea ~30: Ajustar pesos del ensemble
FINBERT_WEIGHT = 0.40
VADER_WEIGHT = 0.25
TEXTBLOB_WEIGHT = 0.15
LEXICON_WEIGHT = 0.20

# Línea ~200: Cambiar número de noticias
MAX_NEWS_TO_ANALYZE = 7  # Aumentar para más contexto
```

### Personalización de RecommendationAgent

Editar `backend/agents/recommendation_agent.py`:

```python
# Línea ~40: Ajustar ponderación de factores
TECHNICAL_WEIGHT = 0.40
PREDICTION_WEIGHT = 0.35
SENTIMENT_WEIGHT = 0.15
RISK_WEIGHT = 0.10

# Línea ~300: Modificar umbrales de recomendación
if score >= 8.0:
    return "compra_fuerte"
elif score >= 6.5:
    return "compra"
# ... personalizar umbrales
```

---

## Configuración de Cache

### Cache en Memoria (Default)

```python
# backend/config.py
CACHE_BACKEND = "memory"
CACHE_MARKET_DATA_TTL = 300  # 5 minutos
```

**Ventajas:** Simple, sin dependencias
**Desventajas:** No persiste entre reinicios, no compartido entre workers

---

### Cache con Redis (Producción)

**Instalación:**
```bash
sudo apt install redis-server
sudo systemctl start redis
```

**Configuración en .env:**
```bash
CACHE_BACKEND=redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

**Instalar cliente Python:**
```bash
pip install redis
```

**Implementación en código:**
```python
# backend/cache.py
import redis
from config import settings

redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    decode_responses=True
)

def get_cached(key):
    return redis_client.get(key)

def set_cached(key, value, ttl=300):
    redis_client.setex(key, ttl, value)
```

---

## Deployment en Producción

### Usando Docker

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      - db
      - redis

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: financial_tracker
      POSTGRES_USER: fintrack_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    depends_on:
      - backend

volumes:
  postgres_data:
```

**Ejecutar:**
```bash
docker-compose up -d
```

---

### Usando Systemd (Linux)

**Archivo de servicio del backend:**
```ini
# /etc/systemd/system/financial-tracker.service
[Unit]
Description=Financial Tracker API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/financial-tracker
Environment="PATH=/opt/financial-tracker/venv/bin"
ExecStart=/opt/financial-tracker/venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

**Activar:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable financial-tracker
sudo systemctl start financial-tracker
sudo systemctl status financial-tracker
```

---

## Optimización de Performance

### Gunicorn con Workers Múltiples

```bash
# Instalar
pip install gunicorn

# Ejecutar con 4 workers
gunicorn backend.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120
```

**Cálculo de workers óptimos:**
```
workers = (2 x CPU_cores) + 1
```

---

### Optimización de Base de Datos

**Índices recomendados (PostgreSQL/MySQL):**
```sql
-- Índice en ticker para búsquedas rápidas
CREATE INDEX idx_alertas_ticker ON alertas(ticker);

-- Índice compuesto para alertas por usuario
CREATE INDEX idx_alertas_usuario_fecha ON alertas(usuario_id, fecha_creacion DESC);

-- Índice para métricas
CREATE INDEX idx_metricas_ticker_fecha ON metricas_modelo(ticker, fecha DESC);
```

---

### Monitoring y Logging

**Prometheus + Grafana (recomendado):**

```bash
# Instalar prometheus-fastapi-instrumentator
pip install prometheus-fastapi-instrumentator

# En backend/main.py
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app)
```

**Sentry para error tracking:**
```bash
pip install sentry-sdk[fastapi]

# En backend/main.py
import sentry_sdk
sentry_sdk.init(dsn="your-sentry-dsn")
```

---

## Variables de Entorno por Ambiente

### Estructura recomendada

```
.env.development   # Desarrollo local
.env.staging       # Staging/QA
.env.production    # Producción
```

**Cargar según ambiente:**
```bash
# Desarrollo
export ENV=development
uvicorn backend.main:app --reload

# Producción
export ENV=production
uvicorn backend.main:app --host 0.0.0.0
```

---

## Backup y Recuperación

### Backup de SQLite

```bash
# Backup
sqlite3 financial_tracker.db ".backup 'backup.db'"

# Restaurar
cp backup.db financial_tracker.db
```

### Backup de PostgreSQL

```bash
# Backup
pg_dump -U fintrack_user financial_tracker > backup.sql

# Restaurar
psql -U fintrack_user financial_tracker < backup.sql
```

---

## Contacto y Soporte

Para preguntas sobre configuración avanzada, consulta la documentación principal en `README.md` o abre un issue en el repositorio.
