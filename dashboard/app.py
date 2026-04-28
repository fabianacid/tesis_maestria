"""
Dashboard Streamlit - Sistema Multiagente de Alertas Financieras

Interfaz profesional para análisis de activos financieros.
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Optional, Dict, Any

# Configuración de la página
st.set_page_config(
    page_title="Sistema de seguimiento y alertas para activos financieros",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #16213e;
        border-bottom: 2px solid #e8e8e8;
        padding-bottom: 8px;
        margin-bottom: 15px;
    }
    .card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 15px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a1a2e;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.05rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.82rem !important;
    }
    .positive { color: #28a745; }
    .negative { color: #dc3545; }
    .neutral { color: #6c757d; }
    .action-buy {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .action-sell {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .action-hold {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# URL del backend
API_URL = "http://localhost:8000"


def fmt_num(value, decimals=2):
    return f"{value:.{decimals}f}".replace(".", ",")


def fmt_price(value, decimals=2):
    return f"${fmt_num(value, decimals)}"


def fmt_pct(value, decimals=2, sign=False):
    s = "+" if sign and value >= 0 else ""
    return f"{s}{fmt_num(value, decimals)}%"


# ============================================
# Funciones de autenticación
# ============================================

def login(username: str, password: str) -> Optional[str]:
    """Realiza login y retorna token JWT."""
    try:
        response = requests.post(
            f"{API_URL}/auth/login",
            data={"username": username, "password": password}
        )
        if response.status_code == 200:
            return response.json()["access_token"]
        return None
    except Exception as e:
        st.error(f"Error de conexión: {str(e)}")
        return None


def register(username: str, email: str, password: str) -> bool:
    """Registra un nuevo usuario."""
    try:
        response = requests.post(
            f"{API_URL}/auth/register",
            json={"username": username, "email": email, "password": password}
        )
        return response.status_code == 201
    except Exception:
        return False


def forgot_password(email: str) -> tuple[bool, str]:
    """Solicita recuperación de contraseña."""
    try:
        response = requests.post(
            f"{API_URL}/auth/forgot-password",
            json={"email": email}
        )
        if response.status_code == 200:
            data = response.json()
            return True, data.get("message", "Email enviado")
        return False, "Error al procesar solicitud"
    except Exception as e:
        return False, f"Error de conexión: {str(e)}"


def reset_password(token: str, new_password: str) -> tuple[bool, str]:
    """Resetea contraseña con token."""
    try:
        response = requests.post(
            f"{API_URL}/auth/reset-password",
            json={"token": token, "new_password": new_password}
        )
        if response.status_code == 200:
            data = response.json()
            return True, data.get("message", "Contraseña actualizada")
        else:
            error_detail = response.json().get("detail", "Token inválido o expirado")
            return False, error_detail
    except Exception as e:
        return False, f"Error: {str(e)}"


def change_password(current_password: str, new_password: str) -> tuple[bool, str]:
    """Cambia contraseña del usuario autenticado."""
    try:
        response = requests.put(
            f"{API_URL}/auth/change-password",
            headers=get_headers(),
            json={"current_password": current_password, "new_password": new_password}
        )
        if response.status_code == 200:
            data = response.json()
            return True, data.get("message", "Contraseña actualizada")
        else:
            error_detail = response.json().get("detail", "Contraseña actual incorrecta")
            return False, error_detail
    except Exception as e:
        return False, f"Error: {str(e)}"


def get_headers() -> Dict[str, str]:
    """Retorna headers con token de autenticación."""
    if "token" in st.session_state:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}


# ============================================
# Funciones de API
# ============================================

def get_prediction(ticker: str) -> Optional[Dict[str, Any]]:
    """Obtiene análisis completo para un ticker."""
    try:
        # Obtener umbrales personalizados del session_state
        umbral_warning = st.session_state.get("umbral_warning", 3.0)
        umbral_critical = st.session_state.get("umbral_critical", 7.0)

        response = requests.get(
            f"{API_URL}/predict/{ticker}",
            headers=get_headers(),
            params={
                "umbral_warning": umbral_warning,
                "umbral_critical": umbral_critical
            }
        )
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            # Ticker no encontrado
            return None
        else:
            st.error(f"Error del servidor: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ No se pudo conectar al servidor backend. Verifica que esté corriendo en http://localhost:8000")
        return None
    except Exception as e:
        st.error(f"Error obteniendo datos: {str(e)}")
        return None


def get_alerts() -> list:
    """Obtiene lista de alertas del usuario."""
    try:
        response = requests.get(f"{API_URL}/alerts", headers=get_headers())
        if response.status_code == 200:
            return response.json().get("alertas", [])
        return []
    except Exception:
        return []


# ============================================
# Componentes de UI
# ============================================

def render_login_form():
    """Renderiza el formulario de login."""
    st.subheader("Iniciar Sesión")

    with st.form("login_form"):
        username = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Ingresar", use_container_width=True)

        if submitted:
            if username and password:
                token = login(username, password)
                if token:
                    st.session_state.token = token
                    st.session_state.username = username
                    st.success("Bienvenido")
                    st.rerun()
                else:
                    st.error("Usuario o contraseña incorrectos")
            else:
                st.warning("Complete todos los campos")

    # Link de recuperación de contraseña
    if st.button("¿Olvidaste tu contraseña?", use_container_width=True, type="secondary"):
        st.session_state.auth_view = "forgot_password"
        st.rerun()


def render_register_form():
    """Renderiza el formulario de registro."""
    st.subheader("Crear Cuenta")

    with st.form("register_form"):
        username = st.text_input("Usuario")
        email = st.text_input("Email")
        password = st.text_input("Contraseña", type="password")
        password2 = st.text_input("Confirmar Contraseña", type="password")
        submitted = st.form_submit_button("Registrarse", use_container_width=True)

        if submitted:
            if not all([username, email, password, password2]):
                st.warning("Complete todos los campos")
            elif password != password2:
                st.error("Las contraseñas no coinciden")
            elif len(password) < 6:
                st.error("La contraseña debe tener al menos 6 caracteres")
            else:
                if register(username, email, password):
                    st.success("Cuenta creada. Ahora puede iniciar sesión")
                else:
                    st.error("Error al crear cuenta. El usuario puede ya existir.")


def render_forgot_password_form():
    """Renderiza el formulario de recuperación de contraseña."""
    st.subheader("🔐 Recuperar Contraseña")
    st.markdown("Ingresa tu email y te enviaremos instrucciones para resetear tu contraseña.")

    with st.form("forgot_password_form"):
        email = st.text_input("Email registrado", placeholder="tu_email@example.com")
        submitted = st.form_submit_button("Enviar Instrucciones", use_container_width=True, type="primary")

        if submitted:
            if email:
                success, message = forgot_password(email)
                if success:
                    st.success(message)
                    st.info("Si tu email está registrado, recibirás un link de recuperación. Revisa tu bandeja de entrada y spam.")
                    st.info("⏰ El link expira en 1 hora por seguridad.")
                    st.warning("Si no configuraste SMTP, el token aparece en los logs del backend. Puedes usarlo manualmente.")
                else:
                    st.info(message)
            else:
                st.warning("Ingresa tu email")

    # Botones fuera del formulario
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Volver al login", use_container_width=True):
            st.session_state.auth_view = "login"
            st.rerun()

    with col2:
        if st.button("Tengo el token →", use_container_width=True, type="primary"):
            st.session_state.auth_view = "reset_password"
            st.rerun()


def render_reset_password_form():
    """Renderiza el formulario de reseteo de contraseña con token."""
    st.subheader("🔑 Resetear Contraseña")
    st.markdown("Ingresa el token que recibiste por email y tu nueva contraseña.")

    with st.form("reset_password_form"):
        token = st.text_input("Token de recuperación", placeholder="Token de 64 caracteres", help="Copia el token del email o de los logs del backend")
        new_password = st.text_input("Nueva contraseña", type="password", placeholder="Mínimo 6 caracteres")
        new_password2 = st.text_input("Confirmar nueva contraseña", type="password")
        submitted = st.form_submit_button("Actualizar Contraseña", use_container_width=True, type="primary")

        if submitted:
            if not all([token, new_password, new_password2]):
                st.warning("Complete todos los campos")
            elif new_password != new_password2:
                st.error("Las contraseñas no coinciden")
            elif len(new_password) < 6:
                st.error("La contraseña debe tener al menos 6 caracteres")
            else:
                success, message = reset_password(token, new_password)
                if success:
                    st.success(message)
                    st.success("✅ Contraseña actualizada exitosamente. Ya puedes iniciar sesión.")
                    st.balloons()
                    # Volver al login después de 2 segundos
                    import time
                    time.sleep(2)
                    st.session_state.auth_view = "login"
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
                    st.info("Verifica que el token sea correcto y no haya expirado (1 hora de validez).")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Volver al login", use_container_width=True):
            st.session_state.auth_view = "login"
            st.rerun()
    with col2:
        if st.button("Solicitar nuevo token", use_container_width=True):
            st.session_state.auth_view = "forgot_password"
            st.rerun()


def render_change_password_section():
    """Renderiza la sección de cambio de contraseña para usuarios autenticados."""
    st.subheader("🔒 Cambiar Contraseña")
    st.markdown("Actualiza tu contraseña de forma segura.")

    with st.form("change_password_form"):
        current_password = st.text_input("Contraseña actual", type="password")
        new_password = st.text_input("Nueva contraseña", type="password", placeholder="Mínimo 6 caracteres")
        new_password2 = st.text_input("Confirmar nueva contraseña", type="password")
        submitted = st.form_submit_button("Actualizar Contraseña", use_container_width=True, type="primary")

        if submitted:
            if not all([current_password, new_password, new_password2]):
                st.warning("Complete todos los campos")
            elif new_password != new_password2:
                st.error("Las contraseñas nuevas no coinciden")
            elif len(new_password) < 6:
                st.error("La contraseña debe tener al menos 6 caracteres")
            elif current_password == new_password:
                st.warning("La nueva contraseña debe ser diferente a la actual")
            else:
                success, message = change_password(current_password, new_password)
                if success:
                    st.success(message)
                    st.success("✅ Tu contraseña ha sido actualizada exitosamente.")
                    st.info("Por seguridad, tu sesión seguirá activa con el token actual.")
                else:
                    st.error(f"❌ {message}")


def render_auth_page():
    """Renderiza la página de autenticación."""
    st.markdown("<h1 class='main-header'>Sistema de seguimiento y alertas para activos financieros</h1>", unsafe_allow_html=True)
    st.markdown("Plataforma de análisis para inversores minoristas")
    st.markdown("---")

    # Inicializar vista de autenticación si no existe
    if "auth_view" not in st.session_state:
        st.session_state.auth_view = "login"

    # Renderizar según la vista seleccionada
    if st.session_state.auth_view == "forgot_password":
        render_forgot_password_form()
    elif st.session_state.auth_view == "reset_password":
        render_reset_password_form()
    else:
        # Vista por defecto: login y registro
        col1, col2 = st.columns(2)
        with col1:
            render_login_form()
        with col2:
            render_register_form()


def render_sidebar():
    """Renderiza el sidebar con opciones."""
    with st.sidebar:
        st.markdown("### Panel de Control")

        if "username" in st.session_state:
            st.write(f"Usuario: **{st.session_state.username}**")
            st.markdown("---")

        # Selector de ticker
        ticker = st.text_input(
            "Símbolo del Activo",
            value=st.session_state.get("ticker", "AAPL"),
            help="Ej: AAPL, GOOGL, MSFT, AMZN"
        ).upper()

        st.session_state.ticker = ticker

        if st.button("Analizar", type="primary", use_container_width=True):
            st.session_state.run_analysis = True

        st.markdown("---")
        st.markdown("**Accesos Rápidos**")

        popular = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META"]
        cols = st.columns(2)
        for i, t in enumerate(popular):
            if cols[i % 2].button(t, use_container_width=True, key=f"btn_{t}"):
                st.session_state.ticker = t
                st.session_state.run_analysis = True
                st.rerun()

        st.markdown("---")

        # Configuración de alertas
        st.markdown("**Configuración de Alertas**")

        umbral_warning = st.slider(
            "Umbral Advertencia (%)",
            min_value=0.5,
            max_value=10.0,
            value=st.session_state.get("umbral_warning", 3.0),
            step=0.5,
            help="Genera alerta de advertencia si la variación supera este porcentaje"
        )
        st.session_state.umbral_warning = umbral_warning

        umbral_critical = st.slider(
            "Umbral Crítico (%)",
            min_value=1.0,
            max_value=15.0,
            value=st.session_state.get("umbral_critical", 7.0),
            step=0.5,
            help="Genera alerta crítica si la variación supera este porcentaje"
        )
        st.session_state.umbral_critical = umbral_critical

        st.caption(f"Advertencia: >{umbral_warning}% | Crítico: >{umbral_critical}%")

        st.markdown("---")

        # Sección de configuración de cuenta
        st.markdown("**Configuración de Cuenta**")

        with st.expander("🔒 Cambiar Contraseña"):
            render_change_password_section()

        st.markdown("---")
        if st.button("Cerrar Sesión", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def render_market_metrics(mercado: Dict):
    """Renderiza métricas de mercado."""
    st.markdown("<p class='section-header'>Datos de Mercado</p>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        variacion = mercado['variacion_diaria']
        delta_color = "normal" if variacion >= 0 else "inverse"
        st.metric("Precio Actual", fmt_price(mercado['ultimo_precio']), fmt_pct(variacion), delta_color=delta_color)

    with col2:
        st.metric("Precio Anterior", fmt_price(mercado['precio_anterior']))

    with col3:
        st.metric("Media Móvil (20)", fmt_price(mercado['media_movil_20']))

    with col4:
        senal = mercado['senal']
        st.metric(
            "Señal Técnica",
            senal.upper(),
            help=(
                "¿Qué significa en la práctica?\n\n"
                "ALCISTA → La mayoría de los indicadores técnicos coinciden en que el precio tiene momentum positivo: "
                "está por encima de sus promedios históricos, el volumen acompaña la suba y no hay señales de sobrecompra extrema. "
                "Es una señal de que la tendencia de corto/mediano plazo favorece al comprador.\n\n"
                "BAJISTA → Lo opuesto: el precio cae por debajo de sus promedios, el volumen confirma la baja o hay señales de debilidad.\n\n"
                "NEUTRAL → Los indicadores se contradicen entre sí o el movimiento es demasiado pequeño para definir una dirección clara.\n\n"
                "Cómo se calcula:\n"
                "Se combinan 15+ indicadores en 4 grupos ponderados:\n"
                "• Tendencia (35%): medias móviles, MACD, ADX\n"
                "• Momentum (30%): RSI, Estocástico, Williams %R\n"
                "• Volatilidad (15%): Bollinger Bands, ATR\n"
                "• Volumen (20%): OBV, MFI, VWAP\n\n"
                "Score ≥ 0,1 → ALCISTA | Score ≤ -0,1 → BAJISTA | Entre ambos → NEUTRAL"
            )
        )


def render_prediction_card(prediccion: Dict, precio_actual: float = 0):
    """Renderiza tarjeta de predicción."""
    st.markdown("<p class='section-header'>Predicción del Modelo</p>", unsafe_allow_html=True)

    variacion = prediccion['variacion_pct']
    precio_predicho = prediccion['precio_predicho']

    # Determinar dirección (zona neutra: |variacion| <= 0.5% = ruido de mercado)
    if variacion > 2:
        direccion = "ALZA SIGNIFICATIVA"
        color_class = "positive"
    elif variacion > 0.5:
        direccion = "Alza moderada"
        color_class = "positive"
    elif variacion < -2:
        direccion = "BAJA SIGNIFICATIVA"
        color_class = "negative"
    elif variacion < -0.5:
        direccion = "Baja moderada"
        color_class = "negative"
    else:
        direccion = "Estable"
        color_class = "neutral"

    # Usar probabilidad real del modelo si está disponible
    prob_subida_real = prediccion.get('prob_subida')
    if prob_subida_real is not None:
        prob_subida = prob_subida_real * 100
    else:
        prob_subida = 50 + (variacion / 2)
    prob_subida = max(0, min(100, prob_subida))
    prob_bajada = 100 - prob_subida

    if variacion > 0:
        dir_label = "⬆️ SUBIDA"
        prob_display = prob_subida
    else:
        dir_label = "⬇️ BAJADA"
        prob_display = prob_bajada

    st.markdown(f"""
        <div class='card'>
            <div class='metric-label'>Predicción del Modelo (Horizonte: 3 días)</div>
            <div class='metric-value {color_class}' style='margin-bottom:15px'>{direccion}</div>
            <div style='display:flex;gap:40px;flex-wrap:wrap;margin-bottom:15px'>
                <div>
                    <div class='metric-label'>Precio Actual</div>
                    <div style='font-size:1.3rem;font-weight:600'>{fmt_price(precio_actual)}</div>
                </div>
                <div>
                    <div class='metric-label'>Precio Proyectado (3 días)</div>
                    <div class='metric-value {color_class}'>{fmt_price(precio_predicho)}</div>
                </div>
                <div>
                    <div class='metric-label'>Variación Esperada</div>
                    <div class='metric-value {color_class}'>{fmt_pct(variacion, sign=True)}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    precision_val = prediccion.get('metricas', {}).get('precision', 0) * 100
    accuracy_val = prediccion.get('metricas', {}).get('accuracy', 0) * 100
    st.metric(
        "Probabilidad de dirección",
        f"{dir_label}: {fmt_num(prob_display, 1)}%",
        help=(
            "Cada modelo genera una probabilidad de subida con predict_proba().\n\n"
            "Resultado = promedio ponderado:\n"
            "P(subida) = Σ ( prob_modelo × peso_modelo )\n\n"
            "El peso de cada modelo es proporcional a su accuracy en validación cruzada temporal (5 folds).\n\n"
            "Modelos: Random Forest · Gradient Boosting · XGBoost · LightGBM\n\n"
            "P(bajada) = 1 − P(subida)\n\n"
            "⚠️ Esta probabilidad refleja la CONFIANZA del modelo, no su tasa de acierto histórica.\n"
            f"Precisión histórica: {fmt_num(precision_val, 1)}% | Accuracy: {fmt_num(accuracy_val, 1)}%"
        )
    )
    st.caption(
        f"⚠️ Precisión histórica: {fmt_num(precision_val, 1)}% | Accuracy: {fmt_num(accuracy_val, 1)}%"
        " — La probabilidad refleja la confianza del modelo, no su tasa de acierto histórica."
    )

    # Detalles de modelos
    with st.expander("Ver detalles de los modelos de IA"):
        modelos_detalle = prediccion.get('modelos_detalle', {})
        predicciones = modelos_detalle.get('predicciones', {})
        pesos = modelos_detalle.get('pesos', {})

        if predicciones:
            nombres_modelos = {
                "ridge": "Ridge Regression",
                "xgboost": "XGBoost",
                "lightgbm": "LightGBM",
                "gradient_boosting": "Gradient Boosting",
                "random_forest": "Random Forest",
                "linear_regression": "Regresión Lineal"
            }

            st.markdown("**Predicción por modelo:**")

            # Crear DataFrame para mostrar
            data = []
            for modelo, pred in predicciones.items():
                peso = pesos.get(modelo, 0) * 100
                nombre = nombres_modelos.get(modelo, modelo)
                data.append({"Modelo": nombre, "P(Subida)": f"{fmt_num(pred * 100, 0)}%", "Peso": f"{fmt_num(peso, 1)}%"})

            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            mejor = prediccion.get('parametros', {}).get('mejor_modelo', 'N/A')
            st.info(
                f"Modelo con mejor rendimiento: **{nombres_modelos.get(mejor, mejor)}**  \n"
                "ℹ️ Se elige el modelo con mayor **accuracy** promedio en validación cruzada temporal (5 folds). "
                "Ese accuracy determina también su peso en el ensemble: mayor accuracy = mayor influencia en la probabilidad final. "
                "No necesariamente es el que predice con mayor confianza hoy, sino el que históricamente acertó más veces."
            )

        # Métricas de clasificación (predicción de dirección)
        st.markdown("---")
        st.markdown("**Métricas del Modelo (Clasificación):**")
        st.caption("El modelo predice si el precio SUBE ⬆️ o BAJA ⬇️")
        metricas = prediccion.get('metricas', {})
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            accuracy = metricas.get('accuracy', 0) * 100
            st.metric("Accuracy", f"{fmt_num(accuracy, 1)}%", help="De todas las predicciones (SUBIDA y BAJADA), % de veces que el modelo acertó. Un modelo aleatorio obtendría ~50%.")
        with col2:
            precision = metricas.get('precision', 0) * 100
            st.metric("Precisión", f"{fmt_num(precision, 1)}%", help="De las veces que el modelo predijo SUBIDA, % en que realmente subió. Baja precisión = muchas falsas alarmas de subida.")
        with col3:
            recall = metricas.get('recall', 0) * 100
            st.metric("Recall", f"{fmt_num(recall, 1)}%", help="De todas las subidas reales que ocurrieron, % que el modelo logró detectar. Bajo recall = el modelo se pierde muchas subidas.")
        with col4:
            f1 = metricas.get('f1', 0) * 100
            st.metric("F1-Score", f"{fmt_num(f1, 1)}%", help="Media armónica de Precisión y Recall: 2×(P×R)/(P+R). Penaliza fuertemente cuando uno de los dos es bajo. Valor mostrado: promedio de los 5 folds de validación cruzada temporal.")


def render_sentiment_card(sentimiento: Dict):
    """Renderiza tarjeta de sentimiento."""
    st.markdown("<p class='section-header'>Análisis de Sentimiento</p>", unsafe_allow_html=True)

    sent = sentimiento['sentimiento']
    color_class = "positive" if sent == "positivo" else "negative" if sent == "negativo" else "neutral"

    explicacion = sentimiento.get('explicacion_simple', f'El sentimiento del mercado es {sent}')
    # Limpiar emojis de la explicación
    explicacion = explicacion.replace('📈', '').replace('📉', '').replace('➡️', '').replace('😊', '').replace('😟', '').replace('😐', '').strip()

    que_significa = sentimiento.get('que_significa', '')
    que_significa = que_significa.replace('⚠️', '').replace('💡', '').strip()

    st.markdown(f"""
        <div class='card'>
            <div class='metric-label'>Sentimiento del Mercado</div>
            <div class='metric-value {color_class}' style='margin-bottom:15px'>{sent.upper()}</div>
            <p style='color:#444;margin-bottom:10px'>{explicacion}</p>
            <p style='color:#666;font-size:0.9rem'><strong>Interpretación:</strong> {que_significa}</p>
        </div>
    """, unsafe_allow_html=True)

    # Detalles técnicos
    with st.expander("Ver metodología de análisis"):
        como_se_calcula = sentimiento.get('como_se_calcula', '')
        # Limpiar emojis
        como_se_calcula = como_se_calcula.replace('📊', '').strip()

        if como_se_calcula:
            st.markdown(como_se_calcula)

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Score",
                f"{fmt_num(sentimiento['score'], 4)}",
                help=(
                    "Rango: -1 (muy negativo) a +1 (muy positivo).\n\n"
                    "Paso 1 — Cada noticia es analizada por 4 modelos:\n"
                    "  · FinBERT  × 0.40  (IA especializada en finanzas)\n"
                    "  · VADER    × 0.25  (léxico de palabras positivas/negativas)\n"
                    "  · Lexicon  × 0.20  (diccionario de términos bursátiles)\n"
                    "  · TextBlob × 0.15  (polaridad general del texto)\n"
                    "  → score_noticia = Σ(score_modelo × peso) / Σ(pesos activos)\n\n"
                    "Paso 2 — Cada noticia tiene un peso de relevancia según\n"
                    "  cuántas veces menciona el ticker y su contexto financiero.\n"
                    "  → score_final = Σ(score_noticia × relevancia) / Σ(relevancias)\n\n"
                    "Clasificación del score final:\n"
                    "  ≥ 0.60  → Muy positivo\n"
                    "  ≥ 0.25  → Positivo\n"
                    "  ≥ 0.10  → Ligeramente positivo\n"
                    " -0.10 a 0.10 → Neutral\n"
                    "  ≤ -0.10 → Ligeramente negativo\n"
                    "  ≤ -0.25 → Negativo\n"
                    "  ≤ -0.60 → Muy negativo"
                )
            )
        with col2:
            st.metric(
                "Confianza",
                f"{fmt_num(sentimiento['confianza'] * 100, 1)}%",
                help=(
                    "Promedio ponderado de la confianza individual de cada modelo:\n"
                    "  Σ(confianza_modelo × peso_modelo) / Σ(pesos)\n\n"
                    "Cada modelo devuelve su propia certeza (ej: FinBERT da probabilidad 0-1).\n"
                    "Los modelos con mayor peso (FinBERT × 0.40) influyen más en el resultado.\n\n"
                    "Ajuste por noticias reales: +0.30 (techo en 95%) cuando se analizan\n"
                    "noticias reales de Yahoo Finance, por mayor calidad de los datos.\n\n"
                    "95% = alta certeza de los modelos + noticias reales disponibles."
                )
            )


def render_recommendation_card(recomendacion: Dict):
    """Renderiza tarjeta de recomendación."""
    st.markdown("<p class='section-header'>Recomendación</p>", unsafe_allow_html=True)

    tipo = recomendacion['tipo']

    # Determinar clase CSS
    if tipo == "compra":
        action_class = "action-buy"
        action_text = "COMPRA"
    elif tipo == "venta":
        action_class = "action-sell"
        action_text = "VENTA"
    else:
        action_class = "action-hold"
        action_text = "MANTENER"

    accion_sugerida = recomendacion.get('accion_sugerida', action_text)
    # Limpiar emojis
    accion_sugerida = accion_sugerida.replace('🟢', '').replace('🔴', '').replace('🟡', '').strip()

    explicacion = recomendacion.get('explicacion_simple', '')
    nivel_riesgo = recomendacion.get('nivel_riesgo_simple', '')
    # Limpiar emojis
    nivel_riesgo = nivel_riesgo.replace('🛡️', '').strip()

    st.markdown(f"""
        <div class='card {action_class}' style='border-radius:12px;padding:25px'>
            <div style='font-size:1.8rem;font-weight:700;margin-bottom:15px'>{accion_sugerida}</div>
            <p style='font-size:1rem;line-height:1.6;margin-bottom:15px'>{explicacion}</p>
            <div style='background:rgba(255,255,255,0.5);padding:12px;border-radius:8px'>
                <strong>{nivel_riesgo}</strong>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.caption("_Herramienta de apoyo a la decisión, no como reemplazo del juicio del inversor._")

    # Detalles del análisis
    with st.expander("Ver análisis detallado"):
        porque = recomendacion.get('porque_esta_recomendacion', '')
        # Limpiar emojis
        porque = porque.replace('📊', '').replace('✅', '').replace('❌', '').replace('🎯', '')
        porque = porque.replace('📈', '').replace('📉', '').replace('➡️', '')
        porque = porque.replace('🤖', '').replace('😊', '').replace('😟', '').replace('😐', '')
        porque = porque.replace('🛡️', '').replace('⚠️', '').replace('1️⃣', '1.').replace('2️⃣', '2.').replace('3️⃣', '3.').replace('4️⃣', '4.')

        if porque:
            st.markdown(porque)

        st.markdown("---")
        st.markdown("**Factores considerados:**")
        factores = recomendacion.get('factores', {})
        composite = factores.get('composite_score', None)
        risk_lvl = factores.get('risk_level', 'N/A')
        prob = factores.get('prob_profit', 0) * 100

        # Fila 1: señal técnica + variación esperada
        col1, col2 = st.columns(2)
        with col1:
            senal_raw = factores.get('senal_mercado', 'N/A')
            senal_display = senal_raw.capitalize()
            st.metric(
                "Señal Técnica",
                senal_display,
                help=(
                    "¿Qué significa en la práctica?\n\n"
                    "ALCISTA → La mayoría de los indicadores técnicos coinciden en que el precio tiene momentum positivo: "
                    "está por encima de sus promedios históricos, el volumen acompaña la suba y no hay señales de sobrecompra extrema.\n\n"
                    "BAJISTA → El precio cae por debajo de sus promedios, el volumen confirma la baja o hay señales de debilidad.\n\n"
                    "NEUTRAL → Los indicadores se contradicen entre sí o el movimiento es demasiado pequeño para definir dirección.\n\n"
                    "Peso en la recomendación final: 40%"
                )
            )
        with col2:
            st.metric(
                "Variación Esperada",
                fmt_pct(factores.get('variacion_pct', 0)),
                help=(
                    "Cambio porcentual predicho por el modelo ML para los próximos 3 días.\n\n"
                    "Peso en la recomendación: 35%\n\n"
                    "Ensemble de 4 modelos (cada uno vota con su propia predicción):\n"
                    "Random Forest + Gradient Boosting + XGBoost + LightGBM\n\n"
                    "Negativo = bajada esperada  |  Positivo = subida esperada"
                )
            )

        # Fila 2: probabilidad de ganancia + score compuesto
        col3, col4 = st.columns(2)
        with col3:
            if prob >= 50:
                prob_label = f"{fmt_num(prob, 0)}% (favorable)"
            else:
                prob_label = f"{fmt_num(prob, 0)}% (desfavorable)"
            st.metric(
                "Prob. Ganancia",
                prob_label,
                help=(
                    "Probabilidad estimada de que la operación resulte en ganancia.\n\n"
                    "Fórmula:\n"
                    "P = 0.5 + (score × 0.3) + (confianza_ML × 0.1) − (riesgo × 0.1)\n"
                    "Acotada entre 20% y 80% para evitar extremos irreales.\n\n"
                    "Variables:\n"
                    "• score: puntuación compuesta del análisis [-1, +1]\n"
                    "• confianza_ML: certeza del modelo de machine learning [0, 1]\n"
                    "• riesgo: score de riesgo global (volatilidad + VaR + régimen) [0, 1]\n\n"
                    "Interpretación:\n"
                    "> 50% → la operación tiene más probabilidades de ganar que de perder\n"
                    "< 50% → la operación tiene más probabilidades de perder que de ganar"
                )
            )
        with col4:
            if composite is not None:
                if composite > 0.3:
                    score_label = f"{fmt_num(composite, 2)} (muy favorable)"
                elif composite > 0.1:
                    score_label = f"{fmt_num(composite, 2)} (favorable)"
                elif composite > -0.1:
                    score_label = f"{fmt_num(composite, 2)} (neutral)"
                elif composite > -0.3:
                    score_label = f"{fmt_num(composite, 2)} (levemente desfavorable)"
                else:
                    score_label = f"{fmt_num(composite, 2)} (desfavorable)"
                st.metric(
                    "Score Compuesto",
                    score_label,
                    help=(
                        "Puntuación final [-1, +1] que combina los 4 grupos de factores ponderados:\n\n"
                        "score = Σ (valor_factor × peso_factor)\n\n"
                        "Grupos y pesos:\n"
                        "• Técnico (40%): tendencia, momentum, volatilidad, volumen, soporte/resistencia\n"
                        "• Predicción ML (35%): señal del modelo, confianza, acuerdo entre modelos\n"
                        "• Sentimiento (15%): score NLP, tendencia, impacto de noticias\n"
                        "• Riesgo (10%): retorno ajustado, régimen de mercado, correlación\n\n"
                        "Umbrales de decisión:\n"
                        "> +0.50 → Compra fuerte\n"
                        "+0.20 a +0.50 → Compra\n"
                        "+0.05 a +0.20 → Compra débil\n"
                        "-0.05 a +0.05 → Mantener\n"
                        "-0.20 a -0.05 → Venta débil\n"
                        "-0.50 a -0.20 → Venta\n"
                        "< -0.50 → Venta fuerte\n\n"
                        f"Nivel de riesgo actual: {risk_lvl}"
                    )
                )


def render_alert_card(alerta: Dict):
    """Renderiza tarjeta de alerta con los 6 niveles de severidad."""
    SEVERITY_CONFIG = {
        "emergency": {
            "label": "EMERGENCIA",
            "bg": "#4a0010", "border": "#ff0033", "color": "#ffffff",
            "icon": "🚨"
        },
        "critical": {
            "label": "ALERTA CRÍTICA",
            "bg": "#fff0f0", "border": "#dc3545", "color": "#721c24",
            "icon": "🔴"
        },
        "high": {
            "label": "ALERTA ALTA",
            "bg": "#fff4e5", "border": "#e8700a", "color": "#7a3600",
            "icon": "🟠"
        },
        "medium": {
            "label": "ADVERTENCIA",
            "bg": "#fffbe6", "border": "#ffc107", "color": "#856404",
            "icon": "🟡"
        },
        "low": {
            "label": "AVISO",
            "bg": "#e8f4fd", "border": "#17a2b8", "color": "#0c5460",
            "icon": "🔵"
        },
        "info": {
            "label": "INFORMACIÓN",
            "bg": "#f0f9f0", "border": "#28a745", "color": "#155724",
            "icon": "🟢"
        },
    }

    if alerta['tiene_alerta']:
        nivel = alerta.get('nivel', 'info')
        cfg = SEVERITY_CONFIG.get(nivel, SEVERITY_CONFIG['info'])
        st.markdown(
            f"""<div style="background:{cfg['bg']};border-left:5px solid {cfg['border']};
            border-radius:6px;padding:12px 16px;margin-bottom:8px;">
            <span style="font-weight:700;color:{cfg['color']};">
            {cfg['icon']} {cfg['label']}</span>
            <span style="color:{cfg['color']};margin-left:8px;">{alerta['mensaje']}</span>
            </div>""",
            unsafe_allow_html=True
        )
    else:
        st.success("Sin alertas activas para este activo")


def render_price_chart(ticker: str, prediction_data: Dict):
    """Renderiza gráfico de precios con datos históricos reales."""
    mercado = prediction_data['mercado']
    prediccion = prediction_data['prediccion']

    precios_historicos = mercado.get('precios_recientes', [])
    fechas_historicas = mercado.get('fechas_recientes', [])

    if precios_historicos and fechas_historicas:
        dates = pd.to_datetime(fechas_historicas)
        prices = precios_historicos
    else:
        # Fallback: línea plana en precio actual si no hay datos
        dates = pd.date_range(end=datetime.now(), periods=30, freq='B')
        prices = [mercado['ultimo_precio']] * 30

    fig = go.Figure()

    # Línea de precios
    fig.add_trace(go.Scatter(
        x=dates, y=prices,
        mode='lines',
        name='Precio',
        line=dict(color='#2563eb', width=2)
    ))

    # Media móvil
    ma_values = pd.Series(prices).rolling(window=20, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=dates, y=ma_values,
        mode='lines',
        name='MA(20)',
        line=dict(color='#f59e0b', width=1, dash='dash')
    ))

    # Proyección de 3 días (horizonte de predicción)
    # Calcular trayectoria lineal desde precio actual hasta precio predicho en 3 días
    precio_actual = mercado['ultimo_precio']
    precio_predicho = prediccion['precio_predicho']

    # Crear puntos para días 1, 2 y 3
    future_dates = [dates[-1] + pd.Timedelta(days=i) for i in range(1, 4)]
    step = (precio_predicho - precio_actual) / 3
    future_prices = [precio_actual + step * i for i in range(1, 4)]

    # Línea de proyección
    fig.add_trace(go.Scatter(
        x=[dates[-1]] + future_dates,
        y=[precio_actual] + future_prices,
        mode='lines+markers',
        name='Proyección 3 días',
        line=dict(color='#10b981', width=2, dash='dot'),
        marker=dict(color='#10b981', size=8)
    ))

    # Resaltar el punto final (día 3)
    fig.add_trace(go.Scatter(
        x=[future_dates[-1]],
        y=[precio_predicho],
        mode='markers',
        name='Objetivo día 3',
        marker=dict(color='#10b981', size=14, symbol='diamond', line=dict(color='white', width=2)),
        showlegend=False
    ))

    fig.update_layout(
        title=dict(text=f"Evolución de Precio - {ticker}", font=dict(size=16)),
        xaxis_title="Fecha",
        yaxis_title="Precio (USD)",
        hovermode='x unified',
        showlegend=True,
        height=350,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_xaxes(gridcolor='#e5e5e5')
    fig.update_yaxes(gridcolor='#e5e5e5')

    st.plotly_chart(fig, use_container_width=True)


TIPO_ALERTA_LABELS = {
    "movimiento_precio": "Movimiento de Precio",
    "pico_volatilidad": "Pico de Volatilidad",
    "cambio_tendencia": "Cambio de Tendencia",
    "anomalia_detectada": "Anomalía Detectada",
    "volumen_inusual": "Volumen Inusual",
    "cambio_sentimiento": "Cambio de Sentimiento",
    "divergencia_prediccion": "Divergencia de Predicción",
    "ruptura_correlacion": "Ruptura de Correlación",
    "patron_detectado": "Patrón Detectado",
}


def render_alerts_history():
    """Renderiza historial de alertas con filtros por tipo."""
    st.markdown("<p class='section-header'>Historial de Alertas</p>", unsafe_allow_html=True)

    alerts = get_alerts()

    if not alerts:
        st.info("No hay alertas registradas")
        return

    df = pd.DataFrame(alerts)

    if 'fecha_creacion' in df.columns:
        df['fecha_creacion'] = pd.to_datetime(df['fecha_creacion']).dt.strftime('%Y-%m-%d %H:%M')

    # --- Filtros ---
    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])

    with col_f1:
        tipos_disponibles = sorted(df['tipo'].dropna().unique().tolist())
        etiquetas = [TIPO_ALERTA_LABELS.get(t, t) for t in tipos_disponibles]
        tipo_seleccionados = st.multiselect(
            "Filtrar por tipo de alerta",
            options=tipos_disponibles,
            default=tipos_disponibles,
            format_func=lambda t: TIPO_ALERTA_LABELS.get(t, t),
            key="filter_tipo"
        )

    with col_f2:
        tickers_disponibles = sorted(df['ticker'].dropna().unique().tolist())
        ticker_sel = st.multiselect(
            "Filtrar por activo",
            options=tickers_disponibles,
            default=tickers_disponibles,
            key="filter_ticker_hist"
        )

    with col_f3:
        solo_no_leidas = st.checkbox("Solo no leídas", value=False, key="filter_leidas")

    # Aplicar filtros
    df_filtrado = df[df['tipo'].isin(tipo_seleccionados) & df['ticker'].isin(ticker_sel)]
    if solo_no_leidas:
        df_filtrado = df_filtrado[df_filtrado['leida'] == False]

    st.caption(f"Mostrando {len(df_filtrado)} de {len(df)} alertas")

    # Tabla con etiquetas legibles para el tipo
    df_display = df_filtrado.copy()
    df_display['tipo'] = df_display['tipo'].map(lambda t: TIPO_ALERTA_LABELS.get(t, t))
    df_display['leida'] = df_display['leida'].map({True: "Leída", False: "Pendiente"})

    st.dataframe(
        df_display[['ticker', 'tipo', 'mensaje', 'variacion_pct', 'leida', 'fecha_creacion']],
        use_container_width=True,
        hide_index=True,
        column_config={
            "ticker": st.column_config.TextColumn("Activo"),
            "tipo": st.column_config.TextColumn("Tipo de Alerta"),
            "mensaje": st.column_config.TextColumn("Mensaje", width="large"),
            "variacion_pct": st.column_config.NumberColumn("Variación %", format="%.2f%%"),
            "leida": st.column_config.TextColumn("Estado"),
            "fecha_creacion": st.column_config.TextColumn("Fecha"),
        }
    )

    if len(df_filtrado) > 0:
        col1, col2 = st.columns(2)

        with col1:
            tipo_counts = df_filtrado['tipo'].map(lambda t: TIPO_ALERTA_LABELS.get(t, t)).value_counts()
            fig = px.pie(
                values=tipo_counts.values,
                names=tipo_counts.index,
                title="Distribución por Tipo de Alerta",
                color_discrete_sequence=[
                    '#ef4444', '#f59e0b', '#3b82f6', '#10b981',
                    '#8b5cf6', '#ec4899', '#14b8a6', '#f97316', '#6366f1'
                ]
            )
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            ticker_counts = df_filtrado['ticker'].value_counts().head(5)
            fig = px.bar(
                x=ticker_counts.index,
                y=ticker_counts.values,
                title="Alertas por Activo",
                color=ticker_counts.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)


def render_main_dashboard():
    """Renderiza el dashboard principal."""
    render_sidebar()

    # Logo y header profesional
    st.markdown("""
    <div style='text-align: center; padding: 20px 0; margin-bottom: 30px;'>
        <div style='display: inline-block; background: linear-gradient(135deg, #87ceeb 0%, #4db8e8 100%);
                    padding: 15px 30px; border-radius: 15px; box-shadow: 0 4px 15px rgba(77, 184, 232, 0.4);'>
            <h1 style='color: white; margin: 0; font-size: 2.5rem; font-weight: 700; letter-spacing: 2px;'>
                Sistema Multi Agente
            </h1>
            <p style='color: #f0f0f0; margin: 5px 0 0 0; font-size: 0.95rem; letter-spacing: 1px;'>
                Seguimiento y alerta de activos financieros.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Análisis de Activos", "Historial de Alertas"])

    with tab1:
        if st.session_state.get("run_analysis", False):
            ticker = st.session_state.get("ticker", "AAPL")

            with st.spinner(f"Analizando {ticker}..."):
                data = get_prediction(ticker)

            st.session_state.run_analysis = False

            if data:
                st.session_state.last_analysis = data
            else:
                st.error(f"❌ Ticker '{ticker}' no encontrado")
                st.warning(f"""
                **El ticker '{ticker}' no existe en Yahoo Finance o no tiene datos disponibles.**

                Por favor verifica:
                - ✅ Que el símbolo esté escrito correctamente
                - ✅ Que sea un ticker válido (ejemplos: AAPL, GOOGL, MSFT, TSLA)
                - ✅ Que el activo esté listado en mercados financieros

                **Tickers populares que puedes probar:**
                - **Tech**: AAPL, GOOGL, MSFT, AMZN, TSLA, META, NVDA
                - **Finance**: JPM, BAC, WFC, GS
                - **Energía**: XOM, CVX, COP
                """)
                st.info("💡 Usa los botones de 'Accesos Rápidos' en el panel lateral para analizar tickers populares.")

        if "last_analysis" in st.session_state:
            data = st.session_state.last_analysis

            st.markdown(f"### {data['ticker']}")
            st.caption(f"Última actualización: {data['fecha_analisis'][:19].replace('T', ' ')}")

            render_alert_card(data['alerta'])

            st.markdown("---")
            render_price_chart(data['ticker'], data)

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                render_market_metrics(data['mercado'])
                st.markdown("")
                render_prediction_card(data['prediccion'], data['mercado']['ultimo_precio'])

            with col2:
                render_sentiment_card(data['sentimiento'])
                st.markdown("")
                render_recommendation_card(data['recomendacion'])

        else:
            st.info("Seleccione un activo en el panel lateral y presione 'Analizar' para comenzar")

    with tab2:
        if "token" in st.session_state:
            render_alerts_history()
        else:
            st.warning("Inicie sesión para ver el historial de alertas")


def main():
    """Función principal de la aplicación."""
    if "token" not in st.session_state:
        render_auth_page()
    else:
        render_main_dashboard()


if __name__ == "__main__":
    main()
