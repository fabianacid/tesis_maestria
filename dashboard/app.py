"""
Dashboard Streamlit - Sistema Multiagente de Análisis y Optimización de Portafolios Financieros

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
    page_title="Sistema Multiagente de Análisis y Optimización de Portafolios Financieros",
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


def analyze_portfolio(tickers: list, weights: list) -> Optional[Dict[str, Any]]:
    """Envía solicitud de análisis de portafolio al backend."""
    try:
        response = requests.post(
            f"{API_URL}/portfolio/analyze",
            headers=get_headers(),
            json={"tickers": tickers, "weights": weights},
            timeout=180,
        )
        if response.status_code == 200:
            return response.json()
        else:
            detail = response.json().get("detail", f"Error {response.status_code}")
            st.error(f"Error del servidor: {detail}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("No se pudo conectar al backend. Verifica que esté corriendo en http://localhost:8000")
        return None
    except requests.exceptions.Timeout:
        st.error("El análisis tardó demasiado. Intenta con menos activos o espera un momento.")
        return None
    except Exception as exc:
        st.error(f"Error: {str(exc)}")
        return None


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
    st.markdown("<h1 class='main-header'>Sistema Multiagente de Análisis y Optimización de Portafolios Financieros</h1>", unsafe_allow_html=True)
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
    # Barra visual de probabilidad
    st.markdown(f"""
    <div style='margin:12px 0 4px 0'>
        <div style='font-size:0.82rem;color:#666;margin-bottom:6px'>
            Probabilidad de dirección
            <span style='font-size:0.78rem;margin-left:6px;opacity:0.7'>
                (confianza del modelo — no es la tasa de acierto histórica)
            </span>
        </div>
        <div style='display:flex;border-radius:6px;overflow:hidden;height:28px'>
            <div style='width:{prob_subida:.0f}%;background:#28a745;display:flex;align-items:center;
                        justify-content:center;color:white;font-size:0.85rem;font-weight:600'>
                ⬆️ {prob_subida:.0f}%
            </div>
            <div style='width:{prob_bajada:.0f}%;background:#dc3545;display:flex;align-items:center;
                        justify-content:center;color:white;font-size:0.85rem;font-weight:600'>
                ⬇️ {prob_bajada:.0f}%
            </div>
        </div>
        <div style='font-size:0.78rem;color:#888;margin-top:4px'>
            Tasa de acierto histórica del modelo: {fmt_num(accuracy_val, 1)}%
            (un modelo aleatorio obtendría 50%)
        </div>
    </div>
    """, unsafe_allow_html=True)

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
                "El peso de cada modelo en el ensemble se calcula por **AUC-ROC** (capacidad discriminativa), "
                "que es más robusto que accuracy cuando las clases están desbalanceadas. "
                "No necesariamente es el que predice con mayor confianza hoy, sino el que históricamente mejor discriminó subidas de bajadas."
            )

        # Métricas de clasificación (predicción de dirección)
        st.markdown("---")
        st.markdown("**¿Qué tan confiable es el modelo?**")
        st.caption("Comparación vs un modelo aleatorio (50% de acierto por azar). Valores calculados con validación cruzada temporal (5 períodos).")
        metricas = prediccion.get('metricas', {})
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            accuracy = metricas.get('accuracy', 0) * 100
            delta_acc = accuracy - 50
            st.metric(
                "Tasa de acierto",
                f"{fmt_num(accuracy, 1)}%",
                delta=f"{delta_acc:+.1f}pp vs azar",
                delta_color="normal" if delta_acc >= 0 else "inverse",
                help="De cada 100 predicciones, cuántas fueron correctas. Un modelo aleatorio acierta el 50% — cualquier valor por encima suma valor real."
            )
            st.progress(min(accuracy / 100, 1.0))
        with col2:
            precision = metricas.get('precision', 0) * 100
            st.metric(
                "Precisión de subidas",
                f"{fmt_num(precision, 1)}%",
                help="Cuando el modelo dice 'va a subir', ¿cuántas veces tiene razón? Alto = pocas falsas alarmas de compra."
            )
            st.progress(min(precision / 100, 1.0))
        with col3:
            recall = metricas.get('recall', 0) * 100
            st.metric(
                "Detección de subidas",
                f"{fmt_num(recall, 1)}%",
                help="De todas las subidas reales que ocurrieron, ¿cuántas detectó el modelo? Alto = no se perdió oportunidades."
            )
            st.progress(min(recall / 100, 1.0))
        with col4:
            f1 = metricas.get('f1', 0) * 100
            st.metric(
                "Equilibrio general",
                f"{fmt_num(f1, 1)}%",
                help="Combina precisión y detección en un solo número. Penaliza fuertemente si uno de los dos es bajo. Es la métrica más completa del modelo."
            )
            st.progress(min(f1 / 100, 1.0))


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


def _fmt_ratio(val, fmt=".2f", suffix=""):
    if val is None:
        return "N/D"
    return f"{val:{fmt}}{suffix}"


def render_sec_card(sec_data: Dict):
    """Renderiza sección de datos fundamentales SEC."""
    st.markdown("<p class='section-header'>Datos Fundamentales (SEC / yfinance)</p>", unsafe_allow_html=True)

    if not sec_data or not sec_data.get("disponible", False):
        st.info("Datos fundamentales no disponibles para este activo (puede ser no-US o ETF).")
        return

    signal = sec_data.get("fundamental_signal", "neutral")
    score = sec_data.get("fundamental_score", 0.0)
    resumen = sec_data.get("resumen", "")
    company = sec_data.get("company_name", "")
    ratios = sec_data.get("ratios", {})
    balance = sec_data.get("balance", {})
    filings = sec_data.get("recent_filings", [])

    color_map = {"positivo": "#28a745", "negativo": "#dc3545", "neutral": "#6c757d"}
    color = color_map.get(signal, "#6c757d")

    st.markdown(f"""
        <div class='card' style='border-left: 4px solid {color}; padding: 15px;'>
            <div style='font-size:1.1rem;font-weight:600;color:{color};margin-bottom:8px;'>
                {company} — Señal fundamental: {signal.upper()}
                (score: {score:+.2f})
            </div>
            <p style='color:#444;margin:0'>{resumen}</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Valuación**")
        st.metric("P/E Ratio", _fmt_ratio(ratios.get("pe_ratio"), ".1f", "x"),
                  help="Precio / Ganancia. < 15 = barato, > 30 = caro")
        st.metric("P/B Ratio", _fmt_ratio(ratios.get("pb_ratio"), ".2f", "x"),
                  help="Precio / Valor en libros")
        st.metric("EV/EBITDA", _fmt_ratio(ratios.get("ev_ebitda"), ".1f", "x"))
        mc = ratios.get("market_cap")
        if mc:
            if mc >= 1e12:
                mc_str = f"${mc/1e12:.2f}T"
            elif mc >= 1e9:
                mc_str = f"${mc/1e9:.1f}B"
            else:
                mc_str = f"${mc/1e6:.0f}M"
            st.metric("Market Cap", mc_str)

    with col2:
        st.markdown("**Rentabilidad**")
        roe = ratios.get("roe")
        st.metric("ROE", _fmt_ratio(roe * 100 if roe else None, ".1f", "%"),
                  help="Return on Equity. >15% = bueno")
        roa = ratios.get("roa")
        st.metric("ROA", _fmt_ratio(roa * 100 if roa else None, ".1f", "%"),
                  help="Return on Assets")
        pm = ratios.get("profit_margin")
        st.metric("Margen Neto", _fmt_ratio(pm * 100 if pm else None, ".1f", "%"))
        om = ratios.get("operating_margin")
        st.metric("Margen Operativo", _fmt_ratio(om * 100 if om else None, ".1f", "%"))

    with col3:
        st.markdown("**Crecimiento y Deuda**")
        rg = ratios.get("revenue_growth")
        st.metric("Crecimiento Ingresos", _fmt_ratio(rg * 100 if rg else None, ".1f", "%"))
        eg = ratios.get("earnings_growth")
        st.metric("Crecimiento Ganancias", _fmt_ratio(eg * 100 if eg else None, ".1f", "%"))
        st.metric("Deuda/Capital", _fmt_ratio(ratios.get("debt_to_equity"), ".2f", "x"),
                  help="< 0.5 = bajo apalancamiento")
        st.metric("Current Ratio", _fmt_ratio(ratios.get("current_ratio"), ".2f"),
                  help="> 1 = liquidez adecuada")

    # Balance resumido
    with st.expander("Ver Balance Resumido"):
        b_col1, b_col2 = st.columns(2)
        def _fmt_millones(v):
            if v is None:
                return "N/D"
            if abs(v) >= 1e12:
                return f"${v/1e12:.2f}T"
            if abs(v) >= 1e9:
                return f"${v/1e9:.2f}B"
            return f"${v/1e6:.1f}M"

        with b_col1:
            st.metric("Ingresos TTM", _fmt_millones(balance.get("revenue_ttm")))
            st.metric("Utilidad Neta TTM", _fmt_millones(balance.get("net_income_ttm")))
            st.metric("Flujo de Caja Operativo", _fmt_millones(balance.get("operating_cash_flow")))
        with b_col2:
            st.metric("Deuda Total", _fmt_millones(balance.get("total_debt")))
            st.metric("Efectivo", _fmt_millones(balance.get("cash_and_equivalents")))
            st.metric("Flujo de Caja Libre", _fmt_millones(balance.get("free_cash_flow")))

    # Filings SEC recientes
    if filings:
        with st.expander(f"Filings SEC recientes ({len(filings)})"):
            st.caption("Fuente: SEC EDGAR — filings más recientes (10-K, 10-Q, 8-K)")
            for f in filings:
                badge_color = {
                    "10-K": "#17a2b8", "10-Q": "#28a745",
                    "8-K": "#ffc107", "DEF 14A": "#6c757d"
                }.get(f["form_type"], "#6c757d")
                st.markdown(
                    f"""<div style='display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid #eee'>
                        <span style='background:{badge_color};color:white;padding:2px 8px;border-radius:4px;
                        font-size:0.8rem;font-weight:600;min-width:55px;text-align:center'>{f["form_type"]}</span>
                        <span style='color:#666;font-size:0.85rem'>{f["filing_date"]}</span>
                        <span style='color:#333;font-size:0.9rem'>{f["description"] or f["form_type"]}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
    else:
        st.caption("No se encontraron filings SEC para este activo (puede no estar listado en EE.UU.)")


def _color_by_rec(tipo: str) -> str:
    return {"compra": "#28a745", "venta": "#dc3545"}.get(tipo, "#ffc107")


def render_portfolio_tab():
    """Renderiza la pestaña de análisis de portafolio."""
    st.markdown("### Análisis de Portafolio")
    st.markdown("Seleccioná los activos, asigná los pesos y presioná **Analizar Portafolio** para obtener métricas y optimización de Markowitz.")

    TICKERS_POPULARES = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META",
        "JPM", "V", "MA", "UNH", "HD", "PG", "KO", "DIS",
        "NFLX", "PYPL", "INTC", "AMD", "ORCL", "CRM", "ADBE",
        "BRK-B", "XOM", "CVX", "GS", "BAC", "WMT", "COST"
    ]

    # --- Paso 1: selección de activos ---
    st.markdown("**Paso 1 — Seleccioná los activos**")
    col_sel, col_custom = st.columns([3, 2])
    with col_sel:
        selected = st.multiselect(
            "Activos de la lista",
            options=TICKERS_POPULARES,
            default=st.session_state.get("pf_selected", ["AAPL", "MSFT", "GOOGL", "AMZN"]),
            help="Podés elegir hasta 15 activos",
        )
        st.session_state.pf_selected = selected
    with col_custom:
        custom_input = st.text_input(
            "Agregar tickers personalizados",
            value=st.session_state.get("pf_custom", ""),
            placeholder="Ej: BMA.BA, GGAL.BA, BABA",
            help="Separados por coma — para tickers no disponibles en la lista",
        )
        st.session_state.pf_custom = custom_input

    custom_tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
    all_tickers = selected + [t for t in custom_tickers if t not in selected]

    if len(all_tickers) < 2:
        st.info("Seleccioná al menos 2 activos para continuar.")
        return

    if len(all_tickers) > 15:
        st.warning("Máximo 15 activos. Se usarán los primeros 15.")
        all_tickers = all_tickers[:15]

    # --- Paso 2 y 3: pesos + submit dentro de form (evita re-render al escribir) ---
    st.markdown("**Paso 2 — Asigná los pesos (%)** — se normalizan automáticamente a 100%")
    n = len(all_tickers)
    default_w = round(100.0 / n, 1)

    with st.form("portfolio_weights_form"):
        cols_per_row = min(n, 5)
        rows = [all_tickers[i:i+cols_per_row] for i in range(0, n, cols_per_row)]
        weights_raw = []
        for row_tickers in rows:
            cols = st.columns(len(row_tickers))
            for col, ticker in zip(cols, row_tickers):
                w = col.number_input(
                    ticker,
                    min_value=0.0,
                    max_value=100.0,
                    value=float(st.session_state.get(f"pf_w_{ticker}", default_w)),
                    step=1.0,
                    format="%.1f",
                )
                weights_raw.append((ticker, w))

        submitted = st.form_submit_button("Analizar Portafolio", type="primary", use_container_width=True)

    total = sum(w for _, w in weights_raw)
    if total > 0:
        weights_norm = {t: round(w / total, 4) for t, w in weights_raw}
        st.caption(f"Suma actual: {total:.1f}%  →  pesos: {', '.join(f'{t} {v*100:.1f}%' for t, v in weights_norm.items())}")
    else:
        st.error("Los pesos deben ser mayores a 0.")
        return

    if submitted:
        for t, w in weights_raw:
            st.session_state[f"pf_w_{t}"] = w
        w_list = list(weights_norm.values())
        with st.spinner(f"Analizando portafolio de {n} activos... esto puede tomar 1-2 minutos"):
            pf_data = analyze_portfolio(all_tickers, w_list)
        if pf_data:
            st.session_state.last_portfolio = pf_data
        else:
            return

    if "last_portfolio" not in st.session_state:
        st.info("Completá los pasos anteriores y presioná **Analizar Portafolio**.")
        return

    pf = st.session_state.last_portfolio
    metricas = pf["metricas"]
    opt = pf["optimizacion"]
    activos = pf["activos"]
    tickers_ok = pf["tickers"]

    st.caption(f"Análisis al: {pf['fecha_analisis'][:19].replace('T', ' ')} | Activos: {', '.join(tickers_ok)}")

    # --- Recomendación general ---
    rec_text = pf.get("recomendacion_portafolio", "")
    sharpe_temp = metricas["sharpe_ratio"]
    if sharpe_temp >= 1:
        _bg, _border, _col, _sem = "#d4edda", "#28a745", "#155724", "🟢"
        _titulo = "Portafolio en buen estado"
    elif sharpe_temp >= 0.5:
        _bg, _border, _col, _sem = "#fff3cd", "#ffc107", "#856404", "🟡"
        _titulo = "Portafolio con margen de mejora"
    else:
        _bg, _border, _col, _sem = "#f8d7da", "#dc3545", "#721c24", "🔴"
        _titulo = "Portafolio requiere ajustes"

    st.markdown(f"""
    <div style='background:{_bg};border:2px solid {_border};color:{_col};
                border-radius:12px;padding:18px 22px;margin-bottom:12px'>
        <div style='font-size:1.25rem;font-weight:700;margin-bottom:8px'>{_sem} {_titulo}</div>
        <p style='font-size:0.95rem;line-height:1.7;margin:0'>{rec_text}</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Alertas del portafolio ---
    alertas = pf.get("alertas", [])
    if alertas:
        for a in alertas:
            if a["nivel"] == "critical":
                st.error(f"Alerta ({a['tipo']}): {a['mensaje']}")
            else:
                st.warning(f"Aviso ({a['tipo']}): {a['mensaje']}")

    st.markdown("---")

    # --- Métricas clave ---
    st.markdown("<p class='section-header'>Métricas del Portafolio</p>", unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    sharpe = metricas["sharpe_ratio"]
    vol = metricas["volatility"]
    var_95 = metricas["var_95"]
    div_ratio = metricas["diversification_ratio"]
    ret_esp = metricas["expected_return"]

    # Contexto visible (sin necesidad de hover)
    sharpe_ctx = ("Excelente riesgo/retorno ✅" if sharpe >= 2
                  else "Buena riesgo/retorno ✅" if sharpe >= 1
                  else "Riesgo/retorno moderado ⚠️" if sharpe >= 0.5
                  else "Riesgo/retorno bajo ❌")
    sharpe_color = "normal" if sharpe >= 0.5 else "inverse"

    vol_ctx = ("Volatilidad baja 🟢" if vol < 15
               else "Volatilidad moderada 🟡" if vol < 25
               else "Volatilidad alta 🔴")

    var_ctx = f"En el peor 5% de días: −{abs(var_95):.1f}%"

    div_ctx = ("Bien diversificado ✅" if div_ratio >= 1.3
               else "Diversificación moderada ⚠️" if div_ratio >= 1.1
               else "Baja diversificación ❌")

    ret_ctx = ("Retorno alto ✅" if ret_esp >= 15
               else "Retorno moderado 👍" if ret_esp >= 8
               else "Retorno bajo ⚠️" if ret_esp >= 0
               else "Retorno negativo ❌")

    m1.metric("Retorno Esperado (anual)", f"{ret_esp:.1f}%",
              delta=ret_ctx,
              delta_color="normal" if ret_esp >= 0 else "inverse",
              help="Retorno anualizado ponderado (histórico + predicción ML)")
    m2.metric("Volatilidad (anual)", f"{vol:.1f}%",
              delta=vol_ctx,
              delta_color="off",
              help="Oscilación típica del portafolio. Menor = más estable.")
    m3.metric("Sharpe Ratio", f"{sharpe:.2f}",
              delta=sharpe_ctx,
              delta_color=sharpe_color,
              help="Retorno por unidad de riesgo. >1 = bueno, >2 = excelente. Rf = 4.5% anual")
    m4.metric("VaR 95%", f"{var_95:.1f}%",
              delta=var_ctx,
              delta_color="off",
              help="En el peor 5% de los días, la pérdida esperada (anualizado)")
    m5.metric("Diversificación", f"{div_ratio:.2f}",
              delta=div_ctx,
              delta_color="off",
              help=">1 = los activos se compensan entre sí. 1 = sin diversificación")

    if metricas.get("beta_portfolio") is not None:
        beta = metricas['beta_portfolio']
        beta_ctx = ("Más volátil que el mercado" if beta > 1.1
                    else "Menos volátil que el mercado" if beta < 0.9
                    else "Similar volatilidad al mercado")
        st.caption(f"Beta vs S&P 500: {beta:.2f} — {beta_ctx}")

    st.markdown("---")

    # --- Composición y análisis individual ---
    col_izq, col_der = st.columns([1, 2])

    with col_izq:
        st.markdown("<p class='section-header'>Composición Actual</p>", unsafe_allow_html=True)

        # Pie chart de pesos actuales
        weights_actual = {a["ticker"]: a["weight"] for a in activos}
        fig_pie = go.Figure(go.Pie(
            labels=list(weights_actual.keys()),
            values=[round(v * 100, 1) for v in weights_actual.values()],
            hole=0.35,
            marker_colors=["#2563eb", "#10b981", "#f59e0b", "#ef4444",
                           "#8b5cf6", "#ec4899", "#14b8a6", "#f97316"][:len(activos)],
            textinfo="label+percent",
        ))
        fig_pie.update_layout(height=260, margin=dict(l=0, r=0, t=20, b=0), showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_der:
        st.markdown("<p class='section-header'>Resumen por Activo</p>", unsafe_allow_html=True)
        rows = []
        for a in activos:
            rec_icon = {"compra": "BUY", "venta": "SELL", "mantener": "HOLD"}.get(
                a["tipo_recomendacion"], a["tipo_recomendacion"].upper()
            )
            rows.append({
                "Ticker": a["ticker"],
                "Peso %": round(a["weight"] * 100, 1),
                "Precio": f"${a['price']:.2f}",
                "Ret. Esp. %": round(a["expected_return"], 1),
                "Volat. %": round(a["volatility"], 1),
                "Recom.": rec_icon,
                "Mercado": a["senal_mercado"].capitalize(),
                "Sentiment": a["sentimiento"].capitalize(),
                "Fundamental": a["fundamental_signal"].capitalize(),
            })
        df_activos = pd.DataFrame(rows)
        st.table(df_activos)

    st.markdown("---")

    # --- Matriz de correlación ---
    st.markdown("<p class='section-header'>Matriz de Correlación</p>", unsafe_allow_html=True)
    corr_data = metricas["correlation_matrix"]
    if corr_data:
        corr_df = pd.DataFrame(corr_data)
        fig_corr = go.Figure(go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns.tolist(),
            y=corr_df.index.tolist(),
            colorscale="RdYlGn",
            zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr_df.values],
            texttemplate="%{text}",
            showscale=True,
        ))
        fig_corr.update_layout(
            height=350, margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="", yaxis_title="",
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("Verde = correlación negativa (mayor diversificación) | Rojo = correlación positiva (menor diversificación)")

    st.markdown("---")

    # --- Optimización de Markowitz ---
    st.markdown("<p class='section-header'>Optimización de Markowitz</p>", unsafe_allow_html=True)

    if not opt.get("disponible", False):
        st.warning("Optimización no disponible (scipy no instalado o datos insuficientes).")
    else:
        col_ms, col_mv = st.columns(2)

        with col_ms:
            st.markdown("**Portafolio de Máximo Sharpe**")
            st.metric("Sharpe Óptimo", f"{opt['max_sharpe_sharpe']:.2f}")
            st.metric("Retorno Esperado", f"{opt['max_sharpe_return']:.1f}%")
            st.metric("Volatilidad", f"{opt['max_sharpe_volatility']:.1f}%")

            w_ms = opt["max_sharpe_weights"]
            fig_ms = go.Figure(go.Bar(
                x=list(w_ms.keys()),
                y=[v * 100 for v in w_ms.values()],
                marker_color="#2563eb",
                text=[f"{v*100:.1f}%" for v in w_ms.values()],
                textposition="outside",
            ))
            fig_ms.update_layout(
                title="Pesos óptimos (Max Sharpe)", height=250,
                margin=dict(l=0, r=0, t=35, b=0),
                yaxis_title="Peso (%)", showlegend=False,
            )
            st.plotly_chart(fig_ms, use_container_width=True)

        with col_mv:
            st.markdown("**Portafolio de Mínima Varianza**")
            min_sharpe = (opt["min_variance_return"] - 4.5) / opt["min_variance_volatility"] if opt["min_variance_volatility"] else 0
            st.metric("Sharpe", f"{min_sharpe:.2f}")
            st.metric("Retorno Esperado", f"{opt['min_variance_return']:.1f}%")
            st.metric("Volatilidad Mínima", f"{opt['min_variance_volatility']:.1f}%")

            w_mv = opt["min_variance_weights"]
            fig_mv = go.Figure(go.Bar(
                x=list(w_mv.keys()),
                y=[v * 100 for v in w_mv.values()],
                marker_color="#10b981",
                text=[f"{v*100:.1f}%" for v in w_mv.values()],
                textposition="outside",
            ))
            fig_mv.update_layout(
                title="Pesos óptimos (Mín. Varianza)", height=250,
                margin=dict(l=0, r=0, t=35, b=0),
                yaxis_title="Peso (%)", showlegend=False,
            )
            st.plotly_chart(fig_mv, use_container_width=True)

        # Frontera eficiente
        frontier = opt.get("efficient_frontier", [])
        if frontier:
            st.markdown("**Frontera Eficiente**")
            fe_vols = [p["volatility"] for p in frontier]
            fe_rets = [p["expected_return"] for p in frontier]
            fe_sharpes = [p["sharpe"] for p in frontier]

            fig_fe = go.Figure()
            fig_fe.add_trace(go.Scatter(
                x=fe_vols, y=fe_rets, mode="lines+markers",
                marker=dict(color=fe_sharpes, colorscale="Viridis", showscale=True,
                            colorbar=dict(title="Sharpe"), size=8),
                line=dict(color="#888", width=1),
                name="Frontera eficiente",
                hovertemplate="Volat: %{x:.1f}%<br>Ret: %{y:.1f}%<br>Sharpe: %{marker.color:.2f}",
            ))
            # Punto portafolio actual
            port_vol_actual = metricas["volatility"]
            port_ret_actual = metricas["expected_return"]
            fig_fe.add_trace(go.Scatter(
                x=[port_vol_actual], y=[port_ret_actual],
                mode="markers", marker=dict(color="red", size=12, symbol="star"),
                name="Portafolio actual",
            ))
            fig_fe.add_trace(go.Scatter(
                x=[opt["max_sharpe_volatility"]], y=[opt["max_sharpe_return"]],
                mode="markers", marker=dict(color="blue", size=12, symbol="diamond"),
                name="Max Sharpe",
            ))
            fig_fe.add_trace(go.Scatter(
                x=[opt["min_variance_volatility"]], y=[opt["min_variance_return"]],
                mode="markers", marker=dict(color="green", size=12, symbol="square"),
                name="Mín. Varianza",
            ))
            fig_fe.update_layout(
                height=380,
                xaxis_title="Volatilidad anualizada (%)",
                yaxis_title="Retorno esperado (%)",
                margin=dict(l=0, r=0, t=20, b=0),
                hovermode="closest",
            )
            st.plotly_chart(fig_fe, use_container_width=True)
            st.caption(
                "La frontera eficiente muestra las combinaciones óptimas de riesgo-retorno. "
                "Portafolios sobre la curva dominan a los que están debajo."
            )

        # Comparación de pesos: actual vs óptimo
        st.markdown("**Comparación: Pesos Actuales vs Óptimos (Max Sharpe)**")
        comp_rows = []
        for t in tickers_ok:
            w_act = pf["weights"].get(t, 0)
            w_opt = opt["max_sharpe_weights"].get(t, 0)
            diff = w_opt - w_act
            accion = "Aumentar" if diff > 0.03 else ("Reducir" if diff < -0.03 else "Mantener")
            comp_rows.append({
                "Ticker": t,
                "Actual %": f"{w_act*100:.1f}%",
                "Óptimo Max Sharpe %": f"{w_opt*100:.1f}%",
                "Diferencia pp": f"{diff*100:+.1f}",
                "Acción Sugerida": accion,
            })
        st.table(pd.DataFrame(comp_rows))

        # --- Conclusión y plan de acción ---
        st.markdown("---")
        st.markdown("<p class='section-header'>Conclusión y Plan de Acción</p>", unsafe_allow_html=True)

        aumentar = [r for r in comp_rows if r["Acción Sugerida"] == "Aumentar"]
        reducir  = [r for r in comp_rows if r["Acción Sugerida"] == "Reducir"]
        mantener = [r for r in comp_rows if r["Acción Sugerida"] == "Mantener"]

        sharpe_actual = metricas["sharpe_ratio"]
        sharpe_optimo = opt["max_sharpe_sharpe"]
        mejora_posible = sharpe_optimo - sharpe_actual

        if mejora_posible > 0.3:
            mejora_msg = f"Rebalanceando hacia los pesos óptimos podrías mejorar el Sharpe de **{sharpe_actual:.2f}** a **{sharpe_optimo:.2f}** (+{mejora_posible:.2f})."
        elif mejora_posible > 0.05:
            mejora_msg = f"El portafolio está cerca del óptimo (Sharpe actual {sharpe_actual:.2f} vs óptimo {sharpe_optimo:.2f})."
        else:
            mejora_msg = f"El portafolio ya está muy cerca del óptimo de Markowitz (Sharpe {sharpe_actual:.2f})."

        lineas = [mejora_msg]
        if aumentar:
            tickers_a = ", ".join(f"**{r['Ticker']}** ({r['Óptimo Max Sharpe %']})" for r in aumentar)
            lineas.append(f"📈 **Aumentar exposición en**: {tickers_a}")
        if reducir:
            tickers_r = ", ".join(f"**{r['Ticker']}** ({r['Óptimo Max Sharpe %']})" for r in reducir)
            lineas.append(f"📉 **Reducir exposición en**: {tickers_r}")
        if mantener and not aumentar and not reducir:
            lineas.append("✅ La distribución actual está alineada con los pesos óptimos — no se requieren cambios significativos.")

        # Señal fundamental: ¿cuántos activos tienen señal positiva?
        activos_positivos = [a for a in activos if a.get("fundamental_signal") == "positivo"]
        activos_negativos = [a for a in activos if a.get("fundamental_signal") == "negativo"]
        if activos_positivos:
            lineas.append(f"✅ {len(activos_positivos)} activo(s) con señal fundamental positiva: {', '.join(a['ticker'] for a in activos_positivos)}")
        if activos_negativos:
            lineas.append(f"⚠️ {len(activos_negativos)} activo(s) con señal fundamental negativa: {', '.join(a['ticker'] for a in activos_negativos)} — revisar antes de aumentar posición")

        lineas.append("_Esta herramienta es de apoyo — no constituye asesoramiento financiero._")

        for l in lineas:
            st.markdown(f"- {l}")


def render_executive_summary(data: Dict):
    """Tarjeta de resumen ejecutivo con semáforo y veredicto en lenguaje simple."""
    tipo = data['recomendacion']['tipo']
    variacion = data['prediccion']['variacion_pct']
    sent = data['sentimiento']['sentimiento']
    senal_merc = data['mercado'].get('senal', 'neutral')
    prob_subida = data['prediccion'].get('prob_subida', 0.5) * 100

    if tipo == 'compra':
        semaforo, bg, border, color_text, veredicto = "🟢", "#d4edda", "#28a745", "#155724", "SEÑAL DE COMPRA"
    elif tipo == 'venta':
        semaforo, bg, border, color_text, veredicto = "🔴", "#f8d7da", "#dc3545", "#721c24", "SEÑAL DE VENTA"
    else:
        semaforo, bg, border, color_text, veredicto = "🟡", "#fff3cd", "#ffc107", "#856404", "MANTENER"

    dir_text = f"subida del {variacion:.1f}%" if variacion > 0 else f"baja del {abs(variacion):.1f}%"
    sent_map = {"positivo": "positivo", "negativo": "negativo", "neutral": "neutral",
                "muy positivo": "muy positivo", "muy negativo": "muy negativo",
                "ligeramente positivo": "levemente positivo", "ligeramente negativo": "levemente negativo"}
    sent_display = sent_map.get(sent, sent)
    senal_display = {"alcista": "alcista (favorable)", "bajista": "bajista (desfavorable)", "neutral": "neutral"}.get(senal_merc, senal_merc)

    accuracy_pct = data['prediccion'].get('metricas', {}).get('accuracy', 0) * 100
    confianza_str = f"El modelo acierta el {accuracy_pct:.0f}% de las veces en promedio." if accuracy_pct > 0 else ""

    st.markdown(f"""
    <div style='background:{bg};border:2px solid {border};color:{color_text};
                border-radius:12px;padding:20px 25px;margin-bottom:15px'>
        <div style='font-size:1.6rem;font-weight:700;margin-bottom:10px'>{semaforo} {veredicto}</div>
        <p style='font-size:1rem;line-height:1.7;margin:0'>
            El modelo predice una <strong>{dir_text}</strong> en los próximos 3 días
            (confianza: <strong>{prob_subida:.0f}%</strong>).
            Las noticias muestran un sentimiento <strong>{sent_display}</strong>
            y los indicadores técnicos son <strong>{senal_display}</strong>.
            {confianza_str}
        </p>
        <p style='font-size:0.8rem;margin-top:10px;opacity:0.75'>
            Herramienta de apoyo a la decisión — no constituye asesoramiento financiero.
        </p>
    </div>
    """, unsafe_allow_html=True)


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

    tab1, tab2, tab3, tab4 = st.tabs(["Análisis de Activos", "Historial de Alertas", "Portafolio", "Backtesting"])

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

            render_executive_summary(data)

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

            # Datos fundamentales SEC
            st.markdown("---")
            if data.get("sec_data"):
                render_sec_card(data["sec_data"])
            else:
                st.caption("Datos fundamentales no incluidos en esta consulta.")

        else:
            st.info("Seleccione un activo en el panel lateral y presione 'Analizar' para comenzar")

    with tab2:
        if "token" in st.session_state:
            render_alerts_history()
        else:
            st.warning("Inicie sesión para ver el historial de alertas")

    with tab3:
        render_portfolio_tab()

    with tab4:
        render_backtest_tab()


def render_backtest_tab():
    """Renderiza la pestaña de backtesting walk-forward."""
    st.markdown("### Backtesting Walk-Forward con Señales ML")
    st.markdown(
        "Reentrena el ensemble de ML cada **trimestre** (63 días) con una ventana de "
        "**504 días** y simula operaciones con **Backtrader**. "
        "Comisión: 0.1% por operación. Tasa libre de riesgo: 4.5% anual."
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        bt_ticker = st.text_input(
            "Ticker",
            value=st.session_state.get("bt_ticker", "AAPL"),
            placeholder="AAPL",
            key="bt_ticker_input",
        )
    with col2:
        bt_years = st.selectbox("Período", options=[1, 2, 3, 5], index=2, key="bt_years_input")
    with col3:
        bt_cash = st.number_input("Capital inicial (USD)", min_value=1000, value=10000, step=1000, key="bt_cash_input")

    if st.button("Ejecutar Backtest", type="primary", use_container_width=True):
        ticker_clean = bt_ticker.strip().upper()
        if not ticker_clean:
            st.error("Ingresá un ticker válido.")
            return
        st.session_state.bt_ticker = ticker_clean

        with st.spinner(f"Ejecutando backtest de {ticker_clean} ({bt_years} años)... ~1-2 minutos"):
            try:
                resp = requests.post(
                    f"{st.session_state.get('api_url', 'http://localhost:8000')}/backtest/{ticker_clean}",
                    params={"years": bt_years, "initial_cash": bt_cash},
                    timeout=300,
                )
                if resp.status_code == 200:
                    st.session_state.last_backtest = resp.json()
                else:
                    st.error(f"Error {resp.status_code}: {resp.json().get('detail', 'Error desconocido')}")
                    return
            except Exception as e:
                st.error(f"Error al conectar con el backend: {e}")
                return

    if "last_backtest" not in st.session_state:
        st.info("Configurá los parámetros y presioná **Ejecutar Backtest**.")
        return

    bt_data = st.session_state.last_backtest
    m = bt_data["metrics"]
    bm = bt_data["benchmark_metrics"]

    st.caption(
        f"Período: {bt_data['start_date']} → {bt_data['end_date']} | "
        f"Ticker: {bt_data['ticker']} | "
        f"Señales: {bt_data['buy_signals']} compras, {bt_data['sell_signals']} ventas"
    )

    # ── Métricas comparativas ─────────────────────────────────────────────
    st.markdown("#### Métricas: Estrategia ML vs Buy & Hold")
    cols = st.columns(6)
    metrics_def = [
        ("CAGR", "cagr", "%"),
        ("Sharpe", "sharpe_ratio", ""),
        ("Sortino", "sortino_ratio", ""),
        ("Max Drawdown", "max_drawdown", "%"),
        ("Retorno Total", "total_return", "%"),
        ("Alpha vs B&H", "alpha", "pp"),
    ]
    for col, (label, key, unit) in zip(cols, metrics_def):
        val = m[key]
        ref = bm[key]
        delta = val - ref
        color = "normal" if abs(delta) < 0.5 else ("inverse" if key == "max_drawdown" else "normal")
        col.metric(
            label,
            f"{val:.1f}{unit}",
            delta=f"{delta:+.1f}{unit}" if key != "alpha" else None,
            delta_color=color,
        )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Trades realizados", m["num_trades"])
    col_b.metric("Win Rate", f"{m['win_rate']:.0f}%",
                 help="% de operaciones que terminaron en ganancia")
    col_c.metric("Trade promedio", f"{m['avg_trade_pct']:+.1f}%")

    # ── Interpretación en lenguaje simple ─────────────────────────────────
    st.markdown("#### ¿Qué significan estos resultados?")
    interpretaciones = []

    # Sortino
    sortino = m.get('sortino_ratio', 0)
    sortino_bm = bm.get('sortino_ratio', 0)
    if sortino > 2:
        interpretaciones.append(f"✅ **Sortino {sortino:.2f}**: Excelente manejo del riesgo — las caídas son pequeñas en relación al retorno obtenido.")
    elif sortino > 1:
        interpretaciones.append(f"👍 **Sortino {sortino:.2f}**: Buen manejo del riesgo a la baja.")
    else:
        interpretaciones.append(f"⚠️ **Sortino {sortino:.2f}**: El portafolio tuvo caídas considerables en relación al retorno. Buy & Hold obtuvo {sortino_bm:.2f}.")

    # Calmar
    calmar = m.get('calmar_ratio', 0)
    if calmar > 1:
        interpretaciones.append(f"✅ **Calmar {calmar:.2f}**: El retorno anual superó ampliamente la caída máxima del portafolio.")
    elif calmar > 0.5:
        interpretaciones.append(f"👍 **Calmar {calmar:.2f}**: El retorno cubre razonablemente la caída máxima observada.")
    else:
        interpretaciones.append(f"⚠️ **Calmar {calmar:.2f}**: La caída máxima fue grande en relación al retorno anual — el portafolio tuvo períodos de pérdida prolongados.")

    # Alpha
    alpha = m.get('alpha', 0)
    if alpha > 5:
        interpretaciones.append(f"✅ **Alpha {alpha:+.1f}pp**: La estrategia ML superó al Buy & Hold por {alpha:.1f} puntos porcentuales — añade valor real.")
    elif alpha > 0:
        interpretaciones.append(f"👍 **Alpha {alpha:+.1f}pp**: La estrategia ML superó ligeramente al Buy & Hold.")
    elif alpha > -5:
        interpretaciones.append(f"⚠️ **Alpha {alpha:+.1f}pp**: La estrategia ML rindió levemente por debajo del Buy & Hold. Considerar costos de transacción.")
    else:
        interpretaciones.append(f"❌ **Alpha {alpha:+.1f}pp**: Buy & Hold superó claramente a la estrategia ML en este período.")

    # Max Drawdown
    mdd = m.get('max_drawdown', 0)
    if abs(mdd) < 10:
        interpretaciones.append(f"✅ **Max Drawdown {mdd:.1f}%**: Caída máxima baja — el portafolio fue estable.")
    elif abs(mdd) < 20:
        interpretaciones.append(f"👍 **Max Drawdown {mdd:.1f}%**: Caída máxima moderada — típica en estrategias activas.")
    else:
        interpretaciones.append(f"⚠️ **Max Drawdown {mdd:.1f}%**: En el peor momento, el portafolio perdió {abs(mdd):.1f}% de su valor pico. Alta tolerancia al riesgo requerida.")

    # Win Rate
    wr = m.get('win_rate', 0)
    if wr >= 55:
        interpretaciones.append(f"✅ **Win Rate {wr:.0f}%**: Más de la mitad de las operaciones fueron rentables.")
    elif wr >= 45:
        interpretaciones.append(f"👍 **Win Rate {wr:.0f}%**: Tasa de acierto en operaciones cercana al promedio del mercado.")
    else:
        interpretaciones.append(f"⚠️ **Win Rate {wr:.0f}%**: Menos de la mitad de las operaciones resultaron en ganancia. Revisar la estrategia de entrada/salida.")

    for interp in interpretaciones:
        st.markdown(f"- {interp}")

    # ── Tabla comparativa ─────────────────────────────────────────────────
    with st.expander("Tabla comparativa completa"):
        rows = []
        all_keys = ["total_return", "cagr", "sharpe_ratio", "sortino_ratio",
                    "max_drawdown", "calmar_ratio", "alpha", "beta", "num_trades", "win_rate", "avg_trade_pct"]
        labels_map = {
            "total_return": "Retorno Total (%)",
            "cagr": "CAGR (%)",
            "sharpe_ratio": "Sharpe Ratio",
            "sortino_ratio": "Sortino Ratio",
            "max_drawdown": "Max Drawdown (%)",
            "calmar_ratio": "Calmar Ratio",
            "alpha": "Alpha (pp)",
            "beta": "Beta",
            "num_trades": "N° Trades",
            "win_rate": "Win Rate (%)",
            "avg_trade_pct": "Trade Promedio (%)",
        }
        for key in all_keys:
            rows.append({"Métrica": labels_map[key], "Estrategia ML": m[key], "Buy & Hold": bm.get(key, "-")})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Curva de equity ───────────────────────────────────────────────────
    st.markdown("#### Curva de Equity: Estrategia ML vs Buy & Hold")
    eq = bt_data.get("equity_curve", [])
    if eq:
        eq_df = pd.DataFrame(eq)
        eq_df["date"] = pd.to_datetime(eq_df["date"])

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=eq_df["date"], y=eq_df["value"],
            name="Estrategia ML",
            line=dict(color="#2196F3", width=2),
        ))
        fig_eq.add_trace(go.Scatter(
            x=eq_df["date"], y=eq_df["benchmark"],
            name="Buy & Hold",
            line=dict(color="#FF9800", width=2, dash="dash"),
        ))
        fig_eq.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Valor del portafolio (USD)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400,
            margin=dict(l=40, r=20, t=30, b=40),
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        # Drawdown
        eq_df["drawdown"] = (eq_df["value"] / eq_df["value"].cummax() - 1) * 100
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=eq_df["date"], y=eq_df["drawdown"],
            fill="tozeroy", fillcolor="rgba(244,67,54,0.2)",
            line=dict(color="#F44336", width=1),
            name="Drawdown ML",
        ))
        fig_dd.update_layout(
            xaxis_title="Fecha", yaxis_title="Drawdown (%)",
            height=200, margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    # ── Historial de trades ────────────────────────────────────────────────
    trades = bt_data.get("trades", [])
    if trades:
        st.markdown(f"#### Historial de Operaciones ({len(trades)} trades)")
        trades_df = pd.DataFrame(trades)
        trades_df["pnl_pct"] = trades_df["pnl_pct"].apply(lambda x: f"{x:+.2f}%")
        trades_df.columns = ["Fecha Apertura", "Fecha Cierre", "Precio Entrada", "Precio Salida", "P&L %", "Resultado"]
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
    else:
        st.info("No se ejecutaron trades en el período de backtest.")


def main():
    """Función principal de la aplicación."""
    if "token" not in st.session_state:
        render_auth_page()
    else:
        render_main_dashboard()


if __name__ == "__main__":
    main()
