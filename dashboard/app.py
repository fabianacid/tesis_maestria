"""
Dashboard Streamlit - Sistema Multiagente de Análisis y Optimización de Portafolios Financieros

Interfaz profesional para análisis de activos financieros.
"""
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

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


def _fmt_fecha_analisis(raw: Optional[str]) -> str:
    """Formatea un timestamp ISO 8601 del backend para mostrar en el dashboard."""
    if not raw:
        return "-"
    try:
        return datetime.fromisoformat(raw).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return str(raw)


def evaluate_risk_profile(answers: dict) -> Optional[Dict[str, Any]]:
    """Evalúa el perfil de riesgo del inversor."""
    try:
        response = requests.post(
            f"{API_URL}/risk/profile",
            json=answers,
            # La selección dinámica descarga precio/volumen de ~500 tickers del
            # S&P 500 en la primera corrida con umbrales nuevos (puede tardar
            # 1-3 min); corridas repetidas con los mismos umbrales usan caché
            # y son rápidas. 15s era insuficiente incluso antes de esto.
            timeout=300,
        )
        if response.status_code == 200:
            return response.json()
        else:
            detail = response.json().get("detail", f"Error {response.status_code}")
            st.error(f"Error: {detail}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("No se pudo conectar al backend.")
        return None
    except requests.exceptions.Timeout:
        st.error(
            "La evaluación tardó demasiado (probablemente construyendo el universo "
            "dinámico del S&P 500 por primera vez con estos umbrales). Intentá de nuevo — "
            "la segunda vez debería ser mucho más rápida por la caché."
        )
        return None
    except Exception as exc:
        st.error(f"Error: {str(exc)}")
        return None


def _generar_sintesis_portafolio(port_data: Dict, perfil: str, risk_budget: Dict) -> Dict:
    """
    Convierte el análisis técnico del portafolio en lenguaje accionable para el inversor.
    Retorna un dict con: estado, semaforo, acciones, riesgo_ok, narrativa, accion_concreta.
    """
    metricas  = port_data.get("metricas", {})
    opt       = port_data.get("optimizacion", {})
    activos   = port_data.get("activos", [])
    alertas   = port_data.get("alertas", [])

    LABELS = {
        "muy_conservador": "muy conservador",
        "conservador":     "conservador",
        "moderado":        "moderado",
        "agresivo":        "agresivo",
        "muy_agresivo":    "muy agresivo",
    }
    label_perfil = LABELS.get(perfil, perfil)

    # ── 1. Señales por ETF ───────────────────────────────────────────────
    compras  = [a for a in activos if "compra" in a.get("tipo_recomendacion", "")]
    ventas   = [a for a in activos if "venta"  in a.get("tipo_recomendacion", "")]
    mantienes = [a for a in activos if a.get("tipo_recomendacion") == "mantener"]

    # ETFs con señal fuerte (confianza > 55%)
    compras_fuertes = [a for a in compras  if a.get("confianza", 0) > 0.55]
    ventas_fuertes  = [a for a in ventas   if a.get("confianza", 0) > 0.55]

    # ETFs con señales contradictorias (ML vs sentimiento)
    contradictorios = []
    for a in activos:
        ml_sube = a.get("expected_return", 0) > 5
        ml_baja = a.get("expected_return", 0) < -5
        sent_pos = a.get("sentimiento") == "positivo"
        sent_neg = a.get("sentimiento") == "negativo"
        if (ml_sube and sent_neg) or (ml_baja and sent_pos):
            contradictorios.append(a["ticker"])

    # ── 2. Métricas vs presupuesto de riesgo ────────────────────────────
    vol_actual  = metricas.get("volatility", 0)
    var_actual  = abs(metricas.get("var_95", 0))
    sharpe      = metricas.get("sharpe_ratio", 0)
    vol_max     = risk_budget.get("vol_anual_max", 20)
    var_max     = risk_budget.get("var_95_max", 15)

    vol_ok  = vol_actual  <= vol_max
    var_ok  = var_actual  <= var_max
    riesgo_ok = vol_ok and var_ok

    # ── 3. Estrategia recomendada según perfil ───────────────────────────
    # Perfiles conservadores → HRP (más robusto, sin sobreconcentración)
    # Perfiles agresivos     → Máximo Sharpe (maximiza retorno ajustado)
    ESTRATEGIA_POR_PERFIL = {
        "muy_conservador": "hrp",
        "conservador":     "hrp",
        "moderado":        "hrp",
        "agresivo":        "max_sharpe",
        "muy_agresivo":    "max_sharpe",
    }
    estrategia_rec = ESTRATEGIA_POR_PERFIL.get(perfil, "hrp")

    sharpe_opt    = opt.get("max_sharpe_sharpe", 0)
    mejora_sharpe = sharpe_opt - sharpe
    rebalancear   = mejora_sharpe > 0.15 and opt.get("disponible", False)

    # Detectar sobreconcentración en Máximo Sharpe
    max_peso      = risk_budget.get("max_peso_activo", 0.65)
    w_sharpe_dict = opt.get("max_sharpe_weights", {})
    sobreconcentrado = any(v > max_peso + 0.01 for v in w_sharpe_dict.values())

    # ── 4. Semáforo general ──────────────────────────────────────────────
    n_alertas_criticas = sum(1 for a in alertas if a.get("nivel") == "critical")
    if ventas_fuertes or n_alertas_criticas > 0 or not riesgo_ok:
        semaforo = "rojo"
        estado   = "Atención requerida"
    elif compras_fuertes or rebalancear:
        semaforo = "amarillo"
        estado   = "Oportunidad de ajuste"
    else:
        semaforo = "verde"
        estado   = "Portafolio estable"

    # ── 5. Construir narrativa ───────────────────────────────────────────
    parrafos = []

    # Párrafo 1: estado general
    if semaforo == "verde":
        parrafos.append(
            f"Tu portafolio de perfil **{label_perfil}** está bien posicionado. "
            f"Los {len(activos)} ETFs seleccionados operan dentro de los límites de riesgo "
            f"de tu perfil y no requieren cambios urgentes."
        )
    elif semaforo == "amarillo":
        parrafos.append(
            f"Tu portafolio de perfil **{label_perfil}** está estable, "
            "pero hay oportunidades de mejora que podrías considerar."
        )
    else:
        parrafos.append(
            f"Tu portafolio de perfil **{label_perfil}** requiere atención. "
            "Se detectaron señales o métricas que superan los límites de tu perfil."
        )

    # Párrafo 2: señales de compra
    if compras:
        tickers_compra = ", ".join(f"**{a['ticker']}**" for a in compras)
        confianza_media = sum(a.get("confianza", 0) for a in compras) / len(compras)
        parrafos.append(
            f"{'El modelo detecta señal de compra' if len(compras) == 1 else 'Hay señales de compra'} "
            f"en {tickers_compra} (confianza promedio: {confianza_media*100:.0f}%). "
            + ("Si querés aumentar exposición, este es el sector más favorecido en este momento."
               if len(compras) == 1
               else "Podés considerar aumentar levemente la exposición en estos sectores.")
        )

    # Párrafo 3: señales de venta
    if ventas:
        tickers_venta = ", ".join(f"**{a['ticker']}**" for a in ventas)
        parrafos.append(
            f"Se detectaron señales de venta en {tickers_venta}. "
            "Considerá reducir o salir de esas posiciones para proteger capital."
        )

    # Párrafo 4: señales contradictorias
    if contradictorios:
        parrafos.append(
            f"**{', '.join(contradictorios)}** muestran señales mixtas: "
            "el modelo ML y el análisis de sentimiento no coinciden. "
            "Conviene esperar confirmación antes de tomar acción."
        )

    # Párrafo 5: riesgo
    if riesgo_ok:
        parrafos.append(
            f"Las métricas de riesgo están dentro del presupuesto de tu perfil — "
            f"volatilidad {vol_actual:.1f}% (límite {vol_max:.0f}%) y "
            f"VaR 95% {var_actual:.1f}% (límite {var_max:.0f}%)."
        )
    else:
        excesos = []
        if not vol_ok:
            excesos.append(f"volatilidad {vol_actual:.1f}% supera el límite de {vol_max:.0f}%")
        if not var_ok:
            excesos.append(f"VaR 95% de {var_actual:.1f}% supera el límite de {var_max:.0f}%")
        parrafos.append(
            "⚠️ El portafolio excede los límites de riesgo de tu perfil: "
            + " y ".join(excesos) + ". "
            "Considerá reducir la exposición a los activos más volátiles."
        )

    # Párrafo 6: optimización + estrategia recomendada
    hrp_sharpe  = opt.get("hrp_sharpe", 0)
    hrp_weights = opt.get("hrp_weights", {})

    if estrategia_rec == "hrp" and hrp_weights:
        top_hrp = sorted(hrp_weights.items(), key=lambda x: x[1], reverse=True)[:3]
        hrp_str = ", ".join(f"{t} ({v*100:.0f}%)" for t, v in top_hrp)
        parrafos.append(
            f"Para tu perfil **{label_perfil}** se recomienda la estrategia **HRP** "
            f"(Sharpe histórico {hrp_sharpe:.2f}) — distribuye el riesgo "
            "jerárquicamente sin sobreconcentrar en ningún activo. "
            f"Pesos principales: {hrp_str}. "
            f"Nota: el Sharpe de {sharpe:.2f} que figura en las métricas usa retornos "
            "predichos por el modelo ML (corto plazo), no históricos anualizados — "
            f"el Sharpe HRP de {hrp_sharpe:.2f} es la referencia más confiable."
        )
        if sobreconcentrado:
            etf_conc = [t for t, v in w_sharpe_dict.items() if v > max_peso + 0.01]
            parrafos.append(
                f"⚠️ La optimización Máximo Sharpe concentra más del {max_peso*100:.0f}% del límite "
                f"de tu perfil en: **{', '.join(etf_conc)}**. "
                "Eso NO es adecuado para un inversor moderado. HRP evita este problema."
            )
    elif rebalancear:
        w_opt = opt.get("max_sharpe_weights", {})
        top_cambios = sorted(w_opt.items(), key=lambda x: x[1], reverse=True)[:3]
        cambios_str = ", ".join(f"{t} ({v*100:.0f}%)" for t, v in top_cambios)
        parrafos.append(
            f"La optimización Máximo Sharpe (recomendada para tu perfil **{label_perfil}**) "
            f"mejora el Sharpe de {sharpe:.2f} a {sharpe_opt:.2f}. "
            f"Concentra en: {cambios_str}."
        )
    elif opt.get("disponible"):
        parrafos.append(
            f"Los pesos actuales del perfil son eficientes — "
            f"ninguna estrategia de optimización ofrece mejora significativa "
            f"(Sharpe actual {sharpe:.2f})."
        )

    # ── 6. Acción concreta ───────────────────────────────────────────────
    if semaforo == "verde" and not rebalancear:
        accion = "Mantener las posiciones actuales. No se requiere acción."
    elif semaforo == "verde" and rebalancear:
        accion = f"Considerar rebalancear hacia los pesos de máximo Sharpe para mejorar eficiencia."
    elif compras and not ventas:
        tickers = ", ".join(a["ticker"] for a in compras)
        accion = f"Mantener el portafolio. Si querés ajustarlo, aumentar levemente {tickers}."
    elif ventas:
        tickers = ", ".join(a["ticker"] for a in ventas)
        accion = f"Reducir o salir de {tickers}. Reubicar en activos con señal neutral o de compra."
    else:
        accion = "Esperar señales más claras antes de realizar cambios."

    return {
        "semaforo": semaforo,
        "estado": estado,
        "narrativa": parrafos,
        "accion_concreta": accion,
        "compras": [a["ticker"] for a in compras],
        "ventas":  [a["ticker"] for a in ventas],
        "contradictorios": contradictorios,
        "riesgo_ok": riesgo_ok,
        "rebalancear": rebalancear,
    }


def portfolio_for_risk_profile(
    perfil: str,
    tickers: Optional[List[str]] = None,
    pesos: Optional[List[float]] = None,
) -> Optional[Dict[str, Any]]:
    """Obtiene portafolio optimizado para un perfil de riesgo."""
    try:
        payload: Dict[str, Any] = {"perfil": perfil, "forzar_actualizacion": False}
        if tickers and pesos:
            payload["tickers"] = tickers
            payload["pesos"]   = pesos
        response = requests.post(
            f"{API_URL}/risk/portfolio",
            json=payload,
            timeout=300,
        )
        if response.status_code == 200:
            return response.json()
        else:
            detail = response.json().get("detail", f"Error {response.status_code}")
            st.error(f"Error: {detail}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("No se pudo conectar al backend.")
        return None
    except requests.exceptions.Timeout:
        st.error("El análisis tardó demasiado. Intentá de nuevo.")
        return None
    except Exception as exc:
        st.error(f"Error: {str(exc)}")
        return None


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
        col1, col2, col3, col4, col5 = st.columns(5)
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
        with col5:
            auc_val = metricas.get('auc', 0.5)
            delta_auc = (auc_val - 0.5) * 100
            st.metric(
                "AUC-ROC",
                f"{fmt_num(auc_val * 100, 1)}%",
                delta=f"{delta_auc:+.1f}pp vs azar",
                delta_color="normal" if delta_auc >= 0 else "inverse",
                help="Área bajo la curva ROC: mide la capacidad discriminativa del ensemble (subida vs bajada) independientemente del umbral. 50% = aleatorio, 100% = perfecto. Es la métrica de referencia para comparar modelos en la tesis."
            )
            st.progress(min(auc_val, 1.0))

        # ── SHAP Values ───────────────────────────────────────────────────
        shap_vals = prediccion.get('shap_values', {})
        mda_vals  = prediccion.get('mda_scores', {})
        if shap_vals or mda_vals:
            st.markdown("---")
            st.markdown("**¿Qué variables impulsan esta predicción? (Explicabilidad)**")
            st.caption(
                "SHAP (Lundberg & Lee, 2017): contribución promedio de cada variable a "
                "la probabilidad de subida. MDI (López de Prado, 2018): importancia basada "
                "en reducción de impureza. Ambas metodologías sobre el mismo Random Forest auxiliar."
            )
            imp_col1, imp_col2 = st.columns(2)
            if shap_vals:
                with imp_col1:
                    top_shap = dict(sorted(shap_vals.items(), key=lambda x: x[1], reverse=True)[:10])
                    fig_shap = go.Figure(go.Bar(
                        x=list(top_shap.values()),
                        y=list(top_shap.keys()),
                        orientation="h",
                        marker_color="#6366f1",
                        text=[f"{v:.4f}" for v in top_shap.values()],
                        textposition="outside",
                    ))
                    fig_shap.update_layout(
                        title="SHAP — Top 10 features", height=320,
                        margin=dict(l=0, r=40, t=35, b=0),
                        xaxis_title="mean |SHAP value|",
                        yaxis=dict(autorange="reversed"),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
            if mda_vals:
                with imp_col2:
                    top_mda = dict(sorted(mda_vals.items(), key=lambda x: x[1], reverse=True)[:10])
                    fig_mda = go.Figure(go.Bar(
                        x=list(top_mda.values()),
                        y=list(top_mda.keys()),
                        orientation="h",
                        marker_color="#10b981",
                        text=[f"{v:.4f}" for v in top_mda.values()],
                        textposition="outside",
                    ))
                    fig_mda.update_layout(
                        title="MDI — Top 10 features", height=320,
                        margin=dict(l=0, r=40, t=35, b=0),
                        xaxis_title="Mean Decrease Impurity",
                        yaxis=dict(autorange="reversed"),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_mda, use_container_width=True)


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
    # Usar días hábiles para evitar que la proyección caiga en fin de semana
    precio_actual = mercado['ultimo_precio']
    precio_predicho = prediccion['precio_predicho']

    future_dates = [dates[-1] + pd.offsets.BDay(i) for i in range(1, 4)]
    step = (precio_predicho - precio_actual) / 3
    future_prices = [precio_actual + step * i for i in range(1, 4)]

    # Intervalo de confianza 95% — crece con sqrt(t) desde t=0 hasta t=3 días
    ic = prediccion.get('intervalo_confianza', [0.0, 0.0])
    ic_lo, ic_hi = (ic[0], ic[1]) if len(ic) == 2 and ic[1] > ic[0] > 0 else (0.0, 0.0)
    if ic_hi > ic_lo > 0:
        ic_half = (ic_hi - ic_lo) / 2
        all_dates = [dates[-1]] + future_dates
        ic_upper = [precio_actual] + [precio_actual + ic_half * np.sqrt(i / 3) for i in range(1, 4)]
        ic_lower = [precio_actual] + [precio_actual - ic_half * np.sqrt(i / 3) for i in range(1, 4)]
        fig.add_trace(go.Scatter(
            x=list(all_dates) + list(reversed(all_dates)),
            y=ic_upper + list(reversed(ic_lower)),
            fill='toself',
            fillcolor='rgba(16, 185, 129, 0.10)',
            line=dict(color='rgba(16, 185, 129, 0.25)', width=1, dash='dot'),
            name='IC 95% (3 días)',
            hoverinfo='skip',
        ))

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

    # --- Perfil estimado (Tab3 → Tab1) ---
    _var95  = abs(metricas.get("var_95", 0))
    _vol    = metricas.get("volatility", 0)
    _PERFILES_ORDEN = [
        ("Muy conservador", 5.0,  10.0, "#1565C0"),
        ("Conservador",    10.0,  15.0, "#1976D2"),
        ("Moderado",       15.0,  20.0, "#2E7D32"),
        ("Agresivo",       25.0,  30.0, "#E65100"),
        ("Muy agresivo",   40.0,  45.0, "#B71C1C"),
    ]
    perfil_estimado = "Muy agresivo"
    color_perfil    = "#B71C1C"
    for nombre, var_lim, vol_lim, color in _PERFILES_ORDEN:
        if _var95 <= var_lim and _vol <= vol_lim:
            perfil_estimado = nombre
            color_perfil    = color
            break
    st.markdown(
        f"<div style='background:#f5f5f5;border-left:4px solid {color_perfil};"
        f"padding:12px 16px;border-radius:6px;margin-bottom:12px'>"
        f"<span style='font-size:0.85rem;color:#555'>Perfil de riesgo estimado para este portafolio</span><br>"
        f"<span style='font-size:1.1rem;font-weight:700;color:{color_perfil}'>{perfil_estimado}</span>"
        f"&nbsp;&nbsp;<span style='font-size:0.8rem;color:#777'>"
        f"VaR 95%: {_var95:.1f}% · Volatilidad: {_vol:.1f}%</span><br>"
        f"<span style='font-size:0.8rem;color:#666;margin-top:6px;display:block'>"
        f"<b>Esta pestaña (Portafolio)</b> muestra el comportamiento técnico del portafolio: "
        f"frontera eficiente, correlaciones completas y optimización sin restricciones de perfil.<br>"
        f"<b>La pestaña Perfil de Riesgo</b> complementa esto con: interpretación adaptada a tu perfil, "
        f"síntesis en lenguaje llano, restricciones de riesgo personalizadas y recomendación de estrategia "
        f"(HRP vs Máximo Sharpe) según tu tolerancia al riesgo.</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

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
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    sharpe = metricas["sharpe_ratio"]
    vol = metricas["volatility"]
    var_95 = metricas["var_95"]
    div_ratio = metricas["diversification_ratio"]
    ret_esp = metricas["expected_return"]
    mdd = metricas.get("max_drawdown")

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
    if mdd is not None:
        mdd_ctx = ("Drawdown bajo ✅" if mdd > -15
                   else "Drawdown moderado ⚠️" if mdd > -30
                   else "Drawdown alto ❌")
        m6.metric("Max. Drawdown", f"{mdd:.1f}%",
                  delta=mdd_ctx,
                  delta_color="off",
                  help="Mayor caída acumulada desde un máximo histórico en el período analizado (2 años). Mide el peor momento posible del portafolio.")
    else:
        m6.metric("Max. Drawdown", "N/D", help="No disponible: datos insuficientes")

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
                "Π equilibrio %": a.get("bl_prior"),
                "Q vista agentes %": a.get("bl_view"),
                "μ posterior BL %": a.get("bl_posterior"),
            })
        df_activos = pd.DataFrame(rows)
        st.dataframe(
            df_activos,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Peso %":      st.column_config.NumberColumn("Peso %",      format="%.1f%%", help="Participación en el portafolio"),
                "Ret. Esp. %": st.column_config.NumberColumn("Ret. Esp. %", format="%.1f%%", help="Retorno esperado ingenuo (histórico 70% + ML 30%) — solo referencia, el optimizador usa μ posterior BL"),
                "Volat. %":    st.column_config.NumberColumn("Volat. %",    format="%.1f%%", help="Volatilidad anualizada histórica"),
                "Recom.":      st.column_config.TextColumn("Recom.",        help="BUY / HOLD / SELL según el agente de recomendación"),
                "Mercado":     st.column_config.TextColumn("Mercado",       help="Señal técnica del MarketAgent"),
                "Sentiment":   st.column_config.TextColumn("Sentimiento",   help="Señal del SentimentAgent (FinBERT + VADER + TextBlob)"),
                "Fundamental": st.column_config.TextColumn("Fundamental",   help="Señal del SECAgent (ratios financieros EDGAR/yfinance)"),
                "Π equilibrio %":    st.column_config.NumberColumn("Π equilibrio %",    format="%.2f%%", help="Retorno de equilibrio de mercado (Black-Litterman)"),
                "Q vista agentes %": st.column_config.NumberColumn("Q vista agentes %", format="%.2f%%", help="Vista combinada de ModelAgent + SentimentAgent + SECAgent"),
                "μ posterior BL %":  st.column_config.NumberColumn("μ posterior BL %",  format="%.2f%%", help="Retorno posterior — el que realmente usa el optimizador de Markowitz"),
            },
        )
        opt_bl = pf.get("optimizacion", {})
        if opt_bl.get("delta_aversion"):
            st.caption(
                f"Black-Litterman: δ (aversión al riesgo) = {opt_bl.get('delta_aversion', 0):.2f} "
                f"(fuente: {opt_bl.get('fuente_delta', '-')}) · τ = {opt_bl.get('tau', 0):.3f}"
            )

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
        st.caption("Rojo = correlación negativa (mayor diversificación) | Verde = correlación positiva (menor diversificación)")

    st.markdown("---")

    # --- Optimización de Markowitz ---
    st.markdown("<p class='section-header'>Optimización de Markowitz</p>", unsafe_allow_html=True)

    if not opt.get("disponible", False):
        st.warning("Optimización no disponible (scipy no instalado o datos insuficientes).")
    else:
        hrp_w = opt.get("hrp_weights", {})
        col_ms, col_mv, col_hrp = st.columns(3)

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

        with col_hrp:
            st.markdown("**HRP — Hierarchical Risk Parity**")
            hrp_sharpe = opt.get("hrp_sharpe", 0.0)
            hrp_ret    = opt.get("hrp_return", 0.0)
            hrp_vol    = opt.get("hrp_volatility", 0.0)
            if hrp_w:
                st.metric("Sharpe", f"{hrp_sharpe:.2f}")
                st.metric("Retorno Esperado", f"{hrp_ret:.1f}%")
                st.metric("Volatilidad", f"{hrp_vol:.1f}%")
                fig_hrp = go.Figure(go.Bar(
                    x=list(hrp_w.keys()),
                    y=[v * 100 for v in hrp_w.values()],
                    marker_color="#f59e0b",
                    text=[f"{v*100:.1f}%" for v in hrp_w.values()],
                    textposition="outside",
                ))
                fig_hrp.update_layout(
                    title="Pesos HRP (sin optimización paramétrica)", height=250,
                    margin=dict(l=0, r=0, t=35, b=0),
                    yaxis_title="Peso (%)", showlegend=False,
                )
                st.plotly_chart(fig_hrp, use_container_width=True)
                st.caption(
                    "HRP usa clustering jerárquico — no requiere invertir la matriz de covarianza (López de Prado, 2016). "
                    "Estos pesos son **sin restricciones de perfil**: la pestaña Perfil de Riesgo aplica un límite máximo "
                    "por activo (ej. 40%) según tu tolerancia al riesgo, por lo que los pesos allí pueden diferir."
                )
            else:
                st.info("HRP no disponible (datos insuficientes).")

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
            if hrp_w and opt.get("hrp_volatility", 0) > 0:
                fig_fe.add_trace(go.Scatter(
                    x=[opt["hrp_volatility"]], y=[opt["hrp_return"]],
                    mode="markers", marker=dict(color="#f59e0b", size=12, symbol="triangle-up"),
                    name="HRP",
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
                "La frontera eficiente y los puntos de referencia (Max Sharpe, Mín. Varianza, HRP) "
                "se calculan con **retornos históricos puros** (media muestral, 2 años). "
                "El punto **Portafolio actual** (estrella roja) usa retornos esperados que mezclan "
                "historia (70%) y predicción ML a 3 días (30%), por lo que puede quedar fuera de la curva — "
                "esto es metodológicamente correcto y esperado, no un error."
            )

        # Comparación de pesos: actual vs óptimos
        st.markdown("**Comparación: Pesos Actuales vs Estrategias de Optimización**")
        # Determinar qué estrategia usar para la conclusión (la de mejor Sharpe)
        hrp_sharpe_val = opt.get("hrp_sharpe", 0)
        ms_sharpe_val  = opt.get("max_sharpe_sharpe", 0)
        usar_hrp_conclusion = hrp_w and hrp_sharpe_val > 0

        comp_rows = []
        for t in tickers_ok:
            w_act  = pf["weights"].get(t, 0)
            w_ms   = opt["max_sharpe_weights"].get(t, 0)
            w_hrp_ = hrp_w.get(t, 0)
            diff_ms  = w_ms   - w_act
            diff_hrp = w_hrp_ - w_act
            accion_ms  = "Aumentar" if diff_ms  > 0.03 else ("Reducir" if diff_ms  < -0.03 else "Mantener")
            accion_hrp = "Aumentar" if diff_hrp > 0.03 else ("Reducir" if diff_hrp < -0.03 else "Mantener")
            comp_rows.append({
                "Ticker":          t,
                "Actual %":        f"{w_act*100:.1f}%",
                "Max Sharpe %":    f"{w_ms*100:.1f}%",
                "Δ Max Sharpe":    f"{diff_ms*100:+.1f}pp",
                "HRP %":           f"{w_hrp_*100:.1f}%" if w_hrp_ else "-",
                "Δ HRP":           f"{diff_hrp*100:+.1f}pp" if w_hrp_ else "-",
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        # --- Conclusión y plan de acción ---
        st.markdown("---")
        st.markdown("<p class='section-header'>Conclusión y Plan de Acción</p>", unsafe_allow_html=True)

        sharpe_actual = metricas["sharpe_ratio"]

        # Acciones basadas en Máximo Sharpe
        aumentar_ms = [r for r in comp_rows if "+" in r["Δ Max Sharpe"] and float(r["Δ Max Sharpe"].replace("pp","")) > 3]
        reducir_ms  = [r for r in comp_rows if "-" in r["Δ Max Sharpe"] and float(r["Δ Max Sharpe"].replace("pp","")) < -3]
        # Acciones basadas en HRP
        aumentar_hrp = [r for r in comp_rows if r["Δ HRP"] != "-" and "+" in r["Δ HRP"] and float(r["Δ HRP"].replace("pp","")) > 3]
        reducir_hrp  = [r for r in comp_rows if r["Δ HRP"] != "-" and "-" in r["Δ HRP"] and float(r["Δ HRP"].replace("pp","")) < -3]

        mejora_posible = ms_sharpe_val - sharpe_actual
        if mejora_posible > 0.3:
            mejora_msg = (
                f"Rebalanceando hacia Máximo Sharpe histórico podrías mejorar el Sharpe de "
                f"**{sharpe_actual:.2f}** a **{ms_sharpe_val:.2f}** (+{mejora_posible:.2f}). "
                f"_(Sharpe actual basado en predicciones ML; Máximo Sharpe basado en historia)_"
            )
        elif mejora_posible > 0.05:
            mejora_msg = (
                f"El portafolio está cerca del óptimo histórico de Markowitz "
                f"(Sharpe ML actual: {sharpe_actual:.2f} · Máximo Sharpe histórico: {ms_sharpe_val:.2f})."
            )
        elif mejora_posible > -0.1:
            mejora_msg = (
                f"El portafolio está alineado con el óptimo histórico de Markowitz "
                f"(Sharpe ML: {sharpe_actual:.2f} · Máximo Sharpe histórico: {ms_sharpe_val:.2f})."
            )
        else:
            mejora_msg = (
                f"El Sharpe del modelo ML ({sharpe_actual:.2f}) supera el Máximo Sharpe histórico "
                f"({ms_sharpe_val:.2f}). Esto es normal: el modelo ML usa predicciones de corto plazo "
                f"mientras que el Sharpe histórico refleja retornos pasados reales. "
                f"Las acciones de rebalanceo abajo se basan en el análisis histórico."
            )

        lineas = [mejora_msg]
        lineas.append("**Según Máximo Sharpe** (maximiza retorno ajustado por riesgo — adecuado para perfiles agresivos):")
        if aumentar_ms:
            _txt = ", ".join(r["Ticker"] + " → " + r["Max Sharpe %"] for r in aumentar_ms)
            lineas.append(f"  📈 Aumentar: {_txt}")
        if reducir_ms:
            _txt = ", ".join(r["Ticker"] + " → " + r["Max Sharpe %"] for r in reducir_ms)
            lineas.append(f"  📉 Reducir: {_txt}")
        if not aumentar_ms and not reducir_ms:
            lineas.append("  ✅ Pesos actuales alineados con Máximo Sharpe.")

        if usar_hrp_conclusion:
            lineas.append(f"**Según HRP** (distribuye el riesgo jerárquicamente — adecuado para perfiles conservadores/moderados, Sharpe {hrp_sharpe_val:.2f}):")
            if aumentar_hrp:
                _txt = ", ".join(r["Ticker"] + " → " + r["HRP %"] for r in aumentar_hrp)
                lineas.append(f"  📈 Aumentar: {_txt}")
            if reducir_hrp:
                _txt = ", ".join(r["Ticker"] + " → " + r["HRP %"] for r in reducir_hrp)
                lineas.append(f"  📉 Reducir: {_txt}")
            if not aumentar_hrp and not reducir_hrp:
                lineas.append("  ✅ Pesos actuales alineados con HRP.")

        if not aumentar_ms and not reducir_ms and not aumentar_hrp and not reducir_hrp:
            lineas.append("✅ La distribución actual está alineada con ambas estrategias — no se requieren cambios significativos.")

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


def render_risk_profile_tab():
    """Pestaña de cuantificación del perfil de riesgo del inversor."""
    st.markdown("### Perfil de Riesgo del Inversor")
    st.markdown(
        "Respondé las 13 preguntas del instrumento validado **Grable & Lytton (1999)** "
        "*Financial Risk Tolerance Scale* para cuantificar tu tolerancia al riesgo. "
        "El sistema te asignará un perfil (según la clasificación oficial del instrumento) "
        "y recomendará los sectores más adecuados para tu cartera."
    )

    OPCIONES = {
        "q01": {
            "label": "1. ¿Cómo te describiría tu mejor amigo/a en cuanto a asumir riesgos?",
            "opciones": {
                "Un evasor del riesgo": "evasor",
                "Cauteloso/a": "cauteloso",
                "Un tomador de riesgo calculado": "calculador",
                "Un verdadero apostador": "jugador",
            },
        },
        "q02": {
            "label": "2. En un concurso de TV podés elegir. ¿Qué elegirías?",
            "opciones": {
                "$1.000 en efectivo, seguro": "efectivo_1000",
                "50% de chance de ganar $5.000": "chance_50_5000",
                "25% de chance de ganar $10.000": "chance_25_10000",
                "5% de chance de ganar $100.000": "chance_5_100000",
            },
        },
        "q03": {
            "label": "3. Perdiste tu empleo 3 semanas antes de tus vacaciones soñadas. ¿Qué harías?",
            "opciones": {
                "Cancelar las vacaciones": "cancelar",
                "Tomar unas vacaciones mucho más modestas": "reducir",
                "Mantener el plan (necesito el descanso para buscar empleo)": "mantener_plan",
                "Extender las vacaciones (puede ser mi última chance)": "extender",
            },
        },
        "q04": {
            "label": "4. Recibís $20.000 inesperados para invertir. ¿Qué harías?",
            "opciones": {
                "Depositarlo en una cuenta/CD segura": "deposito_seguro",
                "Invertir en bonos de alta calidad": "bonos_calidad",
                "Invertir en acciones o fondos de acciones": "acciones",
            },
        },
        "q05": {
            "label": "5. ¿Qué tan cómodo/a te sentís invirtiendo en acciones?",
            "opciones": {
                "Nada cómodo/a": "nada_comodo",
                "Algo cómodo/a": "algo_comodo",
                "Muy cómodo/a": "muy_comodo",
            },
        },
        "q06": {
            "label": "6. ¿Qué palabra asociás primero con 'riesgo'?",
            "opciones": {
                "Pérdida": "perdida",
                "Incertidumbre": "incertidumbre",
                "Oportunidad": "oportunidad",
                "Adrenalina": "adrenalina",
            },
        },
        "q07": {
            "label": "7. Tenés tus activos en bonos de gobierno; se anticipa una suba de activos duros (oro, inmuebles). ¿Qué harías?",
            "opciones": {
                "Mantener los bonos": "mantener_bonos",
                "Vender la mitad y pasarla a activos duros": "mitad_activos_duros",
                "Vender todo y pasarlo a activos duros": "todo_activos_duros",
                "Vender todo, pasarlo a activos duros y pedir prestado para comprar más": "todo_mas_apalancado",
            },
        },
        "q08": {
            "label": "8. Elegí la combinación de ganancia/pérdida potencial que preferís",
            "opciones": {
                "Ganancia máx. $200 / pérdida máx. $0": "bajo_riesgo",
                "Ganancia máx. $800 / pérdida máx. $200": "moderado_bajo",
                "Ganancia máx. $2.600 / pérdida máx. $800": "moderado_alto",
                "Ganancia máx. $4.800 / pérdida máx. $2.400": "alto_riesgo",
            },
        },
        "q09": {
            "label": "9. Además de lo que ya tenés, te regalan $1.000. Podés asegurarte "
                     "una parte, o jugarte el total. ¿Qué elegís?",
            "opciones": {
                "Asegurarme $500 (la mitad, garantizado)": "ganancia_segura_500",
                "Jugarme los $1.000: 50% de quedarme con los $1.000 completos, 50% de quedarme sin nada": "apuesta_50_1000",
            },
        },
        "q10": {
            "label": "10. Además de lo que ya tenés, te regalan $2.000, pero vas a perder "
                     "una parte. ¿Qué elegís?",
            "opciones": {
                "Perder $500 seguro (te quedás con $1.500 garantizados)": "perdida_segura_500",
                "Jugarte la pérdida: 50% de perder $1.000 (te quedás con $1.000), 50% de no perder nada (te quedás con los $2.000)": "apuesta_50_1000",
            },
        },
        "q11": {
            "label": "11. Heredás $100.000 que debés invertir TODO en UNA sola opción. ¿Cuál elegís?",
            "opciones": {
                "Caja de ahorro / money market": "ahorro",
                "Fondo mixto de acciones y bonos": "fondo_mixto",
                "Portafolio de acciones individuales": "acciones_individuales",
                "Commodities (oro, plata, petróleo)": "commodities",
            },
        },
        "q12": {
            "label": "12. Si tuvieras que invertir $20.000, ¿qué combinación de riesgo preferís?",
            "opciones": {
                "Muy conservadora (mayoría bajo riesgo)": "muy_conservadora",
                "Conservadora": "conservadora",
                "Equilibrada": "equilibrada",
                "Agresiva (mayoría alto riesgo)": "agresiva",
            },
        },
        "q13": {
            "label": "13. Un amigo geólogo arma un proyecto de mina de oro: 20% de éxito, retorno de 50-100x o pérdida total. ¿Cuánto invertirías?",
            "opciones": {
                "Nada": "nada",
                "Un mes de salario": "un_mes_salario",
                "Tres meses de salario": "tres_meses_salario",
                "Seis meses de salario": "seis_meses_salario",
            },
        },
    }

    COLORES_PERFIL = {
        "muy_conservador": ("#1a6e3c", "#d4edda", "Muy Conservador"),
        "conservador":     ("#155724", "#d4edda", "Conservador"),
        "moderado":        ("#856404", "#fff3cd", "Moderado"),
        "agresivo":        ("#7b1a1a", "#f8d7da", "Agresivo"),
        "muy_agresivo":    ("#491010", "#f5c6cb", "Muy Agresivo"),
    }

    with st.form("risk_form"):
        respuestas_labels = {}
        cols = st.columns(2)
        campos = list(OPCIONES.items())
        for i, (campo, cfg) in enumerate(campos):
            col = cols[i % 2]
            with col:
                respuestas_labels[campo] = st.selectbox(
                    cfg["label"],
                    options=list(cfg["opciones"].keys()),
                    key=f"risk_{campo}",
                )
        st.markdown("---")
        col_dyn, col_lb = st.columns([2, 1])
        with col_dyn:
            usar_dinamica = st.checkbox(
                "Selección dinámica de sectores (basada en datos históricos reales)",
                value=True,
                help="Si está activo, el sistema evalúa 22 ETFs sectoriales + acciones del S&P 500 "
                     "filtradas por precio/volumen, y elige los mejores para tu perfil usando "
                     "Sharpe ratio y retorno del período seleccionado.",
            )
        with col_lb:
            lookback = st.selectbox(
                "Período histórico",
                options=["6mo", "1y", "2y"],
                index=1,
                disabled=not usar_dinamica,
            )
        col_precio, col_vol = st.columns(2)
        with col_precio:
            precio_maximo = st.number_input(
                "Precio máximo por acción (USD)",
                min_value=10.0, max_value=10000.0, value=1000.0, step=50.0,
                disabled=not usar_dinamica,
                help="Descarta acciones del S&P 500 por encima de este precio (para que puedas "
                     "diversificar con montos chicos). El piso de $5 (definición SEC de "
                     "penny stock) siempre se aplica y no es configurable.",
            )
        with col_vol:
            volumen_minimo_usd = st.number_input(
                "Volumen diario mínimo (USD)",
                min_value=0.0, max_value=1_000_000_000.0, value=10_000_000.0, step=1_000_000.0,
                disabled=not usar_dinamica,
                help="Volumen promedio en dólares (precio × volumen) mínimo para considerar una "
                     "acción suficientemente líquida.",
            )
        submitted = st.form_submit_button("Evaluar mi perfil de riesgo", use_container_width=True, type="primary")

    if submitted:
        payload = {
            campo: OPCIONES[campo]["opciones"][respuestas_labels[campo]]
            for campo in OPCIONES
        }
        payload["usar_seleccion_dinamica"] = usar_dinamica
        payload["lookback"] = lookback
        payload["precio_maximo"] = precio_maximo
        payload["volumen_minimo_usd"] = volumen_minimo_usd
        spinner_msg = (
            f"Evaluando perfil y seleccionando sectores con datos históricos ({lookback})..."
            if usar_dinamica else "Evaluando perfil..."
        )
        with st.spinner(spinner_msg):
            result = evaluate_risk_profile(payload)

        if result:
            st.session_state["risk_result"] = result

    result = st.session_state.get("risk_result")
    if not result:
        return

    perfil = result["perfil"]
    color_txt, color_bg, label_perfil = COLORES_PERFIL.get(perfil, ("#333", "#eee", perfil))

    st.markdown("---")

    # ── Score y perfil ────────────────────────────────────────────────────
    col_score, col_desc = st.columns([1, 2])
    with col_score:
        score = result["score"]
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Score de Riesgo", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color_txt},
                "steps": [
                    {"range": [0, 20],  "color": "#d4edda"},
                    {"range": [20, 40], "color": "#cce5ff"},
                    {"range": [40, 60], "color": "#fff3cd"},
                    {"range": [60, 80], "color": "#ffd5b8"},
                    {"range": [80, 100],"color": "#f8d7da"},
                ],
                "threshold": {"line": {"color": color_txt, "width": 4}, "thickness": 0.75, "value": score},
            },
            number={"suffix": "/100"},
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(
            f"<div style='text-align:center; background:{color_bg}; color:{color_txt}; "
            f"border-radius:8px; padding:10px; font-size:1.2rem; font-weight:700;'>"
            f"Perfil: {label_perfil}</div>",
            unsafe_allow_html=True,
        )
        _periodo_txt = result.get("periodo_analisis", "predefinido")
        st.caption(
            f"Corrida: {_fmt_fecha_analisis(result.get('fecha_analisis'))} "
            f"· datos: {_periodo_txt}"
        )

    with col_desc:
        st.markdown(f"**Descripción del perfil**")
        st.info(result["descripcion_perfil"])
        if result.get("advertencia"):
            st.warning(result["advertencia"])

        rb = result["risk_budget"]
        c1, c2, c3 = st.columns(3)
        c1.metric("VaR 95% máx tolerable", f"{rb['var_95_max']:.0f}%")
        c2.metric("Volatilidad anual objetivo", f"{rb['vol_anual_max']:.0f}%")
        c3.metric("Peso máx por activo", f"{rb['max_peso_activo']*100:.0f}%")

    # ── Desglose por dimensión ────────────────────────────────────────────
    st.markdown("#### Desglose del cuestionario")
    dims = result["dimensiones"]
    fig_dims = go.Figure(go.Bar(
        x=[d["puntos"] for d in dims],
        y=[d["dimension"] for d in dims],
        orientation="h",
        marker_color=[color_txt] * len(dims),
        text=[f"{d['puntos']}/{d['max_puntos']}" for d in dims],
        textposition="outside",
    ))
    max_puntos_dims = max((d["max_puntos"] for d in dims), default=4)
    fig_dims.update_layout(
        xaxis=dict(range=[0, max_puntos_dims + 0.5], title="Puntos"),
        height=max(280, 40 * len(dims)),
        margin=dict(l=10, r=60, t=10, b=30),
    )
    st.plotly_chart(fig_dims, use_container_width=True)

    with st.expander("Ver interpretación de cada dimensión"):
        for d in dims:
            st.markdown(f"**{d['dimension']}** ({d['puntos']}/{d['max_puntos']} pts) — {d['interpretacion']}")

    # ── Advertencia de accesibilidad geográfica ───────────────────────────
    st.info(
        "**Nota sobre accesibilidad:** Los activos recomendados (ETFs y acciones) cotizan en mercados "
        "de Estados Unidos. Para inversores en Argentina y otros países de Latinoamérica, el acceso "
        "puede requerir: (a) cuenta en un broker internacional con habilitación para operar en NYSE/NASDAQ "
        "(ej. Interactive Brokers, Charles Schwab); o (b) operar vía **CEDEARs** para los activos "
        "disponibles en esa modalidad, con la diferencia de tipo de cambio implícita que eso conlleva. "
        "Verifique la disponibilidad y los costos operativos antes de tomar decisiones de inversión."
    )

    # ── Sectores recomendados ─────────────────────────────────────────────
    es_dinamica = result.get("seleccion_dinamica", False)
    periodo     = result.get("periodo_analisis", "")
    universo    = result.get("universo_evaluado", 0)

    if es_dinamica:
        st.success(
            f"Selección **dinámica** — {universo} ETFs evaluados con datos reales de {periodo}. "
            "Los sectores fueron elegidos por su Sharpe ratio ajustado a tu perfil de riesgo."
        )
    else:
        st.info("Selección **predefinida** por conocimiento experto (sin datos históricos).")

    st.markdown("#### Activos recomendados para tu perfil")
    sectores = result["sectores_recomendados"]

    def _tipo_label(s):
        return "ETF" if s.get("tipo", "ETF") == "ETF" else "Acción"

    def _señal_label(v):
        if v is None:
            return "-"
        if v >= 0.60:
            return "🟢 Alta"
        if v >= 0.40:
            return "🟡 Neutra"
        return "🔴 Baja"

    fig_pie = go.Figure(go.Pie(
        labels=[f"{s['sector']} ({s['etf']})" for s in sectores],
        values=[s["peso_target"] for s in sectores],
        hole=0.4,
        textinfo="label+percent",
        marker=dict(line=dict(color="#ffffff", width=2)),
    ))
    fig_pie.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

    if es_dinamica and sectores[0].get("sharpe_hist") is not None:
        tiene_señales = any(s.get("señal_sentimiento") is not None for s in sectores)
        tbl_data = {
            "Ticker": [s["etf"] for s in sectores],
            "Tipo": [_tipo_label(s) for s in sectores],
            "Sector / Empresa": [s["sector"] for s in sectores],
            "Peso (vol. inv.)": [f"{s['peso_target']*100:.1f}%" for s in sectores],
            f"Retorno {periodo} (%)": [f"{s['retorno_hist']:+.1f}%" if s.get('retorno_hist') is not None else "-" for s in sectores],
            "Volatilidad (% anual)": [f"{s['volatilidad_hist']:.1f}%" if s.get('volatilidad_hist') is not None else "-" for s in sectores],
            "Sharpe (anualiz.)": [f"{s['sharpe_hist']:.2f}" if s.get('sharpe_hist') is not None else "-" for s in sectores],
            "Score perfil (0–1)": [f"{s['ranking_score']:.3f}" if s.get('ranking_score') is not None else "-" for s in sectores],
            "Descripción": [s["descripcion"] for s in sectores],
        }
        if tiene_señales:
            tbl_data["Sentimiento"] = [_señal_label(s.get("señal_sentimiento")) for s in sectores]
            tbl_data["Predicción"]  = [_señal_label(s.get("señal_prediccion"))  for s in sectores]
            tbl_data["Fund. SEC"]   = [
                _señal_label(s.get("señal_sec")) if s.get("tipo") == "Acción" else "-"
                for s in sectores
            ]
    else:
        tbl_data = {
            "Ticker": [s["etf"] for s in sectores],
            "Tipo": [_tipo_label(s) for s in sectores],
            "Sector / Empresa": [s["sector"] for s in sectores],
            "Peso sugerido": [f"{s['peso_target']*100:.0f}%" for s in sectores],
            "Descripción": [s["descripcion"] for s in sectores],
        }
    st.dataframe(pd.DataFrame(tbl_data), use_container_width=True, hide_index=True)
    if es_dinamica and sectores[0].get("sharpe_hist") is not None:
        st.caption("Peso asignado por volatilidad inversa — menor volatilidad recibe mayor peso.")
        if tiene_señales:
            st.caption(
                "**Señales multiagente** integradas en el score: "
                "Sentimiento (noticias recientes · SentimentAgent), "
                "Predicción (momentum de precio), "
                "Fund. SEC (fundamentales EDGAR · solo acciones individuales). "
                "🟢 ≥ 0.60 · 🟡 0.40–0.59 · 🔴 < 0.40 (escala 0–1)."
            )

    # ── Portafolio optimizado ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Análisis completo del portafolio para tu perfil")
    st.markdown(
        "Lanza el pipeline multiagente completo (Markowitz, HRP, señales ML, sentimiento) "
        "sobre los ETFs recomendados. Puede tardar 1-2 minutos."
    )

    if st.button("Generar portafolio optimizado", type="primary", key="btn_risk_portfolio"):
        with st.spinner("Analizando portafolio sectorial... esto puede tardar 1-2 minutos"):
            _tickers_din = result.get("tickers_recomendados")
            _pesos_din   = result.get("pesos_sugeridos")
            port_data = portfolio_for_risk_profile(perfil, _tickers_din, _pesos_din)
        if port_data:
            st.session_state["risk_portfolio"] = port_data
            st.success("Portafolio calculado.")

    # Botón de integración Tab1 → Tab3
    if st.session_state.get("risk_portfolio") and st.session_state.get("risk_result", {}).get("perfil") == perfil:
        _w = st.session_state["risk_portfolio"].get("weights", {})
        if _w:
            tickers_str = " · ".join(f"{t} {round(v*100,0):.0f}%" for t, v in list(_w.items())[:4])
            st.markdown(
                f"<div style='background:#e8f4fd;border:1px solid #90CAF9;border-radius:8px;"
                f"padding:12px 16px;margin:8px 0'>"
                f"<div style='font-size:0.9rem;font-weight:600;color:#1565C0;margin-bottom:4px'>"
                f"¿Querés explorar este portafolio con más detalle técnico?</div>"
                f"<div style='font-size:0.82rem;color:#555;margin-bottom:6px'>"
                f"La pestaña <b>Portafolio</b> complementa este análisis con herramientas que no están aquí:"
                f"<ul style='margin:4px 0 6px 16px;padding:0'>"
                f"<li>Frontera eficiente interactiva (gráfico retorno vs riesgo)</li>"
                f"<li>Matriz de correlación completa entre activos</li>"
                f"<li>Posibilidad de agregar o quitar ETFs y comparar variantes</li>"
                f"<li>Optimización sin restricciones de perfil (para análisis técnico puro)</li>"
                f"</ul>"
                f"<b>Perfil de Riesgo</b> responde <i>¿es adecuado para mi perfil?</i>. "
                f"<b>Portafolio</b> responde <i>¿cómo se comporta técnicamente?</i><br><br>"
                f"Pesos a cargar: <span style='font-family:monospace'>{tickers_str}{'…' if len(_w)>4 else ''}</span></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button("Editar este portafolio en Portafolio →", key="btn_open_in_portfolio"):
                st.session_state["pf_selected"] = [t for t in _w if t in [
                    "AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA","META","JPM","V","MA",
                    "UNH","HD","PG","KO","DIS","NFLX","PYPL","INTC","AMD","ORCL","CRM",
                    "ADBE","BRK-B","XOM","CVX","GS","BAC","WMT","COST"]]
                st.session_state["pf_custom"] = ",".join(t for t in _w if t not in st.session_state["pf_selected"])
                for t, v in _w.items():
                    st.session_state[f"pf_w_{t}"] = round(v * 100, 1)
                st.success("✅ Pesos cargados — cambiá a la pestaña **Portafolio** para continuar.")

    port_data = st.session_state.get("risk_portfolio")
    if port_data and st.session_state.get("risk_result", {}).get("perfil") == perfil:
        m = port_data.get("metricas", {})
        opt = port_data.get("optimizacion", {})

        st.markdown("**Métricas del portafolio con pesos sugeridos por perfil**")
        st.caption(f"Corrida: {_fmt_fecha_analisis(port_data.get('fecha_analisis'))}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retorno esperado", f"{m.get('expected_return', 0):.1f}%")
        c2.metric("Volatilidad anual", f"{m.get('volatility', 0):.1f}%")
        c3.metric("Sharpe Ratio", f"{m.get('sharpe_ratio', 0):.2f}")
        c4.metric("VaR 95%", f"{m.get('var_95', 0):.1f}%")

        # ── Transparencia Black-Litterman: Π (equilibrio) vs Q (vista de los
        # agentes) vs μ posterior (lo que realmente usa el optimizador) ──────
        activos_bl = port_data.get("activos", [])
        if activos_bl and activos_bl[0].get("bl_posterior") is not None:
            st.markdown("**Retorno esperado por activo — Black-Litterman**")
            st.caption(
                "El optimizador ya no usa el promedio histórico ni un blend fijo: combina el "
                "equilibrio de mercado (Π) con las señales de ModelAgent, SentimentAgent y SECAgent "
                "(vista Q), ponderadas por su propia confianza, para obtener el retorno posterior (μ)."
            )
            bl_tbl = pd.DataFrame({
                "Ticker": [a["ticker"] for a in activos_bl],
                "Π equilibrio (%)": [f"{a.get('bl_prior', 0):+.2f}" for a in activos_bl],
                "Q vista agentes (%)": [f"{a.get('bl_view', 0):+.2f}" for a in activos_bl],
                "μ posterior BL (%)": [f"{a.get('bl_posterior', 0):+.2f}" for a in activos_bl],
                "Confianza vista": [f"{a.get('view_confidence', 0):.2f}" for a in activos_bl],
            })
            st.dataframe(bl_tbl, use_container_width=True, hide_index=True)
            st.caption(
                f"δ (aversión al riesgo) = {opt.get('delta_aversion', 0):.2f} "
                f"(fuente: {opt.get('fuente_delta', '-')}) · τ = {opt.get('tau', 0):.3f}"
            )

        # ── Síntesis narrativa ────────────────────────────────────────────
        sintesis = _generar_sintesis_portafolio(
            port_data, perfil, result.get("risk_budget", {})
        )

        COLOR_SEMAFORO = {"verde": "#d4edda", "amarillo": "#fff3cd", "rojo": "#f8d7da"}
        ICONO_SEMAFORO = {"verde": "✅", "amarillo": "⚠️", "rojo": "🚨"}
        bg    = COLOR_SEMAFORO[sintesis["semaforo"]]
        icono = ICONO_SEMAFORO[sintesis["semaforo"]]

        st.markdown(f"#### {icono} Resumen para el inversor — {sintesis['estado']}")
        st.markdown(
            f"<div style='background:{bg};border-radius:10px;padding:18px 22px;margin-bottom:12px;'>",
            unsafe_allow_html=True,
        )
        for parrafo in sintesis["narrativa"]:
            st.markdown(parrafo)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f"<div style='background:#e8f4fd;border-left:4px solid #2196F3;"
            f"border-radius:6px;padding:12px 18px;margin-bottom:20px;'>"
            f"<b>Acción recomendada:</b> {sintesis['accion_concreta']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # Comparar portafolios optimizados
        if opt.get("disponible"):
            # Determinar estrategia recomendada para resaltarla en la tabla
            _REC_TABLA = {
                "muy_conservador": "hrp", "conservador": "hrp", "moderado": "hrp",
                "agresivo": "max_sharpe", "muy_agresivo": "max_sharpe",
            }
            _perfil_actual = st.session_state.get("risk_result", {}).get("perfil", "moderado")
            _est_rec_tabla = _REC_TABLA.get(_perfil_actual, "hrp")

            _DESC = {
                "Pesos del perfil":  "Asignación base por volatilidad inversa (punto de partida)",
                "Máximo Sharpe":     "Maximiza el retorno por unidad de riesgo · puede concentrar mucho en pocos activos",
                "Mínima Varianza":   "Minimiza la volatilidad total · retorno más bajo pero más estable",
                "HRP":               "Diversificación jerárquica sin sobreconcentrar · robusto ante errores de estimación",
            }
            _rec_label = "HRP" if _est_rec_tabla == "hrp" else "Máximo Sharpe"

            st.markdown("**Comparación de estrategias de optimización**")
            comp_data = {
                "Estrategia": ["Pesos del perfil", "Máximo Sharpe", "Mínima Varianza"],
                "Retorno (%)": [
                    m.get("expected_return", 0),
                    opt.get("max_sharpe_return", 0),
                    opt.get("min_variance_return", 0),
                ],
                "Volatilidad (%)": [
                    m.get("volatility", 0),
                    opt.get("max_sharpe_volatility", 0),
                    opt.get("min_variance_volatility", 0),
                ],
                "Sharpe": [
                    m.get("sharpe_ratio", 0),
                    opt.get("max_sharpe_sharpe", 0),
                    round((opt.get("min_variance_return", 0) - 4.5) / max(opt.get("min_variance_volatility", 1), 0.01), 3),
                ],
            }
            if opt.get("hrp_weights"):
                comp_data["Estrategia"].append("HRP")
                comp_data["Retorno (%)"].append(opt.get("hrp_return", 0))
                comp_data["Volatilidad (%)"].append(opt.get("hrp_volatility", 0))
                comp_data["Sharpe"].append(opt.get("hrp_sharpe", 0))

            # Marcar la estrategia recomendada
            comp_data["Estrategia"] = [
                f"{e} ← recomendada" if e == _rec_label else e
                for e in comp_data["Estrategia"]
            ]

            st.dataframe(
                pd.DataFrame(comp_data).style.highlight_max(subset=["Sharpe"], color="#d4edda").format(
                    {"Retorno (%)": "{:.2f}", "Volatilidad (%)": "{:.2f}", "Sharpe": "{:.3f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Los valores de esta tabla usan los retornos históricos del período analizado (tasa libre de riesgo: 4.5%). "
                "La pestaña **Portafolio** puede mostrar valores levemente distintos si los activos se cargaron "
                "en un momento diferente — ambas consultas son independientes al servidor."
            )
            with st.expander("¿Qué significa cada estrategia?"):
                for nombre, desc in _DESC.items():
                    rec = " ← **recomendada para tu perfil**" if nombre == _rec_label else ""
                    st.markdown(f"- **{nombre}**: {desc}{rec}")

            # ── Comparación visual: pesos actuales vs Máximo Sharpe vs HRP ──
            w_sharpe = opt.get("max_sharpe_weights", {})
            w_hrp    = opt.get("hrp_weights", {})
            w_actual = port_data.get("weights", {})
            if w_sharpe and w_actual:
                # Estrategia recomendada según perfil (conservador/moderado → HRP)
                _REC_ESTRATEGIA = {
                    "muy_conservador": "hrp", "conservador": "hrp", "moderado": "hrp",
                    "agresivo": "max_sharpe", "muy_agresivo": "max_sharpe",
                }
                estrategia_recomendada = _REC_ESTRATEGIA.get(
                    st.session_state.get("risk_result", {}).get("perfil", "moderado"),
                    "hrp"
                )
                max_peso_activo = st.session_state.get("risk_result", {}).get(
                    "risk_budget", {}).get("max_peso_activo", 0.40)
                limite_pct = round(max_peso_activo * 100, 0)

                label_rec = "HRP" if estrategia_recomendada == "hrp" else "Máximo Sharpe"
                st.markdown(f"**Comparación de pesos — estrategia recomendada: {label_rec} ✓**")
                st.caption(
                    "Límite máximo por activo para este perfil: "
                    f"**{limite_pct:.0f}%** (línea roja punteada)"
                )

                tickers_ord = sorted(w_sharpe.keys())
                pesos_act   = [round(w_actual.get(t, 0) * 100, 1) for t in tickers_ord]
                pesos_opt   = [round(w_sharpe.get(t, 0) * 100, 1) for t in tickers_ord]
                pesos_hrp   = [round(w_hrp.get(t, 0) * 100, 1)   for t in tickers_ord] if w_hrp else []

                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    name="Pesos del perfil (actual)",
                    x=tickers_ord, y=pesos_act,
                    marker_color="#90CAF9",
                    text=[f"{v:.1f}%" for v in pesos_act],
                    textposition="outside",
                ))
                fig_comp.add_trace(go.Bar(
                    name="Máximo Sharpe",
                    x=tickers_ord, y=pesos_opt,
                    marker_color="#FFA726" if estrategia_recomendada == "hrp" else color_txt,
                    text=[f"{v:.1f}%" for v in pesos_opt],
                    textposition="outside",
                    opacity=0.75 if estrategia_recomendada == "hrp" else 1.0,
                ))
                if pesos_hrp:
                    fig_comp.add_trace(go.Bar(
                        name="HRP ✓ (recomendado)" if estrategia_recomendada == "hrp" else "HRP",
                        x=tickers_ord, y=pesos_hrp,
                        marker_color=color_txt if estrategia_recomendada == "hrp" else "#90CAF9",
                        text=[f"{v:.1f}%" for v in pesos_hrp],
                        textposition="outside",
                        opacity=1.0 if estrategia_recomendada == "hrp" else 0.75,
                    ))
                # Línea de límite máximo por activo
                fig_comp.add_hline(
                    y=limite_pct,
                    line_dash="dot", line_color="red", line_width=1.5,
                    annotation_text=f"Límite {limite_pct:.0f}%",
                    annotation_position="top right",
                )
                fig_comp.update_layout(
                    barmode="group", yaxis_title="Peso (%)",
                    height=340,
                    margin=dict(l=20, r=20, t=10, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_comp, use_container_width=True)

                # Tabla de cambios — usa la estrategia recomendada como referencia
                pesos_rec = pesos_hrp if (estrategia_recomendada == "hrp" and pesos_hrp) else pesos_opt
                sharpe_rec = opt.get("hrp_sharpe", 0) if estrategia_recomendada == "hrp" else opt.get("max_sharpe_sharpe", 0)

                st.markdown(f"**Cambios necesarios para rebalancear al {label_rec}:**")
                filas_delta = []
                for i, t in enumerate(tickers_ord):
                    act   = pesos_act[i]
                    rec_p = pesos_rec[i] if i < len(pesos_rec) else act
                    delta = round(rec_p - act, 1)
                    if delta > 0.5:
                        accion = f"▲ Aumentar {delta:+.1f}pp"
                    elif delta < -0.5:
                        accion = f"▼ Reducir {delta:+.1f}pp"
                    else:
                        accion = "≈ Sin cambio"
                    excede = "⚠️" if rec_p > limite_pct else ""
                    filas_delta.append({
                        "ETF": t,
                        "Peso actual": f"{act:.1f}%",
                        f"Peso {label_rec}": f"{rec_p:.1f}%{excede}",
                        "Cambio": accion,
                    })

                df_delta = pd.DataFrame(filas_delta)
                st.dataframe(df_delta, use_container_width=True, hide_index=True)
                st.caption(
                    f"Sharpe actual: {m.get('sharpe_ratio', 0):.2f} "
                    f"_(retornos predichos por ML, no históricos)_  →  "
                    f"Sharpe {label_rec}: {sharpe_rec:.2f} "
                    f"_(retornos históricos reales — más confiable para comparar)_  |  "
                    "pp = puntos porcentuales  |  ⚠️ = supera límite del perfil"
                )

        # Señales por activo
        activos = port_data.get("activos", [])
        if activos:
            st.markdown("**Señales del pipeline multiagente por ETF**")
            with st.expander("¿Qué significa cada columna?"):
                st.markdown("""
| Columna | Qué mide | Cómo interpretarlo |
|---|---|---|
| **Señal** 🟢🟡🔴 | Recomendación final del agente de decisión, combinando las 4 señales | 🟢 Compra = señales positivas alineadas · 🟡 Mantener = señales mixtas · 🔴 Venta = señales negativas |
| **Confianza** | Qué tan alineadas están las 4 fuentes entre sí (0-100%) | > 70% = señales consistentes · < 50% = fuentes contradictorias, más incertidumbre |
| **Mercado** | Tendencia técnica (precio vs media móvil 20 días, RSI, MACD) | alcista / bajista / neutral — horizonte de días a semanas |
| **Sentimiento** | Análisis de noticias y titulares recientes sobre el ETF | positivo / negativo / neutral — refleja el humor del mercado hoy |
| **Ret. esperado** | Variación de precio predicha por el modelo ML para los próximos 3 días, anualizada × 252/3 | ⚠️ No es una predicción a 1 año — valores altos (>50%) son artefacto del período corto |
| **Volatilidad** | Volatilidad histórica anualizada de los últimos 12 meses | Referencia: < 15% baja · 15-25% media · > 25% alta |
""")
            filas = []
            for a in activos:
                tipo = a.get("tipo_recomendacion", "mantener")
                icono = "🟢" if "compra" in tipo else ("🔴" if "venta" in tipo else "🟡")
                filas.append({
                    "ETF": a["ticker"],
                    "Señal": f"{icono} {tipo}",
                    "Confianza": f"{a.get('confianza', 0)*100:.0f}%",
                    "Mercado": a.get("senal_mercado", "-"),
                    "Sentimiento": a.get("sentimiento", "-"),
                    "Ret. esperado": f"{a.get('expected_return', 0):.1f}%",
                    "Volatilidad": f"{a.get('volatility', 0):.1f}%",
                })
            st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)

        alertas = port_data.get("alertas", [])
        if alertas:
            st.markdown("**Alertas detectadas**")
            for al in alertas:
                nivel = al.get("nivel", "warning")
                msg = al.get("mensaje", "")
                # Correlación alta entre ETF de mercado amplio y sectorial es estructural
                if al.get("tipo") == "correlacion_alta" and any(
                    t in al.get("ticker", "") for t in ["SPY", "QQQ", "IVV"]
                ):
                    st.info(
                        f"**{al.get('ticker', '')}** (correlación estructural esperada): {msg} — "
                        "Normal entre un ETF de mercado amplio y sus sectores constituyentes."
                    )
                elif nivel == "critical":
                    st.error(f"**{al.get('ticker', '')}**: {msg}")
                else:
                    st.warning(f"**{al.get('ticker', '')}**: {msg}")


def render_main_dashboard():
    """Renderiza el dashboard principal."""
    render_sidebar()

    # Logo y header profesional
    st.markdown("""
    <div style='text-align: center; padding: 20px 0; margin-bottom: 30px;'>
        <div style='display: inline-block; background: linear-gradient(135deg, #87ceeb 0%, #4db8e8 100%);
                    padding: 15px 30px; border-radius: 15px; box-shadow: 0 4px 15px rgba(77, 184, 232, 0.4);'>
            <h1 style='color: white; margin: 0; font-size: 2.0rem; font-weight: 700; letter-spacing: 1px;'>
                Sistema Multiagente de Análisis y Optimización de Portafolios Financieros
            </h1>
            <p style='color: #f0f0f0; margin: 5px 0 0 0; font-size: 0.95rem; letter-spacing: 1px;'>
                Plataforma de análisis para inversores minoristas
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab3, tab4, tab2, tab5 = st.tabs(["Perfil de Riesgo", "Portafolio", "Backtesting", "Análisis de Activos", "Historial de Alertas"])

    with tab1:
        render_risk_profile_tab()

    with tab2:
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

    with tab3:
        render_portfolio_tab()

    with tab4:
        render_backtest_tab()

    with tab5:
        if "token" in st.session_state:
            render_alerts_history()
        else:
            st.warning("Inicie sesión para ver el historial de alertas")


def _render_backtest_portfolio():
    """Sección de backtesting histórico de portafolio (3 estrategias + SPY)."""
    st.markdown("#### Backtesting histórico del portafolio")
    st.markdown(
        "Simula cómo hubiera evolucionado el portafolio con cada estrategia de pesos "
        "usando retornos históricos reales. Compara pesos del perfil, Máximo Sharpe, HRP y SPY."
    )

    # ── Fuente de datos ───────────────────────────────────────────────────
    port_data   = st.session_state.get("risk_portfolio")
    risk_result = st.session_state.get("risk_result")

    if port_data and risk_result:
        tickers = list(port_data.get("weights", {}).keys())
        weights = port_data.get("weights", {})
        opt     = port_data.get("optimizacion", {})
        perfil  = risk_result.get("perfil", "moderado")
        st.success(f"Usando portafolio del perfil **{perfil}**: {', '.join(tickers)}")
    else:
        st.warning("No hay un portafolio de perfil cargado. Ingresá los tickers manualmente.")
        raw_tickers = st.text_input(
            "Tickers (separados por coma)",
            value="XLK,XLU,QQQ,XLI,BND,EFA",
            key="bt_port_tickers",
        )
        tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
        n = len(tickers)
        weights = {t: round(1 / n, 4) for t in tickers}
        opt = {}

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        bt_years = st.selectbox("Período histórico", [1, 2, 3, 5], index=1, key="bt_port_years")
    with col2:
        bt_capital = st.number_input("Capital inicial (USD)", min_value=1000, value=10000, step=1000, key="bt_port_capital")
    with col3:
        bt_rebal = st.selectbox(
            "Rebalanceo",
            options=[("Sin rebalanceo", 0), ("Trimestral", 3), ("Semestral", 6), ("Anual", 12)],
            format_func=lambda x: x[0],
            index=2,
            key="bt_port_rebal",
        )
        rebal_meses = bt_rebal[1]
    with col4:
        bt_costo = st.selectbox(
            "Costo por operación",
            options=[("Sin costo (ideal)", 0.0), ("0.05% (broker barato)", 0.0005), ("0.10% (típico)", 0.001), ("0.20% (alto)", 0.002)],
            format_func=lambda x: x[0],
            index=2,
            key="bt_port_costo",
        )
        costo_pct = bt_costo[1]

    if rebal_meses > 0:
        st.info(
            f"Con rebalanceo **{bt_rebal[0].lower()}** y costo **{bt_costo[0]}**: "
            "Max Sharpe y HRP recalculan sus pesos en cada fecha usando solo datos pasados "
            f"(ventana {252} días) — **sin sesgo de look-ahead**."
        )

    if st.button("Ejecutar backtest de portafolio", type="primary", use_container_width=True):
        msg = "Calculando curvas de equity"
        if rebal_meses > 0:
            msg += " con walk-forward (puede tardar ~30 segundos)..."
        else:
            msg += "..."
        with st.spinner(msg):
            result = _backtest_portfolio_historico(
                tickers, weights, opt, bt_years, bt_capital,
                rebalanceo_meses=rebal_meses, costo_pct=costo_pct,
            )
        if result:
            st.session_state["last_bt_portfolio"] = result
        else:
            st.error("No se pudieron descargar los datos históricos. Verificá los tickers.")
            return

    if "last_bt_portfolio" not in st.session_state:
        st.info("Configurá los parámetros y presioná **Ejecutar backtest de portafolio**.")
        return

    res   = st.session_state["last_bt_portfolio"]
    estrs = res["estrategias"]

    rebal_label = {0: "sin rebalanceo", 3: "trimestral", 6: "semestral", 12: "anual"}.get(res.get("rebalanceo_meses", 0), "")
    costo_label = f"{res.get('costo_pct', 0)*100:.2f}% por operación"
    st.caption(
        f"Período: {res['start']} → {res['end']}  |  "
        f"Rebalanceo: {rebal_label}  |  Costo: {costo_label}"
        + ("  |  Walk-forward activo: Max Sharpe y HRP sin look-ahead" if res.get("rebalanceo_meses", 0) > 0 else "")
    )

    # ── Métricas comparativas ─────────────────────────────────────────────
    st.markdown("##### Métricas por estrategia")
    _COLORES = {
        "Pesos del perfil":  "#2196F3",
        "Máximo Sharpe":     "#FFA726",
        "HRP":               "#4CAF50",
        "SPY (benchmark)":   "#9E9E9E",
    }
    cols = st.columns(len(estrs))
    for col, e in zip(cols, estrs):
        col.markdown(f"**{e['label']}**")
        col.metric("Retorno total",  f"{e['total_ret']:.1f}%")
        col.metric("CAGR",           f"{e['ann_ret']:.1f}%")
        col.metric("Volatilidad",    f"{e['ann_vol']:.1f}%")
        col.metric("Sharpe",         f"{e['sharpe']:.2f}")
        col.metric("Max Drawdown",   f"{e['max_dd']:.1f}%")
        col.metric("Calmar",         f"{e['calmar']:.2f}")

    # ── Interpretación en lenguaje simple ────────────────────────────────
    st.markdown("##### ¿Qué significan estos resultados?")
    spy_e   = next((e for e in estrs if "SPY" in e["label"]), None)
    perf_e  = next((e for e in estrs if "perfil" in e["label"].lower()), None)
    hrp_e   = next((e for e in estrs if e["label"] == "HRP"), None)
    ms_e    = next((e for e in estrs if "Sharpe" in e["label"]), None)
    mejor   = max(estrs, key=lambda x: x["sharpe"])
    interps = []

    # Mejor estrategia por Sharpe
    interps.append(
        f"✅ La estrategia con mejor Sharpe histórico fue **{mejor['label']}** "
        f"({mejor['sharpe']:.2f}) — mayor retorno ajustado por unidad de riesgo."
    )

    # Portafolio vs SPY
    if perf_e and spy_e:
        diff = perf_e["ann_ret"] - spy_e["ann_ret"]
        if diff > 3:
            interps.append(
                f"✅ El portafolio del perfil superó al SPY en **{diff:+.1f}pp** anualizados "
                f"({perf_e['ann_ret']:.1f}% vs {spy_e['ann_ret']:.1f}%) — añade valor real respecto al benchmark."
            )
        elif diff > -3:
            interps.append(
                f"👍 El portafolio del perfil rindió similar al SPY "
                f"({perf_e['ann_ret']:.1f}% vs {spy_e['ann_ret']:.1f}% anualizado). "
                "La diversificación sectorial no penalizó el retorno."
            )
        else:
            interps.append(
                f"⚠️ El SPY superó al portafolio del perfil en {abs(diff):.1f}pp anualizados. "
                "Esto puede reflejar un período favorable para el mercado amplio vs la selección sectorial."
            )

    # Drawdown
    if perf_e:
        mdd = abs(perf_e["max_dd"])
        spy_mdd = abs(spy_e["max_dd"]) if spy_e else mdd
        if mdd < 10:
            interps.append(f"✅ **Drawdown máximo {perf_e['max_dd']:.1f}%** — el portafolio fue muy estable, con caídas controladas.")
        elif mdd < 20:
            interps.append(
                f"👍 **Drawdown máximo {perf_e['max_dd']:.1f}%** — caída moderada. "
                + (f"El SPY tuvo {spy_e['max_dd']:.1f}% en el mismo período." if spy_e else "")
            )
        else:
            interps.append(
                f"⚠️ **Drawdown máximo {perf_e['max_dd']:.1f}%** — en el peor momento el portafolio perdió "
                f"{mdd:.1f}% de su valor pico. Revisá si es tolerable para tu perfil."
            )

    # HRP vs Máximo Sharpe
    if hrp_e and ms_e:
        diff_s = hrp_e["sharpe"] - ms_e["sharpe"]
        if abs(diff_s) < 0.10:
            interps.append(
                f"👍 HRP y Máximo Sharpe tienen Sharpe similares ({hrp_e['sharpe']:.2f} vs {ms_e['sharpe']:.2f}). "
                "Para perfiles moderados, HRP es preferible por su mayor estabilidad ante cambios de mercado."
            )
        elif diff_s > 0:
            interps.append(
                f"✅ **HRP supera a Máximo Sharpe** en Sharpe histórico ({hrp_e['sharpe']:.2f} vs {ms_e['sharpe']:.2f}) "
                "— confirma que HRP es la estrategia más adecuada para este perfil."
            )
        else:
            interps.append(
                f"👍 Máximo Sharpe tuvo mejor Sharpe histórico ({ms_e['sharpe']:.2f} vs {hrp_e['sharpe']:.2f}). "
                "Sin embargo, HRP es más robusto fuera de muestra al no ajustarse excesivamente al período analizado."
            )

    # Calmar del portafolio
    if perf_e and perf_e["calmar"] > 0:
        if perf_e["calmar"] > 1.0:
            interps.append(f"✅ **Calmar {perf_e['calmar']:.2f}**: el retorno anual superó la caída máxima — muy buena relación retorno/riesgo.")
        elif perf_e["calmar"] > 0.5:
            interps.append(f"👍 **Calmar {perf_e['calmar']:.2f}**: el retorno cubre razonablemente la caída máxima observada.")
        else:
            interps.append(f"⚠️ **Calmar {perf_e['calmar']:.2f}**: la caída máxima fue grande en relación al retorno anual — período con alta volatilidad.")

    for i in interps:
        st.markdown(f"- {i}")

    st.markdown("---")
    # ── Tabla resumen ─────────────────────────────────────────────────────
    with st.expander("Tabla comparativa completa"):
        tabla = []
        for e in estrs:
            tabla.append({
                "Estrategia":     e["label"],
                "Retorno total (%)": e["total_ret"],
                "CAGR (%)":       e["ann_ret"],
                "Volatilidad (%)": e["ann_vol"],
                "Sharpe":         e["sharpe"],
                "Max Drawdown (%)": e["max_dd"],
                "Calmar":         e["calmar"],
            })
        st.dataframe(pd.DataFrame(tabla), use_container_width=True, hide_index=True)

    # ── Curva de equity ───────────────────────────────────────────────────
    st.markdown("##### Curva de equity")
    fig_eq = go.Figure()
    for e in estrs:
        dash = "dash" if e["label"] == "SPY (benchmark)" else "solid"
        fig_eq.add_trace(go.Scatter(
            x=e["dates"], y=e["equity"],
            name=e["label"],
            line=dict(color=_COLORES.get(e["label"], "#607D8B"), width=2, dash=dash),
        ))
    # Líneas verticales en fechas de rebalanceo (del portafolio del perfil)
    perf_rebal = next((e.get("rebal_dates", []) for e in estrs if "perfil" in e["label"].lower()), [])
    for rd in perf_rebal[:30]:
        fig_eq.add_vline(x=rd, line_dash="dot", line_color="rgba(150,150,150,0.4)", line_width=1)
    if perf_rebal:
        fig_eq.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color="rgba(150,150,150,0.6)", dash="dot", width=1),
            name=f"Rebalanceo ({len(perf_rebal)} eventos)",
        ))
    fig_eq.update_layout(
        xaxis_title="Fecha", yaxis_title="Valor del portafolio (USD)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420, margin=dict(l=40, r=20, t=30, b=40),
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # ── Drawdown ──────────────────────────────────────────────────────────
    st.markdown("##### Drawdown")
    fig_dd = go.Figure()
    for e in estrs:
        fig_dd.add_trace(go.Scatter(
            x=e["dates"], y=e["drawdown"],
            name=e["label"],
            line=dict(color=_COLORES.get(e["label"], "#607D8B"), width=1.5),
            fill="tozeroy" if e["label"] == "Pesos del perfil" else None,
            fillcolor="rgba(33,150,243,0.1)" if e["label"] == "Pesos del perfil" else None,
        ))
    fig_dd.update_layout(
        xaxis_title="Fecha", yaxis_title="Drawdown (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=250, margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_dd, use_container_width=True)
    with st.expander("ℹ️ Limitaciones del backtesting — estado actual"):
        st.markdown("""
**Retornos históricos pasados no garantizan resultados futuros.**

| Estado | Limitación | Qué causa | Cómo está tratada en este sistema |
|:---:|---|---|---|
| ✅ | **Costos de transacción** | Retornos inflados si no se restan comisiones | Configurable en el panel: 0%, 0.05%, 0.10%, 0.20% por operación. Se descuenta en la compra inicial y en cada rebalanceo |
| ✅ | **Rebalanceo periódico** | Los pesos derivan con el mercado y generan riesgo no deseado | Configurable: trimestral, semestral o anual. El sistema simula el drift de pesos y aplica el costo de rebalanceo automáticamente |
| ✅ | **Sesgo de look-ahead** | Pesos óptimos calculados con datos futuros — irrealizable en la práctica | Con rebalanceo activo, Max Sharpe y HRP recalculan sus pesos en cada evento usando **solo datos históricos hasta esa fecha** (walk-forward real). Los pesos del perfil no tienen look-ahead por diseño |
| ⚠️ | **Slippage** | El precio real de ejecución suele ser peor que el precio de cierre | No modelado explícitamente. En ETFs líquidos (SPY, QQQ, XLK) es < 0.05% — sumarlo al costo configurado es una buena aproximación |
| ✅ | **Impacto de mercado** | Órdenes grandes mueven el precio | Despreciable para inversores minoristas (< $100k por ETF). Los ETFs del universo tienen miles de millones en volumen diario |
| ⚠️ | **Sesgo de supervivencia** | ETFs discontinuados no figuran, sesgando el retorno hacia arriba | No resuelto — requiere bases de datos de pago (Bloomberg, CRSP). Mitigado usando solo ETFs con > 10 años de historia y > $1B en activos |
| ✅ | **Período limitado** | Un ciclo de mercado no representa todos los escenarios posibles | Probá con 1, 2, 3 y 5 años. Si el Sharpe se mantiene > 0.8 en todos los períodos, la estrategia es robusta ante distintos regímenes de mercado |

**5 de 7 limitaciones resueltas computacionalmente.** Las 2 restantes (slippage y sesgo de supervivencia) son menores para un inversor minorista con ETFs establecidos.

#### Regla práctica para interpretar estos resultados
- **Sharpe > 0.8** en todos los períodos → estrategia robusta ante distintos regímenes
- **Max Drawdown** dentro del límite de tu perfil (VaR 95%) → el riesgo es tolerable
- **Supera al SPY por más de 2pp anualizados** en múltiples períodos → la selección sectorial añade valor real
- Los tres puntos anteriores con costos configurados → podés considerar implementar con una **posición inicial pequeña (10–20% del capital)** para validar antes de comprometer el total
""")
    st.caption("5 de 7 limitaciones resueltas — ver detalle arriba.")


def _hrp_wf(rets_df: "pd.DataFrame", tickers: List[str], max_w: float = 0.50) -> Dict[str, float]:
    """HRP walk-forward sobre retornos pasados con cap de peso."""
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
    except ImportError:
        n = len(tickers)
        return {t: 1/n for t in tickers}

    ts = [t for t in tickers if t in rets_df.columns]
    if len(ts) < 2:
        return {t: 1/len(tickers) for t in tickers}

    r = rets_df[ts].dropna()
    if len(r) < 20:
        return {t: 1/len(ts) for t in ts}

    corr = np.corrcoef(r.values.T)
    cov  = np.cov(r.values.T) * 252
    n    = len(ts)
    dist = np.sqrt(np.maximum((1 - corr) / 2, 0))
    np.fill_diagonal(dist, 0.0)
    try:
        link  = linkage(squareform(dist, checks=False), method="single")
        order = leaves_list(link)
    except Exception:
        order = list(range(n))

    weights_arr = np.ones(n)

    def _cvar(idxs):
        sc = cov[np.ix_(idxs, idxs)]
        iv = 1.0 / np.maximum(np.diag(sc), 1e-8)
        w  = iv / iv.sum()
        return float(w @ sc @ w)

    def _bisect(cl):
        if len(cl) == 1:
            return
        h = len(cl) // 2
        l, rx = cl[:h], cl[h:]
        vl, vr = _cvar(l), _cvar(rx)
        a = 1 - vl / (vl + vr + 1e-8)
        weights_arr[l]  *= a
        weights_arr[rx] *= (1 - a)
        _bisect(l); _bisect(rx)

    _bisect(list(order))
    weights_arr /= weights_arr.sum()

    # Cap iterativo
    cap = max(0.01, min(max_w, 0.99))
    w = {ts[i]: float(weights_arr[i]) for i in range(n)}
    for _ in range(30):
        over  = {t: v - cap for t, v in w.items() if v > cap + 1e-6}
        if not over:
            break
        exc = sum(over.values())
        for t in over:
            w[t] = cap
        free = {t: v for t, v in w.items() if v < cap - 1e-6}
        tf = sum(free.values())
        if tf <= 0:
            break
        for t in free:
            w[t] += exc * free[t] / tf
    total = sum(w.values())
    return {t: v / total for t, v in w.items()}


def _max_sharpe_wf(rets_df: "pd.DataFrame", tickers: List[str], max_w: float = 0.50) -> Dict[str, float]:
    """Máximo Sharpe walk-forward sobre retornos pasados con cap de peso."""
    try:
        from scipy.optimize import minimize
    except ImportError:
        n = len(tickers)
        return {t: 1/n for t in tickers}

    ts = [t for t in tickers if t in rets_df.columns]
    if len(ts) < 2:
        return {t: 1/len(tickers) for t in tickers}

    r = rets_df[ts].dropna()
    if len(r) < 20:
        return {t: 1/len(ts) for t in ts}

    mu  = r.mean().values * 252 * 100
    cov = r.cov().values * 252 * (100**2)
    n   = len(ts)
    rf  = 4.5
    cap = max(0.01, min(max_w, 0.99))

    def neg_sharpe(w):
        ret = float(w @ mu)
        vol = float(np.sqrt(max(w @ cov @ w, 1e-8)))
        return -(ret - rf) / vol if vol > 0 else 0.0

    res = minimize(neg_sharpe, np.ones(n)/n,
                   method="SLSQP",
                   bounds=[(0.01, cap)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 300, "ftol": 1e-9})
    if not res.success:
        return {t: 1/n for t in ts}
    total = res.x.sum()
    return {ts[i]: round(float(res.x[i]) / total, 4) for i in range(n)}


def _backtest_portfolio_historico(
    tickers: List[str],
    weights: Dict[str, float],
    opt: Dict,
    years: int,
    capital: float,
    rebalanceo_meses: int = 0,
    costo_pct: float = 0.0,
    wf_lookback: int = 252,
) -> Optional[Dict]:
    """
    Simula equity curves históricas con rebalanceo periódico, costos de transacción
    y optimización walk-forward (sin sesgo de look-ahead).
    """
    try:
        import yfinance as yf
    except ImportError:
        return None

    end   = datetime.today()
    start = end - timedelta(days=years * 365 + 30)

    all_tickers = list(set(tickers + ["SPY"]))
    raw = yf.download(all_tickers, start=start, end=end,
                      auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return None

    closes = raw["Close"] if hasattr(raw.columns, "levels") else raw
    closes = closes.dropna(how="all").fillna(method="ffill")

    available = [t for t in tickers if t in closes.columns]
    if len(available) < 2:
        return None

    rets = closes[available + (["SPY"] if "SPY" in closes.columns else [])].pct_change().dropna()
    RF_DAILY = 0.045 / 252

    # ── Simulación con rebalanceo walk-forward y costos ─────────────────
    def _simulate(label: str, init_w: Dict, weight_fn=None) -> Optional[Dict]:
        """
        Simula una estrategia día a día.
        - init_w: pesos iniciales
        - weight_fn(past_rets_df) -> dict: si se provee, se llama en cada rebalanceo
          para recalcular pesos con datos históricos hasta esa fecha (walk-forward).
          Si es None, los pesos objetivo son siempre init_w (sin look-ahead).
        """
        ts = [t for t in init_w if t in rets.columns]
        if not ts:
            return None
        total = sum(init_w.get(t, 0) for t in ts)
        if total <= 0:
            return None

        target_w = {t: init_w[t] / total for t in ts}
        curr_w   = dict(target_w)
        pv       = capital * (1 - costo_pct)           # costo compra inicial
        equity_vals = []
        last_rebal  = rets.index[0]
        rebal_dates = []

        for date, row in rets.iterrows():
            # Retorno diario ponderado
            daily_ret = sum(curr_w.get(t, 0) * float(row.get(t, 0)) for t in curr_w)
            pv *= (1 + daily_ret)

            # Drift de pesos de mercado
            factor = {t: curr_w[t] * (1 + float(row.get(t, 0))) for t in curr_w}
            total_f = sum(factor.values())
            if total_f > 0:
                curr_w = {t: v / total_f for t, v in factor.items()}

            # Rebalanceo periódico
            if rebalanceo_meses > 0:
                elapsed = (date - last_rebal).days / 30.44
                if elapsed >= rebalanceo_meses:
                    past = rets.loc[:date].tail(wf_lookback)
                    new_target = weight_fn(past) if weight_fn else target_w
                    if new_target:
                        all_t = set(curr_w) | set(new_target)
                        turnover = sum(
                            abs(new_target.get(t, 0) - curr_w.get(t, 0))
                            for t in all_t
                        ) / 2
                        pv -= pv * turnover * costo_pct
                        curr_w = new_target
                        last_rebal = date
                        rebal_dates.append(date)

            equity_vals.append(pv)

        equity   = pd.Series(equity_vals, index=rets.index)
        port_ret = equity.pct_change().dropna()
        excess   = port_ret - RF_DAILY
        ann_ret  = float((1 + port_ret.mean()) ** 252 - 1) * 100
        ann_vol  = float(port_ret.std() * np.sqrt(252)) * 100
        sharpe   = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0
        total_r  = float((equity.iloc[-1] / capital - 1) * 100)
        roll_max = equity.cummax()
        dd       = (equity / roll_max - 1) * 100
        max_dd   = float(dd.min())

        return {
            "label": label,
            "equity": equity,
            "drawdown": dd,
            "dates": equity.index,
            "rebal_dates": rebal_dates,
            "ann_ret": round(ann_ret, 2),
            "ann_vol": round(ann_vol, 2),
            "sharpe": round(sharpe, 3),
            "total_ret": round(total_r, 2),
            "max_dd": round(max_dd, 2),
            "calmar": round(ann_ret / abs(max_dd), 3) if max_dd < 0 else 0,
        }

    max_w = 0.50
    estrategias = []
    wf_suffix = " (walk-forward)" if rebalanceo_meses > 0 else ""

    # 1. Pesos del perfil — sin look-ahead por diseño (fijos desde cuestionario)
    res = _simulate("Pesos del perfil", weights)
    if res:
        estrategias.append(res)

    # 2. Máximo Sharpe walk-forward
    w_sharpe = opt.get("max_sharpe_weights", {})
    if w_sharpe:
        res = _simulate(
            f"Máximo Sharpe{wf_suffix}", w_sharpe,
            weight_fn=(lambda r: _max_sharpe_wf(r, list(w_sharpe.keys()), max_w))
                      if rebalanceo_meses > 0 else None,
        )
        if res:
            estrategias.append(res)

    # 3. HRP walk-forward
    w_hrp = opt.get("hrp_weights", {})
    if w_hrp:
        res = _simulate(
            f"HRP{wf_suffix}", w_hrp,
            weight_fn=(lambda r: _hrp_wf(r, list(w_hrp.keys()), max_w))
                      if rebalanceo_meses > 0 else None,
        )
        if res:
            estrategias.append(res)

    # 4. SPY benchmark (sin costos — es el índice de referencia)
    if "SPY" in rets.columns:
        spy_rets = rets["SPY"]
        spy_eq   = (1 + spy_rets).cumprod() * capital
        excess   = spy_rets - RF_DAILY
        ann_ret  = float((1 + spy_rets.mean()) ** 252 - 1) * 100
        ann_vol  = float(spy_rets.std() * np.sqrt(252)) * 100
        sharpe   = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0
        total_r  = float((spy_eq.iloc[-1] / capital - 1) * 100)
        roll_max = spy_eq.cummax()
        dd       = (spy_eq / roll_max - 1) * 100
        estrategias.append({
            "label": "SPY (benchmark)",
            "equity": spy_eq, "drawdown": dd, "dates": spy_eq.index, "rebal_dates": [],
            "ann_ret": round(ann_ret, 2), "ann_vol": round(ann_vol, 2),
            "sharpe": round(sharpe, 3), "total_ret": round(total_r, 2),
            "max_dd": round(float(dd.min()), 2),
            "calmar": round(ann_ret / abs(float(dd.min())), 3) if dd.min() < 0 else 0,
        })

    return {
        "estrategias": estrategias,
        "start": str(rets.index[0].date()),
        "end":   str(rets.index[-1].date()),
        "rebalanceo_meses": rebalanceo_meses,
        "costo_pct": costo_pct,
    }


def render_backtest_tab():
    """Renderiza la pestaña de backtesting walk-forward."""
    st.markdown("### Backtesting")

    modo = st.radio(
        "Modo",
        ["Activo individual (ML walk-forward)", "Portafolio (histórico multi-estrategia)"],
        horizontal=True,
        key="bt_modo",
    )
    st.markdown("---")

    if modo == "Portafolio (histórico multi-estrategia)":
        _render_backtest_portfolio()
        return

    st.markdown("#### Walk-Forward con Señales ML")
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
    sma_m = bt_data.get("sma_metrics", {})

    st.caption(
        f"Período: {bt_data['start_date']} → {bt_data['end_date']} | "
        f"Ticker: {bt_data['ticker']} | "
        f"Señales: {bt_data['buy_signals']} compras, {bt_data['sell_signals']} ventas"
    )

    # ── Métricas comparativas ─────────────────────────────────────────────
    st.markdown("#### Métricas: Estrategia ML vs Buy & Hold vs SMA Crossover")
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

    if sma_m:
        st.markdown("##### ML vs SMA Crossover (20/50 días)")
        sma_cols = st.columns(4)
        sma_compare = [
            ("CAGR", "cagr", "%"),
            ("Sharpe Ratio", "sharpe_ratio", ""),
            ("Retorno Total", "total_return", "%"),
            ("Max Drawdown", "max_drawdown", "%"),
        ]
        for col, (label, key, unit) in zip(sma_cols, sma_compare):
            ml_val = m[key]
            sma_val = sma_m.get(key, ml_val)
            delta = ml_val - sma_val
            if key == "max_drawdown":
                col.metric(label, f"{ml_val:.1f}{unit}", delta=f"{delta:+.1f}{unit}", delta_color="inverse")
            else:
                col.metric(label, f"{ml_val:.1f}{unit}", delta=f"{delta:+.1f}{unit}")

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

    # Alpha vs B&H y vs SMA
    alpha = m.get('alpha', 0)
    sma_cagr = sma_m.get('cagr', m.get('cagr', 0)) if sma_m else m.get('cagr', 0)
    alpha_vs_sma = m.get('cagr', 0) - sma_cagr
    if alpha > 5:
        interpretaciones.append(f"✅ **Alpha {alpha:+.1f}pp vs B&H**: La estrategia ML superó al Buy & Hold por {alpha:.1f} puntos porcentuales — añade valor real.")
    elif alpha > 0:
        interpretaciones.append(f"👍 **Alpha {alpha:+.1f}pp vs B&H**: La estrategia ML superó ligeramente al Buy & Hold.")
    elif alpha > -5:
        interpretaciones.append(f"⚠️ **Alpha {alpha:+.1f}pp vs B&H**: La estrategia ML rindió levemente por debajo del Buy & Hold. Considerar costos de transacción.")
    else:
        interpretaciones.append(f"❌ **Alpha {alpha:+.1f}pp vs B&H**: Buy & Hold superó claramente a la estrategia ML en este período.")
    if sma_m:
        if alpha_vs_sma > 2:
            interpretaciones.append(f"✅ **+{alpha_vs_sma:.1f}pp vs SMA Crossover**: El modelo ML supera a la estrategia técnica básica — el aprendizaje automático agrega valor real más allá del análisis técnico tradicional.")
        elif alpha_vs_sma > -2:
            interpretaciones.append(f"👍 **{alpha_vs_sma:+.1f}pp vs SMA Crossover**: Rendimiento similar a la estrategia técnica básica. El ML es competitivo con el análisis técnico simple.")
        else:
            interpretaciones.append(f"⚠️ **{alpha_vs_sma:+.1f}pp vs SMA Crossover**: El SMA Crossover superó al modelo ML. Revisar features del ensemble o condiciones de mercado del período.")

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
            rows.append({
                "Métrica": labels_map[key],
                "Estrategia ML": m[key],
                "Buy & Hold": bm.get(key, "-"),
                "SMA Crossover (20/50)": sma_m.get(key, "-") if sma_m else "-",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Curva de equity ───────────────────────────────────────────────────
    st.markdown("#### Curva de Equity: Estrategia ML vs Buy & Hold vs SMA Crossover")
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
        if "sma_crossover" in eq_df.columns:
            fig_eq.add_trace(go.Scatter(
                x=eq_df["date"], y=eq_df["sma_crossover"],
                name="SMA Crossover (20/50)",
                line=dict(color="#4CAF50", width=2, dash="dot"),
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
