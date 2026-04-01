"""
Registro diario de predicciones del sistema para comparación con realidad.
Ejecutar una vez por día mientras el backend esté corriendo.
Guarda predicciones en: resultados_semana.xlsx
"""

import requests
import pandas as pd
from datetime import datetime, date
from pathlib import Path
import sys

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────
API_URL     = "http://localhost:8000"
TICKERS     = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META"]
EXCEL_FILE  = "resultados_semana.xlsx"
CREDENTIALS = {"username": "carlos2026", "password": "Carlos2026"}
# ──────────────────────────────────────────────────────────────────────────────


def obtener_token() -> str:
    """Obtiene token JWT del backend."""
    resp = requests.post(
        f"{API_URL}/auth/login",
        data=CREDENTIALS,
        timeout=15
    )
    if resp.status_code != 200:
        print(f"  Error de autenticación: {resp.status_code} - {resp.text}")
        sys.exit(1)
    return resp.json()["access_token"]


def analizar_ticker(ticker: str, token: str) -> dict:
    """Llama al endpoint de predicción y retorna los datos relevantes."""
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(
        f"{API_URL}/predict/{ticker}",
        headers=headers,
        timeout=60
    )
    if resp.status_code != 200:
        print(f"  [{ticker}] Error {resp.status_code}")
        return None

    data = resp.json()
    hoy  = date.today().isoformat()
    ahora = datetime.now().strftime("%H:%M:%S")

    mercado       = data.get("mercado", {})
    prediccion    = data.get("prediccion", {})
    sentimiento   = data.get("sentimiento", {})
    recomendacion = data.get("recomendacion", {})
    metricas      = prediccion.get("metricas", {})
    parametros    = prediccion.get("parametros", {})

    variacion_pred = prediccion.get("variacion_pct") or 0
    direccion      = "SUBIDA" if variacion_pred > 0 else "BAJADA"

    return {
        # Fecha y ticker
        "fecha":                  hoy,
        "hora":                   ahora,
        "ticker":                 ticker,

        # Precio actual
        "precio_actual":          mercado.get("ultimo_precio"),
        "variacion_dia_pct":      mercado.get("variacion_diaria"),
        "sma_20":                 mercado.get("media_movil_20"),
        "señal_tecnica":          mercado.get("senal"),

        # Predicción
        "precio_predicho":        prediccion.get("precio_predicho"),
        "variacion_predicha_pct": variacion_pred,
        "horizonte_dias":         prediccion.get("horizonte_dias", 3),
        "direccion":              direccion,
        "mejor_modelo":           parametros.get("mejor_modelo"),

        # Métricas del modelo
        "accuracy":               metricas.get("accuracy"),
        "precision":              metricas.get("precision"),
        "recall":                 metricas.get("recall"),
        "f1":                     metricas.get("f1"),
        "auc":                    metricas.get("auc"),

        # Sentimiento
        "sentimiento_score":      sentimiento.get("score"),
        "sentimiento_cat":        sentimiento.get("sentimiento"),

        # Recomendación
        "recomendacion":          recomendacion.get("tipo"),
        "nivel_riesgo":           recomendacion.get("nivel_riesgo_simple") or recomendacion.get("nivel"),

        # Columnas para completar después (precio real a t+3)
        "precio_real_t3":         None,
        "direccion_real":         None,
        "prediccion_correcta":    None,
    }


def guardar_excel(registros: list):
    """Agrega los registros nuevos al Excel existente (o lo crea)."""
    df_nuevo = pd.DataFrame(registros)
    archivo  = Path(EXCEL_FILE)

    if archivo.exists():
        df_existente = pd.read_excel(archivo)
        # Evitar duplicados: misma fecha + ticker
        df_existente = df_existente[
            ~df_existente[["fecha", "ticker"]].apply(tuple, axis=1).isin(
                df_nuevo[["fecha", "ticker"]].apply(tuple, axis=1)
            )
        ]
        df_final = pd.concat([df_existente, df_nuevo], ignore_index=True)
    else:
        df_final = df_nuevo

    df_final.sort_values(["fecha", "ticker"], inplace=True)
    df_final.to_excel(archivo, index=False)
    return len(df_nuevo), len(df_final)


def main():
    hoy = date.today().isoformat()
    print(f"\n{'='*55}")
    print(f"  REGISTRO DIARIO — {hoy}")
    print(f"{'='*55}")

    # Autenticación
    print("\n  Autenticando...")
    token = obtener_token()
    print("  Token obtenido.")

    # Analizar cada ticker
    registros = []
    for ticker in TICKERS:
        print(f"\n  Analizando {ticker}...", end=" ")
        resultado = analizar_ticker(ticker, token)
        if resultado:
            registros.append(resultado)
            dir_pred  = resultado.get("direccion") or "?"
            precio    = resultado.get("precio_actual") or 0
            pred      = resultado.get("precio_predicho") or 0
            variacion = resultado.get("variacion_predicha_pct") or 0
            print(f"OK  |  ${precio:.2f} -> ${pred:.2f} ({variacion:+.2f}%)  [{dir_pred}]")
        else:
            print("FALLO")

    if not registros:
        print("\n  No se obtuvieron datos. Verificá que el backend esté corriendo.")
        sys.exit(1)

    # Guardar en Excel
    nuevos, total = guardar_excel(registros)
    print(f"\n{'='*55}")
    print(f"  Guardado en: {EXCEL_FILE}")
    print(f"  Filas agregadas hoy: {nuevos}")
    print(f"  Total de registros: {total}")
    print(f"{'='*55}\n")
    print("  RECORDATORIO: en 3 días completar columnas:")
    print("  'precio_real_t3', 'direccion_real', 'prediccion_correcta'")
    print()


if __name__ == "__main__":
    main()
