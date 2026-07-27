import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -------------------------------------------------------
# Funciones de ajuste
# -------------------------------------------------------

def exponencial(t, A, B):
    return A * B**t

def logistica(t, K, A, r):
    return K / (1 + A * np.exp(-r * t))

# -------------------------------------------------------
# Lectura de datos
# -------------------------------------------------------

def parsear(texto):
    ts, ys = [], []

    for linea in texto.split("\n"):

        if linea.strip() == "":
            continue

        try:
            t_str, y_str = linea.split(",")
            ts.append(float(t_str))
            ys.append(float(y_str))

        except:
            return None, None

    return np.array(ts), np.array(ys)

# -------------------------------------------------------
# Interfaz
# -------------------------------------------------------

st.title("Ajuste por cuadrados mínimos")

st.write(
    "Ingresá una pareja (t,y) por línea. "
    "Los valores deben separarse con una coma."
)

default_text = """1,12
2,20
3,32
4,45
5,61
6,79
7,90
8,97"""

texto = st.text_area("Datos", value=default_text, height=180)

t, y = parsear(texto)

if t is not None and len(t) >= 2:

    n_total = len(t)

    n_ajuste = st.slider(
        "Cantidad de datos utilizados para el ajuste",
        min_value=2,
        max_value=n_total,
        value=n_total
    )

else:
    n_ajuste = 2

tipo_ajuste = st.selectbox(
    "Tipo de ajuste",
    ["Exponencial", "Logístico"]
)

# -------------------------------------------------------
# Ajuste
# -------------------------------------------------------

if st.button("Ejecutar"):

    t, y = parsear(texto)

    if t is None or len(t) < 2:
        st.warning("Formato inválido.")

    else:

        t_fit = t[:n_ajuste]
        y_fit = y[:n_ajuste]

        try:

            if tipo_ajuste == "Exponencial":

                if np.any(y_fit <= 0):
                    st.error("Todos los valores de y deben ser positivos.")
                    st.stop()

                m, c = np.polyfit(t_fit, np.log(y_fit), 1)

                A = np.exp(c)
                B = np.exp(m)

                y_pred = exponencial(t_fit, A, B)

                x_line = np.linspace(
                    min(t),
                    max(t),
                    400
                )

                y_line = exponencial(x_line, A, B)

                ecuacion = rf"y={A:.4f}\,{B:.4f}^{{t}}"

            else:

                K0 = 1.2 * np.max(y_fit)
                A0 = max(K0 / y_fit[0] - 1, 0.1)
                r0 = 0.2

                parametros, _ = curve_fit(
                    logistica,
                    t_fit,
                    y_fit,
                    p0=[K0, A0, r0],
                    maxfev=20000
                )

                K, A, r = parametros

                y_pred = logistica(t_fit, K, A, r)

                x_line = np.linspace(
                    min(t),
                    max(t),
                    400
                )

                y_line = logistica(x_line, K, A, r)

                ecuacion = (
                    rf"y=\frac{{{K:.4f}}}"
                    rf"{{1+{A:.4f}e^{{-{r:.4f}t}}}}"
                )

            # ------------------------------------------
            # Error cuadrático
            # ------------------------------------------

            ss = np.sum((y_fit - y_pred) ** 2)

            # ------------------------------------------
            # Gráfico
            # ------------------------------------------

            fig, ax = plt.subplots(figsize=(8,5))

            # Puntos usados
            ax.scatter(
                t_fit,
                y_fit,
                color="tab:blue",
                s=50,
                label="Datos ajustados"
            )

            # Puntos no usados
            if n_ajuste < len(t):

                ax.scatter(
                    t[n_ajuste:],
                    y[n_ajuste:],
                    color="gray",
                    s=50,
                    label="Datos no ajustados"
                )

            # Curva

            ax.plot(
                x_line,
                y_line,
                color="tab:orange",
                linewidth=2,
                label="Ajuste"
            )

            # Residuos

            for ti, yi, ypi in zip(t_fit, y_fit, y_pred):

                ax.plot(
                    [ti, ti],
                    [yi, ypi],
                    color="red"
                )

            ax.set_xlabel("t")
            ax.set_ylabel("y")

            ax.grid(True)

            ax.legend()

            st.pyplot(fig)

            st.latex(ecuacion)

            st.write(
                f"Suma de cuadrados de residuos = {ss:.4f}"
            )

        except Exception as e:

            st.error("No fue posible realizar el ajuste.")

            st.exception(e)
