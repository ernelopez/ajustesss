import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Ajuste por cuadrados mínimos")

st.write(
    "Ingresá los datos (una pareja t,y por línea). "
    "Los valores deben ingresarse separados por una coma; "
    "el punto decimal es el punto."
)

default_text = "1.1,2.3\n1.9,3.8\n2.5,7.1\n3.0,5.9"

texto = st.text_area("Datos", value=default_text, height=150)

tipo_ajuste = st.selectbox(
    "Tipo de ajuste",
    ["Lineal", "Exponencial"]
)


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


if st.button("Ejecutar"):

    t, y = parsear(texto)

    if t is None or len(t) < 2:
        st.warning("Formato inválido o faltan datos.")

    else:

        if tipo_ajuste == "Lineal":

            # Ajuste lineal
            a, b = np.polyfit(t, y, 1)

            y_pred = a * t + b

            x_line = np.linspace(0, max(t), 300)
            y_line = a * x_line + b

            ecuacion = rf"y={a:.4f}\,t"
            if b >= 0:
                ecuacion += rf"+{b:.4f}"
            else:
                ecuacion += rf"{b:.4f}"

        else:

            # Ajuste exponencial
            if np.any(y <= 0):
                st.error("Para el ajuste exponencial todos los valores de y deben ser positivos.")
                st.stop()

            m, c = np.polyfit(t, np.log(y), 1)

            A = np.exp(c)
            B = np.exp(m)

            y_pred = A * B**t

            x_line = np.linspace(0, max(t), 300)
            y_line = A * B**x_line

            ecuacion = rf"y={A:.4f}\,{B:.4f}^{{t}}"

        # Suma de cuadrados de residuos
        ss = np.sum((y - y_pred) ** 2)

        # Gráfico
        fig, ax = plt.subplots()

        ax.scatter(t, y, label="Datos")
        ax.plot(x_line, y_line, label="Ajuste")

        # Residuos
        for ti, yi, ypi in zip(t, y, y_pred):
            ax.plot([ti, ti], [yi, ypi], color="red")

        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        ax.set_xlabel("t")
        ax.set_ylabel("y")

        ax.legend()

        st.pyplot(fig)

        st.latex(ecuacion)
        st.write(f"Suma de cuadrados de residuos: {ss:.4f}")
