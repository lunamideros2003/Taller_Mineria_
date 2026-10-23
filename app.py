from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Segmentacion de clientes con K-Means",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Mall_Customers.csv"
FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
RANDOM_STATE = 42
K_FINAL = 5

PERFIL_INFO = {
    0: {
        "nombre": "Clientes promedio",
        "color": "#4C78A8",
        "descripcion": "Ingreso y gasto medios; es el grupo mas numeroso y representa el comportamiento tipico.",
        "estrategia": "Cross-selling y up-selling, promociones estacionales y programa de fidelizacion para aumentar la frecuencia de compra.",
    },
    1: {
        "nombre": "Clientes de alto valor",
        "color": "#54A24B",
        "descripcion": "Alto ingreso y alto gasto; son los clientes mas rentables del negocio.",
        "estrategia": "Retencion prioritaria, beneficios exclusivos, atencion personalizada y acceso anticipado a productos premium.",
    },
    2: {
        "nombre": "Clientes con potencial",
        "color": "#F58518",
        "descripcion": "Ingreso bajo pero gasto alto; en su mayoria son clientes jovenes.",
        "estrategia": "Ofertas de entrada, pagos a cuotas, gamificacion y fidelizacion temprana; monitorear su evolucion de ingreso.",
    },
    3: {
        "nombre": "Bajo valor con alto ingreso",
        "color": "#E45756",
        "descripcion": "Alto ingreso pero bajo gasto; existe capacidad de compra que no se esta aprovechando.",
        "estrategia": "Reactivacion, pruebas de productos premium, ofertas personalizadas e investigacion de barreras de compra.",
    },
    4: {
        "nombre": "Bajo valor con bajo ingreso",
        "color": "#B279A2",
        "descripcion": "Ingreso y gasto bajos; aportan menor rentabilidad por cliente.",
        "estrategia": "Ofertas de bajo costo, canales digitales y campanas masivas de bajo presupuesto.",
    },
}


@st.cache_data(show_spinner=False)
def cargar_datos(ruta: str) -> pd.DataFrame:
    return pd.read_csv(ruta)


@st.cache_resource(show_spinner="Entrenando el modelo K-Means...")
def entrenar_modelo(ruta: str, k: int):
    datos = pd.read_csv(ruta)
    X = datos[FEATURES]
    scaler = StandardScaler().fit(X)
    X_esc = scaler.transform(X)
    modelo = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_esc)
    return scaler, modelo, X_esc


@st.cache_data(show_spinner=False)
def calcular_metricas(ruta: str, max_k: int = 10) -> pd.DataFrame:
    datos = pd.read_csv(ruta)
    X = datos[FEATURES].to_numpy()
    X_esc = StandardScaler().fit_transform(X)
    filas = []
    for k in range(1, max_k + 1):
        modelo = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_esc)
        silueta = silhouette_score(X_esc, modelo.labels_) if k >= 2 else np.nan
        filas.append({"K": k, "Inercia": modelo.inertia_, "Silueta": silueta})
    return pd.DataFrame(filas)


def construir_resumen(datos: pd.DataFrame) -> pd.DataFrame:
    resumen = (
        datos.groupby(["Cluster", "Perfil"])
        .agg(
            Clientes=("CustomerID", "count"),
            Edad=("Age", "mean"),
            Ingreso=("Annual Income (k$)", "mean"),
            Gasto=("Spending Score (1-100)", "mean"),
        )
        .reset_index()
    )
    resumen["% clientes"] = 100 * resumen["Clientes"] / len(datos)
    resumen = resumen[["Cluster", "Perfil", "Clientes", "% clientes", "Edad", "Ingreso", "Gasto"]]
    return resumen.round(1)


def mostrar_figura(figura):
    st.pyplot(figura)
    plt.close(figura)


def figura_codo(metricas: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(metricas["K"], metricas["Inercia"], marker="o", linewidth=2.2, color="#4C78A8")
    ax.axvline(K_FINAL, color="#E45756", linestyle="--", linewidth=1.5, label=f"K = {K_FINAL}")
    ax.set_title("Metodo del codo")
    ax.set_xlabel("Numero de clusters (K)")
    ax.set_ylabel("Inercia (WCSS)")
    ax.set_xticks(metricas["K"])
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def figura_silueta(metricas: pd.DataFrame):
    datos = metricas.dropna(subset=["Silueta"]).copy()
    mejor = datos.loc[datos["Silueta"].idxmax()]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(datos["K"], datos["Silueta"], marker="o", linewidth=2.2, color="#54A24B")
    ax.scatter([mejor["K"]], [mejor["Silueta"]], s=150, color="#E45756", zorder=5)
    ax.axvline(K_FINAL, color="#E45756", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.annotate(
        f"K = {int(mejor['K'])}\nSilueta = {mejor['Silueta']:.3f}",
        xy=(mejor["K"], mejor["Silueta"]),
        xytext=(mejor["K"] + 1.2, max(0.25, mejor["Silueta"] - 0.08)),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=10,
    )
    ax.set_title("Coeficiente de silueta")
    ax.set_xlabel("Numero de clusters (K)")
    ax.set_ylabel("Silueta promedio")
    ax.set_xticks(datos["K"])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def figura_clusters(datos: pd.DataFrame, centroides: pd.DataFrame, punto_nuevo=None):
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for cluster in sorted(datos["Cluster"].unique()):
        cluster = int(cluster)
        grupo = datos[datos["Cluster"] == cluster]
        info = PERFIL_INFO.get(
            cluster,
            {"nombre": f"Cluster {cluster}", "color": "#777777"},
        )
        ax.scatter(
            grupo["Annual Income (k$)"],
            grupo["Spending Score (1-100)"],
            color=info["color"],
            s=55,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
            label=f"C{cluster}: {info['nombre']} (n={len(grupo)})",
        )

    ax.scatter(
        centroides["Annual Income (k$)"],
        centroides["Spending Score (1-100)"],
        marker="X",
        s=240,
        color="black",
        edgecolors="white",
        linewidths=1.2,
        label="Centroides",
        zorder=5,
    )

    for cluster, fila in centroides.iterrows():
        ax.annotate(
            f"C{cluster}",
            xy=(fila["Annual Income (k$)"], fila["Spending Score (1-100)"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    if punto_nuevo is not None:
        ax.scatter(
            [punto_nuevo[0]],
            [punto_nuevo[1]],
            marker="*",
            s=450,
            color="#FFD700",
            edgecolors="black",
            linewidths=1.2,
            label="Nuevo cliente",
            zorder=6,
        )

    ax.set_title("Segmentacion de clientes con K-Means")
    ax.set_xlabel("Ingreso anual (k$)")
    ax.set_ylabel("Puntaje de gasto (1-100)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    st.title("Segmentacion de clientes con K-Means")
    st.markdown(
        "**Taller final de Mineria de Datos.** Esta aplicacion entrena un modelo K-Means "
        "sobre el dataset Mall Customers y permite explorar los segmentos y clasificar un nuevo cliente."
    )

    if not DATA_PATH.exists():
        st.error(
            "No se encontro el archivo Mall_Customers.csv. "
            "Ubique el archivo en la misma carpeta que app.py."
        )
        st.stop()

    df = cargar_datos(str(DATA_PATH)).copy()
    scaler, modelo, X_esc = entrenar_modelo(str(DATA_PATH), K_FINAL)
    df["Cluster"] = modelo.labels_
    df["Perfil"] = df["Cluster"].map(lambda c: PERFIL_INFO[int(c)]["nombre"])
    centroides = pd.DataFrame(
        scaler.inverse_transform(modelo.cluster_centers_),
        columns=FEATURES,
    )
    metricas = calcular_metricas(str(DATA_PATH))
    silueta_final = silhouette_score(X_esc, modelo.labels_)
    resumen = construir_resumen(df)

    with st.sidebar:
        st.header("Resumen del modelo")
        st.metric("Clientes", len(df))
        st.metric("Clusters (K)", K_FINAL)
        st.metric("Silueta promedio", f"{silueta_final:.3f}")
        st.metric("Inercia", f"{modelo.inertia_:,.2f}")
        st.divider()
        st.subheader("Segmentos")
        for cluster in range(K_FINAL):
            cantidad = int((df["Cluster"] == cluster).sum())
            st.markdown(f"**C{cluster} - {PERFIL_INFO[cluster]['nombre']}:** {cantidad} clientes")
        st.divider()
        st.caption(
            "K = 5 fue seleccionado con el metodo del codo y el coeficiente de silueta. "
            "El modelo usa ingreso anual y puntaje de gasto."
        )

    tab_codo, tab_clusters, tab_nuevo, tab_datos = st.tabs(
        ["Codo y silueta", "Clusters y perfiles", "Nuevo cliente", "Datos"]
    )

    with tab_codo:
        st.subheader("Seleccion del numero de clusters")
        columna_izquierda, columna_derecha = st.columns(2)
        with columna_izquierda:
            st.markdown("**Metodo del codo**")
            mostrar_figura(figura_codo(metricas))
        with columna_derecha:
            st.markdown("**Coeficiente de silueta**")
            mostrar_figura(figura_silueta(metricas))

        st.markdown("### Metricas por valor de K")
        st.dataframe(metricas.round(4), width="stretch", hide_index=True)
        st.info(
            "K = 5 es el punto donde la inercia deja de caer abruptamente y ademas "
            "obtiene el mayor coeficiente de silueta (0.5547)."
        )
        with st.expander("Como interpretar estas metricas"):
            st.markdown(
                "- **Inercia (WCSS):** suma de distancias al cuadrado de cada cliente a su centroide. "
                "Valores menores indican clusters mas compactos, pero siempre disminuye al aumentar K.\n"
                "- **Codo:** punto donde agregar un cluster adicional produce una mejora marginal pequena.\n"
                "- **Silueta:** combina cohesion y separacion. Cerca de 1 indica clusters bien definidos; "
                "cerca de 0 indica solapamiento; valores negativos sugieren asignaciones incorrectas."
            )

    with tab_clusters:
        st.subheader("Clusters generados")
        mostrar_figura(figura_clusters(df, centroides))

        st.markdown("### Resumen por segmento")
        st.dataframe(resumen, width="stretch", hide_index=True)

        st.markdown("### Interpretacion y estrategia")
        cluster_sel = st.selectbox(
            "Seleccione un cluster",
            options=sorted(int(c) for c in df["Cluster"].unique()),
            format_func=lambda c: f"C{c} - {PERFIL_INFO[c]['nombre']}",
        )
        info = PERFIL_INFO[cluster_sel]
        grupo = df[df["Cluster"] == cluster_sel]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Clientes", len(grupo))
        col2.metric("Ingreso promedio", f"{grupo['Annual Income (k$)'].mean():.1f} k$")
        col3.metric("Gasto promedio", f"{grupo['Spending Score (1-100)'].mean():.1f}")
        col4.metric("Edad promedio", f"{grupo['Age'].mean():.1f}")

        st.markdown(f"**Descripcion:** {info['descripcion']}")
        st.markdown(f"**Estrategia recomendada:** {info['estrategia']}")

    with tab_nuevo:
        st.subheader("Clasificar un nuevo cliente")
        st.caption(
            "El cluster se asigna con el ingreso anual y el puntaje de gasto. "
            "El genero y la edad se muestran como contexto del perfil."
        )

        with st.form("formulario_nuevo_cliente"):
            col1, col2, col3, col4 = st.columns(4)
            genero = col1.selectbox("Genero", ["Female", "Male"])
            edad = col2.slider("Edad", 18, 100, 35)
            ingreso = col3.slider("Ingreso anual (k$)", 0, 200, 60)
            gasto = col4.slider("Puntaje de gasto (1-100)", 1, 100, 50)
            enviar = st.form_submit_button("Clasificar cliente", type="primary")

        if enviar:
            nuevo = pd.DataFrame([[ingreso, gasto]], columns=FEATURES)
            nuevo_esc = scaler.transform(nuevo)
            cluster = int(modelo.predict(nuevo_esc)[0])
            info = PERFIL_INFO.get(
                cluster,
                {"nombre": f"Cluster {cluster}", "descripcion": "", "estrategia": ""},
            )
            distancias = np.linalg.norm(modelo.cluster_centers_ - nuevo_esc, axis=1)

            st.success(f"El cliente pertenece al cluster {cluster}: {info['nombre']}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Ingreso", f"{ingreso} k$")
            col2.metric("Gasto", f"{gasto} / 100")
            col3.metric("Edad", edad)
            col4.metric("Genero", genero)

            st.markdown(f"**Descripcion del segmento:** {info['descripcion']}")
            st.markdown(f"**Estrategia recomendada:** {info['estrategia']}")

            st.markdown("### Ubicacion del nuevo cliente")
            mostrar_figura(figura_clusters(df, centroides, punto_nuevo=(ingreso, gasto)))

            st.markdown("### Distancia a cada centroide")
            tabla_distancias = pd.DataFrame(
                {
                    "Cluster": list(range(K_FINAL)),
                    "Perfil": [PERFIL_INFO[c]["nombre"] for c in range(K_FINAL)],
                    "Distancia (espacio escalado)": distancias,
                }
            ).sort_values("Distancia (espacio escalado)")
            st.dataframe(tabla_distancias.round(3), width="stretch", hide_index=True)
            st.caption("El cliente se asigna al centroide con la menor distancia euclidiana.")

    with tab_datos:
        st.subheader("Base de clientes segmentada")
        opciones = sorted(int(c) for c in df["Cluster"].unique())
        seleccion = st.multiselect(
            "Filtrar por cluster",
            options=opciones,
            default=opciones,
            format_func=lambda c: f"C{c} - {PERFIL_INFO[c]['nombre']}",
        )
        filtrado = df[df["Cluster"].isin(seleccion)]
        st.dataframe(filtrado, width="stretch", hide_index=True)
        st.download_button(
            "Descargar CSV segmentado",
            data=filtrado.to_csv(index=False).encode("utf-8"),
            file_name="clientes_segmentados.csv",
            mime="text/csv",
        )

    st.divider()
    st.caption("Taller final de Mineria de Datos - Segmentacion de clientes con K-Means")


if __name__ == "__main__":
    main()

