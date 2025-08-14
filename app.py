import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import gensim.downloader as api

# -----------------------
# Cargar modelo fastText español
# -----------------------
st.title("Visualización 3D de embeddings en español")

st.write("Zoom, rota y explora el espacio semántico de palabras en español.")

@st.cache_resource
def load_model():
    return api.load("fasttext-wiki-news-subwords-300")  # Modelo grande multilingüe (incluye español)

model = load_model()

# Lista de palabras en español (puedes ampliar)
palabras = [
    "gato", "perro", "ratón", "león", "tigre", "manzana", "banana", "naranja", "coche", "avión", "tren",
    "computadora", "teléfono", "internet", "música", "guitarra", "piano", "batería", "fútbol", "baloncesto",
    "río", "montaña", "océano", "ciudad", "pueblo", "escuela", "universidad", "profesor", "estudiante",
    "amistad", "amor", "trabajo", "familia", "libro", "poesía", "historia", "ciencia", "arte", "película",
    "cielo", "mar", "tierra", "fuego", "aire", "bosque", "desierto", "lluvia", "nieve", "sol"
]

# Filtrar solo las que están en el vocabulario
palabras = [p for p in palabras if p in model]

# Obtener embeddings
embeddings = np.array([model[p] for p in palabras])

# -----------------------
# Reducir a 3D
# -----------------------
pca = PCA(n_components=3)
coords = pca.fit_transform(embeddings)

df = pd.DataFrame({
    "palabra": palabras,
    "x": coords[:, 0],
    "y": coords[:, 1],
    "z": coords[:, 2]
})

# -----------------------
# Visualización 3D
# -----------------------
fig = px.scatter_3d(
    df,
    x="x",
    y="y",
    z="z",
    hover_name="palabra",
    opacity=0.8
)

fig.update_traces(marker=dict(size=6))
fig.update_layout(
    title="Embeddings de palabras en 3D (PCA en español)",
    scene=dict(
        xaxis_title="PCA 1",
        yaxis_title="PCA 2",
        zaxis_title="PCA 3"
    )
)

st.plotly_chart(fig, use_container_width=True)