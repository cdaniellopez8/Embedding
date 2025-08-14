import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# ---------------------
# Simulación de embeddings
# ---------------------
np.random.seed(42)
n_palabras = 500
dim = 100
palabras = [f"palabra_{i}" for i in range(n_palabras)]
embeddings = np.random.rand(n_palabras, dim)

# ---------------------
# Reducción de dimensión a 2D (PCA)
# ---------------------
pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)

df = pd.DataFrame({
    "palabra": palabras,
    "x": coords[:, 0],
    "y": coords[:, 1]
})

# ---------------------
# Streamlit UI
# ---------------------
st.title("Visualización interactiva de embeddings")

st.write("Usa el zoom para acercarte y ver las palabras con más detalle.")

fig = px.scatter(
    df,
    x="x",
    y="y",
    hover_name="palabra",  # Aparecen al pasar el mouse
    opacity=0.7,
    width=900,
    height=700
)

fig.update_traces(marker=dict(size=6))
fig.update_layout(
    title="Embeddings (PCA)",
    hovermode="closest"
)

st.plotly_chart(fig, use_container_width=True)