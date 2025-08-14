import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
import spacy

# -----------------------
# Cargar modelo spaCy en español
# -----------------------
@st.cache_resource
def load_model():
    return spacy.load("es_core_news_md")

nlp = load_model()

# -----------------------
# Lista amplia de palabras en español
# -----------------------
palabras = [
    "gato", "perro", "ratón", "león", "tigre", "oso", "caballo", "vaca", "oveja", "cabra",
    "manzana", "banana", "naranja", "sandía", "melón", "pera", "uva", "fresa", "cereza", "limón",
    "coche", "avión", "tren", "barco", "bicicleta", "moto", "camión", "metro", "cohete", "submarino",
    "computadora", "teléfono", "internet", "televisión", "radio", "cámara", "impresora", "robot", "dron", "satélite",
    "música", "guitarra", "piano", "batería", "violín", "flauta", "trompeta", "canto", "danza", "teatro",
    "fútbol", "baloncesto", "tenis", "natación", "atletismo", "voleibol", "rugby", "surf", "boxeo", "ciclismo",
    "río", "montaña", "océano", "ciudad", "pueblo", "bosque", "desierto", "lago", "cascada", "isla",
    "escuela", "universidad", "profesor", "estudiante", "biblioteca", "libro", "cuaderno", "pizarra", "lápiz", "pluma",
    "amistad", "amor", "trabajo", "familia", "hogar", "salud", "dinero", "felicidad", "tristeza", "miedo",
    "historia", "ciencia", "arte", "filosofía", "literatura", "poesía", "matemáticas", "física", "química", "biología",
    "cielo", "mar", "tierra", "fuego", "aire", "lluvia", "nieve", "viento", "tormenta", "huracán",
    "pan", "queso", "carne", "pescado", "huevo", "arroz", "pasta", "ensalada", "sopa", "pizza",
    "lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo", "enero", "febrero", "marzo",
    "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
]

# -----------------------
# Sidebar con buscador
# -----------------------
st.sidebar.header("Buscador de palabra")
palabra_buscar = st.sidebar.text_input("Escribe una palabra en español:").strip().lower()

# Agregar palabra buscada si está en el vocabulario de spaCy
if palabra_buscar and palabra_buscar not in palabras:
    palabras.append(palabra_buscar)

# -----------------------
# Obtener embeddings
# -----------------------
embeddings = np.array([nlp(p).vector for p in palabras])

# -----------------------
# Reducir a 3D con PCA
# -----------------------
pca = PCA(n_components=3)
coords = pca.fit_transform(embeddings)

# -----------------------
# DataFrame para graficar
# -----------------------
df = pd.DataFrame({
    "palabra": palabras,
    "x": coords[:, 0],
    "y": coords[:, 1],
    "z": coords[:, 2],
    "color": ["red" if p == palabra_buscar else "blue" for p in palabras]
})

# -----------------------
# Visualización 3D
# -----------------------
fig = px.scatter_3d(
    df,
    x="x",
    y="y",
    z="z",
    color="color",
    hover_name="palabra",
    opacity=0.8,
    width=900,
    height=900
)

fig.update_traces(marker=dict(size=6))
fig.update_layout(
    title="Visualización 3D de embeddings en español",
    scene=dict(
        xaxis_title=None,
        yaxis_title=None,
        zaxis_title=None
    ),
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)