import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import spacy
from sklearn.decomposition import PCA
from spacy.cli import download

try:
    nlp = spacy.load("es_core_news_md")
except OSError:
    download("es_core_news_md")
    nlp = spacy.load("es_core_news_md")

# Cargar modelo de spaCy en español
@st.cache_resource
def load_model():
    return spacy.load("es_core_news_md")

nlp = load_model()

# Lista de palabras (tu lista completa)
palabras = [
    # Animales
    "gato", "perro", "ratón", "lobo", "zorro", "tigre", "oso", "caballo", "vaca", "oveja", "cabra", "conejo",
    "ardilla", "murciélago", "ciervo", "jirafa", "elefante", "león", "pantera", "puma", "gorila", "mono", "chimpancé",
    "orangután", "ballena", "delfín", "tiburón", "foca", "pingüino", "águila", "halcón", "búho", "paloma", "canario",
    "loro", "pavo", "pollo", "gallina", "gallo", "pato", "cisne", "ganso", "serpiente", "cocodrilo", "lagarto",
    "iguana", "sapo", "tortuga", "cangrejo", "langosta", "camello", "hipopótamo", "rinoceronte", "zebra", "búfalo",
    "bisonte", "cabrito", "potro", "mula", "lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo",
    "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
    # Colores
    "azul", "rojo", "blanco", "negro", "verde", "naranja", "celeste", "cian", "magenta", "marron", "rosa",
    # Sentimientos
    "alegría", "tristeza", "ira", "miedo", "sorpresa", "asco", "amor", "odio", "esperanza", "desesperación",
    "nostalgia", "orgullo", "vergüenza", "culpa", "gratitud", "compasión", "envidia", "celos", "ansiedad", "calma",
    "satisfacción", "melancolía", "euforia", "pena", "soledad", "confianza", "desconfianza", "admiración", "rechazo", "ternura",
    # Filosofía
    "existencia", "esencia", "libertad", "determinismo", "moralidad", "ética", "virtud", "justicia", "verdad", "mentira",
    "belleza", "fealdad", "tiempo", "eternidad", "infinito", "causalidad", "realidad", "ilusión", "ser", "nada",
    "pensamiento", "razón", "emoción", "conocimiento", "ignorancia", "sabiduría", "duda", "certeza", "voluntad", "destino",
    # Frutas y verduras
    "manzana", "banana", "sandía", "melón", "pera", "uva", "fresa", "cereza", "limón", "kiwi", "papaya", "mango", "guayaba",
    "maracuyá", "piña", "durazno", "ciruela", "granada", "higo", "tomate", "lechuga", "zanahoria", "cebolla", "ajo", "pepino",
    "calabaza", "berenjena", "brócoli", "coliflor", "espinaca", "pimiento", "chile", "maíz", "guisante", "haba", "apio",
    "rábano", "alcachofa", "batata",
    # Transportes
    "carro", "avión", "tren", "barco", "bicicleta", "moto", "camión", "metro", "cohete", "submarino", "tractor", "helicóptero",
    "patinete", "globo", "yate", "velero", "canoa", "kayak", "autobús", "tranvía",
    # Tecnología
    "computadora", "teléfono", "internet", "televisión", "radio", "cámara", "impresora", "robot", "dron", "satélite",
    "teclado", "pantalla", "auriculares", "micrófono", "altavoz", "software", "hardware", "servidor", "red",
    # Lugares
    "ciudad", "pueblo", "aldea", "capital", "barrio", "calle", "avenida", "plaza", "puente", "parque", "jardín", "museo",
    "biblioteca", "universidad", "escuela", "estadio", "cine", "teatro", "restaurante", "hotel", "playa", "montaña", "río",
    "lago", "mar", "océano", "desierto", "bosque", "selva", "caverna", "volcán", "catarata", "isla", "península", "glaciar",
    "acantilado", "valle", "llanura", "pradera", "costa",
    # Conceptos abstractos
    "amistad", "felicidad", "valentía", "fe", "paz", "guerra", "honor", "perdón", "solidaridad", "paciencia",
    "depresión", "entusiasmo", "energía", "fuerza", "debilidad", "riqueza", "pobreza", "éxito", "fracaso", "motivación",
    "inspiración", "creatividad"
]

# Obtener vectores válidos
tokens = [nlp(word) for word in palabras]
all_vectors = [token.vector for token in tokens]

valid_data = [(word, vec) for word, vec in zip(palabras, all_vectors) if vec is not None and np.any(vec)]
valid_words, vectors = zip(*valid_data)
vectors = np.array(vectors)

# Reducir a 3D
pca = PCA(n_components=3)
coords = pca.fit_transform(vectors)

# DataFrame para Plotly
df = pd.DataFrame(coords, columns=["x", "y", "z"])
df["word"] = valid_words
df["color"] = "gray"

# Sidebar
with st.sidebar:
    st.markdown(
        "<h3 style='text-align: center; color: #4B4B4B;'>🔍 Buscar palabra</h3>",
        unsafe_allow_html=True
    )
    query = st.text_input("", placeholder="palabra a buscar...")

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; font-size: 0.85em; color: gray;">
        Desarrollado por <strong>Carlos D. López P.</strong>
    </div>
    """, unsafe_allow_html=True)

# --- Título y descripción ---
st.title(" Visualización de embeddings en 3D (Español)")

st.markdown("""
Esta aplicación muestra una representación tridimensional de **vectores semánticos** de palabras en español.
Cada palabra está ubicada en el espacio según su similitud de significado calculada con un **modelo de lenguaje**.

 **Cómo funciona:**
- El modelo `es_core_news_md` de spaCy asigna a cada palabra un vector de 300 dimensiones.
- Estos vectores se reducen a 3 dimensiones usando **PCA** para que puedan visualizarse.
- La similitud entre palabras se calcula usando **similitud de coseno** (1 = idénticos, 0 = no relacionados, -1 = opuestos).
- Si escribes una palabra en el buscador de la izquierda, se resaltará en rojo junto con sus **10 palabras más cercanas**.
- Debajo del gráfico verás una **tabla** con esas palabras cercanas y su similitud de coseno.
""")

# Inicializar listas
closest_words = []
farthest_words = []

# Si hay búsqueda
if query.strip() and query in valid_words:
    query_vec = nlp(query).vector
    sims = np.dot(vectors, query_vec) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vec))
    
    # Más cercanas
    closest_idx = sims.argsort()[::-1][1:11]
    closest_words = [(valid_words[i], sims[i]) for i in closest_idx]
    df.loc[df["word"].isin([w for w, _ in closest_words]), "color"] = "Más cercanas"
    
    # Menos cercanas
    farthest_idx = sims.argsort()[:10]
    farthest_words = [(valid_words[i], sims[i]) for i in farthest_idx]
    df.loc[df["word"].isin([w for w, _ in farthest_words]), "color"] = "Más lejanas"

# Visualización 3D
fig = px.scatter_3d(df, x="x", y="y", z="z", text="word", color="color",
                    color_discrete_map={"gray": "gray", "Más cercanas": "red", "Más lejanas": "blue"})
fig.update_traces(marker=dict(size=4), textposition="top center")
fig.update_layout(scene=dict(xaxis_title='', yaxis_title='', zaxis_title=''))


# Mostrar gráfico
st.plotly_chart(fig, use_container_width=True)

# Mostrar tablas **debajo del gráfico**
if closest_words:
    st.subheader(f"Palabras más cercanas a '{query}'")
    st.dataframe(pd.DataFrame(closest_words, columns=["Palabra", "Similitud coseno"]))
    
if farthest_words:
    st.subheader(f"Palabras menos cercanas a '{query}'")
    st.dataframe(pd.DataFrame(farthest_words, columns=["Palabra", "Similitud coseno"]))







