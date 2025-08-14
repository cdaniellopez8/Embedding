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
# Lista gigante de palabras en español
# -----------------------
palabras = [
    # Animales
    "gato","perro","ratón","león","tigre","oso","caballo","vaca","oveja","cabra","cerdo","mono","elefante","jirafa","cebra","hipopótamo","rinoceronte","lobo","zorro","conejo","ardilla","murciélago","delfín","ballena","tiburón","pingüino","águila","halcón","pato","gallina","pavo","cisne","buho","colibrí","flamenco",
    # Frutas
    "manzana","banana","naranja","sandía","melón","pera","uva","fresa","cereza","limón","kiwi","mango","papaya","piña","ciruela","durazno","granada","maracuyá","frambuesa","arándano","higo","guayaba","mandarina","coco","lichi",
    # Vehículos
    "coche","avión","tren","barco","bicicleta","moto","camión","metro","cohete","submarino","patineta","triciclo","helicóptero","yate","velero","globo","tractor","limusina","autobús",
    # Tecnología
    "computadora","teléfono","internet","televisión","radio","cámara","impresora","robot","dron","satélite","teclado","ratón","pantalla","altavoz","micrófono","consola","tablet","proyector",
    # Música
    "música","guitarra","piano","batería","violín","flauta","trompeta","canto","danza","teatro","arpa","saxofón","clarinete","trombón","banjo","acordeón",
    # Deportes
    "fútbol","baloncesto","tenis","natación","atletismo","voleibol","rugby","surf","boxeo","ciclismo","golf","esquí","snowboard","esgrima","karate","judo","taekwondo","gimnasia","escalada",
    # Lugares
    "río","montaña","océano","ciudad","pueblo","bosque","desierto","lago","cascada","isla","playa","valle","glaciar","cueva","volcán","pradera","acantilado","selva","campo",
    # Educación
    "escuela","universidad","profesor","estudiante","biblioteca","libro","cuaderno","pizarra","lápiz","pluma","mochila","regla","calculadora","tiza",
    # Emociones
    "amistad","amor","trabajo","familia","hogar","salud","dinero","felicidad","tristeza","miedo","ira","esperanza","ansiedad","orgullo","vergüenza","confianza","celos","paz","odio",
    # Ciencias y artes
    "historia","ciencia","arte","filosofía","literatura","poesía","matemáticas","física","química","biología","geografía","astronomía","pintura","escultura","fotografía","cine",
    # Naturaleza
    "cielo","mar","tierra","fuego","aire","lluvia","nieve","viento","tormenta","huracán","terremoto","inundación","marea","nube","rayo","granizo","rocío","brisa",
    # Comida
    "pan","queso","carne","pescado","huevo","arroz","pasta","ensalada","sopa","pizza","hamburguesa","taco","arepa","empanada","paella","curry","helado","chocolate","galleta","pastel","flan",
    # Tiempo
    "lunes","martes","miércoles","jueves","viernes","sábado","domingo","enero","febrero","marzo","abril","mayo","junio","julio","agosto","septiembre","octubre","noviembre","diciembre",
    # Profesiones
    "doctor","abogado","ingeniero","maestro","enfermero","piloto","chef","mecánico","bombero","policía","arquitecto","científico","pintor","cantante","actor","escritor","programador","carpintero","panadero","electricista","fotógrafo",
    # Colores
    "rojo","azul","verde","amarillo","naranja","morado","rosa","negro","blanco","gris","marrón","beige","turquesa","violeta","dorado","plateado",
    # Objetos cotidianos
    "mesa","silla","cama","puerta","ventana","lámpara","reloj","cuadro","alfombra","cortina","armario","espejo","sofá","televisor","teléfono","taza","plato","cuchara","tenedor","cuchillo",
]

# -----------------------
# Sidebar con buscador
# -----------------------
st.sidebar.header("Buscador de palabra")
palabra_buscar = st.sidebar.text_input("Escribe una palabra en español:").strip().lower()

if palabra_buscar and palabra_buscar not in palabras:
    palabras.append(palabra_buscar)

# -----------------------
# Obtener embeddings
# -----------------------
embeddings = np.array([nlp(p).vector for p in palabras])

# -----------------------
# Reducir a 3D
# -----------------------
pca = PCA(n_components=3)
coords = pca.fit_transform(embeddings)

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
    width=1000,
    height=1000
)

fig.update_traces(marker=dict(size=5))
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