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

st.set_page_config(
    page_title="Embedding Visualization",
    page_icon="🌐",  
    layout="wide"
)

# Lista de palabras (tu lista completa)
palabras = [
    # Animales
    "gato", "perro", "raton", "lobo", "zorro", "tigre", "oso", "caballo", "vaca", "oveja", "cabra", "conejo",
    "ardilla", "murcielago", "ciervo", "jirafa", "elefante", "leon", "pantera", "puma", "gorila", "mono", "chimpance",
    "orangutan", "ballena", "delfin", "tiburon", "foca", "pinguino", "aguila", "halcon", "buho", "paloma", "canario",
    "loro", "pavo", "pollo", "gallina", "gallo", "pato", "cisne", "ganso", "serpiente", "cocodrilo", "lagarto",
    "iguana", "sapo", "tortuga", "cangrejo", "langosta", "camello", "hipopotamo", "rinoceronte", "zebra", "bufalo",
    "bisonte", "cabrito", "potro", "mula", "lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo",
    "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
    # Colores
    "azul", "rojo", "blanco", "negro", "verde", "naranja", "celeste", "cian", "magenta", "marron", "rosa",
    # Sentimientos
    "alegria", "tristeza", "ira", "miedo", "sorpresa", "asco", "amor", "odio", "esperanza", "desesperacion",
    "nostalgia", "orgullo", "verguenza", "culpa", "gratitud", "compasion", "envidia", "celos", "ansiedad", "calma",
    "satisfaccion", "melancolia", "euforia", "pena", "soledad", "confianza", "desconfianza", "admiracion", "rechazo", "ternura",
    # Filosofia
    "existencia", "esencia", "libertad", "determinismo", "moralidad", "etica", "virtud", "justicia", "verdad", "mentira",
    "belleza", "fealdad", "tiempo", "eternidad", "infinito", "causalidad", "realidad", "ilusion", "ser", "nada",
    "pensamiento", "razon", "emocion", "conocimiento", "ignorancia", "sabiduria", "duda", "certeza", "voluntad", "destino",
    # Frutas y verduras
    "manzana", "banana", "sandia", "melon", "pera", "uva", "fresa", "cereza", "limon", "kiwi", "papaya", "mango", "guayaba",
    "maracuya", "pina", "durazno", "ciruela", "granada", "higo", "tomate", "lechuga", "zanahoria", "cebolla", "ajo", "pepino",
    "calabaza", "berenjena", "brocoli", "coliflor", "espinaca", "pimiento", "chile", "maiz", "guisante", "haba", "apio",
    "rabano", "alcachofa", "batata",

    # Elementos quimicos
    "hidrogeno", "helio", "litio", "berilio", "boro", "carbono", "nitrogeno", "oxigeno", "fluor", "neon",
    "sodio", "magnesio", "aluminio", "silicio", "fosforo", "azufre", "cloro", "argon", "potasio", "calcio",
    "escandio", "titanio", "vanadio", "cromo", "manganeso", "hierro", "cobalto", "niquel", "cobre", "zinc",
    "galio", "germanio", "arsenico", "selenio", "bromo", "cripton", "rubidio", "estroncio", "itrio", "circonio",
    "niobio", "molibdeno", "tecnecio", "rutenio", "rodio", "paladio", "plata", "cadmio", "indio", "estano",
    "antimonio", "telurio", "yodo", "xenon", "cesio", "bario", "lantano", "cerio", "praseodimio", "neodimio",
    "prometio", "samario", "europio", "gadolinio", "terbio", "disprosio", "holmio", "erbio", "iterbio",
    "lutecio", "hafnio", "tantalio", "wolframio", "renio", "osmio", "iridio", "platino", "oro", "mercurio",
    "talio", "plomo", "bismuto", "polonio", "astato", "radon", "francio", "radio", "actinio", "torio",
    "protactinio", "uranio", "neptunio", "plutonio", "americio", "curio", "berkelio", "californio", "einsteinio",
    "fermio", "mendelevio", "nobelio", "lawrencio",

    # Dinero y economia
    "dinero", "moneda", "billete", "banco", "inversion", "capital", "finanzas", "ahorro", "gasto",
    "ganancia", "perdida", "credito", "deficit", "saldo", "presupuesto", "deuda", "impuesto", "ingreso", "interes",
    "lucro", "beneficio", "dividendo", "patrimonio", "hipoteca", "prestamo", "pago", "cobro", "transaccion", "tarifa",
    "cotizacion", "divisa", "mercado", "comercio", "contrato", "venta", "compra", "exportacion", "importacion", "precio",
    "tarjeta", "cuenta", "subsidio", "bono", "subasta", "acciones", "coste", "oferta", "demanda", "fondo",
    "arancel", "liquidez", "rentabilidad", "riesgo", "flujo", "inflacion", "deflacion", "valorizacion", "desvalorizacion", "cheque",
    "billetera", "saldo", "ahorrador", "inversionista", "cotista", "pagador", "deudor", "acreedor", "financista", "cajero",

    # Ciudades del mundo
    "bogota", "paris", "londres", "roma", "madrid", "berlin", "amsterdam", "bruselas", "viena", "lisboa",
    "atenas", "estocolmo", "oslo", "helsinki", "copenhague", "moscu", "varsovia", "praga", "budapest", "dublin",
    "zurich", "ginebra", "munich", "frankfurt", "barcelona", "valencia", "sevilla", "granada", "bilbao", "san sebastian",
    "tokio", "osaka", "kioto", "seul", "pekin", "shanghai", "hong kong", "taipei", "singapur", "bangkok",
    "kuala lumpur", "manila", "hanoi", "yakarta", "delhi", "bombay", "calcuta", "dubai", "abu dabi", "doha",
    "el cairo", "casablanca", "marrakech", "tunez", "johannesburgo", "ciudad del cabo", "nairobi", "lagos", "accra", "dakar",
    "nueva york", "los angeles", "chicago", "san francisco", "miami", "houston", "toronto", "vancouver", "montreal", "ottawa",
    "mexico d.f.", "guadalajara", "monterrey", "buenos aires", "cordoba", "rosario", "santiago", "valparaiso", "lima", "cusco",
    "quito", "guayaquil", "la paz", "santa cruz", "asuncion", "montevideo", "punta del este", "caracas", "maracaibo", "san juan",
    "medellin", "barranquilla", "santa marta", "cartagena", "valledupar", "junior", "carnaval", "cali", "senior", "desarrolador", 
    
    # Transportes
    "carro", "avion", "tren", "barco", "bicicleta", "moto", "camion", "metro", "cohete", "submarino", "tractor", "helicoptero",
    "patinete", "globo", "yate", "velero", "canoa", "kayak", "autobus", "tranvia",
    # Tecnologia
    "computadora", "telefono", "internet", "television", "radio", "camara", "impresora", "robot", "dron", "satelite",
    "teclado", "pantalla", "auriculares", "microfono", "altavoz", "software", "hardware", "servidor", "red",
    # Lugares
    "ciudad", "pueblo", "aldea", "capital", "barrio", "calle", "avenida", "plaza", "puente", "parque", "jardin", "museo",
    "biblioteca", "universidad", "escuela", "estadio", "cine", "teatro", "restaurante", "hotel", "playa", "montana", "rio",
    "lago", "mar", "oceano", "desierto", "bosque", "selva", "caverna", "volcan", "catarata", "isla", "peninsula", "glaciar",
    "acantilado", "valle", "llanura", "pradera", "costa",
    # Conceptos abstractos
    "amistad", "felicidad", "valentia", "fe", "paz", "guerra", "honor", "perdon", "solidaridad", "paciencia",
    "depresion", "entusiasmo", "energia", "fuerza", "debilidad", "riqueza", "pobreza", "exito", "fracaso", "motivacion",
    "inspiracion", "creatividad"
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
    query = st.text_input("Escribe que palabra quieres buscar en el embedding!", placeholder="No te demores... (ง︡'-'︠)ง")

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















