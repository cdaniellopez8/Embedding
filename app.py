import spacy
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np

@st.cache_resource
def load_model():
    return spacy.load("es_core_news_md")

nlp = load_model()

palabras = ["gato", "perro", "amor", "ciudad", "música", "fútbol", "playa", "montaña", "libro", "escuela"]

embeddings = np.array([nlp(p).vector for p in palabras])

pca = PCA(n_components=3)
coords = pca.fit_transform(embeddings)

df = pd.DataFrame({"palabra": palabras, "x": coords[:,0], "y": coords[:,1], "z": coords[:,2]})
fig = px.scatter_3d(df, x="x", y="y", z="z", hover_name="palabra")
st.plotly_chart(fig, use_container_width=True)