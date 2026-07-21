import streamlit as st
from agente import invocar_llm, responder_pregunta, crear_modelo_embeddings, nombre_indice
from langchain_pinecone import PineconeVectorStore

st.set_page_config(page_title="Asistente BimBam", page_icon="🤖")
st.title("🤖 Asistente Bim-Bam-Buy")

@st.cache_resource
def cargar_recursos():
    modelo_embeddings = crear_modelo_embeddings()
    vectorstore = PineconeVectorStore(
        index_name=nombre_indice,
        embedding=modelo_embeddings
    )
    llm = invocar_llm()
    return vectorstore, llm

vectorstore, llm = cargar_recursos()

if "historial" not in st.session_state:
    st.session_state.historial = []

for msg in st.session_state.historial:
    with st.chat_message(msg["rol"]):
        st.write(msg["contenido"])

pregunta = st.chat_input("Escribe tu pregunta...")

if pregunta:
    st.session_state.historial.append({"rol": "user", "contenido": pregunta})
    with st.chat_message("user"):
        st.write(pregunta)

    with st.spinner("Pensando..."):
        respuesta = responder_pregunta(pregunta, vectorstore, llm)

    st.session_state.historial.append({"rol": "assistant", "contenido": respuesta})
    with st.chat_message("assistant"):
        st.write(respuesta)