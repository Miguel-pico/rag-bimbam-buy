
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings , ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
prompt = ChatPromptTemplate.from_template("""
Responde la pregunta basándote únicamente en el siguiente contexto.
Si la respuesta no está en el contexto, di que no tienes esa información.
Contexto:
{context}
Pregunta: {question}
""")

nombre_indice="langchain-rag"
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(nombre_indice)
def obtener_documentos():
    lista_documentos=[]
    obtener_ruta=Path("archivos").glob("*.pdf")
    for documento in obtener_ruta :
        try:
            cargar_archivos=PyMuPDFLoader(str(documento))
            lista_documentos.extend(cargar_archivos.load())
            print(f"Archivo cargado: {documento.name}")
        except Exception as e:
            print(f"Error cargando archivo: {documento.name}: {e}")
    return lista_documentos
def fragmentar_documentos():
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
    fragmentos=splitter.split_documents(obtener_documentos())
    return fragmentos
def crear_modelo_embeddings():
    google_api_key=os.getenv("Google_Api_Key")
    modelo_embeddings=GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=google_api_key
    )
    return modelo_embeddings
def guardar_fragmentos_en_pinecone():
    modelo_embeddings = crear_modelo_embeddings()
    fragmentos=fragmentar_documentos()
    vectorstore = PineconeVectorStore.from_documents(
        documents=fragmentos,
        embedding=modelo_embeddings,
        index_name=nombre_indice
    )
    return vectorstore
prompt = ChatPromptTemplate.from_template("""
Responde la pregunta basándote únicamente en el siguiente contexto.
Si la respuesta no está en el contexto, di que no tienes esa información.
Contexto:
{context}
Pregunta: {question}
""")
def formatear_documentos(docs):
    return "\n\n".join(doc.page_content for doc in docs)
def responder_pregunta(pregunta, vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(pregunta)
    contexto = formatear_documentos(docs)
    cadena = prompt | llm | StrOutputParser()
    respuesta = cadena.invoke({"context": contexto, "question": pregunta})
    return respuesta
def chat(vectorstore, llm):
    print("=" * 50)
    print("Asistente BimBam - Escribe 'salir' para terminar")
    print("=" * 50)
    
    while True:
        pregunta = input("\nTú: ")
        if pregunta.lower() in ["salir", "exit"]:
            print("¡Hasta luego!")
            break
        
        respuesta = responder_pregunta(pregunta, vectorstore, llm)
        print(f"\nAsistente: {respuesta}")
if __name__ == "__main__":
    #name=fragmentar_documentos()
    #print(name[18].page_content)
    #modelo = crear_modelo_embeddings()
    #vector_prueba = modelo.embed_query("prueba")
    #print(len(vector_prueba))
    #print(index.describe_index_stats())
    vectorstore=guardar_fragmentos_en_pinecone()
    #respuesta=vectorstore.similarity_search("cual son los casos elegibles para un reembolso o devolucion", k=1)
    #print(respuesta)
    llm=ChatGoogleGenerativeAI(
        model='gemini-3.1-flash-lite',
        temperature=0
    )
    chat(vectorstore, llm)
