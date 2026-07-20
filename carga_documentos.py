import os
import tempfile
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
google_api_key=os.getenv("Google_Api_Key")
def cargar_documentos_pdf():
    lista_documentos=[]    #alt+91 para abrir corchetes
    ruta_documentos=Path("archivos").glob("*.pdf")
    for documento in ruta_documentos :
        try:
            cargar_archivos=PyMuPDFLoader(str(documento))
            lista_documentos.extend(cargar_archivos.load())
            print(f"Archivo cargado: {documento.name}")
        except Exception as e:
            print(f"Error cargando archivo: {documento.name}: {e}")
    print(f"Total de documentos cargados: {len(lista_documentos)}")
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
    fragmentos=splitter.split_documents(lista_documentos)
    modelo_embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",google_api_key=google_api_key)
    vectorstore=InMemoryVectorStore.from_documents(documents=fragmentos,embedding=modelo_embeddings)
    retriever=vectorstore.as_retriever(search_kwargs={"k":2})
    query="cual son los casos elegibles para un reembolso o devolucion"
    query_embed=modelo_embeddings.embed_query(query)
    fragmentos_similares=retriever.invoke(query)
    fragmentos_similares = [fragmento.page_content for fragmento in fragmentos_similares]
    print(fragmentos_similares)

if __name__ == "__main__":
    cargar_documentos_pdf()