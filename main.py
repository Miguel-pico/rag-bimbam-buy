import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core import __version__ as core_version
from langchain_google_genai import ChatGoogleGenerativeAI
 modelo_embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
def main():
    print("hola mundo voy a programar")
    #comprobar mi variable de entornop
    valor = os.getenv("google_api_key")
    print(f"Mi variable de entorno es: {valor}")
    #combropar que el modelo este funcionando
    llm=ChatGoogleGenerativeAI(
        model='gemini-3.1-flash-lite',
        temperature=0
    )
    response=llm.invoke("que es rag")
    print(response)
if __name__ == "__main__":
    main()