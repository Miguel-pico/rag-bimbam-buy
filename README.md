# Asistente BimBamBuy - RAG con LangChain, Gemini y Pinecone

Asistente conversacional que responde preguntas basándose en documentos PDF (Política de Reembolsos y Devoluciones de BimBam) usando RAG (Retrieval-Augmented Generation).

## Tecnologías

- Python
- Streamlit (interfaz web)
- LangChain
- Google Gemini (LLM y embeddings)
- Pinecone (base de datos vectorial)

## Requisitos previos

- Python 3.12+
- Cuenta de [Google AI Studio](https://aistudio.google.com/) (API key de Gemini)
- Cuenta de [Pinecone](https://www.pinecone.io/) (API key + índice creado con dimensión compatible con tu modelo de embeddings)

## Instalación (local)

1. Clona el repositorio:
```bash
git clone https://github.com/Miguel-pico/rag-bimbam-buy.git
cd rag-bimbam-buy
```

2. Crea y activa un entorno virtual:
```bash
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Linux/Mac
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

4. Crea un archivo `.env` en la raíz del proyecto con tus API keys:
```
GOOGLE_API_KEY=tu_key_aqui
PINECONE_API_KEY=tu_key_aqui
```

5. Coloca tus archivos PDF en la carpeta `archivos/`

## Primera configuración (subir documentos a Pinecone)

Antes de usar la app por primera vez, debes subir tus PDFs al índice de Pinecone. Con tus PDFs ya en la carpeta `archivos/`, ejecuta una sola vez:

```python
from asistente import guardar_fragmentos_en_pinecone
guardar_fragmentos_en_pinecone()
```

Esto solo es necesario la primera vez, o cuando agregues documentos nuevos.

## Uso

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`.

## Despliegue

Este proyecto está desplegado en Hugging Face Spaces:

🔗 [Ver aplicación en vivo](https://huggingface.co/spaces/tu_usuario/asistente-bimbam)

### Cómo desplegar tu propia copia

1. Crea un Space en [Hugging Face](https://huggingface.co/new-space) con SDK **Docker** + plantilla de Streamlit
2. Sube tu código (sin `venv/` ni `.env`)
3. Configura tus API keys en **Settings → Repository secrets**:
   - `GOOGLE_API_KEY`
   - `PINECONE_API_KEY`
4. El Space se construirá y desplegará automáticamente

## Estructura del proyecto

```
proyecto/
├── app.py             # Interfaz de Streamlit (punto de entrada)
├── asistente.py       # Lógica principal (carga, fragmentación, vectorstore, RAG)
├── requirements.txt
├── .gitignore
├── .env               # No incluido (contiene tus API keys)
└── archivos/          # PDFs a procesar
```

## Cómo funciona

1. **Carga de documentos**: se leen los PDFs de la carpeta `archivos/` con `PyMuPDFLoader`
2. **Fragmentación**: los documentos se dividen en fragmentos de 1000 caracteres con `RecursiveCharacterTextSplitter`
3. **Embeddings**: cada fragmento se convierte en un vector numérico usando el modelo `gemini-embedding-001`
4. **Almacenamiento**: los vectores se guardan en un índice de Pinecone
5. **Consulta**: cuando el usuario hace una pregunta, se busca por similitud los fragmentos más relevantes (retriever)
6. **Respuesta**: el modelo `gemini-3.1-flash-lite` genera una respuesta en lenguaje natural basada únicamente en esos fragmentos