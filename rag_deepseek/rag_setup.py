import os
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Configuración inicial
DOCUMENTS_DIR = "./documentos"
CHROMA_DB_PATH = "./chroma_db"

def setup_rag():
    # Cargar documentos PDF desde el directorio
    print("Cargando documentos PDF...")
    loader = PyPDFDirectoryLoader(DOCUMENTS_DIR)
    documents = loader.load()

    # Dividir documentos en fragmentos más pequeños
    print("Dividiendo documentos en fragmentos...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Generar embeddings con intfloat/multilingual-e5-large
    print("Generando embeddings con intfloat/multilingual-e5-large...")
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # Crear base de datos vectorial con ChromaDB
    print("Creando base de datos ChromaDB...")
    vectorstore = Chroma.from_documents(texts, embedding_model, persist_directory=CHROMA_DB_PATH)
    vectorstore.persist()
    print(f"Base de datos ChromaDB creada en: {CHROMA_DB_PATH}")

if __name__ == "__main__":
    setup_rag()