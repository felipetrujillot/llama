# rag_handler.py

import os
import warnings
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # Importación actualizada
from typing import List

# Ignorar advertencias de deprecación específicas de LangChain
warnings.filterwarnings("ignore", category=UserWarning, module='langchain')

# Configuración
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Puedes elegir otro modelo de HuggingFace
CHROMA_DB_DIR = "chroma_db"  # Directorio donde se almacenará la base de datos

# Inicializar el modelo de embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def cargar_documento(ruta_documento: str):
    """
    Carga un documento PDF o Word y lo divide en páginas o secciones.
    """
    if ruta_documento.lower().endswith(".pdf"):
        loader = PyPDFLoader(ruta_documento)
    elif ruta_documento.lower().endswith((".doc", ".docx")):
        loader = UnstructuredWordDocumentLoader(ruta_documento)
    else:
        raise ValueError("Formato de documento no soportado. Usa PDF o Word.")
    
    documentos = loader.load()
    return documentos

def crear_vectorstore(documentos: List, persistencia: bool = True):
    """
    Crea una base de datos vectorial en ChromaDB a partir de los documentos proporcionados.
    """
    vectorstore = Chroma.from_documents(documents=documentos, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
    if persistencia:
        vectorstore.persist()
    return vectorstore

def inicializar_vectorstore():
    """
    Inicializa la base de datos vectorial existente o crea una nueva si no existe.
    """
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    return vectorstore

def agregar_documento_a_vectorstore(ruta_documento: str):
    """
    Carga un documento y lo agrega a la base de datos vectorial.
    """
    documentos = cargar_documento(ruta_documento)
    vectorstore = inicializar_vectorstore()
    vectorstore.add_documents(documents=documentos)
    vectorstore.persist()
    print(f"Documento '{ruta_documento}' agregado exitosamente a la base de datos vectorial.")

def obtener_contexto(query: str, k: int = 5) -> List[str]:
    """
    Recupera los k fragmentos más relevantes de la base de datos vectorial para la consulta dada.
    Ajusta k si es mayor que el número de documentos disponibles.
    """
    vectorstore = inicializar_vectorstore()
    total_documentos = len(vectorstore)
    
    if total_documentos < k:
        k = total_documentos
        print(f"Advertencia: Solo hay {total_documentos} documentos disponibles. Ajustando k a {k}.")
    
    resultados = vectorstore.similarity_search(query, k=k)
    contextos = [doc.page_content for doc in resultados]
    return contextos

# Script principal para agregar documentos (opcional)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agregar documentos a la base de datos vectorial de ChromaDB.")
    parser.add_argument("ruta_documento", type=str, help="Ruta al documento PDF o Word a agregar.")
    args = parser.parse_args()

    agregar_documento_a_vectorstore(args.ruta_documento)
