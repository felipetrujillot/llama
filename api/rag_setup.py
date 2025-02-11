import os
import logging
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pypdf import PdfReader

# Configuración inicial
DOCUMENTS_DIR = "./documentos"
CHROMA_DB_PATH = "./chroma_db"

# Habilitar registro detallado
logging.basicConfig(level=logging.DEBUG)

def is_pdf_encrypted(file_path):
    """Verifica si un archivo PDF está cifrado."""
    reader = PdfReader(file_path)
    return reader.is_encrypted

def decrypt_pdf(file_path, password):
    """Desbloquea un archivo PDF cifrado."""
    reader = PdfReader(file_path)
    if reader.is_encrypted:
        try:
            reader.decrypt(password)
            logging.info(f"Archivo desbloqueado: {file_path}")
        except Exception as e:
            logging.error(f"No se pudo desbloquear el archivo {file_path}: {e}")
            raise
    return reader

def setup_rag():
    # Cargar documentos PDF desde el directorio
    print("Cargando documentos PDF...")
    loader = PyPDFDirectoryLoader(DOCUMENTS_DIR)
    
    try:
        documents = loader.load()
    except Exception as e:
        logging.error(f"Error al cargar documentos: {e}")
        return

    # Dividir documentos en fragmentos más pequeños
    print("Dividiendo documentos en fragmentos...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Generar embeddings con intfloat/multilingual-e5-large
    print("Generando embeddings con intfloat/multilingual-e5-large...")
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # Crear base de datos vectorial con ChromaDB
    print("Creando base de datos ChromaDB...")
    try:
        vectorstore = Chroma.from_documents(texts, embedding_model, persist_directory=CHROMA_DB_PATH)
        vectorstore.persist()
        print(f"Base de datos ChromaDB creada en: {CHROMA_DB_PATH}")
    except Exception as e:
        logging.error(f"Error al crear la base de datos ChromaDB: {e}")

if __name__ == "__main__":
    # Verificar si hay archivos PDF cifrados en el directorio
    for root, _, files in os.walk(DOCUMENTS_DIR):
        for file in files:
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(root, file)
                if is_pdf_encrypted(file_path):
                    logging.warning(f"El archivo está cifrado: {file_path}")
                    password = input(f"Ingrese la contraseña para desbloquear {file_path}: ")
                    decrypt_pdf(file_path, password)

    # Ejecutar el flujo principal
    setup_rag()