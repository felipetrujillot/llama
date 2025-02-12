import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm
import argparse

# Configuración inicial
def parse_args():
    parser = argparse.ArgumentParser(description="Configuración del sistema RAG.")
    parser.add_argument("--documents_dir", default="./documentos", help="Directorio de documentos PDF.")
    parser.add_argument("--chroma_db_path", default="./chroma_db", help="Ruta de la base de datos ChromaDB.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Tamaño de los fragmentos de texto.")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Superposición entre fragmentos.")
    return parser.parse_args()

def validate_documents_directory(documents_dir):
    if not os.path.exists(documents_dir):
        raise FileNotFoundError(f"El directorio '{documents_dir}' no existe.")
    if not os.listdir(documents_dir):
        raise ValueError(f"El directorio '{documents_dir}' está vacío.")
    print(f"Directorio '{documents_dir}' validado correctamente.")

def load_or_create_vectorstore(texts, embedding_model, chroma_db_path):
    if os.path.exists(chroma_db_path) and os.listdir(chroma_db_path):
        print("Cargando base de datos ChromaDB existente...")
        return Chroma(persist_directory=chroma_db_path, embedding_function=embedding_model)
    else:
        print("Creando nueva base de datos ChromaDB...")
        vectorstore = Chroma.from_documents(texts, embedding_model, persist_directory=chroma_db_path)
        # Eliminamos la llamada a persist(), ya no es necesaria
        return vectorstore

def setup_rag(documents_dir, chroma_db_path, chunk_size, chunk_overlap):
    try:
        # Validar directorio de documentos
        validate_documents_directory(documents_dir)

        # Cargar documentos PDF desde el directorio
        print("Cargando documentos PDF...")
        loader = PyPDFDirectoryLoader(documents_dir)
        documents = loader.load()

        # Dividir documentos en fragmentos más pequeños
        print("Dividiendo documentos en fragmentos...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(tqdm(documents, desc="Procesando documentos"))

        # Generar embeddings con intfloat/multilingual-e5-large
        print("Generando embeddings con intfloat/multilingual-e5-large...")
        embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

        # Crear o cargar base de datos vectorial con ChromaDB
        vectorstore = load_or_create_vectorstore(texts, embedding_model, chroma_db_path)
        print(f"Base de datos ChromaDB lista en: {chroma_db_path}")

    except Exception as e:
        print(f"Ocurrió un error: {str(e)}")

if __name__ == "__main__":
    args = parse_args()
    DOCUMENTS_DIR = args.documents_dir
    CHROMA_DB_PATH = args.chroma_db_path
    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap

    setup_rag(DOCUMENTS_DIR, CHROMA_DB_PATH, CHUNK_SIZE, CHUNK_OVERLAP)