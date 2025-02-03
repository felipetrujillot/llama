import os
import chromadb
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Directorio de documentos
DOCUMENTS_DIR = "./documentos"

# Cargar documentos PDF
def load_documents():
    documents = []
    for file in os.listdir(DOCUMENTS_DIR):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DOCUMENTS_DIR, file))
            documents.extend(loader.load())
    return documents

# Indexar documentos en ChromaDB
def index_documents():
    print("Cargando documentos...")
    documents = load_documents()

    if not documents:
        print("No se encontraron documentos PDF en", DOCUMENTS_DIR)
        return

    # Crear embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Crear base de datos vectorial en ChromaDB
    vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")

    print("Indexación completada con éxito.")

if __name__ == "__main__":
    index_documents()
