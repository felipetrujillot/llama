# rag_system.py

import os
import transformers
import torch
from colorama import init, Fore, Style
import logging
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class RAGSystem:
    def __init__(self, documentos_dir="documentos"):
        # Configurar el nivel de logging para transformers a ERROR
        logging.getLogger("transformers").setLevel(logging.ERROR)

        # Inicializar colorama
        init(autoreset=True)

        # Configuración del modelo de embeddings
        # self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")
        self.vector_store = None
        self.documentos_dir = documentos_dir

    def load_documents_from_directory(self):
        """
        Carga todos los documentos PDF y Word desde el directorio especificado.
        """
        documentos_path = os.path.join(os.getcwd(), self.documentos_dir)
        if not os.path.exists(documentos_path):
            print(Fore.RED + f"El directorio '{self.documentos_dir}' no existe. Creándolo..." + Style.RESET_ALL)
            os.makedirs(documentos_path)
            print(Fore.YELLOW + f"Por favor, coloca tus documentos en el directorio '{self.documentos_dir}' y reinicia el script." + Style.RESET_ALL)
            return []

        documents = []
        supported_extensions = [".pdf", ".docx", ".doc"]
        for filename in os.listdir(documentos_path):
            file_path = os.path.join(documentos_path, filename)
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in supported_extensions:
                print(Fore.RED + f"Formato de archivo no soportado: {filename}" + Style.RESET_ALL)
                continue

            try:
                if ext == ".pdf":
                    loader = PyMuPDFLoader(file_path)
                elif ext in [".docx", ".doc"]:
                    loader = Docx2txtLoader(file_path)
                loaded = loader.load()
                documents.extend(loaded)
                print(Fore.GREEN + f"Documento cargado: {filename}" + Style.RESET_ALL)
            except Exception as e:
                print(Fore.RED + f"Error al cargar {filename}: {e}" + Style.RESET_ALL)
        return documents

    def create_vector_store(self, documents):
        """
        Crea un vector store a partir de los documentos proporcionados.
        """
        if not documents:
            print(Fore.RED + "No hay documentos para procesar." + Style.RESET_ALL)
            return

        # Dividir los documentos en fragmentos más pequeños
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # Crear vector store usando FAISS
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        print(Fore.GREEN + "Vector store creado exitosamente." + Style.RESET_ALL)

    def similarity_search(self, query, k=4):
        """
        Realiza una búsqueda de similitud para la consulta dada y retorna los k fragmentos más relevantes.
        """
        if not self.vector_store:
            print(Fore.RED + "El vector store no ha sido creado. Ejecuta create_vector_store primero." + Style.RESET_ALL)
            return []
        return self.vector_store.similarity_search(query, k=k)

    def save_vector_store(self, path="faiss_index"):
        """
        Guarda el vector store en el disco.
        """
        if self.vector_store:
            self.vector_store.save_local(path, allow_dangerous_deserialization=True)
            print(Fore.GREEN + f"Vector store guardado en: {path}" + Style.RESET_ALL)
        else:
            print(Fore.RED + "El vector store no ha sido creado. No se puede guardar." + Style.RESET_ALL)

    def load_vector_store(self, path="faiss_index"):
        """
        Carga un vector store previamente guardado desde el disco.
        """
        if os.path.exists(path):
            self.vector_store = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
            print(Fore.GREEN + f"Vector store cargado desde: {path}" + Style.RESET_ALL)
        else:
            print(Fore.RED + f"El archivo '{path}' no existe." + Style.RESET_ALL)
