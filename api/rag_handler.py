# rag_system.py

import os
import transformers
import torch
from colorama import init, Fore, Style
import logging
from langchain.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

class RAGSystem:
    def __init__(self):
        # Configurar el nivel de logging para transformers a ERROR
        logging.getLogger("transformers").setLevel(logging.ERROR)

        # Inicializar colorama
        init(autoreset=True)

        # Configuración del modelo de embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None

    def load_documents(self, file_paths):
        documents = []
        for file_path in file_paths:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".pdf":
                loader = PyMuPDFLoader(file_path)
            elif ext in [".docx", ".doc"]:
                loader = Docx2txtLoader(file_path)
            else:
                print(Fore.RED + f"Formato de archivo no soportado: {file_path}" + Style.RESET_ALL)
                continue
            try:
                loaded = loader.load()
                documents.extend(loaded)
                print(Fore.GREEN + f"Documento cargado: {file_path}" + Style.RESET_ALL)
            except Exception as e:
                print(Fore.RED + f"Error al cargar {file_path}: {e}" + Style.RESET_ALL)
        return documents

    def create_vector_store(self, documents):
        # Dividir los documentos en fragmentos más pequeños
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # Crear vector store usando FAISS
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        print(Fore.GREEN + "Vector store creado exitosamente." + Style.RESET_ALL)

    def similarity_search(self, query, k=4):
        if not self.vector_store:
            print(Fore.RED + "El vector store no ha sido creado. Ejecuta create_vector_store primero." + Style.RESET_ALL)
            return []
        return self.vector_store.similarity_search(query, k=k)

    def save_vector_store(self, path):
        if self.vector_store:
            self.vector_store.save_local(path)
            print(Fore.GREEN + f"Vector store guardado en: {path}" + Style.RESET_ALL)
        else:
            print(Fore.RED + "El vector store no ha sido creado. No se puede guardar." + Style.RESET_ALL)

    def load_vector_store(self, path):
        if os.path.exists(path):
            self.vector_store = FAISS.load_local(path, self.embeddings)
            print(Fore.GREEN + f"Vector store cargado desde: {path}" + Style.RESET_ALL)
        else:
            print(Fore.RED + f"El archivo {path} no existe." + Style.RESET_ALL)
