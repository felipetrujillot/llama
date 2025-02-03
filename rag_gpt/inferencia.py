import time
import torch
import chromadb
from colorama import Fore, Style
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer

# Modelo LLM
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
print("Cargando modelo LLM...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Cargar base de datos de ChromaDB
print("Cargando base de datos de ChromaDB...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Preguntas a responder
PREGUNTAS = [
    {'pregunta': 'Hazme un muy breve resumen del documento'},
    {'pregunta': '¿Cuál es el plazo de implementación?'},
    {'pregunta': '¿Hay boleta de garantía?'},
    {'pregunta': '¿Cuándo es la fecha del periodo de preguntas?'},
    {'pregunta': '¿Cuándo es la fecha de entrega de propuesta?'},
    {'pregunta': '¿Cuándo es la fecha de respuesta de la propuesta?'},
    {'pregunta': '¿Cuándo es la fecha de firma del contrato?'},
    {'pregunta': '¿Cuáles son los límites legales de responsabilidad?'},
    {'pregunta': '¿Hay multas por incumplimiento?'},
    {'pregunta': '¿Hay marcas asociadas del RFP?'},
    {'pregunta': '¿Se exigen certificaciones?'},
    {'pregunta': '¿Hay gente en modalidad remota, teletrabajo?'},
    {'pregunta': '¿Se permite subcontratar?'},
    {'pregunta': '¿Cuál es el formato de pago?'},
    {'pregunta': '¿Cómo se entrega la propuesta y condiciones?'},
    {'pregunta': '¿Se aceptan condiciones comerciales?'},
]

# Colores de respuesta
COLORES = [
    Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA,
    Fore.CYAN, Fore.WHITE, Fore.LIGHTRED_EX, Fore.LIGHTGREEN_EX,
    Fore.LIGHTYELLOW_EX, Fore.LIGHTBLUE_EX, Fore.LIGHTMAGENTA_EX,
    Fore.LIGHTCYAN_EX, Fore.LIGHTWHITE_EX, Fore.RED, Fore.GREEN
]

# Función para responder preguntas
def responder_pregunta(pregunta, color):
    start_time = time.time()

    # Obtener contexto de ChromaDB
    docs = vectorstore.similarity_search(pregunta, k=3)
    contexto = "\n\n".join([doc.page_content for doc in docs])

    # Formatear prompt
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Inferencia
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    respuesta = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    end_time = time.time()
    tiempo = end_time - start_time

    # Mostrar respuesta con color y tiempo
    print(f"{color}Pregunta: {pregunta}{Style.RESET_ALL}")
    print(f"{color}Respuesta: {respuesta}{Style.RESET_ALL}")
    print(f"{color}Tiempo de respuesta: {tiempo:.2f} segundos{Style.RESET_ALL}\n")

if __name__ == "__main__":
    print("Comenzando inferencia...\n")
    for i, pregunta_data in enumerate(PREGUNTAS):
        responder_pregunta(pregunta_data["pregunta"], COLORES[i % len(COLORES)])
