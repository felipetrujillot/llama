import time
from colorama import Fore, Style, init
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Inicializar Colorama
init(autoreset=True)

# Configuración inicial
CHROMA_DB_PATH = "./chroma_db"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Preguntas a responder
QUESTIONS = [
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

# Cargar modelo Qwen 2.5
print("Cargando modelo Qwen 2.5...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

# Cargar ChromaDB
print("Cargando base de datos ChromaDB...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)

def generate_response(prompt, context):
    # Crear el mensaje con el contexto explícito
    messages = [
        {"role": "system", "content": "Eres un asistente experto. Usa el contexto proporcionado para responder la pregunta."},
        {"role": "user", "content": f"Pregunta: {prompt}\n\nContexto:\n{context}"}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def main():
    for question in QUESTIONS:
        start_time = time.time()
        query = question['pregunta']
        print(f"\nPregunta: {query}")

        # Recuperar contexto relevante del RAG
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        # Mostrar el contexto recuperado (para depuración)
        print(Fore.CYAN + "Contexto recuperado:")
        print(context[:500] + "...")  # Mostrar solo los primeros 500 caracteres

        # Generar respuesta con Qwen 2.5
        response = generate_response(query, context)
        elapsed_time = time.time() - start_time

        # Mostrar respuesta y tiempo en colores
        print(Fore.GREEN + f"Respuesta: {response}")
        print(Fore.YELLOW + f"Tiempo de respuesta: {elapsed_time:.2f} segundos" + Style.RESET_ALL)

if __name__ == "__main__":
    main()