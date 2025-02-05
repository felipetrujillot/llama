from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import time

# Inicializar FastAPI
app = FastAPI()

# Configuración inicial
CHROMA_DB_PATH = "./chroma_db"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"  # Modelo de razonamiento

# Cargar modelo DeepSeek-R1-Distill-Qwen-14B
print("Cargando modelo DeepSeek-R1-Distill-Qwen-14B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

# Cargar ChromaDB con el modelo de embeddings
print("Cargando base de datos ChromaDB...")
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)

# Modelo de entrada para la API
class QuestionRequest(BaseModel):
    pregunta: str

# Función para generar respuestas
def generate_response(prompt, context):
    # Crear el mensaje con el contexto explícito
    messages = [
        {"role": "system", "content": """
        Eres un asistente experto diseñado para responder preguntas basadas exclusivamente en el contexto proporcionado.
        Piensa paso a paso antes de responder. Si la información solicitada no está presente en el contexto, NO INVENTES respuestas. 
        En su lugar, indica claramente que no se encontró la información y, si es posible, proporciona sugerencias generales o información relacionada que pueda ser útil.
        Aquí tienes un ejemplo:
        Pregunta: ¿Cuál es el plazo de implementación?
        Contexto: El plazo de implementación es de 6 meses según lo especificado en el documento.
        Respuesta: El plazo de implementación es de 6 meses.
        ---
        Ahora, responde la siguiente pregunta:
        """},
        {"role": "user", "content": f"""
        Pregunta: {prompt}
        Contexto (RFP):
        {context}
        """}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,  # Reducir la creatividad para respuestas más precisas
        do_sample=True
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# Endpoint para enviar preguntas
@app.post("/pregunta/")
async def responder_pregunta(request: QuestionRequest):
    start_time = time.time()
    query = request.pregunta

    # Recuperar contexto relevante del RAG
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # Verificar si el contexto es relevante
    if not context.strip() or "no encontrado" in context.lower():
        response = "No se encontró información relevante en el contexto proporcionado."
    else:
        # Generar respuesta con DeepSeek-R1-Distill-Qwen-14B
        response = generate_response(query, context)

    elapsed_time = time.time() - start_time
    return {
        "pregunta": query,
        "respuesta": response,
        "tiempo_respuesta_segundos": round(elapsed_time, 2)
    }

# Mensaje de bienvenida
@app.get("/")
async def root():
    return {"mensaje": "Bienvenido a la API de RAG con DeepSeek-R1-Distill-Qwen-14B. Envía tus preguntas al endpoint /pregunta/."}