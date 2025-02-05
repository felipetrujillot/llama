from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from fastapi.responses import StreamingResponse
import asyncio
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

# Función para generar respuestas en tiempo real
async def generate_response_stream(prompt, context):
    # Crear el mensaje con el contexto explícito
    messages = [
        {"role": "system", "content": """
        Eres un asistente experto diseñado para responder preguntas basadas exclusivamente en el contexto proporcionado.
        Piensa paso a paso antes de responder. Si la información solicitada no está presente en el contexto, NO INVENTES respuestas. 
        En su lugar, indica claramente que no se encontró la información y, si es posible, proporciona sugerencias generales o información relacionada que pueda ser útil.
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

    # Generar tokens uno por uno
    for token in model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        streamer=None  # Usamos un generador manual
    ):
        decoded_token = tokenizer.decode(token, skip_special_tokens=True)
        yield decoded_token  # Enviamos cada token generado
        await asyncio.sleep(0.01)  # Pequeña pausa para evitar sobrecarga

# Endpoint para enviar preguntas con streaming
@app.post("/pregunta-stream/")
async def responder_pregunta_stream(request: QuestionRequest):
    query = request.pregunta

    # Recuperar contexto relevante del RAG
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # Verificar si el contexto es relevante
    if not context.strip() or "no encontrado" in context.lower():
        async def no_context_stream():
            yield "No se encontró información relevante en el contexto proporcionado."
        return StreamingResponse(no_context_stream(), media_type="text/plain")

    # Devolver la respuesta en tiempo real
    return StreamingResponse(
        generate_response_stream(query, context),
        media_type="text/plain"
    )

# Endpoint básico para verificar el estado de la API
@app.get("/status")
async def status():
    return {"status": "ok"}

# Mensaje de bienvenida
@app.get("/")
async def root():
    return {"mensaje": "Bienvenido a la API de RAG con DeepSeek-R1-Distill-Qwen-14B. Envía tus preguntas al endpoint /pregunta-stream/."}