from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from fastapi.responses import StreamingResponse
import asyncio
import torch
import threading

# Inicializar FastAPI
app = FastAPI()

# Configuración inicial
CHROMA_DB_PATH = "./chroma_db"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

# Cargar modelo
print("Cargando modelo DeepSeek-R1-Distill-Qwen-14B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,  # Usa float16 si la GPU lo permite
    device_map="auto"
)

# Cargar ChromaDB con el modelo de embeddings
print("Cargando base de datos ChromaDB...")
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)

# Modelo de entrada para la API
class QuestionRequest(BaseModel):
    pregunta: str

# **Corrección del streaming**
async def generate_response_stream(prompt, context):
    messages = [
        {"role": "system", "content": """
        Eres un asistente experto diseñado para responder preguntas basadas exclusivamente en el contexto proporcionado.
        No inventes respuestas. Si la información no está en el contexto, indica que no se encontró.
        """},
        {"role": "user", "content": f"Pregunta: {prompt}\nContexto:\n{context}"}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, decode_kwargs={"skip_special_tokens": True})

    # Iniciar la generación en un hilo separado
    thread = threading.Thread(
        target=model.generate,
        kwargs={
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "max_new_tokens": 512,
            "temperature": 0.7,
            "do_sample": True,
            "streamer": streamer
        }
    )
    thread.start()

    # **Corrección: Usa `asyncio.to_thread()` para no bloquear FastAPI**
    for token in await asyncio.to_thread(lambda: list(streamer)):
        yield token
        await asyncio.sleep(0.01)

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
