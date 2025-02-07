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
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Cargar modelo
print("Cargando modelo...")
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

# Función para generar tokens uno por uno
async def generate_response_stream(prompt, context):
    messages = [
        {"role": "system", "content": "Eres un asistente experto que responde preguntas basadas exclusivamente en el contexto proporcionado."},
        {"role": "user", "content": f"Pregunta: {prompt}\nContexto:\n{context}"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, decode_kwargs={"skip_special_tokens": True})
    
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
    
    try:
        # Bandera para detectar el inicio de la respuesta generada
        response_started = False
        accumulated_text = ""

        for token in streamer:
            # Decodificar el token
            decoded_token = tokenizer.decode(token, skip_special_tokens=True)
            accumulated_text += decoded_token

            # Detectar el inicio de la respuesta generada
            if not response_started and "<think>" in accumulated_text:
                response_started = True
                # Eliminar todo el contenido previo a <think>
                accumulated_text = accumulated_text[accumulated_text.index("<think>"):]

            # Solo enviar tokens si la respuesta ha comenzado
            if response_started:
                yield decoded_token
                await asyncio.sleep(0.01)  # Retraso artificial para forzar el envío inmediato
    finally:
        # Asegurarse de que la conexión se cierre correctamente
        yield ""  # Enviar un chunk vacío al final

@app.post("/pregunta-stream/")
async def responder_pregunta_stream(request: QuestionRequest):from fastapi import FastAPI
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
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Cargar modelo
print("Cargando modelo...")
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

# Función para generar tokens uno por uno
async def generate_response_stream(prompt, context):
    messages = [
        {"role": "system", "content": "Eres un asistente experto que responde preguntas basadas exclusivamente en el contexto proporcionado."},
        {"role": "user", "content": f"Pregunta: {prompt}\nContexto:\n{context}"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, decode_kwargs={"skip_special_tokens": True})
    
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
    
    try:
        # Bandera para detectar el inicio de la respuesta generada
        response_started = False
        accumulated_text = ""

        for token in streamer:
            # Decodificar el token
            decoded_token = tokenizer.decode(token, skip_special_tokens=True)
            accumulated_text += decoded_token

            # Detectar el inicio de la respuesta generada
            if not response_started and "<think>" in accumulated_text:
                response_started = True
                # Eliminar todo el contenido previo a <think>
                accumulated_text = accumulated_text[accumulated_text.index("<think>"):]

            # Solo enviar tokens si la respuesta ha comenzado
            if response_started:
                yield decoded_token
                await asyncio.sleep(0.01)  # Retraso artificial para forzar el envío inmediato
    finally:
        # Asegurarse de que la conexión se cierre correctamente
        yield ""  # Enviar un chunk vacío al final

@app.post("/pregunta-stream/")
async def responder_pregunta_stream(request: QuestionRequest):
    query = request.pregunta
    
    # Recuperar contexto relevante del RAG
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    if not context.strip() or "no encontrado" in context.lower():
        async def no_context_stream():
            yield "No se encontró información relevante en el contexto proporcionado."
        return StreamingResponse(no_context_stream(), media_type="text/plain")
    
    return StreamingResponse(
        generate_response_stream(query, context),
        media_type="text/plain"
    )

@app.get("/status")
async def status():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"mensaje": "Bienvenido a la API de RAG con DeepSeek-R1-Distill-Llama-8B. Envía tus preguntas al endpoint /pregunta-stream/."}
    query = request.pregunta
    
    # Recuperar contexto relevante del RAG
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    if not context.strip() or "no encontrado" in context.lower():
        async def no_context_stream():
            yield "No se encontró información relevante en el contexto proporcionado."
        return StreamingResponse(no_context_stream(), media_type="text/plain")
    
    return StreamingResponse(
        generate_response_stream(query, context),
        media_type="text/plain"
    )

@app.get("/status")
async def status():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"mensaje": "Bienvenido a la API de RAG con DeepSeek-R1-Distill-Llama-8B. Envía tus preguntas al endpoint /pregunta-stream/."}