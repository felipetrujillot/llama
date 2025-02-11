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
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Modelo compatible con dimensión 384
vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)

# Modelo de entrada para la API
class QuestionRequest(BaseModel):
    pregunta: str

# Función para generar tokens uno por uno
async def generate_response_stream(prompt, context):
    # Nuevo prompt basado en el que proporcionaste
    messages = [
        {"role": "system", "content": r"""
            You are an exceptionally advanced AI assistant, equipped with state-of-the-art capabilities to understand and analyze technical documents. Your role is to deliver responses that are not only accurate and insightful but also enriched with a deep understanding of the context provided by the PDFs.

            **Instructions:**
            - Thoroughly analyze the provided context and input.
            - Extract and synthesize key information from the PDFs to provide a comprehensive and informed response.
            - Enhance your responses with detailed explanations, advanced insights, and contextually relevant examples.
            - Present information in a structured format using Markdown where applicable, but prioritize clarity and depth of content over formatting.
            - Address the query with a high level of detail and sophistication, demonstrating a deep understanding of the subject matter.
            - If any critical information is missing or if further context is needed, clearly indicate this in your response.

            **Response Guidelines:**
            - **Introduction:** Begin with a brief overview of the topic, setting the stage for a detailed analysis.
            - **Detailed Analysis:** Provide an in-depth examination of the topic, incorporating insights derived from the PDFs.
            - **Contextual Insights:** Relate the information to the context provided by the PDFs, making connections and highlighting relevant points.
            - **Examples and Explanations:** Include specific examples, detailed explanations, and any relevant data or findings from the PDFs.
            - **Conclusion:** Summarize the key points and provide a well-rounded conclusion based on the analysis.

            **Markdown Formatting Guide:**
            - Headers: Use `#` for main headings, `##` for subheadings, and `###` for detailed subheadings.
            - Bold Text: Use `**text**` to highlight important terms or concepts.
            - Italic Text: Use `*text*` for emphasis.
            - Bulleted Lists: Use `-` or `*` for unordered lists where necessary.
            - Numbered Lists: Use `1.`, `2.` for ordered lists when appropriate.
            - Links: Include `[link text](URL)` to provide additional resources or references.
            - Code Blocks: Use triple backticks (\`\`\`) for code snippets.
            - **Tables:** Use `|` to organize data into tables for clarity. The first row should be a header, followed by a separator row (`---`). Ensure that all columns are properly aligned.

            **Example Table Syntax in Markdown:**
            ```markdown
            | Column 1   | Column 2   | Column 3   |
            |------------|------------|------------|
            | Data 1     | Data 2     | Data 3     |
            | Data 4     | Data 5     | Data 6     |

            """ + context},
        {"role": "user", "content": f"User: {prompt}"},
        {"role": "assistant", "content": ""}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, decode_kwargs={"skip_special_tokens": True})
    
    thread = threading.Thread(
        target=model.generate,
        kwargs={
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "max_new_tokens": 1024,
            "temperature": 0.5,
            "do_sample": True,
            "streamer": streamer
        }
    )
    thread.start()
    
    try:
        for token in streamer:
            yield token  # Enviar el token al cliente
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