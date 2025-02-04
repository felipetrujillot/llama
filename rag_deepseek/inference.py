import time
from colorama import Fore, Style, init
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Inicializar Colorama
init(autoreset=True)

# Configuración inicial
CHROMA_DB_PATH = "./chroma_db"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"  # Modelo de razonamiento

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

# Cargar modelo DeepSeek-R1-Distill-Qwen-14B
print("Cargando modelo DeepSeek-R1-Distill-Qwen-14B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

# Cargar ChromaDB con el modelo de embeddings
print("Cargando base de datos ChromaDB...")
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)

def generate_response(prompt, context):
    # Crear el mensaje con el contexto explícito
    messages = [
        {"role": "system", "content": """
You are an exceptionally advanced AI assistant, equipped with state-of-the-art capabilities to understand and analyze technical documents. Your role is to deliver responses that are not only accurate and insightful but also enriched with a deep understanding of the context provided by the PDFs.
**Important Notes:**
- The context or documents provided are referred to as an "RFP" (Request for Proposal). Understand that all references to the context or documents imply they are part of an RFP.
- You must ALWAYS respond in SPANISH, regardless of the language of the instructions or prompts. Ensure your responses are clear, detailed, and exclusively in Spanish.
**Instructions:**
- Thoroughly analyze the provided context and input.
- Extract and synthesize key information from the RFP to provide a comprehensive and informed response.
- Enhance your responses with detailed explanations, advanced insights, and contextually relevant examples.
- Present information in a structured format using Markdown where applicable, but prioritize clarity and depth of content over formatting.
- Address the query with a high level of detail and sophistication, demonstrating a deep understanding of the subject matter.
- If any critical information is missing or if further context is needed, clearly indicate this in your response.
**Response Guidelines:**
- **Introduction:** Begin with a brief overview of the topic, setting the stage for a detailed analysis.
- **Detailed Analysis:** Provide an in-depth examination of the topic, incorporating insights derived from the RFP.
- **Contextual Insights:** Relate the information to the context provided by the RFP, making connections and highlighting relevant points.
- **Examples and Explanations:** Include specific examples, detailed explanations, and any relevant data or findings from the RFP.
- **Conclusion:** Summarize the key points and provide a well-rounded conclusion based on the analysis.
---
Now, answer the following question based on the provided context:
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
        max_new_tokens=1024,
        temperature=0.7,  # Reducir la creatividad para respuestas más precisas
        do_sample=True
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

        # Verificar si el contexto es relevante
        if not context.strip() or "no encontrado" in context.lower():
            response = "No se encontró información relevante en el contexto proporcionado."
        else:
            # Generar respuesta con DeepSeek-R1-Distill-Qwen-14B
            response = generate_response(query, context)

        elapsed_time = time.time() - start_time

        # Mostrar respuesta y tiempo en colores
        print(Fore.GREEN + f"Respuesta: {response}")
        print(Fore.YELLOW + f"Tiempo de respuesta: {elapsed_time:.2f} segundos" + Style.RESET_ALL)

if __name__ == "__main__":
    main()