from transformers import AutoModelForCausalLM, AutoTokenizer 
import torch
import time
from colorama import init, Fore, Style
import PyPDF2

def extract_text_from_pdf(pdf_path):
    """
    Extrae el texto de un archivo PDF.
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])
        return text
    except Exception as e:
        print(Fore.RED + f"Error al extraer texto del PDF: {e}" + Style.RESET_ALL)
        return ""

def new_prompt(context: str, question: str) -> str:
    """
    Genera un prompt formateado para un modelo de lenguaje.
    """
    prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an exceptionally advanced AI assistant, equipped with state-of-the-art capabilities to understand and analyze technical documents. Your role is to deliver responses that are not only accurate and insightful but also enriched with a deep understanding of the context provided by the PDFs.

**Instructions:**
- Thoroughly analyze the provided context and input.
- Extract and synthesize key information from the PDFs to provide a comprehensive and informed response.
- Enhance your responses with detailed explanations, advanced insights, and contextually relevant examples.
- Present information in a structured format using Markdown where applicable, but prioritize clarity and depth of content over formatting.
- Address the query with a high level of detail and sophistication, demonstrating a deep understanding of the subject matter.
- If any critical information is missing or if further context is needed, clearly indicate this in your response.

**Response Guidelines:**
- **Introduction:** Brief overview of the topic.
- **Detailed Analysis:** In-depth examination incorporating insights from PDFs.
- **Contextual Insights:** Connections and highlights from PDFs.
- **Examples and Explanations:** Specific examples and findings.
- **Conclusion:** Well-rounded summary.

Context: {context}
<|eot_id|><|start_header_id|>user<|end_header_id|>

User: {question}
<|start_header_id|>assistant<|end_header_id|>
"""
    return prompt

def cargar_modelo(model_name):
    """
    Carga el modelo y el tokenizador de Hugging Face.
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        print(Fore.RED + f"Error al cargar el modelo '{model_name}': {e}" + Style.RESET_ALL)
        return None, None

def responder_pregunta(model, tokenizer, texto_documento, pregunta):
    """
    Genera una respuesta a una pregunta basada en un documento.
    """
    prompt_completo = new_prompt(texto_documento, pregunta)
    model_inputs = tokenizer([prompt_completo], return_tensors="pt").to(model.device)
    
    start_time = time.time()
    try:
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.2,
            top_p=0.95,
            top_k=50
        )
        respuesta = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    except Exception as e:
        respuesta = f"Error al generar la respuesta: {e}"
    
    tiempo_respuesta = time.time() - start_time
    return pregunta, respuesta, tiempo_respuesta

def main():
    init(autoreset=True)
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model, tokenizer = cargar_modelo(model_name)
    if not model or not tokenizer:
        return
    
    pdf_path = "./documentos/amsa.pdf"
    print(Fore.MAGENTA + "Extrayendo texto del PDF..." + Style.RESET_ALL)
    texto_documento = extract_text_from_pdf(pdf_path)
    if not texto_documento:
        print(Fore.RED + "No se pudo extraer texto del PDF." + Style.RESET_ALL)
        return
    
    preguntas = [
        "Hazme un muy breve resumen del documento",
        "¿Cuál es el plazo de implementación?",
        "¿Hay boleta de garantía?",
        "¿Cuándo es la fecha del periodo de preguntas?",
        "¿Cuándo es la fecha de entrega de propuesta?",
        "¿Cuándo es la fecha de respuesta de la propuesta?",
        "¿Cuándo es la fecha de firma del contrato?",
        "¿Cuáles son los límites legales de responsabilidad?",
        "¿Hay multas por incumplimiento?",
        "¿Hay marcas asociadas del RFP?",
        "¿Se exigen certificaciones?",
        "¿Hay gente en modalidad remota, teletrabajo?",
        "¿Se permite subcontratar?",
        "¿Cuál es el formato de pago?",
        "¿Cómo se entrega la propuesta y condiciones?",
        "¿Cuándo es la fecha de entrega de propuesta?",
        "¿Se aceptan condiciones comerciales?",
    ]
    
    for idx, pregunta in enumerate(preguntas, start=1):
        print(Fore.YELLOW + f"\nPregunta {idx}: {pregunta}" + Style.RESET_ALL)
        pregunta, respuesta, tiempo = responder_pregunta(model, tokenizer, texto_documento, pregunta)
        print(Fore.GREEN + f"Respuesta: {respuesta}" + Style.RESET_ALL)
        print(Fore.CYAN + f"Tiempo de respuesta: {tiempo:.2f} segundos" + Style.RESET_ALL)
        print("-" * 50)

if __name__ == "__main__":
    main()
