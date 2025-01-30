from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from colorama import init, Fore, Style
import PyPDF2

def extract_text_from_pdf(pdf_path):
    """
    Extrae el texto de un archivo PDF.

    Args:
        pdf_path (str): Ruta al archivo PDF.

    Returns:
        str: Texto extraído del PDF.
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    print(Fore.YELLOW + f"Advertencia: No se pudo extraer texto de la página {page_num}." + Style.RESET_ALL)
        return text
    except FileNotFoundError:
        print(Fore.RED + f"Error: El archivo PDF en la ruta '{pdf_path}' no se encontró." + Style.RESET_ALL)
        return ""
    except Exception as e:
        print(Fore.RED + f"Error al extraer texto del PDF: {e}" + Style.RESET_ALL)
        return ""

def cargar_modelo(model_name):
    """
    Carga el modelo y el tokenizador de Hugging Face.

    Args:
        model_name (str): Nombre del modelo en Hugging Face.

    Returns:
        tuple: Modelo y tokenizador cargados.
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        print(Fore.RED + f"Error al cargar el modelo '{model_name}': {e}" + Style.RESET_ALL)
        return None, None

def definir_preguntas():
    """
    Define la lista de preguntas a responder.

    Returns:
        list: Lista de diccionarios con preguntas.
    """
    preguntas = [
        {
            'pregunta': 'Hazme un muy breve resumen del documento',
        },
        {
            'pregunta': '¿Cuál es el plazo de implementación?',
        },
        {
            'pregunta': '¿Hay boleta de garantía?',
        },
        {
            'pregunta': '¿Cuándo es la fecha del periodo de preguntas?',
        },
        {
            'pregunta': '¿Cuándo es la fecha de entrega de propuesta?',
        },
        {
            'pregunta': '¿Cuándo es la fecha de respuesta de la propuesta?',
        },
        {
            'pregunta': '¿Cuándo es la fecha de firma del contrato?',
        },
        {
            'pregunta': '¿Cuáles son los límites legales de responsabilidad?',
        },
        {
            'pregunta': '¿Hay multas por incumplimiento?',
        },
        {
            'pregunta': '¿Hay marcas asociadas del RFP?',
        },
        {
            'pregunta': '¿Se exigen certificaciones?',
        },
        {
            'pregunta': '¿Hay gente en modalidad remota, teletrabajo?',
        },
        {
            'pregunta': '¿Se permite subcontratar?',
        },
        {
            'pregunta': '¿Cuál es el formato de pago?',
        },
        {
            'pregunta': '¿Cómo se entrega la propuesta y condiciones?',
        },
        {
            'pregunta': '¿Se aceptan condiciones comerciales?',
        },
    ]
    return preguntas

def responder_preguntas(model, tokenizer, texto_documento, preguntas):
    """
    Responde a una lista de preguntas basadas en el texto del documento.

    Args:
        model: Modelo de lenguaje.
        tokenizer: Tokenizador del modelo.
        texto_documento (str): Texto extraído del PDF.
        preguntas (list): Lista de diccionarios con preguntas.

    Returns:
        list: Lista de diccionarios con respuestas y tiempos.
    """
    respuestas = []
    for idx, item in enumerate(preguntas, start=1):
        pregunta = item['pregunta']
        
        # Crear la entrada del sistema y el usuario
        messages = [
            {"role": "system", "content": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                "You are an exceptionally advanced AI assistant, equipped with state-of-the-art capabilities to understand and analyze technical documents. "
                "Your role is to deliver responses that are not only accurate and insightful but also enriched with a deep understanding of the context provided by the PDFs.\n\n"
                "**Instructions:**\n"
                "- Thoroughly analyze the provided context and input.\n"
                "- Extract and synthesize key information from the PDFs to provide a comprehensive and informed response.\n"
                "- Enhance your responses with detailed explanations, advanced insights, and contextually relevant examples.\n"
                "- Present information in a structured format using Markdown where applicable, but prioritize clarity and depth of content over formatting.\n"
                "- Address the query with a high level of detail and sophistication, demonstrating a deep understanding of the subject matter.\n"
                "- If any critical information is missing or if further context is needed, clearly indicate this in your response.\n\n"
                "**Response Guidelines:**\n"
                "- **Introduction:** Begin with a brief overview of the topic, setting the stage for a detailed analysis.\n"
                "- **Detailed Analysis:** Provide an in-depth examination of the topic, incorporating insights derived from the PDFs.\n"
                "- **Contextual Insights:** Relate the information to the context provided by the PDFs, making connections and highlighting relevant points.\n"
                "- **Examples and Explanations:** Include specific examples, detailed explanations, and any relevant data or findings from the PDFs.\n"
                "- **Conclusion:** Summarize the key points and provide a well-rounded conclusion based on the analysis.\n\n"
                "**Markdown Formatting Guide:**\n"
                "- Headers: Use `#` for main headings, `##` for subheadings, and `###` for detailed subheadings.\n"
                "- Bold Text: Use `**text**` to highlight important terms or concepts.\n"
                "- Italic Text: Use `*text*` for emphasis.\n"
                "- Bulleted Lists: Use `-` or `*` for unordered lists where necessary.\n"
                "- Numbered Lists: Use `1.`, `2.` for ordered lists when appropriate.\n"
                "- Links: Include `[link text](URL)` to provide additional resources or references.\n"
                "- Code Blocks: Use triple backticks (` ``` `) for code snippets.\n"
                "- Tables: Use `|` to organize data into tables for clarity.\n\n"
                "Context: "+texto_documento+"\n"
                "<|eot_id|><|start_header_id|>user<|end_header_id|>"},
            
            {"role": "user", "content": "User: " + pregunta + "\n<|start_header_id|>assistant<|end_header_id|>"}
        ]

        
        # Aplicar la plantilla de chat
        prompt_completo = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenizar el prompt
        model_inputs = tokenizer([prompt_completo], return_tensors="pt").to(model.device)
        
        # Medir el tiempo de respuesta
        start_time = time.time()
        
        try:
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=256,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.2,  # Ajusta la creatividad de la respuesta
                top_p=0.95,
                top_k=50
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            respuesta = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        except Exception as e:
            respuesta = f"Error al generar la respuesta: {e}"
        
        end_time = time.time()
        tiempo_respuesta = end_time - start_time
        
        # Almacenar la respuesta
        respuestas.append({
            'pregunta': pregunta,
            'respuesta': respuesta,
            'tiempo': tiempo_respuesta
        })
        
        # Mostrar progreso
        print(Fore.GREEN + f"Pregunta {idx}/{len(preguntas)} procesada." + Style.RESET_ALL)
    
    return respuestas

def mostrar_respuestas(respuestas):
    """
    Muestra las respuestas junto con los tiempos de respuesta.

    Args:
        respuestas (list): Lista de diccionarios con respuestas y tiempos.
    """
    for idx, item in enumerate(respuestas, start=1):
        print(Fore.BLUE + f"\n=== Pregunta {idx} ===" + Style.RESET_ALL)
        print(Fore.YELLOW + f"Pregunta: {item['pregunta']}" + Style.RESET_ALL)
        print(f"Respuesta: {item['respuesta']}")
        print(Fore.CYAN + f"Tiempo de respuesta: {item['tiempo']:.2f} segundos" + Style.RESET_ALL)
        print("-" * 50)

def main():
    # Inicializa colorama
    init(autoreset=True)

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Asegúrate de que este modelo está disponible

    # Carga el modelo y el tokenizador
    model, tokenizer = cargar_modelo(model_name)
    if model is None or tokenizer is None:
        return

    # Ruta al documento PDF
    pdf_path = "./documentos/amsa.pdf"  # Reemplaza con la ruta de tu PDF

    # Extrae el texto del PDF
    print(Fore.MAGENTA + "Extrayendo texto del PDF..." + Style.RESET_ALL)
    texto_documento = extract_text_from_pdf(pdf_path)
    if not texto_documento:
        print(Fore.RED + "No se pudo extraer texto del PDF. Terminando el script." + Style.RESET_ALL)
        return

    # Define las preguntas
    preguntas = definir_preguntas()

    # Responde a las preguntas
    print(Fore.MAGENTA + "Generando respuestas a las preguntas..." + Style.RESET_ALL)
    respuestas = responder_preguntas(model, tokenizer, texto_documento, preguntas)

    # Muestra las respuestas
    print(Fore.MAGENTA + "\nMostrando todas las respuestas:" + Style.RESET_ALL)
    mostrar_respuestas(respuestas)

if __name__ == "__main__":
    main()
