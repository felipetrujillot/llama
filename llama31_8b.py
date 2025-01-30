import os
import time
import torch
import PyPDF2
from colorama import init, Fore, Style

# Librerías de Hugging Face
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM
)

# Accelerate para offloading
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

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
    Carga el modelo de manera que se permita offloading a CPU si la VRAM se llena,
    usando accelerate y sin cuantizar (float16).

    Args:
        model_name (str): Nombre del modelo en Hugging Face (e.g. meta-llama/Meta-Llama-3.1-8B-Instruct).

    Returns:
        model, tokenizer: Instancias del modelo y tokenizador listos para inferencia.
    """
    try:
        print(Fore.CYAN + f"Cargando modelo {model_name} con Accelerate (sin cuantizar, float16)..." + Style.RESET_ALL)

        # 1. Cargar la configuración del modelo (vacía)
        config = AutoConfig.from_pretrained(model_name)

        # 2. Crear un "modelo vacío"
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)

        # 3. Cargar pesos con dispatch (offload dinámico)
        # - device_map="auto": va poniendo capas en GPU y CPU según la memoria.
        # - offload_folder="./offload_cpu": carpeta para almacenar temporalmente capas en CPU.
        # - offload_state_dict=True: para liberar la memoria de GPU cuando no se usen capas
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint_folder=model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            offload_folder="./offload_cpu",
            offload_state_dict=True
        )

        # 4. Cargar el tokenizador
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Configuración extra (opcional)
        model.eval()  # Modo evaluación

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

def convertir_chat_en_prompt(messages):
    """
    Convierte una lista de mensajes con roles (system, user, assistant, etc.)
    a un string que será usado como prompt para el modelo.

    Args:
        messages (list of dict): Cada dict tiene keys 'role' y 'content'.
    
    Returns:
        str: Prompt unificado.
    """
    # Ejemplo simple: concatenamos "[ROLE]: content\n\n"
    # Puedes personalizarlo según el formateado que el modelo requiera.
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"{role.capitalize()}: {content}\n\n"
    return prompt.strip()

def responder_preguntas(model, tokenizer, texto_documento, preguntas):
    """
    Responde a una lista de preguntas basadas en el texto del documento
    usando 'model.generate' directamente, con un prompt en formato 'chat'.

    Args:
        model: Modelo de lenguaje (cargado con Accelerate).
        tokenizer: Tokenizador correspondiente.
        texto_documento (str): Texto extraído del PDF.
        preguntas (list): Lista de diccionarios con preguntas.

    Returns:
        list: Lista de diccionarios con (pregunta, respuesta, tiempo).
    """
    respuestas = []
    total = len(preguntas)

    for idx, item in enumerate(preguntas, start=1):
        pregunta = item['pregunta']

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

        # Construimos el prompt
        prompt = convertir_chat_en_prompt(messages)

        # Tokenizamos
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Medimos tiempo de generación
        start_time = time.time()

        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,  # Ajusta si deseas respuestas más largas/cortas
                    temperature=0.2,
                    top_p=0.95,
                    top_k=50,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decodificamos la respuesta generada
            # Nota: El prompt queda incluido al principio, así que es útil recortar
            # solo la parte nueva si queremos. Pero aquí, para simplificar, decodificamos todo.
            respuesta_completa = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Si quieres eliminar el prompt inicial:
            # respuesta = respuesta_completa[len(prompt):].strip()
            # Pero a veces es más fácil decodificar todo y filtrar.
            respuesta = respuesta_completa.strip()

        except Exception as e:
            respuesta = f"Error al generar la respuesta: {e}"

        end_time = time.time()
        tiempo_respuesta = end_time - start_time

        respuestas.append({
            'pregunta': pregunta,
            'respuesta': respuesta,
            'tiempo': tiempo_respuesta
        })

        # Muestra progreso
        print(Fore.GREEN + f"Pregunta {idx}/{total} procesada." + Style.RESET_ALL)

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
    # (Opcional) establecer variable de entorno desde Python (o hazlo desde terminal):
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Inicializa colorama para colores en la consola
    init(autoreset=True)

    # Nombre del modelo en Hugging Face
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # 1. Cargar el modelo y el tokenizador
    model, tokenizer = cargar_modelo(model_name)
    if model is None or tokenizer is None:
        print(Fore.RED + "No se pudo cargar el modelo. Saliendo..." + Style.RESET_ALL)
        return

    # 2. Ruta al documento PDF
    pdf_path = "./documentos/amsa.pdf"  # Ajusta con la ruta real

    # 3. Extraer el texto del PDF
    print(Fore.MAGENTA + "Extrayendo texto del PDF..." + Style.RESET_ALL)
    texto_documento = extract_text_from_pdf(pdf_path)
    if not texto_documento:
        print(Fore.RED + "No se pudo extraer texto del PDF. Terminando el script." + Style.RESET_ALL)
        return

    # 4. Definir las preguntas
    preguntas = definir_preguntas()

    # 5. Responder a las preguntas
    print(Fore.MAGENTA + "Generando respuestas a las preguntas..." + Style.RESET_ALL)
    respuestas = responder_preguntas(model, tokenizer, texto_documento, preguntas)

    # 6. Mostrar las respuestas
    print(Fore.MAGENTA + "\nMostrando todas las respuestas:" + Style.RESET_ALL)
    mostrar_respuestas(respuestas)

if __name__ == "__main__":
    main()
