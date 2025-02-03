import os
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
                    print(Fore.YELLOW + f"Advertencia: No se pudo extraer texto de la página {page_num} en {pdf_path}." + Style.RESET_ALL)
        return text
    except Exception as e:
        print(Fore.RED + f"Error al extraer texto del PDF {pdf_path}: {e}" + Style.RESET_ALL)
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
    return [
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

        messages = [
            {"role": "system", "content": "Eres Amalia, creada por Entel. Eres una asistente útil y precisa."},
            {"role": "user", "content": f"Basándote en los siguientes documentos, responde la siguiente pregunta.\n\nDocumentos:\n{texto_documento}\n\nPregunta: {pregunta}"}
        ]

        prompt_completo = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

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
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            respuesta = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        except Exception as e:
            respuesta = f"Error al generar la respuesta: {e}"

        end_time = time.time()
        tiempo_respuesta = end_time - start_time

        respuestas.append({
            'pregunta': pregunta,
            'respuesta': respuesta,
            'tiempo': tiempo_respuesta
        })

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

def obtener_texto_de_todos_los_pdfs(directorio):
    """
    Obtiene y concatena el texto de todos los archivos PDF en un directorio.

    Args:
        directorio (str): Ruta del directorio que contiene los PDFs.

    Returns:
        str: Texto concatenado de todos los PDFs.
    """
    texto_completo = ""
    archivos_pdf = [f for f in os.listdir(directorio) if f.lower().endswith(".pdf")]

    if not archivos_pdf:
        print(Fore.RED + "No se encontraron archivos PDF en el directorio." + Style.RESET_ALL)
        return ""

    for archivo in archivos_pdf:
        ruta_pdf = os.path.join(directorio, archivo)
        print(Fore.CYAN + f"Procesando archivo: {archivo}" + Style.RESET_ALL)
        texto_extraido = extract_text_from_pdf(ruta_pdf)
        if texto_extraido:
            texto_completo += texto_extraido + "\n"

    return texto_completo

def main():
    init(autoreset=True)

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    model, tokenizer = cargar_modelo(model_name)
    if model is None or tokenizer is None:
        return

    directorio_pdfs = "./documentos/"

    print(Fore.MAGENTA + "Extrayendo texto de todos los PDFs en el directorio..." + Style.RESET_ALL)
    texto_documento = obtener_texto_de_todos_los_pdfs(directorio_pdfs)

    if not texto_documento:
        print(Fore.RED + "No se pudo extraer texto de los PDFs. Terminando el script." + Style.RESET_ALL)
        return

    preguntas = definir_preguntas()

    print(Fore.MAGENTA + "Generando respuestas a las preguntas..." + Style.RESET_ALL)
    respuestas = responder_preguntas(model, tokenizer, texto_documento, preguntas)

    print(Fore.MAGENTA + "\nMostrando todas las respuestas:" + Style.RESET_ALL)
    mostrar_respuestas(respuestas)

if __name__ == "__main__":
    main()
