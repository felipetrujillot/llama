import torch
from transformers import pipeline
import PyPDF2
import time
from colorama import init, Fore, Style

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

def definir_preguntas():
    """
    Define la lista de preguntas a responder.

    Returns:
        list: Lista de diccionarios con preguntas.
    """
    preguntas = [
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
    return preguntas

def configurar_pipeline(model_id):
    """
    Configura el pipeline de generación de texto.

    Args:
        model_id (str): Identificador del modelo en Hugging Face.

    Returns:
        pipeline: Pipeline de generación de texto.
    """
    try:
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        return pipe
    except Exception as e:
        print(Fore.RED + f"Error al configurar el pipeline con el modelo '{model_id}': {e}" + Style.RESET_ALL)
        return None

def responder_preguntas(pipe, texto_documento, preguntas):
    """
    Responde a una lista de preguntas basadas en el texto del documento.

    Args:
        pipe (pipeline): Pipeline de generación de texto.
        texto_documento (str): Texto extraído del PDF.
        preguntas (list): Lista de diccionarios con preguntas.

    Returns:
        list: Lista de diccionarios con respuestas y tiempos.
    """
    respuestas = []
    for idx, item in enumerate(preguntas, start=1):
        pregunta = item['pregunta']

        # Crear el prompt combinando el mensaje del sistema y la pregunta
        prompt = (
            "Eres Amalia, creada por Entel. Eres una asistente útil y precisa.\n\n"
            f"Documento:\n{texto_documento}\n\n"
            f"Pregunta: {pregunta}\nRespuesta:"
        )

        # Medir el tiempo de respuesta
        start_time = time.time()

        try:
            # Generar la respuesta
            outputs = pipe(
                prompt,
                max_new_tokens=256,
                temperature=0.2,  # Baja temperatura para respuestas más determinísticas
                top_p=0.95,
                top_k=50,
                eos_token_id=pipe.tokenizer.eos_token_id if hasattr(pipe.tokenizer, 'eos_token_id') else None,
                pad_token_id=pipe.tokenizer.eos_token_id if hasattr(pipe.tokenizer, 'eos_token_id') else None,
            )
            # Obtener el texto generado y eliminar el prompt
            respuesta_completa = outputs[0]['generated_text']
            respuesta = respuesta_completa[len(prompt):].strip()
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

    # Identificador del modelo en Hugging Face
    model_id = "meta-llama/Llama-3.2-3B-Instruct"  # Reemplaza con el modelo correcto si es necesario

    # Configura el pipeline
    print(Fore.MAGENTA + "Configurando el pipeline de generación de texto..." + Style.RESET_ALL)
    pipe = configurar_pipeline(model_id)
    if pipe is None:
        print(Fore.RED + "No se pudo configurar el pipeline. Terminando el script." + Style.RESET_ALL)
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
    respuestas = responder_preguntas(pipe, texto_documento, preguntas)

    # Muestra las respuestas
    print(Fore.MAGENTA + "\nMostrando todas las respuestas:" + Style.RESET_ALL)
    mostrar_respuestas(respuestas)

if __name__ == "__main__":
    main()