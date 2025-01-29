import transformers
import torch
import time
from colorama import init, Fore, Style
import PyPDF2
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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
    Carga el modelo y el tokenizador de Hugging Face con prioridad en GPU
    y offloading mínimo a CPU, utilizando device_map="auto" y max_memory.

    Args:
        model_name (str): Nombre del modelo en Hugging Face.

    Returns:
        pipeline: Pipeline de generación de texto usando dicho modelo.
    """
    try:
        print(Fore.CYAN + f"Cargando modelo {model_name} con prioridad a GPU..." + Style.RESET_ALL)

        # Carga el tokenizador
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Carga del modelo con device_map="auto" y prioridad a GPU.
        # Ajusta los valores de 'max_memory' según tu hardware:
        max_memory = {
            0: "24GiB",    # Asigna hasta 24 GiB de la GPU 0
            "cpu": "8GiB"  # Permite offloading parcial de hasta 8 GiB en CPU
        }

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            max_memory=max_memory,
        )

        # Construimos el pipeline con el modelo y el tokenizador
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

        return pipe
    except Exception as e:
        print(Fore.RED + f"Error al cargar el modelo '{model_name}': {e}" + Style.RESET_ALL)
        return None

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

def responder_preguntas(pipeline_llama, texto_documento, preguntas):
    """
    Responde a una lista de preguntas basadas en el texto del documento utilizando el pipeline.

    Args:
        pipeline_llama: Pipeline de generación de texto (modelo) cargado.
        texto_documento (str): Texto extraído del PDF.
        preguntas (list): Lista de diccionarios con preguntas.

    Returns:
        list: Lista de diccionarios con respuestas y tiempos.
    """
    respuestas = []
    total = len(preguntas)

    for idx, item in enumerate(preguntas, start=1):
        pregunta = item['pregunta']

        # Mensajes en formato chat (system + user)
        messages = [
            {
                "role": "system",
                "content": "Eres Amalia, creada por Entel. Eres una asistente útil y precisa."
            },
            {
                "role": "user",
                "content": (
                    f"Basándote en el siguiente documento, responde a la siguiente pregunta.\n\n"
                    f"Documento:\n{texto_documento}\n\n"
                    f"Pregunta: {pregunta}"
                )
            }
        ]

        start_time = time.time()

        try:
            # Genera la respuesta con el pipeline
            output = pipeline_llama(
                messages,
                max_new_tokens=256,
                temperature=0.2,   # Ajusta la creatividad de la respuesta
                top_p=0.95,
                top_k=50
            )

            # El pipeline retorna una lista de outputs; tomamos la primera salida
            respuesta_completa = output[0]["generated_text"]

            # Ajusta el postprocesado según sea necesario
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
    # Inicializa colorama
    init(autoreset=True)

    # Nombre del modelo (asegúrate de que el modelo existe en Hugging Face)
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Carga el pipeline con el modelo, priorizando GPU y offloading mínimo a CPU
    pipeline_llama = cargar_modelo(model_name)
    if pipeline_llama is None:
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
    respuestas = responder_preguntas(pipeline_llama, texto_documento, preguntas)

    # Muestra las respuestas
    print(Fore.MAGENTA + "\nMostrando todas las respuestas:" + Style.RESET_ALL)
    mostrar_respuestas(respuestas)

if __name__ == "__main__":
    main()
