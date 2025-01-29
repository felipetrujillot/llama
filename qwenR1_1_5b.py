import time
import torch
from colorama import init, Fore, Style
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    IMPORTANTE: trust_remote_code=True para que se use la lógica interna de Qwen.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        return model, tokenizer
    except Exception as e:
        print(Fore.RED + f"Error al cargar el modelo '{model_name}': {e}" + Style.RESET_ALL)
        return None, None

def definir_preguntas():
    """
    Define la lista de preguntas a responder.
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

def responder_preguntas(model, tokenizer, texto_documento, preguntas):
    """
    Responde a una lista de preguntas basadas en el texto del documento.
    Genera un prompt manual, sin plantillas de chat específicas de Qwen.
    """
    respuestas = []
    for idx, item in enumerate(preguntas, start=1):
        pregunta = item['pregunta']
        
        # Construimos un prompt manual
        prompt = (
            "Instrucciones: Eres Amalia, creada por Entel. "
            "Basado en el siguiente documento, responde de manera breve y precisa.\n\n"
            f"Documento:\n{texto_documento}\n\n"
            f"Pregunta: {pregunta}\n\n"
            "Respuesta:"
        )

        # Convertimos el prompt en tensores
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Medir el tiempo de inferencia
        start_time = time.time()

        try:
            # Generamos la respuesta
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,           # Modo sampling
                temperature=0.6,          # Ajusta según prefieras
                top_p=0.9,                # Ajusta según prefieras
                repetition_penalty=1.15,  # Puede ayudar a evitar bucles repetitivos
            )
            # Decodificamos la respuesta
            respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            respuesta = f"Error al generar la respuesta: {e}"

        end_time = time.time()
        tiempo_respuesta = end_time - start_time
        
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

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    pdf_path = "./documentos/amsa.pdf"  # Ajusta la ruta a tu PDF

    # Carga el modelo y el tokenizador (con trust_remote_code)
    model, tokenizer = cargar_modelo(model_name)
    if model is None or tokenizer is None:
        return

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
