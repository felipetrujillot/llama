import torch
from transformers import pipeline
import PyPDF2
import time
from colorama import init, Fore, Style
import pyperclip  # Importar pyperclip para copiar al portapapeles

def extract_text_from_pdf(pdf_path):
    """
    Extrae el texto de un archivo PDF.
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
        {'pregunta': '¿Se aceptan condiciones comerciales?'}
    ]

def configurar_pipeline(model_id):
    """
    Configura el pipeline de generación de texto.
    """
    try:
        return pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
    except Exception as e:
        print(Fore.RED + f"Error al configurar el pipeline con el modelo '{model_id}': {e}" + Style.RESET_ALL)
        return None

def responder_preguntas(pipe, texto_documento, preguntas):
    """
    Responde a una lista de preguntas basadas en el texto del documento.
    """
    respuestas = []
    for idx, item in enumerate(preguntas, start=1):
        pregunta = item['pregunta']
        prompt = (
            "Eres Amalia, creada por Entel. Eres una asistente útil y precisa.\n\n"
            f"Documento:\n{texto_documento}\n\n"
            f"Pregunta: {pregunta}\nRespuesta:"
        )
        start_time = time.time()
        try:
            outputs = pipe(
                prompt,
                max_new_tokens=256,
                temperature=0.2,
                top_p=0.95,
                top_k=50,
                eos_token_id=pipe.tokenizer.eos_token_id if hasattr(pipe.tokenizer, 'eos_token_id') else None,
                pad_token_id=pipe.tokenizer.eos_token_id if hasattr(pipe.tokenizer, 'eos_token_id') else None,
            )
            respuesta_completa = outputs[0]['generated_text']
            respuesta = respuesta_completa[len(prompt):].strip()
        except Exception as e:
            respuesta = f"Error al generar la respuesta: {e}"
        tiempo_respuesta = time.time() - start_time
        respuestas.append({'pregunta': pregunta, 'respuesta': respuesta, 'tiempo': tiempo_respuesta})
        print(Fore.GREEN + f"Pregunta {idx}/{len(preguntas)} procesada." + Style.RESET_ALL)
    return respuestas

def mostrar_respuestas(respuestas):
    """
    Muestra las respuestas y las copia al portapapeles o las guarda en un archivo si falla.
    """
    texto_completo = ""
    for idx, item in enumerate(respuestas, start=1):
        texto_completo += f"\n=== Pregunta {idx} ===\n"
        texto_completo += f"Pregunta: {item['pregunta']}\n"
        texto_completo += f"Respuesta: {item['respuesta']}\n"
        texto_completo += f"Tiempo de respuesta: {item['tiempo']:.2f} segundos\n"
        texto_completo += "-" * 50 + "\n"
    try:
        pyperclip.copy(texto_completo)
        print(Fore.MAGENTA + "\nLas respuestas han sido copiadas al portapapeles." + Style.RESET_ALL)
    except pyperclip.PyperclipException:
        with open("respuestas.txt", "w", encoding="utf-8") as f:
            f.write(texto_completo)
        print(Fore.MAGENTA + "Las respuestas han sido guardadas en 'respuestas.txt' porque no se pudo copiar al portapapeles." + Style.RESET_ALL)

def main():
    init(autoreset=True)
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    print(Fore.MAGENTA + "Configurando el pipeline de generación de texto..." + Style.RESET_ALL)
    pipe = configurar_pipeline(model_id)
    if pipe is None:
        print(Fore.RED + "No se pudo configurar el pipeline. Terminando el script." + Style.RESET_ALL)
        return
    pdf_path = "./documentos/amsa.pdf"
    print(Fore.MAGENTA + "Extrayendo texto del PDF..." + Style.RESET_ALL)
    texto_documento = extract_text_from_pdf(pdf_path)
    if not texto_documento:
        print(Fore.RED + "No se pudo extraer texto del PDF. Terminando el script." + Style.RESET_ALL)
        return
    preguntas = definir_preguntas()
    print(Fore.MAGENTA + "Generando respuestas a las preguntas..." + Style.RESET_ALL)
    respuestas = responder_preguntas(pipe, texto_documento, preguntas)
    print(Fore.MAGENTA + "\nMostrando todas las respuestas:" + Style.RESET_ALL)
    mostrar_respuestas(respuestas)

if __name__ == "__main__":
    main()

