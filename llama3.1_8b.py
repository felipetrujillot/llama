import torch 
from transformers import pipeline
import PyPDF2
import time
from colorama import init, Fore, Style

def extract_text_from_pdf(pdf_path):
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
    return [
        {'pregunta': 'Hazme un muy breve resumen del documento'},
        # {'pregunta': '¿Cuál es el plazo de implementación?'},
        # {'pregunta': '¿Hay boleta de garantía?'},
        # {'pregunta': '¿Cuándo es la fecha del periodo de preguntas?'},
        # {'pregunta': '¿Cuándo es la fecha de entrega de propuesta?'},
        # {'pregunta': '¿Cuándo es la fecha de respuesta de la propuesta?'},
        # {'pregunta': '¿Cuándo es la fecha de firma del contrato?'},
        # {'pregunta': '¿Cuáles son los límites legales de responsabilidad?'},
        # {'pregunta': '¿Hay multas por incumplimiento?'},
        # {'pregunta': '¿Se exigen certificaciones?'},
        # {'pregunta': '¿Se permite subcontratar?'},
        # {'pregunta': '¿Cuál es el formato de pago?'},
    ]

def configurar_pipeline(model_id):
    try:
        pipe = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        return pipe
    except Exception as e:
        print(Fore.RED + f"Error al configurar el pipeline con el modelo '{model_id}': {e}" + Style.RESET_ALL)
        return None

def responder_preguntas(pipe, texto_documento, preguntas):
    respuestas = []
    for idx, item in enumerate(preguntas, start=1):
        pregunta = item['pregunta']
        
        messages = [
            {"role": "system", "content": "Eres Amalia, creada por Entel. Eres una asistente útil y precisa."},
            {"role": "user", "content": f"Basándote en el siguiente documento, responde a la siguiente pregunta.\n\nDocumento:\n{texto_documento}\n\nPregunta: {pregunta}"}
        ]
        
        start_time = time.time()
        try:
            outputs = pipe(
                messages,
                max_new_tokens=256,
            )
            
            if isinstance(outputs, list) and len(outputs) > 0 and 'generated_text' in outputs[0]:
                respuesta = outputs[0]['generated_text'].strip()
            else:
                respuesta = "Error: No se pudo generar una respuesta válida."
        except Exception as e:
            respuesta = f"Error al generar la respuesta: {e}"

        end_time = time.time()
        tiempo_respuesta = end_time - start_time
        
        respuestas.append({'pregunta': pregunta, 'respuesta': respuesta, 'tiempo': tiempo_respuesta})
        print(Fore.GREEN + f"Pregunta {idx}/{len(preguntas)} procesada." + Style.RESET_ALL)
    
    return respuestas

def mostrar_respuestas(respuestas):
    for idx, item in enumerate(respuestas, start=1):
        print(Fore.BLUE + f"\n=== Pregunta {idx} ===" + Style.RESET_ALL)
        print(Fore.YELLOW + f"Pregunta: {item['pregunta']}" + Style.RESET_ALL)
        print(f"Respuesta: {item['respuesta']}")
        print(Fore.CYAN + f"Tiempo de respuesta: {item['tiempo']:.2f} segundos" + Style.RESET_ALL)
        print("-" * 50)

def main():
    init(autoreset=True)
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    print(Fore.MAGENTA + "Configurando el pipeline de generación de texto..." + Style.RESET_ALL)
    pipe = configurar_pipeline(model_id)
    if pipe is None:
        return
    
    pdf_path = "./documentos/amsa.pdf"
    print(Fore.MAGENTA + "Extrayendo texto del PDF..." + Style.RESET_ALL)
    texto_documento = extract_text_from_pdf(pdf_path)
    if not texto_documento:
        return
    
    preguntas = definir_preguntas()
    print(Fore.MAGENTA + "Generando respuestas a las preguntas..." + Style.RESET_ALL)
    respuestas = responder_preguntas(pipe, texto_documento, preguntas)
    
    print(Fore.MAGENTA + "\nMostrando todas las respuestas:" + Style.RESET_ALL)
    mostrar_respuestas(respuestas)

if __name__ == "__main__":
    main()
