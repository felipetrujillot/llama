from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from colorama import init, Fore, Style
import PyPDF2
from accelerate import infer_auto_device_map, dispatch_model, cpu_offload_with_hook
import gc

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

def cargar_modelo(model_name):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            offload_folder="offload"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device_map = infer_auto_device_map(
            model,
            max_memory={0: "50GiB", "cpu": "50GiB"},
            no_split_module_classes=["DecoderLayer"]
        )
        model = dispatch_model(model, device_map=device_map)
        model, hook = cpu_offload_with_hook(model, torch.device("cuda"))
        return model, tokenizer, hook
    except Exception as e:
        print(Fore.RED + f"Error al cargar el modelo '{model_name}': {e}" + Style.RESET_ALL)
        return None, None, None

def definir_preguntas():
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
        {'pregunta': '¿Se exigen certificaciones?'},
        {'pregunta': '¿Hay gente en modalidad remota, teletrabajo?'},
        {'pregunta': '¿Se permite subcontratar?'},
        {'pregunta': '¿Cuál es el formato de pago?'},
        {'pregunta': '¿Cómo se entrega la propuesta y condiciones?'},
    ]

def responder_preguntas(model, tokenizer, hook, texto_documento, preguntas):
    respuestas = []
    for idx, item in enumerate(preguntas, start=1):
        gc.collect()
        torch.cuda.empty_cache()
        
        print(Fore.CYAN + "\nUso de memoria antes de la inferencia:" + Style.RESET_ALL)
        print(torch.cuda.memory_summary(device=None, abbreviated=False))
        
        pregunta = item['pregunta']
        messages = [
            {"role": "system", "content": "You are a helpful and knowledgeable assistant. Think step-by-step."},
            {"role": "user", "content": f"Based on the following document, answer this question:\n\nDocument:\n{texto_documento}\n\nQuestion: {pregunta}"}
        ]
        prompt_completo = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([prompt_completo], return_tensors="pt").to(model.device)
        start_time = time.time()
        try:
            with torch.no_grad():  # Evita almacenar gradientes innecesarios
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=50,  # Reducido para ahorrar memoria
                    temperature=0.2,
                    top_p=0.85,
                    top_k=20
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                respuesta = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        except Exception as e:
            respuesta = f"Error al generar la respuesta: {e}"
        
        del model_inputs, generated_ids  # Libera memoria
        gc.collect()
        torch.cuda.empty_cache()
        hook.offload()  # Mueve el modelo de vuelta a la CPU
        
        end_time = time.time()
        respuestas.append({
            'pregunta': pregunta,
            'respuesta': respuesta,
            'tiempo': end_time - start_time
        })
        
        print(Fore.CYAN + "\nUso de memoria después de la inferencia:" + Style.RESET_ALL)
        print(torch.cuda.memory_summary(device=None, abbreviated=False))
        
        print(Fore.GREEN + f"Pregunta {idx}/{len(preguntas)} procesada." + Style.RESET_ALL)
    return respuestas

def main():
    init(autoreset=True)
    model_name = "Qwen/QwQ-32B-Preview"
    model, tokenizer, hook = cargar_modelo(model_name)
    if model is None or tokenizer is None:
        return
    pdf_path = "./documentos/amsa.pdf"
    print(Fore.MAGENTA + "Extrayendo texto del PDF..." + Style.RESET_ALL)
    texto_documento = extract_text_from_pdf(pdf_path)
    if not texto_documento:
        print(Fore.RED + "No se pudo extraer texto del PDF. Terminando el script." + Style.RESET_ALL)
        return
    preguntas = definir_preguntas()
    print(Fore.MAGENTA + "Generando respuestas a las preguntas..." + Style.RESET_ALL)
    respuestas = responder_preguntas(model, tokenizer, hook, texto_documento, preguntas)
    print(Fore.MAGENTA + "\nMostrando todas las respuestas:" + Style.RESET_ALL)
    mostrar_respuestas(respuestas)

if __name__ == "__main__":
    main()
