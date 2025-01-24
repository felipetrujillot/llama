# llm_chat.py

import transformers
import torch
from colorama import init, Fore, Style
import logging
import time
import sys
from rag_system import RAGSystem

# Configurar el nivel de logging para transformers a ERROR
logging.getLogger("transformers").setLevel(logging.ERROR)

# Inicializar colorama
init(autoreset=True)

# Configuración del modelo y tokenizer para generación de respuestas
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Mensaje de sistema inicial
system_message = (
    "Eres Amalia, una asistente virtual inteligente creada para proporcionar información útil e "
    "insightful sobre cualquier tema. Siempre respondes en un tono amable y profesional en español. "
    "No proporciones traducciones o respuestas en ningún otro idioma. "
    "No proporciones respuestas redundantes. "
    "No proporciones respuestas incompletas. "
    "No hagas preguntas de seguimiento ni ofrezcas sugerencias adicionales después de responder."
)

# Función para obtener la respuesta del modelo en streaming
def get_response_streaming(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    generation_kwargs = {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }

    generated_ids = model.generate(
        input_ids,
        **generation_kwargs,
        return_dict_in_generate=True,
        output_scores=True,
    )

    new_tokens = generated_ids.sequences[0][input_ids.shape[-1]:]

    response = ""
    start_time = time.perf_counter()
    for token_id in new_tokens:
        char = tokenizer.decode(token_id)
        response += char
        print(char, end='', flush=True)
        time.sleep(0.02)  # Ajusta el retraso según tus preferencias
    end_time = time.perf_counter()

    time_taken = end_time - start_time
    print(Fore.YELLOW + f" (Tiempo: {time_taken:.2f} segundos)" + Style.RESET_ALL)
    return response, time_taken

def main():
    print(Fore.GREEN + "Amalia: Hola, ¿en qué puedo ayudarte hoy?" + Style.RESET_ALL)
    
    # Inicializar el sistema RAG
    rag = RAGSystem()

    # Cargar el vector store existente o procesar documentos
    vector_store_path = "faiss_index"
    if os.path.exists(vector_store_path):
        rag.load_vector_store(vector_store_path)
    else:
        documents = rag.load_documents_from_directory()
        if not documents:
            print(Fore.RED + f"No se encontraron documentos en el directorio '{rag.documentos_dir}'. Por favor, agrega documentos y reinicia el script." + Style.RESET_ALL)
            sys.exit(1)
        rag.create_vector_store(documents)
        rag.save_vector_store(vector_store_path)

    while True:
        try:
            # Obtener entrada del usuario
            user_input = input(Fore.BLUE + "User: " + Style.RESET_ALL)
            if user_input.lower() in ["salir", "exit", "quit"]:
                print(Fore.GREEN + "Amalia: ¡Hasta luego!" + Style.RESET_ALL)
                break

            if not user_input.strip():
                print(Fore.RED + "Por favor, ingresa una pregunta válida." + Style.RESET_ALL)
                continue

            # Recuperar documentos relevantes usando RAG
            docs = rag.similarity_search(user_input, k=4)
            if not docs:
                context = "Lo siento, no encontré información relevante en los documentos cargados."
            else:
                context = "\n".join([doc.page_content for doc in docs])

            # Crear prompt con contexto
            prompt = (
                f"{system_message}\n"
                f"Contexto del documento:\n{context}\n\n"
                f"Usuario: {user_input}\n"
                f"Amalia:"
            )

            # Obtener respuesta del modelo y el tiempo tomado en streaming
            print(Fore.GREEN + "Amalia: ", end='', flush=True)
            assistant_response, time_taken = get_response_streaming(prompt)

        except KeyboardInterrupt:
            print("\n" + Fore.GREEN + "Amalia: ¡Hasta luego!" + Style.RESET_ALL)
            break
        except UnicodeDecodeError:
            print(Fore.RED + "Error de codificación en la entrada. Por favor, intenta de nuevo." + Style.RESET_ALL)
            continue
        except Exception as e:
            print(Fore.RED + f"Se produjo un error: {e}" + Style.RESET_ALL)
            continue

if __name__ == "__main__":
    main()
