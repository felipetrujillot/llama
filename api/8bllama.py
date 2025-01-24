import transformers 
import torch
from colorama import init, Fore, Style
import logging
import time
import sys

# Importaciones adicionales para RAG
from rag_handler import obtener_contexto

# Configurar el nivel de logging para transformers a ERROR
logging.getLogger("transformers").setLevel(logging.ERROR)

# Inicializar colorama
init(autoreset=True)

# Configuración del modelo y tokenizer
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Mensaje de sistema inicial
system_message = {
    "role": "system",
    "content": (
        "Eres Amalia, una asistente virtual inteligente creada para proporcionar información útil e "
        "insightful sobre cualquier tema. Siempre respondes en un tono amable y profesional en español. "
        "No proporciones traducciones o respuestas en ningún otro idioma. "
        "No proporciones respuestas redundantes. "
        "No proporciones respuestas incompletas. "
        "No hagas preguntas de seguimiento ni ofrezcas sugerencias adicionales después de responder."
    ),
}

# Función para obtener la respuesta del modelo en streaming
def get_response_streaming(messages, user_query):
    # Obtener contexto de RAG
    contextos = obtener_contexto(user_query, k=5)  # Puedes ajustar k según tus necesidades
    contexto_relevante = "\n".join(contextos)
    
    # Crear un prompt con el contexto
    prompt = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            prompt += f"{content}\n"
        elif role == "user":
            prompt += f"Usuario: {content}\n"
        elif role == "assistant":
            prompt += f"Amalia: {content}\n"
    
    # Incluir el contexto recuperado
    prompt += f"Contexto relevante:\n{contexto_relevante}\n"
    prompt += f"Usuario: {user_query}\nAmalia: "
    
    # Tokenizar el prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    # Generar tokens
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

    # Obtener los nuevos tokens generados
    new_tokens = generated_ids.sequences[0][input_ids.shape[-1]:]

    # Decodificar los tokens uno por uno
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
    print(Fore.GREEN + "Amalia: Hola, ¿en qué puedo ayudarte hoy?")
    messages = [system_message]

    while True:
        try:
            # Obtener entrada del usuario
            user_input = input(Fore.BLUE + "User: " + Style.RESET_ALL)
            if user_input.lower() in ["salir", "exit", "quit"]:
                print(Fore.GREEN + "Amalia: ¡Hasta luego!")
                break

            # Añadir el mensaje del usuario a la conversación
            messages.append({"role": "user", "content": user_input})

            # Obtener respuesta del modelo y el tiempo tomado en streaming
            print(Fore.GREEN + "Amalia: ", end='', flush=True)
            assistant_response, time_taken = get_response_streaming(messages, user_input)

            # Añadir la respuesta del asistente a la conversación
            messages.append({"role": "assistant", "content": assistant_response})

        except KeyboardInterrupt:
            print("\n" + Fore.GREEN + "Amalia: ¡Hasta luego!")
            break
        except UnicodeDecodeError:
            print(Fore.RED + "Error de codificación en la entrada. Por favor, intenta de nuevo." + Style.RESET_ALL)
            continue

if __name__ == "__main__":
    main()
