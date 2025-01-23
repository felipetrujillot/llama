import transformers
import torch
from colorama import init, Fore, Style
import logging
import time  # Importar el módulo time

# Configurar el nivel de logging para transformers a ERROR
logging.getLogger("transformers").setLevel(logging.ERROR)

# Inicializar colorama
init(autoreset=True)

# Configuración del modelo
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# Mensaje de sistema inicial
system_message = {
    "role": "system",
    "content": (
        "Eres Amalia, una asistente virtual inteligente creada para proporcionar información útil e "
        "insightful sobre cualquier tema. Siempre respondes en un tono amable y profesional en español. "
        "No proporciones traducciones o respuestas en ningún otro idioma."
    ),
}

# Función para obtener la respuesta del modelo
def get_response(messages):
    # Unir los mensajes en un solo texto
    prompt = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            prompt += f"{content}\n"
        elif role == "user":
            prompt += f"Usuario: {content}\n"
        elif role == "assistant":
            prompt += f"Amalia: {content}\n"  # Mantener el prefijo "Amalia: "

    # Medir el tiempo de generación de la respuesta
    start_time = time.perf_counter()  # Inicio del contador
    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    end_time = time.perf_counter()  # Fin del contador

    generated_text = outputs[0]["generated_text"]

    # Calcular el tiempo transcurrido
    time_taken = end_time - start_time

    # Extraer solo la respuesta del asistente
    response = generated_text[len(prompt):].strip().split("\n")[0]

    # Eliminar el prefijo "Amalia: " si está presente
    amalia_prefix = "Amalia: "
    if response.startswith(amalia_prefix):
        response = response[len(amalia_prefix):].strip()

    return response, time_taken  # Devolver la respuesta y el tiempo

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

            # Obtener respuesta del modelo y el tiempo tomado
            assistant_response, time_taken = get_response(messages)

            # Mostrar la respuesta del asistente con el tiempo tomado
            print(Fore.GREEN + f"Amalia: {assistant_response} " +
                  Fore.YELLOW + f"(Tiempo: {time_taken:.2f} segundos)" + Style.RESET_ALL)

            # Añadir la respuesta del asistente a la conversación
            messages.append({"role": "assistant", "content": assistant_response})

        except KeyboardInterrupt:
            print("\n" + Fore.GREEN + "Amalia: ¡Hasta luego!")
            break

if __name__ == "__main__":
    main()
