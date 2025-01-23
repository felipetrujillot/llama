import transformers
import torch
from colorama import init, Fore, Style
import logging

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
            prompt += f"Amalia: {content}\n"

    # Generar la respuesta
    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    generated_text = outputs[0]["generated_text"]

    # Extraer solo la respuesta del asistente
    response = generated_text[len(prompt):].strip().split("\n")[0]
    return response

def main():
    print(Fore.GREEN + "Amalia: Hola, ¿en qué puedo ayudarte hoy?")
    messages = [system_message]

    while True:
        try:
            # Obtener entrada del usuario
            user_input = input(Fore.BLUE + "Tú: " + Style.RESET_ALL)
            if user_input.lower() in ["salir", "exit", "quit"]:
                print(Fore.GREEN + "Amalia: ¡Hasta luego!")
                break

            # Añadir el mensaje del usuario a la conversación
            messages.append({"role": "user", "content": user_input})

            # Obtener respuesta del modelo
            assistant_response = get_response(messages)

            # Mostrar la respuesta del asistente
            print(Fore.GREEN + f"Amalia: {assistant_response}")

            # Añadir la respuesta del asistente a la conversación
            messages.append({"role": "assistant", "content": assistant_response})

        except KeyboardInterrupt:
            print("\n" + Fore.GREEN + "Amalia: ¡Hasta luego!")
            break

if __name__ == "__main__":
    main()
