import torch
from transformers import pipeline, AutoTokenizer
from colorama import init, Fore, Style
import logging
import time  # Importar el módulo time para medir el tiempo de respuesta

# Inicializar colorama
init(autoreset=True)

# Configurar el nivel de registro para suprimir mensajes innecesarios
logging.getLogger("transformers").setLevel(logging.ERROR)

# Configuración del modelo
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Determinar el tipo de dato y el dispositivo
if torch.cuda.is_available():
    device = 0  # Asigna a la primera GPU
    capabilities = torch.cuda.get_device_capability(device)
    major, minor = capabilities
    if major >= 8:
        dtype = torch.bfloat16
        print(Fore.GREEN + "Usando torch.bfloat16 para optimizar el rendimiento.")
    else:
        dtype = torch.float16
        print(Fore.GREEN + "Usando torch.float16 para optimizar el rendimiento.")
else:
    device = -1  # CPU
    dtype = torch.float32
    print(Fore.RED + "CUDA no está disponible. Se usará la CPU.")

# Crear el pipeline con la configuración adecuada
print(Fore.GREEN + "Cargando el modelo, por favor espera...")
pipe = pipeline(
    "text-generation",
    model=MODEL_ID,
    torch_dtype=dtype,
    device=device,
)

# Cargar el tokenizer para obtener `eos_token_id` si es necesario
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Definir el prompt inicial (system prompt) en inglés
system_prompt = (
    "You are Amalia, an intelligent virtual assistant created to provide helpful and insightful information on any topic. "
    "You always respond in a friendly and professional tone in Spanish."
)

# Inicializar los mensajes con el prompt del sistema
messages = [
    {"role": "system", "content": system_prompt},
]

print(Fore.GREEN + "Hola, soy Amalia, tu asistente virtual. ¡Estoy aquí para ayudarte con cualquier pregunta o duda que tengas! Escribe 'salir' para terminar la conversación.")

while True:
    try:
        # Obtener entrada del usuario con color azul
        user_input = input(Fore.BLUE + "Tú: " + Style.RESET_ALL).strip()

        # Salir del chat si el usuario escribe "salir"
        if user_input.lower() == "salir":
            print(Fore.GREEN + "Amalia: Gracias por usar mis servicios. ¡Hasta luego!")
            break

        if user_input == "":
            continue  # Ignorar entradas vacías

        # Agregar el mensaje del usuario a los mensajes
        messages.append({"role": "user", "content": user_input})

        # Preparar el prompt concatenando los mensajes
        prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])

        # Iniciar el contador de tiempo antes de generar la respuesta
        start_time = time.time()

        # Generar respuesta con el modelo usando pipeline
        response = pipe(
            prompt,
            max_new_tokens=4096,        # Reducido para evitar respuestas excesivamente largas
            do_sample=True,
            temperature=0.5,           # Reducido para mayor coherencia
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Finalizar el contador de tiempo después de generar la respuesta
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Obtener el texto generado
        generated_text = response[0]["generated_text"].strip()

        # Extraer la respuesta de Amalia
        # Se asume que la respuesta comienza después de "Amalia:"
        if "Amalia:" in generated_text:
            response_text = generated_text.split("Amalia:")[-1].strip()
        else:
            # Fallback si "Amalia:" no está presente
            response_text = generated_text

        # Agregar la respuesta del modelo a los mensajes
        messages.append({"role": "assistant", "content": response_text})

        # Mostrar la respuesta con color amarillo
        print(Fore.YELLOW + "Amalia: " + Style.RESET_ALL + response_text)

        # Mostrar el tiempo de respuesta con dos decimales
        print(Fore.CYAN + f"Tiempo de respuesta: {elapsed_time:.2f} segundos" + Style.RESET_ALL)

    except KeyboardInterrupt:
        print(Fore.GREEN + "\nChat finalizado por el usuario. ¡Hasta luego!")
        break
    except Exception as e:
        print(Fore.RED + f"Error: {e}")
