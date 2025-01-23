import torch
from transformers import pipeline
from colorama import init, Fore, Style
import logging

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

# Mensajes iniciales del sistema en español
system_prompt = (
    "Eres Nova, un asistente virtual inteligente creado para proporcionar información útil y perspicaz sobre cualquier tema. "
    "Siempre respondes en un tono amable y profesional en español."
)

# Inicializar los mensajes con el prompt del sistema
messages = [
    {"role": "system", "content": system_prompt},
]

while True:
    try:
        # Obtener entrada del usuario con color azul
        user_input = input(Fore.BLUE + "Tú: " + Style.RESET_ALL)
        
        # Salir del chat si el usuario escribe "salir"
        if user_input.strip().lower() == "salir":
            print(Fore.GREEN + "Amalia: Gracias por usar mis servicios. ¡Hasta luego!")
            break
        
        if user_input.strip() == "":
            continue  # Ignorar entradas vacías
        
        # Agregar el mensaje del usuario a los mensajes
        messages.append({"role": "user", "content": user_input})
        
        # Preparar el prompt concatenando los mensajes
        prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])
        
        # Generar respuesta con el modelo usando pipeline
        response = pipe(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id if 'tokenizer' in globals() else None,  # Asegura que se detenga correctamente
        )
        
        # Obtener el texto generado
        generated_text = response[0]["generated_text"].strip()
        
        # Extraer la respuesta de Nova
        # Se asume que la respuesta comienza después de "Nova:"
        if "Nova:" in generated_text:
            response_text = generated_text.split("Nova:")[-1].strip()
        else:
            response_text = generated_text  # Fallback en caso de que "Nova:" no esté presente
        
        # Agregar la respuesta del modelo a los mensajes
        messages.append({"role": "assistant", "content": response_text})
        
        # Mostrar la respuesta con color amarillo
        print(Fore.YELLOW + "Nova: " + Style.RESET_ALL + response_text)
    
    except KeyboardInterrupt:
        print(Fore.GREEN + "\nChat finalizado por el usuario. ¡Hasta luego!")
        break
    except Exception as e:
        print(Fore.RED + f"Error: {e}")
