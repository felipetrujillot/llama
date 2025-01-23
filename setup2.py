import torch
from transformers import pipeline
# Configuración del modelo
model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
import torch
from transformers import pipeline

# Configuración del modelo
model_id = "meta-llama/Llama-3.1-8B-Instruct"

# Determinar el tipo de dato y el dispositivo
if torch.cuda.is_available():
    device = 0  # Asigna a la primera GPU
    capabilities = torch.cuda.get_device_capability(device)
    major, minor = capabilities
    if major >= 8:
        dtype = torch.bfloat16
        print("Usando torch.bfloat16 para optimizar el rendimiento.")
    else:
        dtype = torch.float16
        print("Usando torch.float16 para optimizar el rendimiento.")
else:
    device = -1  # CPU
    dtype = torch.float32
    print("CUDA no está disponible. Se usará la CPU.")

# Crear el pipeline con la configuración adecuada
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=dtype,
    device=device,
)

# Mensajes iniciales del sistema
messages = [
    {"role": "system", "content": "You are Nova, an intelligent virtual assistant created to provide helpful and insightful information on any topic. You always respond in a friendly and professional tone."},
]

print("Hola, soy Nova, tu asistente virtual. ¡Estoy aquí para ayudarte con cualquier pregunta o duda que tengas! Escribe 'salir' para terminar la conversación.")

while True:
    # Obtener entrada del usuario
    user_input = input("Tú: ")

    # Salir del chat si el usuario escribe "salir"
    if user_input.lower() == "salir":
        print("Nova: Gracias por usar mis servicios. ¡Hasta luego!")
        break

    # Agregar el mensaje del usuario a los mensajes
    messages.append({"role": "user", "content": user_input})

    # Preparar el prompt concatenando los mensajes
    prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])

    # Generar respuesta con el modelo
    response = pipe(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
    )

    # Obtener el texto generado
    generated_text = response[0]["generated_text"].strip()

    # Agregar la respuesta del modelo a los mensajes
    messages.append({"role": "assistant", "content": generated_text})

    # Mostrar la respuesta
    print(f"Nova: {generated_text}")

# Mensajes iniciales del sistema
messages = [
    {"role": "system", "content": "You are Nova, an intelligent virtual assistant created to provide helpful and insightful information on any topic. You always respond in a friendly and professional tone."},
]

print("Hola, soy Nova, tu asistente virtual. ¡Estoy aquí para ayudarte con cualquier pregunta o duda que tengas! Escribe 'salir' para terminar la conversación.")

user_input = open('archivo.txt').read()

# Agregar el mensaje del usuario a los mensajes
messages.append({"role": "user", "content": user_input})

# Generar respuesta con el modelo
response = pipe(
    [{"role": msg["role"], "content": msg["content"]} for msg in messages],
    max_new_tokens=9999999,
)

# Obtener el texto generado
generated_text = response[0]["generated_text"]

# Agregar la respuesta del modelo a los mensajes
messages.append({"role": "assistant", "content": generated_text})

# Mostrar la respuesta
print(f"Nova: {generated_text}")
