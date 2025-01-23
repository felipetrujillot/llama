import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
from colorama import init, Fore, Style
import logging

# Inicializar colorama
init(autoreset=True)

# Configurar el nivel de registro para suprimir mensajes innecesarios
logging.getLogger("transformers").setLevel(logging.ERROR)

# Configuración del modelo
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Cargar el tokenizer y el modelo
print(Fore.GREEN + "Cargando el modelo, por favor espera...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Configurar el modelo para usar CUDA y torch.float16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cuda"  # Utiliza la GPU disponible
)

# Definir el prompt inicial (system prompt) en español
system_prompt = (
    "Eres Nova, un asistente virtual inteligente creado para proporcionar información útil y perspicaz sobre cualquier tema. "
    "Siempre respondes en un tono amable y profesional en español."
)

# Crear una función para generar respuestas
def generate_response(user_input):
    # Formatear la entrada combinando el prompt del sistema y la entrada del usuario
    input_text = f"{system_prompt}\nUsuario: {user_input}\nNova:"
    
    # Tokenizar la entrada
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    # Generar la respuesta
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
    
    # Decodificar la salida
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extraer solo la respuesta de Nova
    # Se asume que la respuesta comienza después de "Nova:"
    response = generated_text.split("Nova:")[-1].strip()
    
    return response

def main():
    print(Fore.GREEN + "Bienvenido al chat con Nova. Escribe 'salir' para terminar.")
    while True:
        try:
            # Entrada del usuario con color azul
            user_input = input(Fore.BLUE + "Tú: " + Style.RESET_ALL)
            if user_input.strip().lower() == "salir":
                print(Fore.GREEN + "Chat finalizado. ¡Hasta luego!")
                break

            if user_input.strip() == "":
                continue  # Ignorar entradas vacías

            # Generar respuesta del asistente
            response = generate_response(user_input)

            # Mostrar la respuesta con color amarillo
            print(Fore.YELLOW + "Nova: " + Style.RESET_ALL + response)
        
        except KeyboardInterrupt:
            print(Fore.GREEN + "\nChat finalizado por el usuario. ¡Hasta luego!")
            break
        except Exception as e:
            print(Fore.RED + f"Error: {e}")

if __name__ == "__main__":
    main()
