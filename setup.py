import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from colorama import init, Fore, Style

# Inicializar colorama
init(autoreset=True)

# Configuración del modelo
MODEL_NAME = "huggingface/llama-3.1-8B-Instruct"  # Reemplaza con el nombre correcto si es diferente

# Cargar el tokenizer y el modelo
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cuda",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# Asignar pad_token_id al eos_token_id si no está configurado
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Definir el prompt inicial (system prompt)
system_prompt = (
    "System: Eres Nova, un asistente virtual inteligente creado para proporcionar información útil y perspicaz sobre cualquier tema. "
    "Siempre respondes en un tono amable y profesional en español."
)

# Función para generar respuestas
def generate_response(user_input, chat_history_ids=None):
    # Formatear la entrada
    input_text = f"{system_prompt}\nUser: {user_input}\nNova:"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    # Generar la respuesta
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=500,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True
        )

    # Decodificar la salida
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extraer la respuesta de Nova
    # Asumiendo que la respuesta de Nova viene después de "Nova:"
    response = generated_text.split("Nova:")[-1].strip()

    return response

def main():
    print(Fore.GREEN + "Bienvenido al chat con Nova. Escribe 'salir' para terminar.")
    while True:
        # Entrada del usuario
        user_input = input(Fore.BLUE + "Tú: " + Style.RESET_ALL)
        if user_input.lower() == "salir":
            print(Fore.GREEN + "Chat finalizado. ¡Hasta luego!")
            break

        # Generar respuesta
        response = generate_response(user_input)

        # Mostrar la respuesta
        print(Fore.YELLOW + "Nova: " + Style.RESET_ALL + response)

if __name__ == "__main__":
    main()
