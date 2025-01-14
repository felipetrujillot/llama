import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_model():
    """
    Configura el modelo y lo inicializa con DeepSpeed para utilizar múltiples GPUs.
    """
    print("Inicializando el modelo en múltiples GPUs...")

    # Cargar modelo y tokenizer
    print("Cargando el modelo y el tokenizador...")
    model_name = "EleutherAI/gpt-neo-125M"  # Reemplaza con el modelo que estés usando
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Configuración de DeepSpeed
    ds_config = {
        "replace_method": "auto",
        "tensor_parallel": {
            "tp_size": torch.cuda.device_count()  # Número de GPUs disponibles
        }
    }

    # Inicializar el modelo con DeepSpeed
    print("Inicializando el modelo con DeepSpeed...")
    model = deepspeed.init_inference(
        model=model,
        config=ds_config,
        mp_size=torch.cuda.device_count()
    )

    return model, tokenizer

def main():
    """
    Punto de entrada principal para inicializar y ejecutar el modelo.
    """
    try:
        # Inicializar modelo y tokenizer
        model, tokenizer = setup_model()
        
        # Prueba básica de generación de texto
        prompt = "DeepSpeed es una herramienta poderosa para"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_length=50)
        print("\nResultado generado:\n", tokenizer.decode(outputs[0], skip_special_tokens=True))

    except Exception as e:
        print("Error durante la inicialización o ejecución del modelo:", e)

if __name__ == "__main__":
    main()
