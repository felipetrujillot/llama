import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM

def setup_model():
    model_name = "EleutherAI/gpt-neo-125M"  # Cambia al modelo que estés usando
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Configuración de DeepSpeed
    ds_config = {
        "replace_with_kernel_inject": True,
        "tensor_parallel": {
            "tp_size": 4  # Número de GPUs disponibles
        }
    }

    model = deepspeed.init_inference(
        model,
        config=ds_config,
        dtype=torch.float16  # Ajusta según lo necesario
    )
    return model, tokenizer

def main():
    try:
        print("Inicializando el modelo y el tokenizador...")
        model, tokenizer = setup_model()

        # Generación de ejemplo
        prompt = "Hello, world! This is an example of"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=50)
        print("Resultado generado:", tokenizer.decode(outputs[0]))
    except Exception as e:
        print("Error durante la inicialización o ejecución del modelo:", str(e))

if __name__ == "__main__":
    main()
