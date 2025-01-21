import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def main():
    # 1) Inicializa Accelerator con la config de accelerate_config.yaml
    accelerator = Accelerator()

    # 2) Nombre del modelo en Hugging Face
    model_id = "meta-llama/Llama-3.3-70B-Instruct"

    # 3) Carga del modelo y tokenizer en CPU (no device_map="auto" aquí)
    #    Asegúrate de tener permisos para descargar este repo si es privado.
    print("Cargando el modelo en CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  # O float16 si lo prefieres
        device_map=None,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 4) Prepara el modelo con Accelerator (esto aplica DeepSpeed Zero Offload)
    #    No necesitas llamar directamente a deepspeed.initialize().
    model = accelerator.prepare(model)

    # 5) Crea la pipeline de Hugging Face con el modelo “acelerado”
    #    El pipeline estará listo para inferencia con Zero Offload.
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=accelerator.device  # Asigna la GPU que maneja Accelerate
    )

    # 6) Prueba de inferencia
    print("Generando respuesta...")
    prompt = "Explain the difference between a pirate and a ninja."
    outputs = pipe(prompt, max_new_tokens=128)

    # 7) Imprimir resultado
    print(outputs[0]["generated_text"])

if __name__ == "__main__":
    main()
