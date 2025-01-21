from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model_id = "meta-llama/Llama-3.3-70B-Instruct"

pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=model_id,
    torch_dtype=torch.bfloat16,   # O float16 si presentas OOM
    device_map="auto",            # ¡Clave para offload!
)

prompt = "Explain the difference between a pirate and a ninja."
output = pipe(prompt, max_new_tokens=128)
print(output[0]["generated_text"])
