import torch
import transformers
from transformers import pipeline
import sys
import io

# 文本生成
print(f"version:{transformers.__version__}")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

genertor = pipeline("text-generation",
                    model="gpt2",
                    device=0 if torch.cuda.is_available() else -1)

prompt = "introduce something about vancouver"
generated = genertor(prompt, 
                     max_length=100,
                     num_return_sequences=2,
                     temperature=0.7,
                     do_sample=True)
for i, gen in enumerate(generated):
    print(f"generation text{i + 1}:{gen['generated_text']}")