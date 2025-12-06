import torch
import transformers
from transformers import pipeline
import sys
import io

# 文本生成
print(f"version:{transformers.__version__}")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

genertor = pipeline("text-generation",
                    model="Qwen/Qwen2.5-0.5B-Instruct",
                    device=0 if torch.cuda.is_available() else -1)

prompt = "人工智能的未来发展将会带来哪些变化"
generated = genertor(prompt, 
                     max_length=256,
                     num_return_sequences=2,
                     temperature=0.7,
                     do_sample=True
                     )
for i, gen in enumerate(generated):
    print(f"generation text{i + 1}:{gen['generated_text']}")