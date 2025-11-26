from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
#模型量化

def quantize_model(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    quantize_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    return quantize_model, tokenizer

quantize_model, toknizer = quantize_model("bert-base-chinese")

import time
text = "this is a testing text"
inputs = toknizer(text, return_tensors="pt")

start_time = time.time()
with torch.no_grad():
    outputs = quantize_model(**inputs)

end_time = time.time()
print(f"{end_time - start_time:.4f} seconds")