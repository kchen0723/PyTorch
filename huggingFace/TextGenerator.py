import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
#基础文本生成

class TextGenerator:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompt, max_length=100, num_return_sequences=1, temperature=0.7, top_p=0.9):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)

        return generated_texts

generator = TextGenerator()
prompt = "The future of artificial intelligence is"
generated_texts = generator.generate(prompt, max_length=150, num_return_sequences=3)

for i, text in enumerate(generated_texts):
    print(f"{i + 1}: {text}")