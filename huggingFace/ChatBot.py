import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
#对话系统

class ChatBot:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.chat_history_ids = None

    def chat(self, user_input):
        new_user_input_ids = self.tokenizer.encode(
            user_input + self.tokenizer.eos_token,
            return_tensor='pt'
        )

        bot_input_ids = torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1) \
                        if self.chat_history_ids is not None else new_user_input_ids
        
        self.chat_history_ids = self.model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=0.7,
            do_sample=True
        )

        response=self.tokenizer.decode(
            self.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_toekns=True
        )

        return response
    
    def reset_chat(self):
        self.chat_history_ids = None

chatobot = ChatBot()

print("starting, quit to quit")
while True:
    user_input = input("")
    if user_input.lower() == "quit":
        break

    response = chatbot.chat(user_input)
    print("{response}")