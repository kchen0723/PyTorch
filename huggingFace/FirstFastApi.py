import torch
import json
import time
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    stream: bool = False

app = FastAPI()

# Allow ChatUI to connect
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Qwen instruction model
# chatbot = pipeline(
#     "text-generation",
#     model="Qwen/Qwen2.5-0.5B-Instruct",
#     device=0 if torch.cuda.is_available() else -1
# )

chatbot = pipeline(
    "text-generation",
    model="gpt2",
    device=0 if torch.cuda.is_available() else -1
)

# HuggingFace ChatUI expects this exact endpoint
@app.get("/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "gpt2",
                "object": "model",
                "owned_by": "local",
                "permission": []
            }
        ]
    }

@app.post("/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    # Concatenate all user and assistant messages for multi-turn
    prompt = ""
    for msg in request.messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            prompt += f"用户: {content}\n"
        elif role == "assistant":
            prompt += f"助手: {content}\n"

    prompt += "助手: "

    response = chatbot(
        prompt,
        max_new_tokens=128,
        do_sample=True,
        top_p=0.9,
        temperature=0.9,
        pad_token_id=chatbot.tokenizer.eos_token_id
    )
    print(f"request is: {request}")
    print(f"response is: {response}")
    generated = response[0]["generated_text"]
    # Only take new output after the prompt
    reply = generated[len(prompt):].strip()
    if not reply:
        reply = "抱歉，我暂时无法生成回答。"

    print(f"reply is: {reply}")
    response_id = f"chatcmpl-{int(time.time())}"
    result = {
        "id": response_id,
        "object": "chat.completion",
        "created": int(time.time()),        
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "text": "hello, this is my reply",
                # "message": {"role": "assistant", "content": "hello, this is my reply"},
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": 2,
            "total_tokens": len(prompt.split()) + 2
        }
    }
    result_string = json.dumps(result)
    print(result_string)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# use this command to check the endpoint: 
# curl -X POST http://127.0.0.1:8000/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"gpt2\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}"
# since huggingface chatui using stream, we need to change above codes to support stream
