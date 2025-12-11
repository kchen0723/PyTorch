import torch
import json
import time
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    stream: bool = False

app = FastAPI()
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# Allow ChatUI to connect
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
chatbot = pipeline(
    "text-generation",
    model = model_name,
    device=0 if torch.cuda.is_available() else -1
)

#OPENAI_BASE_URL=http://127.0.0.1:8000/ for huggingface chatui, please use this URL.
@app.get("/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "gpt2",
                "object": model_name,
                "owned_by": "local",
                "permission": []
            }
        ]
    }

# Generator for streaming tokens
async def generate_stream(prompt: str, model: str):
    response = chatbot(
        prompt,
        max_new_tokens=128,
        do_sample=True,
        top_p=0.9,
        temperature=0.9,
        pad_token_id=chatbot.tokenizer.eos_token_id
    )
    generated = response[0]["generated_text"]
    reply = generated[len(prompt):].strip()

    response_id = f"chatcmpl-{int(time.time())}"

    # Stream token by token (or chunk)
    for i, token in enumerate(reply.split()):
        chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": token + " "},
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.05)  # simulate streaming delay

    # Final stop message
    final_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Concatenate messages
    prompt = ""
    for msg in request.messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            prompt += f"用户: {content}\n"
        elif role == "assistant":
            prompt += f"助手: {content}\n"
    prompt += "助手: "

    if request.stream:
        return StreamingResponse(
            generate_stream(prompt, request.model),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming fallback
        response = chatbot(
            prompt,
            max_new_tokens=128,
            do_sample=True,
            top_p=0.9,
            temperature=0.9,
            pad_token_id=chatbot.tokenizer.eos_token_id
        )
        generated = response[0]["generated_text"]
        reply = generated[len(prompt):].strip()
        response_id = f"chatcmpl-{int(time.time())}"
        result = {
            "id": response_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(reply.split()),
                "total_tokens": len(prompt.split()) + len(reply.split())
            }
        }
        return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# use this command to check the endpoint: 
# curl -X POST http://127.0.0.1:8000/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"gpt2\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}"
# since huggingface chatui using stream, we need to change above codes to support stream