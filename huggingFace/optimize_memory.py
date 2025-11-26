import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
#内存优化

def optimize_memory():
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese", gradient_checkpointing=True)

    from torch.cuda.amp import autocase, GradScaler
    scaler = GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    def train_step_with_amp(batch):
        optimizer.zero_grad()
        with autocast():
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss.item()
    
    return train_step_with_amp

if torch.cuda.is_available():
    train_step = optimizer_memory()
    print("memory optimized")