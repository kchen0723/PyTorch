#内存不足问题
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback, AutoModel
)

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    dataloader_pin_memory=False,
)

model = AutoModel.from_pretrained("bert-base-chinese")
model.gradient_checkpointing_enable()

import torch
torch.cuda.empty_cache()
