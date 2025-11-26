#内存不足问题

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_acculation_steps=4,
    dataloader_pin_memory=False,
)

model = AutoModel.from_pretrained("ber-base-chinese", gradient_checkpointing=True)

import torch
torch.cuda.empty_cache()
