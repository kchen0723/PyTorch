from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
#使用accelerate进行分布式训练

def train_with_accelerate():
    accelator = Accelerator()

    model_name = "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    model.train()

    for epoch in range(3):
        for batch in train_datloader:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss

            accelator.batchward(loss)
            optimizer.step()

            if accelerator.is_main_process:
                print(f"loss: {loss.item():.4f}")
    
    accelator.wait_for_everyone()
    unwrapped_model = accelator.unwrap_model(model)
    unwrapped_model.save_pretrained("./accelarated_model", save_function=accelator.save)