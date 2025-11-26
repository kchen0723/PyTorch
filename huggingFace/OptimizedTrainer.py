#梯度累积

class OptimizedTrainer:
    def __init__(self, model, tokenizer, accumlation_steps=4):
        self.model = model
        self.tokenizer = tokenizer
        self.accumlation_steps = accumlation_steps
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    def train_step(self, batch, step):

        outputs = self.model(**batch)
        loss = outputs.loss / self.accumlation_steps
        loss.backward()

        if(step + 1) % self.accumlation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return loss.item() * self.accumlation_steps
    
    def train(self, dataloader, epochs=3):
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            for step, batch in enumalate(dataloader):
                loss = self.train_step(batch, step)
                total_loss += loss
                if step % 100 == 0:
                    print(f"epoch: {epoch}, step {step}, loss: {loss:.4f}")
            
            avg_loss = total_loss / len(dataloader)
            print(f"epoch:{epoch}, avg loss: {avg_loss:.4f}")