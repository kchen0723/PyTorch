from transformers import pipeline
import torch
from torch.utils.data import DataLoader
#批量处理

class BatchProcessor:
    def __init__(self, task, model_name=None, batch_size=32):
        self.pipeline = pipeline(task, model=model_name, device=0 if torch.cuda.is_available() else -1)
        self.batch_size = batch_size
    
    def process_texts(self, texts):
        results = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            batch_results = self.pipeline(batch)
            results.extend(batch_results)
        
        return batch_results
    
processor = BatchProcessor("sentiment-analysis", batch_size=16)
texts = ["text" + str(i) for i in range(100)]
results = processor.process_texts(texts)

print(f"processed {len(results)} texts")