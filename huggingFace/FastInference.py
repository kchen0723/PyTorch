import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#推理加速

class FastInference:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    @torch.no_grad()
    def predicate_batch(self, texts, batch_size=32):
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            predications = torch.softmax(outputs.logits, dim=-1)

            for j, pred in enumerate(predications):
                predicated_class = torch.argmax(pred).item()
                confidence = pred[predicated_class].item()
                results.append({
                    "text": batch_texts[j],
                    "class": predicated_class,
                    "confidence": confidence
                })

        return results

fast_inference = FastInference("./my_classifier")
texts=["text1", "text2", "text3"] * 100
results = fast_inference.predicate_batch(texts, batch_size=16)
print(f"processed {len(results)} texts")