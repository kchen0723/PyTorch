from transformers import Pipeline, pipeline
import torch
#自定义Pipeline

class CustomSentimentPipeline(Pipeline):
    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
  
    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        postprocess_params = {}

        if "max_length" in kwargs:
            preprocess_params["max_length"] = kwargs["max_length"]

        if "threshold" in kwargs:
            postprocess_params["threshold"] = kwargs["threshold"]
        
        return preprocess_params, {}, postprocess_params
    
    def preprocess(self, text, max_length=512):
        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length
        )
    
    def _forward(self, model_inputs):
        with torch.no_grad():
            outputs = self.model(**model_inputs)
        return outputs
    
    def postprocess(self, model_outputs, threshold=0.5):
        logits = model_outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicated_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicated_class].item()

        return {
            "label": "POSITIVE" if predicated_class == 1 else "NEGATIVE",
            "score": confidence,
            "confident": confidence > threshold
        }
    
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from transformers.pipelines import PIPELINE_REGISTRY

model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

PIPELINE_REGISTRY.register_pipeline(
    "custom-sentiment",
    pipeline_class=CustomSentimentPipeline,
    pt_model=AutoModelForSequenceClassification,
)

custom_pipeline = pipeline("custom-sentiment", model=model, tokenizer=tokenizer)
result=custom_pipeline("this produce is very good", threshold=0.8)
print(result)