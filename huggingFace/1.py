from transformers import pipeline
import torch

classifier = pipeline("text-classification",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if torch.cuda.is_available() else -1)
review = "The apartment was clean and cozy, but the location was noisy"
result = classifier(review)
print(result)

review = "The apartment was spotless and in a perfect location! will definitely come back."
result = classifier(review)
print(result)

review = "The apartment was spotless and in a pefect location! will definitely come back."
result = classifier(review)
print(result)