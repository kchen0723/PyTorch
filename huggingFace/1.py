from transformers import pipeline

classifier = pipeline("sentiment-analysis")
review = "The apartment was clean and cozy, but the location was noisy"
result = classifier(review)
print(result)

review = "The apartment was spotless and in a perfect location! will definitely come back."
result = classifier(review)
print(result)

# classifier = pipeline("sentiment-analysis",
#     model="cardiffnlp/twitter-roberta-base-sentiment-latest")
# review = "The apartment was spotless and in a pefect location! will definitely come back."
# result = classifier(review)
# print(result)