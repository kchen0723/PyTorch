from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

reviews = [
    "The apartment was spotless and in a pefect location! will definitely come back.",
    "It was okay, but the wifi was unreliable and the check-in process was confusing",
    "Absolutely terrible! The place was nothing like the photos and it was very noisy."
]

for review in reviews:
    result = sentiment_pipeline(review)[0]
    print(f"Review: {review}")
    print(f"Sentiment: {result['label']}, Confidence: {result['score']:.4f}\n")