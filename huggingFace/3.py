from transformers import pipeline
# 零样本分类管道
classifier = pipeline("zero-shot-classification")

# 定义你关心的主题（标签）
topic_candidates = ["cleanliness", "location", "value for money", "host communication", "amenities", "check-in process"]

# 对评论进行分类
review = "The host was very responsive and the apartment had a great view, but it wasn't as clean as I expected."
result = classifier(review, topic_candidates)

print(f"Review: {review}")
print("Topics mentioned:")
for topic, score in zip(result['labels'], result['scores']):
    print(f"  - {topic}: {score:.4f}")