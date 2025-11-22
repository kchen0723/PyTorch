from transformers import pipeline
# 命名实体识别管道
ner_pipeline = pipeline("ner", aggregation_strategy="simple")

review = "The apartment in Soho was close to the subway and had a beautiful balcony."
results = ner_pipeline(review)

print(f"Review: {review}")
print("Extracted entities:")
for entity in results:
    print(f"  - {entity['word']} ({entity['entity_group']})")