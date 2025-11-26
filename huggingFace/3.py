from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")  #情感分析
generator = pipeline("text-generation")              #文本生成
qa_pipeline = pipeline("question-answering")         #问答系统 
# summarizer = pipeline("summarization")               #文本摘要

from transformers import AutoModel, AutoTokenizer
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "Helllo, how are you?"
encoded = tokenizer(text, return_tensors="pt")
print(encoded)

decoded = tokenizer.decode(encoded["input_ids"][0])
print(decoded)