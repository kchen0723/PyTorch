from transformers import pipeline
import sys
import io
#问答系统

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
qa_pipeline = pipeline("question-answering",
                       model="bert-base-chinese")

context = """"
人工智能（AI）是计算机科学的一个分支，它试图理解智能的实质，
并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
人工智能的研究领域包括机器学习，深度学习，自然语言处理和计算机视觉等方面。
"""""

question="人工智能的研究领域包括哪些？"

answer = qa_pipeline(question, context)
print(answer)