#分词问题

def robust_tokenize(tokenizer, texts, max_length=512):

    if isinstance(texts, str):
        texts = [texts]

    processed_texts = []
    for text in texts:
        if text is None or text == "":
            text="[EMPTY]"
        text = text.replace("\x100", "")
        processed_texts.append(str(text))

    try:
        encoded = tokenizer(
            processed_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return encoded
    except Exception as e:
        print(f"error: {e}")
        return tokenizer(
            ["[ERROR]"] * len(processed_texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
    