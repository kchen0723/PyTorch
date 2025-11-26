#模型加载

def safe_load_model(model_name, num_labels=None):
    try:
        if num_labels:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                ignore_mismatched_size=True
            )
        else:
            model = AutoModel.from_pretrained(model_name)

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer
    
    except Exception as e:
        print(f"errro loadeing: {e}")
        return safe_load_model("bert-base-chinese", num_labels)