import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
#文本分类示例

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return{
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def prepare_data():
    data = {
        'text': [
            '这个产品质量很好，非常满意',
            '服务态度差，不推荐',
            '价格合理，值得购买',
            '物流太慢了，体验不好',
            '功能强大，使用方便'
        ] * 100,
        'label': [1, 0, 1, 0, 1] * 100
    }

    df = pd.DataFrame(data)
    return train_test_split(df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42)

def train_classifier():
    train_texts, test_texts, train_labels, test_labels = prepare_data()

    model_name = 'hfl/chinese-roberta-wwm-ext'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)
    test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model("./my_classifier")
    return trainer

def predict_with_trained_model(text):
    tokenizer = AutoTokenizer.from_pretrained("./my_classifier")
    model = AutoModelForSequenceClassification.from_pretrained("./my_classifier")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        predications = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicated_class = torch.argmax(predications, dim=-1).item()
        confidence = predications[0][predicated_class].item()

    return predicated_class, confidence

if __name__ == "__main__":
    trainer = train_classifier()
    test_text = "this product is very good"
    pred_class, confidence = predict_with_trained_model(test_text)
    print(pred_class)
    print(confidence)