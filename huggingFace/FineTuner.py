from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
#使用Trainer API进行微调

class FineTuner:
    def __init__(self, model_name, num_labels):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def prepare_dataset(self, texts, labels, max_length=512):
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )

        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encoding = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encoding.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item
            
            def __len__(self):
                return len(self.lables)

        return CustomDataset(encodings, labels)
    
    def compute_metrics(self, eval_pred):
        predications, labels = eval_pred
        predications = predications.argmax(axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predications, average='weighted')
        acc = accuracy_score(labels, predications)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, train_texts, train_labels, val_texts, val_labels, output_dir="./fine_tuned_model"):
        train_dataset = self.prepare_dataset(train_texts, train_labels)
        val_dataset = self.prepare_dataset(val_texts, val_labels)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            evaluation_strategy='steps',
            eval_steps=500,
            save_strategy='steps',
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        trainer.train()
        self.tokenizer.save_pretrained(output_dir)

        return trainer
    
def fine_tune_example():
    train_texts = ["文本1", "文本2", "文本3"] * 100
    train_labels = [0, 1, 2] * 100
    val_texts = ["验证文本1", "验证文本2"] * 50
    val_labels = [0, 1] * 50

    fine_tuner = FineTuner("bert-base-chinese", num_labels=3)
    trainer = fine_tuner.train(train_texts, train_labels, val_texts, val_labels)
    print("done")
    print(trainer)

fine_tune_example()