from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader

train_iter, test_iter = IMDB(split=('train', 'test'))
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

def encode(text):
    return vocab(tokenizer(text))

def collate_batch(batch):
    labels, texts = [], []
    for label, text in batch:
        labels.append(1 if label == 'pos' else 0)
        texts.append(encode(text))
    return labels, texts

train_iter, test_iter = IMDB(split=('train', 'test'))
train_dataloader = DataLoader(list(train_iter), batch_size=32, collate_fn=collate_batch)

for labels, texts in train_dataloader:
    print(labels)
    print(texts)
    break