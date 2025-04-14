import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.dimension_per_head = d_model // num_heads

        self.Weight_query = nn.Linear(d_model, d_model)
        self.Weight_key = nn.Linear(d_model, d_model)
        self.Weight_value = nn.Linear(d_model, d_model)
        self.Weight_output = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Query, Key, Value, mask=None):
        atten_scores = torch.matmul(Query, Key.transpose(-2, -1)) / math.sqrt(self.dimension_per_head)
        if mask is not None:
            atten_scores = atten_scores.masked_fill(mask == 0, -1e9)

        atten_probs = torch.softmax(atten_scores, dim=-1)
        output = torch.matmul(atten_probs, Value)
        return output
    
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.dimension_per_head).transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, _, seq_length, dimension_per_head = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, Query, Key, Value, mask=None):
        Query = self.split_heads(self.Weight_query(Query))
        Key = self.split_heads(self.Weight_key(Key))
        Value = self.split_heads(self.Weight_value(Value))

        atten_output = self.scaled_dot_product_attention(Query, Key, Value, mask)
        output = self.Weight_output(self.combine_heads(atten_output))
        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, dimension_of_inner_feed_foward_layer):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, dimension_of_inner_feed_foward_layer)
        self.fc2 = nn.Linear(dimension_of_inner_feed_foward_layer, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
        
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dimension_of_inner_feed_foward_layer, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, dimension_of_inner_feed_foward_layer)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        atten_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(atten_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dimension_of_inner_feed_foward_layer, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, dimension_of_inner_feed_foward_layer)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, target_mask):
        atten_output = self.self_attn(x, x, x, target_mask)
        x = self.norm1(x + self.dropout(atten_output))

        atten_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(atten_output))
        ff_output = self.feed_forward(x)
        x=self.norm3(x + self.dropout(ff_output))
        return x
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, target_vocab_size, d_model, num_heads, num_layers, dimension_of_inner_feed_foward_layer, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(target_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dimension_of_inner_feed_foward_layer, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dimension_of_inner_feed_foward_layer, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, target):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        target_mask = (target != 0).unsqueeze(1).unsqueeze(3)
        seq_length = target.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        target_mask = target_mask & nopeak_mask
        return src_mask, target_mask
    
    def forward(self, src, target):
        src_mask, target_mask = self.generate_mask(src, target)
        self_embeded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        enc_output = self_embeded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        target_embeded = self.dropout(self.positional_encoding(self.decoder_embedding(target)))
        dec_output = target_embeded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, target_mask)

        output = self.fc(dec_output)
        return output
    
src_vocab_size = 5000
target_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
dimension_of_inner_feed_foward_layer = 2048
max_seq_length = 100
dropout = 0.1

transformer = Transformer(src_vocab_size, target_vocab_size, d_model, num_heads, num_layers, dimension_of_inner_feed_foward_layer, max_seq_length, dropout)

src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))
target_data = torch.randint(1, target_vocab_size, (64, max_seq_length))

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()
for epoch in range(100):
    optimizer.zero_grad()
    output = transformer(src_data, target_data[:, :-1])
    loss = criterion(
        output.contiguous().view(-1, target_vocab_size), target_data[:, 1:].contiguous().view(-1)
    )
    loss.backward()
    optimizer.step()
    print(f"{epoch}, {loss.item()}")

transformer.eval()
val_src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))
val_target_data = torch.randint(1, target_vocab_size, (64, max_seq_length))
with torch.no_grad():
    val_output = transformer(val_src_data, val_target_data[:, :-1])
    val_loss = criterion(val_output.contiguous().view(-1, target_vocab_size), val_target_data[:, 1:].contiguous().view(-1))
    print(f"{val_loss.item()}")