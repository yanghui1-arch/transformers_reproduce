import torch
import torch.nn as nn
import math
from torch.nn import functional as F

class Transformer(nn.Module):
    def __init__(self, 
                 src_seq_len:int, 
                 src_vocab_size:int,
                 tgt_seq_len:int,
                 tgt_vocab_size:int,
                 hidden_size:int,
                 blocks:int=2) -> None:
        
        super().__init__()
        self.src_seq_len = src_seq_len
        self.src_vocab_size = src_vocab_size
        self.tgt_seq_len = tgt_seq_len
        self.tgt_vocab_size = tgt_vocab_size
        self.blocks = blocks

        self.src_embeddings = Embeddings(self.src_seq_len, self.src_vocab_size, hidden_size)
        self.tgt_embeddings = Embeddings(self.tgt_seq_len, self.tgt_vocab_size, hidden_size)

        self.transformers = TransformerBlock(self.src_seq_len, self.tgt_seq_len, hidden_size)
        self.project = Project(hidden_size, tgt_vocab_size)
    
    def forward(self,
                src:torch.Tensor,
                tgt:torch.Tensor,
                enc_mask:torch.BoolTensor=None,
                dec_mask:torch.BoolTensor=None):
        # (bts, seq_len, vocab_size) -> (bts, seq_len, hidden_size) -> (bts, seq_len, hidden_size)
        _x = self.transformers(self.src_embeddings(src), self.tgt_embeddings(tgt), enc_mask, dec_mask)
        # (bts, seq_len, hidden_size) -> (bts, seq_len)
        output_probs = torch.softmax(self.project(_x), dim=-1)
        return output_probs


class TransformerBlock(nn.Module):
    def __init__(self, src_seq_len:int, tgt_seq_len:int, hidden_size:int, blocks:int=6) -> None:
        super().__init__()
        self.encoder = Encoder(src_seq_len, hidden_size, blocks)
        self.decoder = Decoder(tgt_seq_len, hidden_size, blocks)

    def forward(self,
                src:torch.Tensor,
                target:torch.Tensor,
                enc_mask: torch.BoolTensor=None,
                dec_mask: torch.BoolTensor=None):
        # x: (bts, seq, hs) -> (bts, seq, hs)
        encoder_output = self.encoder(src, enc_mask)
        return self.decoder(encoder_output, target, enc_mask, dec_mask)


class Embeddings(nn.Module):
    def __init__(self, 
                seq_len:int, 
                vocab_size:int, 
                hidden_size:int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.input_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = PositionalEncoding(hidden_size, seq_len)

    def forward(self,
                x:torch.Tensor):
        # (bts, seq_len) -> (bts, seq_len, vocab_size) -> (bts, seq_len, hidden_size)
        _x = self.input_embedding(x)
        # (bts, seq_len, hidden_size) -> (bts, seq_len, hidden_size)
        return self.position_embedding(_x)

class Project(nn.Module):
    def __init__(self, hidden_size, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)


class Encoder(nn.Module):
    def __init__(self,
                 seq_len:int,
                 hidden_size:int,
                 blocks:int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.encoder_layer = nn.ModuleList([EncoderBlock(hidden_size) for _ in range(blocks)])

    def forward(self, 
                x:torch.Tensor,
                mask:torch.BoolTensor=None):
        # (bts, seq, hidden_size)
        for block in self.encoder_layer:
            x = block(x, mask)
        return x

class Decoder(nn.Module):

    def __init__(self,
                 seq_len:int,
                 hidden_size:int,
                 blocks:int,
                 heads:int=8) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.blocks = blocks
        self.decoder_layer = nn.ModuleList(DecoderBlock(hidden_size) for _ in range(self.blocks))
    
    def forward(self, 
                encoder_output: torch.Tensor,
                target: torch.Tensor,
                enc_mask: torch.BoolTensor,
                dec_mask: torch.BoolTensor):
        for block in self.decoder_layer:
            target = block(encoder_output, target, enc_mask, dec_mask)

        return target

class DecoderBlock(nn.Module):
    def __init__(self, 
                 hidden_size:int,
                 heads:int=8) -> None:
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(heads)
        self.norm = Normalization(hidden_size)
        self.ff = FeedForward(hidden_size)

    def forward(self,
                encoder_output: torch.Tensor,
                target: torch.Tensor,
                enc_mask: torch.BoolTensor,
                dec_mask: torch.BoolTensor):
        _x = self.norm(target + self.multi_head_attention(target, target, target, dec_mask))
        _x = self.norm(_x + self.multi_head_attention(_x, encoder_output, encoder_output, enc_mask))
        _x = self.norm(_x + self.ff(_x))
        return _x

class EncoderBlock(nn.Module):
    def __init__(self,
                 hidden_size:int,
                 heads:int=8,
                 d_ff:int=2048
                 ) -> None:
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(heads, d_ff, d_model=hidden_size)
        self.norm = Normalization(hidden_size)
        self.ff = FeedForward(hidden_size)
    
    def forward(self,
                x: torch.Tensor,
                mask:torch.BoolTensor=None):
        
        # (bts, seq_len, hidden_size)
        _x = self.norm(x + self.multi_head_attention(x, x, x, mask))
        output = self.norm(_x + self.ff(_x))
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 heads:int,
                 d_ff:int=2048,
                 d_k:int=64,
                 d_v:int=64,
                 d_model:int=512,
                 ) -> None:
        super().__init__()
        self.heads = heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff
        self.d_model = d_model

        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_o = nn.Linear(self.d_v, self.d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,
                q:torch.Tensor,
                k:torch.Tensor,
                v:torch.Tensor,
                mask: None):

        batch_size, seq_len, d_model = q.shape[0], q.shape[1], q.shape[2]
        # (bts, seq, d_model)
        query = self.w_q(q) 
        key = self.w_k(k)
        value = self.w_v(v)
        # (bts, h, seq_len, d_k)
        query = query.view(batch_size, seq_len, self.heads, -1).permunate(0, 2, 1, 3)
        key = key.view(batch_size, seq_len, self.heads, -1).permunate(0, 2, 1, 3)
        value = value.view(batch_size, seq_len, self.heads, -1).permunate(0, 2, 1, 3)

        # (bts, h, seq_len, seq_len) -> (bts, h, seq_len, d_k)
        attention_score = self.softmax(torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)) @ value
        if mask is not None:
            # attention_mask consists of {0, 1}
            attention_score = attention_score.masked_fill_(mask == 0, 1e-9)

        # (bts, h, seq_len, d_k)
        attention_score = attention_score.permunate(0, 2, 1, 3).view(batch_size, seq_len, d_model)

        return self.w_o(attention_score)

class Normalization(nn.Module):

    def __init__(self, hidden_size) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = 1e-6

    def forward(self,
                x: torch.Tensor
                ):
        
        norm = self.alpha * (x - torch.mean(x, dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) - self.eps) + self.bias
        return norm

class FeedForward(nn.Module):
    def __init__(self, 
                hidden_size:int,
                d_ff:int=2048) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(hidden_size, d_ff, bias=True)
        self.linear2 = nn.Linear(d_ff, hidden_size, bias=True)

    def forward(self,
                x:torch.Tensor
                ):
        _x = F.relu(self.linear1(x))
        return self.linear2(_x)

class PositionalEncoding(nn.Module):

    def __init__(self, 
                d_model: int, 
                seq_len: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model).requires_grad_(False)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # (bts, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]) # (batch, seq_len, d_model)
        return x