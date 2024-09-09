from dataclasses import dataclass

@dataclass
class Config:
    block_size: int = 8
    emb_dim: int = 256
    head_size: int = 32
    num_heads: int = 8
    num_layers: int = 2
    vocab_size: int = -1 # vocab size of the tokenizer