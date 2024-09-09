import torch
import torch.nn as nn
import torch.nn.functional as F

class MHSA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb_dim = cfg.emb_dim
        self.head_size = cfg.head_size
        self.block_size = cfg.block_size
        self.num_heads = cfg.num_heads

        self.layer_qkv = nn.Linear(self.emb_dim, 3* self.emb_dim)
        self.output = nn.Linear(self.emb_dim, self.emb_dim)
        # Create a buffer for the mask (buffers are tensors that are not updated during backpropagation)
        self.register_buffer("mask", torch.tril(torch.ones(self.block_size, self.block_size)))

    def forward(self, x):

        B, T, C = x.shape # batch size, block size, emb dim
        H = self.num_heads

        projection = self.layer_qkv(x).view(B, T,H,3*self.head_size)
        projection = projection.view(B,T,3,H,self.head_size).permute(0,3,1,2,4)
        query, key, value = projection.split([1,1,1], dim=-2)
        # for easier processing we cut the dimensions into chunks for faster MHSA implementation (as fast as single head attention)
        query = query.view(B,H,T,-1)
        key = key.view(B,H,T,-1)
        value = value.view(B,H,T,-1)
        scores = torch.matmul(query, key.transpose(-2,-1))/(self.head_size**0.5)
        scores = scores.masked_fill(self.mask[:T, :T]==0, float('-inf'))
        probs = torch.nn.functional.softmax(scores, dim=-1)
        weighted_sum = probs @ value
        out = weighted_sum.permute(0,2,1,3).contiguous().view(B,T,-1)
        out = self.output(out)
        return out

class Feedforward(nn.Module):
    def __init__(self, config):
        super().__init__()
        emb_dim = config.emb_dim

        self.feedforward = nn.Sequential(
            nn.Linear(emb_dim, emb_dim *2),
            nn.GELU(),
            nn.Linear(emb_dim*2, emb_dim*2),
            nn.GELU(),
            nn.Linear(emb_dim*2, emb_dim),
            nn.Dropout(p=0.1)
        )

    def forward(self, x):
        x = self.feedforward(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.mhsa = MHSA(config)
        self.feedforward = Feedforward(config)
        self.norm1 = nn.LayerNorm(config.emb_dim)
        self.norm2 = nn.LayerNorm(config.emb_dim)
    def forward(self, x):

        x = self.mhsa(self.norm1(x))
        x = self.feedforward(self.norm2(x))
        return x
    
class MiniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb_dim = config.emb_dim
        self.block_size = config.block_size
        self.num_layers = config.num_layers
        self.vocab_size = config.vocab_size

        self.word = nn.Embedding(config.vocab_size, config.emb_dim)
        self.position = nn.Embedding(config.block_size, config.emb_dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range (config.num_layers)])
        self.final = nn.LayerNorm(config.emb_dim)
        self.final_logits = nn.Linear(config.emb_dim, config.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idxs):
        # idxs: (B, T)
        batch_size, seq_len = idxs.shape

        assert seq_len <= self.block_size, f"Sequence length exceeds block size of {self.block_size}"

        # Note: position embeddings are encodings of the position indices (NOT the actual tokens)
        word_emb = self.word(idxs)
        position_ind = torch.arange(seq_len, device=idxs.device).unsqueeze(0)
        position_emb = self.position(position_ind)
        embed = word_emb + position_emb
        for block in self.blocks:
          embed = block(embed)
        embed = self.final(embed)
        logits = self.final_logits(embed)
        return logits

    @torch.no_grad()
    def generate(self, idxs, max_new_tokens=20):
        '''
        Takes in a sequence of indices (the tokenized sentence) and generates new tokens
        Note that the input indices should not be longer than the block size
        Returns the input sequence with the generated tokens appended (these should be decoded using the Tokenizer)

        Params
        ------
        idxs: torch.Tensor
            (B, T) tensor of token indices
        max_new_tokens: int
            Maximum number of new tokens to generate
        '''

        # idxs: (B, T)
        for _ in range(max_new_tokens):
            idxs_trimmed = idxs[:, -self.block_size:] # trim to block size

            logits = self(idxs_trimmed) # (B, T, V)

            logits = logits[:, -1, :] # (B, V)

            probs = F.softmax(logits, dim=-1) # (B, V)

            next_idx = torch.multinomial(probs, num_samples=1) # (B, 1)

            idxs = torch.cat((idxs, next_idx), dim=1) # (B, T+1)

        return idxs