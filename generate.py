import torch
import torch.nn as nn
from torch.nn import functional as F
import re

# Load the model configuration and vocabulary mappings
#model_path = '/Users/gazirahman/coursework/qac159/proj/model.pth'
model_path = '/zfshomes/hpc180/proj/model_GPU.pth'
#config_path = '/Users/gazirahman/coursework/qac159/proj/config.pth'
config_path = '/zfshomes/hpc180/proj/config_GPU.pth'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model configuration and mappings
model_config = torch.load(config_path, weights_only=True)
vocab_size = model_config['vocab_size']
n_embd = model_config['n_embd']
block_size = model_config['block_size']
n_layer = model_config['n_layer']
n_head = model_config['n_head']
dropout = model_config['dropout']
tokentonum = model_config['tokentonum']
numtotoken = model_config['numtotoken']

def process_text(text):
    pattern = r'\w+|[^\w\s]|\s'
    return re.findall(pattern, text)

def encode(text):
    tokens = process_text(text)
    encoding = [tokentonum.get(token, tokentonum[' ']) for token in tokens]
    return encoding

def decode(nums):
    tokens = [numtotoken.get(num, ' ') for num in nums]
    decoding = ''.join(tokens)
    return decoding

# Define your model classes (ensure these match your training script)
class Head(nn.Module):
    """One head of self-attention"""
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B,T,C)
        q = self.query(x)  # (B,T,C)
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        wei = self.dropout(wei)
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    def __init__(self, n_embd, num_heads, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # Concatenate along the embedding dimension
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_layer, n_head, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # Final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        loss = None
        if targets is not None:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # Crop to last block_size tokens
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # Focus on last time step
            probs = F.softmax(logits, dim=-1)  # Get probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample from distribution
            idx = torch.cat((idx, idx_next), dim=1)  # Append sampled index
        return idx

# Instantiate the model with the correct parameters
model = BigramLanguageModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    block_size=block_size,
    n_layer=n_layer,
    n_head=n_head,
    dropout=dropout
)

# Load the model state_dict
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model = model.to(device)
model.eval()

# Generate text using your own input
custom_input = input("Enter the text that you'd like to be completed: ")
context_indices = encode(custom_input)
context = torch.tensor([context_indices], dtype=torch.long, device=device)
max_new_tokens = 200
generated_indices = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
generated_text = decode(generated_indices)
print(generated_text)
