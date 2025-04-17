# author: Gazi Rahman (grahman@wesleyan.edu)
# credit: Andrej Karpathy (https://youtu.be/kCc8FmEb1nY?si=USOan7FBimdlvX4R)
#               Model architecture/code/expertise
#         Rick Riordan (https://www.kaggle.com/datasets/shobhit043/percy-jackson-first-5-books)
#               Text from iconic series Percy Jackson and the Olympians
# description: I've taken Andrej's model of a small pre-trained text
# transformer and (1) implemented tokenization and (2) optimized it to run
# and compile on my school's high performance computing cluster. While
# previously, it was a miracle when the model would spell a single word
# correctly, now that the focus is less granular and more on sentence
# structure, it can make coherent sub-clauses and sometimes entire
# sentences. I attribute most of the intellectuals regarding the actual
# model structure to Andrej, while I implemented the data preparation,
# processing, and HPC aspects. I owe a humongous thank you to Rick
# Riordan as well, who authored the books that I loved as a child.

# load modules
# pytorch is critical for our machine learning practices and parallel data 
# processing
# re is used to process text
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data
import re

# set seed for repeatability
torch.manual_seed(1337)

# load dataset (text file)
textfile = '/zfshomes/hpc180/proj/input.txt'
#textfile = '/Users/gazirahman/coursework/qac159/proj/input2.txt'
with open(textfile, 'r', encoding='utf-8') as f:
    text = f.read()

# process_text(text): take text string and return array of token strings
# a token is a space, word, number, or punctuation mark
# eg:
# > process("Hi 5, bro!")
# ["Hi" , " " , "5" , "," , " " , "bro" , "!"]
def process_text(text):
    pattern = r'\w+|[^\w\s]|\s'
    return re.findall(pattern, text)

# split the text into unique tokens
# create mappings from tokens to numerical ids and vice versa
tokens = process_text(text)
tokens = list(set(tokens))
vocab_size = len(tokens)
tokentonum = {t: i for i, t in enumerate(tokens)}
numtotoken = {i: t for i, t in enumerate(tokens)}

# encode(text): take text string, turn into token array, return array of 
# corresponding numerical identifiers
# eg:
# > tokentonum = {"Hi": 1, " ": 2, "5": 3, ",": 4:, "bro": 5, "!": 6}
# > encode("Hi 5, bro!")
# [1 , 2 , 3 , 4 , 2 , 5 , 6]
def encode(text):
    tokens = process_text(text)
    encoding = [tokentonum[token] for token in tokens]
    return encoding

# decode(nums): take array of token ids, turn into tokens, return text string
# of combined tokens
# eg:
# > numtotoken = {1: "Hi", 2: " ", 3: "5", 4: ",", 5: "bro", 6: "!"}
# > tokenids = [1 , 2 , 3 , 4 , 2 , 5 , 6]
# > decode(tokenids)
# "Hi 5, bro!"
def decode(nums):
    tokens = [numtotoken[num] for num in nums]
    decoding = ''.join(tokens)
    return decoding

# encode all of our data using a pytorch builtin tensor which is an optimized
# datatype for pytorch's computations
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) 
train_data = data[:n]
val_data = data[n:]

# model/training hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16
block_size = 32
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
eval_iters = 200
n_embd = 64
n_head = 8
n_layer = 4
dropout = 0.0

# TextDataset object takes our text and turns it into an optimized dataset
# for PyTorch's DataLoader functions, which enable multiprocessing in data
# loading and processing
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y

# each 'head' is a point of attention in the language model; we want to use
# multiple heads so when generating a sentence, there are multiple points
# of attention indicating that each part of the text depends on other
# parts of the text
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

# we want to string together heads for multiple points of attention in parallel
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# adding layers to the model
class FeedFoward(nn.Module):

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

# not sure about this component
class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# stringing together the actual language model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

if __name__ == '__main__':

    # create dataset and data loaders with multiple workers
    train_dataset = TextDataset(train_data, block_size)
    val_dataset = TextDataset(val_data, block_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # initialize model
    model = BigramLanguageModel()

    # Wrap the model with DataParallel for multi-GPU training
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)

    # move model to device to use multiple gpu's
    model = model.to(device)

    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

    # custom loss function provided by tutorial
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split, loader in [('train', train_loader), ('val', val_loader)]:
            losses = []
            for k, (X, Y) in enumerate(loader):
                if k >= eval_iters:
                    break
                X, Y = X.to(device), Y.to(device)
                logits, loss = model(X, Y)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
        model.train()
        return out

    # optimizer for pytorch
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # initialize the training iterator
    train_iter = iter(train_loader)

    for iter in range(max_iters):

        # Evaluate the loss on train and val sets at intervals
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Get the next batch of data
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)

        xb, yb = xb.to(device), yb.to(device)

        # Evaluate the loss and perform backpropagation
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Generate text from the model
    starting_text = "I looked at "
    context_indices = encode(starting_text)
    context = torch.tensor([context_indices], dtype=torch.long, device=device)
    # If using DataParallel, access the 'module' attribute
    if isinstance(model, nn.DataParallel):
       generated = model.module.generate(context, max_new_tokens=50)[0].tolist()
    else:
       generated = model.generate(context, max_new_tokens=50)[0].tolist()
    print(decode(generated))

    # If using DataParallel
    model_path = '/zfshomes/hpc180/proj/model.pth'
    #model_path = '/Users/gazirahman/coursework/qac159/proj/model.pth'
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)

    config_path = '/zfshomes/hpc180/proj/config.pth'
    #config_path = '/Users/gazirahman/coursework/qac159/proj/config.pth'
    model_config = {
        'vocab_size': vocab_size,
        'n_embd': n_embd,
        'block_size': block_size,
        'n_layer': n_layer,
        'n_head': n_head,
        'dropout': dropout,
        'tokentonum': tokentonum,
        'numtotoken': numtotoken
    }
    torch.save(model_config, config_path)