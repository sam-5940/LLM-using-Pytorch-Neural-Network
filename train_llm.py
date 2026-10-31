import os
import urllib.request
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------
# 1. Hyperparameters & Configuration
# ---------------------------------------------------------
batch_size = 32         # Independent sequences processed in parallel
block_size = 64         # Maximum context length for predictions
max_iters = 10000        # Total training steps
eval_interval = 300     # Evaluation frequency
learning_rate = 3e-4    # Learning rate for AdamW
eval_iters = 100        # Batches used for computing validation loss
n_embd = 128            # Embedding dimension
n_head = 4              # Number of attention heads (n_embd // n_head = 32 per head)
n_layer = 4             # Number of Transformer blocks
dropout = 0.2           # Dropout rate for regularization

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ---------------------------------------------------------
# 2. Dataset Preparation
# ---------------------------------------------------------
dataset_file = 'tiny_shakespeare.txt'
if not os.path.exists(dataset_file):
    print("Downloading Tiny Shakespeare dataset...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, dataset_file)

with open(dataset_file, 'r', encoding='utf-8') as f:
    text = f.read()

# Character-level vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train / Validation Split (90% / 10%)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    source = train_data if split == 'train' else val_data
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# ---------------------------------------------------------
# 3. Model Architecture
# ---------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention with causal masking."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # Key, Query, Value projections in a single linear layer
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask buffer
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.size()

        # Compute Q, K, V
        q, k, v = self.c_attn(x).split(n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm residual connections."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniLLM(nn.Module):
    """Complete Decoder-Only Language Model."""
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: share weights between input embeddings and output projection
        self.token_embedding.weight = self.lm_head.weight

        # Parameter initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=20):
        """Autoregressive generation with temperature and top-k filtering."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ---------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------
if __name__ == '__main__':
    model = MiniLLM().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("\nStarting training...")
    for iter_step in range(max_iters):
        if iter_step % eval_interval == 0 or iter_step == max_iters - 1:
            losses = estimate_loss(model)
            print(f"Step {iter_step:4d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")

        xb, yb = get_batch('train')
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # ---------------------------------------------------------
    # 5. Inference / Sampling
    # ---------------------------------------------------------
    print("\n--- Generating Sample Text ---")
    start_context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(start_context, max_new_tokens=400, temperature=0.8, top_k=20)
    print(decode(generated[0].tolist()))

    # Optional: Save weights
    torch.save(model.state_dict(), 'mini_llm.pt')
    print("\nModel saved to mini_llm.pt")