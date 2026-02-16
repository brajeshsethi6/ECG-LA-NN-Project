# 🧠 Multi-Head Attention Block Walkthrough

**File Path:** `src/models/attention.py`

This module implements the global context-aware mechanism of the LA-NN architecture. It consists of two main components: **Positional Encoding** and the **Multi-Head Attention Block**.

---

## 1. `PositionalEncoding` Class
Since attention mechanisms are permutation-invariant (they don't "see" order), we must inject time-series sequence data with information about the position of each sample.

### Line-by-Line Explanation

| Line | Code | Explanation |
| :--- | :--- | :--- |
| **6** | `class PositionalEncoding(nn.Module):` | Defines the module for adding positional info to embeddings. |
| **7** | `def __init__(self, d_model, max_len=5000):` | Constructor. `d_model` is the hidden dimension, `max_len` is the maximum sequence length supported. |
| **10** | `pe = torch.zeros(max_len, d_model)` | Initializes a zero matrix for the positional encoding table. |
| **11** | `position = torch.arange(0, max_len...` | Creates a column vector from `0` to `max_len-1` representing position indices. |
| **12** | `div_term = torch.exp(...)` | Calculates the scaling frequencies using a log-space formula based on the Attention paper (Vaswani et al.). |
| **14** | `pe[:, 0::2] = torch.sin(...)` | Applies the Sine function to even-indexed dimensions ($2i$). |
| **15** | `pe[:, 1::2] = torch.cos(...)` | Applies the Cosine function to odd-indexed dimensions ($2i+1$). |
| **17** | `pe = pe.unsqueeze(0)` | Adds a batch dimension to allow broadcasting across different batch sizes. |
| **18** | `self.register_buffer('pe', pe)` | Saves the table in the model state but marks it as non-trainable (fixed weights). |
| **20** | `def forward(self, x):` | Execution method. `x` is the input tensor of shape `(batch, seq, dim)`. |
| **22** | `x = x + self.pe[:, :x.size(1), :]` | **The Core Step**: Adds the precomputed positional vectors to the input vectors. |

---

## 2. `MultiHeadAttentionBlock` Class
This is the primary refinement layer that allows the model to selectively focus on critical parts of the ECG beat (like the P-wave vs the R-peak).

### Line-by-Line Explanation

| Line | Code | Explanation |
| :--- | :--- | :--- |
| **25** | `class MultiHeadAttentionBlock(nn.Module):` | Main block containing Attention, FFN, and LayerNorm. |
| **28** | `self.ln1 = nn.LayerNorm(d_model)` | First Layer Normalization (Pre-Norm architecture). |
| **29** | `self.mha = nn.MultiheadAttention(...)` | PyTorch's optimized Multi-Head Attention module. `num_heads` split the dimension to attend to different features. |
| **30** | `self.ln2 = nn.LayerNorm(d_model)` | Second Layer Normalization before the Feed-Forward network. |
| **31-36** | `self.ffn = nn.Sequential(...)` | The Feed-Forward Network: Expands the dimension by 4x, applies ReLU, and projects back. |
| **38** | `self.positional_encoding = ...` | Instantiates the ordering injector defined above. |

### Execution Flow (`forward`)

1.  **Ordering**: `x = self.positional_encoding(x)` — Adds time-step awareness to the LNN hidden states.
2.  **Normalization 1**: `x_norm = self.ln1(x)` — Stabilizes training by normalizing the input mean/variance.
3.  **Self-Attention**: `attn_out, attn_weights = self.mha(...)` — The Q (Query), K (Key), and V (Value) are all `x_norm`. The model calculates how much each time step should relate to every other time step.
4.  **Residual Connection 1**: `x = x + self.dropout(attn_out)` — Adds the attention output back to the original input to prevent gradient vanishing.
5.  **Normalization 2**: `x_norm = self.ln2(x)` — Normalizes before the non-linear transformation.
6.  **Refinement**: `ffn_out = self.ffn(x_norm)` — Refines the attended features through dense layers.
7.  **Residual Connection 2**: `x = x + ffn_out` — Final summation for the output block.

---
**Why this matters for ECG**: In an arrhythmia, the relationship between the P-wave (early) and the R-peak (middle) is vital. Attention allows the model to link these far-apart events mathematically.
