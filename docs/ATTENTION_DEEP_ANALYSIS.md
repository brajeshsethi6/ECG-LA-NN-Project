# 👁️ Deep Dive: Multi-Head Attention Module

**File Location**: `src/models/attention.py`

This document details the **Global Context** mechanism of the LA-NN architecture. While the LNN handles the "flow" of time, this module handles the "relationships" across the entire heartbeat.

---

## 1. Class: `PositionalEncoding`
**The "Time-Stamper"**

### 📌 Role
Attention mechanisms look at all data points simultaneously (like a bag of words). They don't inherently know that point $t=0$ comes before $t=1$. This class injects that order information mathematically.

### 🧠 The Math
We add specific sine and cosine waves to the feature vector.
$$ PE(pos, 2i) = \sin(pos / 10000^{2i/d_{model}}) $$
$$ PE(pos, 2i+1) = \cos(pos / 10000^{2i/d_{model}}) $$

*   **Low Frequencies**: Identify general position (Start vs End of beat).
*   **High Frequencies**: Identify precise position (neighboring relationships).

### 🔍 Code Logic (`forward`)
```python
x = x + self.pe[:, :x.size(1), :]
```
We actively **add** the position signal to the LNN features. The neural network learns to "untangle" them later.

---

## 2. Class: `MultiHeadAttentionBlock`
**The "Context Engine"**

This is a Transformer-Encoder block (similar to BERT/GPT layers), customized for our ECG signal.

### ⚙️ Parameters (`__init__`)
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `d_model` | `int` | The dimension of the input features (must match LNN output). |
| `num_heads` | `int` | How many "types" of relationships to look for (e.g., 8). |
| `dropout` | `float` | Regularization to prevent overfitting. |

### 📥 Input -> 📤 Output
*   **Input**: `x` from `LNNEncoder`. Shape: `(Batch, 180, Hidden_Dim)`
*   **Output**: `x` (Contextualized). Shape: `(Batch, 180, Hidden_Dim)`

---

## 3. Detailed Layer Breakdown

This module implements a standard **Transformer Block** structure.

### 🔹 Component A: Pre-Layer Normalization (`ln1`)
*   **Code**: `x_norm = self.ln1(x)`
*   **Why**: We standardize the numbers (mean=0, var=1) *before* doing the heavy math. This stabilizes training, preventing values from exploding.

### 🔹 Component B: Multi-Head Attention (`mha`)
*   **Code**: `attn_out, _ = self.mha(x_norm, x_norm, x_norm)`
*   **Role**: The core logic. It splits the features into `num_heads` chunks.
    *   **Head 1** might focus on the **P-wave** relation to QRS.
    *   **Head 2** might focus on the **T-wave** height.
    *   **Head 3** might focus on the **QT interval** duration.
*   **Self-Attention**: Note we pass `x_norm` three times (Query, Key, Value). The beat attends to *itself*.

### 🔹 Component C: Residual Connection
*   **Code**: `x = x + self.dropout(attn_out)`
*   **Why**: Crucial for Deep Learning. We add the *original* input back to the *new* output.
    *   This allows the network to say "Assume the LNN was right, and just add small adjustments."
    *   It prevents "vanishing gradients" during backpropagation.

### 🔹 Component D: Feed-Forward Network (`ffn`)
*   **Structure**: `Linear -> ReLU -> Linear`
*   **Role**: After the Attention mechanism says *where* to look, this layer processes *what* was found. It considers the new context and extracts higher-level patterns.

---

## 4. Integration: Why include this in LA-NN?

### The limitation of LNN (and varying $\tau$)
The Liquid network is excellent at adapting to *immediate* changes (e.g., "The slope is increasing fast right now!"). However, by the time it reaches the T-wave (at the end), it might have "forgotten" the exact shape of the P-wave (at the start).

### The Attention Solution
The Output of this block allows the Classifier (the final step) to ask global questions:

> "I see a weird QRS complex here. Was there a P-wave 180ms ago?"

1.  **LNN Output**: "Here is the QRS shape."
2.  **Attention Output**: "Here is the QRS shape, enriched with information copied from the P-wave timestamp."

This combination allows the model to correctly identify **Fusion Beats**, which are defined specifically by the timing relationship between atrial (P) and ventricular (QRS) activity.
