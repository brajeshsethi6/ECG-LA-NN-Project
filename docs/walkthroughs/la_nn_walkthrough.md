# 🏥 Bio-LANN (Liquid-Attention NN) Walkthrough

**File Path:** `src/models/la_nn.py`

This file is the "Glue" of the project. It integrates the biological LTC cells, standard LSTM memory, and global Attention into a single cohesive model.

---

## 🏗️ Model Structure (`__init__`)

| Line | Code | Explanation |
| :--- | :--- | :--- |
| **7** | `class BioLANN(nn.Module):` | The main assembly class for the LA-NN architecture. |
| **25, 26** | `self.lstm = nn.LSTMCell(...)` | **Mixed Memory**: An LSTM handles the high-frequency temporal shortcuts, providing a stable "memory" for the LTC cell. |
| **29-33** | `self.ltc_cell = BiologicalLTCCell(...)` | The **Liquid** core. Models the physics-based ODE dynamics of the signal. |
| **36-40** | `self.attention = MultiHeadAttentionBlock(...)` | The **Attention** module for global context (summarizing the whole heartbeat). |
| **43-46** | `self.classifier = nn.Sequential(...)` | Final classification head that maps features to the 5 AAMI categories. |

---

## 🌊 Logic Flow (`forward` pass)

The model processes an ECG sequence of shape `(Batch, 180, 1)`.

### 1. State Initialization (Lines 53–57)
The model starts with "blank slate" hidden states for both the LTC and LSTM components. Everything is zero-initialized on the correct device (CPU or CUDA).

### 2. The Recurrent Loop (Lines 62–75)
Unlike standard feed-forward models, this loop iterates through every time step `t` of the ECG signal (180 steps).

*   **Step A: LSTM Update (Line 67)**: `h_lstm, c_lstm = self.lstm(x_t, (h_lstm, c_lstm))`
    *   The digital memory updates its internal state based on the current raw sample.
*   **Step B: LTC Update (Line 71)**: `h_ltc = self.ltc_cell(x_t, h_lstm)`
    *   **CRITICAL INTERACTION**: The LTC cell doesn't just look at the raw input `x_t`; it is *driven* by the LSTM hidden state. This is why it is called "Mixed Memory"—it combines biological ODEs with robust digital memory.
*   **Step C: Recording (Line 75)**: `outputs.append(h_ltc.unsqueeze(1))`
    *   We save the "brain state" of the LTC neuron at every millisecond of the signal.

### 3. Feature Refinement & Classification (Lines 78–90)

*   **Step D: Concatenation (Line 78)**: `rnn_out = torch.cat(outputs, dim=1)`
    *   Converts the list of 180 states into a single sequence tensor.
*   **Step E: Attention (Line 81)**: `attn_out, weights = self.attention(rnn_out)`
    *   Re-weights the entire sequence. If the model sees an anomaly at step 45, the attention weights will "light up" for that specific timeframe.
*   **Step F: Pooling (Line 85)**: `pooled = torch.mean(attn_out, dim=1)`
    *   Global Average Pooling: Summarizes the 180 time-steps into a single "feature vector" representing the whole heartbeat.
*   **Step G: Logits (Line 88)**: `logits = self.classifier(pooled)`
    *   Final projection to class probabilities (Normal, Ventricular, etc.).

---

## 🔬 Interpetability Feature (`get_biological_params`)

| Line | Code | Explanation |
| :--- | :--- | :--- |
| **92-98** | `def get_biological_params(self):` | This allows researchers to "peek inside" the neuron after training. It exposes the learned `gleak` (leaks) and `cm` (capacitance) to see if they align with biological realism. |

---
**Summary Flow**: 
Signal `x` → **LSTM** (Memory) → **LTC** (Dynamics) → **Attention** (Focus) → **Mean Pool** (Summary) → **Linear** (Diagnosis).
