# LA-NN: Comprehensive Architecture & Flow Reference

## 🌐 1. Core Idea

### The Challenge: "Static vs. Dynamic"
Traditional Deep Learning models (like CNNs used in standard ECG monitors) treat a heartbeat like a static image. They look at the "shape" of the wave.
**The Problem**: A heart condition isn't just about the *shape*; it's about the *timing*—the slight delays or accelerations in the electrical signal.

### The Solution: "Liquid" Neural Networks (LNN) + Attention
The **LA-NN (Liquid-Attention Neural Network)** is a hybrid architecture designed to solve this:
1.  **Liquid Component (LNN)**: Uses differential equations to model the continuous *time* evolution of the signal. It treats the ECG as a "flowing" stream rather than a static picture.
    *   *Why?* To capture the specific temporal dynamics (d/dt) of arrhythmias like PVCs or Fusion beats.
2.  **Attention Component**: Uses Transformer-style attention to look at the whole beat at once.
    *   *Why?* To understand context—e.g., "Is this QRS complex weird because of the P-wave that happened 200ms ago?"

---

## 🔄 2. Complete Data Flow (The "Lifecycle of a Heartbeat")

The system follows a strict pipeline: **Ingest -> Segment -> Liquid Encode -> Attend -> Classify -> Decide**.

### Step 1: Data Ingestion (`src/data/preprocessing.py`)
*   **Source**: The MIT-BIH Arrhythmia Database.
*   **Action**: 
    1.  Downloads raw WFDB files (`.dat`, `.hea`).
    2.  Reads annotations (`.atr`) to find the exact millisecond where a heartbeat (R-peak) occurs.
*   **Why Needed**: We cannot analyze a continuous 24-hour signal all at once. We must break it down into individual medical events.

### Step 2: Segmentation & Normalization (`src/data/preprocessing.py`)
*   **Function**: `process_record`
*   **Action**: 
    1.  Extracts a window of **180 samples** (0.5 seconds at 360Hz) centered on the R-peak.
    2.  Applies **Z-Score Normalization** (`normalize_signal`).
*   **Why Needed**:
    *   *Windowing*: Focuses the model on one beat at a time.
    *   *Normalization*: ECGs vary in amplitude based on patient body type. Normalization ($ \frac{x - \mu}{\sigma} $) ensures the model sees "relative shape" rather than raw voltage, making it universal.

### Step 3: Liquid Encoding (`src/models/lnn.py`)
*   **Function**: `LNNEncoder` -> `LiquidTimeConstantCell`
*   **Action**: The 180-sample sequence is fed into the Liquid cells.
    *   **ODE Solver**: Inside `ode_step`, the cell solves a differential equation: 
        $$ \frac{dh}{dt} = \frac{-h(t) + f(x(t))}{\tau} $$
    *   **Time Constant ($\tau$)**: This is the "Liquid" part. The network *learns* how fast or slow to react to changes in the signal for every single millisecond.
*   **Why Needed**: This allows the model to adapt. For a sharp spike (R-peak), $\tau$ becomes small (fast reaction). For a slow wave (T-wave), $\tau$ becomes large (slow reaction). CNNs cannot do this.

### Step 4: Contextual Attention (`src/models/attention.py`)
*   **Function**: `MultiHeadAttentionBlock`
*   **Action**: The output from the LNN (a sequence of features) is passed here.
    1.  **Positional Encoding**: Adds "time tags" to the features so the model knows which part is the beginning vs. the end of the beat.
    2.  **Self-Attention**: The model compares every time step with every other time step.
*   **Why Needed**: LNNs are great at immediate dynamics (step-by-step), but they can forget what happened at the start of the beat by the time they reach the end. Attention fixes this by looking at the *entire* global picture instantly.

### Step 5: Classification (`src/models/la_nn.py`)
*   **Function**: `LANN.forward`
*   **Action**:
    1.  **Global Pooling**: Averages the features across all time steps to get one "summary vector" for the beat.
    2.  **Linear Layer**: Maps this vector to 5 output numbers (Probabilities for N, S, V, F, Q).
*   **Why Needed**: To translate abstract neural features into human-readable medical categories.

### Step 6: Clinical Decision Logic (`src/api/stream_engine.py`)
*   **Function**: `get_alert_level`
*   **Action**: Checks the probabilities against safety thresholds.
    *   If $P(V) > 0.8$ OR $P(F) > 0.8$ -> **ALARM (Level 3)**.
*   **Why Needed**: Doctors don't care about "98% Probability". They care about "Is the patient dying?". This layer translates math into clinical protocols.

---

## 📚 3. Function-Level Deep Dive

### A. `src/models/lnn.py`

#### `compute_tau(x, h)`
*   **Logic**: Calculates the time-constant $\tau$ based on current input $x$ and hidden state $h$.
*   **Why**: This is the heart of the "Liquid" architecture. It dynamically adjusts the system's "speed" of processing.

#### `ode_step(x, h, tau, f_val)`
*   **Logic**: Performs one step of Euler Integration. $h_{new} = h_{old} + \frac{dh}{dt} \times \Delta t$.
*   **Why**: Neural networks usually jump from layer to layer. This function smooths the transition, mimicking biological neurons.

### B. `src/models/attention.py`

#### `PositionalEncoding`
*   **Logic**: Adds sine and cosine waves of different frequencies to the input.
*   **Why**: Without this, the Attention mechanism (which is set-based) wouldn't know the order of the signal. It would treat the ECG as a "bag of points" rather than a time-series.

### C. `src/training/trainer.py`

#### `class_weights`
*   **Logic**: `[1.0, 30.0, 15.0, 100.0, 15.0]`
*   **Why**: 90% of heartbeats are Normal. If we treated them equally, the model would just guess "Normal" every time and get 90% accuracy but miss every fatal arrhythmia. We penalize missing a "Fusion" beat 100x more than missing a Normal beat.

#### `CosineAnnealingWarmRestarts`
*   **Logic**: Periodically resets the learning rate and decays it.
*   **Why**: Helps the model jump out of "local minima" (traps in the loss landscape) and find a better, more robust solution.

### D. `src/api/stream_engine.py`

#### `stream_samples` (Async Generator)
*   **Logic**: Yields small bundles of data (e.g., 15 samples) and sleeps to match real-time duration (e.g., 41ms).
*   **Why**: To simulate a real hardware device feeding data to the UI. It ensures the dashboard looks "live" and smooth.

#### `run_inference`
*   **Logic**: Triggered only when the stream index matches a known R-peak.
*   **Why**: Efficiency. We don't need to run the heavy AI model on empty noise between heartbeats. We only run it when a beat actually happens.

---

## 4. Why This Architecture? (Defense Justification)

| Feature | Competitor (CNN/RNN) | **LA-NN (Our Approach)** | Advantage |
| :--- | :--- | :--- | :--- |
| **Temporal Modeling** | Fixed kernels / Vanishing gradients | **ODEs (Liquid)** | Adapts to irregular heart rates naturally. |
| **Context** | Limited receptive field | **Global Attention** | Sees the P-Q-R-S-T relation instantly. |
| **Explainability** | "Black Box" | **Tau + Attention Weights** | We can visualize *where* (Attention) and *how fast* (Tau) the model looked. |
| **Edge Deployment** | Large parameters | **Compact State** | ODEs are parameter-efficient, ideal for portable monitors. |

---
*Reference Document for LA-NN Project | 2026*
