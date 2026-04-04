# 🧶 Complete Code Flow & Architecture Verification

This document is the **definitive reference** for the code execution path within the LA-NN project. It details the exact sequence of function calls for both **Training** and **Real-Time Inference**.

---

## 📂 1. High-Level Directory Map

| Path | Role | Key Functionality |
| :--- | :--- | :--- |
| `main.py` | **Entry Point (Training)** | Orchestrates data loading, model init, and training loops. |
| `streamlit_app.py` | **Entry Point (Demo)** | Runs the User Interface and Real-Time visualization logic. |
| `src/data/preprocessing.py` | **Data ETLL** | Extract (read signals), Transform (Windowing -180/+180), Load (Tensoring). |
| `src/models/` | **Neural Architecture** | Contains the custom logical blocks: `lnn.py` (Time), `attention.py` (Context), `la_nn.py` (Assembly). |
| `src/api/stream_engine.py` | **Simulation Engine** | Creates the "Live Stream" effect by yielding data in real-time buckets. |
| `src/training/trainer.py` | **Optimization Loop** | Handles the backpropagation, loss calculation, and epoch management. |

---

## 🚆 2. Execution Flow: Training Pipeline (`main.py`)

This specific flow runs when you execute `python main.py`.

### Phase A: Initialization
1.  **`Config` Setup** (`src/config.py`):
    *   Loads hyperparameters: `Batch Size=32`, `Epochs=20`, `Learning Rate=1e-3`.
    *   Sets device: `cuda` or `cpu`.

2.  **Data Loading** (`src/data/preprocessing.py` -> `load_data()`):
    *   **Scan**: Finds all MIT-BIH records in `data/mitdb`.
    *   **Loop**: For each record:
        *   `process_record()`: Reads signal + annotations.
        *   `normalize_signal()`: Applies Z-Score scaling ($ \frac{x-\mu}{\sigma} $).
        *   **Splitting**: Slices the signal into segments of 180 samples centered on R-peaks.
    *   **Dataset Creation**: Warps tensors into `PyTorch DataLoader` (Train/Val/Test).

### Phase B: Model Construction (`src/models/la_nn.py`)
1.  **`LANN.__init__`**:
    *   Builds `LNNEncoder` (3 layers of Liquid Cells).
    *   Builds `MultiHeadAttentionBlock` (8 Heads).
    *   Builds `Classifier` (Linear: Hidden Dim -> 5 Classes).

### Phase C: The Training Loop (`src/training/trainer.py`)
1.  **`Trainer.train()`** starts:
    *   **For each Epoch**:
        *   **For each Batch** in TrainLoader:
            1.  **Forward Pass**: `output = model(input)`
            2.  **Loss Calculation**: `CrossEntropyLoss(output, target)` (Weighted).
            3.  **Backward Pass**: `loss.backward()` (Calculates gradients).
            4.  **Optimizer Step**: `optimizer.step()` (Updates weights).
    *   **Validation**: Every epoch, checks accuracy on the Validation Set.
    *   **Checkpoint**: Saves `la_nn_best.pth` if validation loss improves.

---

## 🏥 3. Execution Flow: Real-Time Inference (`streamlit_app.py`)

This flow runs during the Demo command `streamlit run streamlit_app.py`.

### Phase A: UI & Engine Startup
1.  **`load_model()`**:
    *   Instantiates `LANN` structure.
    *   Loads weights from `la_nn_best.pth`.
    *   Sets mode to `.eval()` (freezes dropout/gradients).
2.  **`ECGStreamEngine` Init** (`src/api/stream_engine.py`):
    *   Loads the specific MIT-BIH record (e.g., '101') OR the uploaded CSV.
    *   Pre-calculates "Ground Truth" peak locations for simulation accuracy.

### Phase B: The Real-Time Loop
The `while st.session_state.running:` loop in `streamlit_app.py` drives the experience.

1.  **Data Fetching**:
    *   `engine.signal` yields the next `BUNDLE_SIZE` (15 samples).
    *   Appends to `st.session_state.ecg_buffer`.
2.  **Peak Detection Check**:
    *   Checks if current index matches a known R-peak.
    *   **IF Peak Detected**:
        *   **Call**: `engine.run_inference(current_idx)`.
        *   **Extraction**: Grabs `[idx-90 : idx+90]` window.
        *   **Normalization**: Re-applies Z-Score to this single window.
        *   **Inference**: Passing tensor `(1, 180, 1)` to Model -> Softmax.
        *   **Result**: Returns Class (e.g., 'V') and Confidence (e.g., 0.98).
3.  **Visualization Update**:
    *   **Logic**: If Class != 'N', change trace color to **RED**.
    *   **Plotly**: Redraws the line chart with the new buffer.
4.  **Clinical Logging**:
    *   If abnormal, append metadata + waveform snapshot to `st.session_state.anomalies`.

### Phase C: Report Generation
1.  **User Click**: "SAVE CLINICAL REPORT".
2.  **Aggregation**: Concatenates `full_session_buffer` into a single Time-Series.
3.  **HTML Rendering**:
    *   Draws the "Full Continuous Trace".
    *   Overlays Red 'X' markers at `anomaly_indices`.
    *   Iterates through `anomalies` list to create the grid of snapshot sub-plots.
    *   Encodes to Base64 for download.

---

## 📐 4. Data Transformation Reference (Tensor Shapes)

Tracking the shape of the data matrix is the best way to understand the Deep Learning flow.

| Stage | Data Shape | Description |
| :--- | :--- | :--- |
| **Raw Input** | `(180, 1)` | A single heartbeat window (0.5s). |
| **Batching** | `(Batch, 180, 1)` | Grouped for processing (Batch=1 during inference). |
| **LNN Encoder** | `(Batch, 180, 32)` | Time-series expanded to Hidden Dimensions (32 features). |
| **Attention** | `(Batch, 180, 32)` | Context applied. Shape stays same, but values are enriched. |
| **Global Pooling** | `(Batch, 32)` | Averaged across time. Time dimension collapses. |
| **Classifier** | `(Batch, 5)` | Reduced to 5 Logic scores (one per Class). |
| **Softmax** | `(Batch, 5)` | Converted to Probabilities (Sum = 1.0). |

---

*Verified for LA-NN Project Structure v1.0*
