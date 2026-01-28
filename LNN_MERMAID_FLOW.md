# LNN (Liquid Neural Network) - Complete Flow Diagram

## Overall Architecture Flow

```mermaid
graph TD
    A["Input Data<br/>(batch, input_dim)"] --> B["BiologicalLTCCell<br/>Initialization"]
    H["Hidden State<br/>(batch, hidden_dim)"] --> B
    
    B --> C["Parameter Initialization<br/>Using torch.rand"]
    C --> C1["Biological Parameters<br/>Cm, Gleak, Vleak"]
    C --> C2["Synaptic Parameters<br/>w, sigma, mu, erev"]
    C --> C3["Sensory Parameters<br/>sensory_w, sensory_sigma<br/>sensory_mu, sensory_erev"]
    
    A --> D["Forward Pass<br/>forward(x, h, dt)"]
    H --> D
    
    D --> E["Pre-compute Sensory Activations"]
    E --> E1["Softplus on sensory_w<br/>F.softplus(sensory_w)"]
    E --> E2["Sigmoid Activation<br/>sigmoid(sigma * v_pre - mu)"]
    E --> E3["Compute sensory_w_act<br/>= sensory_w_act * sigmoid"]
    E --> E4["Sum for Numerator & Denominator<br/>sensory_num, sensory_den"]
    
    D --> F["ODE Integration Setup<br/>Semi-implicit Euler"]
    F --> F1["Compute cm_t<br/>F.softplus(cm) / dt_step"]
    F --> F2["Compute gleak<br/>F.softplus(gleak)"]
    
    F1 --> G["ODE Steps Loop<br/>for _ in range(ode_steps)"]
    F2 --> G
    E4 --> G
    
    G --> G1["Compute Recurrent Activations<br/>w_act = softplus(w) * sigmoid"]
    G1 --> G2["Sum Recurrent Contributions<br/>w_num, w_den"]
    G2 --> G3["Semi-implicit Euler Update<br/>Numerator = cm_t*v + gleak*vleak + w_num"]
    G3 --> G4["Compute Denominator<br/>= cm_t + gleak + w_den + epsilon"]
    G4 --> G5["Update Membrane Potential<br/>v_pre = numerator / denominator"]
    
    G5 --> G["Loop Back<br/>ode_steps times"]
    
    G --> H1["Output Membrane Potential<br/>(batch, hidden_dim)"]
    H1 --> I["Return Updated Hidden State"]
```

## Detailed Process Flow

```mermaid
graph TD
    START["START: Process Input"] --> INIT["Initialize Parameters<br/>Biological + Synaptic"]
    
    INIT --> SENSORY["SENSORY INPUT PROCESSING"]
    SENSORY --> S1["Apply Softplus to sensory_w<br/>w_sensory_pos = softplus(sensory_w)"]
    S1 --> S2["Compute Sigmoid Gate<br/>gate = sigmoid(sensory_sigma*(x-sensory_mu))"]
    S2 --> S3["Weighted Activation<br/>sensory_w_act = w_sensory_pos * gate"]
    S3 --> S4["Aggregate Contributions<br/>sensory_num = sum(w_act * erev)<br/>sensory_den = sum(w_act)"]
    
    S4 --> ODE["ODE INTEGRATION LOOP<br/>Semi-implicit Euler Method"]
    
    ODE --> O1["Step 1 of ode_steps"]
    O1 --> O2["Recurrent Processing<br/>w_act_rec = softplus(w) * sigmoid(mu,sigma)"]
    O2 --> O3["Compute Sums<br/>w_num = sum(w_act * erev) + sensory_num<br/>w_den = sum(w_act) + sensory_den"]
    O3 --> O4["Apply Conductance Model<br/>numerator = cm_t*v_pre + gleak*vleak + w_num"]
    O4 --> O5["Denominator<br/>denominator = cm_t + gleak + w_den + epsilon"]
    O5 --> O6["Update State<br/>v_pre = numerator / denominator"]
    O6 --> O7{"More ODE Steps?"}
    
    O7 -->|Yes| O1
    O7 -->|No| OUTPUT["OUTPUT<br/>Final Hidden State (v_pre)"]
    
    OUTPUT --> END["END: Return State"]
```

## Biological Conductance Model

```mermaid
graph LR
    A["Membrane Potential (v)"] --> B["Leak Conductance<br/>gleak * vleak"]
    A --> C["Input Conductance<br/>sum(sensory_w * erev)"]
    A --> D["Recurrent Conductance<br/>sum(w * erev)"]
    
    B --> E["Total Conductance<br/>Model"]
    C --> E
    D --> E
    
    E --> F["Semi-implicit Euler<br/>v_new = numerator / denominator"]
    
    F --> G["Next Time Step"]
```

## Parameter Types and Purpose

```mermaid
graph TD
    PARAMS["LTC Cell Parameters"] --> BIO["Biological Parameters"]
    PARAMS --> SYN["Synaptic Parameters"]
    PARAMS --> SENS["Sensory Parameters"]
    
    BIO --> B1["cm: Membrane Capacitance<br/>Controls charging rate"]
    BIO --> B2["gleak: Leak Conductance<br/>Passive ion leakage"]
    BIO --> B3["vleak: Leak Reversal Potential<br/>Resting membrane potential"]
    
    SYN --> S1["w: Weight Matrix<br/>Hidden-to-Hidden strength"]
    SYN --> S2["sigma: Synaptic Sensitivity<br/>Sigmoid sharpness"]
    SYN --> S3["mu: Synaptic Threshold<br/>Sigmoid center"]
    SYN --> S4["erev: Reversal Potential<br/>Excitatory/Inhibitory"]
    
    SENS --> SE1["sensory_w: Input Weights<br/>Input-to-Hidden strength"]
    SENS --> SE2["sensory_sigma: Input Sensitivity"]
    SENS --> SE3["sensory_mu: Input Threshold"]
    SENS --> SE4["sensory_erev: Input Reversal Potential"]
```

## Key Equations

```mermaid
graph TD
    E1["Sigmoid Gate Function<br/>g(v) = sigmoid(sigma * (v - mu))"] --> E2["Softplus Constraint<br/>softplus(x) = log(1 + exp(x))"]
    
    E2 --> E3["Weighted Activation<br/>w_act = softplus(w) * g(v)"]
    
    E3 --> E4["Aggregation<br/>sum_g = sum(w_act)<br/>sum_v = sum(w_act * erev)"]
    
    E4 --> E5["Semi-implicit Update<br/>v_new = (cm_t*v + gleak*vleak + sum_v)<br/>/ (cm_t + gleak + sum_g + epsilon)"]
    
    E5 --> E6["Stability Property<br/>Handles Stiff ODEs<br/>Unconditionally Stable"]
```

## Data Flow through Time Steps

```mermaid
graph LR
    V0["v(t=0)<br/>Initial Hidden State"] --> ODE1["ODE Step 1"]
    ODE1 --> V1["v(t=dt/ode_steps)"]
    V1 --> ODE2["ODE Step 2"]
    ODE2 --> V2["v(t=2*dt/ode_steps)"]
    V2 --> ODE3["ODE Step ..."]
    ODE3 --> VN["v(t=dt)<br/>Final State"]
    
    INPUT["x (constant)<br/>during all steps"] -.->|Used in all steps| ODE1
    INPUT -.-> ODE2
    INPUT -.-> ODE3
```

## Computational Complexity

```mermaid
graph TD
    A["Input: (batch_size, input_dim)"] --> B["Hidden: (batch_size, hidden_dim)"]
    B --> C["Forward Pass Complexity"]
    
    C --> C1["Sensory Processing: O(input_dim * hidden_dim)"]
    C --> C2["ODE Loop iterations: ode_steps times"]
    
    C2 --> C3["Each ODE Step: O(hidden_dim²)"]
    C3 --> C4["Total: O(ode_steps * hidden_dim²)"]
    
    C4 --> D["Output: (batch_size, hidden_dim)"]
```

This diagram shows:
1. **Initialization** of biological parameters using random values
2. **Sensory processing** for input transformation
3. **ODE integration loop** with semi-implicit Euler method
4. **Conductance-based model** for membrane potential updates
5. **Parameter types** and their biological significance
6. **Time-stepping** through ODE integration
7. **Overall data flow** through the network

---

# Complete System Architecture - All Files Integration

## Project-Wide Data Flow

```mermaid
graph TB
    subgraph DATA["Data Processing"]
        D1["Raw ECG Data<br/>MIT-BIH Database"]
        D2["preprocessing.py<br/>Signal Normalization<br/>Window Extraction"]
        D3["Normalized Sequences<br/>Shape: batch x window_size"]
    end
    
    subgraph CONFIG["Configuration"]
        C1["config.py<br/>Hyperparameters<br/>Paths<br/>Training Settings"]
    end
    
    subgraph MODELS["Neural Network Models"]
        M1["ltc_cell.py<br/>BiologicalLTCCell<br/>ODE Integration"]
        M2["attention.py<br/>MultiHeadAttentionBlock<br/>Self-Attention"]
        M3["la_nn.py<br/>BioLANN<br/>Complete Architecture"]
    end
    
    subgraph TRAINING["Training Pipeline"]
        T1["trainer.py<br/>Trainer Class<br/>Training Loop"]
        T2["metrics.py<br/>Performance Metrics<br/>Evaluation Functions"]
    end
    
    subgraph API["Real-time Inference"]
        A1["stream_engine.py<br/>ECGStreamEngine<br/>Real-time Processing"]
        A2["streamlit_app.py<br/>Interactive Dashboard<br/>Visualization"]
    end
    
    subgraph OUTPUT["Output & Storage"]
        O1["Models Directory<br/>la_nn_best.pth"]
        O2["Logs Directory<br/>metrics_*.csv"]
        O3["Interactive UI<br/>Predictions & Charts"]
    end
    
    D1 --> D2
    D2 --> D3
    
    C1 --> D2
    C1 --> MODELS
    C1 --> TRAINING
    
    D3 --> T1
    MODELS --> T1
    
    T1 --> T2
    T2 --> O2
    T1 --> O1
    
    O1 --> API
    MODELS --> API
    A1 --> A2
    A2 --> O3
    
    CONFIG -.->|Provides| API
```

## Detailed File Interaction Diagram

```mermaid
graph LR
    subgraph Entry["Entry Points"]
        MAIN["main.py<br/>Training Pipeline"]
        STREAMLIT["streamlit_app.py<br/>Interactive Demo"]
    end
    
    subgraph CONFIG["Configuration Layer"]
        CFG["config.py<br/>- SAMPLING_RATE: 360<br/>- WINDOW_SIZE: 180<br/>- HIDDEN_DIM: 48<br/>- ODE_STEPS: 6<br/>- NUM_CLASSES: 5"]
    end
    
    subgraph DATAPROC["Data Processing"]
        LOAD["preprocessing.py<br/>load_data()<br/>normalize_signal()"]
        LOADERS["DataLoaders<br/>train/val/test"]
    end
    
    subgraph MODELCORE["Model Architecture"]
        LTC["ltc_cell.py<br/>BiologicalLTCCell"]
        ATT["attention.py<br/>MultiHeadAttentionBlock"]
        LANN["la_nn.py<br/>BioLANN<br/>Combines LTC + LSTM<br/>+ Attention"]
    end
    
    subgraph TRAIN["Training"]
        TRAINER["trainer.py<br/>Trainer Class<br/>- train()<br/>- validate()<br/>- save_checkpoint()"]
    end
    
    subgraph EVAL["Evaluation"]
        METRICS["metrics.py<br/>evaluate_model()<br/>- Accuracy<br/>- Precision<br/>- Recall<br/>- F1-Score"]
    end
    
    subgraph REALTIME["Real-time System"]
        ENGINE["stream_engine.py<br/>ECGStreamEngine<br/>- buffer management<br/>- preprocessing<br/>- inference"]
        APP["streamlit_app.py<br/>- Load model<br/>- Stream visualization<br/>- Real-time prediction"]
    end
    
    subgraph PERSIST["Persistence"]
        MODELS["models_saved/<br/>la_nn_best.pth"]
        LOGS["logs/<br/>metrics_*.csv"]
    end
    
    CFG --> LOAD
    MAIN --> CFG
    STREAMLIT --> CFG
    
    LOAD --> LOADERS
    LOADERS --> TRAINER
    
    CFG --> LTC
    CFG --> ATT
    LTC --> LANN
    ATT --> LANN
    LANN --> TRAINER
    
    TRAINER --> METRICS
    TRAINER --> MODELS
    METRICS --> LOGS
    
    MODELS --> ENGINE
    CFG --> ENGINE
    ENGINE --> APP
    APP --> APP
```

## Data Flow Through Training Pipeline

```mermaid
graph TD
    A["main.py<br/>Program Start"] --> B["Load Config<br/>config.py"]
    B --> C["Ensure Directories<br/>Config.ensure_dirs()"]
    
    C --> D["Load & Preprocess Data<br/>preprocessing.py"]
    D --> E["Create DataLoaders<br/>train/val/test"]
    
    E --> F["Initialize Model<br/>la_nn.py"]
    F --> F1["Create BiologicalLTCCell<br/>ltc_cell.py"]
    F --> F2["Create MultiHeadAttention<br/>attention.py"]
    F1 --> F3["BiologicalLANN<br/>Complete Model"]
    F2 --> F3
    
    F3 --> G["Initialize Trainer<br/>trainer.py"]
    G --> H["Training Loop<br/>for each epoch"]
    
    H --> H1["Forward Pass<br/>Input → LSTM<br/>→ LTC Cell<br/>→ Attention<br/>→ Classification"]
    H1 --> H2["Compute Loss<br/>CrossEntropyLoss"]
    H2 --> H3["Backward Pass<br/>Backpropagation"]
    H3 --> H4["Update Parameters<br/>Adam Optimizer"]
    H4 --> H5["Validate on Val Set<br/>metrics.py"]
    
    H5 --> H6["Save Best Model<br/>models_saved/"]
    H5 --> H7["Log Metrics<br/>logs/"]
    
    H7 --> H8{"More Epochs?"}
    H8 -->|Yes| H
    H8 -->|No| I["Load Best Model"]
    
    I --> J["Evaluate on Test Set<br/>metrics.py"]
    J --> K["Output Results<br/>Accuracy, F1, etc."]
```

## Real-time Inference Data Flow

```mermaid
graph LR
    ECG["ECG Sensor<br/>Real-time Data"] --> STREAM["streamlit_app.py<br/>Streamlit Interface"]
    
    STREAM --> ENGINE["stream_engine.py<br/>ECGStreamEngine"]
    
    ENGINE --> BUFFER["Buffer Management<br/>Sliding Window<br/>WINDOW_SIZE=180"]
    
    BUFFER --> PREPROCESS["Preprocessing<br/>Normalization<br/>normalize_signal()"]
    
    PREPROCESS --> MODEL["Load BioLANN Model<br/>models_saved/la_nn_best.pth"]
    
    MODEL --> INFERENCE["Forward Pass<br/>Get Prediction"]
    
    INFERENCE --> POST["Post-processing<br/>softmax → class label<br/>confidence score"]
    
    POST --> VIZ["Visualization<br/>- ECG Plot<br/>- Class Probability<br/>- Arrhythmia Type<br/>- Timestamps"]
    
    VIZ --> DISPLAY["Display Results<br/>Streamlit UI"]
```

## Component Dependencies Map

```mermaid
graph TD
    MAIN["main.py"]
    STREAM["streamlit_app.py"]
    
    CFG["config.py<br/>Core Configuration"]
    
    MODELS["src/models/"]
    DATA["src/data/"]
    TRAIN["src/training/"]
    EVAL["src/evaluation/"]
    API["src/api/"]
    
    MAIN --> CFG
    MAIN --> DATA
    MAIN --> MODELS
    MAIN --> TRAIN
    MAIN --> EVAL
    
    STREAM --> CFG
    STREAM --> MODELS
    STREAM --> DATA
    STREAM --> API
    
    MODELS --> |ltc_cell.py| MODELS
    MODELS --> |attention.py| MODELS
    MODELS --> |la_nn.py| MODELS
    
    DATA --> |preprocessing.py| DATA
    TRAIN --> |trainer.py| TRAIN
    EVAL --> |metrics.py| EVAL
    API --> |stream_engine.py| API
    
    TRAIN -.->|Uses| MODELS
    TRAIN -.->|Logs to| EVAL
    API -.->|Loads| MODELS
    API -.->|Uses| DATA
```

## Module Responsibilities

```mermaid
graph TD
    subgraph CONFIG["Configuration & Setup"]
        C1["config.py<br/>- Define all hyperparameters<br/>- Set device CPU/GPU<br/>- Define paths<br/>- Training parameters"]
    end
    
    subgraph DATA["Data Layer"]
        D1["preprocessing.py<br/>- load_data: Load MIT-BIH records<br/>- normalize_signal: Z-score norm<br/>- create window: Extract sequences<br/>- return: DataLoaders"]
    end
    
    subgraph MODEL["Model Layer"]
        M1["ltc_cell.py<br/>- BiologicalLTCCell<br/>- ODE integration<br/>- Conductance model<br/>- Parameter initialization"]
        M2["attention.py<br/>- MultiHeadAttentionBlock<br/>- Self-attention mechanism<br/>- Multi-head computation<br/>- Contextualization"]
        M3["la_nn.py<br/>- BioLANN class<br/>- Integrate: LSTM + LTC + Attention<br/>- Bidirectional processing<br/>- Output classification"]
    end
    
    subgraph TRAIN["Training Layer"]
        T1["trainer.py<br/>- Trainer class<br/>- train(): Main loop<br/>- validate(): Val metrics<br/>- save_checkpoint(): Model save<br/>- Handle early stopping"]
    end
    
    subgraph EVAL["Evaluation Layer"]
        E1["metrics.py<br/>- evaluate_model()<br/>- Compute: Accuracy<br/>- Compute: Precision<br/>- Compute: Recall/F1<br/>- Log results"]
    end
    
    subgraph INFERENCE["Inference Layer"]
        I1["stream_engine.py<br/>- ECGStreamEngine<br/>- Buffer: Sliding window<br/>- normalize: Apply norm<br/>- predict: Model inference<br/>- Return: Label + confidence"]
        I2["streamlit_app.py<br/>- Dashboard UI<br/>- Real-time display<br/>- Interactive charts<br/>- User controls"]
    end
    
    C1 -.->|Provides params| DATA
    C1 -.->|Provides params| MODEL
    C1 -.->|Provides params| TRAIN
    
    DATA -->|Feeds| TRAIN
    MODEL -->|Used by| TRAIN
    TRAIN -->|Saves| EVAL
    EVAL -->|Metrics| TRAIN
    
    TRAIN -->|Produces| I1
    MODEL -->|Used by| I1
    DATA -->|normalize| I1
    I1 -->|Inference| I2
    I2 -->|Display| I2
```

## File Interaction Summary

| File | Purpose | Inputs | Outputs | Key Functions |
|------|---------|--------|---------|---|
| **main.py** | Entry point for training | Config | Best model, Metrics | - Orchestrates pipeline |
| **config.py** | Global configuration | - | Hyperparameters | - SAMPLING_RATE, HIDDEN_DIM |
| **preprocessing.py** | Data loading & normalization | Raw ECG data | DataLoaders | load_data(), normalize_signal() |
| **ltc_cell.py** | Biological LTC cell | x, h | Updated hidden state | forward(), __init__() |
| **attention.py** | Multi-head attention | Input sequences | Attention output | forward() |
| **la_nn.py** | Complete model | Input tensor | Class logits | forward(), __init__() |
| **trainer.py** | Training orchestration | Model, DataLoaders | Trained model | train(), validate() |
| **metrics.py** | Evaluation metrics | Predictions, Labels | Metric scores | evaluate_model() |
| **stream_engine.py** | Real-time processing | Raw ECG stream | Predictions | buffer_input(), predict() |
| **streamlit_app.py** | Web UI dashboard | Model, Data | Visual dashboard | Page layouts, Charts |

This architecture ensures:
- **Modularity**: Each file has a clear responsibility
- **Reusability**: Components can be imported and used independently
- **Scalability**: Easy to add new features or modify existing ones
- **Maintainability**: Clear separation of concerns

---

# Individual Component Flow Diagrams

## 1. Attention Module Flow (attention.py)

### Positional Encoding Flow

```mermaid
graph TD
    A["Input Tensor<br/>(batch, seq_len, d_model)"] --> B["PositionalEncoding<br/>__init__ phase"]
    
    B --> B1["Create Position Array<br/>(max_len, 1)<br/>values: 0,1,2,...,max_len"]
    B1 --> B2["Compute Div Term<br/>10000^(2i/d_model)"]
    B2 --> B3["Compute PE Values<br/>Even indices: sin(pos/div_term)<br/>Odd indices: cos(pos/div_term)"]
    B3 --> B4["PE Shape: (1, max_len, d_model)<br/>Register as buffer"]
    
    A --> C["Forward Pass<br/>forward(x)"]
    B4 --> C
    C --> C1["Add Positional Info<br/>x = x + pe[:, :seq_len, :]"]
    C1 --> D["Output<br/>(batch, seq_len, d_model)<br/>with positional info"]
```

### Multi-Head Attention Block Flow

```mermaid
graph TD
    X["Input x<br/>(batch, seq_len, d_model)"] --> A["Add Positional Encoding<br/>PositionalEncoding(x)"]
    
    A --> B["Pre-Norm Attention<br/>Branch 1"]
    B --> B1["Layer Norm<br/>LayerNorm(x)"]
    B1 --> B2["Multi-Head Attention<br/>Query=Norm(x)<br/>Key=Norm(x)<br/>Value=Norm(x)"]
    B2 --> B3["Compute Attention Weights<br/>softmax(QK^T/√d_k)"]
    B3 --> B4["Attention Output<br/>V * weights"]
    B4 --> B5["Dropout & Residual<br/>x = x + dropout(attn_out)"]
    
    B5 --> C["Pre-Norm FFN<br/>Branch 2"]
    C --> C1["Layer Norm<br/>LayerNorm(x)"]
    C1 --> C2["Feed-Forward Network<br/>Linear(d_model, 4*d_model)"]
    C2 --> C3["ReLU Activation<br/>ReLU(hidden)"]
    C3 --> C4["Output Projection<br/>Linear(4*d_model, d_model)"]
    C4 --> C5["Dropout<br/>Dropout(ffn_out)"]
    C5 --> C6["Residual Connection<br/>x = x + ffn_out"]
    
    C6 --> D["Output<br/>(batch, seq_len, d_model)"]
    B3 --> E["Attention Weights<br/>(batch, num_heads, seq_len, seq_len)"]
    D --> F["Return<br/>x, attn_weights"]
    E --> F
```

### Attention Computation Details

```mermaid
graph TD
    Q["Query<br/>(batch, seq_len, d_model)"] --> PROJ["Project Q, K, V<br/>into num_heads"]
    K["Key<br/>(batch, seq_len, d_model)"] --> PROJ
    V["Value<br/>(batch, seq_len, d_model)"] --> PROJ
    
    PROJ --> SPLIT["Split into Heads<br/>(batch*num_heads, seq_len, d_k)"]
    
    SPLIT --> SCORE["Compute Scores<br/>QK^T / √d_k"]
    SCORE --> SOFT["Apply Softmax<br/>softmax(scores)"]
    SOFT --> WEIGHT["Attention Weights<br/>(batch*num_heads, seq_len, seq_len)"]
    
    WEIGHT --> APPLY["Apply to Values<br/>weights @ V"]
    APPLY --> CONCAT["Concatenate Heads<br/>(batch, seq_len, d_model)"]
    CONCAT --> LINEAR["Output Projection<br/>Linear(d_model, d_model)"]
    LINEAR --> OUT["Output<br/>(batch, seq_len, d_model)"]
```

---

## 2. BioLANN Model Flow (la_nn.py)

### Complete BioLANN Forward Pass

```mermaid
graph TD
    INPUT["Input Sequence<br/>(batch, seq_len, input_dim)"] --> INIT["Initialize States"]
    
    INIT --> H_LTC["h_ltc<br/>(batch, hidden_dim)<br/>zeros"]
    INIT --> H_LSTM["h_lstm<br/>(batch, hidden_dim)<br/>zeros"]
    INIT --> C_LSTM["c_lstm<br/>(batch, hidden_dim)<br/>zeros"]
    
    INPUT --> RNN_LOOP["RNN Processing Loop<br/>for t in range(seq_len)"]
    
    RNN_LOOP --> T["Iterate through time steps"]
    
    T --> T1["Extract x_t<br/>x[:, t, :] shape: batch, input_dim"]
    
    T1 --> LSTM["LSTM Step<br/>Mixed Memory Component"]
    LSTM --> LSTM1["LSTMCell Forward<br/>h_lstm, c_lstm = lstm(x_t, h_lstm, c_lstm)"]
    
    LSTM1 --> LTC["LTC Step<br/>Biological Component"]
    LTC --> LTC1["BiologicalLTCCell Forward<br/>h_ltc = ltc_cell(x_t, h_lstm)"]
    LTC1 --> LTC2["ODE Integration<br/>Semi-implicit Euler<br/>6 steps"]
    
    LTC2 --> STORE["Store Output<br/>outputs.append(h_ltc)<br/>shape: (batch, 1, hidden_dim)"]
    
    STORE --> LOOP_CHECK{"More<br/>time steps?"}
    LOOP_CHECK -->|Yes| T1
    LOOP_CHECK -->|No| CONCAT
    
    CONCAT["Concatenate Outputs<br/>rnn_out = cat(outputs)<br/>shape: (batch, seq_len, hidden_dim)"] --> ATT["Attention Block<br/>MultiHeadAttentionBlock"]
    
    ATT --> ATT1["Add Positional Encoding"]
    ATT1 --> ATT2["Pre-Norm Attention<br/>Self-attention on sequence"]
    ATT2 --> ATT3["Pre-Norm FFN<br/>Feed-forward refinement"]
    ATT3 --> ATT_OUT["attn_out, weights<br/>(batch, seq_len, hidden_dim)"]
    
    ATT_OUT --> POOL["Global Aggregation<br/>Average Pooling"]
    POOL --> POOL1["pooled = mean(attn_out, dim=1)<br/>shape: (batch, hidden_dim)"]
    
    POOL1 --> CLASS["Classification Head"]
    CLASS --> CLASS1["Dropout<br/>Dropout(pooled)"]
    CLASS1 --> CLASS2["Linear Layer<br/>Linear(hidden_dim, num_classes)"]
    CLASS2 --> LOGITS["logits<br/>(batch, num_classes)"]
    
    LOGITS --> OUT["Return<br/>logits, attn_weights"]
    ATT_OUT --> OUT
```

### Mixed Memory: LSTM + LTC Integration

```mermaid
graph TD
    X["Input x_t<br/>(batch, input_dim)"]
    H_PREV["Previous States<br/>h_lstm, c_lstm<br/>h_ltc"]
    
    X --> LSTM["LSTM Cell<br/>Update short-term memory"]
    H_PREV --> LSTM
    
    LSTM --> LSTM_OUT["LSTM Output<br/>h_lstm_new, c_lstm_new<br/>Captures temporal<br/>dependencies"]
    
    LSTM_OUT --> LTC["LTC Cell<br/>Update biological state<br/>Uses LSTM output as input"]
    X --> LTC
    
    LTC --> LTC_OUT["LTC Output<br/>h_ltc_new<br/>Biological dynamics<br/>+ temporal info"]
    
    LSTM_OUT -.->|Influences| LTC
    
    LTC_OUT --> FINAL["Final Hidden State<br/>h_ltc_new<br/>Combines:
    - LSTM memory
    - LTC biology
    - Input dynamics"]
```

### Parameter Flow in BioLANN

```mermaid
graph TD
    PARAMS["BioLANN Parameters"] --> LSTM["LSTM Parameters<br/>W_ii, W_if, W_ig, W_io<br/>W_hi, W_hf, W_hg, W_ho<br/>b_i, b_f, b_g, b_o"]
    
    PARAMS --> LTC["LTC Cell Parameters<br/>cm, gleak, vleak<br/>w, sigma, mu, erev<br/>sensory_w, sensory_sigma<br/>sensory_mu, sensory_erev"]
    
    PARAMS --> ATT["Attention Parameters<br/>Q_proj, K_proj, V_proj<br/>output_proj<br/>num_heads weights"]
    
    PARAMS --> CLASS["Classification Parameters<br/>linear weight<br/>linear bias"]
    
    LSTM --> TOTAL["Total Trainable<br/>Parameters"]
    LTC --> TOTAL
    ATT --> TOTAL
    CLASS --> TOTAL
```

### Training Forward-Backward Flow

```mermaid
graph TD
    BATCH["Batch Data<br/>(batch, seq_len, input_dim)"] --> FWD["Forward Pass<br/>BioLANN.forward()"]
    
    FWD --> OUT["Logits<br/>(batch, num_classes)"]
    
    LABELS["Ground Truth<br/>(batch)"] --> LOSS["Loss Computation<br/>CrossEntropyLoss"]
    OUT --> LOSS
    
    LOSS --> LOSS_VAL["Loss Value<br/>scalar"]
    
    LOSS_VAL --> BACKWARD["Backward Pass<br/>loss.backward()"]
    
    BACKWARD --> GRAD["Compute Gradients<br/>for all parameters<br/>∂L/∂param"]
    
    GRAD --> UPDATE["Update Parameters<br/>Adam Optimizer<br/>param = param - lr * ∂L/∂param"]
    
    UPDATE --> NEXT["Next Iteration"]
```

---

## 3. LTC Cell Flow (ltc_cell.py)

### Detailed LTC Computation Flow

```mermaid
graph TD
    INPUT["Input x<br/>(batch, input_dim)"] --> SENSORY["Sensory Processing<br/>Input → Hidden mapping"]
    HIDDEN["Hidden State h<br/>(batch, hidden_dim)"] --> SENSORY
    
    SENSORY --> S1["Apply Softplus<br/>sensory_w_pos = softplus(sensory_w)"]
    S1 --> S2["Compute Gate<br/>sigmoid(sensory_sigma*(x-sensory_mu))"]
    S2 --> S3["Weighted Activation<br/>sensory_w_act = w_pos * gate<br/>(batch, hidden_dim)"]
    
    S3 --> S4["Aggregate Input<br/>sensory_num = sum(w_act*erev, dim=1)"]
    S3 --> S5["Aggregate Conductance<br/>sensory_den = sum(w_act, dim=1)"]
    
    S4 --> SETUP["Setup ODE Parameters"]
    S5 --> SETUP
    
    SETUP --> ST1["cm_t = softplus(cm) / dt_step<br/>Scaled membrane capacitance"]
    SETUP --> ST2["gleak = softplus(gleak)<br/>Leak conductance"]
    SETUP --> ST3["vleak = vleak<br/>Leak potential"]
    
    ST1 --> ODE["ODE Integration Loop<br/>Semi-implicit Euler"]
    ST2 --> ODE
    ST3 --> ODE
    
    ODE --> ODE_ITER["For each ODE step (1 to 6)"]
    
    ODE_ITER --> REC["Recurrent Processing<br/>Hidden → Hidden"]
    REC --> REC1["w_act = softplus(w) * sigmoid(mu, sigma)<br/>(batch, hidden_dim, hidden_dim)"]
    REC1 --> REC2["w_num = sum(w_act*erev) + sensory_num"]
    REC2 --> REC3["w_den = sum(w_act) + sensory_den"]
    
    REC3 --> UPDATE["Semi-implicit Euler Update"]
    UPDATE --> NUM["numerator = cm_t*v_pre + gleak*vleak + w_num"]
    UPDATE --> DEN["denominator = cm_t + gleak + w_den + epsilon"]
    UPDATE --> NEW_V["v_pre_new = numerator / denominator<br/>(batch, hidden_dim)"]
    
    NEW_V --> CHECK{"All ODE steps<br/>complete?"}
    CHECK -->|No| ODE_ITER
    CHECK -->|Yes| OUTPUT["Output<br/>v_pre<br/>(batch, hidden_dim)"]
```

### Conductance Model Components

```mermaid
graph TD
    V["Membrane Potential<br/>v_pre"]
    
    V --> CM["Capacitive Current<br/>I_cap = C_m * dv/dt"]
    V --> GL["Leak Current<br/>I_leak = g_leak * (v - E_leak)"]
    V --> GS["Synaptic Current (Input)<br/>I_sensory = sum(g_s * (v - E_s))"]
    V --> GR["Synaptic Current (Recurrent)<br/>I_recurrent = sum(g_r * (v - E_r))"]
    
    CM --> EQ["Total Current Balance:<br/>I_cap + I_leak + I_sensory + I_recurrent = 0"]
    GL --> EQ
    GS --> EQ
    GR --> EQ
    
    EQ --> SOLVE["Rearrange for v_new:<br/>C_m*v + g_leak*E_leak<br/>+ sum(g_s*E_s) + sum(g_r*E_r)<br/>C_m + g_leak + sum(g_s) + sum(g_r)"]
    
    SOLVE --> FINAL["v_new = numerator / denominator"]
```

### Parameter Initialization Details

```mermaid
graph TD
    RAND["torch.rand(shape)<br/>Uniform [0, 1)"]
    
    RAND --> CM["cm = rand(hidden_dim) * 0.2 + 0.4<br/>Range: [0.4, 0.6]<br/>Membrane Capacitance"]
    RAND --> GL["gleak = rand(hidden_dim) * 0.1 + 0.01<br/>Range: [0.01, 0.11]<br/>Leak Conductance"]
    RAND --> VL["vleak = rand(hidden_dim) * 0.4 - 0.2<br/>Range: [-0.2, 0.2]<br/>Leak Potential"]
    
    RAND --> W["w = rand(hidden_dim, hidden_dim) * 0.1<br/>Range: [0, 0.1]<br/>Recurrent Weights"]
    RAND --> SIGMA["sigma = rand(hidden, hidden) * 5 + 3<br/>Range: [3, 8]<br/>Sensitivity"]
    RAND --> MU["mu = rand(hidden, hidden) * 0.5 + 0.3<br/>Range: [0.3, 0.8]<br/>Threshold"]
    RAND --> EREV["erev = rand(hidden, hidden) * 2 - 1<br/>Range: [-1, 1]<br/>Reversal Potential"]
    
    RAND --> SW["sensory_w = rand(input, hidden) * 0.1"]
    RAND --> SSIG["sensory_sigma = rand(input, hidden) * 5 + 3"]
    RAND --> SMU["sensory_mu = rand(input, hidden) * 0.5 + 0.3"]
    RAND --> SEREV["sensory_erev = rand(input, hidden) * 2 - 1"]
    
    CM --> PARAMS["Initialized Parameters<br/>Ready for training"]
    GL --> PARAMS
    VL --> PARAMS
    W --> PARAMS
    SIGMA --> PARAMS
    MU --> PARAMS
    EREV --> PARAMS
    SW --> PARAMS
    SSIG --> PARAMS
    SMU --> PARAMS
    SEREV --> PARAMS
```

### ODE Integration Stability

```mermaid
graph TD
    EXP["Explicit Euler<br/>v(t+Δt) = v(t) + Δt*f(v,t)"] --> PROB1["Stability Issues<br/>May diverge for<br/>stiff ODEs"]
    
    IMP["Implicit Euler<br/>v(t+Δt) = v(t) + Δt*f(v(t+Δt),t+Δt)"] --> PROB2["Expensive<br/>Requires iteration to solve"]
    
    SEMI["Semi-implicit Euler<br/>(Used in LTC)"] --> ADVAN["Advantages:"]
    
    ADVAN --> A1["✓ Unconditionally stable<br/>Works for stiff ODEs"]
    ADVAN --> A2["✓ Computationally efficient<br/>No iteration needed"]
    ADVAN --> A3["✓ Accurate dynamics<br/>Preserves biological properties"]
    ADVAN --> A4["✓ Handles multi-timescale<br/>processes"]
    
    A1 --> CHOICE["Semi-implicit Euler<br/>is CHOSEN for<br/>LTC Cell"]
    A2 --> CHOICE
    A3 --> CHOICE
    A4 --> CHOICE
```

### Sigmoid Activation in Synaptic Gates

```mermaid
graph TD
    V["Membrane Potential v"]
    MU["Threshold μ"]
    SIGMA["Sensitivity σ"]
    
    V --> COMPUTE["Compute: σ * (v - μ)"]
    MU --> COMPUTE
    SIGMA --> COMPUTE
    
    COMPUTE --> EXPONENT["e^(σ * (v - μ))"]
    
    EXPONENT --> SIGMOID["sigmoid = 1 / (1 + e^-(σ*(v-μ)))<br/>or<br/>sigmoid = e^(σ*(v-μ)) / (1 + e^(σ*(v-μ)))"]
    
    SIGMOID --> GATE["Synaptic Gate<br/>Range: [0, 1]"]
    
    GATE --> INTERP["Interpretation:<br/>- Gate ≈ 0: Synapse closed, no transmission<br/>- Gate ≈ 1: Synapse open, full transmission<br/>- Smooth transition around μ, controlled by σ"]
```

This complete breakdown shows:
1. **Attention**: How positional encoding and multi-head attention work
2. **BioLANN**: How LSTM and LTC integrate, then attention, then classification
3. **LTC**: The biological conductance model, ODE integration, and parameter initialization

