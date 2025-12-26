# LA-NN Code Deep Dive Demo

This document provides a deep dive into the code structure and logic of the **Liquid Attention Neural Network (LA-NN)** project, specifically focusing on the `LA_NN_Colab.ipynb` notebook. This project is designed for real-time ECG signal analysis using a hybrid architecture.

## 1. Setup and Libraries
The notebook starts by installing necessary dependencies (`wfdb`, `torch`, `tqdm`, etc.) and importing standard libraries for data manipulation and deep learning.

## 2. Configuration (`Config` class)
This section defines the global configuration for the project. Key parameters include:
- **Data Processing**: Sampling rate (360Hz), Window size (180 samples), and AAMI Classes (N, S, V, F, Q).
- **Model Hyperparameters**:
  - `HIDDEN_DIM`: 48
  - `NUM_LNN_LAYERS`: 2
  - `ODE_STEPS`: 6 (Number of internal integration steps for the Liquid Neural Network)
  - `TAU_MIN` / `TAU_MAX`: Range for the time constant $\tau$.
- **Training**: Batch size, Learning rate, and Device selection (CUDA/CPU).
- **Clinical Thresholds**:
  - `THETA_NORMAL` (0.9) and `THETA_CRITICAL` (0.8) are used later in the logic simulation to trigger alarms.

## 3. Model Architecture
The core innovation lies in the hybrid **LA-NN** architecture, combining Liquid Neural Networks (LNN) with Multi-Head Attention.

### A. Liquid Time Constant Cell (`LiquidTimeConstantCell`)
This is the fundamental building block. It models the hidden state dynamics using an Ordinary Differential Equation (ODE):
$$ \frac{dh}{dt} = \frac{-h(t) + f(x(t), h(t))}{\tau(x(t), h(t))} $$
- **`compute_tau`**: Dynamically computes the time constant $\tau$ based on input $x$ and current state $h$.
- **`compute_f`**: Computes the nonlinearity.
- **`ode_step`**: Performs an Euler integration step to update the state $h$.

### B. LNN Encoder (`LNNEncoder`)
This module stacks multiple `LiquidTimeConstantCell` layers.
- It processes the input sequence time-step by time-step.
- The output of one layer flows into the next.
- The final output gives a sequence of hidden states.

### C. Multi-Head Attention Block (`MultiHeadAttentionBlock`)
To capture long-term dependencies in the ECG signal:
- **Positional Encoding**: Adds temporal information to the sequence.
- **Multi-Head Attention**: Allows the model to focus on different parts of the signal simultaneously.
- **Residual Connections & LayerNorm**: Standard Transformer-style blocks for stability.

### D. The Full Model (`LANN`)
Combines the components:
1. **Input** $\to$ **LNN Encoder** (Extracts dynamic features)
2. **LNN Output** $\to$ **Attention Block** (Refines features/context)
3. **Aggregation**: Global Average Pooling triggers a unified feature vector.
4. **Classifier**: A final Linear layer outputs class logits.

## 4. Data Pipeline
The pipeline handles the MIT-BIH Arrhythmia Database.

- **`AAMI_MAPPING`**: Maps the specific MIT-BIH annotation codes to the 5 standard AAMI classes (N, S, V, F, Q).
- **`process_record`**:
  - Reads the signal and annotations.
  - Segments individual heartbeats based on R-peak locations (window centered on peak).
  - Normalizes the signal (Z-score).
- **`load_data`**:
  - Downloads the dataset if missing.
  - Iterates through all records to build the full dataset.
  - **Stratified Split**: Ensures proper class distribution across Train (70%), Validation (15%), and Test (15%) sets.
  - Returns PyTorch `DataLoader` objects.

## 5. Trainer
The `Trainer` class handles the training loop.
- **Loss Function**: `CrossEntropyLoss` with **Weighted Classes**. This is crucial due to the high imbalance in ECG data (Normal beats far outnumber Arrythmias).
- **Optimizer**: `AdamW` with Cosine Annealing scheduler.
- **Gradient Clipping**: Applied to prevent exploding gradients, which can occur in recurrent/ODE systems.
- **Validation**: Tracks accuracy and saves the best model checkpoint.

## 6. Evaluation & Clinical Logic
### Metrics
Standard classification metrics are generated:
- **Classification Report**: Precision, Recall, F1-Score for each class.

### Clinical Decision Logic Simulation
A unique feature of this project is the post-processing logic to simulate a real-world monitoring scenario:
- **Level 1 (Normal)**: If probability of 'N' > `THETA_NORMAL`.
- **Level 3 (Alarm)**: If probability of 'V' or 'F' > `THETA_CRITICAL`.
- **Level 2 (Monitor)**: Default state if neither condition is met.

The code calculates true alarms vs. false alarms to demonstrate clinical utility beyond raw accuracy.

## 7. Main Execution
The `main()` function ties everything together:
1. Sets up directories.
2. Loads data.
3. Initializes the model.
4. Trains the model.
5. Loads the best saved model.
6. Runs final evaluation on the Test set.
