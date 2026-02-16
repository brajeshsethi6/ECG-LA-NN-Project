# 🩺 LA-NN Project: Code Walkthrough

This document provides a guided tour through the **Liquid-Attention Neural Network (LA-NN)** codebase for real-time ECG analysis. This architecture is designed for high-performance heart rhythm monitoring with a minimal memory footprint, making it ideal for edge devices.

---

## 🏗️ 1. Project Overview & Architecture

The LA-NN project is structured to transition raw ECG signals into clinical insights through a multi-stage pipeline:

1.  **Incoming Signal**: Raw 1D ECG lead data.
2.  **Preprocessing**: Segmentation of signals around R-peaks (AAMI standards).
3.  **LNN Encoder**: Temporal feature extraction using Biophysical Liquid Time-Constant (LTC) neurons.
4.  **Attention Mechanism**: Global context capture to identify long-range irregularities.
5.  **Classifier**: MLP that maps latent features to 5 AAMI heartbeat classes.
6.  **Clinical Decision Logic**: Tiered alerting system based on prediction confidence and class severity.

---

## ⚙️ 2. The Control Center: `src/config.py`

Everything starts with the `Config` class. It centralizes all hyperparameters, directory paths, and clinical thresholds.

-   **Window Size**: 180 samples (90 before R-peak, 90 after).
-   **AAMI Classes**: `N` (Normal), `S` (Supraventricular), `V` (Ventricular), `F` (Fusion), `Q` (Unknown).
-   **ODE Steps**: 6 steps for the Euler integration, balancing stability and speed.
-   **Clinical Thresholds**: `THETA_NORMAL` (0.9) and `THETA_CRITICAL` (0.8) define when an alert is triggered.

---

## 🔍 3. Data Pipeline: `src/data/preprocessing.py`

This module manages the transformation of medical datasets (like MIT-BIH) into tensors ready for deep learning.

-   **Filtering**: High-pass and notch filtering to remove powerline noise and baseline wander.
-   **R-Peak Detection**: Identification of the "QRS complex" to segment individual beats.
-   **Normalization**: Min-Max scaling of signals to ensure gradient stability.
-   **Augmentation**: Adding synthetic noise to improve model robustness.

---

## 💧 4. The Core Model: `src/models/`

### 🧬 `ltc_cell.py`: The Biological Neuron
Unlike traditional RNNs/LSTMs, the `BiologicalLTCCell` implements a **physics-based ODE**.
-   It models the membrane potential ($v$) of a neuron using liquid time constants.
-   **Semi-implicit Euler**: Uses a highly stable mathematical integration method to handle "stiff" differential equations.
-   **Interpretability**: Parameters like `gleak` (leak conductance) and `cm` (capacitance) have direct biological equivalents.

### 🧠 `attention.py`: Global Context
Standard RNNs often forget the beginning of a long sequence.
-   **Multi-Head Attention**: Allows the model to "look back" at the entire heartbeat window simultaneously.
-   It assigns weights to specific parts of the signal (e.g., the P-wave or T-wave) that are most relevant for classification.

### 🔗 `la_nn.py`: The Assembly
The `BioLANN` class brings it all together.
1.  **LNN/LSTM Layer**: Processes the sequence step-by-step.
2.  **Attention Layer**: Re-weights the recurrent states.
3.  **Readout**: A final linear layer outputs probability distributions.

---

## 🚂 5. Training Logic: `src/training/trainer.py`

The `Trainer` class handles the heavy lifting of optimization.
-   **Loss Function**: Cross-Entropy with class balancing (to handle rare heart conditions).
-   **Early Stopping**: Prevents overfitting by monitoring validation accuracy.
-   **Checkpointing**: Saves the best model state to the `models_saved/` directory.

---

## 📊 6. The User Interface: `streamlit_app.py`

The frontend is a sophisticated medical dashboard built with Streamlit.
-   **Live Monitoring**: Simulates a bedside monitor with real-time waveform plotting.
-   **Clinical Reports**: Automatically generates summaries of detected anomalies.
-   **Interpretability View**: Visualizes the Attention weights and LNN state gradients so doctors can see *why* a model made a specific prediction.

---

## 🚨 7. Clinical Decision Logic

Located within `src/evaluation/metrics.py`, this logic converts AI probabilities into actionable alerts:

| Condition | Logic | Action |
| :--- | :--- | :--- |
| **Normal** | $P(N) > THETA\_NORMAL$ | Keep Monitoring |
| **Inconclusive** | $P(N) < THETA\_NORMAL$ but no high anomaly score | Flag for Review |
| **Critical** | $P(V \text{ or } F) > THETA\_CRITICAL$ | **Trigger ALARM** |

---

## 🚀 How to Run the Walkthrough

1.  **View Config**: `cat src/config.py`
2.  **Run Training**: `python main.py`
3.  **Launch Dashboard**: `streamlit run streamlit_app.py`

---
*Created by the LA-NN Team | 2026*
