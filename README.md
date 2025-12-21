
# Real-Time ECG Signal Analysis Using Liquid-Attention Neural Network (LA-NN)

## Project Overview
This project implements the **LA-NN** architecture, a hybrid model combining **Liquid Neural Networks (LNNs)** and **Multi-Head Self-Attention**, designed for real-time cardiac anomaly detection on edge devices.

## Project Structure
The project follows a modular deployment-ready structure:

```
LA-NN-Project/
├── src/
│   ├── config.py          # Centralized configuration (Hyperparameters, Paths)
│   ├── models/            # Neural Network Modules
│   │   ├── lnn.py         # Liquid Time-Constant (LTC) Cells & Encoder
│   │   ├── attention.py   # Multi-Head Attention & Positional Encoding
│   │   └── la_nn.py       # Main LA-NN Assembly
│   ├── data/              # Data Pipeline
│   │   └── preprocessing.py # Mock Data Generator & WFDB Integration Logic
│   ├── training/          # Training Loop
│   │   └── trainer.py     # Trainer Class
│   └── evaluation/        # Validation & Metrics
│       └── metrics.py     # Classification metrics & Clinical Decision Logic
├── main.py                # Entry point
└── requirements.txt       # Dependencies
```

## Setup & Usage

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Training (Simulation):**
   ```bash
   python main.py
   ```
   *Note: Currently runs with mock data. To use real data, update `src/data/preprocessing.py` to point to your MIT-BIH dataset path.*

## Architecture Details
- **LNN Encoder:** 2 Layers, Hidden Dim=48, ODE Steps=3.
- **Attention:** 4 Headers, Pre-Norm.
- **Optimization:** Designed for low latency and small memory footprint fitting edge constraints.

## Clinical Decision Logic
The system implements a tiered alert mechanism:
- **Normal:** High confidence in 'N' class.
- **Monitor:** Low confidence or non-critical anomalies.
- **Alarm:** Critical confidence in 'V' (Ventricular) or 'F' (Fusion) beats.
