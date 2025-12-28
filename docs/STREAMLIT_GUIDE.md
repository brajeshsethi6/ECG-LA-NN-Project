# LA-NN Streamlit Dashboard Guide 🚀

This is the premium Streamlit version of the clinical ECG monitoring system. It provides a more integrated, single-page experience for demonstrating the model's real-time capabilities.

## 1. How to Run

Ensure you have installed the new dependencies:
```bash
pip install streamlit pandas plotly streamlit-echarts
```

Run the Streamlit application:
```bash
streamlit run streamlit_app.py
```

The dashboard will open automatically in your browser (usually at `http://localhost:8501`).

## 2. Key Features

- **Cyberpunk Clinical Theme**: Custom dark-mode UI with "LCD glow" effects and glassmorphism.
- **Adaptive Metrics**: Heart Rate, Model Prediction, and Alert Status update in real-time as heartbeats are detected.
- **Streaming ECG Trace**: Fast, responsive plotting of the MLII lead signal.
- **Deep Analysis**:
    - **Confidence Analysis**: Live progress bars showing the model's class probabilities (N, S, V, F, Q).
    - **Diagnosis Log**: A scrolling, terminal-style log of all clinical events.
- **Patient Database**: Easily switch between normal and arrhythmia records (100, 106, 200, 203).

## 3. Demo Steps

1. **Select Record 100**: Click **START**. Observe the stable "Normal" state.
2. **Switch to Record 200**: Click **START**. Notice the "Level 3: Critical" alarm and the "Ventricular" classification.
3. **Analyze Confidence**: Point out how the 'V' bar dominates when the arrhythmia appears.
4. **Technical Detail**: Mention the LA-NN architecture (ODE Steps and Tau parameters) in the sidebar.

---
*Created for the LA-NN Deep Dive Demo.*
