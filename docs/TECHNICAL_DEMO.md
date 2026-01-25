# LA-NN: Technical Demo Guide

This guide outlines how to perform a full technical demonstration of the **Liquid-Attention Neural Network for ECG Monitoring**.

---

## 1. Demo Preparation

### Installation
Ensure all dependencies are installed and the MIT-BIH dataset is present in `data/mitdb`.
```bash
pip install -r requirements.txt
```

### Model Check
The demo relies on a pre-trained model. Verify that `models_saved/la_nn_best.pth` exists. If not, run a quick training simulation:
```bash
python main.py
```

---

## 2. Part 1: Real-Time Monitoring Demo (Streamlit)
The highlight of the project is the interactive dashboard.

### How to Run:
```bash
streamlit run streamlit_app.py
```

### Presentation Script:
1.  **Overview**: "This is our real-time clinical monitoring dashboard. It simulates a hospital bedside monitor powered by a Liquid Neural Network."
2.  **Record Selection**: Select **Record 200** (Ventricular Tachycardia). This shows the model's ability to detect high-risk anomalies.
3.  **Start Monitoring**: Click **START**. Point out the scrolling cyan trace.
4.  **The 'Liquid' Advantage**: Explain that the LNN is modeling the temporal dynamics of the signal. Contrast it with traditional CNNs that treat ECG like an image.
5.  **Detection Event**: When the trace turns **RED**, highlight the **Level 3: CRITICAL** alert. Show the **Confidence Analysis** bars showing high probability for the 'V' (Ventricular) class.
6.  **Clinical Review**: Scroll down to the "Clinical Review" section to see the historical snapshots of detected anomalies.

---

## 3. Part 2: Custom Data Upload
Showcase flexibility by uploading external data.

1.  In the sidebar, expand **➕ ADD NEW ECG REPORT**.
2.  Upload `sample_ecg_test.csv` (provided in the root).
3.  Select **UPLOADED** from the dropdown and click **START**.
4.  This demonstrates the system's readiness for real-world devices outside the MIT-BIH environment.

---

## 4. Part 3: Professional Diagnostic Report
Finalize the demo by generating a clinical document.

1.  Click **STOP** on the monitoring session.
2.  Click **📄 SAVE CLINICAL REPORT**.
3.  Open the generated HTML file. Show the high-resolution grid, the "Critical Point Analysis", and the professional layout. This proves the system is "End-to-End" from signal to diagnosis.

---

## 5. Technical Highlights (Q&A Prep)
- **Why LNN?**: Because they are computationally efficient (edge-ready) and naturally handle time-series data via Differential Equations.
- **Why Attention?**: To provide a global context of the whole heartbeat, allowing the model to focus on subtle irregularities in the P-wave or QRS complex.
- **Latency**: Inference takes less than 2ms, making it truly real-time.

---
*Technical Demo Assets: LA-NN Project | 2026*
