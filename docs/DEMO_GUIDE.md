
# LA-NN Real-time Demo Guide 🚀

This system demonstrates the clinical alerting capabilities of your Liquid Neural Network (LA-NN) architecture.

## 1. Prerequisites
Ensure you have installed the new dependencies:
```bash
pip install -r requirements.txt
```

## 2. Running the Demo
1.  **Start the Backend**:
    Run the following command in your terminal:
    ```bash
    python app.py
    ```
    This will start the FastAPI server at `http://localhost:8000`.

2.  **Open the Monitor**:
    Navigate to the `frontend/` folder and open `index.html` in any modern web browser (Chrome/Edge recommended).
    - Or use a Live Server extension.

## 3. Demo Storyboard (The "Professor" Walkthrough)

### Phase 1: Normal Stabilization
- Select **Record 100** and click **START MONITOR**.
- Point out the **Green Status** ("Level 1: Stable").
- Show the heart rate (around 70-80 BPM).
- **Explanation**: "The LA-NN is processing temporal windows of heartbeats. Because it's a Liquid Neural Network, it tracks the 'state' of the signal dynamically."

### Phase 2: Arrhythmia Detection
- Switch to **Record 106** or **200**.
- Watch for the **Red Alert** ("Level 3: Critical").
- Show the **Diagnosis History** updating in real-time.
- **Key Technical Note**: "Observe the model confidence bars on the left. Notice how the 'V' (Ventricular) probability spikes instantly when the waveform widens."

### Phase 3: The Deep Dive
- Talk about the **ODE Steps** (6) and **Tau Range** (0.1 - 10.0) shown in the footer.
- Explain that this isn't just a CNN; it's a model that understands the physical dynamics of heart electricity.

## 4. Troubleshooting
- **No data?** Ensure `app.py` is running and the record exists in your `data/mitdb/` folder.
- **Disconnected?** Check if the browser block WebSocket connections (unlikely for localhost).
