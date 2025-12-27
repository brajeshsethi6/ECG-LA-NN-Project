
import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import sys
import torch
import torch.nn.functional as F
import asyncio
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.config import Config
    from src.models.la_nn import LANN
    from src.api.stream_engine import ECGStreamEngine
    from src.data.preprocessing import normalize_signal
except ImportError:
    st.error("Could not load project modules. Ensure you are running from the project root.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="LA-NN Clinical Monitoring System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Cyberpunk/Clinical Theme ---
st.markdown("""
    <style>
        .stApp {
            background-color: #050510;
            color: #e0e0e0;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0c0c1e;
            border-right: 1px solid #1f1f3e;
        }
        
        /* Metric Card Styling */
        .metric-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            border-color: #00f2ff;
            background: rgba(0, 242, 255, 0.05);
        }
        .metric-label {
            font-size: 10px;
            font-weight: 700;
            color: #8888aa;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 8px;
        }
        .metric-value {
            font-size: 32px;
            font-weight: 800;
            font-family: 'OCR A Std', monospace;
        }
        
        /* Status Text Colors */
        .text-normal { color: #3eff8b; text-shadow: 0 0 10px rgba(62, 255, 139, 0.5); }
        .text-monitor { color: #ffeb3e; text-shadow: 0 0 10px rgba(255, 235, 62, 0.5); }
        .text-critical { color: #ff3e3e; text-shadow: 0 0 10px rgba(255, 62, 62, 0.5); }
        
        /* LCD/Glow Effects */
        .lcd-glow {
            font-family: 'Courier New', monospace;
            color: #00f2ff;
            text-shadow: 0 0 15px rgba(0, 242, 255, 0.8);
        }
        
        /* Scrolling Log */
        .log-container {
            background: #000;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 10px;
            font-family: 'Consolas', monospace;
            font-size: 11px;
            height: 250px;
            overflow-y: auto;
        }
        
        /* Titles */
        h1, h2, h3 {
            font-family: 'Inter', sans-serif;
            letter-spacing: -0.5px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Resource Loading ---
@st.cache_resource
def load_model():
    Config.ensure_dirs()
    model = LANN(Config)
    model_path = os.path.join(Config.MODELS_DIR, 'la_nn_best.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    return model.to(Config.DEVICE).eval()

# Execute loading
model_res = load_model()

# --- State Management ---
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.running = False
    st.session_state.history = []
    st.session_state.ecg_buffer = []
    st.session_state.anomalies = [] # Store abnormal segments
    st.toast("✓ AI Model Ready", icon='✅')

# --- Sidebar Controls ---
with st.sidebar:
    st.image("https://img.icons8.com/nolan/256/heart-monitor.png", width=80)
    st.title("LA-NN ECG Monitor")
    st.markdown("Liquid Neural Network Hybrid Monitoring")
    
    st.markdown("---")
    record_map = {
        "100": "Normal Sinus (100)",
        "106": "Ventricular Bigeminy (106)",
        "200": "Ventricular Tachycardia (200)",
        "203": "Complex Arrhythmia (203)"
    }
    selected_record = st.selectbox("Select Patient Record", list(record_map.keys()), format_func=lambda x: record_map[x])
    
    col_btn1, col_btn2 = st.columns(2)
    if col_btn1.button("START", use_container_width=True, type="primary"):
        st.session_state.running = True
        st.session_state.history = []
        st.session_state.ecg_buffer = []
        
    if col_btn2.button("STOP", use_container_width=True):
        st.session_state.running = False

    st.markdown("---")
    st.markdown("### Model Config")
    st.code(f"""
ODE Steps: {Config.ODE_STEPS}
Tau Range: {Config.TAU_MIN}-{Config.TAU_MAX}
Device: {Config.DEVICE}
Latency: < 2ms
    """)

# --- Main Layout ---
# Header with extra styling
st.markdown(f"""
    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;'>
        <h2 style='margin:0;'>Patient Monitor: <span class='lcd-glow'>#{selected_record}</span></h2>
        <div style='background: rgba(0,255,100,0.1); padding: 5px 15px; border-radius: 20px; border: 1px solid rgba(0,255,100,0.3); font-size: 10px; font-weight: bold; color: #3eff8b;'>
            SYSTEM STATUS: ONLINE
        </div>
    </div>
""", unsafe_allow_html=True)

# Top row metrics
m_col1, m_col2, m_col3 = st.columns(3)

hr_placeholder = m_col1.empty()
state_placeholder = m_col2.empty()
alert_placeholder = m_col3.empty()

# ECG Plot
st.markdown("### Real-time ECG Trace")
plot_placeholder = st.empty()

# Bottom analysis row
b_col1, b_col2 = st.columns([1, 1.5])

with b_col1:
    st.markdown("### Confidence Analysis")
    conf_placeholder = st.empty()

with b_col2:
    st.markdown("### Diagnosis Log")
    log_placeholder = st.empty()

# --- Monitoring Loop ---
if st.session_state.running:
    engine = ECGStreamEngine(model_res, record_name=selected_record)
    
    # Constants
    FS = 360
    BUNDLE_SIZE = 15
    MAX_BUFFER = 1000 # Visible window
    
    current_idx = 0
    trace_color = "#00f2ff" # Default cyan
    
    # Simple HR calculation (demo only)
    hr_val = 72
    
    while st.session_state.running:
        # Get data from engine's signal
        end_idx = current_idx + BUNDLE_SIZE
        if end_idx >= len(engine.signal):
            st.session_state.running = False
            st.info("End of record reached.")
            break
            
        samples = engine.signal[current_idx : end_idx]
        st.session_state.ecg_buffer.extend(samples.tolist())
        
        if len(st.session_state.ecg_buffer) > MAX_BUFFER:
            st.session_state.ecg_buffer = st.session_state.ecg_buffer[-MAX_BUFFER:]
            
        # Check for R-peaks and Run Inference
        prediction_result = None
        for i in range(current_idx, end_idx):
            if i in engine.ann_samples:
                res = engine.run_inference(i)
                if res:
                    prediction_result = res
                    # Randomize HR slightly for demo effect on detections
                    hr_val = np.random.randint(68, 76) if prediction_result['prediction'] == 'N' else np.random.randint(90, 110)
                    
                    # Update Trace Color based on status
                    if prediction_result['prediction'] != 'N':
                        trace_color = "#ff3e3e" # Red for abnormality
                        # "Push" for analysis: Capture segment
                        anomaly_segment = engine.signal[max(0, i-180) : min(len(engine.signal), i+180)]
                        st.session_state.anomalies.append({
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "type": prediction_result['prediction'],
                            "conf": prediction_result['confidence'],
                            "data": anomaly_segment.tolist()
                        })
                        # Keep only last 50 anomalies to capture a full record's worth of data
                        if len(st.session_state.anomalies) > 50:
                            st.session_state.anomalies.pop(0)
                    else:
                        trace_color = "#00f2ff" # Reset to Cyan for Normal
                    
                    break
        
        # --- UI Updates ---
        
        # 1. Metrics
        hr_placeholder.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Heart Rate</div>
                <div class='metric-value lcd-glow'>{hr_val} <span style='font-size:14px; color:#666;'>BPM</span></div>
            </div>
        """, unsafe_allow_html=True)
        
        p_class = "text-normal"
        p_label = "NORMAL"
        a_class = "text-normal"
        a_label = "LEVEL 1: STABLE"
        
        if prediction_result:
            pred = prediction_result['prediction']
            if pred == 'V':
                p_label = "VENTRICULAR"
                p_class = "text-critical"
            elif pred == 'F':
                p_label = "FUSION"
                p_class = "text-monitor"
            elif pred == 'N':
                p_label = "NORMAL"
                p_class = "text-normal"
            else:
                p_label = "ARRHYTHMIA"
                p_class = "text-monitor"
                
            level = prediction_result['alert_level']
            if level == 1:
                a_label = "LEVEL 1: STABLE"
                a_class = "text-normal"
            elif level == 2:
                a_label = "LEVEL 2: MONITOR"
                a_class = "text-monitor"
            else:
                a_label = "LEVEL 3: CRITICAL"
                a_class = "text-critical"
                
            # Log history
            ts = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{ts}] {pred} detected (Conf: {prediction_result['confidence']:.2%})"
            st.session_state.history.insert(0, log_entry)
            
            # Confidence bars
            with conf_placeholder.container():
                for k, v in prediction_result['probabilities'].items():
                    color = "green" if k == 'N' else "red" if k in ['V', 'F'] else "blue"
                    st.write(f"**{k} Class Confidence**")
                    st.progress(v)
                    st.markdown(f"<p style='font-size:10px; margin-top:-10px;'>Probability: {v:.4f}</p>", unsafe_allow_html=True)
        
        state_placeholder.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Model Prediction</div>
                <div class='metric-value {p_class}'>{p_label}</div>
            </div>
        """, unsafe_allow_html=True)
        
        alert_placeholder.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Alert Status</div>
                <div class='metric-value {a_class}'>{a_label}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # 2. Plotting (Using Plotly for dynamic coloring)
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=st.session_state.ecg_buffer, 
            mode='lines',
            line=dict(color=trace_color, width=2),
            name='ECG Lead MLII'
        ))
        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', range=[-2, 3])
        )
        plot_placeholder.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # 3. History
        with log_placeholder.container():
            st.markdown(f"<div class='log-container'>{'<br>'.join(st.session_state.history[:20])}</div>", unsafe_allow_html=True)
            
        current_idx += BUNDLE_SIZE
        time.sleep(BUNDLE_SIZE / FS)
        
else:
    # Idle State
    hr_placeholder.markdown("<div class='metric-card'><div class='metric-label'>Heart Rate</div><div class='metric-value'>--</div></div>", unsafe_allow_html=True)
    state_placeholder.markdown("<div class='metric-card'><div class='metric-label'>Model Prediction</div><div class='metric-value'>IDLE</div></div>", unsafe_allow_html=True)
    alert_placeholder.markdown("<div class='metric-card'><div class='metric-label'>Alert Status</div><div class='metric-value'>STANDBY</div></div>", unsafe_allow_html=True)
    
    st.info("👈 Select a record and click 'START' to begin real-time clinical monitoring.")

# --- Clinical Analysis Section (Anomalies) ---
if st.session_state.anomalies:
    st.markdown("---")
    st.markdown(f"### 🔍 Clinical Review: All Detected Abnormalities ({len(st.session_state.anomalies)})")
    
    # Display all anomalies in a 3-column grid
    num_anomalies = len(st.session_state.anomalies)
    for i in range(0, num_anomalies, 3):
        cols = st.columns(3)
        for j in range(3):
            idx = i + j
            if idx < num_anomalies:
                anomaly = st.session_state.anomalies[idx]
                with cols[j]:
                    st.markdown(f"**{anomaly['type']} Class** ({idx+1}) | {anomaly['time']}")
                    import plotly.express as px
                    fig_anom = px.line(anomaly['data'], height=150)
                    fig_anom.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        paper_bgcolor='rgba(255,0,0,0.05)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis_visible=False,
                        yaxis_visible=False
                    )
                    fig_anom.update_traces(line_color='#ff3e3e')
                    st.plotly_chart(fig_anom, use_container_width=True, key=f"anom_{idx}")
                    st.caption(f"Confidence: {anomaly['conf']:.2%}")
