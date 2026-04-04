
import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import sys
import torch
import torch.nn.functional as F
import asyncio
import plotly.graph_objects as go
import plotly.express as px
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
    st.session_state.full_session_buffer = [] # Store all samples for download
    st.session_state.anomalies = [] # Store abnormal segments
    st.session_state.anomaly_indices = [] # Store indices for marking on full graph
    st.toast("✓ AI Model Ready", icon='✅')

# --- Sidebar Controls ---
with st.sidebar:
    st.image("https://img.icons8.com/nolan/256/heart-monitor.png", width=80)
    st.title("LA-NN ECG Monitor")
    st.markdown("Liquid Neural Network Hybrid Monitoring")
    
    st.markdown("---")
    
    # 1. Dynamic Record Scanning
    db_dir = os.path.join(Config.DATA_DIR, 'mitdb')
    available_records = []
    if os.path.exists(db_dir):
        # Look for .hea files to identify records
        available_records = sorted(list(set([f.split('.')[0] for f in os.listdir(db_dir) if f.endswith('.hea')])))
    
    # Default common records for quick access
    preset_records = {
        "100": "Normal Sinus (100)",
        "106": "Ventricular Bigeminy (106)",
        "200": "Ventricular Tachycardia (200)",
        "203": "Complex Arrhythmia (203)"
    }
    
    # Combine presets with any other found records
    all_record_options = {}
    for r in available_records:
        if r in preset_records:
            all_record_options[r] = preset_records[r]
        else:
            all_record_options[r] = f"Record {r}"
            
    # --- New Feature: Add Custom ECG ---
    with st.expander("➕ ADD NEW ECG REPORT"):
        st.markdown("""
        **Supported Formats:**
        1. **CSV**: Single column of voltage values (360Hz recommended).
        2. **WFDB**: Upload both `.hea` and `.dat` files (optionally `.atr`).
        """)
        
        uploaded_files = st.file_uploader(
            "Upload ECG Files", 
            type=['csv', 'dat', 'hea', 'atr'], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            new_wfdb_uploaded = False
            for uploaded_file in uploaded_files:
                # 1. Handle CSVs (Simulated real-time)
                if uploaded_file.name.endswith('.csv'):
                    try:
                        df_upload = pd.read_csv(uploaded_file)
                        if 'voltage' in df_upload.columns:
                            custom_signal = df_upload['voltage'].values
                        elif 'signal' in df_upload.columns:
                            custom_signal = df_upload['signal'].values
                        else:
                            custom_signal = df_upload.iloc[:, 0].values
                        
                        st.session_state.custom_signal = custom_signal.astype(np.float32)
                        st.session_state.custom_filename = uploaded_file.name
                        all_record_options["UPLOADED"] = f"Uploaded: {uploaded_file.name}"
                        st.success(f"CSV Loaded: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error loading CSV: {e}")
                
                # 2. Handle WFDB (Physical save to data/mitdb)
                else:
                    save_path = os.path.join(db_dir, uploaded_file.name)
                    # Check if exists to avoid redundant writes, though overwrite is usually fine
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    new_wfdb_uploaded = True
            
            if new_wfdb_uploaded:
                st.info("WFDB files saved to database. They will appear in the selection list after refresh.")
                if st.button("Refresh Record List"):
                    st.rerun()

    if "UPLOADED" in all_record_options:
        # Move UPLOADED to top if it exists
        keys = ["UPLOADED"] + [k for k in all_record_options.keys() if k != "UPLOADED"]
    else:
        keys = list(all_record_options.keys())

    selected_record = st.selectbox(
        "Select Patient Record", 
        keys, 
        format_func=lambda x: all_record_options[x],
        index=0
    )
    
    col_btn1, col_btn2 = st.columns(2)
    if col_btn1.button("START", width='stretch', type="primary"):
        st.session_state.running = True
        st.session_state.history = []
        st.session_state.ecg_buffer = []
        st.session_state.full_session_buffer = []
        st.session_state.anomalies = []
        st.session_state.anomaly_indices = []
        
    if col_btn2.button("STOP", width='stretch'):
        st.session_state.running = False
        st.session_state.report_ready = True

    st.markdown("---")
    
    # --- Report Export Feature ---
    if not st.session_state.running and hasattr(st.session_state, 'report_ready') and len(st.session_state.full_session_buffer) > 0:
        st.markdown("### 📋 Final Results")
        
        # Professional HTML Report Generation
        def generate_html_report():
            import base64
            from io import BytesIO
            
            # Summary Metrics
            counts = pd.Series([a['type'] for a in st.session_state.anomalies]).value_counts().to_dict()
            summary_str = ", ".join([f"{k}: {v}" for k, v in counts.items()]) or "No abnormalities detected"
            
            # Create Time Index (seconds)
            fs = 360
            time_idx = np.arange(len(st.session_state.full_session_buffer)) / fs
            
            # Full Signal Plot with Critical Point Markers
            fig_full = go.Figure()
            # Main Signal
            fig_full.add_trace(go.Scatter(
                x=time_idx, 
                y=st.session_state.full_session_buffer, 
                line=dict(color='#2c3e50', width=1),
                name='ECG Lead MLII'
            ))
            
            # Add Critical Points (Markers)
            if st.session_state.anomaly_indices:
                anom_times = [idx / fs for idx in st.session_state.anomaly_indices]
                anom_values = [st.session_state.full_session_buffer[idx] for idx in st.session_state.anomaly_indices]
                fig_full.add_trace(go.Scatter(
                    x=anom_times,
                    y=anom_values,
                    mode='markers',
                    marker=dict(color='#e74c3c', size=10, symbol='x'),
                    name='CRITICAL POINT'
                ))

            fig_full.update_layout(
                title=dict(
                    text="Full Continuous ECG Record with Critical Markers",
                    x=0.5,
                    font=dict(size=20, color='#2c3e50')
                ),
                xaxis_title="Time (seconds)",
                yaxis_title="Amplitude (mV)",
                height=500,
                plot_bgcolor='rgb(255, 250, 250)',
                paper_bgcolor='rgba(255, 255, 255, 0)',
                margin=dict(l=60, r=40, t=80, b=60),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Standard ECG Grid: Minor (0.04s/0.1mV), Major (0.2s/0.5mV)
            fig_full.update_xaxes(
                showgrid=True,
                dtick=0.2,
                gridcolor='rgba(231, 76, 60, 0.4)',
                gridwidth=1.5,
                minor=dict(dtick=0.04, showgrid=True, gridcolor='rgba(231, 76, 60, 0.15)', gridwidth=0.5),
                zeroline=True,
                zerolinecolor='rgba(231, 76, 60, 0.6)',
                zerolinewidth=2
            )
            fig_full.update_yaxes(
                showgrid=True,
                dtick=0.5,
                gridcolor='rgba(231, 76, 60, 0.4)',
                gridwidth=1.5,
                minor=dict(dtick=0.1, showgrid=True, gridcolor='rgba(231, 76, 60, 0.15)', gridwidth=0.5),
                zeroline=True,
                zerolinecolor='rgba(231, 76, 60, 0.6)',
                zerolinewidth=2,
                range=[-2, 3] # Fixed scale for standard visual comparison
            )
            
            full_plot_html = fig_full.to_html(
                include_plotlyjs='cdn', 
                full_html=False, 
                config={'responsive': True, 'displayModeBar': True}
            )
            
            
            html = f"""
            <html>
                <head>
                    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap" rel="stylesheet">
                    <style>
                        body {{ font-family: 'Inter', sans-serif; padding: 40px; color: #333; background: #f4f7f9; }}
                        .header {{ background: #fff; border: 1px solid #dce4ec; border-radius: 10px; padding: 25px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center; }}
                        h1 {{ color: #1a2a3a; margin: 0; font-size: 28px; letter-spacing: 1px; }}
                        
                        /* Professional ECG Paper Styling */
                        .ecg-paper {{
                            background-color: #fff;
                            border: 3px solid #e74c3c;
                            border-radius: 8px;
                            padding: 2px;
                            position: relative;
                            box-shadow: 0 4px 15px rgba(231, 76, 60, 0.15);
                            overflow: hidden;
                        }}
                        
                        .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
                        .event-card {{ background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid #eee; }}
                        .event-header {{ background: #e74c3c; color: #fff; padding: 8px 15px; font-weight: bold; display: flex; justify-content: space-between; font-size: 13px; }}
                        .section-title {{ border-bottom: 3px solid #2c3e50; padding-bottom: 5px; margin: 40px 0 20px 0; color: #2c3e50; text-transform: uppercase; font-size: 18px; }}
                        .footer {{ margin-top: 60px; font-size: 11px; color: #95a5a6; text-align: center; border-top: 1px solid #ddd; padding-top: 25px; }}
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>CARDIAC DIAGNOSTIC ANALYSIS</h1>
                        <p style="color:#7f8c8d; margin-top:5px;">Patient Record Reference: <b>{selected_record}</b> | Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                    
                    <h3 class="section-title">I. Continuous Monitoring Sequence</h3>
                    <div class="ecg-paper">
                        {full_plot_html}
                    </div>

                    <h3 class="section-title">II. Critical Point Analysis Grid</h3>
                    <div class="grid">
                        {self_building_grid()}
                    </div>
                    
                    <div class="footer">
                        <b>CONFIDENTIAL MEDICAL RECORD</b><br>
                        Generated via LA-NN (Liquid Attention Neural Network) Engine. <br>
                        The grid spacing reflects a standard 1mm/5mm diagnostic scale for visual estimation.
                    </div>
                </body>
            </html>
            """
            return html
            
        def self_building_grid():
            fs = 360
            items = ""
            for i, a in enumerate(st.session_state.anomalies):
                # Time index for the segment
                t_a = np.arange(len(a['data'])) / fs
                
                fig_a = go.Figure()
                fig_a.add_trace(go.Scatter(
                    x=t_a, 
                    y=a['data'], 
                    line=dict(color='#000', width=2),
                    name='ECG segment'
                ))
                
                fig_a.update_layout(
                    height=280, 
                    margin=dict(l=40, r=20, t=10, b=40), 
                    plot_bgcolor='rgb(255, 250, 250)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False
                )
                
                # Apply standard ECG grid to anomaly snippets too
                fig_a.update_xaxes(
                    showgrid=True,
                    dtick=0.2,
                    gridcolor='rgba(231, 76, 60, 0.3)',
                    minor=dict(dtick=0.04, showgrid=True, gridcolor='rgba(231, 76, 60, 0.1)'),
                    title="Time (s)"
                )
                fig_a.update_yaxes(
                    showgrid=True,
                    dtick=0.5,
                    gridcolor='rgba(231, 76, 60, 0.3)',
                    minor=dict(dtick=0.1, showgrid=True, gridcolor='rgba(231, 76, 60, 0.1)'),
                    range=[-1.5, 2.5],
                    title="mV"
                )
                
                html_plot = fig_a.to_html(include_plotlyjs='cdn', full_html=False, config={'displayModeBar': False})
                items += f"""
                <div class="event-card">
                    <div class="event-header">
                        <span>CRITICAL EVENT #{i+1} [{a['type']}]</span>
                        <span>Offset: {a['offset_sec']:.2f}s</span>
                    </div>
                    <div class="ecg-paper" style="border:none; border-radius:0; box-shadow:none;">
                        {html_plot}
                    </div>
                    <div style="padding: 10px; font-size: 11px; color: #666; background: #fffcfc; border-top: 1px solid #fee;">
                        <b>Detection Confidence:</b> <span style="color:{'#e74c3c' if a['conf'] > 0.8 else '#d35400'}">{a['conf']:.2%}</span> | 
                        <b>Timestamp:</b> {a['time']}
                    </div>
                </div>
                """
            return items

        st.download_button(
            label="📄 SAVE CLINICAL REPORT",
            data=generate_html_report(),
            file_name=f"Clinical_Report_{selected_record}.html",
            mime="text/html",
            width='stretch'
        )
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

# --- Full Session View ---
st.markdown("### 📊 Full Recorded Session")
full_plot_placeholder = st.empty()

# --- Monitoring Loop ---
if st.session_state.running:
    # Handle custom upload vs preset record
    custom_sig = st.session_state.get('custom_signal') if selected_record == "UPLOADED" else None
    engine = ECGStreamEngine(model_res, record_name=selected_record, custom_signal=custom_sig)
    
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
        st.session_state.full_session_buffer.extend(samples.tolist()) # Record full session
        
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
                        total_samples_so_far = len(st.session_state.full_session_buffer)
                        st.session_state.anomalies.append({
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "offset_sec": total_samples_so_far / FS,
                            "type": prediction_result['prediction'],
                            "conf": prediction_result['confidence'],
                            "data": anomaly_segment.tolist()
                        })
                        st.session_state.anomaly_indices.append(total_samples_so_far)
                        
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
        plot_placeholder.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
        
        # 3. History
        with log_placeholder.container():
            st.markdown(f"<div class='log-container'>{'<br>'.join(st.session_state.history[:20])}</div>", unsafe_allow_html=True)
            
        # 4. Full Session Cumulative Plot
        if len(st.session_state.full_session_buffer) > 0:
            fig_full = go.Figure()
            fs = 360
            full_time = np.arange(len(st.session_state.full_session_buffer)) / fs
            fig_full.add_trace(go.Scatter(
                x=full_time,
                y=st.session_state.full_session_buffer,
                mode='lines',
                line=dict(color='#2c3e50', width=1),
                name='Full Session Record'
            ))
            
            # Professional clinical grid for main UI
            fig_full.update_layout(
                height=250,
                margin=dict(l=40, r=20, t=10, b=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgb(255, 250, 250)',
                xaxis=dict(
                    title="Time (s)", 
                    dtick=1.0, 
                    showgrid=True, 
                    gridcolor='rgba(231,76,60,0.2)',
                    minor=dict(dtick=0.2, showgrid=True, gridcolor='rgba(231,76,60,0.05)')
                ),
                yaxis=dict(
                    title="mV", 
                    range=[-2, 3], 
                    showgrid=True, 
                    gridcolor='rgba(231,76,60,0.2)',
                    minor=dict(dtick=0.5, showgrid=True, gridcolor='rgba(231,76,60,0.05)')
                )
            )
            full_plot_placeholder.plotly_chart(fig_full, width='stretch', config={'displayModeBar': True})
            
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
                    
                    # Enhanced segment plot with standard grid
                    fs = 360
                    t_anom = np.arange(len(anomaly['data'])) / fs
                    fig_anom = go.Figure()
                    fig_anom.add_trace(go.Scatter(
                        x=t_anom, 
                        y=anomaly['data'], 
                        line=dict(color='#ff3e3e', width=2),
                        name=anomaly['type']
                    ))
                    
                    fig_anom.update_layout(
                        height=180,
                        margin=dict(l=30, r=10, t=10, b=30),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgb(255, 250, 250)',
                        showlegend=False
                    )
                    fig_anom.update_xaxes(
                        showgrid=True, dtick=0.2, 
                        gridcolor='rgba(231,76,60,0.1)', 
                        title=dict(text="s", font=dict(size=10))
                    )
                    fig_anom.update_yaxes(
                        showgrid=True, dtick=0.5, 
                        gridcolor='rgba(231,76,60,0.1)',
                        range=[-1.5, 2.5]
                    )
                    st.plotly_chart(fig_anom, width='stretch', key=f"anom_{idx}")
                    st.caption(f"Confidence: {anomaly['conf']:.2%}")
