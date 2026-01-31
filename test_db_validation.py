
import os
import sys
import numpy as np
import torch
import wfdb
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.config import Config
from src.models.la_nn import BioLANN
from src.data.preprocessing import normalize_signal, AAMI_MAPPING

def run_validation(record_name='106'):
    print(f"--- Starting Database Validation for Record {record_name} ---")
    
    # 1. Load Model
    model = BioLANN(
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_classes=Config.NUM_CLASSES,
        ode_steps=Config.ODE_STEPS,
        num_heads=Config.NUM_ATTENTION_HEADS,
        dropout=Config.DROPOUT
    )
    
    # Load weights
    model_path = os.path.join(Config.MODELS_DIR, 'la_nn_best.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(Config.MODELS_DIR, 'la_nn_epoch_10.pth')
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=Config.DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded weights from {model_path}")
    else:
        print("ERROR: No model weights found!")
        return

    model.to(Config.DEVICE).eval()
    
    # 2. Load WFDB Record
    db_dir = os.path.join(Config.DATA_DIR, 'mitdb')
    record_path = os.path.join(db_dir, record_name)
    
    if not os.path.exists(record_path + ".hea"):
        print(f"Downloading record {record_name}...")
        wfdb.dl_database('mitdb', db_dir, records=[record_name])
        
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')
    
    signal = record.p_signal[:, 0] # Lead MLII
    symbols = annotation.symbol
    samples = annotation.sample
    
    # 3. Process and Predict
    half_window = Config.WINDOW_SIZE // 2
    predictions = []
    ground_truth = []
    
    print(f"Processing {len(symbols)} annotations...")
    
    for i in tqdm(range(len(symbols))):
        symbol = symbols[i]
        if symbol in AAMI_MAPPING:
            r_peak = samples[i]
            
            # Boundary check
            if r_peak - half_window < 0 or r_peak + half_window > len(signal):
                continue
                
            # Extract and Normalize
            segment = signal[r_peak - half_window : r_peak + half_window]
            segment = normalize_signal(segment)
            
            # Prepare Tensor
            x = torch.FloatTensor(segment).unsqueeze(0).unsqueeze(-1).to(Config.DEVICE)
            
            # Predict
            with torch.no_grad():
                outputs, _ = model(x)
                pred_idx = torch.argmax(outputs, dim=1).item()
            
            predictions.append(pred_idx)
            ground_truth.append(AAMI_MAPPING[symbol])
            
    # 4. Results
    labels = Config.AAMI_CLASSES
    print("\n" + "="*50)
    print(f"VALIDATION REPORT: Record {record_name}")
    print("="*50)
    
    # Classification Report
    print(classification_report(ground_truth, predictions, target_names=labels, labels=range(len(labels))))
    
    # Overall Accuracy
    correct = (np.array(predictions) == np.array(ground_truth)).sum()
    total = len(ground_truth)
    print(f"Overall Accuracy: {correct/total:.2%} ({correct}/{total})")
    
    # Confusion Matrix
    cm = confusion_matrix(ground_truth, predictions, labels=range(len(labels)))
    print("\nConfusion Matrix:")
    cm_df = pd.DataFrame(cm, index=[f"True {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
    print(cm_df)

if __name__ == "__main__":
    # Test with record 106 which has a lot of Ventricular beats
    run_validation('106')
