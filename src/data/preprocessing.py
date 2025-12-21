import os
import numpy as np
import torch
import wfdb
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Mapping from MIT-BIH annotations to AAMI classes
# Class indices: N=0, S=1, V=2, F=3, Q=4
AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal
    'A': 1, 'a': 1, 'J': 1, 'S': 1,          # Supraventricular (S)
    'V': 2, 'E': 2,                          # Ventricular (V)
    'F': 3,                                  # Fusion (F)
    '/': 4, 'f': 4, 'Q': 4                   # Unknown/Paced (Q)
    # Ignored: [, ], !, x, (, ), p, t, u, `, ', ~, +, "
}

class ECGDataset(Dataset):
    def __init__(self, signals, labels):
        """
        Args:
            signals (np.array): Shape (num_samples, seq_len, input_dim)
            labels (np.array): Shape (num_samples,)
        """
        self.signals = torch.FloatTensor(signals)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

def generate_mock_data(num_samples=1000, seq_len=180, input_dim=1, num_classes=5):
    """
    Generates random mock data for testing the pipeline.
    """
    signals = np.random.randn(num_samples, seq_len, input_dim).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=(num_samples,))
    return signals, labels

def normalize_signal(signal):
    """Z-score normalization."""
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:
        return signal - mean
    return (signal - mean) / std

def get_records(db_dir):
    """
    Returns a list of record names from the MIT-BIH Arrhythmia Database.
    Excludes records with significant noise or paced rhythms if necessary, 
    but for now we include the standard set.
    """
    # Standard MIT-BIH records
    records = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '111', 
        '112', '113', '114', '115', '116', '117', '118', '119', '121', '122', '123', '124',
        '200', '201', '202', '203', '205', '207', '208', '209', '210', '212', '213', 
        '214', '215', '219', '220', '221', '222', '223', '228', '230', '231', '232', '233', '234'
    ]
    return records

def process_record(record_name, db_dir, window_size):
    """
    Reads a WFDB record, extracts beats based on annotations, and labels them.
    """
    # Read signal
    record = wfdb.rdrecord(os.path.join(db_dir, record_name))
    signal = record.p_signal[:, 0] # Use lead MLII (usually channel 0)
    
    # Read annotations
    annotation = wfdb.rdann(os.path.join(db_dir, record_name), 'atr')
    symbols = annotation.symbol
    samples = annotation.sample
    
    processed_beats = []
    processed_labels = []
    
    half_window = window_size // 2
    
    for i, symbol in enumerate(symbols):
        if symbol in AAMI_MAPPING:
            r_peak = samples[i]
            
            # Check boundaries
            if r_peak - half_window < 0 or r_peak + half_window > len(signal):
                continue
                
            # Segment
            beat = signal[r_peak - half_window : r_peak + half_window]
            
            # Normalize
            beat = normalize_signal(beat)
            
            processed_beats.append(beat)
            processed_labels.append(AAMI_MAPPING[symbol])
            
    return processed_beats, processed_labels

def load_data(config):
    """
    Main function to load and preprocess data from MIT-BIH.
    """
    print(f"Loading Data from {config.DATA_DIR}...")
    
    db_name = 'mitdb'
    db_dir = os.path.join(config.DATA_DIR, db_name)
    
    # 1. Check if database exists, if not download
    if not os.path.exists(db_dir):
        print(f"Downloading {db_name} to {db_dir}...")
        os.makedirs(db_dir, exist_ok=True)
        wfdb.dl_database(db_name, db_dir)
        print("Download complete.")
    else:
        print(f"Database found at {db_dir}")

    # 2. Process Records
    all_signals = []
    all_labels = []
    
    records = get_records(db_dir)
    print(f"Processing {len(records)} records...")
    
    for rec in tqdm(records):
        try:
            beats, labels = process_record(rec, db_dir, config.WINDOW_SIZE)
            all_signals.extend(beats)
            all_labels.extend(labels)
        except Exception as e:
            print(f"Error processing record {rec}: {e}")
            
    # Convert to numpy
    X = np.array(all_signals)
    y = np.array(all_labels)
    
    # Reshape X to (num_samples, seq_len, input_dim)
    # The current shape is (num_samples, seq_len). 
    # We need to add the feature dimension.
    X = X[..., np.newaxis]
    
    print(f"Total Heartbeats Processed: {X.shape[0]}")
    print(f"Signal Shape: {X.shape}, Label Shape: {y.shape}")
    
    # 3. Stratified Split (Train: 70%, Val: 15%, Test: 15%)
    # First split: Train (70%) vs Temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # Second split: Val (15% of total -> 50% of Temp) vs Test (15% of total -> 50% of Temp)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    print(f"Train/Val/Test Splits: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    
    # 4. Create Datasets and Loaders
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
    test_dataset = ECGDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader
