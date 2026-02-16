
import os
import numpy as np
import wfdb
import torch
import torch.nn.functional as F
from src.config import Config
from src.data.preprocessing import normalize_signal
import time
import asyncio
from scipy.signal import find_peaks

class ECGStreamEngine:
    def __init__(self, model, record_name='100', custom_signal=None):
        self.model = model
        self.config = Config
        self.record_name = record_name
        self.db_dir = os.path.join(Config.DATA_DIR, 'mitdb')
        
        # AAMI Mapping for ground truth
        self.AAMI_MAPPING = {
            'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',  # Normal
            'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',            # Supraventricular
            'V': 'V', 'E': 'V',                                # Ventricular
            'F': 'F',                                          # Fusion
            '/': 'Q', 'f': 'Q', 'Q': 'Q'                       # Unknown/Paced
        }
        
        if custom_signal is not None:
            self.signal = custom_signal
            self.ann_samples = self._detect_peaks(self.signal)
            self.ground_truth = {}  # No ground truth for custom signals
            self.record_name = "Uploaded CSV"
        else:
            # Load record
            record = wfdb.rdrecord(os.path.join(self.db_dir, self.record_name))
            self.signal = record.p_signal[:, 0]  # MLII channel
            
            # Load annotations for "Ground Truth" markers in demo
            try:
                annotation = wfdb.rdann(os.path.join(self.db_dir, self.record_name), 'atr')
                self.ann_samples = set(annotation.sample)
<<<<<<< HEAD
                # Store ground truth mapping
                from src.data.preprocessing import AAMI_MAPPING
                self.gt_mapping = {
                    samp: sym for samp, sym in zip(annotation.sample, annotation.symbol) 
                    if sym in AAMI_MAPPING
                }
            except Exception:
                # If no annotation found, detect peaks automatically
                self.ann_samples = self._detect_peaks(self.signal)
                self.gt_mapping = {}
=======
                
                # Store ground truth mapping: sample_index -> AAMI class
                self.ground_truth = {}
                for idx, sample in enumerate(annotation.sample):
                    symbol = annotation.symbol[idx]
                    if symbol in self.AAMI_MAPPING:
                        self.ground_truth[sample] = self.AAMI_MAPPING[symbol]
            except Exception:
                # If no annotation found, detect peaks automatically
                self.ann_samples = self._detect_peaks(self.signal)
                self.ground_truth = {}
>>>>>>> d8744a7 (feat: Enhance accuracy tracking and performance display in ECG monitoring app)
        
        self.current_idx = 0
        self.window_size = Config.WINDOW_SIZE
        self.half_window = self.window_size // 2
        
        # Reverse mapping for display
        self.classes = Config.AAMI_CLASSES
        
    async def stream_samples(self, bundle_size=15):
        """
        Simulates real-time streaming of ECG samples in bundles.
        Yields: (list_of_samples, prediction_if_any)
        """
        fs = 360
        bundle_interval = bundle_size / fs  # Time for one bundle
        
        while self.current_idx < len(self.signal):
            start_time = time.time()
            bundled_samples = []
            final_prediction = None
            
            # Fill bundle
            for _ in range(bundle_size):
                if self.current_idx >= len(self.signal):
                    break
                    
                sample = self.signal[self.current_idx]
                bundled_samples.append(float(sample))
                
                # Check for peak
                if self.current_idx in self.ann_samples:
                    # Run inference on peak
                    pred = self.run_inference(self.current_idx)
                    if pred:
                        final_prediction = pred
                
                self.current_idx += 1
            
            if bundled_samples:
                yield bundled_samples, final_prediction
            
            # Synchronize with real time
            elapsed = time.time() - start_time
            sleep_time = max(0, bundle_interval - elapsed)
            await asyncio.sleep(sleep_time)

    def run_inference(self, r_peak_idx):
        """Runs model inference on the window surrounding r_peak_idx"""
        start = r_peak_idx - self.half_window
        end = r_peak_idx + self.half_window
        
        if start < 0 or end > len(self.signal):
            return None
            
        # 1. Extract and Normalize
        segment = self.signal[start:end]
        segment = normalize_signal(segment)
        
        # 2. Prepare for Model
        # Input shape: (1, seq_len, 1)
        x = torch.FloatTensor(segment).unsqueeze(0).unsqueeze(-1).to(Config.DEVICE)
        
        # 3. Model Forward
        self.model.eval()
        with torch.no_grad():
            outputs, _ = self.model(x)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
        
        # 4. Get ground truth if available
        ground_truth_label = self.ground_truth.get(r_peak_idx, None)
            
<<<<<<< HEAD
        # 4. Result dict
        # Get ground truth if available
        from src.data.preprocessing import AAMI_MAPPING
        gt_symbol = self.gt_mapping.get(r_peak_idx, None)
        gt_class = None
        if gt_symbol:
            # Map MIT-BIH symbols to AAMI class names
            inv_map = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}
            gt_class = inv_map.get(AAMI_MAPPING[gt_symbol])

=======
        # 5. Result dict
>>>>>>> d8744a7 (feat: Enhance accuracy tracking and performance display in ECG monitoring app)
        result = {
            "prediction": self.classes[pred_idx],
            "ground_truth": gt_class,
            "confidence": float(probs[pred_idx]),
            "probabilities": {self.classes[i]: float(probs[i]) for i in range(len(self.classes))},
            "alert_level": self.get_alert_level(probs),
            "ground_truth": ground_truth_label,
            "is_correct": ground_truth_label == self.classes[pred_idx] if ground_truth_label else None
        }
        return result

    def get_alert_level(self, probs):
        """Clinical Decision Logic"""
        # Indices: N=0, S=1, V=2, F=3, Q=4
        p_N = probs[0]
        p_V = probs[2]
        p_F = probs[3]
        
        if p_N > Config.THETA_NORMAL:
            return 1 # Normal
        elif p_V > Config.THETA_CRITICAL or p_F > Config.THETA_CRITICAL:
            return 3 # Critical
        else:
            return 2 # Monitor

    def _detect_peaks(self, signal):
        """Simple peak detector for custom signals without annotations"""
        # Normalize for peak detection
        sig = (signal - np.mean(signal)) / np.std(signal)
        # Focus on R-peaks (positive)
        peaks, _ = find_peaks(sig, height=1.5, distance=180)
        return set(peaks)
