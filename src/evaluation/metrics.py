
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import torch.nn.functional as F

def evaluate_model(model, loader, config):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    device = config.DEVICE
    
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            outputs, _ = model(signals)
            probs = F.softmax(outputs, dim=1)
            
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=config.AAMI_CLASSES))
    
    # Clinical/Decision Logic Simulation
    apply_decision_logic(all_probs, all_labels, config)

def apply_decision_logic(probs, true_labels, config):
    """
    Applies the tiered alert system described in the thesis.
    Level 1: Normal (N > theta_normal)
    Level 3: Critical (V or F > theta_critical)
    Level 2: Monitor (Everything else)
    """
    print("\n--- Clinical Decision Logic Simulation ---")
    
    probs = np.array(probs)
    true_labels = np.array(true_labels)
    
    # Class indices
    # ['N', 'S', 'V', 'F', 'Q'] -> [0, 1, 2, 3, 4]
    idx_N = 0
    idx_V = 2
    idx_F = 3
    
    total = len(true_labels)
    alarms_triggered = 0
    correct_alarms = 0
    missed_alarms = 0 # Critical events not alarmed
    
    for i in range(total):
        prob_vector = probs[i]
        true_label = true_labels[i]
        
        # Decision Logic
        action = "MONITOR" # Default Level 2
        
        if prob_vector[idx_N] > config.THETA_NORMAL:
            action = "NORMAL"
        elif (prob_vector[idx_V] > config.THETA_CRITICAL) or (prob_vector[idx_F] > config.THETA_CRITICAL):
            action = "ALARM"
        
        # Validation
        if action == "ALARM":
            alarms_triggered += 1
            if true_label in [idx_V, idx_F]:
                correct_alarms += 1
        
        # Check for missed critical events (False Negatives for Alarm)
        if true_label in [idx_V, idx_F] and action != "ALARM":
            missed_alarms += 1
            
    print(f"Total Samples: {total}")
    print(f"Total Alarms Triggered: {alarms_triggered}")
    print(f"True Alarms (Correctly identified V/F): {correct_alarms}")
    print(f"False Alarms: {alarms_triggered - correct_alarms}")
    print(f"Missed Critical Events (Critical FN): {missed_alarms}")
