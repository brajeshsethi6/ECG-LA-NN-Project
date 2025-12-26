"""
Test Script for LA-NN Model
This script loads the trained model and evaluates it on the test dataset.
"""

import sys
import os

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Config
from src.data.preprocessing import load_data
from src.models.la_nn import LANN
from src.evaluation.metrics import evaluate_model
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import torch.nn.functional as F
from datetime import datetime

def log_test(message, log_file):
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

def test_model():
    """
    Main testing function that:
    1. Loads the trained model
    2. Loads test data
    3. Evaluates performance with detailed metrics
    """
    # 1. Setup
    Config.ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(Config.LOGS_DIR, f'test_results_{timestamp}.log')
    
    log_test("="*60, log_file)
    log_test("LA-NN Model Testing Result", log_file)
    log_test(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_test("="*60, log_file)
    
    log_test(f"\nDevice: {Config.DEVICE}", log_file)
    log_test(f"Model Path: {Config.MODELS_DIR}/la_nn_best.pth", log_file)
    
    # 2. Check if model exists
    model_path = f"{Config.MODELS_DIR}/la_nn_best.pth"
    if not os.path.exists(model_path):
        log_test(f"\n❌ Error: Model file not found at {model_path}", log_file)
        log_test("Please train the model first using main.py", log_file)
        return
    
    # 3. Load Data
    log_test("\n" + "="*60, log_file)
    log_test("Loading Test Data...", log_file)
    log_test("="*60, log_file)
    train_loader, val_loader, test_loader = load_data(Config)
    log_test(f"✓ Test samples loaded: {len(test_loader.dataset)}", log_file)
    
    # 4. Initialize Model
    log_test("\n" + "="*60, log_file)
    log_test("Initializing Model Architecture...", log_file)
    log_test("="*60, log_file)
    model = LANN(Config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_test(f"Total Parameters: {total_params:,}", log_file)
    log_test(f"Trainable Parameters: {trainable_params:,}", log_file)
    
    # 5. Load Trained Weights
    log_test("\n" + "="*60, log_file)
    log_test("Loading Trained Model Weights...", log_file)
    log_test("="*60, log_file)
    try:
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
        log_test("✓ Model weights loaded successfully", log_file)
    except Exception as e:
        log_test(f"❌ Error loading model: {e}", log_file)
        return
    
    model = model.to(Config.DEVICE)
    model.eval()
    
    # 6. Run Evaluation
    log_test("\n" + "="*60, log_file)
    log_test("Running Model Evaluation on Test Set...", log_file)
    log_test("="*60, log_file)
    # Redirect stdout briefly to capture evaluate_model output if possible, 
    # but for simplicity we'll just run it. evaluate_model already prints.
    evaluate_model(model, test_loader, Config)
    
    # 7. Additional Detailed Metrics
    log_test("\n" + "="*60, log_file)
    log_test("Computing Additional Metrics...", log_file)
    log_test("="*60, log_file)
    compute_detailed_metrics(model, test_loader, Config, log_file)
    
    log_test("\n" + "="*60, log_file)
    log_test("Testing Complete! Results saved to " + log_file, log_file)
    log_test("="*60, log_file)

def compute_detailed_metrics(model, loader, config, log_file):
    """
    Compute additional detailed metrics and save to log.
    """
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
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Confusion Matrix
    log_test("\nConfusion Matrix:", log_file)
    cm = confusion_matrix(all_labels, all_preds)
    log_test(f"Classes: {config.AAMI_CLASSES}", log_file)
    log_test(str(cm), log_file)
    
    # Per-class accuracy
    log_test("\nPer-Class Accuracy:", log_file)
    for i, class_name in enumerate(config.AAMI_CLASSES):
        class_mask = all_labels == i
        if class_mask.sum() > 0:
            class_acc = (all_preds[class_mask] == all_labels[class_mask]).mean()
            log_test(f"  {class_name}: {class_acc:.4f} ({class_mask.sum()} samples)", log_file)
    
    # Overall Accuracy
    overall_acc = (all_preds == all_labels).mean()
    log_test(f"\nOverall Accuracy: {overall_acc:.4f}", log_file)
    
    # ROC-AUC (One-vs-Rest)
    try:
        log_test("\nROC-AUC Scores (One-vs-Rest):", log_file)
        for i, class_name in enumerate(config.AAMI_CLASSES):
            if i in all_labels:
                binary_labels = (all_labels == i).astype(int)
                class_probs = all_probs[:, i]
                auc = roc_auc_score(binary_labels, class_probs)
                log_test(f"  {class_name}: {auc:.4f}", log_file)
            else:
                log_test(f"  {class_name}: N/A (no samples in test set)", log_file)
        
        auc_macro = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        log_test(f"\nMacro-Average ROC-AUC: {auc_macro:.4f}", log_file)
    except Exception as e:
        log_test(f"Could not compute ROC-AUC: {e}", log_file)
    
    # Critical vs Non-Critical Performance
    log_test("\n" + "-"*60, log_file)
    log_test("Critical Event Detection Performance:", log_file)
    log_test("-"*60, log_file)
    
    critical_indices = [2, 3]  
    critical_mask = np.isin(all_labels, critical_indices)
    non_critical_mask = ~critical_mask
    
    if critical_mask.sum() > 0:
        critical_recall = (all_preds[critical_mask] == all_labels[critical_mask]).mean()
        log_test(f"Critical Event Recall: {critical_recall:.4f}", log_file)
        log_test(f"Critical Events in Test Set: {critical_mask.sum()}", log_file)
    
    if non_critical_mask.sum() > 0:
        non_critical_acc = (all_preds[non_critical_mask] == all_labels[non_critical_mask]).mean()
        log_test(f"Non-Critical Accuracy: {non_critical_acc:.4f}", log_file)
        log_test(f"Non-Critical Events in Test Set: {non_critical_mask.sum()}", log_file)

if __name__ == "__main__":
    test_model()
