
import sys
import os

# Add src to python path to specific modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Config
from src.data.preprocessing import load_data
from src.models.la_nn import LANN
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_model
import torch

def main():
    print("Initializing LA-NN Project...")
    
    # 1. Setup
    Config.ensure_dirs()
    print(f"Device: {Config.DEVICE}")
    
    # 2. Data Loading
    train_loader, val_loader, test_loader = load_data(Config)
    
    # 3. Model Initialization
    model = LANN(Config)
    print("Model Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Trainable Parameters: {total_params}")
    
    # 4. Training
    trainer = Trainer(model, Config, train_loader, val_loader)
    trainer.train()
    
    # 5. Evaluation (Load best model)
    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load(f"{Config.MODELS_DIR}/la_nn_best.pth"))
    evaluate_model(model, test_loader, Config)

if __name__ == "__main__":
    main()
