
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
    # 1. Load Data
    train_loader, val_loader, test_loader = load_data(Config)
    
    # 2. Initialize Model
    model = LANN(Config)
    
    # 3. Load Best Model
    model.load_state_dict(torch.load(f"{Config.MODELS_DIR}/la_nn_best.pth"))
    
    # 4. Evaluation
    evaluate_model(model, test_loader, Config)

if __name__ == "__main__":
    main()
