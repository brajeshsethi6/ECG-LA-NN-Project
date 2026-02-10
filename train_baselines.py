
import sys
import os
import torch
from src.config import Config
from src.data.preprocessing import load_data
from src.models import BioLANN, CNN1D, LSTMModel, TransformerBaseline, LNNBaseline
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_model

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def train_baseline(model_name, model_class, config, train_loader, val_loader, test_loader, **model_kwargs):
    print(f"\n" + "="*50)
    print(f"TRAINING BASELINE: {model_name}")
    print("="*50)
    
    model = model_class(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
        **model_kwargs
    )
    
    # Custom model path for baseline
    model_save_name = f"{model_name.lower().replace(' ', '_')}_best.pth"
    model_path = os.path.join(config.MODELS_DIR, model_save_name)
    
    # We need to temporarily monkey-patch or subclass Trainer to save with custom name
    # Or just use the Trainer and rename the file after training.
    # Let's just use the Trainer and rename.
    
    trainer = Trainer(model, config, train_loader, val_loader)
    
    # Adjust epochs for baselines if needed (shorter for quick test, or same for fair comparison)
    # config.NUM_EPOCHS = 10 # Optional: reduce for faster baseline run
    
    trainer.train()
    
    # Rename the saved model to avoid overwriting la_nn_best.pth
    original_best_path = os.path.join(config.MODELS_DIR, "la_nn_best.pth")
    if os.path.exists(original_best_path):
        os.replace(original_best_path, model_path)
    
    print(f"\nEvaluating {model_name}...")
    model.load_state_dict(torch.load(model_path))
    evaluate_model(model, test_loader, config)

def main():
    Config.ensure_dirs()
    train_loader, val_loader, test_loader = load_data(Config)
    
    # List of models to train
    baselines = [
        ("CNN1D", CNN1D, {}),
        ("LSTM", LSTMModel, {"bidirectional": True}),
        ("Transformer", TransformerBaseline, {"num_layers": 2, "num_heads": 4}),
        ("LNN_No_Attention", LNNBaseline, {"ode_steps": 6})
    ]
    
    for name, model_class, kwargs in baselines:
        train_baseline(name, model_class, Config, train_loader, val_loader, test_loader, **kwargs)

if __name__ == "__main__":
    main()
