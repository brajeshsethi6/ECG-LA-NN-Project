
import torch
from src.models import CNN1D, LSTMModel, TransformerBaseline, LNNBaseline

def test_models():
    batch_size = 4
    seq_len = 180
    input_dim = 1
    num_classes = 5
    hidden_dim = 48
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    models = {
        "CNN1D": CNN1D(input_dim, hidden_dim, num_classes),
        "LSTM": LSTMModel(input_dim, hidden_dim, num_classes),
        "Transformer": TransformerBaseline(input_dim, hidden_dim, num_classes),
        "LNN_Baseline": LNNBaseline(input_dim, hidden_dim, num_classes)
    }
    
    for name, model in models.items():
        print(f"Testing {name}...")
        logits, weights = model(x)
        assert logits.shape == (batch_size, num_classes)
        print(f"{name} output shape: {logits.shape}")
        if weights is not None:
             print(f"{name} weights shape: {weights.shape}")
        else:
             print(f"{name} has no weights output")

if __name__ == "__main__":
    test_models()
    print("All baseline models verified successfully!")
