
import os
import torch

class Config:
    # Project Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
    MODELS_DIR = os.path.join(BASE_DIR, '..', 'models_saved')
    LOGS_DIR = os.path.join(BASE_DIR, '..', 'logs')
    
    # Data Processing
    SAMPLING_RATE = 360
    WINDOW_SIZE = 180  # 90 before, 90 after R-peak
    INPUT_DIM = 1
    NUM_CLASSES = 5
    AAMI_CLASSES = ['N', 'S', 'V', 'F', 'Q']
    
    # Model Hyperparameters
    HIDDEN_DIM = 48
    NUM_LNN_LAYERS = 2
    ODE_STEPS = 6
    NUM_ATTENTION_HEADS = 4
    TAU_MIN = 0.1
    TAU_MAX = 10.0
    BIDIRECTIONAL = True
    DROPOUT = 0.1
    
    # Training Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 15e-4
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 30  # Increased from 1 to 30 for better learning
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Clinical Thresholds (for Decision Logic)
    THETA_NORMAL = 0.9
    THETA_CRITICAL = 0.8  # Critical classes: 'V', 'F'
    
    @staticmethod
    def ensure_dirs():
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        os.makedirs(Config.LOGS_DIR, exist_ok=True)
