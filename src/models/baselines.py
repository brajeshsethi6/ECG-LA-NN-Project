
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ltc_cell import BiologicalLTCCell

class CNN1D(nn.Module):
    """Standard 1D Convolutional Neural Network baseline."""
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.1):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.conv3 = nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim * 4)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.pool(x).squeeze(-1)
        logits = self.fc(x)
        return logits, None  # Return None for weights to maintain interface

class LSTMModel(nn.Module):
    """Standard LSTM baseline."""
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.1, bidirectional=True):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=bidirectional)
        
        multiplier = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * multiplier, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Global Average Pooling over time
        pooled = torch.mean(lstm_out, dim=1)
        logits = self.classifier(pooled)
        return logits, None

class TransformerBaseline(nn.Module):
    """Lightweight Transformer baseline."""
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, num_heads=4, dropout=0.1):
        super(TransformerBaseline, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 2, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        trans_out = self.transformer(x)
        # Global Average Pooling
        pooled = torch.mean(trans_out, dim=1)
        logits = self.classifier(pooled)
        return logits, None

class LNNBaseline(nn.Module):
    """Liquid Neural Network (LNN) without Attention."""
    def __init__(self, input_dim, hidden_dim, num_classes, ode_steps=6, dropout=0.1):
        super(LNNBaseline, self).__init__()
        self.hidden_dim = hidden_dim
        self.ltc_cell = BiologicalLTCCell(input_dim, hidden_dim, ode_steps=ode_steps)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        device = x.device
        h = torch.zeros(batch_size, self.hidden_dim).to(device)
        
        outputs = []
        for t in range(seq_len):
            h = self.ltc_cell(x[:, t, :], h)
            outputs.append(h.unsqueeze(1))
            
        rnn_out = torch.cat(outputs, dim=1)
        # Global Average Pooling (instead of attention)
        pooled = torch.mean(rnn_out, dim=1)
        logits = self.classifier(pooled)
        return logits, None
