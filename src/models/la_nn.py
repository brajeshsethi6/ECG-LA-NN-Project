
import torch
import torch.nn as nn
from .lnn import LNNEncoder
from .attention import MultiHeadAttentionBlock

class LANN(nn.Module):
    def __init__(self, config):
        super(LANN, self).__init__()
        
        self.input_dim = config.INPUT_DIM
        self.hidden_dim = config.HIDDEN_DIM
        self.num_lnn_layers = config.NUM_LNN_LAYERS
        self.ode_steps = config.ODE_STEPS
        self.tau_min = config.TAU_MIN
        self.tau_max = config.TAU_MAX
        self.bidirectional = getattr(config, 'BIDIRECTIONAL', True)
        self.num_heads = config.NUM_ATTENTION_HEADS
        self.num_classes = config.NUM_CLASSES
        self.dropout_rate = config.DROPOUT
        
        # 1. LNN Encoder
        self.lnn_encoder = LNNEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_lnn_layers,
            ode_steps=self.ode_steps,
            tau_min=self.tau_min,
            tau_max=self.tau_max,
            bidirectional=self.bidirectional
        )
        
        # 2. Multi-Head Attention Block
        self.attention_block = MultiHeadAttentionBlock(
            d_model=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_rate
        )
        
        # 3. Classification Head
        # "The final aggregated latent vector is passed through a classification layer"
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # LNN Encoder
        lnn_out = self.lnn_encoder(x) # (batch_size, seq_len, hidden_dim)
        
        # Attention Block
        attn_out, attn_weights = self.attention_block(lnn_out) # (batch_size, seq_len, hidden_dim)
        
        # Global Average Pooling (Aggregation)
        # Check if we should use GAP or specific token. Using GAP is standard for sequence classification without CLS token.
        aggregated_features = torch.mean(attn_out, dim=1) # (batch_size, hidden_dim)
        
        # Classification
        logits = self.classifier(aggregated_features) # (batch_size, num_classes)
        
        return logits, attn_weights

    def get_lnn_taus(self, x):
        """
        Helper method to extract Tau values for interpretability analysis.
        This would require modifying LNNEncoder to return taus.
        Currently a placeholder.
        """
        pass
