
import torch
import torch.nn as nn
from .ltc_cell import BiologicalLTCCell
from .attention import MultiHeadAttentionBlock

class BioLANN(nn.Module):
    """
    Biological Liquid Attention Neural Network (Bio-LANN).
    
    This model integrates the biophysical LTC cell with Mixed Memory (LSTM)
    and Multi-Head Attention for state-of-the-art sequence classification
    rooted in biological realism.
    """
    def __init__(self, input_dim, hidden_dim, num_classes, 
                 ode_steps=6, num_heads=4, dropout=0.1, mixed_memory=True):
        super(BioLANN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mixed_memory = mixed_memory
        
        # 1. Mixed Memory Component (LSTM-based)
        # handles the dynamic short-term temporal features.
        if self.mixed_memory:
            self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        
        # 2. Biological LTC Cell (The 'Liquid' part)
        self.ltc_cell = BiologicalLTCCell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            ode_steps=ode_steps
        )
        
        # 3. Attention Block
        self.attention = MultiHeadAttentionBlock(
            d_model=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 4. Readout / Classification
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        # Initial states
        h_ltc = torch.zeros(batch_size, self.hidden_dim).to(device)
        if self.mixed_memory:
            h_lstm = torch.zeros(batch_size, self.hidden_dim).to(device)
            c_lstm = torch.zeros(batch_size, self.hidden_dim).to(device)
        
        outputs = []
        
        # RNN Loop
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Step A: Mixed Memory Update (LSTM)
            if self.mixed_memory:
                h_lstm, c_lstm = self.lstm(x_t, (h_lstm, c_lstm))
                # The LTC state can be initialized/influenced by the LSTM 
                # or we can pass the LSTM output as input. 
                # Standard Mixed-Memory NCPS: h_ltc = LTC(inputs, h_lstm)
                h_ltc = self.ltc_cell(x_t, h_lstm)
            else:
                h_ltc = self.ltc_cell(x_t, h_ltc)
                
            outputs.append(h_ltc.unsqueeze(1))
            
        # Combine recurrent outputs: (batch, seq_len, hidden_dim)
        rnn_out = torch.cat(outputs, dim=1)
        
        # Step B: Attention Mechanism
        attn_out, weights = self.attention(rnn_out)
        
        # Step C: Global Aggregation (Average Pooling)
        # Summing or averaging the attended features
        pooled = torch.mean(attn_out, dim=1)
        
        # Step D: Classification
        logits = self.classifier(pooled)
        
        return logits, weights

    def get_biological_params(self):
        """Returns the interpretable parameters of the LTC cell."""
        return {
            "gleak": torch.sigmoid(self.ltc_cell.gleak).detach(),
            "cm": torch.sigmoid(self.ltc_cell.cm).detach(),
            "vleak": self.ltc_cell.vleak.detach()
        }
