
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiologicalLTCCell(nn.Module):
    """
    A Liquid Time-Constant (LTC) cell based on the original biological 
    conductance-based model from the NCPS toolkit.
    
    Features:
    - Physical ODE: C_m, G_leak, E_rev
    - Semi-implicit Euler integration (Very Stable)
    - Conductance-based synaptic interaction
    """
    def __init__(self, input_dim, hidden_dim, ode_steps=6, epsilon=1e-8):
        super(BiologicalLTCCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ode_steps = ode_steps
        self.epsilon = epsilon

        # 1. Biological Parameters
        # Membrane Capacitance (C_m)
        self.cm = nn.Parameter(torch.rand(hidden_dim) * 0.2 + 0.4)
        # Leak Conductance (G_leak)
        self.gleak = nn.Parameter(torch.rand(hidden_dim) * 0.1 + 0.01)
        # Leak Reversal Potential (V_leak)
        self.vleak = nn.Parameter(torch.rand(hidden_dim) * 0.4 - 0.2)
        
        # 2. Synaptic Parameters (Hidden-to-Hidden)
        self.w = nn.Parameter(torch.rand(hidden_dim, hidden_dim) * 0.1)
        self.sigma = nn.Parameter(torch.rand(hidden_dim, hidden_dim) * 5 + 3)
        self.mu = nn.Parameter(torch.rand(hidden_dim, hidden_dim) * 0.5 + 0.3)
        self.erev = nn.Parameter(torch.rand(hidden_dim, hidden_dim) * 2 - 1)

        # 3. Sensory Parameters (Input-to-Hidden)
        self.sensory_w = nn.Parameter(torch.rand(input_dim, hidden_dim) * 0.1)
        self.sensory_sigma = nn.Parameter(torch.rand(input_dim, hidden_dim) * 5 + 3)
        self.sensory_mu = nn.Parameter(torch.rand(input_dim, hidden_dim) * 0.5 + 0.3)
        self.sensory_erev = nn.Parameter(torch.rand(input_dim, hidden_dim) * 2 - 1)

    def _sigmoid(self, v_pre, mu, sigma):
        """Standard sigmoid-based synaptic activation function."""
        v_pre = v_pre.unsqueeze(-1) # For broadcasting
        return torch.sigmoid(sigma * (v_pre - mu))

    def forward(self, x, h, dt=1.0):
        """
        Forward pass using Semi-implicit Euler integration.
        x: (batch, input_dim)
        h: (batch, hidden_dim)
        dt: time step
        """
        v_pre = h
        
        # Pre-compute sensory (input) activations
        # We ensure weights are positive using softplus
        sensory_w_act = F.softplus(self.sensory_w) * self._sigmoid(x, self.sensory_mu, self.sensory_sigma)
        # Numerator: Sum(w * E_rev), Denominator: Sum(w)
        sensory_num = torch.sum(sensory_w_act * self.sensory_erev, dim=1)
        sensory_den = torch.sum(sensory_w_act, dim=1)

        # cm_t incorporates the time step relative to ODE unfoldings
        cm_t = F.softplus(self.cm) / (dt / self.ode_steps)
        gleak = F.softplus(self.gleak)

        for _ in range(self.ode_steps):
            # Recurrent activations from current membrane state
            w_act = F.softplus(self.w) * self._sigmoid(v_pre, self.mu, self.sigma)
            
            w_num = torch.sum(w_act * self.erev, dim=1) + sensory_num
            w_den = torch.sum(w_act, dim=1) + sensory_den
            
            # Semi-implicit update formula:
            # v(t+1) = (cm*v(t) + gleak*vleak + sum(w*erev)) / (cm + gleak + sum(w))
            # This is extremely stable and handles stiff ODEs.
            numerator = cm_t * v_pre + gleak * self.vleak + w_num
            denominator = cm_t + gleak + w_den
            
            v_pre = numerator / (denominator + self.epsilon)
            
        return v_pre
