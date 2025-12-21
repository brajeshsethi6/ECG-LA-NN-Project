
import torch
import torch.nn as nn
import torch.nn.functional as F

class LiquidTimeConstantCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, ode_steps, tau_min=0.1, tau_max=10.0):
        super(LiquidTimeConstantCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ode_steps = ode_steps
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.dt = 1.0 / ode_steps

        # Linear layers for Tau computation
        self.linear_tau_input = nn.Linear(input_dim, hidden_dim)
        self.linear_tau_hidden = nn.Linear(hidden_dim, hidden_dim)
        
        # Linear layers for f(.) - the nonlinearity
        self.linear_f_input = nn.Linear(input_dim, hidden_dim)
        self.linear_f_hidden = nn.Linear(hidden_dim, hidden_dim)

    def compute_tau(self, x, h):
        # τ = τmin + (τmax - τmin) * σ(Linear(Input) + Linear(Hidden) + Bias)
        gate = self.linear_tau_input(x) + self.linear_tau_hidden(h)
        sig = torch.sigmoid(gate)
        tau = self.tau_min + (self.tau_max - self.tau_min) * sig
        return tau

    def compute_f(self, x, h):
        # f(.) typically involves a tanh or sigmoid activation
        # The thesis just says f(.), assuming standard RNN-like nonlinearity
        return torch.tanh(self.linear_f_input(x) + self.linear_f_hidden(h))

    def ode_step(self, x, h, tau, f_val):
        # dh/dt = (-h + f(.)) / tau
        # Euler integration: h(t+dt) = h(t) + dh/dt * dt
        dh_dt = (-h + f_val) / tau
        h_new = h + dh_dt * self.dt
        return h_new

    def forward(self, x, h_prev=None):
        if h_prev is None:
            h_prev = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        
        h = h_prev
        
        # We assume x is constant over the ODE steps for this time step, 
        # or we integrate the dynamics. 
        # For standard LNN/LTC, we compute params based on current input and state, 
        # then evolve the state.
        
        for _ in range(self.ode_steps):
            # Recalculate Tau and f at each micro-step? 
            # Usually LTC keeps input constant but state evolves.
            # State `h` is evolving.
            
            tau = self.compute_tau(x, h)
            f_val = self.compute_f(x, h)
            h = self.ode_step(x, h, tau, f_val)
            
        return h

class LNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, ode_steps, tau_min=0.1, tau_max=10.0):
        super(LNNEncoder, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            # First layer takes input_dim, subsequent layers take hidden_dim
            in_d = input_dim if i == 0 else hidden_dim
            self.layers.append(
                LiquidTimeConstantCell(in_d, hidden_dim, ode_steps, tau_min, tau_max)
            )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden states for each layer
        hidden_states = [None] * self.num_layers
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            for layer_idx, layer in enumerate(self.layers):
                h_prev = hidden_states[layer_idx]
                h_new = layer(x_t, h_prev)
                hidden_states[layer_idx] = h_new
                x_t = h_new # Feed output of this layer as input to next
            
            outputs.append(x_t.unsqueeze(1))
            
        # Concatenate outputs along time dimension
        # Shape: (batch_size, seq_len, hidden_dim)
        return torch.cat(outputs, dim=1)
