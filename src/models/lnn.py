
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

    def compute_tau(self, x, h, memory=None):
        # τ = τmin + (τmax - τmin) * σ(Linear(Input) + Linear(Hidden) + Bias)
        gate = self.linear_tau_input(x) + self.linear_tau_hidden(h)
        if memory is not None:
            gate = gate + memory
        sig = torch.sigmoid(gate)
        tau = self.tau_min + (self.tau_max - self.tau_min) * sig
        return tau

    def compute_f(self, x, h, memory=None):
        # f(.) typically involves a tanh or sigmoid activation
        gate = self.linear_f_input(x) + self.linear_f_hidden(h)
        if memory is not None:
            gate = gate + memory
        return torch.tanh(gate)

    def ode_step(self, x, h, tau, f_val):
        # dh/dt = (-h + f(.)) / tau
        # Euler integration: h(t+dt) = h(t) + dh/dt * dt
        dh_dt = (-h + f_val) / tau
        h_new = h + dh_dt * self.dt
        return h_new

    def forward(self, x, h_prev=None, memory=None):
        if h_prev is None:
            h_prev = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        
        h = h_prev
        
        for _ in range(self.ode_steps):
            tau = self.compute_tau(x, h, memory=memory)
            f_val = self.compute_f(x, h, memory=memory)
            h = self.ode_step(x, h, tau, f_val)
            
        return h

class LNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, ode_steps, tau_min=0.1, tau_max=10.0, bidirectional=True):
        super(LNNEncoder, self).__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        
        # Forward layers
        self.forward_layers = nn.ModuleList()
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            self.forward_layers.append(
                LiquidTimeConstantCell(in_d, hidden_dim, ode_steps, tau_min, tau_max)
            )

        # Backward layers
        if self.bidirectional:
            self.backward_layers = nn.ModuleList()
            for i in range(num_layers):
                in_d = input_dim if i == 0 else hidden_dim
                self.backward_layers.append(
                    LiquidTimeConstantCell(in_d, hidden_dim, ode_steps, tau_min, tau_max)
                )
            
            # Projection to keep hidden_dim consistent if needed, 
            # but usually we just double it. Let's provide a projection layer
            # so the rest of the architecture stays the same.
            self.output_projection = nn.Linear(hidden_dim * 2, hidden_dim)

        # Long-term memory components
        self.memory_alpha = 0.9  # Decay rate for long-term memory
        self.memory_projection = nn.Linear(hidden_dim, hidden_dim)

    def _process_direction(self, x, layers, reverse=False):
        batch_size, seq_len, _ = x.size()
        hidden_states = [None] * self.num_layers
        
        # Initial long-term memory state
        longterm_memory = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        
        outputs = []
        indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
        
        for t in indices:
            x_t = x[:, t, :]
            
            for layer_idx, layer in enumerate(layers):
                h_prev = hidden_states[layer_idx]
                
                # Connect with long-term memory in the forward pass of the cell
                # We use a projected version of the long-term memory
                mem_input = self.memory_projection(longterm_memory) if layer_idx == 0 else None
                
                h_new = layer(x_t, h_prev, memory=mem_input)
                hidden_states[layer_idx] = h_new
                x_t = h_new
            
            # Update long-term memory (simple exponential moving average of the last layer)
            # This captures the "history" up to this point.
            longterm_memory = self.memory_alpha * longterm_memory + (1 - self.memory_alpha) * x_t
            
            outputs.append(x_t.unsqueeze(1))
            
        if reverse:
            outputs = outputs[::-1]
            
        return torch.cat(outputs, dim=1)

    def forward(self, x):
        # Forward pass
        forward_out = self._process_direction(x, self.forward_layers, reverse=False)
        
        if self.bidirectional:
            # Backward pass
            backward_out = self._process_direction(x, self.backward_layers, reverse=True)
            # Concatenate forward and backward outputs
            combined = torch.cat([forward_out, backward_out], dim=-1)
            # Project back to hidden_dim (to maintain compatibility with attention block)
            return self.output_projection(combined)
        
        return forward_out
