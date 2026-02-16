# 🧬 Biological LTC Cell Walkthrough

**File Path:** `src/models/ltc_cell.py`

This file is the mathematical heart of the project. It implements a **Liquid Time-Constant (LTC)** neuron, which is a specialized RNN cell where the time constant $\tau$ is not a fixed number, but a dynamic function of the input.

---

## 🏗️ Initialization (`__init__`)
The cell defines parameters that have direct physical meanings in neurobiology.

| Line | Code | Explanation |
| :--- | :--- | :--- |
| **25** | `self.cm = nn.Parameter(...)` | **Membrane Capacitance**: Determines how much energy the neuron can store before it "fires" or changes state. |
| **27** | `self.gleak = nn.Parameter(...)` | **Leak Conductance**: The passive rate at which the neuron loses charge to return to its resting state. |
| **29** | `self.vleak = nn.Parameter(...)` | **Leak Reversal Potential**: The "Resting Potential"—the voltage the neuron naturally settles to when there is no input. |
| **32-35** | `self.w, self.sigma, self.mu...` | **Synaptic Parameters**: Hidden-to-Hidden weights. These model how neurons "talk" to each other within the layer. |
| **38-41** | `self.sensory_w, sensory_sigma...` | **Sensory Parameters**: Input-to-Hidden weights. These model how external signals (the ECG) "drive" the neuron. |

---

## 🌊 Logic Flow (`forward` pass)

The `forward` pass implements **Semi-Implicit Euler Integration** to solve the Ordinary Differential Equation (ODE) governing the neuron.

### 1. Pre-computing Sensory Activations (Lines 59–62)
Before the ODE loop starts, we calculate how the current ECG sample `x` affects the neuron.
*   **Line 59**: We use `softplus` on the weights to ensure they are positive (conductance cannot be negative in biology).
*   **Lines 61, 62**: We compute `sensory_num` (numerator) and `sensory_den` (denominator) which represent the conductance-based "push" from the ECG input.

### 2. The ODE Solver Loop (Lines 68–81)
Because neurons are continuous-time systems, we "unfold" the math into 6 small steps (`ode_steps`) to ensure numerical stability.

*   **Recurrent Activation (Lines 70–73)**:
    *   The neuron looks at its *current* state `v_pre` to decide how much internal feedback to apply.
    *   `w_num` / `w_den`: Combines the internal recurrent feedback with the external sensory input calculated earlier.
*   **The Semi-Implicit Update (Lines 75–81)**:
    *   **The Formula**: `numerator / denominator`
    *   **Numerator (Line 78)**: `cm_t * v_pre + gleak * self.vleak + w_num`
        *   Contains the "old" memory (`cm_t * v_pre`) and the resting goal (`gleak * vleak`).
    *   **Denominator (Line 79)**: `cm_t + gleak + w_den`
        *   **CRITICAL**: This denominator represents the inverse of the time constant ($1/\tau$). Since `w_den` changes based on the input, this proves the network is "Liquid"—its reaction speed ($1/\tau$) is dynamic!
    *   **Update (Line 81)**: The neuron's state `v_pre` is updated for the next sub-step.

---

## 🛠️ Key Mathematical Features

### Softplus Constraint
You will notice `F.softplus()` used throughout. This maps any real number to $(0, \infty)$. 
*   **Why?** In physics and biology, variables like "Capacitance" and "Conductance" cannot be negative. Softplus enforces these laws of physics during AI training.

### Numerical Stability
The `self.epsilon` (Line 81) prevents "Division by Zero" errors if all conductances were to drop to zero, ensuring the code never crashes during training.

---
**Summary Flow**: 
1. Get Input `x`. 
2. Calculate "Conductance" drive. 
3. Run 6 sub-steps of the Physical ODE. 
4. Update the Neuron's Voltage ($V_m$). 
5. Return the new state.
