import torch
import torch.nn as nn

class CustomActivation(nn.Module):
    """
    Implements the custom, dynamic activation function:
    f(x, t) = g_i(t) * [ReLU(x + c_i) + alpha_i(t) * tanh(beta * x)]
    """
    def __init__(self, num_neurons, beta=1.0, k=10.0, tau=0.5):
        """
        Initializes the custom activation layer.

        Args:
            num_neurons (int): The number of neurons in the layer this
                               activation function is applied to.
            beta (float): A constant controlling the intensity of the
                          tanh curvature.
            k (float): A constant controlling the steepness of the
                       soft-freezing gate.
            tau (float): The threshold for the soft-freezing gate.
        """
        super().__init__()
        
        # --- Learnable and State Parameters ---
        self.c = nn.Parameter(torch.zeros(num_neurons)) # Learnable bias shift
        self.alpha = nn.Parameter(torch.ones(num_neurons)) # Adaptive curvature
        
        # State for the soft-freezing gate (not trained with gradients)
        self.register_buffer('moving_avg_activity', torch.zeros(num_neurons))
        
        # --- Constants ---
        self.beta = beta
        self.k = k
        self.tau = tau

    def forward(self, x):
        """
        Applies the custom activation function.

        Args:
            x (torch.Tensor): The input tensor from the previous layer.

        Returns:
            torch.Tensor: The activated tensor.
            torch.Tensor: The gate values for diagnostics.
        """
        # --- 1. Update Moving Average of Activity ---
        # This is a simple way to track neuron usage over time.
        # We use a `no_grad` context to ensure this state update does not leak memory.
        with torch.no_grad():
            current_activity = x.mean(dim=[0, 1]) # Average activity across batch and sequence
            self.moving_avg_activity.mul_(0.9).add_(current_activity * 0.1) # Exponential moving average

        # --- 2. Calculate the Soft-Freezing Gate ---
        # g_i(t) = sigma(k * (m_i(t) - tau))
        gate = torch.sigmoid(self.k * (self.moving_avg_activity - self.tau))
        
        # --- 3. Calculate the Main Activation ---
        # ReLU(x + c_i)
        relu_part = torch.relu(x + self.c)
        
        # alpha_i(t) * tanh(beta * x)
        tanh_part = self.alpha * torch.tanh(self.beta * x)
        
        # --- 4. Combine and Apply the Gate ---
        # f(x, t) = g_i(t) * [ ... ]
        activated_x = gate * (relu_part + tanh_part)
        
        return activated_x, gate