import torch
import torch.nn as nn
import math
from typing import Iterable

class LinearLoRA(nn.Module):
    """
    Wraps a Linear layer with Low-Rank Adaptation (LoRA).
    Forward pass: W'x = W₀x + (scale * (x @ Aᵀ) @ Bᵀ)
    """
    def __init__(self, linear_layer: nn.Linear, rank: int = 4, alpha: float = 16.0):
        super().__init__()
        self.linear = linear_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / max(1, rank)
        
        # Freeze original parameters
        for param in self.linear.parameters():
            param.requires_grad = False

        # Initialize LoRA matrices
        self.A = nn.Parameter(torch.empty(linear_layer.in_features, rank))
        self.B = nn.Parameter(torch.empty(rank, linear_layer.out_features))
        
        self._init_parameters()

    def _init_parameters(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.linear(x)
        lora_output = (x @ self.A) @ self.B
        return base_output + self.scaling * lora_output

def freeze_model(model: nn.Module):
    """Freeze all parameters except input embeddings and output head."""
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    return model

def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 32,
    target_modules: Iterable[str] = ("Linear",),
    # target_modules: Iterable[str] = ("q_proj", "v_proj"),
):
    """
    Recursively replace Linear layers with LoRA-enhanced versions.
    
    Args:
        model: Model to modify
        rank: LoRA rank
        alpha: Scaling factor
        target_modules: Module types to target (substring match)
        verbose: Whether to print replacement info
    """
    for name, child in list(model.named_children()):
        # Process children first
        apply_lora(child, rank, alpha, target_modules)

        # Check if current module should be replaced
        child_type = type(child).__name__
        should_replace = (
            isinstance(child, nn.Linear) and 
            any(target in child_type for target in target_modules)
        )

        if should_replace:
            new_layer = LinearLoRA(child, rank, alpha)
            setattr(model, name, new_layer)
    return model