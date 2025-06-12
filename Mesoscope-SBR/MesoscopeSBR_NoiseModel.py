"""
@Author: Bhupesh BISHNOI
@Project: CNRS MesEnView Computational Imaging Pipeline
@Laboratory: Institute for Research in Geroscience and Rejuvenation (RESTORE) | CNRS UMR5070 | INSERM UMR1301 |
@Laboratory: Clément Ader Institute | Federal University Toulouse Midi-Pyrénées | UMR CNRS 5312 |
@Institute: Centre National de la Recherche Scientifique (CNRS) 
@Institute: Institut National de la Santé et de la Recherche Médicale (INSERM)
@Year: 2024-2025
@License: GNU Lesser General Public License v3.0 (LGPL-3.0)

This block provides author information and licensing details for the code. 
It is intended for internal use within the CNRS institute and adheres to the GNU Lesser General Public License v3.0 (LGPL-3.0).
"""
"""
Applies the Poisson-Gaussian noise model to the given input tensors, `stack` and `rfv`.

The `forward` method modifies both `stack` and `rfv` by adding noise according to the formula:
`forward(x) = x + sqrt(a*x + b) * N(0,1)`, where `a` and `b` are calibrated parameters.

Args:
    stack (Tensor): A tensor representing the noise-free stack of light field views, normalized to [0,1].
    rfv (Tensor): A tensor representing the noise-free refocused volume, normalized to [0,1].

Returns:
    Tuple[Tensor, Tensor]: A tuple containing the modified `stack` and `rfv` tensors, both with Poisson-Gaussian noise added.
"""
from typing import Tuple
from functools import cached_property
import torch
from torch import Tensor
from torch.nn import Module


class PoissonGaussianNoiseModel(Module):
    """Poisson-Gaussian noise model for CM2 sensor: https://ieeexplore.ieee.org/document/4623175"""

    def __init__(self, config: dict):
        """
        Initializes the PoissonGaussianNoiseModel with the specified configuration.

        Args:
            config (dict): A dictionary containing configuration parameters.
                Expected keys:
                - "num_lf_views": The number of light field views (int).
                - "A_MEAN": The mean parameter 'a' used to model the noise (float).
                - "B_MEAN": The mean parameter 'b' used to model the noise (float).
        """
        super().__init__()  # Initialize the parent class (Module)
        self.num_views = config["num_lf_views"]  # Set the number of light field views
        self.a_mean = config.get("A_MEAN")  # Set the mean parameter 'a'
        self.b_mean = config.get("B_MEAN")  # Set the mean parameter 'b'

    @cached_property
    def recip_sqrt_num_views(self) -> torch.Tensor:
        """
        Computes the reciprocal of the square root of the number of views.
        
        Returns:
            torch.Tensor: The computed reciprocal of the square root of num_views.
        """
        return 1 / torch.sqrt(torch.tensor(self.num_views))  # Compute and return the reciprocal of sqrt(num_views)

    def forward(self, stack: Tensor, rfv: Tensor) -> Tuple[Tensor, Tensor]:
        """Applies the Poisson-Gaussian noise model to the given inputs.

        The forward function modifies both `stack` and `rfv` by adding noise
        according to the formula: 
        forward(x) = x + sqrt(a*x + b) * N(0,1), where a and b are calibrated parameters.

        Args:
            stack (Tensor): A tensor representing the noise-free stack of light field views,
                            normalized to [0,1].
            rfv (Tensor): A tensor representing the noise-free refocused volume,
                          normalized to [0,1].

        Note:
            Ideally, backprojection would be performed on the raw light field views with noise added
            to achieve the refocused volume. This implementation simplifies that by adding
            the same noise to both tensors, scaling the standard deviation by the reciprocal of 
            the square root of the number of views. This resembles a "denoising" approach.
        
        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the modified `stack` and `rfv` tensors, 
                                   both with Poisson-Gaussian noise added.
        """
        # Move the reciprocal of sqrt(num_views) to the same device as `stack`
        recip_sqrt_num_views = self.recip_sqrt_num_views.to(stack.device)

        # Add Poisson-Gaussian noise to `stack`
        stack += torch.sqrt(
            torch.clamp(self.a_mean * stack + self.b_mean, min=0)  # Clamp to prevent negative values
        ) * torch.randn(stack.shape).to(stack.device)  # Generate noise and apply it 

        # Add Poisson-Gaussian noise to `rfv`
        rfv += (
            torch.sqrt(torch.clamp(self.a_mean * rfv + self.b_mean, min=0))  # Clamp for rfv
            * torch.randn(rfv.shape).to(rfv.device)  # Generate noise and apply it
            * recip_sqrt_num_views  # Scale noise by the reciprocal of sqrt(num_views)
        )
        
        return stack, rfv  # Return the noisy stack and refocused volume