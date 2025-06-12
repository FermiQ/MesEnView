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
Implements the Pinball Loss, which is commonly used in quantile regression tasks.

The `PinballLoss` class calculates the Pinball Loss between the predicted output and the target values. The Pinball Loss is a piecewise linear function that penalizes predictions above and below the target value differently, based on the specified quantile level.

Args:
    quantile (float): The quantile level for the loss, should be in the range (0, 1). It defines the quantile to be estimated; for instance, 0.10 corresponds to the 10th percentile.
    reduction (str): Specifies the reduction method to apply to the output. Options are "none" (no reduction), "mean" (average of losses), and "sum" (sum of losses).

Returns:
    torch.Tensor: The computed loss value, reduced according to the specified method.
"""
# Author: Anastasios Nikolas Angelopoulos, angelopoulos@berkeley.edu
# https://github.com/aangelopoulos/im2im-uq/blob/main/core/models/losses/pinball.py
# This module implements the Pinball Loss, which is commonly used in quantile regression tasks.

import torch

# PinballLoss class for calculating the Pinball Loss.
class PinballLoss:
    def __init__(self, quantile=0.10, reduction="mean"):
        """
        Initializes the PinballLoss instance.

        Parameters:
        quantile (float): The quantile level for the loss, should be in the range (0, 1).
                          It defines the quantile to be estimated; for instance, 0.10 corresponds to the 10th percentile.
        reduction (str): Specifies the reduction method to apply to the output. 
                         Options are "none" (no reduction), "mean" (average of losses), and "sum" (sum of losses).
        """
        self.quantile = quantile
        assert 0 < self.quantile  # Ensures quantile is positive
        assert self.quantile < 1  # Ensures quantile is less than 1
        self.reduction = reduction

    def __call__(self, output, target):
        """
        Computes the Pinball loss between the output and target tensors.

        Parameters:
        output (torch.Tensor): The predicted values from the model.
        target (torch.Tensor): The true target values.

        Returns:
        torch.Tensor: The computed loss value, reduced according to the specified method.
        """
        assert output.shape == target.shape  # Ensure output and target tensors have the same shape
        loss = torch.zeros_like(target, dtype=torch.float)  # Initialize the loss tensor with the same shape as target
        error = output - target  # Calculate the error between output and target
        smaller_index = error < 0  # Boolean mask for errors smaller than 0
        bigger_index = 0 < error  # Boolean mask for errors greater than 0

        # Calculate loss for predictions lower than the target
        loss[smaller_index] = self.quantile * (abs(error)[smaller_index])
        # Calculate loss for predictions greater than the target
        loss[bigger_index] = (1 - self.quantile) * (abs(error)[bigger_index])

        # Apply reduction method if specified
        if self.reduction == "sum":
            loss = loss.sum()  # Sum the losses if reduction is 'sum'
        if self.reduction == "mean":
            loss = loss.mean()  # Average the losses if reduction is 'mean'

        return loss  # Return the computed loss