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
The SBRNet class is a neural network model that implements a specific architecture using ResNet as the backbone. It consists of two main branches: the view synthesis branch and the refinement branch. The model takes two input tensors, the low-frequency view stack (lf_view_stack) and the refinement view (rfv), and produces an output tensor.

The ResConnection class implements a residual connection with scaling, where the input tensor is added to the output of the sequential layers, and the result is scaled by a constant factor (RSQRT2).

The ResBlock class defines a single residual block, which consists of two convolutional layers with batch normalization.

The ResNetCM2NetBlock class is a sequential model block based on the ResNet architecture. It is initialized based on the specified branch (either 'view_synthesis' or 'refinement') and the configuration dictionary.
"""
from pandas import read_parquet
import torch
import torch.nn as nn
import torch.nn.functional as F

# Importing types for better code clarity
from torch import Tensor
from torch.nn import Module, Conv2d, Sequential
from sbrnet_core.utils.constants import view_combos

# Constant for square root of 2, used for normalization
RSQRT2 = torch.sqrt(torch.tensor(0.5)).item()

# SBRNet is a neural network model that implements a specific architecture using ResNet as the backbone.
class SBRNet(Module):
    def __init__(self, config) -> None:
        """
        Initializes the SBRNet model with the given configuration.

        Parameters:
        config (dict): Configuration dictionary containing model parameters such as
                       backbone type, number of layers, and weight initialization method.
        """
        super().__init__()
        self.config = config

        # Check if the specified backbone is 'resnet', and initialize branches accordingly
        if config["backbone"] == "resnet":
            self.view_synthesis_branch = ResNetCM2NetBlock(config, "view_synthesis")
            self.rfv_branch = ResNetCM2NetBlock(config, "refinement")
        else:
            raise ValueError(
                f"Unknown backbone: {config['backbone']}. Only 'resnet' is supported."
            )
        
        # Creating convolutional layers with ReLU activation
        self.conv_layers_with_relu = nn.Sequential(*[
            nn.Conv2d(config.get("num_gt_layers")*2, config.get("num_gt_layers")*2, kernel_size=3, padding=1),
            nn.ReLU()
            for _ in range(config.get("num_head_layers")-1)
        ])
        
        # Final convolutional layer to reduce channels
        self.end_conv: Module = nn.Conv2d(
            config.get("num_gt_layers") * 2,
            config.get("num_gt_layers"),
            kernel_size=3,
            padding=1,
        )
        # Initialize convolutional layers
        self.init_convs()

    def init_convs(self) -> None:
        """
        Initializes the convolutional layers' weights based on the specified method in the config.
        """
        def init_fn(mod: Module) -> None:
            """
            Initializes the weights of convolutional layers.
            
            Parameters:
            mod (Module): The module to initialize.
            """
            if isinstance(mod, Conv2d):
                weight_init = self.config.get("weight_init", "kaiming_normal")
                if weight_init == "kaiming_normal":
                    nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
                elif weight_init == "xavier_normal":
                    nn.init.xavier_normal_(mod.weight)
                else:
                    raise ValueError(
                        f"Unsupported weight initialization method: {weight_init}"
                    )

        # Apply weight initialization to the branches
        self.view_synthesis_branch.apply(init_fn)
        self.rfv_branch.apply(init_fn)

    def forward(self, lf_view_stack: Tensor, rfv: Tensor) -> Tensor:
        """
        Defines the forward pass of the SBRNet model.

        Parameters:
        lf_view_stack (Tensor): Input tensor representing the low-frequency view stack.
        rfv (Tensor): Input tensor representing the refinement view.

        Returns:
        Tensor: The output of the network after processing both inputs.
        """
        return self.end_conv(
            (self.view_synthesis_branch(lf_view_stack) + self.rfv_branch(rfv)) * RSQRT2
        )

# ResConnection implements residual connections with scaling in the neural network.
class ResConnection(Sequential):
    def forward(self, data: Tensor, scale: float = RSQRT2) -> Tensor:
        """
        Forward pass for ResConnection that adds the input to the output producing a residual connection.

        Parameters:
        data (Tensor): Input tensor to the connection.
        scale (float): Scaling factor for the residual connection (default is RSQRT2).

        Returns:
        Tensor: The output tensor after applying the residual connection.
        """
        return (super().forward(data) + data) * scale

# ResBlock defines a single residual block using batch normalization and convolutional layers.
class ResBlock(Sequential):
    def __init__(self, channels: int) -> None:
        """
        Initializes a residual block consisting of two convolutional layers with batch normalization.

        Parameters:
        channels (int): Number of channels in the input and output tensors of the block.
        """
        super(ResBlock, self).__init__(
            nn.BatchNorm2d(channels),
            ResConnection(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(True),
                nn.BatchNorm2d(channels),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            ),
        )

# ResNetCM2NetBlock is a sequential model block based on the ResNet architecture.
class ResNetCM2NetBlock(Sequential):
    def __init__(self, config, branch: str) -> None:
        """
        Initializes the ResNetCM2NetBlock based on the specified branch and configuration.

        Parameters:
        config (dict): Configuration dictionary containing model parameters.
        branch (str): Specifies the branch ('view_synthesis' or 'refinement') to initialize.

        Raises:
        ValueError: If an unknown branch is specified.
        """
        # Load data from the specified parquet dataset
        df = read_parquet(config["dataset_pq"])
        
        # Determine input channels based on the branch type
        if branch == "view_synthesis":
            inchannels = df.iloc[0].num_views
        elif branch == "refinement":
            inchannels = config["num_rfv_layers"]
        else:
            raise ValueError(
                f"Unknown branch: {branch}. Only 'view_synthesis' and 'refinement' are supported."
            )

        numblocks = config["num_resblocks"]
        outchannels = config["num_gt_layers"]
        super(ResNetCM2NetBlock, self).__init__(
            nn.Conv2d(inchannels, outchannels * 2, kernel_size=3, padding=1),
            *(ResBlock(channels=outchannels * 2) for _ in range(numblocks)),
            nn.BatchNorm2d(outchannels * 2),
            nn.Conv2d(outchannels * 2, outchannels * 2, kernel_size=3, padding=1),
        )