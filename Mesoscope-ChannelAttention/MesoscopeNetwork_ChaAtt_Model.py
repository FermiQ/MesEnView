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
Implements the Residual Channel Attention Network (RCAN) for image restoration.

The RCAN model consists of several key components:
- ChannelAttention: Applies channel attention mechanism to enhance feature representations.
- RCAB: Residual Channel Attention Block that combines convolution, ReLU, and channel attention.
- RG: Residual Group that contains multiple RCABs followed by a convolution layer.
- RCAN: The main RCAN model that applies downscaling, residual groups, upscaling, and final convolution.
- FourierDeconvolution2D_ds: Performs deconvolution in the frequency domain for each point spread function (PSF).
- MultiWienerDeconvolution2D: Applies Wiener deconvolution in the frequency domain for each PSF.
- LSVEnsemble2d: Combines the deconvolution and enhancement models into an ensemble.
- resblock: Residual block for building deeper networks with skip connections.
- cm2netblock: A neural network block that consists of convolutional layers and residual blocks.
- cm2net: The main network class that combines the demixing and reconstruction blocks.

The documentation provides an overview of the key components and their functionality within the RCAN model.

The code defines several neural network models and their components for image restoration tasks. A brief overview of each module:

1. **RCAB (Residual Channel Attention Block)**: This module is designed for image restoration tasks. It includes a convolution layer, a channel attention layer, and a residual connection. The channel attention layer helps in adjusting the channel-wise feature responses adaptively.

2. **RG (Residual Group)**: This module consists of multiple RCABs followed by a convolution layer. It is used to enhance the feature extraction capabilities of the network.

3. **FourierDeconvolution2D**: This module performs Wiener deconvolution in the frequency domain for each PSF. It includes a first convolution, a list of residual blocks, and a second convolution.

4. **LSVEnsemble2D**: This is an ensemble model that combines deconvolution and enhancement processes. It consists of two cm2netblocks for demixing and reconstruction, followed by a final convolutional layer and a sigmoid activation function.

5. **cm2netblock**: This module is a neural network block that includes a first convolution, a list of residual blocks, and a second convolution. It is used in the LSVEnsemble2D model for demixing and reconstruction.

6. **cm2net**: This is the main network class that consists of two cm2netblocks for demixing and reconstruction, followed by a final convolutional layer and a sigmoid activation function.
"""

from utils import *  # Importing utility functions from a utils module
import torch.nn as nn  # Importing neural network module from PyTorch
import torch  # Importing PyTorch
import numpy as np  # Importing NumPy for numerical operations
import torch  # Duplicate import of PyTorch (can be removed)
import torch.nn as nn  # Duplicate import of neural network module (can be removed)
import torch.nn.functional as F  # Importing functional interface for PyTorch neural networks

class ChannelAttention(nn.Module):
    """
    Implements Channel Attention mechanism for enhancing feature representations.
    
    Attributes:
        module (nn.Sequential): The sequence of operations for channel attention.
    
    Parameters:
        num_features (int): Number of input feature channels.
        reduction (int): Reduction ratio for channel attention.
    """
    def __init__(self, num_features, reduction):
        super(ChannelAttention, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Apply adaptive average pooling to get channel-wise statistics
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),  # Reduce dimensions
            nn.ReLU(inplace=True),  # Activation function
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),  # Restore dimensions
            nn.Sigmoid()  # Sigmoid activation to get channel attention weights
        )

    def forward(self, x):
        """
        Forward pass for Channel Attention.
        
        Parameters:
            x (Tensor): Input tensor of shape (N, C, H, W), where
                        N is batch size, C is number of channels, H is height, W is width.
        
        Returns:
            Tensor: Output tensor after applying channel attention.
        """
        return x * self.module(x)  # Scale input by channel attention weights


class RCAB(nn.Module):
    """
    Residual Channel Attention Block (RCAB) for enhancing feature representation.
    
    Attributes:
        module (nn.Sequential): The sequence of operations in the RCAB.
    
    Parameters:
        num_features (int): Number of input feature channels.
        reduction (int): Reduction ratio for channel attention.
    """
    def __init__(self, num_features, reduction):
        super(RCAB, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),  # Convolution layer
            nn.ReLU(inplace=True),  # Activation function
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),  # Convolution layer
            ChannelAttention(num_features, reduction)  # Channel attention layer
        )

    def forward(self, x):
        """
        Forward pass for RCAB.
        
        Parameters:
            x (Tensor): Input tensor of shape (N, C, H, W).
        
        Returns:
            Tensor: Output tensor after applying the RCAB.
        """
        return x + self.module(x)  # Add input (residual connection) to output of module


class RG(nn.Module):
    """
    Residual Group (RG) containing multiple RCABs followed by a convolution layer.
    
    Attributes:
        module (nn.Sequential): The sequence of operations in the RG.
    
    Parameters:
        num_features (int): Number of input feature channels.
        num_rcab (int): Number of RCABs in the group.
        reduction (int): Reduction ratio for channel attention.
    """
    def __init__(self, num_features, num_rcab, reduction):
        super(RG, self).__init__()
        self.module = [RCAB(num_features, reduction) for _ in range(num_rcab)]  # Create multiple RCABs
        self.module.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))  # Final convolution
        self.module = nn.Sequential(*self.module)  # Convert list to sequential module

        # Initialize weights for Conv2d and BatchNorm2d layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # Kaiming initialization for Conv2d
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)  # Set BatchNorm weights to 1
                torch.nn.init.constant_(m.bias, 0)  # Set BatchNorm bias to 0

    def forward(self, x):
        """
        Forward pass for RG.
        
        Parameters:
            x (Tensor): Input tensor of shape (N, C, H, W).
        
        Returns:
            Tensor: Output tensor after applying the RG.
        """
        return x + self.module(x)  # Add input (residual connection) to output of module


class RCAN(nn.Module):
    """
    Residual Channel Attention Network (RCAN) for image restoration.
    
    Parameters:
        num_psfs (int): Number of point spread functions (PSFs).
    """
    def __init__(self,num_psfs):
        super(RCAN, self).__init__()
        scale = 2  # Downscale factor
        num_features = 32  # Number of feature channels
        num_rg = 2  # Number of residual groups
        num_rcab = 3  # Number of RCABs per group
        reduction = 16  # Reduction ratio for channel attention
        
        # Downscale operation
        self.downscale = nn.Sequential(
            nn.PixelUnshuffle(scale),  # Downscale spatial dimensions by a factor of scale
            nn.Conv2d(num_psfs*scale ** 2, num_features, kernel_size=3, padding=1)  # Convolution to extract features
        )
        
        self.rgs = nn.Sequential(*[RG(num_features, num_rcab, reduction) for _ in range(num_rg)])  # Create residual groups
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)  # Convolution layer after RGs
        
        # Upscale operation
        self.upscale = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale ** 2), kernel_size=3, padding=1),  # Convolution to increase channels
            nn.PixelShuffle(scale)  # Upscale spatial dimensions by a factor of scale
        )
        
        self.conv2 = nn.Conv2d(num_features, 1, kernel_size=3, padding=1)  # Final output convolution layer
        self.activation = nn.Sigmoid()  # Sigmoid activation for output

    def forward(self, x):
        """
        Forward pass for RCAN.
        
        Parameters:
            x (Tensor): Input tensor of shape (N, C, H, W).
        
        Returns:
            Tensor: Output tensor after processing through the network.
        """
        x = self.downscale(x)  # Downscale input
        residual = x  # Store residual for skipping connection
        x = self.rgs(x)  # Process through residual groups
        x = self.conv1(x)  # Convolution after RGs
        x += residual  # Add residual connection
        x = self.upscale(x)  # Upscale output
        x = self.conv2(x)  # Final convolution to produce output
        x = self.activation(x)  # Apply activation function
        return x  # Return final output


class FourierDeconvolution2D_ds(nn.Module):
    """
    Performs Deconvolution in the frequency domain for each PSF.
    
    Input: initial_psfs of shape (Y, X, C), initial_K has shape (1, 1, C) for each PSF.
    
    Parameters:
        num_psfs (int): Number of point spread functions.
        ps (int): Downsampling scale factor.
    """
    def __init__(self,num_psfs, ps):
        super(FourierDeconvolution2D_ds, self).__init__()
        self.scale = ps  # Store scale factor
        self.channel = num_psfs  # Number of channels corresponding to PSFs
        
        # Initialize PSFs in real and imaginary parts
        self.psfs_re = nn.Parameter(torch.rand(self.channel, 4200//self.scale, (2100//self.scale)+1) * 0.001)
        self.psfs_im = nn.Parameter(torch.rand(self.channel, 4200//self.scale, (2100//self.scale)+1) * 0.001)
        
        self.ds =  nn.PixelUnshuffle(self.scale)  # Downscale operation
        self.us = nn.PixelShuffle(self.scale)  # Upscale operation
        
        self.conv = nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=1)  # Convolution layer
        torch.nn.init.normal_(self.conv.weight)  # Normal initialization for convolution weights
        self.activation = nn.PReLU()  # PReLU activation function

    def forward(self, y):
        """
        Forward pass for Fourier Deconvolution.
        
        Parameters:
            y (Tensor): Input tensor of shape (N, H, W) to be processed.
        
        Returns:
            Tensor: Output tensor after deconvolution.
        """
        y = y.unsqueeze(1)  # Add channel dimension
        y = self.ds(y)  # Downscale input
        Y = torch.fft.rfft2(y, dim=(-2, -1))  # Compute 2D FFT of the input
        Y = Y.unsqueeze(1)  # Add a dimension for the PSFs
        psfs_re = self.psfs_re[None, ...]  # Expand PSF parameter for broadcasting
        psfs_im = self.psfs_im[None, ...]  # Expand PSF parameter for broadcasting
        psf_freq = torch.complex(psfs_re, psfs_im)  # Create complex PSF frequency representation
        X = Y * psf_freq.unsqueeze(2)  # Multiply in frequency domain
        x = torch.fft.irfft2(X, dim=(-2, -1))  # Compute inverse FFT to get the spatial domain result
        x = self.us(x).squeeze(2)  # Upscale and remove the channel dimension
        return x  # Return the output tensor

    def get_config(self):
        """
        Get configuration parameters for the model.
        
        Returns:
            dict: Configuration containing scale and channel information.
        """
        config = {
            'scale': self.scale,  # Include scale factor
            'channel': self.channel,  # Include number of channels
        }
        return config


class MultiWienerDeconvolution2D(nn.Module):
    """
    Performs Wiener Deconvolution in the frequency domain for each PSF.
    
    Input: initial_psfs of shape (Y, X, C), initial_K has shape (1, 1, C) for each PSF.
    
    Parameters:
        initial_psfs (array): Initial point spread functions.
        initial_Ks (array): Initial noise parameters for Wiener deconvolution.
    """
    def __init__(self, initial_psfs, initial_Ks):
        super(MultiWienerDeconvolution2D, self).__init__()
        initial_psfs = torch.tensor(initial_psfs, dtype=torch.float32)  # Convert to tensor
        initial_Ks = torch.tensor(initial_Ks, dtype=torch.float32)  # Convert to tensor
        
        self.psfs = nn.Parameter(initial_psfs, requires_grad=True)  # PSFs as learnable parameters
        self.Ks = nn.Parameter(initial_Ks, requires_grad=True)  # Noise parameters as learnable parameters

    def forward(self, y):
        """
        Forward pass for Multi Wiener Deconvolution.
        
        Parameters:
            y (Tensor): Input tensor of shape (B, C, H, W).
        
        Returns:
            Tensor: Output tensor after applying Wiener deconvolution.
        """
        y = y.unsqueeze(1)  # Add channel dimension
        y = y.type(torch.complex64)  # Convert to complex type
        Y = torch.fft.fft2(y)  # Compute 2D FFT of the input
        psf = self.psfs.type(torch.complex64)  # Convert PSF to complex type
        H_sum = torch.fft.fft2(psf)  # Compute FFT of PSF
        X = (torch.conj(H_sum) * Y) / (torch.square(torch.abs(H_sum)) + self.Ks)  # Wiener deconvolution formula
        x = torch.real((torch.fft.ifftshift(torch.fft.ifft2(X), dim=(-2, -1))))  # Inverse FFT and return real part
        return x  # Return the output tensor

    def get_config(self):
        """
        Get configuration parameters for the model.
        
        Returns:
            dict: Configuration containing initial PSFs and noise parameters.
        """
        config = super().get_config().copy()  # Get existing config
        config.update({
            'initial_psfs': self.psfs.numpy(),  # Add initial PSFs
            'initial_Ks': self.Ks.numpy()  # Add initial noise parameters
        })
        return config


class LSVEnsemble2d(nn.Module):
    """
    Ensemble model combining deconvolution and enhancement processes.
    
    Parameters:
        deconvolution (nn.Module): A deconvolution model.
        enhancement (nn.Module): An enhancement model.
    """
    def __init__(self, deconvolution, enhancement):
        super(LSVEnsemble2d, self).__init__()
        self.dropout = nn.Dropout(0.1)  # Dropout layer for regularization
        self.deconvolution = deconvolution  # Store deconvolution model
        self.enhancement = enhancement  # Store enhancement model

    def forward(self, x):
        """
        Forward pass for LSV Ensemble.
        
        Parameters:
            x (Tensor): Input tensor to be processed.
        
        Returns:
            Tensor: Final output tensor after deconvolution and enhancement.
        """
        initial_output = self.deconvolution(x)  # Apply deconvolution
        w = initial_output.shape[-1]  # Get width of the output
        h = initial_output.shape[-2]  # Get height of the output
        
        # Normalize output for cropping
        initial_output = initial_output / torch.max(initial_output)
        
        # Crop the output to a specific size
        initial_output = initial_output[..., h//2+1 - 2400 // 2:h//2+1 + 2400 // 2, w//2+1 - 2400 // 2:w//2+1 + 2400 // 2]
        initial_output = initial_output / torch.max(initial_output)  # Re-normalize cropped output
        final_output = self.enhancement(initial_output)  # Apply enhancement
        return final_output  # Return final output


class resblock(nn.Module):
    """
    Residual block for building deeper networks with skip connections.
    
    Parameters:
        channels (int): Number of input and output channels for the block.
    """
    def __init__(self, channels=48):
        super(resblock, self).__init__()
        self.channels = channels  # Store number of channels

        # First convolution layer
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')  # Kaiming initialization for weights
        self.bn1 = nn.BatchNorm2d(channels)  # Batch normalization after first convolution
        self.act = nn.ReLU(inplace=True)  # ReLU activation function this is for the based upon the application for the

        # Second convolution layer
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')  # Kaiming initialization for weights
        self.bn2 = nn.BatchNorm2d(channels)  # Batch normalization after second convolution

    def forward(self, x1):
        """
        Forward pass for the residual block.
        
        Parameters:
            x1 (Tensor): Input tensor of shape (N, C, H, W).
        
        Returns:
            Tensor: Output tensor after applying the residual block.
        """
        x1 = self.conv1(x1)  # Apply first convolution
        x1 = self.act(self.bn1(x1))  # Apply activation and batch normalization
        x1 = self.conv2(x1)  # Apply second convolution
        return (self.bn2(x1))  # Return output after batch normalization

class cm2netblock(nn.Module):
    """
    A neural network block that consists of convolutional layers and residual blocks.
    
    Attributes:
        inchannels (int): Number of input channels for the convolution operations.
        outchannels (int): Number of output channels after the initial convolution.
        numblocks (int): Number of residual blocks to be applied within this block.
        conv1 (nn.Conv2d): First convolutional layer that transforms input channels to output channels.
        act (nn.ReLU): Activation function applied after the first convolution.
        resblocks (nn.ModuleList): List of residual blocks to be applied sequentially.
        conv2 (nn.Conv2d): Second convolutional layer that transforms output channels back to input channels.
    """

    def __init__(self, inchannels, numblocks, outchannels=48):
        """
        Initializes the cm2netblock with specified channels and number of blocks.

        Parameters:
            inchannels (int): Number of input channels.
            numblocks (int): Number of residual blocks to add.
            outchannels (int, optional): Number of output channels after the first convolution. Default is 48.
        """
        super(cm2netblock, self).__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.numblocks = numblocks

        # Initialize the first convolutional layer
        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1)
        # ReLU activation applied after first convolution
        self.act = nn.ReLU(inplace=True)
        # Apply Kaiming normalization to the weights of the first convolution
        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')

        # Create a list of residual blocks
        self.resblocks = nn.ModuleList([resblock(self.outchannels) for i in range(numblocks)])
        
        # Initialize the second convolutional layer
        self.conv2 = nn.Conv2d(outchannels, inchannels, kernel_size=3, padding=1)
        # Apply Kaiming normalization to the weights of the second convolution
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')

    def forward(self, x):
        """
        Defines the forward pass of the cm2netblock.

        Parameters:
            x (torch.Tensor): Input tensor with shape (N, C, H, W), where:
                N = batch size,
                C = number of input channels,
                H = height of the input,
                W = width of the input.

        Returns:
            torch.Tensor: Output tensor after applying the convolution and residual blocks.
        """
        # Apply the first convolution and activation
        x0 = self.act(self.conv1(x))
        x1 = torch.clone(x0)
        
        # Sequentially apply each residual block
        for _, modulee in enumerate(self.resblocks):
            x1 = (modulee(x1) + x1) / 1.414  # Normalize the output with the input of the residual block
        
        # Combine the output of the last residual block with the first layer's output
        x1 = (x1 + x0) / 1.414
        
        # Apply the second convolution and return the output
        return self.conv2(x1)


class cm2net(nn.Module):
    """
    The main network class that consists of two cm2netblocks for demixing and reconstruction,
    followed by a final convolutional layer to produce the output.

    Attributes:
        demix (cm2netblock): The first block used for demixing the input.
        recon (cm2netblock): The second block used for reconstructing the output.
        endconv (nn.Conv2d): Final convolutional layer that reduces the channel dimension to 1.
        activation (nn.Sigmoid): Activation function applied to the output of the final convolution.
    """

    def __init__(self, numBlocks, stackchannels=9, outchannels=48):
        """
        Initializes the cm2net model with demixing and reconstruction blocks.

        Parameters:
            numBlocks (int): The number of residual blocks in each cm2netblock.
            stackchannels (int, optional): Number of input channels to the blocks. Default is 9.
            outchannels (int, optional): Number of output channels for the blocks. Default is 48.
        """
        super(cm2net, self).__init__()
        self.demix = cm2netblock(stackchannels, numblocks=numBlocks, outchannels=outchannels)
        self.recon = cm2netblock(stackchannels, numblocks=numBlocks, outchannels=outchannels)
        self.endconv = nn.Conv2d(stackchannels, 1, kernel_size=3, padding=1)
        self.activation = nn.Sigmoid()

    def forward(self, stack):
        """
        Defines the forward pass of the cm2net model.

        Parameters:
            stack (torch.Tensor): Input tensor with shape (N, C, H, W), where:
                N = batch size,
                C = number of input channels (stackchannels),
                H = height of the input,
                W = width of the input.

        Returns:
            tuple: A tuple containing:
                - demix_result (torch.Tensor): Result after the demixing block.
                - output (torch.Tensor): Final output after reconstruction and activation.
        """
        # Apply the demixing block and activation
        demix_result = self.activation(self.demix(stack))
        
        # Apply the reconstruction block
        output = self.recon(demix_result)  # no squeeze
        
        # Apply the final convolution and activation
        output = self.activation(self.endconv(output))
        
        return demix_result, output