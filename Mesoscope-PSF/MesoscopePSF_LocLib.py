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

#%% Import necessary libraries
import ctypes  # For C libraries interfacing
import numpy.ctypeslib as ctl  # For handling NumPy arrays in C
import numpy as np  # For numerical operations
import h5py as h5  # For handling HDF5 files
from .utilities import psf2cspline_np  # Importing a custom utility for PSF to CSpline conversion
import matplotlib.pyplot as plt  # For plotting and visualization
from tqdm import tqdm  # For creating progress bars
import os  # For interacting with the operating system
import sys  # For accessing system-specific parameters and functions
import tensorflow as tf  # For machine learning and deep learning
from psflearning import io  # Importing input/output utility from psflearning module

#%% Localization library class definition
class localizationlib:
    """
    A class to encapsulate the functionality for localization in microscopy data using 
    Maximum Likelihood Estimation (MLE) techniques, with support for both CPU and GPU processing.

    Attributes:
        usecuda (bool): Flag indicating whether to use CUDA for GPU processing.
    """
    
    def __init__(self, usecuda=False):
        """
        Initializes the localization library and loads the appropriate shared libraries based on the system platform.
        
        Parameters:
            usecuda (bool): A flag indicating whether to attempt to use CUDA for GPU processing. Default is False.
        """
        
        # Obtain the path of the current file and package
        thispath = os.path.dirname(os.path.abspath(__file__))  # Current file's directory
        pkgpath = os.path.dirname(os.path.dirname(thispath))  # Parent package directory
        
        # Load configuration paths from a YAML file
        cfg = io.param.load(pkgpath + '/config/path/config_path.yaml')
        
        # Determine the platform and set paths for shared libraries
        if sys.platform.startswith('win'):
            # Windows-specific paths for shared libraries
            dllpath_cpu_astM = pkgpath + cfg.Paths.spline.win.cpu.astM
            dllpath_gpu_astM = pkgpath + cfg.Paths.spline.win.cuda.astM
            dllpath_cpu_4pi = pkgpath + cfg.Paths.spline.win.cpu.fpi
            dllpath_gpu_4pi = pkgpath + cfg.Paths.spline.win.cuda.fpi
            dllpath_cpu_ast = pkgpath + cfg.Paths.spline.win.cpu.ast
            dllpath_gpu_ast = pkgpath + cfg.Paths.spline.win.cuda.ast
            
            # Check for available GPU devices and load the corresponding libraries
            if tf.config.list_physical_devices('GPU'):
                lib_gpu_astM = ctypes.CDLL(dllpath_gpu_astM)            
                lib_gpu_4pi = ctypes.CDLL(dllpath_gpu_4pi)            
                lib_gpu_ast = ctypes.CDLL(dllpath_gpu_ast)
            else:
                usecuda = False  # Disable CUDA if no GPU is detected

        elif sys.platform.startswith('darwin'):
            # macOS-specific paths for shared libraries
            usecuda = False
            dllpath_cpu_ast = pkgpath + cfg.Paths.spline.mac.cpu.ast
            dllpath_cpu_astM = pkgpath + cfg.Paths.spline.mac.cpu.astM
            dllpath_cpu_4pi = pkgpath + cfg.Paths.spline.mac.cpu.fpi

        elif sys.platform.startswith('linux'):
            # Linux-specific paths for shared libraries
            dllpath_cpu_ast = pkgpath + cfg.Paths.spline.linux.cpu.ast
            dllpath_gpu_ast = pkgpath + cfg.Paths.spline.linux.cuda.ast
            dllpath_cpu_astM = pkgpath + cfg.Paths.spline.linux.cpu.astM
            dllpath_gpu_astM = pkgpath + cfg.Paths.spline.linux.cuda.astM
            dllpath_cpu_4pi = pkgpath + cfg.Paths.spline.linux.cpu.fpi
            dllpath_gpu_4pi = pkgpath + cfg.Paths.spline.linux.cuda.fpi
            
            # Check for available GPU devices and load the corresponding libraries
            if tf.config.list_physical_devices('GPU'):
                lib_gpu_astM = ctypes.CDLL(dllpath_gpu_astM)            
                lib_gpu_4pi = ctypes.CDLL(dllpath_gpu_4pi)            
                lib_gpu_ast = ctypes.CDLL(dllpath_gpu_ast)
            else:
                usecuda = False  # Disable CUDA if no GPU is detected

        try:
            # Attempt to load CPU versions of the shared libraries
            lib_cpu_astM = ctypes.CDLL(dllpath_cpu_astM)        
            lib_cpu_4pi = ctypes.CDLL(dllpath_cpu_4pi)        
            lib_cpu_ast = ctypes.CDLL(dllpath_cpu_ast)
        except:
            # If loading fails, notify the user
            print('MLE CPU fitting is not available')
       
        # Assign the appropriate fitting function based on the CUDA usage flag
        if usecuda:
            self._mleFit_MultiChannel = lib_gpu_astM.GPUmleFit_MultiChannel
            self._mleFit_4Pi = lib_gpu_4pi.GPUmleFit_LM_4Pi
            self._mleFit = lib_gpu_ast.GPUmleFit_LM
        else:
            self._mleFit_MultiChannel = lib_cpu_astM.CPUmleFit_MultiChannel
            self._mleFit_4Pi = lib_cpu_4pi.CPUmleFit_LM_4Pi
            self._mleFit = lib_cpu_ast.CPUmleFit_LM
        
        # Define argument types for MLE fitting functions for proper data handling
        self._mleFit_4Pi.argtypes = [
            ctl.ndpointer(np.float32), # data input as a NumPy array of float32
            ctl.ndpointer(np.int32),   # shared data input as a NumPy array of int32
            ctypes.c_int32,            # number of iterations for fitting
            ctl.ndpointer(np.float32), # spline coefficients as a NumPy array of float32
            ctl.ndpointer(np.float32), # dTAll as a NumPy array of float32
            ctl.ndpointer(np.float32), # phiA as a NumPy array of float32
            ctl.ndpointer(np.float32), # initial z value as a NumPy array of float32
            ctl.ndpointer(np.float32), # initial phase as a NumPy array of float32
            ctl.ndpointer(np.int32),    # data size as a NumPy array of int32
            ctl.ndpointer(np.int32),    # spline size as a NumPy array of int32
            ctl.ndpointer(np.float32), # output P as a NumPy array of float32
            ctl.ndpointer(np.float32), # output CRLB as a NumPy array of float32
            ctl.ndpointer(np.float32)  # output LL as a NumPy array of float32
        ]

        self._mleFit_MultiChannel.argtypes = [
            ctl.ndpointer(np.float32), # data input as a NumPy array of float32
            ctypes.c_int32,            # fitting type as an int32
            ctl.ndpointer(np.int32),   # shared data input as a NumPy array of int32
            ctypes.c_int32,            # number of iterations for fitting
            ctl.ndpointer(np.float32), # spline coefficients as a NumPy array of float32
            ctl.ndpointer(np.float32), # dTAll as a NumPy array of float32
            ctl.ndpointer(np.float32), # variances as a NumPy array of float32
            ctl.ndpointer(np.float32), # initial z value as a NumPy array of float32
            ctl.ndpointer(np.int32),    # data size as a NumPy array of int32
            ctl.ndpointer(np.int32),    # spline size as a NumPy array of int32
            ctl.ndpointer(np.float32), # output P as a NumPy array of float32
            ctl.ndpointer(np.float32), # output CRLB as a NumPy array of float32
            ctl.ndpointer(np.float32)  # output LL as a NumPy array of float32
        ]

        self._mleFit.argtypes = [
            ctl.ndpointer(np.float32), # data input as a NumPy array of float32
            ctypes.c_int32,            # fitting type as an int32
            ctypes.c_int32,            # number of iterations for fitting
            ctl.ndpointer(np.float32), # spline coefficients as a NumPy array of float32
            ctl.ndpointer(np.float32), # variances as a NumPy array of float32
            ctypes.c_float,            # initial z value as a float
            ctl.ndpointer(np.int32),   # data size as a NumPy array of int32
            ctl.ndpointer(np.int32),    # spline size as a NumPy array of int32
            ctl.ndpointer(np.float32), # output P as a NumPy array of float32
            ctl.ndpointer(np.float32), # output CRLB as a NumPy array of float32
            ctl.ndpointer(np.float32)  # output LL as a NumPy array of float32
        ]

def loc_ast_dual(self, psf_data, I_model, pixelsize_z, cor, imgcenter, T, initz=None, plot=False, start_time=0):
    """
    Perform localization of multiple beads in dual-channel data using a maximum likelihood estimation method.

    Parameters:
    - psf_data: np.ndarray
        The point spread function data with shape (Nchannel, Nz, rsz, rsz).
    - I_model: np.ndarray
        The model intensity image, the same shape as psf_data.
    - pixelsize_z: float
        The size of each pixel in the z-dimension.
    - cor: np.ndarray
        The coordinates of the beads in the form of (Nchannel, Nbead, 2).
    - imgcenter: np.ndarray
        The center of the image for transformation, shape (2,).
    - T: np.ndarray
        Transformation matrix with shape (Nchannel, 3, 3).
    - initz: np.ndarray, optional
        Initial z-coordinates for the beads.
    - plot: bool, optional
        Whether to plot the results or not.
    - start_time: float, optional
        The start time for timing the operation.

    Returns:
    - P: np.ndarray
        Estimated parameters for the localization.
    - CRLB: np.ndarray
        Cramer-Rao lower bounds for the estimated parameters.
    - LL: np.ndarray
        Log-likelihood values for the estimated parameters.
    - Iall: np.ndarray
        Coefficients of the spline representation of the PSF.
    - msezRatio: np.ndarray
        Ratio of mean squared errors in z localization.
    - toc: float
        Total elapsed time for the localization process.
    - loc_dict: dict
        A dictionary containing the x, y, and z coordinates of the localization.
    """

    # Get sizes from input data
    rsz = psf_data.shape[-1]  # Size of the PSF
    Nbead = cor.shape[1]       # Number of beads
    Nchannel = cor.shape[0]    # Number of channels

    # Determine number of z-planes based on the shape of psf_data
    if len(psf_data.shape) > 4:
        Nz = psf_data.shape[-3]  # Number of z-planes
    else:
        Nz = 1

    # Calculate the number of fitting parameters
    Nfit = Nbead * Nz
    Nparam = 5  # Number of parameters to estimate

    # Prepare the image model for processing
    offset = np.min(I_model)  # Offset for normalization
    Iall = []  # List to store coefficients
    Imd = I_model - offset  # Subtract offset from the model
    normf = np.max(np.median(np.sum(Imd, axis=(-1, -2)), axis=-1))  # Find normalization factor
    Imd = Imd / normf  # Normalize the model

    # Set up progress bar for spline coefficient calculation
    pbar = tqdm(total=Nchannel, desc='4/6: calculating spline coefficients',
                bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}s] {rate_fmt} {postfix[0]}{postfix[1][time]:>4.2f}s",
                postfix=["total time: ", dict(time=start_time)])

    # Calculate spline coefficients for each channel
    for i in range(Nchannel):
        coeff = psf2cspline_np(Imd[i])  # Convert model to spline coefficients
        Iall.append(coeff)  # Append coefficients to list

        # Update progress bar
        pbar.postfix[1]['time'] = start_time + pbar._time() - pbar.start_t
        pbar.update(1)

    # Close progress bar for spline coefficients
    toc = pbar.postfix[1]['time']
    pbar.close()

    # Stack coefficients into an array
    Iall = np.stack(Iall).astype(np.float32)

    # Reshape and crop psf_data for processing
    data = psf_data.reshape((Nchannel, Nfit, rsz, rsz))
    bxsz = np.min((rsz, 20))  # Box size for cropping
    data = data[:, :, rsz // 2 - bxsz // 2:rsz // 2 + bxsz // 2,
                 rsz // 2 - bxsz // 2:rsz // 2 + bxsz // 2].astype(np.float32)
    data = np.maximum(data, 0.0)  # Ensure no negative values

    # Prepare coordinate transformation
    cor1 = np.concatenate((cor[0], np.ones((Nbead, 1))), axis=1)  # Append ones for homogeneous coordinates
    T1 = np.concatenate((np.expand_dims(np.eye(3), axis=0), T), axis=0)  # Combine identity with transformation
    dx1 = np.zeros((Nbead, Nchannel))  # Placeholder for x-differences
    dy1 = np.zeros((Nbead, Nchannel))  # Placeholder for y-differences

    # Calculate differences in coordinates
    for i in range(Nchannel):
        cor2 = np.matmul(cor1 - imgcenter, T1[i]) + imgcenter  # Apply transformation
        dy1[:, i] = cor2[:, 0] - cor[i][:, 0]  # Y differences
        dx1[:, i] = cor2[:, 1] - cor[i][:, 1]  # X differences

    # Prepare data for fitting
    dTS = np.zeros((Nbead, Nz, Nchannel * 2, Nparam))  # Shape for displacement values
    dTS[:, :, 0:Nchannel, 0] = np.expand_dims(dx1, axis=1)  # Assign dx
    dTS[:, :, 0:Nchannel, 1] = np.expand_dims(dy1, axis=1)  # Assign dy
    dTS[:, :, Nchannel:] = 1  # Placeholder for other parameters
    dTS = dTS.reshape((Nfit, Nchannel * 2, Nparam)).astype(np.float32)  # Reshape for fitting

    # Prepare shared parameters for fitting
    shared = np.array([1, 1, 1, 1, 0])  # Indicates shared parameters among beads
    sharedA = np.repeat(np.expand_dims(shared, axis=0), Nfit, axis=0).astype(np.int32)  # Expand shared array

    ccz = Iall.shape[-3] // 2  # Center index for z-coordinates

    # Initialize z-coordinates
    if initz is None:
        Nzm = Imd.shape[-3]  # Get number of z-planes
        initz = np.linspace(-Nzm * pixelsize_z / 2, Nzm * pixelsize_z / 2, np.int32(Nzm * pixelsize_z / 0.5)) * 0.8 / pixelsize_z + ccz
    else:
        initz = np.array(initz) * 0.5 / pixelsize_z + ccz
    zstart = np.repeat(np.expand_dims(initz, axis=1), Nfit, axis=1).astype(np.float32)  # Expand for fitting

    # Prepare data sizes
    datasize = np.array(np.flip(data.shape)).astype(np.int32)  # Flip data shape for processing
    splinesize = np.array(np.flip(Iall.shape)).astype(np.int32)  # Flip spline size for processing
    varim = np.array((0)).astype(np.float32)  # Placeholder for variance image

    # Initialize arrays for localization
    Pk = np.zeros((Nparam + 1 + (Nchannel - 1) * (Nparam - np.sum(shared)), Nfit)).astype(np.float32)
    CRLBk = np.zeros((Nparam + (Nchannel - 1) * (Nparam - np.sum(shared)), Nfit)).astype(np.float32)
    LLk = np.zeros((Nfit)).astype(np.float32)  # Log-likelihood initialization
    fittype = np.int32(2)  # Fitting type
    iterations = np.int32(100)  # Number of iterations for fitting
    P = np.zeros((Nparam + 1 + (Nchannel - 1) * (Nparam - np.sum(shared)), Nfit)).astype(np.float32)
    CRLB = np.zeros((Nparam + (Nchannel - 1) * (Nparam - np.sum(shared)), Nfit)).astype(np.float32)
    LL = np.zeros((Nfit)).astype(np.float32) - 1e10  # Initialize log-likelihood to a very low value

    # Set up progress bar for localization
    pbar = tqdm(total=len(zstart), desc='5/6: localization',
                bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}s] {rate_fmt} {postfix[0]}{postfix[1][time]:>4.2f}s",
                postfix=["total time: ", dict(time=toc)])

    # Localization process over z-coordinates
    for z0 in zstart:
        self._mleFit_MultiChannel(data, fittype, sharedA, iterations, Iall, dTS,
                                   varim, z0, datasize, splinesize, Pk, CRLBk, LLk)  # Fit the data
        mask = (LLk - LL) > 1e-4  # Create mask for significant updates
        LL[mask] = LLk[mask]  # Update log-likelihoods
        P[:, mask] = Pk[:, mask]  # Update parameters
        CRLB[:, mask] = CRLBk[:, mask]  # Update CRLB

        # Update progress bar
        pbar.postfix[1]['time'] = toc + pbar._time() - pbar.start_t
        pbar.update(1)

    # Close progress bar for localization
    toc = pbar.postfix[1]['time']
    pbar.close()

    # Reshape output parameters
    zf = P[2].reshape((Nbead, Nz))  # Reshape z-coordinates
    xf = P[1].reshape((Nbead, Nz))  # Reshape x-coordinates
    yf = P[0].reshape((Nbead, Nz))  # Reshape y-coordinates

    zg = np.linspace(0, Nz - 1, Nz)  # Generate theoretical z-coordinates
    if Nz > 1:
        # Adjust z-coordinates based on median
        zf = zf - np.median(zf - zg, axis=1, keepdims=True)
        zdiff = zf - zg  # Difference from theoretical z-coordinates
        xf = xf - np.median(xf, axis=1, keepdims=True)  # Adjust x-coordinates
        yf = yf - np.median(yf, axis=1, keepdims=True)  # Adjust y-coordinates
        if Nz > 4:
            zind = range(2, Nz - 2, 1)  # Indices for calculation
        else:
            zind = range(0, Nz, 1)
        zdiff = zdiff - np.mean(zdiff[:, zind], axis=1, keepdims=True)  # Adjust zdiff
        msez = np.mean(np.square((np.median(zf - zg, axis=0) - (zf - zg))[:, zind]), axis=1)  # Mean squared error in z
    else:
        zdiff = zf
        msez = np.array([1.0])  # Default error if single z-plane

    # Calculate msezRatio for single bead case
    if Nbead == 1:
        msezRatio = np.array([1.0])
    else:
        msezRatio = msez / (np.median(msez) + 1e-6)  # Ratio of errors

    # Plotting results if requested
    if plot & (Nz > 1):
        fig = plt.figure(figsize=[12, 6])  # Create figure for plots
        ax = fig.add_subplot(1, 2, 1)  # First subplot for z-coordinates
        plt.plot(zf.transpose(), color=(0.6, 0.6, 0.6))  # Plot z-coordinates
        plt.plot(np.linspace(0, Nz - 1, Nz))  # Plot theoretical line
        ax = fig.add_subplot(1, 2, 2)  # Second subplot for z-diff
        plt.plot((zdiff).transpose(), color=(0.6, 0.6, 0.6))  # Plot difference
        plt.plot(np.median(zdiff, axis=0), color='r')  # Plot median
        plt.plot(zg - zg, color='k')  # Zero line
        ax.set_ylim([-0.1, 0.1] / np.array([pixelsize_z]))  # Set y-limits
        plt.show()  # Show the plot

    # Create a dictionary to hold localization results
    loc_dict = dict(x=xf, y=yf, z=zf)  # Store x, y, z in dict

    return P, CRLB, LL, Iall, msezRatio, toc, loc_dict  # Return results


def loc_4pi(self, psf_data, I_model, A_model, pixelsize_z, cor, imgcenter, T, zT, initz=None, initphi=None, plot=False, start_time=0, linkxy=True):
    """
    Perform localization of beads using 4Pi microscopy model with maximum likelihood estimation.

    Parameters:
    - psf_data: np.ndarray
        The point spread function data with shape (Nchannel, Nz, rsz, rsz).
    - I_model: np.ndarray
        The model intensity image, the same shape as psf_data.
    - A_model: np.ndarray
        The model amplitude image for 4Pi microscopy.
    - pixelsize_z: float
        The size of each pixel in the z-dimension.
    - cor: np.ndarray
        The coordinates of the beads in the form of (Nchannel, Nbead, 2).
    - imgcenter: np.ndarray
        The center of the image for transformation, shape (2,).
    - T: np.ndarray
        Transformation matrix with shape (Nchannel, 3, 3).
    - zT: float
        Parameter for z-translation in 4Pi microscopy.
    - initz: np.ndarray, optional
        Initial z-coordinates for the beads.
    - initphi: np.ndarray, optional
        Initial phase angles for the 4Pi microscopy.
    - plot: bool, optional
        Whether to plot the results or not.
    - start_time: float, optional
        The start time for timing the operation.
    - linkxy: bool, optional
        Indicates whether to link x and y parameters during fitting.

    Returns:
    - P: np.ndarray
        Estimated parameters for the localization.
    - CRLB: np.ndarray
        Cramer-Rao lower bounds for the estimated parameters.
    - LL: np.ndarray
        Log-likelihood values for the estimated parameters.
    - IABall: np.ndarray
        Coefficients of the spline representation of the PSF.
    - msezRatio: np.ndarray
        Ratio of mean squared errors in z localization.
    - toc: float
        Total elapsed time for the localization process.
    - loc_dict: dict
        A dictionary containing the x, y, z coordinates of the localization and phase.
    """

    # Get sizes from input data
    rsz = psf_data.shape[-1]  # Size of the PSF
    Nbead = cor.shape[1]       # Number of beads
    Nchannel = cor.shape[0]    # Number of channels

    # Determine number of z-planes based on the shape of psf_data
    Nz = psf_data.shape[-3]  # Number of z-planes

    # Check for number of phases if psf_data has more than 5 dimensions
    if len(psf_data.shape) > 5:
        Nphase = psf_data.shape[-4]  # Number of phases
    else:
        Nphase = 1

    Nfit = Nbead * Nz * Nphase  # Total number of fitting parameters
    Nparam = 6  # Number of parameters to estimate (including phase)

    # Prepare the image model for processing
    offset = np.min(I_model - 2 * np.abs(A_model))  # Offset for normalization
    Imd = I_model - offset  # Subtract offset from the model
    normf = np.max(np.median(np.sum(Imd[:, 1:-1], axis=(-1, -2)), axis=-1)) * 2.0  # Normalization factor
    Imd = Imd / normf  # Normalize the model
    Amd = A_model / normf  # Normalize the amplitude model

    # Set up progress bar for spline coefficient calculation
    pbar = tqdm(total=Nchannel, desc='4/6: calculating spline coefficients',
                bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}s] {rate_fmt} {postfix[0]}{postfix[1][time]:>4.2f}s",
                postfix=["total time: ", dict(time=start_time)])

    IABall = []  # List to store all coefficients for I, A, and B
    # Calculate spline coefficients for each channel
    for i in range(Nchannel):
        Ii = Imd[i]  # Get the i-th model intensity
        Ai = 2 * np.real(Amd[i])  # Real part of amplitude
        Bi = -2 * np.imag(Amd[i])  # Imaginary part of amplitude
        IAB = [psf2cspline_np(Ai), psf2cspline_np(Bi), psf2cspline_np(Ii)]  # Get spline coefficients
        IAB = np.stack(IAB)  # Stack coefficients
        IABall.append(IAB)  # Append to list

# Update the progress bar with the current elapsed time
pbar.postfix[1]['time'] = start_time + pbar._time() - pbar.start_t  # Calculate the elapsed time
pbar.update(1)  # Increment the progress bar by 1

# Close progress bar for spline coefficients
toc = pbar.postfix[1]['time']  # Store the elapsed time for later use
pbar.close()  # Close the progress bar

# Stack IABall into a 4D array and convert the type to float32
IABall = np.stack(IABall).astype(np.float32)  
# Reshape the PSF data to have dimensions (Nchannel, Nfit, rsz, rsz)
data = psf_data.reshape((Nchannel, Nfit, rsz, rsz))  
# Determine the box size, limited to a maximum of 20
bxsz = np.min((rsz, 20))  
# Crop the data to focus on the center region and convert to float32
data = data[:, :, rsz // 2 - bxsz // 2:rsz // 2 + bxsz // 2, rsz // 2 - bxsz // 2:rsz // 2 + bxsz // 2].astype(np.float32)  
# Ensure that all values in data are non-negative
data = np.maximum(data, 0.0)  

# Concatenate correlation data with a column of ones for additional dimensionality
cor1 = np.concatenate((cor[0], np.ones((Nbead, 1))), axis=1)  
# Create a transformation matrix by concatenating an identity matrix with T
T1 = np.concatenate((np.expand_dims(np.eye(3), axis=0), T), axis=0)  
# Initialize arrays to hold x and y differences
dx1 = np.zeros((Nbead, Nchannel))  
dy1 = np.zeros((Nbead, Nchannel))  

# Loop over each channel to calculate coordinate changes
for i in range(Nchannel):
    cor2 = np.matmul(cor1 - imgcenter, T1[i]) + imgcenter  # Apply transformation
    dy1[:, i] = cor2[:, 0] - cor[i][:, 0]  # Calculate y differences
    dx1[:, i] = cor2[:, 1] - cor[i][:, 1]  # Calculate x differences

# Initialize a multidimensional array to store differences in parameters
dTS = np.zeros((Nbead, Nz * Nphase, Nchannel, Nparam))  
dTS[:, :, :, 0] = np.expand_dims(dx1, axis=1)  # Store x differences
dTS[:, :, :, 1] = np.expand_dims(dy1, axis=1)  # Store y differences
# Reshape dTS to 3D array and convert to float32
dTS = dTS.reshape((Nfit, Nchannel, Nparam)).astype(np.float32)  

# Determine if coordinates (x, y) should be linked across channels
if linkxy:
    shared = np.array([1, 1, 1, 1, 1, 1])  # All parameters are shared
else:
    shared = np.array([0, 0, 1, 1, 1, 1])  # Some parameters are not shared

# Prepare shared parameter array for further computations
sharedA = np.repeat(np.expand_dims(shared, axis=0), Nfit, axis=0).astype(np.int32)  

# Initialize an array for phase parameters
phic = np.array([0, 0, 0, 0])  
# Repeat phase parameters across all fits
phiA = np.repeat(np.expand_dims(phic, axis=0), Nfit, axis=0).astype(np.float32)  

# Calculate the center coordinate in the z dimension
ccz = IABall.shape[-3] // 2  
if initz is None:
    # Initialize z coordinates if not provided
    # Uncomment the following lines if the initialization based on image dimensions is needed
    # Nzm = Imd.shape[-3]
    # initz = np.linspace(-Nzm*pixelsize_z/2, Nzm*pixelsize_z/2, np.int32(Nzm*pixelsize_z/0.5)) * 0.8 / pixelsize_z + ccz
    initz = np.array([-1, 1]) * 0.15 / pixelsize_z + ccz  # Default initialization
else:
    initz = np.array(initz) * 0.15 / pixelsize_z + ccz  # Scale provided initialization

# Repeat initial z coordinates for all fits
zstart = np.repeat(np.expand_dims(initz, axis=1), Nfit, axis=1).astype(np.float32)  

# Initialize phase angle if not provided
if initphi is None:
    initphi = np.array([0, np.pi])  # Default angles
else:
    initphi = np.array(initphi)  # Use provided angles
# Repeat phase angles for all fits
phi_start = np.repeat(np.expand_dims(initphi, axis=1), Nfit, axis=1).astype(np.float32)  

# Store the sizes of the data and spline structures
datasize = np.array(np.flip(data.shape)).astype(np.int32)  
splinesize = np.array(np.flip(IABall.shape)).astype(np.int32)  

# Initialize arrays for parameters, CRLB (Cramér-Rao Lower Bound), and log-likelihood
Pk = np.zeros((Nparam + 1 + (Nchannel - 1) * (Nparam - np.sum(shared)), Nfit)).astype(np.float32)  
CRLBk = np.zeros((Nparam + (Nchannel - 1) * (Nparam - np.sum(shared)), Nfit)).astype(np.float32)  
LLk = np.zeros((Nfit)).astype(np.float32)  # Log-likelihoods
iterations = np.int32(50)  # Number of iterations for fitting
P = np.zeros((Nparam + 1 + (Nchannel - 1) * (Nparam - np.sum(shared)), Nfit)).astype(np.float32)  
CRLB = np.zeros((Nparam + (Nchannel - 1) * (Nparam - np.sum(shared)), Nfit)).astype(np.float32)  
LL = np.zeros((Nfit)).astype(np.float32) - 1e10  # Initialize log-likelihoods to a large negative value

maxN = 3000  # Maximum number of fits to process at once
Nf = np.ceil(Nfit / maxN).astype(np.int32)  # Calculate the number of batches
# Create an array defining the fit indices for each batch
vec = np.linspace(0, Nf * maxN, Nf + 1).astype(np.int32)  
vec[-1] = Nfit  # Ensure the last index equals Nfit

# Initialize the progress bar for the fitting process
pbar = tqdm(total=len(zstart) * len(phi_start), desc='5/6: localization', bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}s] {rate_fmt} {postfix[0]}{postfix[1][time]:>4.2f}s", postfix=["total time: ", dict(time=toc)])

# Loop through each z and phi to perform fitting
for z0 in zstart:
    for phi0 in phi_start:
        for i in range(Nf):
            nfit = vec[i + 1] - vec[i]  # Number of fits in the current batch
            # Initialize arrays for fitting parameters
            ph = np.zeros((Nparam + 1 + (Nchannel - 1) * (Nparam - np.sum(shared)), nfit)).astype(np.float32)  
            ch = np.zeros((Nparam + (Nchannel - 1) * (Nparam - np.sum(shared)), nfit)).astype(np.float32)  
            Lh = np.zeros((nfit)).astype(np.float32)  # Log-likelihood for the current batch

            # Copy relevant data for the current batch
            datai = np.copy(data[:, vec[i]:vec[i + 1]])  
            sharedi = np.copy(sharedA[vec[i]:vec[i + 1]])  
            dts = np.copy(dTS[vec[i]:vec[i + 1]])  
            phiAi = np.copy(phiA[vec[i]:vec[i + 1]])  
            z0i = np.copy(z0[vec[i]:vec[i + 1]])  
            phi0i = np.copy(phi0[vec[i]:vec[i + 1]])  
            datsz = np.array(np.flip(datai.shape)).astype(np.int32)  # Get the shape of the data batch

            # Call the maximum likelihood estimation fitting function
            self._mleFit_4Pi(datai, sharedi, iterations, IABall, dts, phiAi, z0i, phi0i, datsz, splinesize, ph, ch, Lh)  
            
            # Store the results of the fitting
            Pk[:, vec[i]:vec[i + 1]] = ph  
            CRLBk[:, vec[i]:vec[i + 1]] = ch  
            LLk[vec[i]:vec[i + 1]] = Lh  

        # Update the log-likelihoods and parameter estimates based on convergence criteria
        mask = (LLk - LL) > 1e-4  # Create a mask for significant updates
        LL[mask] = LLk[mask]  # Update log-likelihoods
        P[:, mask] = Pk[:, mask]  # Update parameters
        CRLB[:, mask] = CRLBk[:, mask]  # Update CRLBs
        
        # Update the progress bar with the current elapsed time
        pbar.postfix[1]['time'] = toc + pbar._time() - pbar.start_t  
        pbar.update(1)  # Increment the progress bar

# Finalize the closing of the progress bar
toc = pbar.postfix[1]['time']  
pbar.close()  

# Calculate final parameter estimates based on whether x and y coordinates are linked
if linkxy:
    yf = np.mean(P[0].reshape((Nbead, Nphase, Nz)), axis=1)  # Mean y values
    xf = np.mean(P[1].reshape((Nbead, Nphase, Nz)), axis=1)  # Mean x values
    zf = np.mean(P[4].reshape((Nbead, Nphase, Nz)), axis=1)  # Mean z values
    phif = np.mean(np.unwrap(P[5].reshape((Nbead, Nphase * Nz)), axis=1).reshape((Nbead, Nphase, Nz)), axis=1) * zT / 2 / np.pi  # Mean phi values
else:
    yf = P[0:4]  # Unlinked y values
    xf = P[4:8]  # Unlinked x values
    zf = P[10]  # z values
    phif = P[11] * zT / 2 / np.pi  # phi values

# Adjust zf and phif if there are multiple slices in the z dimension
if Nz > 1:
    zg = np.linspace(0, Nz - 1, Nz)  # Create a reference z grid
    zf = zf - np.median(zf - zg, axis=1, keepdims=True)  # Center z values
    phif = phif - np.median(phif - zg, axis=1, keepdims=True)  # Center phi values
    zdiff = zf - zg  # Calculate differences from the reference grid
    phidiff = phif - zg  # Calculate differences from the reference grid

    xf = xf - np.median(xf, axis=1, keepdims=True)  # Center x values
    yf = yf - np.median(yf, axis=1, keepdims=True)  # Center y values

    # Define indices for more robust statistics depending on Nz
    if Nz > 4:
        zind = range(2, Nz - 2, 1)  # Indices for robust filtering
    else:
        zind = range(0, Nz, 1)  # Use all indices for small Nz
    
    # Adjust differences to remove mean effects for robust measures
    zdiff = zdiff - np.mean(zdiff[:, zind], axis=1, keepdims=True)  
    phidiff = phidiff - np.mean(phidiff[:, zind], axis=1, keepdims=True)  
    msez = np.mean(np.square((np.median(zf - zg, axis=0) - (zf - zg))[:, zind]), axis=1)  # Mean squared error in z

else:
    msez = np.array([1.0])  # Default value if Nz is not greater than 1

# Calculate the ratio of mean squared errors
msezRatio = msez / (np.median(msez) + 1e-6)  
if plot & (Nz > 1):
    # Plotting if conditions are met
    fig = plt.figure(figsize=[12, 6])  # Create a figure with specified size
    ax = fig.add_subplot(2, 2, 1)  # First subplot for z values
    plt.plot(zf.transpose(), color=(0.6, 0.6, 0.6))  # Plot z values
    plt.plot(np.linspace(0, Nz - 1, Nz))  # Reference line
    ax = fig.add_subplot(2, 2, 2)  # Second subplot for phi values
    plt.plot(phif.transpose(), color=(0.6, 0.6, 0.6))  # Plot phi values
    plt.plot(np.linspace(0, Nz - 1, Nz))  # Reference line    
    ax = fig.add_subplot(2, 2, 3)  # Third subplot for z differences
    plt.plot((zdiff).transpose(), color=(0.6, 0.6, 0.6))  # Plot z differences
    plt.plot(np.median(zdiff, axis=0), color='r')  # Median line for z differences
    plt.plot(zg - zg, color='k')  # Reference line
    ax.set_ylim([-0.1, 0.1] / np.array([pixelsize_z]))  # Set y-limits
    ax.set_title('z')  # Title for z plot
    ax = fig.add_subplot(2, 2, 4)  # Fourth subplot for phi differences
    plt.plot((phidiff).transpose(), color=(0.6, 0.6, 0.6))  # Plot phi differences
    plt.plot(np.median(phidiff, axis=0), color='r')  # Median line for phi differences
    plt.plot(zg - zg, color='k')  # Reference line
    ax.set_ylim([-0.01, 0.01] / np.array([pixelsize_z]))  # Set y-limits
    ax.set_title('phi')  # Title for phi plot
    plt.show()  # Display the plots
    # Return a dictionary with localization results and relevant metrics
    loc_dict = dict(x=xf, y=yf, z=phif, zast=zf)  
return P, CRLB, LL, IABall, msezRatio, toc, loc_dict  # Return results


def loc_ast(self, psf_data, I_model, pixelsize_z, initz=None, plot=False, start_time=0):
    """
    Estimates the localization of point sources using the provided PSF data.

    Parameters:
    - psf_data (numpy.ndarray): 
        A 3D or 4D array representing the point spread function data, with dimensions corresponding to beads, z positions, and the spatial dimensions.
    - I_model (numpy.ndarray): 
        A 3D array (typically the same dimensions as the spatial dimensions of psf_data) representing the model intensity for the PSF.
    - pixelsize_z (float): 
        The size of a pixel in the z-dimension, used for localization calculations.
    - initz (numpy.ndarray, optional): 
        Initial z-positions for fitting. If None, it will be computed based on the shape of I_model.
    - plot (bool, optional): 
        If True, plots the results of the localization.
    - start_time (float, optional): 
        The starting time for tracking progress (for progress bar).

    Returns:
    - P (numpy.ndarray): 
        Parameters obtained from localization fitting.
    - CRLB (numpy.ndarray): 
        Cramer-Rao Lower Bound values associated with the parameters.
    - LL (numpy.ndarray): 
        Log-likelihood values from the fitting process.
    - coeff (numpy.ndarray): 
        Coefficients for the cubic spline representation of the PSF.
    - msezRatio (numpy.ndarray): 
        Mean square error ratio for the z localization.
    - toc (float): 
        Total elapsed time for the localization process.
    - loc_dict (dict): 
        A dictionary containing the x, y, and z localization results.
    """
    # Extracting dimensions from the PSF data for processing
    rsz = psf_data.shape[-1]  # Size of the PSF in spatial dimensions
    Nbead = psf_data.shape[0]  # Number of beads in the data
    
    # Determine the number of z slices; default to 1 if not present
    Nz = psf_data.shape[-3] if len(psf_data.shape) > 3 else 1
    Nfit = Nbead * Nz  # Total number of fitting parameters (beads * z slices)
    Nparam = 5  # Number of parameters to fit (x, y, z, intensity, and background offset)
    
    # Preprocessing the model intensity image
    offset = np.min(I_model)  # Minimum value in I_model (for background offset)
    Imd = I_model - offset  # Remove background
    normf = np.median(np.sum(Imd, axis=(-1, -2)))  # Normalize based on median intensity
    Imd = Imd / normf  # Normalize the image data
    
    # Setup progress bar for spline coefficient calculation
    pbar = tqdm(total=1, desc='4/6: calculating spline coefficients', 
                bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}s] {rate_fmt} {postfix[0]}{postfix[1][time]:>4.2f}s", 
                postfix=["total time: ", dict(time=start_time)])
                  
    # Calculate the spline coefficients from the PSF data
    coeff = psf2cspline_np(Imd)
                         
    # Update the progress bar with the time taken
    pbar.postfix[1]['time'] = start_time + pbar._time() - pbar.start_t    
    pbar.update(1)
    toc = pbar.postfix[1]['time']    
    pbar.close()  # Close the progress bar
    
    # Prepare data for fitting
    coeff = coeff.astype(np.float32)  # Ensure coefficients are of type float32
    data = psf_data.reshape((Nfit, rsz, rsz))  # Reshape PSF data for fitting
    bxsz = np.min((rsz, 20))  # Determine the box size for cropping
    # Crop the PSF data to the center
    data = data[:, rsz // 2 - bxsz // 2:rsz // 2 + bxsz // 2, rsz // 2 - bxsz // 2:rsz // 2 + bxsz // 2].astype(np.float32)
    data = np.maximum(data, 0.0)  # Ensure no negative values in data
    
    # Initialize z position array
    ccz = coeff.shape[-3] // 2  # Center index for z
    if initz is None:  # If no initial z positions are provided
        Nzm = Imd.shape[0]  # Number of z positions
        # Generate initial z positions based on pixel size and center
        initz = np.linspace(-Nzm * pixelsize_z / 2, Nzm * pixelsize_z / 2, np.int32(Nzm * pixelsize_z / 0.5)) * 0.8 / pixelsize_z + ccz
    else:
        initz = np.array(initz) * 0.5 / pixelsize_z + ccz  # Scale provided initial z positions
    zstart = initz.astype(np.float32)  # Convert to float32

    # Prepare arrays for fitting results
    datasize = np.array(np.flip(data.shape)).astype(np.int32)  # Size of the data for fitting
    splinesize = np.array(np.flip(coeff.shape)).astype(np.int32)  # Size of the spline coefficients
    varim = np.array((0)).astype(np.float32)  # Variance image placeholder
    Pk = np.zeros((Nparam + 1, Nfit)).astype(np.float32)  # Parameter estimates
    CRLBk = np.zeros((Nparam, Nfit)).astype(np.float32)  # Cramer-Rao Lower Bounds
    LLk = np.zeros((Nfit)).astype(np.float32)  # Log-likelihoods
    fittype = np.int32(5)  # Type of fitting method
    iterations = np.int32(100)  # Number of iterations for fitting
    P = np.zeros((Nparam + 1, Nfit)).astype(np.float32)  # Final parameter estimates
    CRLB = np.zeros((Nparam, Nfit)).astype(np.float32)  # Final Cramer-Rao Lower Bounds
    LL = np.zeros((Nfit)).astype(np.float32) - 1e10  # Initialize log-likelihoods to a very low value
    
    # Setup progress bar for localization process
    pbar = tqdm(total=len(zstart), desc='5/6: localization', 
                bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}s] {rate_fmt} {postfix[0]}{postfix[1][time]:>4.2f}s", 
                postfix=["total time: ", dict(time=start_time)])

    # Loop through each initial z position to perform fitting
    for z0 in zstart:
        # Perform maximum likelihood estimation fitting
        self._mleFit(data, fittype, iterations, coeff, varim, z0, datasize, splinesize, Pk, CRLBk, LLk)
        mask = (LLk - LL) > 1e-4  # Identify significant updates in log-likelihood
        LL[mask] = LLk[mask]  # Update log-likelihoods
        P[:, mask] = Pk[:, mask]  # Update parameters
        CRLB[:, mask] = CRLBk[:, mask]  # Update Cramer-Rao Lower Bounds
        
        # Update the progress bar with time taken
        pbar.postfix[1]['time'] = toc + pbar._time() - pbar.start_t    
        pbar.update(1)  # Increment the progress bar

    toc = pbar.postfix[1]['time']  # Total time after localization
    pbar.close()  # Close the progress bar

    # Reshape and adjust the fitting results for output
    zf = P[4].reshape((Nbead, Nz))  # Reshape z fitting results
    xf = P[1].reshape((Nbead, Nz))  # Reshape x fitting results
    yf = P[0].reshape((Nbead, Nz))  # Reshape y fitting results

    zg = np.linspace(0, Nz - 1, Nz)  # Create a reference z position array
    if Nz > 1:
        # Center the z fitting results around the median
        zf = zf - np.median(zf - zg, axis=1, keepdims=True)
        zdiff = zf - zg  # Calculate differences from reference z
        xf = xf - np.median(xf, axis=1, keepdims=True)  # Center x positions
        yf = yf - np.median(yf, axis=1, keepdims=True)  # Center y positions
        zind = range(2, Nz - 2, 1) if Nz > 4 else range(0, Nz, 1)  # Index range for mean calculation
        
        # Adjust the differences based on mean over specified indices
        zdiff = zdiff - np.mean(zdiff[:, zind], axis=1, keepdims=True)
        msez = np.mean(np.square((np.median(zf - zg, axis=0) - (zf - zg))[:, zind]), axis=1)  # Mean square error for z
    else:
        zdiff = zf  # No adjustment needed for single z layer
        msez = np.array([1.0])  # Default MSE value for single z layer

    # Calculate the ratio of MSE for z localization
    msezRatio = np.array([1.0]) if Nbead == 1 else msez / (np.median(msez) + 1e-6)

    # If plotting is enabled and there are multiple z layers
    if plot & (Nz > 1):
        fig = plt.figure(figsize=[12, 6])  # Create a figure for plotting
        ax = fig.add_subplot(1, 2, 1)  # First subplot for z positions
        plt.plot(zf.transpose(), color=(0.6, 0.6, 0.6))  # Plot the localized z positions
        plt.plot(zg)  # Plot the reference z positions
        ax.set_title('z')  # Set title for the z plot
        
        ax = fig.add_subplot(1, 2, 2)  # Second subplot for z differences
        plt.plot((zdiff).transpose(), color=(0.6, 0.6, 0.6))  # Plot the z differences
        plt.plot(np.median(zdiff, axis=0), color='r')  # Plot the median difference
        plt.plot(zg - zg, color='k')  # Reference line at zero
        ax.set_ylim([-0.1, 0.1] / np.array([pixelsize_z]))  # Set y-limits
        plt.show()  # Display the plots

    # Prepare localization results in a dictionary
    loc_dict = dict(x=xf, y=yf, z=zf)  # Dictionary for x, y, and z results

    return P, CRLB, LL, coeff, msezRatio, toc, loc_dict  # Return all computed results