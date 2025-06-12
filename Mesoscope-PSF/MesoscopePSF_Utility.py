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

import tensorflow as tf  # Import TensorFlow for deep learning and tensor operations
import numpy as np  # Import NumPy for numerical operations
import scipy as sp  # Import SciPy for scientific computations
from math import factorial  # Import factorial function from math module
from . import imagetools as nip  # Import custom imagetools module as nip
import numbers  # Import numbers module for numerical type checking
from scipy import ndimage  # Import ndimage for image processing functions
import scipy.fft as fft  # Import FFT functions from SciPy

# Define default data types for TensorFlow operations
defaultTFDataType = "float32"  # Default TensorFlow data type for floating point numbers
defaultTFCpxDataType = "complex64"  # Default TensorFlow data type for complex numbers

#%%
# The functions below are Tensorflow now

def fft3d(tfin):
    """
    Perform a 3D Fast Fourier Transform (FFT) on the input tensor.
    
    Parameters:
    tfin (tf.Tensor): Input tensor of any shape on which FFT is to be applied.
    
    Returns:
    tf.Tensor: The 3D FFT of the input tensor, centered using fftshift.
    """
    return tf.signal.fftshift(tf.signal.fft3d(tf.signal.fftshift(tfin, axes=[-1, -2, -3])), axes=[-1, -2, -3])

def ifft3d(tfin):
    """
    Perform a 3D Inverse Fast Fourier Transform (IFFT) on the input tensor.
    
    Parameters:
    tfin (tf.Tensor): Input tensor of any shape on which IFFT is to be applied.
    
    Returns:
    tf.Tensor: The 3D IFFT of the input tensor, centered using ifftshift.
    """
    return tf.signal.ifftshift(tf.signal.ifft3d(tf.signal.ifftshift(tfin, axes=[-1, -2, -3])), axes=[-1, -2, -3])

def fft2d(tfin):
    """
    Perform a 2D Fast Fourier Transform (FFT) on the input tensor.
    
    Parameters:
    tfin (tf.Tensor): Input tensor of any shape on which FFT is to be applied.
    
    Returns:
    tf.Tensor: The 2D FFT of the input tensor, centered using fftshift.
    """
    return tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(tfin, axes=[-1, -2])), axes=[-1, -2])

def ifft2d(tfin):
    """
    Perform a 2D Inverse Fast Fourier Transform (IFFT) on the input tensor.
    
    Parameters:
    tfin (tf.Tensor): Input tensor of any shape on which IFFT is to be applied.
    
    Returns:
    tf.Tensor: The 2D IFFT of the input tensor, centered using ifftshift.
    """
    return tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(tfin, axes=[-1, -2])), axes=[-1, -2])

def psf2cspline_np(psf):
    """
    Convert a Point Spread Function (PSF) to cubic spline coefficients.
    
    Parameters:
    psf (np.ndarray): Input PSF array with shape (depth, height, width).
    
    Returns:
    np.ndarray: Array of cubic spline coefficients.
    """
    # Initialize a zero array of shape (64, 64)
    A = np.zeros((64, 64))
    # Loop to fill the matrix A based on the grid values
    for i in range(1, 5):
        dx = (i - 1) / 3
        for j in range(1, 5):
            dy = (j - 1) / 3
            for k in range(1, 5):
                dz = (k - 1) / 3
                for l in range(1, 5):
                    for m in range(1, 5):
                        for n in range(1, 5):
                            A[(i - 1) * 16 + (j - 1) * 4 + k - 1, (l - 1) * 16 + (m - 1) * 4 + n - 1] = dx ** (l - 1) * dy ** (m - 1) * dz ** (n - 1)

    # Upsample the PSF by a factor of 3
    psf_up = ndimage.zoom(psf, 3.0, mode='grid-constant', grid_mode=True)[1:-1, 1:-1, 1:-1]
    A = np.float32(A)  # Ensure A is of type float32
    # Calculate the spline coefficients using the matrix A and the upsampled PSF
    coeff = calsplinecoeff(A, psf, psf_up)
    return coeff

def calsplinecoeff(A, psf, psf_up):
    """
    Calculate cubic spline coefficients for the given PSF.
    
    Parameters:
    A (np.ndarray): Coefficient matrix of shape (64, 64).
    psf (np.ndarray): Original PSF array for reference.
    psf_up (np.ndarray): Upsampled PSF array.
    
    Returns:
    np.ndarray: Array containing the cubic spline coefficients.
    """
    # Initialize coefficients array with appropriate shape
    coeff = np.zeros((64, psf.shape[0] - 1, psf.shape[1] - 1, psf.shape[2] - 1))
    # Loop through each voxel in the upsampled PSF to calculate coefficients
    for i in range(coeff.shape[1]):
        for j in range(coeff.shape[2]):
            for k in range(coeff.shape[3]):
                temp = psf_up[i * 3: 3 * (i + 1) + 1, j * 3: 3 * (j + 1) + 1, k * 3: 3 * (k + 1) + 1]
                # Solve the linear system A * x = temp to find spline coefficients
                x = sp.linalg.solve(A, temp.flatten())
                coeff[:, i, j, k] = x  # Store coefficients for this voxel

    return coeff

def nl2noll(n, l):
    """
    Convert (n, l) indices to Noll index j.
    
    Parameters:
    n (int): Radial index.
    l (int): Azimuthal index.
    
    Returns:
    int: The corresponding Noll index j.
    """
    mm = abs(l)  # Absolute value of l
    j = n * (n + 1) / 2 + 1 + max(0, mm - 1)  # Base calculation for j
    # Adjust j based on the parity of n and l
    if ((l > 0) & (np.mod(n, 4) >= 2)) | ((l < 0) & (np.mod(n, 4) <= 1)):
        j = j + 1
    
    return np.int32(j)  # Convert j to int32

def noll2nl(j):
    """
    Convert Noll index j to (n, l) indices.
    
    Parameters:
    j (int): Noll index.
    
    Returns:
    tuple: (n, l) indices corresponding to the given Noll index.
    """
    n = np.ceil((-3 + np.sqrt(1 + 8 * j)) / 2)  # Calculate n
    l = j - n * (n + 1) / 2 - 1  # Calculate l
    # Adjust l based on the parity of n
    if np.mod(n, 2) != np.mod(l, 2):
        l = l + 1
    
    # Adjust l based on the parity of j
    if np.mod(j, 2) == 1:
        l = -l
    
    return np.int32(n), np.int32(l)  # Return n and l as int32

def radialpoly(n, m, rho):
    """
    Calculate the radial polynomial for given n, m and radius rho.
    
    Parameters:
    n (int): Degree of the polynomial.
    m (int): Order of the polynomial.
    rho (np.ndarray): Array representing the radial coordinate.
    
    Returns:
    np.ndarray: Computed radial polynomial values.
    """
    # Calculate normalization factor based on m
    if m == 0:
        g = np.sqrt(n + 1)
    else:
        g = np.sqrt(2 * n + 2)
    r = np.zeros(rho.shape)  # Initialize output array
    # Loop to calculate the polynomial
    for k in range(0, (n - m) // 2 + 1):
        coeff = g * ((-1) ** k) * factorial(n - k) / factorial(k) / factorial((n + m) // 2 - k) / factorial((n - m) // 2 - k)
        p = rho ** (n - 2 * k)
        r += coeff * p  # Accumulate the result

    return r

def genZern1(n_max, xsz):
    """
    Generate Zernike polynomials up to degree n_max.
    
    Parameters:
    n_max (int): Maximum degree of Zernike polynomials to generate.
    xsz (int): Size of the output array (xsz x xsz).
    
    Returns:
    np.ndarray: Array containing the Zernike polynomials.
    """
    Nk = (n_max + 1) * (n_max + 2) // 2  # Calculate number of Zernike polynomials
    Z = np.ones((Nk, xsz, xsz))  # Initialize Zernike polynomial array
    pkx = 2 / xsz  # Scaling factor
    xrange = np.linspace(-xsz / 2 + 0.5, xsz / 2 - 0.5, xsz)  # Generate x range
    [xx, yy] = np.meshgrid(xrange, xrange)  # Create meshgrid from x range
    rho = np.lib.scimath.sqrt((xx * pkx) ** 2 + (yy * pkx) ** 2)  # Calculate radius
    phi = np.arctan2(yy, xx)  # Calculate angle

    # Loop through to generate each Zernike polynomial
    for j in range(0, Nk):
        [n, l] = noll2nl(j + 1)  # Convert Noll index to (n, l)
        m = np.abs(l)  # Absolute value of l
        r = radialpoly(n, m, rho)  # Calculate radial polynomial
        # Assign Zernike polynomial based on the sign of l
        if l < 0:
            Z[j] = r * np.sin(phi * m)
        else:
            Z[j] = r * np.cos(phi * m)
    return Z  # Return the array of Zernike polynomials

def prechirpz1(kpixelsize, pixelsize_x, pixelsize_y, N, M):
    """
    Prepare matrices for the chirp Z-transform.
    
    Parameters:
    kpixelsize (float): Pixel size in k-space.
    pixelsize_x (float): Pixel size in the x direction.
    pixelsize_y (float): Pixel size in the y direction.
    N (int): Number of rows/columns in the k-space matrix.
    M (int): Number of rows/columns in the spatial matrix.
    
    Returns:
    tuple: (A, Bh, C) where A is the transformation matrix, Bh is the FFT of B, and C is the spatial transformation matrix.
    """
    krange = np.linspace(-N / 2 + 0.5, N / 2 - 0.5, N, dtype=np.float32)  # Create k-space range
    [xxK, yyK] = np.meshgrid(krange, krange)  # Create meshgrid for k-space
    xrange = np.linspace(-M / 2 + 0.5, M / 2 - 0.5, M, dtype=np.float32)  # Create spatial range
    [xxR, yyR] = np.meshgrid(xrange, xrange)  # Create meshgrid for spatial
    a = 1j * np.pi * kpixelsize  # Calculate the complex exponent coefficient
    A = np.exp(a * (pixelsize_x * xxK * xxK + pixelsize_y * yyK * yyK))  # Transformation matrix
    C = np.exp(a * (pixelsize_x * xxR * xxR + pixelsize_y * yyR * yyR))  # Spatial transformation matrix

    brange = np.linspace(-(N + M) / 2 + 1, (N + M) / 2 - 1, N + M - 1, dtype=np.float32)  # Range for B matrix
    [xxB, yyB] = np.meshgrid(brange, brange)  # Create meshgrid for B
    B = np.exp(-a * (pixelsize_x * xxB * xxB + pixelsize_y * yyB * yyB))  # Matrix B
    Bh = tf.signal.fft2d(B)  # FFT of matrix B

    return A, Bh, C  # Return the matrices for further processing

def cztfunc1(datain, param):
    """
    Perform the chirp Z-transform using the provided parameters.
    
    Parameters:
    datain (tf.Tensor): Input tensor data to transform.
    param (tuple): Contains (A, Bh, C) matrices necessary for the transformation.
    
    Returns:
    tf.Tensor: Output tensor after applying the chirp Z-transform.
    """
    A = param[0]  # Transformation matrix
    Bh = param[1]  # FFT of B matrix
    C = param[2]  # Spatial transformation matrix
    N = A.shape[0]  # Size of A
    L = Bh.shape[0]  # Size of Bh
    M = C.shape[0]  # Size of C

    # Pad the input data and perform FFT
    Apad = tf.concat((A * datain / N, tf.zeros(datain.shape[0:-1] + (L - N), tf.complex64)), axis=-1)
    Apad = tf.concat((Apad, tf.zeros(Apad.shape[0:-2] + (L - N, Apad.shape[-1]), tf.complex64)), axis=-2)
    Ah = tf.signal.fft2d(Apad)  # FFT of padded matrix
    cztout = tf.signal.ifft2d(Ah * Bh / L)  # Apply inverse FFT
    dataout = C * cztout[..., -M:, -M:]  # Multiply with C and extract the necessary area

    return dataout  # Return the transformed data output