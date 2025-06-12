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
Shift a NumPy array without circular wrapping.

Args:
    arr (numpy.ndarray): The input array to shift.
    shift_x (int): The horizontal shift (positive values shift right, negative left).
    shift_y (int): The vertical shift (positive values shift down, negative up).

Returns:
    numpy.ndarray: The shifted array.
"""
# Import necessary libraries
import numpy as np  # For numerical operations
from typing import Tuple  # To specify types for function parameters

from tifffile import imread, imwrite  # For reading and writing TIFF files
import imageio.v3 as iio  # For reading PNG files
from scipy.signal import fftconvolve  # For convolution operations

def read_tiff(path: str) -> np.ndarray:
    """Read a .tiff image into a numpy array.

    Args:
        path (str): Path to the image.

    Returns:
        np.ndarray: Numpy array representation of the TIFF image.
    """
    return imread(path)  # Read the TIFF image using tifffile

def read_png(path: str) -> np.ndarray:
    """Read a .png image into a numpy array.

    Args:
        path (str): Path to the image.

    Returns:
        np.ndarray: Numpy array representation of the PNG image.
    """
    return iio.imread(path)  # Read the PNG image using imageio

def uint8_to_float(x: np.ndarray) -> np.ndarray:
    """Convert an 8-bit unsigned integer array to a float array in the range [0, 1].

    Args:
        x (np.ndarray): Input array in uint8 format (0-255).

    Returns:
        np.ndarray: Converted array in float32 format, scaled to [0, 1].
    """
    x_float = x.astype(np.float32)  # Convert array to float32
    x_float /= 0xFF  # Scale to [0, 1] by dividing by 255
    return x_float

def full_read_tiff(data_path: str) -> np.ndarray:
    """Read and normalize TIFF files into a float array.

    Args:
        data_path (str): Path to the TIFF file.

    Returns:
        np.ndarray: Normalized numpy array of the TIFF image.
    """
    return uint8_to_float(read_tiff(data_path))  # Read and normalize TIFF

def full_read(data_path: str) -> np.ndarray:
    """Combine reading and normalizing for both TIFF and PNG files.

    Args:
        data_path (str): Path to the image file (TIFF or PNG).

    Returns:
        np.ndarray: Normalized numpy array of the image.
    """
    if data_path.lower().endswith((".tiff", ".tif")):
        return uint8_to_float(read_tiff(data_path))  # For TIFF files
    if data_path.lower().endswith(".png"):
        return uint8_to_float(read_png(data_path))  # For PNG files

def write_tiff(x: np.ndarray, path: str) -> None:
    """Write a numpy array to a TIFF file in 8-bit format.

    Args:
        x (np.ndarray): The numpy array you want to save (in float format).
        path (str): Path where the TIFF file will be saved.
    """
    x = (255 * (linear_normalize(x))).astype("uint8")  # Normalize and convert to uint8
    imwrite(path, x)  # Write the array to a TIFF file

def write_tiff16(x: np.ndarray, path: str) -> None:
    """Write a numpy array to a TIFF file in 16-bit format.

    Args:
        x (np.ndarray): The numpy array you want to save (in float format).
        path (str): Path where the TIFF file will be saved.
    """
    x = (0xFFFF * (linear_normalize(x))).astype("uint16")  # Normalize and convert to uint16
    imwrite(path, x)  # Write the array to a TIFF file

def linear_normalize(x: np.ndarray) -> np.ndarray:
    """Linearly normalize the array to the range [0, 1].

    Args:
        x (np.ndarray): Any sized array.

    Returns:
        np.ndarray: Same sized array, normalized to [0, 1].
    """
    return (x - x.min()) / (x.max() - x.min() + np.finfo(float).eps)  # Normalize

def clip(x: np.ndarray, low: float, high: float) -> np.ndarray:
    """Clip the array values to a specified range.

    Args:
        x (np.ndarray): Array to be clipped.
        low (float): Minimum value; values below this will be set to this value.
        high (float): Maximum value; values above this will be set to this value.

    Returns:
        np.ndarray: Clipped array with values in the range [low, high].
    """
    return np.minimum(np.maximum(x, low), high)  # Clip values to [low, high]

def crop(arr: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    """Crop the center of the array to specified dimensions.

    Args:
        arr (np.ndarray): Input array to be cropped.
        new_height (int): Desired height of the cropped array.
        new_width (int): Desired width of the cropped array.

    Returns:
        np.ndarray: Cropped array with specified dimensions.
    """
    height, width = arr.shape  # Get current dimensions

    # Calculate starting indices for cropping
    start_row = (height - new_height) // 2
    start_col = (width - new_width) // 2

    # Crop the array
    cropped_arr = arr[
        start_row : start_row + new_height, start_col : start_col + new_width
    ]

    return cropped_arr  # Return the cropped array

def pad_3d_array_to_size(
    arr: np.ndarray, target_shape: Tuple[int, int, int]
) -> np.ndarray:
    """Pad a 3D NumPy array to the desired shape along the last two dimensions with zeros.

    Args:
        arr (np.ndarray): 3D NumPy array to be padded.
        target_shape (Tuple[int, int, int]): Desired shape in the format (depth, height, width).

    Returns:
        np.ndarray: Padded 3D NumPy array of the desired shape.
    """
    d, r, c = arr.shape  # Get current shape dimensions
    d_target, H, W = target_shape  # Unpack target shape

    # Calculate padding needed for each dimension
    pad_height = max(0, H - r)
    pad_width = max(0, W - c)

    # Calculate padding for each side
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Create a new array with the target shape, initialized with zeros
    padded_array = np.zeros((d_target, H, W), dtype=arr.dtype)

    # Pad each 2D slice of the original array
    for i in range(d):
        padded_array[i] = np.pad(
            arr[i],
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )

    return padded_array  # Return the padded array

def pad0(arr: np.ndarray) -> np.ndarray:
    """Pad a 2D numpy array for fft2d operation.

    Args:
        arr (np.ndarray): Array to be padded.

    Returns:
        np.ndarray: Padded array to the appropriate size for FFT operations.
    """
    pad_height = arr.shape[0] // 2  # Calculate half height for padding
    pad_width = arr.shape[1] // 2  # Calculate half width for padding
    pad_tuple = ((pad_height, pad_height), (pad_width, pad_width))  # Padding specification

    # Pad the array with zeros
    padded_arr = np.pad(
        arr,
        pad_tuple,
        mode="constant",
        constant_values=0,
    )

    return padded_arr  # Return the padded array

def fft2d(x: np.ndarray) -> np.ndarray:
    """Compute the 2D Fast Fourier Transform of the input array.

    Args:
        x (np.ndarray): Input array to transform.

    Returns:
        np.ndarray: Complex-valued 2D Fast Fourier Transform of the input.
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))  # Perform FFT and shift

def ifft2d(x: np.ndarray) -> np.ndarray:
    """Compute the 2D Inverse Fast Fourier Transform of the input array.

    Args:
        x (np.ndarray): Input array to transform.

    Returns:
        np.ndarray: Complex-valued 2D Inverse Fast Fourier Transform of the input.
    """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))  # Perform inverse FFT and shift

def power_normalize(x: np.ndarray) -> np.ndarray:
    """Normalize the array such that its elements sum to 1.

    Args:
        x (np.ndarray): Any real-valued numpy array.

    Returns:
        np.ndarray: Array normalized so that all elements sum to 1.
    """
    return x / np.sum(x)  # Divide by the sum of the array elements

def normalize_psf_power(psf: np.ndarray) -> np.ndarray:
    """Normalize each slice of a PSF (Point Spread Function) by its power.

    Args:
        psf (np.ndarray): 3D stack of PSFs or a single 2D PSF.

    Returns:
        np.ndarray: Power-normalized PSF.
    """
    if psf.ndim == 2:
        return power_normalize(psf)  # Normalize if it's a single 2D PSF

    # Normalize each slice if it's a 3D PSF
    for z in range(psf.shape[0]):
        tmp = psf[z, :, :]
        psf[z, :, :] = power_normalize(tmp)
    return psf  # Return normalized PSF

def conv2d(obj: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Perform 2D convolution between an object and a kernel.

    Args:
        obj (np.ndarray): Input object array to be convolved.
        kernel (np.ndarray): Convolution kernel (filter).

    Returns:
        np.ndarray: The result of the convolution between the object and the kernel.
    """
    assert obj.shape == kernel.shape  # Ensure shapes match for convolution
    return fftconvolve(obj, kernel, mode="same")  # Perform convolution using FFT

def lsi_fwd_mdl(obj: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Linear shift-invariant (LSI) forward model to propagate an object to the image plane via a PSF.

    Args:
        obj (np.ndarray): Object to be propagated, can be 2D or 3D (z, x, y).
        psf (np.ndarray): Point Spread Function (PSF); must match the object shape.

    Returns:
        np.ndarray: LSI measurement of the object.
    """
    meas = np.zeros((obj.shape[1:]))  # Initialize measurement array
    for z in range(obj.shape[0]):
        meas += conv2d(obj[z, :, :], psf[z, :, :])  # Convolve each slice with the PSF
    return meas / np.max(meas)  # Normalize the measurement

def process_slice(z, obj, psf):
    """Process a single slice by applying convolution with the PSF.

    Args:
        z: Index of the slice to process.
        obj: The object array.
        psf: The Point Spread Function array.

    Returns:
        Result of convolving the specified slice of the object with the PSF.
    """
    return conv2d(obj[z, :, :], psf[z, :, :])  # Convolve the specified slice

def shift_array(arr, shift_x, shift_y):
    """Shift a NumPy array without circular wrapping.

    Args:
        arr (numpy.ndarray): The input array to shift.
        shift_x (int): The horizontal shift (positive values shift right, negative left).
        shift_y (int): The vertical shift (positive values shift down, negative up).

    Returns:
        numpy.ndarray: The shifted array.
    """
    if shift_x == 0 and shift_y == 0:
        return arr  # No shift needed

    h, w = arr.shape  # Get array dimensions
    shifted_arr = np.zeros_like(arr)  # Initialize an array of the same shape

    # Calculate slices for rows and columns based on shift values
    if shift_x >= 0:
        x_start_src, x_end_src, x_start_dst, x_end_dst = 0, w - shift_x, shift_x, w
    else:
        x_start_src, x_end_src, x_start_dst, x_end_dst = -shift_x, w, 0, w + shift_x

    if shift_y >= 0:
        y_start_src, y_end_src, y_start_dst, y_end_dst = 0, h - shift_y, shift_y, h
    else:
        y_start_src, y_end_src, y_start_dst, y_end_dst = -shift_y, h, 0, h + shift_y

    # Copy the shifted region of the input array to the destination
    shifted_arr[y_start_dst:y_end_dst, x_start_dst:x_end_dst] = arr[
        y_start_src:y_end_src, x_start_src:x_end_src
    ]

    return shifted_arr  # Return the shifted array