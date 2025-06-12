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
Zeros out slices in a 3D array along the first axis that are not specified in the list of indices to keep.

Args:
    arr (numpy.ndarray): The 3D array with shape (c, h, w) where c is the number of slices.
    indices_to_keep (list): List of integers representing the indices along the c-axis to retain.

Returns:
    numpy.ndarray: A modified array where slices not indexed in the provided list are set to zero.
"""
import numpy as np
from typing import Tuple, List
from skimage.feature import peak_local_max
from sbrnet_core.utils.constants import CM2_SIZE, FOCUS_LOC, NUM_VIEWS
from sbrnet_core.utils import (
    linear_normalize,
    read_tiff,
    uint8_to_float,
    normalize_psf_power,
    shift_array,
)

# Functions
def load_data(path: str) -> np.ndarray:
    """Loads and normalizes data from a TIFF file.

    Args:
        path (str): Path to the TIFF data file.

    Returns:
        np.ndarray: Normalized image data as a numpy array.
    """
    return uint8_to_float(read_tiff(path))


def load_psf(path: str) -> np.ndarray:
    """Loads and prepares the PSF (Point Spread Function) stack.

    Args:
        path (str): Path to the PSF file.

    Returns:
        np.ndarray: Normalized PSF stack.
    """
    return normalize_psf_power(linear_normalize(read_tiff(path)))


def crop_views(im: np.ndarray, crop_size: int = 512) -> np.ndarray:
    """Crops a given large image into smaller views based on predefined focus locations.

    Args:
        im (np.ndarray): The original image of size 2076x3088.
        crop_size (int, optional): The side length of the square crop, defaults to 512.

    Returns:
        np.ndarray: A 3D array containing cropped views of shape [NUM_VIEWS, crop_size, crop_size].
    """
    stack = np.zeros((NUM_VIEWS, crop_size, crop_size))  # Preallocate the stack for cropped views
    for i, point in enumerate(FOCUS_LOC):
        x, y = point
        x_min = x - crop_size // 2
        x_max = x + crop_size // 2
        y_min = y - crop_size // 2
        y_max = y + crop_size // 2

        # Crop the region from the larger array and store it in the 3D numpy array
        stack[i, :, :] = im[x_min:x_max, y_min:y_max]
    return stack


def get_coord_max(im: np.ndarray) -> np.ndarray:
    """Finds the coordinates of local maxima in the image, filtering out low values.

    Args:
        im (np.ndarray): Input image from which to find the maxima.

    Returns:
        np.ndarray: Array of coordinates of the local maxima.
    """
    fixed_meas = im.copy()  # Create a copy of the image to avoid modifying the original
    fixed_meas[im <= 0.25] = 0  # Filter out values below 0.25 to remove noise
    return peak_local_max(fixed_meas, min_distance=5)  # Find and return local maxima coordinates


def get_meas_mean(im: np.ndarray) -> np.float32:
    """Computes the average value of the peaks in the freespace measurement image.

    Args:
        im (np.ndarray): Input image (2076x3088) from which to compute the average of the peaks.

    Returns:
        np.float32: Average value of all the peak signals in the image.
    """
    coords = get_coord_max(im)  # Get coordinates of local max peaks

    # Extract pixel values for the local maxima coordinates
    pixel_values = [im[x, y] for x, y in coords]

    # Compute and return the average of the pixel values
    return np.mean(pixel_values)


def get_background_mean(im: np.ndarray) -> np.float32:
    """Calculates the mean value of a noise image.

    Args:
        im (np.ndarray): Value noise sample, typically of size 600x600.

    Returns:
        np.float32: The mean of all the values in the noise image.
    """
    return np.mean(im)  # Compute and return the mean of the noise image


def make_bg_img(
    value_img: np.ndarray,
    lens_ap: np.ndarray,
    mla_ap: np.ndarray,
    crop_size: int = 600,
) -> Tuple[np.ndarray, np.float32]:
    """Generates a background image using a synthetic background and apodization masks.

    Args:
        value_img (np.ndarray): Synthetic background of size 600x600.
        lens_ap (np.ndarray): Gaussian apodization mask for lens Field of View (FOV).
        mla_ap (np.ndarray): Apodization mask for Micro Lens Array (MLA).
        crop_size (int, optional): Size of the crops to be taken, defaults to 600.

    Returns:
        Tuple[np.ndarray, np.float32]: A tuple containing the normalized background measurement image
        and its mean value.
    """
    assert value_img.shape == lens_ap.shape  # Ensure the shapes of value image and lens apodization mask match
    bg_mask = np.zeros((CM2_SIZE))  # Preallocate the background mask array
    for point in FOCUS_LOC:
        x, y = point
        x_min = x - crop_size // 2
        x_max = x + crop_size // 2
        y_min = y - crop_size // 2
        y_max = y + crop_size // 2

        # Apply the lens apodization to the synthetic background at the specified focus points
        bg_mask[x_min:x_max, y_min:y_max] = value_img * lens_ap

    bg_mean = get_background_mean(value_img)  # Calculate the mean of the background image
    return linear_normalize(bg_mask * mla_ap), bg_mean  # Return the normalized background and its mean


def make_measurement(
    freespace_img: np.ndarray,
    bg_img: np.ndarray,
    SBR: np.float32,
    bg_mean: np.float32,
) -> np.ndarray:
    """Creates a scattering measurement from freespace and background images.

    Args:
        freespace_img (np.ndarray): Image representing the convolution of freespace PSF and ground truth volume.
        bg_img (np.ndarray): Background image with applied apodization.
        SBR (np.float32): The signal to background ratio.
        bg_mean (np.float32): The mean value of the raw background image.

    Returns:
        np.ndarray: Scattering measurement simulation incorporating value noise.
    """
    meas_mean = get_meas_mean(freespace_img)  # Compute the mean of the measurement image
    S = (bg_mean * SBR - bg_mean) / meas_mean  # Calculate the scaling factor for the freespace image
    scattering_meas = linear_normalize(S * freespace_img + bg_img)  # Create the final scattering measurement
    return scattering_meas


def lf_refocus_step(lf: np.ndarray, shift: int) -> np.ndarray:
    """Performs one step of the shift-and-add operation for light field refocusing.

    Args:
        lf (np.ndarray): 4D array of shape [row_mla, col_mla, H, W] representing the light field.
        shift (int): Number of pixels to shift before adding.

    Returns:
        np.ndarray: Result of the shift-and-add operation for this step, with shape [H, W].
    """
    c_lf = lf.shape[0] // 2 + 1  # Calculate the center index of the microlenses

    out = np.zeros((lf.shape[-2:]))  # Preallocate the output array
    for r in range(lf.shape[0]):
        for c in range(lf.shape[1]):
            # Calculate the shift for the current microlens position
            shift_tuple = (-1 * (c_lf - (c + 1)) * shift, -1 * (c_lf - (r + 1)) * shift)
            out += shift_array(lf[r, c, :, :], *shift_tuple)  # Apply the shift and accumulate

    return out  # Return the result of the shift-and-add operation


def lf_refocus_volume(
    lf: np.ndarray, z_slices: int, max_shift: int, mla_size: tuple = (3, 3)
) -> np.ndarray:
    """Refocuses a light field volume using multiple z slices and the shift-and-add algorithm.

    Args:
        lf (np.ndarray): Cropped light field of shape [num_views, H, W].
        z_slices (int): Number of z slices to generate in the refocused volume.
        max_shift (int): Maximum shift value, usually half of the number of z slices.
        mla_size (tuple, optional): Size of the microlens array, defaults to (3, 3).

    Returns:
        np.ndarray: Refocused volume of shape [z_slices, H, W].
    """
    lf = lf.reshape((*mla_size, *lf.shape[-2:]))  # Reshape the light field array to 4D

    rfv = np.zeros((z_slices, *lf.shape[-2:]))  # Preallocate the refocused volume array

    for ii, z in enumerate(range(z_slices)):
        rfv[z, :, :] = lf_refocus_step(lf=lf, shift=ii - max_shift + 1)  # Perform refocusing for each z slice

    return linear_normalize(rfv)  # Return the normalized refocused volume


def zero_slices_not_in_list(arr, indices_to_keep):
    """Zeros out slices in a 3D array along the first axis that are not specified in the list of indices to keep.

    Args:
        arr (numpy.ndarray): The 3D array with shape (c, h, w) where c is the number of slices.
        indices_to_keep (list): List of integers representing the indices along the c-axis to retain.

    Returns:
        numpy.ndarray: A modified array where slices not indexed in the provided list are set to zero.

    Example:
        >>> c, h, w = 3, 4, 5
        >>> array = np.random.rand(c, h, w)
        >>> indices_to_keep = [0, 2]
        >>> result_array = zero_slices_not_in_list(array, indices_to_keep)
        >>> print(result_array)
    """
    # Create a boolean mask where True corresponds to indices to keep
    mask = np.isin(np.arange(arr.shape[0]), indices_to_keep)

    # Use the mask to zero out slices not in the list
    arr[~mask, :, :] = 0

    return arr  # Return the modified array with zeroed slices