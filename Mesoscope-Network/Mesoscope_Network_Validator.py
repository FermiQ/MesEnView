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
The Python script defines a function `refocusing_and_crop` that takes light field data as input and applies a series of shifts to create a refocused volume. It then crops this volume to the original dimensions. The script also defines a joint model combining a Demixer and Reconstructor, loads pre-trained weights, and uses the model to make predictions on synthetic data points. Finally, it processes and saves the demixed views, refocused volume, and reconstructed volume as TIF files.

1. **Refocusing and Cropping Function** (`refocusing_and_crop`): This function takes light field data and applies a series of shifts to create a refocused volume. It then crops this volume to the original dimensions. The function is defined using a lambda function within the script.

2. **Model Setup**: The script defines constants for the model setup, including the total length of the light field, kernel size for convolution, and padding size for input data. It also instantiates Demixer and Reconstructor models with specified parameters.

3. **Joint Model**: The script creates a joint model combining the Demixer and Reconstructor models. This joint model takes input views and outputs demixed views, refocused volume, and reconstructed volume.

4. **Loading Pre-trained Weights**: The script loads pre-trained weights for the joint model from a specified file.

5. **Synthetic Data Points**: The script defines an array of coordinates for synthetic data points and sets a directory for saving results.

6. **Data Preparation and Prediction**: The script reads measurement data from a TIF file, normalizes it, and pads it to handle edge effects during processing. It then fills an input array with the appropriate sections of the padded measurement data and makes predictions using the joint model on this input data.

7. **Saving Results**: The script processes and saves the demixed views, refocused volume, and reconstructed volume as TIF files in the specified results directory.

Refocus and crop light field data.

The function takes light field data, applies a series of shifts to create a refocused volume, and crops it to the original dimensions.

Parameters:
- vars: A list containing the following elements:
    - lf_data: Tensor of shape [batch, height, width, M, N] representing light field data, 
      where M and N are the number of views in the horizontal and vertical directions.
    - padding: Integer indicating the padding size to be removed after processing.
    - rows: Integer representing the number of rows in the light field data.
    - cols: Integer representing the number of columns in the light field data.

Returns:
- rfv: Tensor of shape [batch, rows, cols, 162] representing the refocused volume.
"""
from __future__ import print_function
import os
from cm2_models import Demixer_ResNet, Reconstructor_ResNet
import numpy as np
import tifffile
import tensorflow as tf

# Set the policy for mixed precision computation to use float16 to optimize performance
tf.keras.mixed_precision.experimental.set_policy('float16')

from tensorflow.keras.layers import Lambda, Input, Concatenate
from tensorflow.keras.models import Model
from skimage import io

def refocusing_and_crop(vars):
    """
    Refocus and crop light field data.

    This function takes light field data, applies a series of shifts to create a refocused volume, 
    and crops it to the original dimensions.

    Parameters:
    - vars: A list containing the following elements:
        - lf_data: Tensor of shape [batch, height, width, M, N] representing light field data, 
          where M and N are the number of views in the horizontal and vertical directions.
        - padding: Integer indicating the padding size to be removed after processing.
        - rows: Integer representing the number of rows in the light field data.
        - cols: Integer representing the number of columns in the light field data.

    Returns:
    - rfv: Tensor of shape [batch, rows, cols, 162] representing the refocused volume.
    """
    lf_data = vars[0]  # Light field data
    padding = vars[1]  # Padding to be removed
    rows = vars[2]     # Number of rows in input data
    cols = vars[3]     # Number of columns in input data
    
    # Reshape the light field data to include padding
    lf_data = tf.reshape(lf_data, [-1, rows + 2 * padding, cols + 2 * padding, 3, 3])
    rfv_list = []  # Initialize list to hold refocused volume frames

    # Iterate through a range of shifts to apply in refocusing
    for shift in range(-17, 19, 1):
        tmp = lf_data[:, :, :, 1, 1]  # Central view
        # Apply various shifts and combines frames
        tmp = tmp + tf.roll(lf_data[:, :, :, 0, 0], shift=[shift, shift], axis=[1, 2])  # Top-left
        tmp = tmp + tf.roll(lf_data[:, :, :, 0, 1], shift=shift, axis=1)  # Top-center
        tmp = tmp + tf.roll(lf_data[:, :, :, 0, 2], shift=[shift, -shift], axis=[1, 2])  # Top-right
        tmp = tmp + tf.roll(lf_data[:, :, :, 1, 0], shift=shift, axis=2)  # Middle-left
        tmp = tmp + tf.roll(lf_data[:, :, :, 1, 2], shift=-shift, axis=2)  # Middle-right
        tmp = tmp + tf.roll(lf_data[:, :, :, 2, 0], shift=[-shift, shift], axis=[1, 2])  # Bottom-left
        tmp = tmp + tf.roll(lf_data[:, :, :, 2, 1], shift=-shift, axis=1)  # Bottom-center
        tmp = tmp + tf.roll(lf_data[:, :, :, 2, 2], shift=[-shift, -shift], axis=[1, 2])  # Bottom-right
        
        # Average the combined frames
        tmp = tmp / 9.0
        tmp = tf.expand_dims(tmp, axis=-1)  # Expand dimensions for concatenation
        rfv_list.append(tmp)  # Add to the list

    # Concatenate all refocused frames along the last axis
    rfv = tf.concat(rfv_list, axis=-1)
    
    # Return cropped refocused volume
    return rfv[:, padding:-padding, padding:-padding, :]

# Define constants for the model setup
tot_len = 1920  # Total length of the light field
kernel_size = 3  # Kernel size for convolution
padding = 32  # Padding size for input data
rows = tot_len - 2 * padding  # Number of rows after padding
cols = rows  # Number of columns, same as rows

# Instantiate Demixer and Reconstructor models with specified parameters
demixer = Demixer_ResNet(rows + 2 * padding, cols + 2 * padding, 3, 64, 16)  # Demixer model
reconstructor = Reconstructor_ResNet(rows, cols, 9, 'add', 36, 80, kernel_size, 64, 16, 16, None)  # Reconstructor model

# Create input layer for the model
input_views = Input(shape=[rows + 2 * padding, cols + 2 * padding, 9], name='input_views')

# Pass input through the demixer to get demixed views
demixed_views = demixer(input_views)

# Apply the refocusing and cropping function
rfv = Lambda(refocusing_and_crop)([demixed_views, padding, rows, cols])

# Crop the demixed views to remove padding
crop_dmx_views = Lambda(lambda x: x[:, padding:-padding, padding:-padding, :])(demixed_views)

# Concatenate cropped demixed views with the refocused volume
views_with_rfv = Concatenate(axis=-1)([crop_dmx_views, rfv])

# Pass the concatenated views through the reconstructor to get a prediction
pred_vol = reconstructor(views_with_rfv)

# Create the joint model to combine inputs and outputs
joint_model = Model(inputs=[input_views], outputs=[demixed_views, rfv, pred_vol])

# Load pre-trained weights for the model from a specified file
joint_model.load_weights('cm2net.hdf5')

# Set the location for experimental data or reference image (commented out)
# ref = io.imread('.tif')
# ref = ref.astype('float32') / 65535.0

# Define locations for synthetic data points as an array of coordinates
loc = np.array(
    [[392, 916], [393, 1562], [394, 2206], [1032, 918], [1032, 1561], [1032, 2204], [1681, 917], [1680, 1561],
     [1679, 2203]])

# Set the directory for saving results and create it if it does not exist
results_dir = 'result/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Read measurement data from a TIF file and normalize it
meas = io.imread('measurement.tif')
meas = meas.astype('float64') / 65535.0  # Normalize pixel values
# Optionally apply histogram matching (commented out)
# meas = match_histograms(meas, ref, multichannel=False)

tmp_pad = 900  # Padding for the measurement data
# Pad the measurement data to handle edge effects during processing
meas = np.pad(meas, ((tmp_pad, tmp_pad), (tmp_pad, tmp_pad)), 'constant', constant_values=0)

# Initialize an empty array to hold the input data for the model
x = np.zeros((1, tot_len, tot_len, 9), 'float16')
# Fill the input array with the appropriate sections of the padded measurement data
for k in range(9):
    x[0, :, :, k] = (meas[loc[k, 0] - (tot_len // 2) + tmp_pad:loc[k, 0] + (tot_len // 2) + tmp_pad,
                     loc[k, 1] - (tot_len // 2) + tmp_pad:loc[k, 1] + (tot_len // 2) + tmp_pad]).astype('float16')

# Make predictions using the joint model on the prepared input data
[pred_dmx, pred_rfv, pred_rec] = joint_model.predict(x, batch_size=1)

# Process and save the demixed views as a TIF file
pred_dmx = (np.transpose(pred_dmx[0, :].squeeze(), [2, 0, 1]) * 65535.0).astype('uint16')
pred_dmx = np.pad(pred_dmx, ((0, 0), (padding, padding), (padding, padding)), 'constant',
                  constant_values=0)
tifffile.imwrite((results_dir + '/dmx.tif'), pred_dmx)

# Process and save the refocused volume as a TIF file
pred_rfv = (np.transpose(pred_rfv[0, :].squeeze(), [2, 0, 1]) * 65535.0).astype('uint16')
pred_rfv = np.pad(pred_rfv, ((0, 0), (padding, padding), (padding , padding)), 'constant',
                  constant_values=0)
tifffile.imwrite((results_dir + '/rfv.tif'), pred_rfv)

# Process and save the reconstructed volume as a TIF file
pred_rec = pred_rec / pred_rec.max()  # Normalize the reconstructed volume
pred_rec = (np.transpose(pred_rec[0, :].squeeze(), [2, 0, 1]) * 65535.0).astype('uint16')
pred_rec = np.pad(pred_rec, ((0, 0), (padding, padding), (padding, padding)), 'constant',
                  constant_values=0)
tifffile.imwrite(results_dir + '/rec.tif', pred_rec)