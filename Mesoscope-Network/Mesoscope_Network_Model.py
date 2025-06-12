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
The code defines two main functions for constructing neural network models: `Demixer_ResNet` and `Reconstructor_ResNet`. These functions are designed to create specific architectures for image processing tasks, utilizing residual blocks and various activation functions.

1. **Demixer_ResNet**:
   - This function constructs a Demixer Residual Network model, which is designed to process view and random field inputs separately before merging them.
   - It takes several parameters to customize the model architecture, including input dimensions, number of views, depth of the random field, output channels, filter size, number of filters, and the number of residual blocks for both view and random field inputs.
   - The function returns a Keras Model instance representing the Demixer ResNet architecture.

2. **Reconstructor_ResNet**:
   - This function builds a Reconstructor Residual Network model, which is used to reconstruct images from the outputs of the Demixer model.
   - It accepts similar parameters as `Demixer_ResNet`, with the addition of a merge mode parameter to specify how the outputs from the view and random field branches should be combined.
   - The function also returns a Keras Model instance, this time representing the Reconstructor ResNet architecture.

3. **res_block_gen**:
   - This function generates a residual block for a convolutional neural network.
   - It takes a model as input and applies a series of operations, including convolution, batch normalization, and activation, before adding a skip connection.
   - The function returns the output tensor after applying the residual block operations.

It provides functions to construct and customize Demixer and Reconstructor ResNet models for image processing tasks, utilizing residual blocks and various activation functions. These models can be tailored to specific requirements by adjusting the input parameters.

Generates a residual block for a convolutional neural network.

Parameters:
- model: The input tensor that will be processed through this residual block.
- kernel_size: The size of the convolution kernel (e.g., 3 for a 3x3 kernel).
- filters: The number of filters to use in the convolution layers.
- strides: The stride with which to slide the convolution kernel.
- activation_func: The activation function to use ('regular' for PReLU, 'swish' for swish activation).
- kernel_reg: Optional kernel regularizer to apply to the convolution layers.

Returns:
- The output tensor after applying the residual block operations.
"""
from __future__ import print_function

# Import necessary Keras layers and functions for building the model
from tensorflow.keras.layers import Conv2D, Input, Concatenate, Activation, Add, BatchNormalization, PReLU
from tensorflow.keras.activations import swish
from tensorflow.keras import Model

def res_block_gen(model, kernel_size, filters, strides, activation_func='regular', kernel_reg=None):
    """
    Generates a residual block for a convolutional neural network.

    Parameters:
    - model: The input tensor that will be processed through this residual block.
    - kernel_size: The size of the convolution kernel (e.g., 3 for a 3x3 kernel).
    - filters: The number of filters to use in the convolution layers.
    - strides: The stride with which to slide the convolution kernel.
    - activation_func: The activation function to use ('regular' for PReLU, 'swish' for swish activation).
    - kernel_reg: Optional kernel regularizer to apply to the convolution layers.

    Returns:
    - The output tensor after applying the residual block operations.
    """
    gen = model  # Store the input tensor for skip connection

    # First convolution layer with Batch Normalization
    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
                   kernel_regularizer=kernel_reg)(model)
    model = BatchNormalization(momentum=0.5)(model)  # Batch normalization with momentum

    # Choose activation function based on input parameter
    if activation_func == 'regular':
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(
            model)  # Parametric ReLU activation
    elif activation_func == 'swish':
        model = swish(model)  # Swish activation

    # Second convolution layer with Batch Normalization
    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
                   kernel_regularizer=kernel_reg)(model)
    model = BatchNormalization(momentum=0.5)(model)  # Batch normalization

    model = Add()([gen, model])  # Add skip connection (residual connection)

    return model  # Return the output tensor of the residual block

def Demixer_ResNet(input_rows, input_cols, filter_size=3, num_filters=64, num_resblocks=16):
    """
    Constructs a Demixer Residual Network model.

    Parameters:
    - input_rows: The number of rows in the input image.
    - input_cols: The number of columns in the input image.
    - filter_size: The size of the convolutional kernels.
    - num_filters: The number of filters used in the convolutional layers.
    - num_resblocks: The number of residual blocks to include in the model.

    Returns:
    - A Keras Model instance representing the Demixer ResNet architecture.
    """
    num_views = 9  # Number of views (channels) in the input image
    gen_input = Input((input_rows, input_cols, num_views))  # Input layer

    # Initial convolution layer
    model = Conv2D(filters=64, kernel_size=filter_size, strides=1, padding="same")(gen_input)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)

    gen_model = model  # Keep reference to the output of the initial layer

    # Add specified number of residual blocks
    for index in range(num_resblocks):
        model = res_block_gen(model, filter_size, num_filters, 1, activation_func='regular')

    # Final convolution layer and skip connection
    model = Conv2D(filters=64, kernel_size=filter_size, strides=1, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = Add()([gen_model, model])  # Add skip connection to initial model output

    model = Conv2D(filters=num_views, kernel_size=filter_size, strides=1, padding="same")(model)  # Final convolution
    model = Activation('sigmoid')(model)  # Sigmoid activation for output

    generator_model = Model(inputs=gen_input, outputs=model)  # Create and return the Keras model

    return generator_model  # Return the constructed generator model

def Reconstructor_ResNet(input_rows, input_cols, num_views, merge_mode='add', rf_depth=50, output_z=32, filter_size=3,
                         num_filters=64, num_resblocks_vs=16, num_resblocks_rf=16, kernel_reg=None):
    """
    Constructs a Reconstructor Residual Network model.

    Parameters:
    - input_rows: The number of rows in the input image.
    - input_cols: The number of columns in the input image.
    - num_views: The number of views (channels) in the input image.
    - merge_mode: Method of combining outputs ('add' for addition, 'concat' for concatenation).
    - rf_depth: The depth of the random field (used in input shape).
    - output_z: The number of channels for the output layer.
    - filter_size: The size of the convolutional kernels.
    - num_filters: The number of filters used in the convolutional layers.
    - num_resblocks_vs: The number of residual blocks for the view input.
    - num_resblocks_rf: The number of residual blocks for the random field input.
    - kernel_reg: Optional kernel regularizer to apply to the convolution layers.

    Returns:
    - A Keras Model instance representing the Reconstructor ResNet architecture.
    """
    input_z = num_views + rf_depth  # Total input channels
    inputs = Input((input_rows, input_cols, input_z))  # Input layer
    print("inputs shape:", inputs.shape)  # Print the input shape for debugging

    # Split input into views and random field
    input_views = inputs[:, :, :, 0:num_views]  # Extract view inputs
    input_rfv = inputs[:, :, :, num_views:]  # Extract random field inputs

    # Process view inputs
    model = Conv2D(filters=num_filters, kernel_size=filter_size, strides=1, padding="same",
                   kernel_regularizer=kernel_reg)(input_views)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)

    gen_model = model  # Keep reference to the output of the initial layer for skip connection

    # Add specified number of residual blocks for view input
    for index in range(num_resblocks_vs):
        model = res_block_gen(model, filter_size, num_filters, 1, activation_func='regular', kernel_reg=kernel_reg)

    # Final convolution layer and skip connection for view input
    model = Conv2D(filters=num_filters, kernel_size=filter_size, strides=1, padding="same",
                   kernel_regularizer=kernel_reg)(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = Add()([gen_model, model])  # Add skip connection

    model = Conv2D(filters=output_z, kernel_size=filter_size, strides=1, padding="same", kernel_regularizer=kernel_reg)(model)  # Final convolution for view input
    model = Activation('sigmoid')(model)  # Sigmoid activation for output

    # Process random field inputs
    model_rf = Conv2D(filters=num_filters, kernel_size=filter_size, strides=1, padding="same",
                      kernel_regularizer=kernel_reg)(input_rfv)
    model_rf = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model_rf)

    gen_model_rf = model_rf  # Keep reference to the output of the initial layer for skip connection

    # Add specified number of residual blocks for random field input
    for index in range(num_resblocks_rf):
        model_rf = res_block_gen(model_rf, filter_size, num_filters, 1, activation_func='regular', kernel_reg=kernel_reg)

    # Final convolution layer and skip connection for random field input
    model_rf = Conv2D(filters=num_filters, kernel_size=filter_size, strides=1, padding="same",
                      kernel_regularizer=kernel_reg)(model_rf)
    model_rf = BatchNormalization(momentum=0.5)(model_rf)
    model_rf = Add()([gen_model_rf, model_rf])  # Add skip connection

    model_rf = Conv2D(filters=output_z, kernel_size=filter_size, strides=1, padding="same",
                      kernel_regularizer=kernel_reg)(model_rf)  # Final convolution for random field input
    model_rf = Activation('sigmoid')(model_rf)  # Sigmoid activation for output

    # Merge the outputs from both branches using the specified merge mode
    if merge_mode == 'add':
        output_volume = Add()([model, model_rf])  # Add the outputs
    elif merge_mode == 'concat':
        output_volume = Concatenate(axis=-1)([model, model_rf])  # Concatenate the outputs along the last axis

    output_volume = Conv2D(filters=output_z, kernel_size=filter_size, strides=1, padding="same",
                           kernel_regularizer=kernel_reg)(output_volume)  # Final convolution for merged output
    output_volume = Activation('sigmoid')(output_volume)  # Sigmoid activation for final output

    reconstructor = Model(inputs=inputs, outputs=output_volume)  # Create and return the Keras model
    return reconstructor  # Return the constructed reconstructor model