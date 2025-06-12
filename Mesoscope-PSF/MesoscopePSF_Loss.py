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
import tensorflow as tf
import numpy as np

def mse_real(model, data, variables=None, mu=None, w=None):
    """
    Calculate the mean squared error (MSE) loss function for a real-valued model.

    Parameters:
    model (Tensor): The predicted values from the model.
    data (Tensor): The true values to compare against.
    variables (list): A list of variables used in the loss computation, expected to contain specific values based on the model.
    mu (float): A scalar used for scaling certain terms in the loss function.
    w (list): A list of weights corresponding to different components of the loss.

    Returns:
    Tensor: The computed loss value.
    """
    # Compute the difference between the model and data
    mydiff = model - data
    mydiff = mydiff[:, 1:-1]
    data = data[:, 1:-1]
    model = model[:, 1:-1]

    # Normalize mean squared error
    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)
    mse_norm2 = tf.reduce_mean(tf.reduce_sum(tf.square(mydiff), axis=(-3, -2, -1)) / tf.math.reduce_max(tf.square(data), axis=(-3, -2, -1))) / data.shape[-3] * 200

    # Compute log-likelihood
    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    # Extract variables
    f = variables[3]
    gxymean = tf.reduce_mean(tf.abs(variables[4]))
    bg = variables[1]
    intensity = variables[2]
    
    # Additional computations for the loss
    s = tf.math.reduce_sum(tf.math.square(f[0] - f[1]) + tf.math.square(f[-1] - f[-2]))
    dfz = tf.math.square(tf.experimental.numpy.diff(f, n=1, axis=-3))
    dfz = tf.reduce_sum(dfz)
    
    Imin = tf.reduce_sum(tf.math.square(tf.math.minimum(f, 0)))
    bgmin = tf.reduce_sum(tf.math.square(tf.math.minimum(bg, 0)))
    intensitymin = tf.reduce_sum(tf.math.square(tf.math.minimum(intensity, 0)))
    fsz = f.shape
    Inorm = tf.reduce_mean(tf.math.square(tf.math.reduce_sum(f, axis=(-1, -2)) - tf.math.reduce_sum(f) / fsz[0]))

    # Compute the total loss
    loss = mse_norm1 * w[0] + mse_norm2 * w[1] + w[2] * dfz + s * w[3] + w[4] * Imin * mu + bgmin * w[5] * mu + intensitymin * w[6] * mu + Inorm * w[7] + gxymean * w[8]

    return loss

def mse_real_4pi(model, data, variables=None, mu=None, w=None):
    """
    Calculate the mean squared error (MSE) loss function for a 4-pi model.

    Parameters:
    model (Tensor): The predicted values from the 4-pi model.
    data (Tensor): The true values to compare against.
    variables (list): A list of variables used in the loss computation, expected to contain specific values based on the model.
    mu (float): A scalar used for scaling certain terms in the loss function.
    w (list): A list of weights corresponding to different components of the loss.

    Returns:
    Tensor: The computed loss value.
    """
    # Compute the difference between the model and data
    mydiff = model - data
    mydiff = mydiff[:, :, 1:-1]
    data = data[:, :, 1:-1]

    # Normalize mean squared error
    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)
    mse_norm2 = tf.reduce_mean(tf.reduce_sum(tf.square(mydiff), axis=(-3, -2, -1)) / tf.math.reduce_max(tf.square(data), axis=(-3, -2, -1))) / data.shape[-3] * 200

    # Extract variables
    f = variables[4]
    bg = variables[1]
    intensity = variables[2]
    gxymean = tf.reduce_mean(tf.abs(variables[-1]))

    # Additional computations for the loss
    s = tf.math.reduce_sum(tf.math.square(f[0] - f[1]) + tf.math.square(f[-1] - f[-2]))
    fsz = f.shape
    Areal = variables[5]
    Aimg = variables[6]
    A = tf.complex(Areal, Aimg)

    dfz = tf.math.square(tf.experimental.numpy.diff(f, n=1, axis=-3)) + tf.math.square(tf.experimental.numpy.diff(Areal, n=1, axis=-3)) + tf.math.square(tf.experimental.numpy.diff(Aimg, n=1, axis=-3))
    dfz = tf.reduce_sum(dfz)

    Imin = tf.reduce_sum(tf.math.square(tf.math.minimum(f - 2 * tf.math.abs(A), 0)))
    bgmin = tf.reduce_sum(tf.math.square(tf.math.minimum(bg, 0)))
    intensitymin = tf.reduce_sum(tf.math.square(tf.math.minimum(intensity, 0)))
    Inorm = tf.reduce_mean(tf.math.square(tf.math.reduce_sum(f, axis=(-1, -2)) - tf.math.reduce_sum(f) / fsz[0]))

    # Compute the total loss
    loss = mse_norm1 * w[0] + mse_norm2 * w[1] + w[2] * dfz + (s + w[3]) + w[4] * Imin * mu + bgmin * mu * w[5] + intensitymin * w[6] * mu + Inorm * w[7] * mu + gxymean * w[8]

    return loss

def mse_real_4pi_All(model, data, loss_func, variables=None, mu=None, w=None, psfnorm=None):
    """
    Calculate the total mean squared error (MSE) loss for all samples in a 4-pi model.

    Parameters:
    model (Tensor): The predicted values from the 4-pi model for all samples.
    data (Tensor): The true values to compare against for all samples.
    loss_func (function): The loss function to be used for individual loss computation.
    variables (list): A list of variables used in the loss computation for each sample.
    mu (float): A scalar used for scaling certain terms in the loss function.
    w (list): A list of weights corresponding to different components of the loss.
    psfnorm (Tensor, optional): Optional normalization factor for the loss.

    Returns:
    Tensor: The total computed loss value across all samples.
    """
    varsize = len(variables)
    var = [None] * (varsize - 1)
    loss = 0.0
    
    # Iterate over each sample in the model
    for i in range(0, model.shape[0]):
        for j in range(1, varsize - 1):
            var[j] = variables[j][i]
        var[0] = variables[0]  # The first variable remains constant
        
        # Compute loss for the current sample
        if psfnorm:
            loss += loss_func(model[i], data[i], var, mu, w, psfnorm[i])
        else:
            loss += loss_func(model[i], data[i], var, mu, w)

    return loss

def mse_real_All(model, data, loss_func, variables=None, mu=None, w=None, psfnorm=None):
    """
    Calculate the total mean squared error (MSE) loss for all samples in a real-valued model.

    Parameters:
    model (Tensor): The predicted values from the model for all samples.
    data (Tensor): The true values to compare against for all samples.
    loss_func (function): The loss function to be used for individual loss computation.
    variables (list): A list of variables used in the loss computation for each sample.
    mu (float): A scalar used for scaling certain terms in the loss function.
    w (list): A list of weights corresponding to different components of the loss.
    psfnorm (Tensor, optional): Optional normalization factor for the loss.

    Returns:
    Tensor: The total computed loss value across all samples.
    """
    varsize = len(variables)
    var = [None] * (varsize - 1)
    loss = 0.0
    
    # Iterate over each sample in the model
    for i in range(0, model.shape[0]):
        for j in range(1, varsize - 1):
            var[j] = variables[j][i]
        var[0] = variables[0]  # The first variable remains constant
        
        # Compute loss for the current sample
        if psfnorm:
            loss += loss_func(model[i], data[i], var, mu, w, psfnorm[i])
        else:
            loss += loss_func(model[i], data[i], var, mu, w)

    return loss

def mse_real_pupil(model, data, variables=None, mu=None, w=None, psfnorm=1.0):
    """
    Calculate the mean squared error (MSE) loss function for a pupil model.

    Parameters:
    model (Tensor): The predicted values from the pupil model.
    data (Tensor): The true values to compare against.
    variables (list): A list of variables used in the loss computation.
    mu (float): A scalar used for scaling certain terms in the loss function.
    w (list): A list of weights corresponding to different components of the loss.
    psfnorm (float): A normalization parameter for the pupil function.

    Returns:
    Tensor: The computed loss value.
    """
    # Compute the difference between the model and data
    mydiff = model - data

    # Normalize mean squared error
    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)
    mse_norm2 = tf.reduce_mean(tf.reduce_sum(tf.square(mydiff), axis=(-3, -2, -1)) / tf.math.reduce_max(tf.square(data), axis=(-3, -2, -1))) / data.shape[-3] * 200

    # Compute log-likelihood
    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    # Extract variables
    pupilR = variables[3]    
    pupilI = variables[4] 
    bg = variables[1]
    intensity = variables[2]
    gxymean = tf.reduce_mean(tf.abs(variables[-1]))   
    Inorm = tf.math.square(tf.math.minimum(psfnorm - 0.97, 0))

    # Additional computations for the loss
    dfxy1 = tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilI, n=1, axis=-1))) + tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilI, n=1, axis=-2)))
    dfxy2 = tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilR, n=1, axis=-1))) + tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilR, n=1, axis=-2)))
    dfxy = dfxy2

    # Compute the total loss
    loss = LL * w[0] + bgmin * w[5] * mu + intensitymin * w[6] * mu + dfxy * w[2] + gxymean * w[8] + Inorm * w[7]

    return loss

def mse_pupil_4pi(model, data, variables=None, mu=None, w=None, psfnorm=[1.0, 1.0]):
    """
    Calculate the mean squared error (MSE) loss function for a 4-pi pupil model.

    Parameters:
    model (Tensor): The predicted values from the 4-pi pupil model.
    data (Tensor): The true values to compare against.
    variables (list): A list of variables used in the loss computation.
    mu (float): A scalar used for scaling certain terms in the loss function.
    w (list): A list of weights corresponding to different components of the loss.
    psfnorm (list): A list of normalization parameters for the pupil function.

    Returns:
    Tensor: The computed loss value.
    """
    # Compute the difference between the model and data
    mydiff = model - data
    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)     
    mse_norm2 = tf.reduce_mean(tf.reduce_sum(tf.square(mydiff), axis=(-3, -2, -1)) / tf.math.reduce_max(tf.square(data), axis=(-3, -2, -1))) / data.shape[-3] * 200

    # Compute log-likelihood
    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)]) 

    # Extract variables
    pupilR1 = variables[4]    
    pupilI1 = variables[5] 
    pupilR2 = variables[6]    
    pupilI2 = variables[7] 
    bg = variables[1]
    intensity = variables[2]
    alpha = variables[9]
    wavelength = variables[10]
    gxymean = tf.reduce_mean(tf.abs(variables[-1]))   

    # Additional computations for the loss
    Inorm = tf.math.square(tf.math.minimum(psfnorm[0] - 0.97, 0)) + tf.math.square(tf.math.minimum(psfnorm[1] - 0.97, 0))

    dfxy1 = tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilI1, n=1, axis=-1))) + tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilI1, n=1, axis=-2)))
    dfxy2 = tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilR1, n=1, axis=-1))) + tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilR1, n=1, axis=-2)))
    dfxy3 = tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilI2, n=1, axis=-1))) + tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilI2, n=1, axis=-2)))
    dfxy4 = tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilR2, n=1, axis=-1))) + tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilR2, n=1, axis=-2)))
    dfxy = dfxy2 + dfxy4

    # Compute the total loss
    bgmin = tf.reduce_sum(tf.math.square(tf.math.minimum(bg, 0)))
    intensitymin = tf.reduce_sum(tf.math.square(tf.math.minimum(intensity, 0)))
    alphamin = tf.reduce_sum(tf.math.square(tf.math.minimum(alpha, 0)))

    loss = LL * w[0] + bgmin * w[5] * mu + intensitymin * w[6] * mu + dfxy * w[2] + alphamin * w[4] * mu + gxymean * w[8] + Inorm * w[7]

    return loss

def mse_real_zernike(model, data, variables=None, mu=None, w=None):
    """
    Calculate the mean squared error (MSE) loss function for a Zernike model.

    Parameters:
    model (Tensor): The predicted values from the Zernike model.
    data (Tensor): The true values to compare against.
    variables (list): A list of variables used in the loss computation.
    mu (float): A scalar used for scaling certain terms in the loss function.
    w (list): A list of weights corresponding to different components of the loss.

    Returns:
    Tensor: The computed loss value.
    """
    # Compute the difference between the model and data
    mydiff = model - data

    # Normalize mean squared error
    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)     
    mse_norm2 = tf.reduce_mean(tf.reduce_sum(tf.square(mydiff), axis=(-3, -2, -1)) / tf.math.reduce_max(tf.square(data), axis=(-3, -2, -1))) / data.shape[-3] * 200

    # Compute log-likelihood
    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    # Extract variables
    bg = variables[1]
    intensity = variables[2]
    zcoeff = variables[3]
    wavelength = variables[5]
    gxymean = tf.reduce_mean(tf.abs(variables[-1]))   
   
    # Additional computations for the loss
    bgmin = tf.reduce_sum(tf.math.square(tf.math.minimum(bg, 0)))
    intensitymin = tf.reduce_sum(tf.math.square(tf.math.minimum(intensity, 0)))
    g1 = tf.reduce_sum(tf.square(zcoeff[0][1:]))
    g2 = tf.reduce_sum(tf.square(zcoeff[1]))

    # Compute the total loss
    loss = LL * w[0] + bgmin * w[5] * mu + intensitymin * w[6] * mu + gxymean * w[8]

    return loss

def mse_zernike_4pi(model, data, variables=None, mu=None, w=None):
    """
    Computes the loss for a Zernike model using the mean squared error (MSE) approach under 4pi illumination.
    
    Parameters:
    model (Tensor): The predicted model output.
    data (Tensor): The actual data to compare against.
    variables (list): A list of variables used in the loss calculation. 
                      Specific indices are expected to contain background, intensity, 
                      intensity phase, Zernike coefficients, alpha, wavelength, 
                      position, and a last variable for GXY mean.
    mu (float): A scaling factor used in the computation of the loss.
    w (list): A list of weights corresponding to different components of the loss.
    
    Returns:
    Tensor: The computed loss value.
    """
    mydiff = model - data  # Calculate the difference between model and data

    # Calculate normalized MSE components
    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)     
    mse_norm2 = tf.reduce_mean(tf.reduce_sum(tf.square(mydiff), axis=(-3, -2, -1)) / 
                                tf.math.reduce_max(tf.square(data), axis=(-3, -2, -1))) / data.shape[-3] * 200

    # Calculate log-likelihood loss
    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])  # Mean of finite values only

    # Extract parameters from the variables list
    bg = variables[1]
    intensity = variables[2]
    intensity_phase = variables[3]
    zcoeff1 = variables[4]
    zcoeff2 = variables[5]
    alpha = variables[7]
    wavelength = variables[8]
    posd = variables[9]
    gxymean = tf.reduce_mean(tf.abs(variables[-1]))    # Mean of absolute values of the last variable

    # Penalties for negative values in parameters
    bgmin = tf.reduce_sum(tf.math.square(tf.math.minimum(bg, 0)))
    intensitymin = tf.reduce_sum(tf.math.square(tf.math.minimum(intensity, 0)))
    alphamin = tf.reduce_sum(tf.math.square(tf.math.minimum(alpha, 0)))

    # Regularization terms based on Zernike coefficients and position
    g1 = tf.reduce_sum(tf.abs(zcoeff1[1][1:]))  # Regularization for second order Zernike coefficients
    g2 = tf.reduce_sum(tf.abs(zcoeff1[0][1:])) * 2 + tf.reduce_sum(tf.abs(zcoeff2[0][1:])) * 2  # First order
    g3 = tf.reduce_sum(tf.abs(zcoeff2[1][1:]))  # Second order
    g4 = tf.reduce_sum(tf.square(posd)) * 2  # Position regularization

    # Compute final loss combining various components
    loss = LL * w[0] + bgmin * w[5] * mu + intensitymin * w[6] * mu + alphamin * w[4] * mu + gxymean * w[8]

    return loss

def mse_zernike_4pi_smlm(model, data, variables=None, mu=None, w=None):
    """
    Computes the loss for a Zernike model using the mean squared error (MSE) approach for single molecule localization microscopy (SMLM) under 4pi illumination.
    
    Parameters:
    model (Tensor): The predicted model output.
    data (Tensor): The actual data to compare against.
    variables (list): A list of variables used in the loss calculation.
                      Specific indices are expected to contain background, intensity, 
                      Zernike coefficients, alpha, z-position, stage position, and sample height.
    mu (float): A scaling factor used in the computation of the loss.
    w (list): A list of weights corresponding to different components of the loss.
    
    Returns:
    Tensor: The computed loss value.
    """
    mydiff = model - data  # Calculate the difference between model and data

    # Calculate normalized MSE components
    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)     
    mse_norm2 = tf.reduce_mean(tf.reduce_sum(tf.square(mydiff), axis=(-2, -1)) / 
                                tf.math.reduce_max(tf.square(data), axis=(-2, -1))) * 200

    # Calculate log-likelihood loss
    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])  # Mean of finite values only

    # Extract parameters from the variables list
    bg = variables[1]
    intensity = variables[2]
    intensity_phase = variables[3]
    zcoeff1 = variables[6]
    zcoeff2 = variables[7]
    alpha = variables[8]
    zpos = variables[0][:, 0, ...]  # z-position from the first variable
    stagepos = variables[4]  # stage position
    sampleheight = variables[5]  # sample height

    # Penalties for negative values in parameters
    bgmin = tf.reduce_sum(tf.math.square(tf.math.minimum(bg, 0)))
    intensitymin = tf.reduce_sum(tf.math.square(tf.math.minimum(intensity, 0)))
    alphamin = tf.reduce_sum(tf.math.square(tf.math.minimum(alpha, 0)))
    zmin = tf.reduce_mean(tf.math.square(tf.math.minimum(zpos, 0))) + \
           tf.reduce_mean(tf.math.square(tf.math.minimum(stagepos, 0))) + \
           tf.reduce_mean(tf.math.square(tf.math.minimum(sampleheight, 0)))

    # Regularization terms based on Zernike coefficients
    g1 = tf.reduce_sum(tf.abs(zcoeff1[1][1:]))  # Regularization for the second order Zernike coefficients
    g2 = tf.reduce_sum(tf.abs(zcoeff1[0][1:])) * 2 + tf.reduce_sum(tf.abs(zcoeff2[0][1:])) * 2  # First order
    g3 = tf.reduce_sum(tf.abs(zcoeff2[1][1:]))  # Second order

    # Compute final loss combining various components
    loss = LL * w[0] + bgmin * w[5] * mu + intensitymin * w[6] * mu + alphamin * w[4] * mu + zmin * w[4] * mu

    return loss

def mse_real_zernike_FD(model, data, variables=None, mu=None, w=None):
    """
    Computes the loss for a real Zernike model using the mean squared error (MSE) approach with finite differences.
    
    Parameters:
    model (Tensor): The predicted model output.
    data (Tensor): The actual data to compare against.
    variables (list): A list of variables used in the loss calculation. 
                      Specific indices are expected to contain background, intensity, 
                      Zernike map, and a last variable for GXY mean.
    mu (float): A scaling factor used in the computation of the loss.
    w (list): A list of weights corresponding to different components of the loss.
    
    Returns:
    Tensor: The computed loss value.
    """
    mydiff = model - data  # Calculate the difference between model and data

    # Calculate normalized MSE components
    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)     
    mse_norm2 = tf.reduce_mean(tf.reduce_sum(tf.square(mydiff), axis=(-3, -2, -1)) / 
                                tf.math.reduce_max(tf.square(data), axis=(-3, -2, -1))) / data.shape[-3] * 200

    # Calculate log-likelihood loss
    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])  # Mean of finite values only

    # Extract parameters from the variables list
    bg = variables[1]
    intensity = variables[2]
    gxymean = tf.reduce_mean(tf.abs(variables[-1]))    # Mean of absolute values of the last variable

    # Penalties for negative values in parameters
    bgmin = tf.reduce_sum(tf.math.square(tf.math.minimum(bg, 0)))
    intensitymin = tf.reduce_sum(tf.math.square(tf.math.minimum(intensity, 0)))

    Zmap = variables[3]  # Zernike map
    # Compute finite differences in the Zernike map
    dfxy = tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(Zmap, n=1, axis=-1))) + \
           tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(Zmap, n=1, axis=-2)))

    # Compute final loss combining various components
    loss = LL * w[0] + bgmin * w[5] * mu + intensitymin * w[6] * mu + dfxy * w[2] + gxymean * w[8]

    return loss

def mse_real_zernike_IMM(model, data, variables=None, mu=None, w=None):
    """
    Computes the loss for a real Zernike model using the mean squared error (MSE) approach with intermediate memory management.
    
    Parameters:
    model (Tensor): The predicted model output.
    data (Tensor): The actual data to compare against.
    variables (list): A list of variables used in the loss calculation. 
                      Specific indices are expected to contain background, intensity, 
                      Zernike map, and a last variable for GXY mean.
    mu (float): A scaling factor used in the computation of the loss.
    w (list): A list of weights corresponding to different components of the loss.
    
    Returns:
    Tensor: The computed loss value.
    """
    mydiff = model - data  # Calculate the difference between model and data

    # Calculate normalized MSE components
    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)     
    mse_norm2 = tf.reduce_mean(tf.reduce_sum(tf.square(mydiff), axis=(-3, -2, -1)) / 
                                tf.math.reduce_max(tf.square(data), axis=(-3, -2, -1))) / data.shape[-3] * 200

    # Calculate log-likelihood loss
    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])  # Mean of finite values only

    # Extract parameters from the variables list
    bg = variables[1]
    intensity = variables[2]
    gxymean = tf.reduce_mean(tf.abs(variables[-1]))    # Mean of absolute values of the last variable

    # Penalties for negative values in parameters
    bgmin = tf.reduce_sum(tf.math.square(tf.math.minimum(bg, 0)))
    intensitymin = tf.reduce_sum(tf.math.square(tf.math.minimum(intensity, 0)))
    zpos = variables[0][:, 1, ...]  # z-position from the first variable
    zmin = tf.reduce_mean(tf.math.square(tf.math.minimum(zpos, 0)))  # Penalty for negative z-positions

    Zmap = variables[3]  # Zernike map
    # Compute finite differences in the Zernike map
    dfz = tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(Zmap, n=1, axis=-1)))

    # Compute final loss combining various components
    loss = LL * w[0] + bgmin * w[5] * mu + intensitymin * w[6] * mu + dfz * w[2] + gxymean * w[8] + zmin * w[4] * mu

    return loss

def mse_real_zernike_FD_smlm(model, data, variables=None, mu=None, w=None):
    """
    Computes the loss for a real Zernike model using the mean squared error (MSE) approach with finite differences for single molecule localization microscopy (SMLM).
    
    Parameters:
    model (Tensor): The predicted model output.
    data (Tensor): The actual data to compare against.
    variables (list): A list of variables used in the loss calculation. 
                      Specific indices are expected to contain background, intensity, 
                      stage position, Zernike map, and others.
    mu (float): A scaling factor used in the computation of the loss.
    w (list): A list of weights corresponding to different components of the loss.
    
    Returns:
    Tensor: The computed loss value.
    """
    mydiff = model - data  # Calculate the difference between model and data

    # Calculate normalized MSE components
    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)     
    mse_norm2 = tf.reduce_mean(tf.reduce_mean(tf.square(mydiff), axis=(-2, -1)) / 
                                tf.math.reduce_max(tf.square(data), axis=(-2, -1))) * 200

    # Calculate log-likelihood loss
    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])  # Mean of finite values only

    # Extract parameters from the variables list
    bg = variables[1]
    intensity = variables[2]
    stagepos = variables[5]  # stage position
    zpos = variables[0][:, 0, ...]  # z-position from the first variable
    bgmin = tf.reduce_mean(tf.math.square(tf.math.minimum(bg, 0)))  # Penalty for negative background
    zmin = tf.reduce_mean(tf.math.square(tf.math.minimum(zpos, 0))) + \
           tf.reduce_mean(tf.math.square(tf.math.minimum(stagepos, 0)))  # Penalty for negative z positions
    intensitymin = tf.reduce_mean(tf.math.square(tf.math.minimum(intensity, 0)))  # Penalty for negative intensity

    Zmap = variables[3]  # Zernike map
    # Compute finite differences in the Zernike map
    dfxy = tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(Zmap, n=1, axis=-1))) + \
           tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(Zmap, n=1, axis=-2)))

    # Compute final loss combining various components
    loss = LL * w[0] + bgmin * w[5] * mu + intensitymin * w[6] * mu + dfxy * w[2] + zmin * w[4] * mu

    return loss

def mse_real_zernike_smlm(model, data, variables=None, mu=None, w=None):
    """
    Computes the loss for a real Zernike model using the mean squared error (MSE) approach for single molecule localization microscopy (SMLM).
    
    Parameters:
    model (Tensor): The predicted model output.
    data (Tensor): The actual data to compare against.
    variables (list): A list of variables used in the loss calculation. 
                      Specific indices are expected to contain background, intensity, 
                      Zernike coefficients, stage position, and z-position.
    mu (float): A scaling factor used in the computation of the loss.
    w (list): A list of weights corresponding to different components of the loss.
    
    Returns:
    Tensor: The computed loss value.
    """
    mydiff = model - data  # Calculate the difference between model and data

    # Calculate normalized MSE components
    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)     
    mse_norm2 = tf.reduce_mean(tf.reduce_mean(tf.square(mydiff), axis=(-2, -1)) / 
                                tf.math.reduce_max(tf.square(data), axis=(-2, -1))) * 200

    # Calculate log-likelihood loss
    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])  # Mean of finite values only

    # Extract parameters from the variables list
    bg = variables[1]
    intensity = variables[2]
    zcoeff = variables[3]  # Zernike coefficients
    stagepos = variables[5]  # stage position
    zpos = variables[0][:, 0, ...]  # z-position from the first variable
    bgmin = tf.reduce_mean(tf.math.square(tf.math.minimum(bg, 0)))  # Penalty for negative background
    zmin = tf.reduce_mean(tf.math.square(tf.math.minimum(zpos, 0))) + \
           tf.reduce_mean(tf.math.square(tf.math.minimum(stagepos, 0)))  # Penalty for negative z positions
    intensitymin = tf.reduce_mean(tf.math.square(tf.math.minimum(intensity, 0)))  # Penalty for negative intensity

    # Regularization terms based on Zernike coefficients
    g1 = tf.reduce_sum(tf.square(zcoeff[0][1:]))  # Regularization for Zernike coefficients
    g2 = tf.reduce_sum(tf.square(zcoeff[1]))  # Regularization for Zernike coefficients

    # Compute final loss combining various components
    loss = LL * w[0] + bgmin * w[5] * mu + intensitymin * w[6] * mu + (g1 + g2) * w[2] + zmin * w[4] * mu

    return loss

def mse_real_pupil_smlm(model, data, variables=None, mu=None, w=None, psfnorm=1.0):
    """
    Computes a custom loss function based on the mean squared error (MSE) and a log-likelihood term.
    
    Parameters:
    - model: A tensor representing the predicted values from the model.
    - data: A tensor representing the ground truth data values.
    - variables: A list or tensor containing specific variables used in the loss computation.
      - Expected indices:
        - [0]: z-position values.
        - [1]: background values.
        - [2]: intensity values.
        - [3]: pupil radius values.
        - [4]: pupil intensity values.
        - [6]: stage position values.
    - mu: A scalar value used to scale certain loss components.
    - w: A tensor or list of weights for each component of the loss function.
    - psfnorm: A scalar value that serves as a normalization factor for the point spread function, defaulted to 1.0.
    
    Returns:
    - A scalar tensor representing the computed loss value.
    """
    
    # Calculate the difference between model predictions and actual data
    mydiff = model - data

    # Compute normalized mean squared error (MSE) using two different normalizations
    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)     
    mse_norm2 = tf.reduce_mean(tf.reduce_mean(tf.square(mydiff), axis=(-2, -1)) / tf.math.reduce_max(tf.square(data), axis=(-2, -1))) * 200
    
    # Calculate the log-likelihood (LL) of the model predictions with respect to the data
    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])  # Take mean of finite values

    # Extract specific variable values for loss computation
    bg = variables[1]        # Background value
    intensity = variables[2] # Intensity value
    pupilR = variables[3]    # Pupil radius (real part)
    pupilI = variables[4]    # Pupil radius (imaginary part)
    stagepos = variables[6]  # Stage position value
    zpos = variables[0][:, 0, ...]  # Z-position values extracted

    # Calculate the mean of the squared minimum of background, z-position, and intensity
    bgmin = tf.reduce_mean(tf.math.square(tf.math.minimum(bg, 0)))
    zmin = tf.reduce_mean(tf.math.square(tf.math.minimum(zpos, 0))) + tf.reduce_mean(tf.math.square(tf.math.minimum(stagepos, 0)))
    intensitymin = tf.reduce_mean(tf.math.square(tf.math.minimum(intensity, 0)))

    # Compute a normalization term related to the point spread function norm
    Inorm = tf.math.square(tf.math.minimum(psfnorm - 0.97, 0))

    # Calculate the sum of squared differences in pupil radius values across two dimensions (x and y)
    dfxy1 = tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilI, n=1, axis=-1))) + tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilI, n=1, axis=-2)))
    dfxy2 = tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilR, n=1, axis=-1))) + tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupilR, n=1, axis=-2)))
    dfxy = dfxy2  # Store the second derivative sum for loss calculation

    # Calculate the final loss value based on various components weighted by their respective weights
    loss = LL * w[0] + bgmin * w[5] * mu + intensitymin * w[6] * mu + dfxy * w[2] + zmin * w[4] * mu + Inorm * w[7]
    
    return loss  # Return the computed loss value