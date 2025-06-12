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
This script is used to train and test a network for image deconvolution and enhancement.

The script sets up an argument parser to accept various configuration parameters for the training and testing process, such as the directory for input data, the network type, the model name, batch size, number of training epochs, and more.

Based on the specified network type, the script initializes the appropriate deconvolution and enhancement models, and then loads the pre-trained model weights. The script then performs inference on a measurement image, normalizes the output, and saves the reconstructed image to a TIFF file.

The script is a Python file used for training and testing a network for image deconvolution and enhancement. It utilizes the PyTorch library for deep learning tasks and the SciPy library for scientific computing. The script is designed to work with TIFF images and can be configured using command-line arguments.

A detailed explanation of the script:

1. **Importing Libraries**: The script starts by importing necessary libraries, including `skimage.io` for image input/output operations, `argparse` for parsing command-line arguments, and `tifffile` for reading and writing TIFF files. It also imports all classes and functions from a custom `model` module.

2. **Setting Up Argument Parser**: The script sets up an argument parser to accept various configuration parameters for the training and testing process. These parameters include the directory for input data, the network type, the model name, batch size, number of training epochs, and more.

3. **Model Initialization**: Based on the specified network type, the script initializes the appropriate deconvolution and enhancement models and loads the pre-trained model weights. The supported network types are 'multiwiener', 'svfourier', and 'cm2net'.

4. **Loading and Preprocessing Measurement Image**: The script loads a measurement image from a TIFF file, normalizes it, and pads it for processing.

5. **Inference**: The script performs inference using the model on the prepared measurement image and normalizes the output for saving.

6. **Saving Reconstructed Image**: Finally, the script prepares the reconstructed image for saving by scaling it to the uint16 range and then saves it to a TIFF file using the `tifffile` library.

The script also includes some utility functions and classes in the `model` module, which are not shown in this code snippet. These functions and classes are used to define the network architectures and perform other necessary operations for image deconvolution and enhancement.
"""

import skimage.io  # Importing skimage library for image input/output operations
import argparse  # Importing argparse module for parsing command-line arguments
import tifffile  # Importing tifffile library for reading and writing TIFF files
from model import *  # Importing all classes and functions from the model module

# Setup argument parser for command-line execution
parser = argparse.ArgumentParser(description='Train the network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Adding various arguments for configuring the model and training process
parser.add_argument('--dir_data', default='test/', dest='dir_data',
                    help='Directory for input data')
parser.add_argument("--network", default='multiwiener', 
                    help='Specify the network type: multiwiener, svfourier, or cm2net')
parser.add_argument('--model_name', default='/multiwiener', dest='model_name',
                    help='Name of the model to load')
parser.add_argument('--batch_size', type=int, default=1, dest='batch_size',
                    help='Size of each training batch')
parser.add_argument("--local_rank", type=int, default=0, dest='local_rank',
                    help='Local rank for distributed training')
parser.add_argument("--num_psf", type=int, default=9,
                    help='Number of Point Spread Functions (PSFs)')
parser.add_argument("--ps", type=int, default=1,
                    help='Parameter for PSF')
parser.add_argument("--ks", type=float, default=10.0,
                    help='Scaling factor for the PSF')
parser.add_argument("--epoch", type=int, default=54,
                    help='Number of training epochs')
parser.add_argument('--dir_chck', default='./checkpoints', dest='dir_chck',
                    help='Directory to save checkpoints')
parser.add_argument("--distributed", type=bool, default=False, dest='distributed',
                    help='Enable distributed training or not')
parser.add_argument('--lr', type=float, default=5e-5, dest='lr',
                    help='Learning rate for the optimizer')
parser.add_argument('--mode', default='test', choices=['train', 'test'], dest='mode',
                    help='Mode of operation: train or test')

# Parse the command-line arguments
args = parser.parse_args(''.split())

# Create a directory for storing results if it doesn't exist
dir_result_test = args.dir_data + args.model_name
if not os.path.exists(os.path.join(dir_result_test)):
    os.makedirs(os.path.join(dir_result_test))  # Create the directory

# Set up GPU configuration based on local rank
args.num_gpu = list(range(torch.cuda.device_count()))  # List all available GPUs
torch.cuda.set_device(args.local_rank)  # Set the current GPU device
args.device = torch.device(f'cuda:{args.local_rank}')  # Assign device for computation

# Initialize the model based on the specified network type
if args.network == 'multiwiener':
    # Load PSF from a TIFF file and preprocess it
    psfs = skimage.io.imread('pretrained_model\psf.tif')
    psfs = np.array(psfs)  # Convert to NumPy array
    psfs = psfs.astype('float32') / psfs.max()  # Normalize PSF
    psfs = psfs[:, 57 * 2:3000, 94 * 2 + 156:4000 - 156]  # Crop PSF
    psfs = np.pad(psfs, ((0, 0), (657, 657), (350, 350)))  # Pad PSF with zeros
    Ks = args.ks * np.ones((args.num_psf, 1, 1))  # Create scaling factors for each PSF
    deconvolution = MultiWienerDeconvolution2D(psfs, Ks).to(args.device)  # Create deconvolution model
    enhancement = RCAN(args.num_psf).to(args.device)  # Create enhancement model
    model = LSVEnsemble2d(deconvolution, enhancement)  # Combine models

if args.network == 'svfourier':
    deconvolution = FourierDeconvolution2D_ds(args.num_psf, args.ps).to(args.device)  # Create Fourier deconvolution model
    enhancement = RCAN(args.num_psf).to(args.device)  # Create enhancement model
    model = LSVEnsemble2d(deconvolution, enhancement)  # Combine models

if args.network == 'cm2net':
    layers = 20  # Number of residual blocks in the model
    model = cm2net(numBlocks=layers, stackchannels=args.num_psf).to(args.device)  # Create cm2net model

# Load pre-trained model weights
params = model.parameters()  # Retrieve model parameters
dict_net = torch.load('pretrained_model/%s.pth' % (args.model_name))  # Load the model state dict
model.load_state_dict(dict_net)  # Update model with loaded parameters
print('Successfully loaded network %s' % (args.network))  # Confirmation message

# Set the model to evaluation mode and perform inference
with torch.no_grad():  # Disable gradient calculation for inference
    model.eval()  # Set the model to evaluation mode
    meas = skimage.io.imread(args.dir_data + 'meas.tif').astype('float32')  # Load and normalize the measurement image
    if args.network == 'cm2net':
        tot_len = 2400  # Total length for cropping this 
        tmp_pad = 900  # Padding size
        meas = np.pad(meas, ((tmp_pad, tmp_pad), (tmp_pad, tmp_pad)), 'constant')  # Pad the measurement image
        # Define locations for cropping the measurement image
        loc = [(664, 1192), (664, 2089), (660, 2982),
               (1564, 1200), (1557, 2094), (1548, 2988),
               (2460, 1206), (2452, 2102), (2444, 2996)]
        # Stack cropped measurements based on defined locations
        meas = np.stack([
            meas[x - (tot_len // 2) + tmp_pad:x + (tot_len // 2) + tmp_pad,
            y - (tot_len // 2) + tmp_pad:y + (tot_len // 2) + tmp_pad] for x, y in loc
        ])
    else:
        # Crop and pad the measurement image for other networks
        meas = meas[57 * 2:3000, 94 * 2 + 156:4000 - 156]
        meas = np.pad(meas, ((657, 657), (350, 350)))  # Pad cropped image

    # Convert measurement to a tensor and move to the configured device
    meas = torch.from_numpy(meas / meas.max()).unsqueeze(0)  # Normalize and add batch dimension this for the application of the basis
    meas = meas.to(args.device)  # Move tensor to the specified device

    # Perform inference using the model
    if args.network == 'cm2net':
        demix_output, output = model(meas)  # For cm2net, get both demixing and output
    else:
        output = model(meas)  # For other networks, just get output

    # Normalize the output for saving
    output_n = (output - torch.min(output)) / (torch.max(output) - torch.min(output))  # Normalize output
    x_recon = output.data.cpu().numpy().squeeze()  # Convert output tensor to NumPy array and remove singleton dimensions
    # Prepare the reconstructed image for saving
    im_recon = (np.clip(x_recon / np.max(x_recon), 0, 1) * 65535).astype(np.uint16)  # Scale to uint16 range
    # Save the reconstructed image to a TIFF file
    tifffile.imwrite((dir_result_test + '/recon.tif'), im_recon.squeeze())  # Write the image to file