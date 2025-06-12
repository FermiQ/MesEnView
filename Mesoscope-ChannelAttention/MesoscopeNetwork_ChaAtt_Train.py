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
This is the main training script for the LSV ensemble model, which includes a deconvolution module and an enhancement module. The script handles command-line arguments, dataset preparation, model initialization, loss function and optimization setup, training and validation loops, and logging of results.

The script supports three different network architectures. It also includes functionality for continuing training from a checkpoint, early stopping, and saving the best performing model.

The main training loop iterates over the training and validation sets, computing losses, SSIM, and PSNR, and logging the results to TensorBoard. The script also saves reconstructed images and PSF visualizations at specified intervals.

The code is a Python script for training a deep learning model, specifically a multi-scale super-resolution network for light-sheet microscopy image processing. It includes various functionalities such as loading datasets, setting up the neural network, defining loss functions and optimization, and implementing early stopping. The script also supports saving the model and its outputs at specified intervals.

To document this script, here's a brief overview of its main components:

1. **Data Preparation**: The script starts by preparing the dataset for training and validation. It loads the dataset, splits it into training and validation sets, and creates data loaders for both.

2. **Model Setup**: Depending on the specified network type, the script initializes different models and sets up the necessary transformations and loss functions.

3. **Training Loop**: The main training loop iterates over the specified number of epochs. For each epoch, it iterates over the training batches, performs a forward pass, calculates losses, updates the model parameters using the optimizer, and logs the training metrics.

4. **Validation**: After each training epoch, the script evaluates the model on the validation set, calculates validation metrics, and logs them.

5. **Model Saving and Early Stopping**: The script checks for early stopping conditions and saves the model and its outputs at specified intervals. It also updates the best model if the current validation SSIM score improves.

6. **TensorBoard Logging**: The script uses TensorBoard for logging the training and validation metrics.

7. **Checkpoint Saving**: The script saves a checkpoint of the model and optimizer state at specified intervals.
"""

from dataset import *  # Import all classes and functions from the dataset module
import os  # Import the os module for interacting with the operating system
import torch  # Import the main PyTorch library
from utils import *  # Import all utility functions and classes
import torch.nn as nn  # Import the neural network module from PyTorch
import skimage.io  # Import the skimage.io module for image input/output
import argparse  # Import argparse to handle command line arguments
from torchvision import transforms  # Import transforms for image preprocessing
import tifffile  # Import tifffile for reading/writing TIFF files
import pandas as pd  # Import pandas for data manipulation and analysis
from pytorch_msssim import MS_SSIM  # Import the multi-scale SSIM loss function
from math import log10, sqrt  # Import logarithmic and square root functions
import torch.optim.lr_scheduler as lr_scheduler  # Import learning rate scheduler
from model import *  # Import all classes and functions from the model module
from tensorboardX import SummaryWriter  # Import SummaryWriter for TensorBoard logging

# Set up argument parsing for command line options
parser = argparse.ArgumentParser(description='Train the network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mode', default='train', choices=['train', 'debug'], dest='mode')  # Mode of operation
parser.add_argument('--train_continue', default='off',  dest='train_continue')  # Option to continue training
parser.add_argument('--computer', default='local',choices=['local', 'scc'], dest='computer')  # Computer type
parser.add_argument("--num_gpu", type=int, default=[1], dest='num_gpu')  # Number of GPUs to use
parser.add_argument('--num_epoch', type=int,  default=150, dest='num_epoch')  # Number of epochs for training
parser.add_argument('--batch_size', type=int, default=4, dest='batch_size')  # Size of training batch
parser.add_argument('--lr', type=float, default=5e-5, dest='lr')  # Learning rate
parser.add_argument('--train_ratio', type=float, default=0.9, dest='train_ratio')  # Ratio of data used for training
parser.add_argument('--dir_chck', default='./checkpoints', dest='dir_chck')  # Directory for saving checkpoints
parser.add_argument('--dir_log', default='./log', dest='dir_log')  # Directory for saving logs
parser.add_argument('--dir_save', default='./save', dest='dir_save')  # Directory for saving results
parser.add_argument('--num_freq_save', type=int,  default=10, dest='num_freq_save')  # Frequency of saving models
parser.add_argument("--local_rank", type=int, default=0, dest='local_rank')  # Local rank for distributed training
parser.add_argument("--early_stop", type=int, default=50, dest='early_stop', help='cancel=None')  # Early stopping patience
parser.add_argument("--num_psf", type=int, default=9)  # Number of point spread functions (PSFs)
parser.add_argument("--network", default='svfourier', help='multiwiener svfourier and cm2net')  # Network type
parser.add_argument("--ks", type=float, default=10.0)  # Parameter for deconvolution
parser.add_argument("--ps", type=int, default=1)  # Parameter for PSF size

# Main execution block
if __name__ == '__main__':
    PARSER = Parser(parser)  # Initialize the parser
    args = PARSER.get_arguments()  # Get parsed arguments
    PARSER.write_args()  # Write arguments to file
    PARSER.print_args()  # Print arguments to console

    torch.manual_seed(3407)  # Set manual seed for reproducibility
    torch.cuda.empty_cache()  # Clear GPU cache to free memory
    args.device = torch.device(0)  # Set device to the first GPU

    # Set data directory based on the selected computer type
    if args.computer=='local':
        args.dir_data = 'T:/simulation beads/2d/debug/'  # Local directory for data
    elif args.computer=='scc':
        args.dir_data='/ad/eng/research/eng_research_cisl/yqw/simulation beads/2d/lsv_2d_dataset/'  # SCC directory for data

    # Create directories for saving training and validation results
    dir_result_val = args.dir_save + '/val/'  # Directory for validation results
    dir_result_train = args.dir_save + '/train/'  # Directory for training results
    if not os.path.exists(os.path.join(dir_result_train)):
        os.makedirs(os.path.join(dir_result_train))  # Create training directory if it doesn't exist
    if not os.path.exists(os.path.join(dir_result_val)):
        os.makedirs(os.path.join(dir_result_val))  # Create validation directory if it doesn't exist

    # Prepare training data based on the selected network
    if args.network == 'cm2net':
        # Create the complete dataset using specified transformations
        transform_train = transforms.Compose([Noisecm2(), ToTensorcm2(), Crop()])  # Define dataset transformations
        whole_set = CM2Dataset(args.dir_data, transform=transform_train)  # Load dataset
        length = len(whole_set)  # Get the total length of the dataset
        train_size, validate_size = int(args.train_ratio * length), length - int(args.train_ratio * length)  # Split sizes for training and validation
        # Randomly split dataset into training and validation sets
        train_set, validate_set = torch.utils.data.random_split(whole_set, [train_size, validate_size])  
        train_set = Subset(train_set, isVal=False)  # Create subset for training
        validate_set = Subset(validate_set, isVal=True)  # Create subset for validation
    else:
        transform_train = transforms.Compose([Noise(), Resize(), ToTensor()])  # Define transformations for other datasets
        whole_set = MyDataset(args.dir_data, transform=transform_train)  # Load the custom dataset
        length = len(whole_set)  # Get dataset length
        train_size, validate_size = int(args.train_ratio*length), length-int(args.train_ratio*length)  # Calculate sizes
        # Randomly split dataset
        train_set, validate_set = torch.utils.data.random_split(whole_set, [train_size, validate_size])  
        print('training images:', len(train_set), 'testing images:', len(validate_set))  # Print dataset sizes

    # Create data loaders for training and validation
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=1, shuffle=True, drop_last=True)  # DataLoader for training data
    val_loader = torch.utils.data.DataLoader(validate_set, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=True)  # DataLoader for validation data

    # Calculate number of batches for training and validation
    num=len(args.num_gpu)  # Get the number of GPUs
    num_batch_train = int((train_size / (args.batch_size*num)) + ((train_size % (args.batch_size*num)) != 0))  # Calculate number of training batches
    num_batch_val = int((validate_size / args.batch_size) + ((validate_size % args.batch_size) != 0))  # Calculate number of validation batches

    ## Setup neural network based on selected architecture
    if args.network == 'multiwiener':
        psfs = skimage.io.imread(args.dir_data + '/psf_v11.tif')  # Read PSF image
        psfs = np.array(psfs)  # Convert to numpy array
        psfs = psfs.astype('float32') / psfs.max()  # Normalize PSFs
        psfs = psfs[:,57 * 2:3000, 94 * 2 + 156:4000 - 156]  # Crop PSFs
        psfs = np.pad(psfs, ((0,0),(657, 657), (350, 350)))  # Pad PSFs
        Ks = args.ks*np.ones((args.num_psf, 1, 1))  # Create a constant array for Ks
        deconvolution = MultiWienerDeconvolution2D(psfs, Ks).to(args.device)  # Initialize multi-Wiener deconvolution model
        enhancement = RCAN(args.num_psf).to(args.device)  # Initialize RCAN enhancement model
        model = LSVEnsemble2d(deconvolution, enhancement)  # Create the ensemble model

    if args.network == 'svfourier':
        # Initialize Fourier deconvolution and enhancement models
        deconvolution = FourierDeconvolution2D_ds(args.num_psf, args.ps).to(args.device)  # Initialize Fourier deconvolution model
        enhancement = RCAN(args.num_psf).to(args.device)  # Initialize RCAN enhancement model
        model = LSVEnsemble2d(deconvolution, enhancement)  # Create the ensemble model

    if args.network == 'cm2net':
        layers = 20  # Number of residual blocks
        # Initialize cm2net model with specified number of blocks and channels
        model = cm2net(numBlocks=layers, stackchannels=args.num_psf).to(args.device)  # Model for cm2net

    # Move model to the specified device
    model = model.to(args.device)

    ## Setup loss functions and optimization
    ssim_loss = MS_SSIM(data_range=1, size_average=True, channel=1)  # Initialize multi-scale SSIM loss
    l2_loss = nn.MSELoss()  # Mean Squared Error loss
    bce_loss = nn.BCELoss()  # Binary Cross Entropy loss
    params = model.parameters()  # Get model parameters
    optimizer = torch.optim.Adam(params, lr=args.lr)  # Initialize Adam optimizer
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 50, eta_min=1e-6)  # Cosine Annealing learning rate scheduler

    ## Load model from checkpoint if continuing training
    st_epoch = 0  # Start epoch for training

    # Logger for tracking losses
    losslogger = pd.DataFrame()  # Initialize loss logger
    if args.train_continue == 'on':
        # Load model, optimizer, starting epoch and loss logger from checkpoint
        model, optimizer, st_epoch, losslogger = load(args.dir_chck, model, optimizer, epoch=[], mode=args.mode)

    # Initialize variables for saving the best model
    best_ssim = 0  # Best SSIM score
    trigger = 0  # Trigger for early stopping
    best_loss = 10e7  # Best loss initialized to a high value

    ## Setup TensorBoard for logging
    dir_log = args.dir_log  # Directory for TensorBoard logs
    if not os.path.exists(os.path.join(dir_log)):
        os.makedirs(os.path.join(dir_log))  # Create log directory if it doesn't exist
    writer = SummaryWriter(log_dir=dir_log)  # Initialize TensorBoard writer

    # Main training loop
    for epoch in range(st_epoch + 1, args.num_epoch + 1):
        ## Training phase
        model.train()  # Set model to training mode
        loss_train = []  # List to store training losses
        ssim_train = []  # List to store training SSIM scores
        psnr_train = []  # List to store training PSNR scores
        for batch, data in enumerate(train_loader, 1):  # Iterate over training batches
            def should(freq):  # Helper function to check if logging is needed
                return freq > 0 and (batch % freq == 0 or batch == num_batch_train)

            # Ground truth shape is [Batch,H,W], Output shape is [Batch,1,H,W]
            if args.network == 'cm2net':
                meas = data['meas'].to(args.device)  # Move measurements to device
                gt = data['gt'].to(args.device)  # Move ground truth to device
                demix = data['demix'].to(args.device)  # Move demixed data to device
                optimizer.zero_grad()  # Zero gradients
                demix_output, output = model(meas)  # Forward pass
                # Calculate losses
                loss_demix = bce_loss(demix_output, demix) + l2_loss(demix_output, demix)
                loss_recon = bce_loss(torch.squeeze(output, 1), gt) + l2_loss(torch.squeeze(output, 1), gt)
                loss = loss_demix + loss_recon  # Total loss
            else:
                meas = data['meas'].to(args.device)  # Move measurements to device
                gt = data['gt'].to(args.device)  # Move ground truth to device
                optimizer.zero_grad()  # Zero gradients
                output = model(meas)  # Forward pass
                loss = bce_loss(torch.squeeze(output, 1), gt) + l2_loss(torch.squeeze(output, 1), gt)  # Total loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            # Normalize outputs for SSIM and PSNR calculations
            output_n = (output - torch.min(output)) / (torch.max(output) - torch.min(output))  # Normalize output
            gt_n = (gt - torch.min(gt)) / (torch.max(gt) - torch.min(gt))  # Normalize ground truth
            ssim = ssim_loss(output_n, gt_n.unsqueeze(1))  # Calculate SSIM
            psnr = 20 * torch.log10(torch.max(output) / sqrt(l2_loss(torch.squeeze(output, 1), gt)))  # Calculate PSNR
            # Collect losses
            loss_train += [loss.item()]  # Store loss
            ssim_train += [ssim.item()]  # Store SSIM
            psnr_train += [psnr.item()]  # Store PSNR

            if args.local_rank == 0:  # Print training progress for the main process
                print('TRAIN: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f SSIM: %.4f'
                      % (epoch, batch, num_batch_train, np.mean(loss_train), np.mean(ssim_train)))

        scheduler.step()  # Update learning rate

        # Save model and outputs at specified frequency
        if args.local_rank == 0 and (epoch % args.num_freq_save) == 0:
            gt = gt.data.cpu().numpy()  # Move ground truth to CPU
            x_recon = torch.squeeze(output, 1).data.cpu().numpy()  # Move output to CPU
            for j in range(gt.shape[0]):  # Iterate over batch
                # Save ground truth and reconstructed images as TIFF files
                im_gt = (np.clip(gt[j, ...]/ np.max(gt[j, ...]), 0, 1) * 255).astype(np.uint8)
                im_recon = (np.clip(x_recon[j, ...] / np.max(x_recon[j, ...]), 0, 1) * 255).astype(np.uint8)
                tifffile.imwrite((dir_result_train + str(epoch) + '_recon' + '.tif'), im_recon.squeeze())  # Save reconstructed image
                tifffile.imwrite((dir_result_train + str(epoch) + '_gt' + '.tif'), im_gt.squeeze())  # Save ground truth image

        ## Validation phase
        with torch.no_grad():  # Disable gradient calculation
            model.eval()  # Set model to evaluation mode
            loss_val = []  # List to store validation losses
            ssim_val = []  # List to store validation SSIM scores
            psnr_val = []  # List to store validation PSNR scores

            for batch, data in enumerate(val_loader, 1):  # Iterate over validation batches
                # Forward simulation (add noise)
                if args.network == 'cm2net':
                    meas = data['meas'].to(args.device)  # Move measurements to device
                    gt = data['gt'].to(args.device)  # Move ground truth to device
                    demix = data['demix'].to(args.device)  # Move demixed data to device
                    demix_output, output = model(meas)  # Forward pass
                    # Calculate losses
                    loss_demix = bce_loss(demix_output, demix) + l2_loss(demix_output, demix)
                    loss_recon = bce_loss(torch.squeeze(output, 1), gt) + l2_loss(torch.squeeze(output, 1), gt)
                    loss = loss_demix + loss_recon  # Total loss
                else:
                    meas = data['meas'].to(args.device)  # Move measurements to device
                    gt = data['gt'].to(args.device)  # Move ground truth to device
                    output = model(meas)  # Forward pass
                    loss = bce_loss(torch.squeeze(output, 1), gt) + l2_loss(torch.squeeze(output, 1), gt)  # Total loss
                output_n = (output - torch.min(output)) / (torch.max(output) - torch.min(output))  # Normalize output
                gt_n = (gt - torch.min(gt)) / (torch.max(gt) - torch.min(gt))  # Normalize ground truth
                ssim = ssim_loss(output_n, gt_n.unsqueeze(1))  # Calculate SSIM
                psnr = 20 * torch.log10(torch.max(output) / sqrt(l2_loss(torch.squeeze(output, 1), gt)))  # Calculate PSNR
                # Collect losses
                loss_val += [loss.item()]  # Store loss
                ssim_val += [ssim.item()]  # Store SSIM
                psnr_val += [psnr.item()]  # Store PSNR

                if args.local_rank == 0:  # Print validation progress for the main process
                    print('VALID: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f SSIM: %.4f'
                          % (epoch, batch, num_batch_val, np.mean(loss_val), np.mean(ssim_val)))

            # Check if the current epoch is the first epoch.
            if epoch == 1:
            # Move ground truth data to CPU and convert it to a NumPy array.
            gt = gt.data.cpu().numpy()
            # Normalize the last slice of ground truth data and convert it to an 8-bit image.
            im_gt = (np.clip(gt[-1, ...] / np.max(gt[-1, ...]), 0, 1) * 255).astype(np.uint8)
            # Save the ground truth image as a TIFF file.
            tifffile.imwrite((dir_result_val + str(epoch) + '_gt' + '.tif'), im_gt.squeeze())

# Check if the process is the main one and if the current epoch is a multiple of num_freq_save.
if args.local_rank == 0 and (epoch % args.num_freq_save) == 0:
    # Move the output data to CPU and convert it to a NumPy array.
    x_recon = output.data.cpu().numpy()
    # Normalize the last reconstructed output and convert it to an 8-bit image.
    im_recon = (np.clip(x_recon[-1, ...] / np.max(x_recon[-1, ...]), 0, 1) * 255).astype(np.uint8)
    # Save the reconstructed image as a TIFF file.
    tifffile.imwrite((dir_result_val + str(epoch) + '_recon' + '.tif'), im_recon.squeeze())

    # If the network type is 'svfourier', retrieve and process PSFs.
    if args.network == 'svfourier':
        # Get the real and imaginary parts of the PSFs and move them to CPU.
        psfs_re = model.deconvolution.psfs_re.detach().cpu().numpy()
        psfs_im = model.deconvolution.psfs_im.detach().cpu().numpy()
        # Combine real and imaginary parts into a complex frequency representation.
        psf_freq = psfs_re + psfs_im * 1j
        # Compute the inverse FFT to get the PSF in spatial domain.
        psf = np.fft.ifftshift(np.fft.irfft2(psf_freq, axes=(-2, -1)))
        # Create a maximum intensity projection of the PSF.
        psf_mip = np.max(psf, 0).squeeze()
        # Normalize to 16-bit range and convert to int16.
        psf_mip = (psf_mip / np.abs(psf_mip).max() * 65535.0).astype('int16')
        # Save the PSF maximum intensity projection as a TIFF file.
        tifffile.imwrite((dir_result_val + str(epoch) + '_psf_mip' + '.tif'), psf_mip, photometric='minisblack')

    # If the network type is 'multiwiener', retrieve and process PSFs.
    if args.network == 'multiwiener':
        # Get the PSFs and move them to CPU.
        psf = model.deconvolution.psfs.detach().cpu().numpy()
        # Create a maximum intensity projection of the PSF.
        psf_mip = np.max(psf, 0).squeeze()
        # Normalize to 16-bit range and convert to int16.
        psf_mip = (psf_mip / np.abs(psf_mip).max() * 65535.0).astype('int16')
        # Save the PSF maximum intensity projection as a TIFF file.
        tifffile.imwrite((dir_result_val + str(epoch) + '_psf_mip' + '.tif'), psf_mip, photometric='minisblack')

# If the process is the main one, log training and validation metrics.
if args.local_rank == 0:
    # Create a DataFrame to store the mean metrics.
    df = pd.DataFrame()
    df['loss_train'] = pd.Series(np.mean(loss_train))  # Mean training loss.
    df['ssim_train'] = pd.Series(np.mean(ssim_train))  # Mean training SSIM.
    df['psnr_train'] = pd.Series(np.mean(psnr_train))  # Mean training PSNR.
    df['loss_val'] = pd.Series(np.mean(loss_val))      # Mean validation loss.
    df['ssim_val'] = pd.Series(np.mean(ssim_val))      # Mean validation SSIM.
    df['psnr_val'] = pd.Series(np.mean(psnr_val))      # Mean validation PSNR.
    # Append the recorded metrics to the logger.
    losslogger = losslogger.append(df)
    # Log each metric to TensorBoard.
    writer.add_scalar('Loss/loss_train', np.mean(loss_train), epoch)
    writer.add_scalar('SSIM/ssim_train', np.mean(ssim_train), epoch)
    writer.add_scalar('PSNR/psnr_train', np.mean(psnr_train), epoch)
    writer.add_scalar('Loss/loss_val', np.mean(loss_val), epoch)
    writer.add_scalar('SSIM/ssim_val', np.mean(ssim_val), epoch)
    writer.add_scalar('PSNR/psnr_val', np.mean(psnr_val), epoch)

# Increment the trigger for early stopping.
trigger += 1
# If the process is the main one and the current SSIM is the best, save the model.
if args.local_rank == 0 and (np.mean(ssim_val) > best_ssim):
    save(args.dir_chck+ '/best_model/', model, optimizer, epoch, losslogger)
    best_ssim = np.mean(ssim_val)  # Update best SSIM.
    print("=>saved best model")  # Notify user of the saved model.
    trigger = 0  # Reset trigger.

# Check for early stopping condition.
if not args.early_stop is not None and args.local_rank == 0:
    if trigger >= args.early_stop:  # If trigger exceeds early stopping threshold.
        print("=> early stop")  # Notify user of early stopping.
        break  # Exit the training loop.

# Save a checkpoint if the process is the main one and the epoch is a multiple of num_freq_save.
if args.local_rank == 0 and (epoch % args.num_freq_save) == 0:
    save(args.dir_chck, model, optimizer, epoch, losslogger)  # Save the model and optimizer state.
