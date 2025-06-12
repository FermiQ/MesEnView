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
from pickle import FALSE
import h5py as h5  # For handling HDF5 files
import czifile as czi  # For reading CZI files
import numpy as np  # For numerical operations
import scipy as sp  # For scientific computations
import matplotlib.pyplot as plt  # For plotting
from matplotlib import gridspec  # For creating grids in plots
from skimage import io  # For image I/O
import sys  # To manipulate the Python runtime environment
import glob  # For file path manipulation
import scipy.io as sio  # For reading MATLAB files
import tensorflow as tf  # For machine learning
import json  # For JSON file handling
from tqdm import tqdm  # For progress bars in loops
from PIL import Image  # For image handling
from omegaconf import OmegaConf  # For configuration management
import os  # For operating system interactions
from tkinter import EXCEPTION, messagebox as mbox  # For GUI interactions
from dotted_dict import DottedDict  # For accessing dictionary items with dot notation
from .dataloader import dataloader  # Importing the dataloader module

# Importing functions and classes from the learning module related to PSF learning
from .learning import ( 
    PreprocessedImageDataSingleChannel,
    PreprocessedImageDataMultiChannel,
    PreprocessedImageDataSingleChannel_smlm,
    PreprocessedImageDataMultiChannel_smlm,
    Fitter,
    PSFVolumeBased,
    PSFPupilBased,
    PSFZernikeBased,
    PSFZernikeBased_FD,
    PSFVolumeBased4pi,
    PSFPupilBased4pi,
    PSFZernikeBased4pi,
    PSFMultiChannel,
    PSFMultiChannel_smlm,
    PSFMultiChannel4pi,
    PSFZernikeBased_vector_smlm,
    PSFPupilBased_vector_smlm,
    PSFZernikeBased_FD_smlm,
    PSFMultiChannel4pi_smlm,
    PSFZernikeBased4pi_smlm,
    L_BFGS_B,
    psf2cspline_np,
    mse_real,
    mse_real_zernike,
    mse_real_zernike_FD,
    mse_real_zernike_smlm,
    mse_real_pupil_smlm,
    mse_real_zernike_FD_smlm,
    mse_real_4pi,
    mse_zernike_4pi,
    mse_zernike_4pi_smlm,
    mse_real_pupil,
    mse_pupil_4pi,
    mse_real_All,
    mse_real_4pi_All
)

#%% Dictionaries mapping PSF types to their corresponding classes and loss functions
PSF_DICT = dict(
    voxel=PSFVolumeBased, 
    pupil=PSFPupilBased,
    pupil_vector=PSFPupilBased,
    zernike=PSFZernikeBased,
    zernike_vector=PSFZernikeBased,
    zernike_FD=PSFZernikeBased_FD,
    zernike_vector_FD=PSFZernikeBased_FD,
    insitu_zernike=PSFZernikeBased_vector_smlm,
    insitu_pupil=PSFPupilBased_vector_smlm,
    insitu_FD=PSFZernikeBased_FD_smlm
)

# Dictionary for mapping loss functions based on PSF types
LOSSFUN_DICT = dict(
    voxel=mse_real, 
    pupil=mse_real_pupil,
    pupil_vector=mse_real_pupil,
    zernike=mse_real_zernike,
    zernike_vector=mse_real_zernike,
    zernike_FD=mse_real_zernike_FD,
    zernike_vector_FD=mse_real_zernike_FD,
    insitu_zernike=mse_real_zernike_smlm,
    insitu_pupil=mse_real_pupil_smlm,
    insitu_FD=mse_real_zernike_FD_smlm
)

# PSF and loss function dictionaries specific to the 4pi configuration
PSF_DICT_4pi = dict(
    voxel=PSFVolumeBased4pi, 
    pupil=PSFPupilBased4pi,
    zernike=PSFZernikeBased4pi,
    insitu_zernike=PSFZernikeBased4pi_smlm
)

LOSSFUN_DICT_4pi = dict(
    voxel=mse_real_4pi, 
    pupil=mse_pupil_4pi,
    zernike=mse_zernike_4pi,
    insitu_zernike=mse_zernike_4pi_smlm
)

# Class definition for the PSF learning library
class psflearninglib:
    def __init__(self, param=None):
        """
        Initializes the psflearninglib instance.
        
        Parameters:
        param (DottedDict): Configuration parameters for the PSF learning process.
        """
        self.param = param  # Store the parameters
        self.loc_FD = None  # Initialize the FD localization variable

    def getpsfclass(self):
        """
        Determines the PSF class and loss function based on the parameters.
        """
        param = self.param
        PSFtype = param.PSFtype  # Get PSF type from parameters
        channeltype = param.channeltype  # Get channel type
        lossfun = LOSSFUN_DICT[PSFtype]  # Get corresponding loss function
        lossfunmulti = None  # Initialize multi-channel loss function variable

        # Determine PSF class based on channel type
        if channeltype == 'single':
            psfclass = PSF_DICT[PSFtype]  # Single channel PSF class
            psfmulticlass = None  # No multi-channel class for single channel
        elif channeltype == 'multi':
            psfclass = PSF_DICT[PSFtype]  # Multi-channel PSF class
            if 'insitu' in PSFtype:
                psfmulticlass = PSFMultiChannel_smlm  # SMLM specific multi-channel class
            else:
                psfmulticlass = PSFMultiChannel  # Regular multi-channel class
            lossfunmulti = mse_real_All  # Multi-channel loss function
        elif channeltype == '4pi':
            psfclass = PSF_DICT_4pi[PSFtype]  # 4pi PSF class
            lossfun = LOSSFUN_DICT_4pi[PSFtype]  # 4pi specific loss function
            if 'insitu' in PSFtype:
                psfmulticlass = PSFMultiChannel4pi_smlm  # SMLM specific for 4pi
            else:
                psfmulticlass = PSFMultiChannel4pi  # Regular 4pi multi-channel class
            lossfunmulti = mse_real_4pi_All  # 4pi multi-channel loss function

        # Store determined classes and loss functions
        self.psf_class = psfclass
        self.psf_class_multi = psfmulticlass
        self.loss_fun = lossfun
        self.loss_fun_multi = lossfunmulti
        return

    def load_data(self, frange=None):
        """
        Loads image data based on the provided configuration parameters.
        
        Parameters:
        frange (tuple): Optional; a range specifying which files to load.
        
        Returns:
        numpy.ndarray: The loaded images as a 5D numpy array.
        """
        param = self.param
        varname = param.varname  # Variable name from parameters
        format = param.format  # Data format from parameters
        channeltype = param.channeltype  # Channel type (single, multi, 4pi)
        PSFtype = param.PSFtype  # PSF type
        ref_channel = param.ref_channel  # Reference channel for multi-channel data
        filelist = param.filelist  # List of files to load

        loader = dataloader(param)  # Create a loader instance
        if not filelist:
            filelist = loader.getfilelist()  # Get the file list if not provided

        if frange:
            filelist = filelist[frange[0]:frange[1]]  # Slice file list based on frange
        
        # Load images based on the specified format
        if format == '.mat':
            imagesall = loader.loadmat(filelist)  # Load MATLAB files
        elif (format == '.tif') or (format == '.tiff'):
            imagesall = loader.loadtiff(filelist)  # Load TIFF files
        elif format == '.czi':
            imagesall = loader.loadczi(filelist)  # Load CZI files
        elif format == '.h5':
            imagesall = loader.loadh5(filelist)  # Load HDF5 files
        else:
            raise TypeError('supported data format is '+'.mat,'+'.tif,'+'.czi,'+'.h5.')

        # Process images based on channel type
        if channeltype == '4pi':
            if 'insitu' in PSFtype:
                images = np.transpose(imagesall, (1, 0, 2, 3, 4))  # Transpose for insitu 4pi
            else:
                if varname:
                    images = np.transpose(imagesall, (1, 0, 2, 3, 4, 5))  # Transpose with variable name
                else:
                    images = np.transpose(imagesall, (1, 0, 3, 2, 4, 5))  # Regular transpose for 4pi
        elif channeltype == 'multi':
            images = np.transpose(imagesall, (1, 0, 2, 3, 4))  # Transpose for multi-channel
            Nchannel = images.shape[0]  # Number of channels
            
            # Calculate defocus values for each channel
            defocus = []
            for i in range(Nchannel):
                defocus.append(param.option.multi.defocus_offset + i * param.option.multi.defocus_delay)
            # Swap defocus values based on reference channel
            defocus[0], defocus[ref_channel] = defocus[ref_channel], defocus[0]
            self.param.option.multi.defocus = defocus
            id = list(range(images.shape[0]))  # Create a list of channel indices
            id[0], id[ref_channel] = id[ref_channel], id[0]  # Swap channel indices
            images = images[id]  # Reorder images based on new indices
        else:
            images = imagesall  # Use the loaded images directly for single channel

        # Reshape images for insitu PSF types
        if 'insitu' in PSFtype:
            if channeltype == 'single':
                images = images.reshape(-1, images.shape[-2], images.shape[-1])  # Reshape single channel
            elif channeltype == 'multi':
                images = images.reshape(images.shape[0], -1, images.shape[-2], images.shape[-1])  # Multi-channel reshape
            elif channeltype == '4pi':
                images = images.reshape(images.shape[0], -1, images.shape[-2], images.shape[-1])  # 4pi reshape
        
        # Swap axes if specified in parameters
        if param.swapxy:
            tmp = np.zeros(images.shape[:-2] + (images.shape[-1], images.shape[-2]), dtype=np.float32)            
            tmp[0:] = np.swapaxes(images[0:], -1, -2)  # Swap the last two axes
            images = tmp

        # Reverse the image direction if specified
        if (param.stage_mov_dir == 'reverse') & (param.datatype == 'bead'):
            images = np.flip(images, axis=-3)  # Flip along the third axis
        
        print(images.shape)  # Print the shape of the loaded images
        return images  # Return the processed images

    def prep_data(self, images):
        """
        Prepares the data for PSF learning by processing and applying necessary transformations.
        
        Parameters:
        images (numpy.ndarray): The images to be prepared for learning.
        
        Returns:
        PreprocessedImageDataSingleChannel or PreprocessedImageDataMultiChannel: 
        An object containing the preprocessed image data.
        """
        param = self.param
        peak_height = param.roi.peak_height  # Peak height for ROI detection
        roi_size = param.roi.roi_size  # Size of the region of interest
        gaus_sigma = param.roi.gauss_sigma  # Gaussian sigma for filtering
        kernel = param.roi.max_kernel  # Maximum kernel size
        pixelsize_x = param.pixel_size.x  # Pixel size in the x direction
        pixelsize_y = param.pixel_size.y  # Pixel size in the y direction
        pixelsize_z = param.pixel_size.z  # Pixel size in the z direction
        bead_radius = param.roi.bead_radius  # Radius of the bead
        showplot = param.plotall  # Flag to show plots
        zT = param.fpi.modulation_period  # Modulation period for the PSF
        PSFtype = param.PSFtype  # PSF type
        channeltype = param.channeltype  # Channel type
        fov = list(param.FOV.values())  # Field of view
        skew_const = param.LLS.skew_const  # Skew constants for correction
        maxNobead = param.roi.max_bead_number  # Maximum number of beads

        # Calculate z-index range based on field of view
        zstart = fov[-3]  # Start index for z
        zend = images.shape[-3] + fov[-2]  # End index for z
        zstep = fov[-1]  # Step size for z
        zind = range(zstart, zend, zstep)  # Range of z indices
        ims = np.swapaxes(images, 0, -3)  # Swap the first and last axes for processing

        ims = ims[zind]  # Select relevant images based on z indices
        images = np.swapaxes(ims, 0, -3)  # Swap axes back to original order

        # Determine if the PSF type is a volume
        if PSFtype == 'voxel':
            isvolume = True
            padpsf = False
        else:
            isvolume = False
            padpsf = False

        # Create a data object based on the channel type
        if channeltype == 'single':
            if 'insitu' in PSFtype:
                dataobj = PreprocessedImageDataSingleChannel_smlm(images)  # SMLM single channel
            else:
                dataobj = PreprocessedImageDataSingleChannel(images)  # Regular single channel
        elif channeltype == '4pi':
            if 'insitu' in PSFtype:
                dataobj = PreprocessedImageDataMultiChannel_smlm(images, PreprocessedImageDataSingleChannel_smlm, is4pi=True)  # SMLM multi-channel 4pi
            else:
                dataobj = PreprocessedImageDataMultiChannel(images, PreprocessedImageDataSingleChannel, is4pi=True)  # Regular multi-channel 4pi
        elif channeltype == 'multi':
            if 'insitu' in PSFtype:
                dataobj = PreprocessedImageDataMultiChannel_smlm(images, PreprocessedImageDataSingleChannel_smlm)  # SMLM multi-channel
            else:
                dataobj = PreprocessedImageDataMultiChannel(images, PreprocessedImageDataSingleChannel)  # Regular multi-channel
        
        # Handle field of view and skew constants
        if fov[2] == 0:
            fov = None  # Set FOV to None if not specified
        if (skew_const[0] == 0.0) & (skew_const[1] == 0.0):
            skew_const = None  # Set skew to None if no skew constants are provided
            
        # Process the data object with the specified parameters
        dataobj.process(
            roi_size=roi_size,
            gaus_sigma=gaus_sigma,
            min_border_dist=list(np.array(roi_size) // 2 + 1),  # Minimum border distance
            min_center_dist=np.max(roi_size),  # Minimum center distance
            FOV=fov,  # Field of view
            max_threshold=peak_height,  # Maximum threshold for peak detection
            max_kernel=kernel,  # Maximum kernel size
            pixelsize_x=pixelsize_x,  # Pixel size in x
            pixelsize_y=pixelsize_y,  # Pixel size in y
            pixelsize_z=pixelsize_z,  # Pixel size in z
            bead_radius=bead_radius,  # Bead radius
            modulation_period=zT,  # Modulation period
            plot=showplot,  # Flag to display plots
            padPSF=padpsf,  # Flag for padding PSF
            isVolume=isvolume,  # Flag indicating if this is a volume
            skew_const=skew_const,  # Skew constants
            max_bead_number=maxNobead  # Maximum bead number
        )
        
        return dataobj  # Return the preprocessed data object

    def initializepsf(self):
        """
        Initializes the PSF object based on the current parameters and configurations.
        
        Returns:
        psfobj: The initialized PSF object ready for use in learning.
        """
        param = self.param  # Get parameters
        w = list(param.loss_weight.values())  # Get loss weights
        optionparam = param.option  # Get options
        batchsize = param.batch_size  # Get batch size

        # Create PSF object based on whether multi-channel is used
        if self.psf_class_multi is None:
            psfobj = self.psf_class(options=optionparam)  # Initialize single PSF object
            if 'vector' in param.PSFtype:
                psfobj.psftype = 'vector'  # Set PSF type to vector if applicable
        else:
            optimizer_single = L_BFGS_B(maxiter=50)  # Set up optimizer for learning
            optimizer_single.batch_size = batchsize  # Set batch size for optimizer
            psfobj = self.psf_class_multi(self.psf_class, optimizer_single, options=optionparam, loss_weight=w
            if 'vector' in param.PSFtype:
                psfobj.PSFtype = 'vector'

        return psfobj
    
    def learn_psf(self, dataobj, time=None):
    """
    Learns the point spread function (PSF) using the provided data object.
    
    Parameters:
    - dataobj: An object containing the data required for learning the PSF.
    - time: Optional; a specific time to consider during the learning process.
    
    Returns:
    - psfobj: An object initialized with PSF data.
    - fitter: An instance of the Fitter class used for fitting the PSF.
    """
    param = self.param  # Accessing parameters from the class
    rej_threshold = list(param.rej_threshold.values())  # Thresholds for rejecting outliers
    maxiter = param.iteration  # Maximum number of iterations for the optimizer
    w = list(param.loss_weight.values())  # Loss weights for the fitting process
    usecuda = param.usecuda  # Flag indicating if CUDA should be used
    showplot = param.plotall  # Flag to display plots
    optionparam = param.option  # Additional options for processing
    channeltype = param.channeltype  # Type of channel (e.g., single, multi)
    PSFtype = param.PSFtype  # Type of PSF to be learned
    roi_size = param.roi.roi_size  # Size of the region of interest
    batchsize = param.batch_size  # Batch size for processing
    pupilfile = optionparam.model.init_pupil_file  # File containing pupil information
    psfobj = self.initializepsf()  # Initializing PSF object

    # Loading pupil parameters from the provided pupil file
    if pupilfile:
        f = h5.File(pupilfile, 'r')  # Open the pupil file in read mode
        if channeltype == 'single':
            # Handling single channel case
            try:
                psfobj.initpupil = np.array(f['res']['pupil'])  # Load pupil data
            except:
                pass

            try:
                psfobj.Zoffset = np.array(f['res']['zoffset'])  # Load Z offset data
            except:
                pass

            try:
                psfobj.initpsf = np.array(f['res']['I_model_reverse']).astype(np.float32)  # Load PSF model
            except:
                psfobj.initpsf = np.array(f['res']['I_model']).astype(np.float32)

            try:
                psfobj.initzcoeff = np.array(f['res']['zernike_coeff']).astype(np.float32)  # Load Zernike coefficients
            except:
                pass
        else:
            # Handling multi-channel case
            Nchannels = len(dataobj.channels)  # Number of channels in the data object
            psfobj.initpupil = [None]*Nchannels  # Initialize pupil list for each channel
            psfobj.initpsf = [None]*Nchannels  # Initialize PSF list for each channel
            if channeltype == '4pi':
                psfobj.initA = [None]*Nchannels  # Initialize A model for 4pi channels
            for k in range(Nchannels):
                # Loading channel-specific data from the file
                try:
                    psfobj.initpupil[k] = np.array(f['res']['channel'+str(k)]['pupil'])  # Load pupil data for channel k
                except:
                    pass
                try:
                    psfobj.Zoffset = np.array(f['res']['channel'+str(k)]['zoffset'])  # Load Z offset for channel k
                except:
                    pass
                psfobj.initpsf[k] = np.array(f['res']['channel'+str(k)]['I_model']).astype(np.float32)  # Load PSF model for channel k
                if channeltype == '4pi':
                    psfobj.initA[k] = np.array(f['res']['channel'+str(k)]['A_model']).astype(np.complex64)  # Load A model for channel k

    # Setting up optimizer for fitting
    optimizer = L_BFGS_B(maxiter=maxiter)  # Create L-BFGS-B optimizer
    optimizer.batch_size = batchsize  # Set batch size for the optimizer
    # Selecting fitter based on whether multiple loss functions are used
    if self.loss_fun_multi:
        fitter = Fitter(dataobj, psfobj, optimizer, self.loss_fun_multi, loss_func_single=self.loss_fun, loss_weight=w)
    else:
        fitter = Fitter(dataobj, psfobj, optimizer, self.loss_fun, loss_weight=w)

    # Retrieving image data from the data object
    _, _, centers, file_idxs = dataobj.get_image_data()  # Get centers and file indices
    centers = np.stack(centers)  # Stack centers for processing
    res, toc = fitter.learn_psf(start_time=time)  # Start learning the PSF

    pos = res[-1][0]  # Extracting position data from results
    zpos = pos[:, 0:1]  # Extracting Z positions
    zpos = zpos - np.mean(zpos)  # Centering Z positions

    # Adjusting centers for voxel PSF type if conditions are met
    if (centers.shape[-1] == 3) & (np.max(np.abs(zpos)) > 2) & (PSFtype == 'voxel'):
        cor = dataobj.centers  # Original centers

        # Adjusting centers based on skew if skew constants are present
        if dataobj.skew_const:
            sk = dataobj.skew_const
            centers1 = np.int32(np.round(np.hstack((cor[:, 0:1] - zpos, cor[:, 1:2] - sk[0] * zpos, cor[:, 2:] - sk[1] * zpos))))
        else:
            centers1 = np.int32(np.round(np.hstack((cor[:, 0:1] - zpos, cor[:, 1:2], cor[:, 2:]))))
        
        # Updating data object with new ROIs based on adjusted centers
        dataobj.cut_new_rois(centers1, file_idxs, roi_size=roi_size)
        offset = np.min((np.quantile(dataobj.rois, 1e-3), 0))  # Adjusting offset
        dataobj.rois = dataobj.rois - offset  # Centering ROIs
        if dataobj.skew_const:
            dataobj.deskew_roi(roi_size)  # Deskewing ROIs if necessary

        fitter.dataobj = dataobj  # Updating fitter with the modified data object
        res, toc = fitter.learn_psf(start_time=time)  # Re-learning PSF with updated data

    # Localizing results based on number of file indices
    if len(file_idxs) == 1:
        locres = fitter.localize(res, channeltype, usecuda=usecuda, plot=showplot, start_time=toc)  # Localize for single file
        res1 = res  # Storing results
    else:
        # Handling outlier removal and localization for multiple files
        if 'insitu' in PSFtype:
            res1, toc = fitter.relearn_smlm(res, channeltype, rej_threshold, start_time=toc)  # Re-learning for SMLM
            locres = fitter.localize_smlm(res1, channeltype, plot=showplot)  # Localize SMLM results
        else:
            locres = fitter.localize(res, channeltype, usecuda=usecuda, plot=showplot, start_time=toc)  # Localize results
            toc = locres[-2]  # Updating toc with localization time
            res1, toc = fitter.relearn(res, channeltype, rej_threshold, start_time=toc)  # Re-learning process
            if res1[0].shape[-2] < res[0].shape[-2]:
                locres = fitter.localize(res1, channeltype, usecuda=usecuda, plot=showplot, start_time=toc)  # Re-localize if necessary

    self.learning_result = res1  # Storing learning results
    self.loc_result = locres  # Storing localization results
    return psfobj, fitter  # Returning PSF object and fitter instance

def localize_FD(self, fitter, initz=None):
    """
    Localizes the results using the Fitter object for FD (Fourier Deconvolution).
    
    Parameters:
    - fitter: An instance of the Fitter class used for localization.
    - initz: Optional; initial Z position for localization.
    
    Returns:
    - loc_FD: Localization results based on Fourier Deconvolution.
    """
    res = self.learning_result  # Getting the learning result
    usecuda = self.param.usecuda  # Fetching CUDA usage parameter
    showplot = self.param.plotall  # Fetching plot display parameter
    channeltype = self.param.channeltype  # Fetching channel type
    # Performing localization using the fitter's method
    loc_FD = fitter.localize_FD(res, channeltype, usecuda=usecuda, initz=initz, plot=showplot)
    self.loc_FD = loc_FD  # Storing localization results
    return loc_FD  # Returning localization results

def iterlearn_psf(self, dataobj, time=None):
    """
    Iteratively learns the PSF by repeating the learning process a specified number of times.
    
    Parameters:
    - dataobj: An object containing the data required for learning the PSF.
    - time: Optional; a specific time to consider during the learning process.
    
    Returns:
    - resfile: The filename where results are saved.
    """
    min_photon = self.param.option.insitu.min_photon  # Minimum photon count for in situ processing
    iterN = self.param.option.insitu.repeat  # Number of iterations for learning
    pz = self.param.pixel_size.z  # Pixel size in the Z direction
    channeltype = self.param.channeltype  # Type of channel
    savename = self.param.savename  # Base name for saving results
    
    # Iterating through the specified number of learning iterations
    for nn in range(0, iterN):
        if nn > 0:
            dataobj.resetdata()  # Resetting data for the next iteration
        psfobj, fitter = self.learn_psf(dataobj, time=time)  # Learning the PSF
        self.param.savename = savename + str(nn)  # Updating save name for the current iteration
        resfile = self.save_result(psfobj, dataobj, fitter)  # Saving results
        self.param.option.model.init_pupil_file = resfile  # Updating pupil file option
        self.param.option.insitu.min_photon = max([min_photon - nn * 0.1, 0.2])  # Adjusting minimum photon count
        
        res = psfobj.res2dict(self.learning_result)  # Converting results to dictionary format
        
        # Plotting results depending on channel type
        if channeltype == 'single':
            self.param.option.insitu.stage_pos = float(res['stagepos'])  # Updating stage position for single channel
            I_model = res['I_model']  # Getting the PSF model
            Nz = I_model.shape[-3]  # Number of Z slices
            zind = range(0, Nz, 4)  # Indices for Z slices to plot
            # Plotting the PSF model slices
            if self.param.plotall:
                fig = plt.figure(figsize=[3 * len(zind), 3])
                for i, id in enumerate(zind):
                    ax = fig.add_subplot(1, len(zind), i + 1)
                    plt.imshow(I_model[id], cmap='twilight')  # Displaying the image slice using a colormap
                    plt.axis('off')  # Hiding the axis
                plt.show()  # Displaying the plot
        else:
            # Handling multi-channel case
            try:
                self.param.option.insitu.stage_pos = float(res['channel0']['stagepos'])  # Updating stage position for channel 0
            except:
                pass
            # Plotting results for each channel
            if self.param.plotall:
                for j in range(0, len(dataobj.channels)):
                    if channeltype == '4pi':
                        I_model = res['channel' + str(j)]['psf_model']  # Getting PSF model for 4pi channels
                    else:
                        I_model = res['channel' + str(j)]['I_model']  # Getting PSF model for other channels
                    Nz = I_model.shape[-3]  # Number of Z slices
                    zind = range(0, Nz, 4)  # Indices for Z slices
                    # Plotting the PSF model slices
                    fig = plt.figure(figsize=[3 * len(zind), 3])
                    for i, id in enumerate(zind):
                        ax = fig.add_subplot(1, len(zind), i + 1)
                        plt.imshow(I_model[id], cmap='twilight')  # Displaying the image slice using a colormap
                        plt.axis('off')  # Hiding the axis
                    plt.show()  # Displaying the plot

    return resfile  # Returning the filename of the saved results

def save_result(self, psfobj, dataobj, fitter):
    """
    Saves the results of the PSF learning process to a file.
    
    Parameters:
    - psfobj: The object containing the learned PSF data.
    - dataobj: The object containing the data used for learning the PSF.
    - fitter: The Fitter instance used for fitting the PSF.
    
    Returns:
    - resfile: The filename where results are saved.
    """
    param = self.param  # Accessing parameters from the class
    res = self.learning_result  # Getting learning results
    locres = self.loc_result  # Getting localization results
    toc = locres[-2]  # Time of computation
    # Progress bar for saving results
    pbar = tqdm(desc='6/6: saving results', bar_format="{desc}: [{elapsed}s] {postfix[0]}{postfix[1][time]:>4.2f}s", postfix=["total time: ", dict(time=toc)])
    
    folder = param.datapath  # Data path for saving results
    savename = param.savename + '_' + param.PSFtype + '_' + param.channeltype  # Constructing save name
    res_dict = psfobj.res2dict(res)  # Converting results to dictionary format
    coeff_reverse = self.gencspline(res_dict, psfobj, keyname='I_model_reverse')  # Generating spline coefficients for reverse model
    coeff = self.gencspline(res_dict, psfobj)  # Generating spline coefficients for the main model

    # Preparing localization results dictionary
    if self.loc_FD is not None:
        locres_dict = dict(P=locres[0], CRLB=locres[1], LL=locres[2], coeff=coeff, coeff_bead=locres[3], loc=locres[-1], loc_FD=self.loc_FD, coeff_reverse=coeff_reverse)
    else:
        locres_dict = dict(P=locres[0], CRLB=locres[1], LL=locres[2], coeff=coeff, coeff_bead=locres[3], loc=locres[-1], coeff_reverse=coeff_reverse)

    # Retrieving image data to save with results
    img, _, centers, file_idxs = dataobj.get_image_data()  # Getting image data
    img = np.stack(img)  # Stacking image data
    rois_dict = dict(cor=np.stack(centers), fileID=np.stack(file_idxs), psf_data=fitter.rois, psf_fit=fitter.forward_images, image_size=img.shape)  # ROIs and image information
    resfile = savename + '.h5'  # Final filename for the saved results
    self.writeh5file(resfile, res_dict, locres_dict, rois_dict)  # Writing all data to HDF5 file

    self.result_file = resfile  # Storing the result file name
    pbar.postfix[1]['time'] = toc + pbar._time() - pbar.start_t  # Updating total time elapsed
    pbar.update()  # Updating progress bar
    pbar.close()  # Closing progress bar
    return resfile  # Returning the filename of the saved results

def writeh5file(self, filename, res_dict, locres_dict, rois_dict):
    """
    Writes the results and localization data to an HDF5 file.
    
    Parameters:
    - filename: The name of the file to save the results.
    - res_dict: A dictionary containing results data.
    - locres_dict: A dictionary containing localization results.
    - rois_dict: A dictionary containing regions of interest data.
    """
    with h5.File(filename, "w") as f:  # Creating the HDF5 file
        f.attrs["params"] = json.dumps(OmegaConf.to_container(self.param))  # Saving parameters as attributes
        g3 = f.create_group("rois")  # Creating group for ROIs
        g1 = f.create_group("res")  # Creating group for results
        g2 = f.create_group("locres")  # Creating group for localization results

        # Saving localization results to HDF5 file
        for k, v in locres_dict.items():
            if isinstance(v, dict):
                gi = g2.create_group(k)  # Creating a subgroup for each key
                for ki, vi in v.items():
                    gi[ki] = vi  # Saving each item in the subgroup
            else:
                g2[k] = v  # Saving item directly in the group
        
        # Saving results to HDF5 file
        for k, v in res_dict.items():
            if isinstance(v, dict):
                gi = g1.create_group(k)  # Creating a subgroup for each key
                for ki, vi in v.items():
                    gi[ki] = vi  # Saving each item in the subgroup
            else:
                g1[k] = v  # Saving item directly in the group
        
        # Saving ROIs to HDF5 file
        for k, v in rois_dict.items():
            g3[k] = v  # Saving each item in the ROIs group
        
    return  # End of function, file is automatically closed


    def gencspline(self, res_dict, psfobj, keyname='I_model'):
    """
    Generates a cubic spline representation of the intensity model based on the provided parameters.

    Parameters:
    - res_dict: dict
        A dictionary containing the results from the model, which should include the intensity model under the specified keyname.
    - psfobj: object
        An object that contains the PSF (Point Spread Function) related methods and properties.
    - keyname: str, optional
        The key name under which the intensity model is stored in res_dict (default is 'I_model').

    Returns:
    - coeff: numpy.ndarray
        The coefficients of the cubic spline representation of the intensity model.
    """
    param = self.param
    coeff = []  # Initialize an empty list to hold coefficients

    # Handle the case for single channel
    if param.channeltype == 'single':
        if keyname in res_dict:
            I_model = res_dict[keyname]  # Retrieve the intensity model
            offset = np.min(I_model)  # Calculate the minimum value for offsetting
            Imd = I_model - offset  # Subtract offset from I_model
            normf = np.median(np.sum(Imd, axis=(-1, -2)))  # Normalize factor
            Imd = Imd / normf  # Normalize the intensity model
            coeff = psf2cspline_np(Imd)  # Convert to cubic spline coefficients
            coeff = coeff.astype(np.float32)  # Ensure the coefficients are in float32 format

    # Handle the case for multi-channel
    if param.channeltype == 'multi':
        if keyname in res_dict['channel0']:
            Nchannel = len(psfobj.sub_psfs)  # Number of channels
            I_model = []
            for i in range(Nchannel):
                I_model.append(res_dict['channel' + str(i)][keyname])  # Collect intensity models for each channel
            I_model = np.stack(I_model)  # Stack them into a 3D array
            offset = np.min(I_model)  # Calculate the minimum value for offsetting
            Iall = []
            Imd = I_model - offset  # Subtract offset
            normf = np.max(np.median(np.sum(Imd, axis=(-1, -2)), axis=-1))  # Determine normalization factor
            Imd = Imd / normf  # Normalize intensity model
            for i in range(Nchannel):
                coeff = psf2cspline_np(Imd[i])  # Convert each channel's model to cubic spline coefficients
                Iall.append(coeff)  # Append coefficients to list
            coeff = np.stack(Iall).astype(np.float32)  # Stack coefficients and convert to float32

    # Handle the case for 4pi channel type
    if param.channeltype == '4pi':
        if keyname in res_dict['channel0']:
            Nchannel = len(psfobj.sub_psfs)  # Number of channels
            I_model = []
            A_model = []
            for i in range(Nchannel):
                I_model.append(res_dict['channel' + str(i)][keyname])  # Collect intensity models
                if keyname == 'I_model':
                    A_model.append(res_dict['channel' + str(i)]['A_model'])  # Collect amplitude models
                else:
                    A_model.append(res_dict['channel' + str(i)]['A_model_reverse'])  # Collect reverse amplitude models
            I_model = np.stack(I_model)  # Stack intensity models
            A_model = np.stack(A_model)  # Stack amplitude models
            offset = np.min(I_model - 2 * np.abs(A_model))  # Calculate offset
            Imd = I_model - offset  # Subtract offset
            normf = np.max(np.median(np.sum(Imd[:, 1:-1], axis=(-1, -2)), axis=-1)) * 2.0  # Determine normalization factor
            Imd = Imd / normf  # Normalize intensity model
            Amd = A_model / normf  # Normalize amplitude model
            IABall = []
            for i in range(Nchannel):
                Ii = Imd[i]  # Current intensity model
                Ai = 2 * np.real(Amd[i])  # Real part of normalized amplitude
                Bi = -2 * np.imag(Amd[i])  # Imaginary part of normalized amplitude
                IAB = [psf2cspline_np(Ai), psf2cspline_np(Bi), psf2cspline_np(Ii)]  # Generate cubic splines for Ai, Bi, and Ii
                IAB = np.stack(IAB)  # Stack all coefficients
                IABall.append(IAB)  # Append to the list
            coeff = np.stack(IABall).astype(np.float32)  # Stack and convert to float32

    return coeff  # Return the coefficients


def genpsf(self, f, Nz=21, xsz=21, stagepos=1.0):
    """
    Generates the Point Spread Function (PSF) based on the provided parameters and configurations.

    Parameters:
    - f: object
        An object that contains information related to the PSF generation such as resolution, pixel size, and regions of interest.
    - Nz: int, optional
        The number of axial points to generate in the PSF (default is 21).
    - xsz: int, optional
        The size of the PSF in the lateral dimensions (default is 21).
    - stagepos: float, optional
        The position of the stage in the axial direction (default is 1.0).

    Returns:
    - f: object
        The updated f object containing the generated PSF results.
    - psfobj: object
        The PSF object initialized with the computed PSF data.
    """
    p = self.param  # Retrieve parameters from the class instance
    # Create a data object with pixel sizes and image size for PSF calculations
    dataobj = DottedDict(pixelsize_x=p.pixel_size.x,
                         pixelsize_y=p.pixel_size.y,
                         pixelsize_z=p.pixel_size.z,
                         image_size=list(f.rois.image_size),
                         rois=np.zeros((Nz, xsz, xsz)))  # Initialize ROIs to zeros

    self.getpsfclass()  # Retrieve PSF class
    psfobj = self.initializepsf()  # Initialize the PSF object

    # Handle case for single-channel PSF generation
    if p.channeltype == 'single':
        sigma = f.res.sigma  # Retrieve sigma from resolution
        Zcoeff = f.res.zernike_coeff  # Retrieve Zernike coefficients
        Zcoeff = Zcoeff.reshape((Zcoeff.shape + (1, 1)))  # Reshape coefficients for processing
        psfobj.data = dataobj  # Assign the data object to the PSF object

        # Check for 'insitu' type to estimate z-offset
        if 'insitu' in p.PSFtype:
            psfobj.stagepos = stagepos / p.pixel_size.z  # Set the stage position
            psfobj.estzoffset(Nz=Nz)  # Estimate z-offset based on the number of axial points
        else:
            # Calculate pupil field based on PSF type
            if psfobj.psftype == 'vector':
                psfobj.calpupilfield('vector', Nz=Nz)  # Calculate pupil field for vector type
            else:
                psfobj.calpupilfield('scalar', Nz=Nz)  # Calculate pupil field for scalar type

        # Check for 'FD' type in PSF generation
        if 'FD' in p.PSFtype:
            # Calculate the ranges for the x and y dimensions
            dx = f.rois.image_size[-1] / f.res.zernike_map.shape[-1] / 2
            dy = f.rois.image_size[-2] / f.res.zernike_map.shape[-2] / 2
            xrange = np.linspace(dx, f.rois.image_size[-1] - dx, f.res.zernike_map.shape[-1], dtype=np.float32)
            yrange = np.linspace(dy, f.rois.image_size[-2] - dy, f.res.zernike_map.shape[-2], dtype=np.float32)
            [xx, yy] = np.meshgrid(xrange, yrange)  # Create a meshgrid for coordinates
            cor = np.vstack((xx.flatten(), yy.flatten())).transpose()  # Flatten and transpose for coordinates

            Zmap = f.res.zernike_map  # Retrieve Zernike map
            batchsize = 200  # Define batch size for processing
            ind = list(np.int32(np.linspace(0, cor.shape[0], cor.shape[0] // batchsize + 2)))  # Generate index ranges for batching
            for i in range(len(ind) - 1):
                I0, _, _ = psfobj.genpsfmodel(sigma, Zmap=Zmap, cor=cor[ind[i]:ind[i + 1]])  # Generate PSF model
                if i == 0:
                    I_model = I0  # Initialize the I_model with the first batch
                else:
                    I_model = np.vstack((I_model, I0))  # Stack subsequent batches to I_model
                # I_model, _, _ = psfobj.genpsfmodel(sigma, Zmap, cor)  # Alternative single call (commented out)
        else:
            I_model, _ = psfobj.genpsfmodel(sigma, Zcoeff)  # Generate PSF using Zernike coefficients
        f.res.I_model = I_model  # Store the generated I_model in the results object

    # Handle the case for multi-channel PSF generation
    elif p.channeltype == 'multi':
        Nchannel = f.rois.cor.shape[0]  # Number of channels
        psfobj.sub_psfs = [None] * Nchannel  # Initialize sub-PSFs for each channel
        for i in range(Nchannel):
            psf = psfobj.psftype(options=psfobj.options)  # Create a new PSF for each channel
            psf.psftype = psfobj.PSFtype  # Assign PSF type to the new instance
            psfobj.sub_psfs[i] = psf  # Store the new PSF object in the list
            sigma = f.res['channel' + str(i)].sigma  # Retrieve the sigma for the current channel
            Zcoeff = f.res['channel' + str(i)].zernike_coeff  # Retrieve Zernike coefficients for current channel
            Zcoeff = Zcoeff.reshape((Zcoeff.shape + (1, 1)))  # Reshape coefficients for processing
            psf.data = dataobj  # Assign the data object to the PSF object

            # Check for 'insitu' type to estimate z-offset for each channel
            if 'insitu' in p.PSFtype:
                psf.stagepos = stagepos / p.pixel_size.z  # Set stage position
                psf.estzoffset(Nz=Nz)  # Estimate z-offset for the PSF
            else:
                # Calculate pupil field based on PSF type
                if psf.psftype == 'vector':
                    psf.calpupilfield('vector', Nz=Nz)  # Vector type
                else:
                    psf.calpupilfield('scalar', Nz=Nz)  # Scalar type

            I_model, _ = psf.genpsfmodel(sigma, Zcoeff)  # Generate PSF model for the current channel
            f.res['channel' + str(i)].I_model = I_model  # Store channel-specific I_model results

    return f, psfobj  # Return the updated object and the PSF object


def calstrehlratio(self, f, xsz=31):
    """
    Calculates the Strehl ratio for the PSF, which is a measure of the quality of the imaging system.

    Parameters:
    - f: object
        An object containing the results of the PSF generation and relevant parameters.
    - xsz: int, optional
        The size of the image for which the Strehl ratio is calculated (default is 31).

    Returns:
    - strehlratio: numpy.ndarray
        The calculated Strehl ratio, indicating the quality of the imaging system.
    """
    f1 = f.copy()  # Create a copy of the original object to work on
    p = self.param  # Retrieve parameters from the class

    # Handle case for single channel
    if p.channeltype == 'single':
        if 'FD' in p.PSFtype:
            f1.res.zernike_map = f.res.zernike_map.copy()  # Copy Zernike map for calculations
            f1.res.zernike_map[1, 0:4] = 0.0  # Modify the Zernike coefficients for FD type
            f1, psfobj = self.genpsf(f1, Nz=1, xsz=xsz)  # Generate PSF for modified coefficients
            I_model = f1.res.I_model / np.sum(f1.res.I_model, axis=(-1, -2), keepdims=True)  # Normalize I_model
            I1 = I_model[:, 0, xsz // 2, xsz // 2]  # Get the central intensity for Strehl ratio calculation

            # Reset Zernike map to calculate the reference PSF
            f1.res.zernike_map = np.zeros(f1.res.zernike_map.shape, dtype=np.float32)
            f1.res.zernike_map[0, 0] = 1  # Set a single coefficient to create reference PSF
            f1, psfobj = self.genpsf(f1, Nz=1, xsz=xsz)  # Generate PSF for the reference
            I_model = f1.res.I_model / np.sum(f1.res.I_model, axis=(-1, -2), keepdims=True)  # Normalize reference I_model
            I0 = I_model[:, 0, xsz // 2, xsz // 2]  # Get the central intensity for reference
            strehlratio = np.float32(I1 / I0)  # Calculate the Strehl ratio
            strehlratio_map = np.reshape(strehlratio, (f.res.zernike_map.shape[-2], f.res.zernike_map.shape[-1]))  # Reshape for visualization
            plt.imshow(strehlratio_map)  # Display the Strehl ratio map
            plt.colorbar()  # Add colorbar for reference
            plt.title('Strehl ratio map', fontsize=15)  # Title for the plot
        else:
            f1.res.zernike_coeff[1, 0:4] = 0.0  # Modify Zernike coefficients for non-FD type
            f1, psfobj = self.genpsf(f1, Nz=1, xsz=xsz)  # Generate PSF with modified coefficients
            I_model = f1.res.I_model / np.sum(f1.res.I_model)  # Normalize I_model
            I1 = I_model[0, xsz // 2, xsz // 2]  # Get the central intensity for Strehl ratio calculation

            # Reset Zernike coefficients to create a reference PSF
            f1.res.zernike_coeff = np.zeros(f1.res.zernike_coeff.shape, dtype=np.float32)
            f1.res.zernike_coeff[0, 0] = 1  # Set a single coefficient for reference
            f1, psfobj = self.genpsf(f1, Nz=1, xsz=xsz)  # Generate PSF for the reference
            I_model = f1.res.I_model / np.sum(f1.res.I_model)  # Normalize reference I_model
            I0 = I_model[0, xsz // 2, xsz // 2]  # Get central intensity for reference
            strehlratio = np.float32(I1 / I0)  # Calculate the Strehl ratio
            print('Strehl ratio: ', strehlratio)  # Print the calculated ratio

    # Handle case for multi-channel
    elif p.channeltype == 'multi':
        Nchannel = f1.rois.cor.shape[0]  # Number of channels
        I1 = []  # List to hold central intensity for each channel
        I0 = []  # List to hold reference central intensity for each channel
        for i in range(Nchannel):
            f1.res['channel' + str(i)].zernike_coeff[1, 0:4] = 0.0  # Modify coefficients for each channel
        f1, psfobj = self.genpsf(f1, Nz=1, xsz=xsz)  # Generate PSF for all channels
        coeff = np.zeros(f1.res.channel0.zernike_coeff.shape, dtype=np.float32)  # Initialize coefficient array for reference
        coeff[0, 0] = 1  # Set a single coefficient for reference

        for i in range(Nchannel):
            I_model = f1.res['channel' + str(i)].I_model / np.sum(f1.res['channel' + str(i)].I_model)  # Normalize each channel's I_model
            I1.append(I_model[0, xsz // 2, xsz // 2])  # Store central intensity for each channel
            f1.res['channel' + str(i)].zernike_coeff = coeff  # Set coefficients to reference for each channel

        f1, psfobj = self.genpsf(f1, Nz=1, xsz=31)  # Generate PSF again for reference
        for i in range(Nchannel):
            I_model = f1.res['channel' + str(i)].I_model / np.sum(f1.res['channel' + str(i)].I_model)  # Normalize reference I_model for each channel
            I0.append(I_model[0, xsz // 2, xsz // 2])  # Store reference central intensity for each channel

        I1 = np.stack(I1)  # Stack central intensities
        I0 = np.stack(I0)  # Stack references
        strehlratio = np.float32(I1 / I0)  # Calculate the Strehl ratio for multi-channel case
        print('Strehl ratio: ', strehlratio)  # Print the calculated ratio

    # Handle case for 4pi channel type
    elif p.channeltype == '4pi':    
    # Determine the number of channels based on the shape of the correlation region (rois)
    Nchannel = f.rois.cor.shape[0]
    
    # Initialize an array to hold the modulation depths for each channel
    mdepth = np.zeros(Nchannel)
    
    # Loop through each channel to extract modulation depth values
    for i in range(0, Nchannel):
        # Access the modulation depth for the current channel from the results dictionary (res)
        # 'channel' + str(i) constructs the key for the current channel's modulation depth
        mdepth[i] = f.res['channel' + str(i)].modulation_depth
    
    # Print the rounded modulation depth values for better readability
    print('modulation depth: ', np.round(mdepth, 2))
    
    # Assign the modulation depths to the strehl ratio variable for further use
    strehlratio = mdepth

# Return the computed strehl ratio (which contains modulation depths for 4pi channels)
return strehlratio
    
    def calfwhm(self, f):
        """
        Calculate the Full Width at Half Maximum (FWHM) for a given dataset.

        This method processes the input object based on its channel type:
        - For 'single' channel types, it generates a PSF and calculates FWHM for the provided data.
        - For 'multi' channel types, it processes each channel independently.

        Parameters:
        - self: The instance of the class where this method is defined.
        - f: An object containing relevant data (including PSF and intensity model) for FWHM calculation.
        
        Returns:
        - fwhmx: FWHM in the x-direction (numpy array).
        - fwhmy: FWHM in the y-direction (numpy array).
        - fwhmz: FWHM in the z-direction (numpy array).
        """
        p = self.param  # Access parameters from the class
        f1 = f.copy()  # Create a copy of the input object f
        
        # Check the type of channel
        if p.channeltype == 'single':
            # Handle single channel case
            if 'FD' in p.PSFtype:
                psfsize = f.res.I_model_bead.shape  # Get PSF size
                f1.res.zernike_map = f.res.zernike_map.copy()  # Copy Zernike map
                f1.res.zernike_map[1, 0:4] = 0.0  # Zero out specific entries of Zernike map
                
                # Generate PSF object
                f1, psfobj = self.genpsf(f1, Nz=psfsize[-3], xsz=psfsize[-1])
                I_model = f1.res.I_model  # Get the intensity model
                fwhmx = np.zeros(I_model.shape[0])  # Initialize arrays for FWHM values
                fwhmy = np.zeros(I_model.shape[0])
                fwhmz = np.zeros(I_model.shape[0])
                
                # Iterate through each model to compute FWHM
                for i, psfi in enumerate(I_model):
                    Ix, xh, Iy, yh, Iz, zh = self.getfwhm(psfi)  # Get FWHM for the current model
                    fwhmx[i] = np.diff(xh) * p.pixel_size.x * 1e3  # Calculate FWHM in x
                    fwhmy[i] = np.diff(yh) * p.pixel_size.y * 1e3  # Calculate FWHM in y
                    fwhmz[i] = np.diff(zh) * p.pixel_size.z * 1e3  # Calculate FWHM in z

                # Create a figure to visualize the FWHM maps
                fig = plt.figure(figsize=[12, 5])
                fwhmx_map = np.reshape(fwhmx, (f.res.zernike_map.shape[-2], f.res.zernike_map.shape[-1]))  # Reshape arrays for plotting
                fwhmy_map = np.reshape(fwhmy, (f.res.zernike_map.shape[-2], f.res.zernike_map.shape[-1]))
                fwhmz_map = np.reshape(fwhmz, (f.res.zernike_map.shape[-2], f.res.zernike_map.shape[-1]))
                
                # Plot FWHM xy map
                ax = fig.add_subplot(121)
                plt.imshow((fwhmx_map + fwhmy_map) / 2)  # Average FWHM in x and y
                clb = plt.colorbar()  # Add color bar
                clb.ax.set_title('nm')  # Title for color bar
                plt.title('FWHMxy map', fontsize=15)  # Title for the plot
                ax = fig.add_subplot(122)
                plt.imshow(fwhmz_map)  # FWHMz map
                clb = plt.colorbar()  # Add color bar
                clb.ax.set_title('nm')  # Title for color bar
                plt.title('FWHMz map', fontsize=15)  # Title for the plot
                fwhmx = fwhmx_map  # Update fwhmx to be the reshaped map
                fwhmy = fwhmy_map  # Update fwhmy to be the reshaped map
                fwhmz = fwhmz_map  # Update fwhmz to be the reshaped map

            else:
                # Handle the case where PSF type does not contain 'FD'
                I_model = f.res.I_model  # Get intensity model
                Imaxh = np.max(I_model) / 2  # Compute half maximum intensity
                Ix, xh, Iy, yh, Iz, zh = self.getfwhm(I_model)  # Get FWHM values
                fwhmx = np.diff(xh) * p.pixel_size.x * 1e3  # Calculate FWHM in x
                fwhmy = np.diff(yh) * p.pixel_size.y * 1e3  # Calculate FWHM in y
                fwhmz = np.diff(zh) * p.pixel_size.z * 1e3  # Calculate FWHM in z
                
                # Create a figure for FWHM plots
                fig = plt.figure(figsize=[12, 4])
                ax = fig.add_subplot(121)
                plt.plot(Ix, 'o-')  # Plot intensity in x
                plt.plot(xh, [Imaxh, Imaxh], '-')  # Plot half maximum line
                plt.plot(Iy, 'o-')  # Plot intensity in y
                plt.plot(yh, [Imaxh, Imaxh], '-')  # Plot half maximum line
                plt.title('FWHMxy: ' + str(np.round((fwhmx[0] + fwhmy[0]) / 2, 2)) + ' nm', fontsize=15)  # Title for the xy FWHM plot
                plt.xlabel('x (pixel)')  # Label for x-axis
                plt.ylabel('intensity')  # Label for y-axis

                ax = fig.add_subplot(122)
                plt.plot(Iz, 'o-')  # Plot intensity in z
                plt.plot(zh, [Imaxh, Imaxh], '-')  # Plot half maximum line
                plt.title('FWHMz: ' + str(np.round(fwhmz[0], 2)) + ' nm', fontsize=15)  # Title for the z FWHM plot
                plt.xlabel('z (pixel)')  # Label for x-axis
                plt.ylabel('intensity')  # Label for y-axis
                
        elif p.channeltype == 'multi':
            # Handle multi channel case
            Nchannel = f.rois.cor.shape[0]  # Get number of channels
            fig = plt.figure(figsize=[4 * Nchannel, 8])  # Create figure with appropriate size
            spec = gridspec.GridSpec(ncols=Nchannel, nrows=2,
                                     width_ratios=list(np.ones(Nchannel)), wspace=0.4,
                                     hspace=0.3, height_ratios=[1, 1])  # Define grid specification for subplots

            fwhmx = []  # Initialize list for FWHM in x across channels
            fwhmy = []  # Initialize list for FWHM in y across channels
            fwhmz = []  # Initialize list for FWHM in z across channels
            
            # Loop through each channel to compute FWHM values
            for i in range(0, Nchannel):
                I_model = f.res['channel' + str(i)].I_model  # Get intensity model for the current channel
                Imaxh = np.max(I_model) / 2  # Compute half maximum intensity
                Ix, xh, Iy, yh, Iz, zh = self.getfwhm(I_model)  # Get FWHM values
                fwhmxi = np.diff(xh) * p.pixel_size.x * 1e3  # Calculate FWHM in x for this channel
                fwhmyi = np.diff(yh) * p.pixel_size.y * 1e3  # Calculate FWHM in y for this channel
                fwhmzi = np.diff(zh) * p.pixel_size.z * 1e3  # Calculate FWHM in z for this channel
                
                # Plot FWHM for the xy dimensions
                ax = fig.add_subplot(spec[i])
                plt.plot(Ix, 'o-')  # Plot intensity in x
                plt.plot(xh, [Imaxh, Imaxh], '-')  # Plot half maximum line
                plt.plot(Iy, 'o-')  # Plot intensity in y
                plt.plot(yh, [Imaxh, Imaxh], '-')  # Plot half maximum line
                plt.title('ch' + str(i) + ' FWHMxy: ' + str(np.round((fwhmxi[0] + fwhmyi[0]) / 2, 2)) + ' nm', fontsize=15)  # Title for the xy FWHM plot of the channel
                plt.xlabel('x (pixel)')  # Label for x-axis
                plt.ylabel('intensity')  # Label for y-axis

                # Plot FWHM for the z dimension
                ax = fig.add_subplot(spec[Nchannel + i])
                plt.plot(Iz, 'o-')  # Plot intensity in z
                plt.plot(zh, [Imaxh, Imaxh], '-')  # Plot half maximum line
                plt.title('ch' + str(i) + ' FWHMz: ' + str(np.round(fwhmzi[0], 2)) + ' nm', fontsize=15)  # Title for the z FWHM plot of the channel
                plt.xlabel('z (pixel)')  # Label for x-axis
                plt.ylabel('intensity')  # Label for y-axis
                
                fwhmx.append(fwhmxi)  # Store FWHM in x for this channel
                fwhmz.append(fwhmzi)  # Store FWHM in z for this channel
            
            fwhmx = np.stack(fwhmx)  # Stack collected FWHM values into an array
            fwhmz = np.stack(fwhmz)  # Stack collected FWHM values into an array

        plt.show()  # Display the plots
        return fwhmx, fwhmy, fwhmz  # Return the calculated FWHM values for x, y, and z

def getfwhm(self, I_model):
        """
        Calculate the Full Width at Half Maximum (FWHM) for the provided intensity model.

        This method identifies the coordinates of the maximum intensity and computes the FWHM
        in the x, y, and z dimensions.

        Parameters:
        - self: The instance of the class where this method is defined.
        - I_model: A numpy array representing the intensity model from which FWHM will be calculated.

        Returns:
        - Ix: Intensity profile in the x-direction.
        - xh: Positions corresponding to the x-intensity profile at FWHM.
        - Iy: Intensity profile in the y-direction.
        - yh: Positions corresponding to the y-intensity profile at FWHM.
        - Iz: Intensity profile in the z-direction.
        - zh: Positions corresponding to the z-intensity profile at FWHM.
        """
        cor = np.unravel_index(np.argmax(I_model), I_model.shape)  # Find index of maximum intensity
        
        # Lateral profiles for FWHM calculation
        Ix = I_model[cor[0], cor[1]]  # Extract x intensity profile
        xh = self.get1dfwhm(Ix, cor[2])  # Calculate FWHM in x dimension

        Iy = I_model[cor[0], :, cor[2]]  # Extract y intensity profile
        yh = self.get1dfwhm(Iy, cor[1])  # Calculate FWHM in y dimension

        # Axial profile for FWHM calculation
        Iz = I_model[:, cor[1], cor[2]]  # Extract z intensity profile
        zh = self.get1dfwhm(Iz, cor[0])  # Calculate FWHM in z dimension

        return Ix, xh, Iy, yh, Iz, zh  # Return profiles for further analysis
    
def get1dfwhm(self, I, cor):
        """
        Calculate the FWHM for a one-dimensional intensity profile.

        This method identifies the points around the half maximum intensity to compute the FWHM.

        Parameters:
        - self: The instance of the class where this method is defined.
        - I: A numpy array representing the intensity profile (1D) from which FWHM will be calculated.
        - cor: The index of the maximum intensity in the profile.

        Returns:
        - A numpy array containing the positions corresponding to the FWHM in the profile.
        """
        Imaxh = np.max(I) / 2  # Compute the half maximum intensity

        # Find the left edge of the FWHM
        x1 = np.argsort(np.abs(I[:cor] - Imaxh))[0]  # Get index close to half maximum on the left side
        if I[x1] > Imaxh:  # Adjust for the edge case
            x1 = [x1, x1 - 1]
        else:
            x1 = [x1, x1 + 1]

        # Find the right edge of the FWHM
        x2 = np.argsort(np.abs(I[cor:] - Imaxh))[0] + cor  # Get index close to half maximum on the right side
        if I[x2] > Imaxh:  # Adjust for the edge case
            x2 = [x2, x2 + 1]
        else:
            x2 = [x2, x2 - 1]

        # Linear interpolation to find precise locations of the FWHM
        g = np.diff(x1) / np.diff(I[x1])  # Gradient for left side
        xh1 = g * (Imaxh - I[x1[0]]) + x1[0]  # Interpolated position for left edge
        x1 = np.array(x1, dtype=np.float64)  # Ensure x1 is a numpy array for calculations
        xh1 = np.minimum(np.maximum(xh1, np.min(x1)), np.max(x1))  # Clamp value to valid range

        g = np.diff(x2) / np.diff(I[x2])  # Gradient for right side
        xh2 = g * (Imaxh - I[x2[0]]) + x2[0]  # Interpolated position for right edge
        x2 = np.array(x2, dtype=np.float64)  # Ensure x2 is a numpy array for calculations
        xh2 = np.minimum(np.maximum(xh2, np.min(x2)), np.max(x2))  # Clamp value to valid range
        
        return np.hstack([xh1, xh2])  # Return combined positions for the FWHM