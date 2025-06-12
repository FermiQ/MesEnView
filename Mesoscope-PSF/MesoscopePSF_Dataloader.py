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

# Import necessary libraries for file handling, image processing, and numerical operations
from pickle import FALSE  # Importing FALSE from pickle module, though it is not used in this code
import h5py as h5  # Importing h5py for reading HDF5 files
import czifile as czi  # Importing czifile for reading CZI files
import numpy as np  # Importing NumPy for numerical operations
from skimage import io  # Importing skimage's io for image reading
import glob  # Importing glob for file name pattern matching
import json  # Importing json for handling JSON data, although it's not used in this code
from PIL import Image  # Importing PIL's Image for image file handling

class dataloader:
    """
    A class to handle the loading of various image file formats and processing them based on given parameters.
    """

    def __init__(self, param=None):
        """
        Initializes the dataloader with specified parameters.

        :param param: Object containing configuration settings for loading data.
        """
        self.param = param

    def getfilelist(self):
        """
        Generates a list of files based on the specified parameters.

        :return: A list of file paths that match the criteria set in the parameters.
        """
        param = self.param
        # If no subfolder is specified, match files in the main data path
        if not param.subfolder:
            filelist = glob.glob(param.datapath + '/*' + param.keyword + '*' + param.format)
        else:
            filelist = []
            # If a subfolder is specified, match files within those subfolders
            folderlist = glob.glob(param.datapath + '/*' + param.subfolder + '*/')
            for f in folderlist:
                filelist.append(glob.glob(f + '/*' + param.keyword + '*' + param.format)[0])
        
        return filelist

    def loadtiff(self, filelist):
        """
        Loads TIFF image files from the provided list.

        :param filelist: A list of file paths for TIFF images to load.
        :return: A NumPy array containing the loaded images.
        """
        param = self.param
        imageraw = []
        for filename in filelist:
            print(filename)  # Print the filename being processed
            if param.datatype == 'smlm':  # Check if the data type is 'smlm'
                dat = []
                fID = Image.open(filename)  # Open the TIFF file

                # Read specified frames from the TIFF file
                for ii in range(param.insitu.frame_range[0], param.insitu.frame_range[1]):
                    fID.seek(ii)  # Move to the specified frame
                    dat.append(np.asarray(fID))  # Append the frame as an array
                dat = np.stack(dat).astype(np.float32)  # Stack frames into a single array
            else:
                # Load the TIFF file using skimage and convert to float32
                dat = np.squeeze(io.imread(filename).astype(np.float32))
            
            # If the channel type is multi, split the channels
            if param.channeltype == 'multi':
                dat = self.splitchannel(dat)

            # Adjust the data based on CCD offset and gain
            dat = (dat - param.ccd_offset) * param.gain
            imageraw.append(dat)  # Append the processed data to the image list
            
        imagesall = np.stack(imageraw)  # Stack all images into a single array

        return imagesall

    def loadmat(self, filelist):
        """
        Loads MAT files from the provided list.

        :param filelist: A list of file paths for MAT files to load.
        :return: A NumPy array containing the loaded images from the MAT files.
        """
        param = self.param
        imageraw = []
        for filename in filelist:
            print(filename)  # Print the filename being processed
            fdata = h5.File(filename, 'r')  # Open the MAT file
            if param.varname:
                name = [param.varname]  # Use the specified variable name if provided
            else:
                name = list(fdata.keys())  # Get all variable names in the file
                
            # Remove metadata and reference keys if they exist
            try:
                name.remove('metadata')
            except:
                pass
            try:
                name.remove('#refs#')
            except:
                pass

            # Process the data based on the specified channel type
            if param.channeltype == 'single':
                dat = np.squeeze(np.array(fdata.get(name[0])).astype(np.float32))
            else:
                if len(name) > 1:
                    dat = []
                    for ch in name:            
                        datai = np.squeeze(np.array(fdata.get(ch)).astype(np.float32))
                        dat.append(datai)  # Append each channel's data
                    dat = np.squeeze(np.stack(dat))  # Stack channels into a single array
                else:
                    dat = np.squeeze(np.array(fdata.get(name[0])).astype(np.float32))
                    dat = self.splitchannel(dat)

            # Adjust the data based on CCD offset and gain
            dat = (dat - param.ccd_offset) * param.gain
            imageraw.append(dat)  # Append the processed data to the image list
        
        imagesall = np.stack(imageraw)  # Stack all images into a single array

        return imagesall
        
    def loadh5(self, filelist):
        """
        Loads HDF5 files from the provided list, currently designed for 'smlm' data.

        :param filelist: A list of file paths for HDF5 files to load.
        :return: A NumPy array containing the loaded images.
        """
        param = self.param
        imageraw = []

        for filename in filelist:
            f = h5.File(filename, 'r')  # Open the HDF5 file
            k = list(f.keys())  # Get the list of top-level keys
            gname = ''
            while len(k) == 1:  # Navigate through groups if there's only one key
                gname += k[0] + '/'
                k = list(f[gname].keys())
            datalist = list(f[gname].keys())  # Get the keys of the deeper level

            # Try to read the data from the appropriate key
            try:
                dat = np.squeeze(np.array(f.get(gname + datalist[0])).astype(np.float32))
            except:
                dat = np.squeeze(np.array(f.get(gname + datalist[0] + '/' + datalist[0])).astype(np.float32))
            
            # Crop the data based on frame range and adjust it
            dat = dat[param.insitu.frame_range[0]:param.insitu.frame_range[1]]
            dat = (dat - param.ccd_offset) * param.gain
            imageraw.append(dat)  # Append the processed data to the image list
        
        imagesall = np.stack(imageraw)  # Stack all images into a single array

        return imagesall

    def loadczi(self, filelist):
        """
        Loads CZI files from the provided list.

        :param filelist: A list of file paths for CZI files to load.
        :return: A NumPy array containing the loaded images.
        """
        param = self.param
        imageraw = []
        for filename in filelist:
            # Read the CZI file and convert the data to float32
            dat = np.squeeze(czi.imread(filename).astype(np.float32))
            # Adjust the data based on CCD offset and gain
            dat = (dat - param.ccd_offset) * param.gain
            imageraw.append(dat)  # Append the processed data to the image list
        
        imagesall = np.stack(imageraw)  # Stack all images into a single array

        return imagesall

    def splitchannel(self, dat):
        """
        Splits the channels of the data based on specified parameters.

        :param dat: A NumPy array containing image data to be split.
        :return: A NumPy array of the split channels based on the specified arrangement and size.
        """
        param = self.param
        if param.dual.channel_arrange:
            # Handle dual channel arrangement
            if param.dual.channel_arrange == 'up-down':
                cc = dat.shape[-2] // 2  # Calculate the channel center
                if param.dual.mirrortype == 'up-down':
                    # Stack the top half and the flipped bottom half
                    dat = np.stack([dat[:, :-cc], np.flip(dat[:, cc:], axis=-2)])
                elif param.dual.mirrortype == 'left-right':
                    # Stack the left half and the flipped right half
                    dat = np.stack([dat[:, :-cc], np.flip(dat[:, cc:], axis=-1)])
                else:
                    # Stack the top half and the bottom half without mirroring
                    dat = np.stack([dat[:, :-cc], dat[:, cc:]])
            else:
                cc = dat.shape[-1] // 2  # Calculate the channel center
                if param.dual.mirrortype == 'up-down':
                    # Stack the left half and the flipped right half
                    dat = np.stack([dat[..., :-cc], np.flip(dat[..., cc:], axis=-2)])
                elif param.dual.mirrortype == 'left-right':
                    # Stack the top half and the flipped bottom half
                    dat = np.stack([dat[..., :-cc], np.flip(dat[..., cc:], axis=-1)])  
                else:
                    # Stack the left half and the right half without mirroring
                    dat = np.stack([dat[..., :-cc], dat[..., cc:]])    

        # Handle multi-channel splitting if specified
        if param.multi.channel_size:
            roisz = param.multi.channel_size  # Get the channel size for splitting
            xdiv = list(range(0, dat.shape[-1], roisz[-1]))  # Calculate divisions along the width
            ydiv = list(range(0, dat.shape[-2], roisz[-2]))  # Calculate divisions along the height
            im = []
            for yd in ydiv[:-1]:  # Loop through height divisions
                for xd in xdiv[:-1]:  # Loop through width divisions
                    im.append(dat[..., yd:yd + roisz[-2], xd:xd + roisz[-1]])  # Append the split image sections

            dat = np.stack(im)  # Stack all split images into a single array

        return dat  # Return the final split data