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
Custom dataset class that handles the loading of measurement and ground truth images.

Attributes:
    dir_data (str): Directory where the data is located.
    Transform (callable, optional): Optional transform to be applied to the data.

Methods:
    __getitem__(self, index): Retrieves the item at the specified index, which consists of the ground truth and measurement images.
    __len__(self): Returns the total number of items in the dataset.

The code defines several classes and functions for image restoration tasks, including dataset loading, data augmentation, and neural network models. Here is a brief description of each class and function:

1. **Dataset**: This class loads images from a specified directory and applies transformations such as resizing, normalization, and noise addition to the images.

2. **Subset**: This class creates a subset of the dataset for validation or training purposes. It can apply random cropping to the images if not in validation mode.

3. **ToTensor**: This class converts numpy arrays in the dataset to PyTorch tensors.

4. **Noise**: This class applies random noise to the measurement images based on calibrated parameters.

5. **Resize**: This class resizes the measurement images by padding them with a specific size and predefined locations.

6. **ToTensorcm2**: This class converts numpy arrays in the CM2 dataset to PyTorch tensors.

7. **Noisecm2**: This class applies random noise to the measurement images in the CM2 dataset based on calibrated parameters.

8. **Crop**: This class crops the measurement, ground truth, and demixed images to a specified size at predefined locations.

These classes and functions are designed to create a pipeline for loading, preprocessing, and augmenting image data for restoration tasks.
"""


import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
import torch.nn
import skimage.io
import glob

class MyDataset(data.Dataset):
    """
    Custom dataset class that handles loading of measurement and ground truth images.

    Attributes:
    dir_data (str): Directory where the data is located.
    transform (callable, optional): Optional transform to be applied to the data.
    """

    def __init__(self, dir_data, transform=None):
        """
        Initializes the MyDataset instance.

        Parameters:
        dir_data (str): Directory where the data files are stored.
        transform (callable, optional): A function/transform to apply to the data.
        """
        self.dir_data = dir_data
        self.transform = transform

    def __getitem__(self, index):
        """
        Retrieves the item at the specified index, which consists of the ground truth 
        and measurement images.

        Parameters:
        index (int): Index of the item to retrieve.

        Returns:
        data (dict): Dictionary containing the ground truth and measurement images.
        """
        # Read measurement image
        meas = skimage.io.imread(self.dir_data + '/meas_{:n}.tif'.format(index + 1))
        # Read ground truth image
        gt = skimage.io.imread(self.dir_data + '/gt_{:n}.tif'.format(index + 1))
        # Crop the measurement image
        meas = meas[57 * 2:3000, 94 * 2 + 156:4000 - 156]
        # Normalize the images and prepare data dictionary
        data = {'gt': gt.astype('float32') / gt.max(), 
                'meas': meas.astype('float32') / meas.max()}
        # Apply transformations if any
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
        int: The number of measurement files in the directory.
        """
        return len(glob.glob(self.dir_data + '/meas_*.tif'))


class CM2Dataset(data.Dataset):
    """
    Custom dataset class for loading measurement, ground truth, and demixed images.

    Attributes:
    dir_data (str): Directory where the data is located.
    transform (callable, optional): Optional transform to be applied to the data.
    """

    def __init__(self, dir_data, transform=None):
        """
        Initializes the CM2Dataset instance.

        Parameters:
        dir_data (str): Directory where the data files are stored.
        transform (callable, optional): A function/transform to apply to the data.
        """
        self.dir_data = dir_data
        self.transform = transform

    def __getitem__(self, index):
        """
        Retrieves the item at the specified index, which consists of the ground truth, 
        measurement, and demixed images.

        Parameters:
        index (int): Index of the item to retrieve.

        Returns:
        data (dict): Dictionary containing the ground truth, measurement, and demixed images.
        """
        # Read measurement image
        meas = skimage.io.imread(self.dir_data + '/meas_{:n}.tif'.format(index + 1))
        # Read ground truth image
        gt = skimage.io.imread(self.dir_data + '/gt_{:n}.tif'.format(index + 1))
        # Read demixed image
        demix = skimage.io.imread(self.dir_data + '/demix_{:n}.tif'.format(index + 1))
        
        # Normalize images and prepare data dictionary
        data = {'gt': gt.astype('float32') / gt.max(), 
                'meas': meas.astype('float32') / meas.max(), 
                'demix': demix.astype('float32') / demix.max()}
        
        # Apply transformations if any
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
        int: The number of measurement files in the directory.
        """
        return len(glob.glob(self.dir_data + '/meas_*.tif'))


class Subset(data.Dataset):
    """
    Custom subset class that either returns the whole dataset item or a random crop 
    of the measurements.

    Attributes:
    dataset (data.Dataset): The original dataset to create a subset from.
    isVal (bool): A flag to indicate if the subset is for validation.
    """

    def __init__(self, dataset, isVal):
        """
        Initializes the Subset instance.

        Parameters:
        dataset (data.Dataset): The original dataset.
        isVal (bool): Indicates whether this subset is for validation.
        """
        self.dataset = dataset
        self.isVal = isVal

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset. If in validation mode, it returns the item 
        directly. Otherwise, it returns a random crop.

        Parameters:
        idx (int): The index of the item to retrieve.

        Returns:
        data (dict): Dictionary containing the cropped ground truth, measurement, and demixed images if applicable.
        """
        p = 256  # Size of the random patch
        if self.isVal:  
            # Return the full item for validation
            data = self.dataset.__getitem__(idx)
            return data
        else:
            # Get the full item
            data = self.dataset.__getitem__(idx)
            gt, meas, demix = data['gt'], data['meas'], data['demix']
            dim = meas.shape[-1]  # Get the dimensions of the measurement image
            
            # Randomly select a starting point for the crop
            a = torch.randint(0, dim - p, (1,))  
            b = torch.randint(0, dim - p, (1,))
            # Crop the images
            data = {'gt': gt[..., a:a + p, b:b + p], 
                    'meas': meas[..., a:a + p, b:b + p], 
                    'demix': demix[..., a:a + p, b:b + p]}
            return data

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
        int: The number of items in the original dataset.
        """
        return self.dataset.__len__()


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        """
        Converts the ground truth and measurement images in the sample from numpy ndarrays 
        to PyTorch tensors.

        Parameters:
        data (dict): Dictionary containing 'gt' and 'meas' images.

        Returns:
        dict: Dictionary containing 'gt' and 'meas' as tensors.
        """
        gt, meas = data['gt'], data['meas']
        return {'gt': torch.from_numpy(gt),
                'meas': torch.from_numpy(meas)}


class Noise(object):
    """Applies noise to the measurement images."""

    def __call__(self, data):
        """
        Applies random noise to the measurement image based on calibrated parameters.

        Parameters:
        data (dict): Dictionary containing 'gt' and 'meas' images.

        Returns:
        dict: Dictionary with the noisy measurement images.
        """
        gt, meas = data['gt'], data['meas']
        
        # Calibration parameters for noise
        amin = 7.8109e-5
        amax = 9.6636e-5
        bmin = 1.3836e-8
        bmax = 1.1204e-7
        
        # Generate random noise parameters
        a = np.random.rand(1) * (amax - amin) + amin  
        b = np.random.rand(1) * (bmax - bmin) + bmin  
        
        # Add noise to the measurement image
        meas += np.sqrt(a * meas + b) * np.random.randn(meas.shape[0], meas.shape[1])
        data = {'gt': gt, 'meas': meas}

        return data


class Resize(object):
    """Resize the measurement images by padding."""

    def __call__(self, data):
        """
        Applies padding to the measurement images.

        Parameters:
        data (dict): Dictionary containing 'gt' and 'meas' images.

        Returns:
        dict: Dictionary with padded measurement images.
        """
        gt, meas = data['gt'], data['meas']
        meas = np.pad(meas, ((657, 657), (350, 350)))  # Pad the measurement image
        data = {'gt': gt, 'meas': meas}
        return data


class ToTensorcm2(object):
    """Convert ndarrays in sample to Tensors for CM2 dataset."""

    def __call__(self, data):
        """
        Converts the ground truth, measurement, and demixed images from numpy ndarrays 
        to PyTorch tensors.

        Parameters:
        data (dict): Dictionary containing 'gt', 'meas', and 'demix' images.

        Returns:
        dict: Dictionary containing 'gt', 'meas', and 'demix' as tensors.
        """
        gt, meas, demix = data['gt'], data['meas'], data['demix']
        return {'gt': torch.from_numpy(gt),
                'meas': torch.from_numpy(meas),
                'demix': torch.from_numpy(demix)}


class Noisecm2(object):
    """Applies noise to the measurement images for CM2 dataset."""

    def __call__(self, data):
        """
        Applies random noise to the measurement image based on calibrated parameters.

        Parameters:
        data (dict): Dictionary containing 'gt', 'meas', and 'demix' images.

        Returns:
        dict: Dictionary with the noisy measurement images.
        """
        gt, meas, demix = data['gt'], data['meas'], data['demix']
        
        # Calibration parameters for noise
        amin = 7.8109e-5
        amax = 9.6636e-5
        bmin = 1.3836e-8
        bmax = 1.1204e-7
        
        # Generate random noise parameters
        a = np.random.rand(1) * (amax - amin) + amin  
        b = np.random.rand(1) * (bmax - bmin) + bmin  
        
        # Add noise to the measurement image
        meas += np.sqrt(a * meas + b) * np.random.randn(meas.shape[0], meas.shape[1])
        data = {'gt': gt, 'meas': meas, 'demix': demix}

        return data


class Crop(object):
    """Crop the measurement images to a specific size based on predefined locations."""

    def __call__(self, data):
        """
        Crops the measurement, ground truth, and demixed images to a specified size 
        at predefined locations.

        Parameters:
        data (dict): Dictionary containing 'gt', 'meas', and 'demix' images.

        Returns:
        dict: Dictionary with the cropped ground truth, measurement, and demixed images.
        """
        gt, meas, demix = data['gt'], data['meas'], data['demix']
        tot_len = 2400  # Desired crop size
        tmp_pad = 900   # Temporary padding size
        # Pad the measurement image
        meas = F.pad(meas, (tmp_pad, tmp_pad, tmp_pad, tmp_pad), 'constant', 0)

        # Predefined cropping locations
        loc = [(664, 1192), (664, 2089), (660, 2982),
               (1564, 1200), (1557, 2094), (1548, 2988),
               (2460, 1206), (2452, 2102), (2444, 2996)]

        # Create cropped images based on the locations
        meas = torch.stack([
            meas[x - (tot_len // 2) + tmp_pad:x + (tot_len // 2) + tmp_pad,
            y - (tot_len // 2) + tmp_pad:y + (tot_len // 2) + tmp_pad] for x, y in loc
        ])
        
        # Prepare final data dictionary
        data = {'gt': gt, 'meas': meas, 'demix': demix}
        return data