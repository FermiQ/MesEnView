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
Defines custom PyTorch dataset classes for loading and preprocessing data for a machine learning model.

The `CustomDataset` class loads data from a Parquet file and provides access to the stack of lightfield views, the refocused volume, and the ground truth target. The `ZarrData` class provides a caching mechanism for efficiently loading data from TIFF files stored in Zarr format. The `PatchDataset` class extends the `CustomDataset` to extract random patches from the data, which can be useful for training models.
"""
import os  # Importing the os module for operating system dependent functionality
from functools import cached_property, cache  # Importing decorators for caching properties and functions
from typing import Tuple  # Importing Tuple type for type hints
from pandas import DataFrame, read_parquet  # Importing DataFrame and read_parquet for handling dataframes and reading parquet files

import numpy as np  # Importing numpy for numerical operations
import torch  # Importing PyTorch for tensor operations and model training
import zarr  # Importing zarr for chunked compressed data storage
from tifffile import imread, TiffFile  # Importing tifffile to read TIFF image files
from torch.utils.data import Dataset  # Importing Dataset class from PyTorch for creating custom datasets


class CustomDataset(Dataset):
    def __init__(self, config: dict):
        """Initialize the CustomDataset.

        Args:
            config (dict): Configuration dictionary containing dataset parameters,
            including the path to the parquet dataset file and scattering type.
        """
        super(CustomDataset, self).__init__()  # Call the parent class constructor
        self.df = read_parquet(config["dataset_pq"])  # Load dataset from parquet file into a DataFrame
        self.scattering = config["scattering"]  # Store scattering type from config

    def __len__(self):
        """Return the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.df)  # Return the length of the DataFrame

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve a single sample from the dataset.

        Args:
            index (int): Index of the data. The input measurement data is stored in the format of 
            meas_{index}.tiff, and the output is stored in the format of gt_vol_{index}.tiff.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Stack of lightfield views, the refocused volume,
            and the ground truth target, all normalized to [0,1] in torch tensor format with 32-bit float.
        """

        # Read the stack of images from the path specified in the DataFrame and normalize
        stack = (
            imread(self.df[f"stack_{self.scattering}_path"].iloc[index]).astype(np.float32)
            / 0xFFFF
        )
        stack = torch.from_numpy(stack)  # Convert numpy array to torch tensor

        # Read the refocused volume from the corresponding path and normalize
        rfv = (
            imread(self.df[f"rfv_{self.scattering}_path"].iloc[index]).astype(np.float32)
            / 0xFFFF
        )
        rfv = torch.from_numpy(rfv)  # Convert numpy array to torch tensor

        # Read the ground truth from the path and normalize
        gt = imread(self.df["gt_path"].iloc[index]).astype(np.float32) / 0xFFFF
        gt = torch.from_numpy(gt)  # Convert numpy array to torch tensor

        return stack, rfv, gt  # Return the stack, refocused volume, and ground truth


class ZarrData:
    def __init__(self, df: DataFrame, datatype: str, scattering: str):
        """Initialize the ZarrData class.

        Args:
            df (DataFrame): DataFrame containing paths to data.
            datatype (str): Type of data to load, must be one of 'stack', 'rfv', or 'gt'.
            scattering (str): The scattering type related to the data.
        """
        self.df = df  # Store the DataFrame

        # Validate the datatype to ensure it is one of the expected values
        if datatype not in ["stack", "rfv", "gt"]:
            raise ValueError("datatype must be one of stack, rfv, gt")

        self.datatype = datatype  # Store the datatype
        self.scattering = scattering  # Store the scattering type
        self.open_zarrs = []  # List to keep track of opened zarr files

    # NOTE: ensure cache is larger than number of items
    @cache
    def __getitem__(self, index: int):
        """Retrieve a zarr dataset for the specified index.

        Args:
            index (int): Index of the data.

        Returns:
            zarr.Container: The zarr data object for the specified index.
        """
        path = self.df[self.datatype + f"_{self.scattering}_path"].iloc[index]  # Get the path from DataFrame
        with TiffFile(path) as img:  # Open the TIFF file using TiffFile
            return zarr.open(img.aszarr())  # Open and return the zarr representation of the image


class PatchDataset(Dataset):
    def __init__(self, dataset: Dataset, config: dict):
        """Initialize the PatchDataset for creating patches from the dataset.

        Args:
            dataset (Dataset): The dataset from which to extract patches, typically a training split.
            config (dict): Configuration dictionary containing parameters such as patch size and scattering type.
        """
        self.dataset = dataset  # Store the base dataset
        self.df = read_parquet(config["dataset_pq"])  # Load dataset metadata from parquet file
        self.patch_size = config["patch_size"]  # Store the patch size from config
        self.scattering = config["scattering"]  # Store scattering type from config

    @cached_property
    def stack(self) -> ZarrData:
        """Lazy load stack data as a ZarrData object.

        Returns:
            ZarrData: Instance of ZarrData for stack data.
        """
        return ZarrData(self.df, "stack", self.scattering)  # Create ZarrData instance for stacks

    @cached_property
    def rfv(self) -> ZarrData:
        """Lazy load refocused volume data as a ZarrData object.

        Returns:
            ZarrData: Instance of ZarrData for refocused volume data.
        """
        return ZarrData(self.df, "rfv", self.scattering)  # Create ZarrData instance for refocused volumes

    @cached_property
    def gt(self) -> ZarrData:
        """Lazy load ground truth data as a ZarrData object.

        Returns:
            ZarrData: Instance of ZarrData for ground truth data.
        """
        return ZarrData(self.df, "gt")  # Create ZarrData instance for ground truth data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve a random patch of the data.

        Args:
            idx (int): Index of the data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A patch of stack, refocused volume, and ground truth 
            data each of size patch_size.
        """
        # Recipe for fast dataloading with zarr courtesy of Mitchell Gilmore mgilm0re@bu.edu
        stack = self.stack[idx]  # Retrieve stack data for the given index
        rfv = self.rfv[idx]  # Retrieve refocused volume data for the given index
        gt = self.gt[idx]  # Retrieve ground truth data for the given index

        # Generate random starting indices for the patch
        row_start = torch.randint(0, stack.shape[-2] - self.patch_size, (1,))
        col_start = torch.randint(0, stack.shape[-1] - self.patch_size, (1,))

        # Create slices for row and column based on the starting indices
        row_slice = slice(row_start, row_start + self.patch_size)
        col_slice = slice(col_start, col_start + self.patch_size)

        # Extract patches from the data and normalize
        stack = torch.from_numpy(
            stack[:, row_slice, col_slice].astype(np.float32) / 0xFFFF
        )
        rfv = torch.from_numpy(rfv[:, row_slice, col_slice].astype(np.float32) / 0xFFFF)
        gt = torch.from_numpy(gt[:, row_slice, col_slice].astype(np.float32) / 0xFFFF)

        return stack, rfv, gt  # Return the extracted patches

    def __len__(self):
        """Return the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.dataset)  # Return the length of the base dataset