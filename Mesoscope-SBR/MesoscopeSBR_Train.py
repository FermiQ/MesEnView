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
The Trainer class encapsulates the training and validation of a deep learning model.

The Trainer class is responsible for the following tasks:
- Initializing the model, configuration, and other necessary components for training.
- Setting up the training and validation data loaders.
- Performing the training loop, including forward passes, loss computation, backpropagation, and model updates.
- Validating the model on the validation dataset and tracking the lowest validation loss.
- Saving the model checkpoint when the validation loss improves.
- Providing utility methods for setting random seeds, initializing optimizers and learning rate schedulers.

The Trainer class is designed to be a reusable component for training deep learning models, abstracting away the boilerplate code and allowing the user to focus on the model architecture and hyperparameter tuning.
"""
import datetime
import logging
import os
from typing import Tuple
import time
from pandas import read_parquet

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from sbrnet_core.sbrnet.dataset import CustomDataset, PatchDataset
from sbrnet_core.sbrnet.noisemodel import PoissonGaussianNoiseModel
from sbrnet_core.sbrnet.losses.pinball import PinballLoss

# Get the current date and time for logging purposes
now = datetime.datetime.now()

# Create a timestamp string for saving models
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

## for debugging
# Clear the CUDA cache for memory management
# torch.cuda.empty_cache()
# Set environment variables for debugging
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
##

# Initialize a logger for the module
logger = logging.getLogger(__name__)

class Trainer:
    """
    Trainer class to encapsulate the training and validation of a deep learning model.

    Attributes:
        model (Module): The neural network model to be trained.
        config (dict): Configuration parameters including learning rate, epochs, etc.
        noise_model (PoissonGaussianNoiseModel): A model to simulate noise in the data.
        learning_rate (float): The learning rate for the optimizer.
        epochs (int): Number of training epochs.
        model_dir (str): Directory to save the trained model.
        lowest_val_loss (float): The lowest validation loss observed.
        training_losses (list): List to store training losses over epochs.
        validation_losses (list): List to store validation losses over epochs.
        random_seed (int): Seed for random number generation.
        use_amp (bool): Flag to use Automatic Mixed Precision training.
        optimizer_name (str): Name of the optimizer to use.
        lr_scheduler_name (str): Name of the learning rate scheduler to use.
        criterion_name (str): Name of the loss criterion to use.
        device (torch.device): Device to run the training on (CPU or GPU).
        scaler (GradScaler): Scaler for mixed precision training.
        criterion (Loss): Loss function used for training.
        train_data_loader (DataLoader): DataLoader for training data.
        val_data_loader (DataLoader): DataLoader for validation data.
    """
    
    def __init__(self, model: Module, config: dict):
        """
        Initializes the Trainer with model and configuration.

        Parameters:
            model (Module): The neural network model to be trained.
            config (dict): Configuration parameters with keys:
                - learning_rate (float): The learning rate for the optimizer.
                - epochs (int): Number of training epochs.
                - model_dir (str): Directory to save the trained model.
                - random_seed (int, optional): Seed for random number generation.
                - use_amp (bool, optional): Flag to use Automatic Mixed Precision training.
                - optimizer (str, optional): Name of the optimizer to use (default: "adam").
                - lr_scheduler (str, optional): Name of the learning rate scheduler to use (default: "cosine_annealing").
                - loss_criterion (str, optional): Name of the loss criterion to use (default: "bce_with_logits").
        """
        self.config = config
        self.model = model
        self.noise_model = PoissonGaussianNoiseModel(config)
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]
        self.model_dir = config["model_dir"]
        self.lowest_val_loss = float("inf")
        self.training_losses = []
        self.validation_losses = []
        self.random_seed = config.get("random_seed", None)
        self.use_amp = config.get("use_amp", False)
        self.optimizer_name = config.get("optimizer", "adam")
        self.lr_scheduler_name = config.get("lr_scheduler", "cosine_annealing")
        self.criterion_name = config.get("loss_criterion", "bce_with_logits")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create the model directory if it does not exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        logger.debug(f"Using device: {self.device}")
        self.scaler = (
            GradScaler() if self.use_amp else None
        )  # Initialize the scaler if using AMP

        # Initialize the loss criterion based on the configuration
        if self.criterion_name == "bce_with_logits":
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.criterion_name == "mse":
            self.criterion = nn.MSELoss()
        elif self.criterion_name == "mae":
            self.criterion = nn.L1Loss()
        else:
            print(
                f"Unknown loss criterion: {self.criterion_name}. Using BCEWithLogitsLoss."
            )
        self.train_data_loader, self.val_data_loader = self._get_dataloaders()

    def _get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Create DataLoaders for training and validation datasets.

        Returns:
            Tuple[DataLoader, DataLoader]: A tuple containing the training and validation DataLoaders.
        """
        
        def split_dataset(dataset, split_ratio):
            """
            Split the dataset into training and validation sets based on the given ratio.

            Parameters:
                dataset (Dataset): The complete dataset to be split.
                split_ratio (float): Ratio of the dataset to be used for training.

            Returns:
                Tuple[Dataset, Dataset]: Training and validation datasets.
            """
            dataset_size = len(dataset)
            train_size = int(split_ratio * dataset_size)
            val_size = dataset_size - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            return train_dataset, val_dataset

        # Create a complete dataset from the configuration
        complete_dataset: Dataset = CustomDataset(self.config)
        split_ratio = self.config["train_split"]
        train_dataset, val_dataset = split_dataset(complete_dataset, split_ratio)

        # only train_dataset is a PatchDataset. val_dataset is full sized images.
        train_dataset = PatchDataset(
            dataset=train_dataset,
            config=self.config,
        )

        # Create DataLoaders for both training and validation datasets
        train_dataloader = DataLoader(
            train_dataset,
            self.config.get("batch_size"),
            shuffle=True,
            num_workers=4,  # Number of subprocesses for data loading
            pin_memory=True,  # Enable pin_memory for faster data transfer to GPU
        )
        val_dataloader = DataLoader(
            val_dataset,
            self.config["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        return train_dataloader, val_dataloader

    def _set_random_seed(self):
        """
        Set the random seed for reproducibility if specified in the configuration.
        """
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_seed)

    def _initialize_optimizer(self):
        """
        Initialize the optimizer based on the configuration.

        Returns:
            Optimizer: An optimizer instance for training the model.
        """
        if self.optimizer_name == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
            )
        elif self.optimizer_name == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(), lr=self.learning_rate, momentum=0.9
            )
        else:
            print(f"Unknown optimizer: {self.optimizer_name}. Using Adam.")
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _initialize_lr_scheduler(self, optimizer):
        """
        Initialize the learning rate scheduler based on the configuration.

        Parameters:
            optimizer (Optimizer): The optimizer for which to initialize the scheduler.

        Returns:
            lr_scheduler: A learning rate scheduler instance.
        """
        if self.lr_scheduler_name == "cosine_annealing":
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config["cosine_annealing_T_max"]
            )
        elif self.lr_scheduler_name == "step_lr":
            # StepLR scheduler
            step_size = self.config.get("step_lr_step_size", 10)
            gamma = self.config.get("step_lr_gamma", 0.5)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif self.lr_scheduler_name == "plateau":
            # Plateau scheduler
            patience = self.config.get("plateau_patience", 10)
            factor = self.config.get("plateau_factor", 0.5)
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=patience, factor=factor, verbose=True
            )
        else:
            print(
                f"Unknown LR scheduler: {self.lr_scheduler_name}. Using Cosine Annealing."
            )
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        return scheduler

    def train(self):
        """
        Perform the training loop for the specified number of epochs.
        - Each epoch includes training on the training dataset followed by validation on the validation dataset.
        - The model is saved if the validation loss improves.
        """
        model_name = f"sbrnet_{timestamp}.pt"  # Generate model name with timestamp
        self.model.to(self.device)  # Move the model to the selected device
        self.noise_model.to(self.device)  # Move the noise model to the selected device
        self._set_random_seed()  # Set the random seed for reproducibility

        optimizer = self._initialize_optimizer()  # Initialize the optimizer
        scheduler = self._initialize_lr_scheduler(optimizer)  # Initialize the learning rate scheduler
        start_time = time.time()  # Record the start time for timing the training process

        if self.use_amp:
            print("Using mixed-precision training with AMP.")  # Notify the use of AMP

        for epoch in range(self.epochs):
            self.model.train()  # Set the model to training mode
            total_loss = 0  # Initialize total loss for the epoch
            for lf_view_stack, rfv, gt in self.train_data_loader:  # Iterate over training batches
                # Move data to the selected device
                lf_view_stack, rfv, gt = (
                    lf_view_stack.to(self.device),
                    rfv.to(self.device),
                    gt.to(self.device),
                )

                # Apply noise model to the inputs
                lf_view_stack, rfv = self.noise_model(lf_view_stack, rfv)

                optimizer.zero_grad()  # Reset gradients for the optimizer

                # Forward pass and loss calculation
                if self.use_amp:
                    with autocast():  # Enable autocasting for mixed-precision
                        output = self.model(lf_view_stack, rfv)  # Forward pass
                        loss = self.criterion(output, gt)  # Compute loss
                    self.scaler.scale(loss).backward()  # Scale the loss and perform backpropagation
                    self.scaler.step(optimizer)  # Step the optimizer
                    self.scaler.update()  # Update the scaler
                else:
                    output = self.model(lf_view_stack, rfv)  # Forward pass
                    loss = self.criterion(output, gt)  # Compute loss
                    loss.backward()  # Backpropagation
                    optimizer.step()  # Step the optimizer
                    
                # Log the debug information for the current epoch
                logger.debug(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {loss.item()}")

                total_loss += loss.item()  # Accumulate the loss

            avg_train_loss = total_loss / len(self.train_data_loader)  # Average loss for the epoch
            self.training_losses.append(avg_train_loss)  # Store training loss
            logger.info(
                f"Epoch [{epoch + 1}/{self.epochs}], Train Loss: {avg_train_loss}"
            )

            val_loss = self.validate()  # Validate the model on the validation set
            self.validation_losses.append(val_loss)  # Store validation loss
            logger.info(
                f"Epoch [{epoch + 1}/{self.epochs}], Validation Loss: {val_loss}"
            )

            # Step the learning rate scheduler based on the validation loss
            if self.lr_scheduler_name == "plateau":
                scheduler.step(val_loss)  # For Plateau scheduler, pass validation loss as an argument
            else:
                scheduler.step()  # Step the scheduler

            # Save the model if the validation loss has improved
            if val_loss < self.lowest_val_loss:
                self.lowest_val_loss = val_loss  # Update the lowest validation loss
                save_state = {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "training_losses": self.training_losses,
                    "validation_losses": self.validation_losses,
                    "time_elapsed": time.time() - start_time,
                }

                # Update the saving state with the configuration parameters
                save_state.update(self.config)
                model_save_path = os.path.join(self.model_dir, model_name)  # Create the save path
                torch.save(save_state, model_save_path)  # Save the model state
                logger.info("Model saved at epoch {}".format(epoch + 1))  # Log the save event

    def validate(self):
        """
        Validate the model and compute the average validation loss.

        Returns:
            float: The average validation loss across the validation dataset.
        """
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0  # Initialize total loss for validation
        with torch.no_grad():  # Disable gradient calculation for validation
            for lf_view_stack, rfv, gt in self.val_data_loader:  # Iterate over validation batches
                lf_view_stack, rfv, gt = (
                    lf_view_stack.to(self.device),
                    rfv.to(self.device),
                    gt.to(self.device),
                )
                output = self.model(lf_view_stack, rfv)  # Forward pass
                loss = self.criterion(output, gt)  # Compute loss
                total_loss += loss.item()  # Accumulate the loss
        return total_loss / len(self.val_data_loader)  # Return average validation loss