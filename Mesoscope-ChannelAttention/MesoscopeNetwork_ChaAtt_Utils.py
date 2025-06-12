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
Utility functions for model training and evaluation.

This module provides a set of utility functions to support model training and evaluation, including:

- `Logger`: A utility class for logging information during model training and evaluation.
- `Parser`: A utility class for parsing command line arguments.
- `set_requires_grad`: A function to set the 'requires_grad' attribute for all parameters in the given networks.
- `save`: A function to save the model state, optimizer state, and loss logger to a specified directory.
- `load`: A function to load the model state, optimizer state, and optionally the loss logger from a specified directory.

The Python script defines utility functions for model training and evaluation in PyTorch. It includes classes for logging and parsing command line arguments and functions for setting the 'requires_grad' attribute for network parameters, saving model states, and loading model states. A detailed explanation of each part of the script:

1. **Logger Class**: This class logs information during model training and evaluation. It has methods to configure and return a logger with a specified handler (stream or file) and to print or write the parsed arguments in a formatted table.

2. **Parser Class**: This class parses command line arguments. It initializes with an ArgumentParser instance and provides methods to get the parsed arguments and write them to a text file.

3. **set_requires_grad Function**: This function sets the 'requires_grad' attribute for all parameters in the given networks. It takes a list of networks or a single network instance and a boolean indicating whether the networks should require gradients.

4. **save function**: This function saves the model state, optimizer state, and loss logger to a specified directory. It takes the directory path, model instance, optimizer instance, current training epoch, and loss logger as inputs.

5. **load Function**: This function loads the model state, optimizer state, and, optionally, the loss logger from a specified directory. It supports loading in 'train' or 'test' modes, with different return values based on the mode.

The script provides utility functions for managing model training and evaluation tasks, including logging, parsing command line arguments, setting parameter gradients, and saving and loading model states.
"""

from __future__ import absolute_import, division, print_function
import torch.distributed as dist  # Importing distributed training functionalities
import os  # Importing OS module for file and directory operations
import logging  # Importing logging for logging messages
from math import log10, sqrt  # Importing mathematical functions
import torch  # Importing PyTorch library
import torch.optim.lr_scheduler as lr_scheduler  # Importing learning rate scheduler module
# import argparse  # Importing argparse for command line argument parsing (commented out)

'''
Class Logger: A utility class for logging information during model training and evaluation.
'''
class Logger:
    def __init__(self, info=logging.INFO, name=__name__):
        """
        Initializes the Logger with a specified logging level and name.

        Parameters:
            info (int): The logging level (default is logging.INFO).
            name (str): The name of the logger (default is the current module's name).
        """
        logger = logging.getLogger(name)  # Creating a logger with the specified name
        logger.setLevel(info)  # Setting the logging level

        self.__logger = logger  # Storing the logger instance

    def get_logger(self, handler_type='stream_handler'):
        """
        Configures and returns the logger with a specified handler.

        Parameters:
            handler_type (str): The type of handler to use ('stream_handler' for console logging,
                                'file_handler' for logging to a file).

        Returns:
            Logger: Configured logger instance.
        """
        if handler_type == 'stream_handler':
            handler = logging.StreamHandler()  # Creating a stream handler for console output
            log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # Setting log format
            handler.setFormatter(log_format)  # Assigning the format to the handler
        else:
            handler = logging.FileHandler('utils.log')  # Creating a file handler for logging to 'utils.log'

        self.__logger.addHandler(handler)  # Adding the handler to the logger

        return self.__logger  # Returning the configured logger


'''
Class Parser: A utility class for parsing command line arguments.
'''
class Parser:
    def __init__(self, parser):
        """
        Initializes the Parser with a command line argument parser.

        Parameters:
            parser (ArgumentParser): An instance of ArgumentParser to parse command line arguments.
        """
        self.__parser = parser  # Storing the parser instance
        self.__args = parser.parse_args()  # Parsing the arguments

    def get_parser(self):
        """
        Returns the argument parser instance.

        Returns:
            ArgumentParser: The stored argument parser instance.
        """
        return self.__parser  # Returning the parser instance

    def get_arguments(self):
        """
        Returns the parsed arguments.

        Returns:
            Namespace: The namespace containing parsed arguments.
        """
        return self.__args  # Returning the parsed arguments

    def write_args(self):
        """
        Writes the parsed arguments to a text file in the specified log directory.
        """
        params_dict = vars(self.__args)  # Converting the arguments to a dictionary

        log_dir = os.path.join(params_dict['dir_log'])  # Getting the log directory from arguments
        args_name = os.path.join(log_dir, 'args.txt')  # Defining the path for arguments text file

        if not os.path.exists(log_dir):  # Checking if the log directory exists
            os.makedirs(log_dir)  # Creating the log directory if it does not exist

        with open(args_name, 'wt') as args_fid:  # Opening the arguments file for writing
            args_fid.write('----' * 10 + '\n')  # Writing a separator line
            args_fid.write('{0:^40}'.format('PARAMETER TABLES') + '\n')  # Writing title
            args_fid.write('----' * 10 + '\n')  # Writing another separator line
            for k, v in sorted(params_dict.items()):  # Iterating through sorted arguments
                args_fid.write('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)) + '\n')  # Writing each parameter
            args_fid.write('----' * 10 + '\n')  # Writing final separator line

    def print_args(self, name='PARAMETER TABLES'):
        """
        Prints the parsed arguments in a formatted table.

        Parameters:
            name (str): The title to print above the arguments (default is 'PARAMETER TABLES').
        """
        params_dict = vars(self.__args)  # Converting the arguments to a dictionary

        print('----' * 10)  # Printing a separator line
        print('{0:^40}'.format(name))  # Printing the title
        print('----' * 10)  # Printing another separator line
        for k, v in sorted(params_dict.items()):  # Iterating through sorted arguments
            if '__' not in str(k):  # Excluding private attributes
                print('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)))  # Printing each parameter
        print('----' * 10)  # Printing final separator line


def set_requires_grad(nets, requires_grad=False):
    """
    Sets the 'requires_grad' attribute for all parameters in the given networks.

    Parameters:
        nets (list or network): A list of networks or a single network instance.
        requires_grad (bool): Whether the networks should require gradients (default is False).
    """
    if not isinstance(nets, list):  # Checking if the input is a list
        nets = [nets]  # Converting single network to a list
    for net in nets:  # Iterating through each network
        if net is not None:  # Checking if the network is not None
            for param in net.parameters():  # Iterating through parameters of the network
                param.requires_grad = requires_grad  # Setting the 'requires_grad' attribute


def save(dir_chck, model, optimizer, epoch, losslogger):
    """
    Saves the model state, optimizer state, and loss logger to a specified directory.

    Parameters:
        dir_chck (str): The directory where the model and training state will be saved.
        model (torch.nn.Module): The model instance whose state will be saved.
        optimizer (torch.optim.Optimizer): The optimizer instance whose state will be saved.
        epoch (int): The current training epoch number.
        losslogger (any): An object to log loss or training metrics.
    """
    if not os.path.exists(dir_chck):  # Checking if the directory exists
        os.makedirs(dir_chck)  # Creating the directory if it does not exist

    torch.save({'model': model.state_dict(),  # Saving the model's state dictionary
                'optim': optimizer.state_dict(),  # Saving the optimizer's state dictionary
                'losslogger': losslogger},  # Saving the loss logger
               '%s/model_epoch%04d.pth' % (dir_chck, epoch))  # Saving to a file with epoch number


def load(dir_chck, model, optimizer=[], epoch=[], mode='train'):
    """
    Loads the model state, optimizer state, and optionally the loss logger from a specified directory.

    Parameters:
        dir_chck (str): The directory from which the model and training state will be loaded.
        model (torch.nn.Module): The model instance to load the state into.
        optimizer (torch.optim.Optimizer, optional): The optimizer instance to load the state into (default is an empty list).
        epoch (list, optional): A list containing the epoch number to load (default is an empty list).
        mode (str): The mode in which to load ('train' or 'test', default is 'train').

    Returns:
        tuple: Depending on the mode, returns the loaded model, optimizer, epoch, and loss logger.
    """
    if not os.path.exists(dir_chck) or not os.listdir(dir_chck):  # Checking if the directory does not exist or is empty
        epoch = 0  # Setting epoch to 0 if no checkpoint exists
        if mode == 'train':  # If in training mode
            return model, optimizer, epoch  # Returning model, optimizer, and epoch
        elif mode == 'test':  # If in testing mode
            return model, epoch  # Returning model and epoch

    if not epoch:  # If no specific epoch is provided
        ckpt = os.listdir(dir_chck)  # Listing checkpoint files in the directory
        ckpt.sort()  # Sorting the checkpoint files
        epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])  # Extracting the epoch number from the latest checkpoint
        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))  # Loading the model state from the checkpoint

    print('Loaded %dth network' % epoch)  # Printing the loaded epoch number
    
    if mode == 'train':  # If in training mode
        model.load_state_dict(dict_net['model'])  # Loading model state
        optimizer.load_state_dict(dict_net['optim'])  # Loading optimizer state
        losslogger = dict_net['losslogger']  # Loading loss logger state

        return model, optimizer, epoch, losslogger  # Returning model, optimizer, epoch, and loss logger

    elif mode == 'test':  # If in testing mode
        model.load_state_dict(dict_net['model'])  # Loading model state
        losslogger = dict_net['losslogger']  # Loading loss logger state

        return model, losslogger  # Returning model and loss logger