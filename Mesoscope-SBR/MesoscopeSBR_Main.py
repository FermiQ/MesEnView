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
Main function to execute the training process.

Args:
    args (argparse.Namespace): Namespace containing command-line arguments as attributes.
"""
import logging  # Import the logging module to enable logging in the application
import argparse  # Import argparse for parsing command-line arguments
from datetime import datetime  # Import datetime to generate timestamps
from torch import compile  # Import the compile function from torch for model compilation

from sbrnet_core.sbrnet.model import SBRNet  # Import the SBRNet model from the SBRNet core module

# from sbrnet_core.config_loader import load_config # Not needed anymore
from sbrnet_core.sbrnet.trainer import Trainer  # Import the Trainer class to handle the training process

# Get the current timestamp as a formatted string
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define the log file path using the current timestamp to differentiate log files
log_file_path = f"/projectMESENVIEW/inserm/.log/logging/sbrnet_train_{current_time}.log"

# Configure the logging to write logs to the specified log file with INFO level
logging.basicConfig(filename=log_file_path, level=logging.INFO)

logger = logging.getLogger(__name__)  # Create a logger for the current module


def main(args):
    """
    Main function to execute the training process.

    :param args: Namespace containing command-line arguments as attributes.
    """

    # Convert the argparse namespace to a dictionary for easier access
    config = vars(args)

    # Compile the SBRNet model with the given configuration
    model = compile(SBRNet(config))

    # Initialize the Trainer with the compiled model and configuration
    trainer = Trainer(model, config)

    logger.info("Starting training...")  # Log the start of training

    trainer.train()  # Start the training process

    logger.info("Training complete.")  # Log the completion of training


if __name__ == "__main__":  # Entry point of the script
    # Create an ArgumentParser object to handle command-line arguments
    parser = argparse.ArgumentParser(
        description="Train SBRNet with command-line parameters."
    )

    # Define command-line arguments and their properties
    # paths
    parser.add_argument(
        "--dataset_pq",
        type=str,
        required=True,
        help="Path to the Parquet dataset file.",  # A description of the argument
    )
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Directory to save trained models."
    )
    parser.add_argument(
        "--scattering",
        type=str,
        required=True,
        choices=["scat", "free"],  # Restrict options for this argument
        help="whether to use scattering or free space data.",
    )

    # training parameters
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="The ratio of training set split.",  # Default is 80% for training
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training."  # Default batch size
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer.",  # Default learning rate
    )
    parser.add_argument(
        "--epochs", type=int, default=20000, help="Number of epochs for training."  # Total training epochs
    )
    parser.add_argument(
        "--backbone", type=str, default="resnet", help="The backbone network type."
    )
    parser.add_argument(
        "--resnet_channels",
        type=int,
        default=48,
        help="Number of channels in resnet backbone.",  # Number of channels for ResNet
    )
    parser.add_argument(
        "--weight_init",
        type=str,
        default="kaiming_normal",
        help="Weight initialization method.",  # Default weight initialization
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Seed for random number generators."  # Seed for reproducibility
    )
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer type.")  # Default optimizer
    parser.add_argument(
        "--criterion_name",
        type=str,
        default="bce_with_logits",
        help="Criterion name for the loss function.",  # Default loss function
    )
    parser.add_argument(
        "--use_amp",
        type=bool,
        default=True,
        help="Whether to use automatic mixed precision or not.",  # Default is to use AMP
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine_annealing",
        help="Learning rate scheduler type.",  # Default learning rate scheduler
    )
    parser.add_argument(
        "--cosine_annealing_T_max",
        type=int,
        default=30,
        help="Maximum number of iterations for cosine annealing scheduler.",  # Iterations for scheduler
    )

    # model parameters
    parser.add_argument(
        "--num_gt_layers", type=int, default=24, help="Number of ground truth layers."  # Default GT layers
    )
    parser.add_argument(
        "--num_lf_views", type=int, default=9, help="Number of light field views."  # Number of LF views
    )
    parser.add_argument(
        "--num_rfv_layers", type=int, default=24, help="Number of RFV layers."  # Default RFV layers
    )
    parser.add_argument(
        "--num_resblocks", type=int, default=20, help="Number of residual blocks."  # Default number of residual blocks
    )
    parser.add_argument(
        "--patch_size", type=int, default=224, help="Size of the patch."  # Size of input patch
    )
    parser.add_argument(
        "--num_head_layers",
        type=int,
        default=3,
        help="number of conv layers for the head.",  # Number of conv layers for the head
    )

    # calibrated parameters for poisson-gaussian noise model
    parser.add_argument(
        "--A_STD",
        type=float,
        default=5.7092e-5,
        help="Standard deviation of A for poisson-gaussian noise model.",  # Std for noise model
    )
    parser.add_argument(
        "--A_MEAN",
        type=float,
        default=1.49e-4,
        help="Mean of A for poisson-gaussian noise model.",  # Mean for noise model
    )
    parser.add_argument(
        "--B_STD",
        type=float,
        default=2.7754e-6,
        help="Standard deviation of B for poisson-gaussian noise model.",  # Std for noise model
    )
    parser.add_argument(
        "--B_MEAN",
        type=float,
        default=5.41e-6,
        help="Mean of B for poisson-gaussian noise model.",  # Mean for noise model
    )

    # Parse the arguments from the command line
    args = parser.parse_args()

    main(args)  # Call the main function with parsed arguments