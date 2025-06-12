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
Processes a single iteration of synthetic data generation.

Parameters:
- i (int): The index of the current iteration, used for naming output files.
- config (dict): A configuration dictionary containing parameters for the process.

Returns:
- pd.DataFrame: A DataFrame containing metadata for the generated synthetic measurement.
"""
import ray
import os
import random
import pandas as pd
import logging
from sbrnet_core.utils.constants import NUM_SLICES, view_combos
from sbrnet_core.utils import (
    normalize_psf_power,
    full_read_tiff,
    full_read,
    lsi_fwd_mdl,
    pad_3d_array_to_size,
    write_tiff16,
    sbrnet_utils,
)

# Initialize a logger for tracking events and debugging
logger = logging.getLogger(__name__)

@ray.remote
def process_single_iteration(i, config):
    """
    Processes a single iteration of synthetic data generation.

    Parameters:
    - i (int): The index of the current iteration, used for naming output files.
    - config (dict): A configuration dictionary containing parameters for the process.

    Returns:
    - pd.DataFrame: A DataFrame containing metadata for the generated synthetic measurement.
    """
    
    # Determine which view combination to use based on the index in the config
    view_combo_index = config["view_ind"] - 1
    view_list = view_combos[view_combo_index]

    # Set the lower and upper bounds for signal-to-background ratio (SBR)
    low_sbr = config["lower_sbr"]
    upper_sbr = config["upper_sbr"]

    # Load the point spread function (PSF) from the specified path and normalize it
    psf_path = config["psf_path"]
    PSF = normalize_psf_power(full_read_tiff(psf_path))
    
    # Define the paths for ground truth (GT) and value data
    gt_folder = config["gt_path"]
    value_folder = config["value_path"]

    # Load lenslet and MLA apodization data
    lens_apodize_path = config["lenslet_apodize_path"]
    LENS_AP = full_read_tiff(lens_apodize_path)
    mla_apodize_path = config["mla_apodize_path"]
    MLA_AP = full_read_tiff(mla_apodize_path)

    # Randomly select an SBR within the specified range
    sbr = random.uniform(low_sbr, upper_sbr)
    
    # Load the ground truth volume and associated value data
    gt_path = os.path.join(gt_folder, f"sim_gt_vol_{i}.tif")
    gt = full_read_tiff(gt_path)
    value_path = os.path.join(value_folder, f"value_{i+1}.png")
    value = full_read(value_path)

    # Create a free space measurement from the ground truth and PSF
    fs_meas = lsi_fwd_mdl(pad_3d_array_to_size(gt, PSF.shape), PSF)

    # Generate a background image from the value data and apodization data
    bg_meas, bg_mean = sbrnet_utils.make_bg_img(value, LENS_AP, MLA_AP)

    # Create a synthetic measurement by combining free space measurement and background
    synthetic_scat_measurement = sbrnet_utils.make_measurement(
        fs_meas, bg_meas, sbr, bg_mean
    )

    # Crop the synthetic measurement to the desired view dimensions
    stack_scat = sbrnet_utils.crop_views(im=synthetic_scat_measurement)
    
    # Retain only the views specified in the view list
    stack_scat = sbrnet_utils.zero_slices_not_in_list(stack_scat, view_list)

    # Refocus the light field into a volume using the specified number of slices
    rfv_scat = sbrnet_utils.lf_refocus_volume(
        lf=stack_scat, z_slices=NUM_SLICES, max_shift=NUM_SLICES // 2 + 1
    )

    # Retain only the views specified in the view list after refocusing
    stack_scat = stack_scat[view_list, :, :]

    #### Process free space measurement #####
    stack_free = sbrnet_utils.crop_views(im=fs_meas)

    # Use the same view list to keep only the relevant views
    stack_free = sbrnet_utils.zero_slices_not_in_list(stack_free, view_list)

    # Refocus the free space light field
    rfv_free = sbrnet_utils.lf_refocus_volume(
        lf=stack_free, z_slices=NUM_SLICES, max_shift=NUM_SLICES // 2 + 1
    )

    # Retain only the relevant views after refocusing for free space
    stack_free = stack_free[view_list, :, :]
    #### End of free space processing #####

    # Define output folder structure and paths for saving generated data
    out_folder = config["out_dir"]
    out_stack_scat_folder = os.path.join(out_folder, "stack_scattering")
    out_rfv_scat_folder = os.path.join(out_folder, "rfv_scattering")
    out_stack_free_folder = os.path.join(out_folder, "stack_freespace")
    out_rfv_free_folder = os.path.join(out_folder, "rfv_freespace")

    out_stack_scat_path = os.path.join(out_stack_scat_folder, f"meas_{i}.tiff")
    out_stack_free_path = os.path.join(out_stack_free_folder, f"meas_{i}.tiff")
    out_rfv_scat_path = os.path.join(out_rfv_scat_folder, f"meas_{i}.tiff")
    out_rfv_free_path = os.path.join(out_rfv_free_folder, f"meas_{i}.tiff")

    # Write the generated stacks to TIFF files
    write_tiff16(stack_scat, out_stack_scat_path)
    write_tiff16(rfv_scat, out_rfv_scat_path)
    write_tiff16(stack_free, out_stack_free_path)
    write_tiff16(rfv_free, out_rfv_free_path)

    # Create a DataFrame to store metadata about the generated data
    rowdata = pd.DataFrame(
        {
            "num_views": [len(view_list)],  # Number of views used
            "view_combo": [view_list],  # The list of views
            "psf_path": [psf_path],  # Path to the PSF
            "lens_apodized_path": [lens_apodize_path],  # Path to lens apodization data
            "mla_apodized_path": [mla_apodize_path],  # Path to MLA apodization data
            "gt_folder": [gt_folder],  # Path to ground truth data
            "value_path": [value_path],  # Path to value data
            "sbr": [sbr],  # Randomly selected SBR value
            "sbr_range": [f"{low_sbr}-{upper_sbr}"],  # SBR range
            "stack_scat_path": [out_stack_scat_path],  # Output path for scattering stack
            "rfv_scat_path": [out_rfv_scat_path],  # Output path for refocused scattering volume
            "stack_free_path": [out_stack_free_path],  # Output path for free space stack
            "rfv_free_path": [out_rfv_free_path],  # Output path for refocused free space volume
            "gt_path": [gt_path],  # Path to ground truth volume
        }
    )
    
    # Log completion of the current iteration
    logger.info(f"Finished iteration {i}")
    return rowdata

def make_synthetic_dataset(config: dict) -> None:
    """
    Generates a synthetic dataset based on the provided configuration.

    Parameters:
    - config (dict): A configuration dictionary containing parameters for dataset generation.
    """

    # Create the output folder if it doesn't exist
    out_folder = config["out_dir"]
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Create subfolders for different types of output data
    out_stack_folder = os.path.join(out_folder, "stack_scattering")
    if not os.path.exists(out_stack_folder):
        os.makedirs(out_stack_folder)

    out_rfv_folder = os.path.join(out_folder, "rfv_scattering")
    if not os.path.exists(out_rfv_folder):
        os.makedirs(out_rfv_folder)

    out_stack_folder = os.path.join(out_folder, "stack_freespace")
    if not os.path.exists(out_stack_folder):
        os.makedirs(out_stack_folder)

    out_rfv_folder = os.path.join(out_folder, "rfv_freespace")
    if not os.path.exists(out_rfv_folder):
        os.makedirs(out_rfv_folder)

    # Check if Ray is to be used for parallel processing
    if config["use_ray"]:
        # Initialize Ray with the number of CPUs
        ray.init(ignore_reinit_error=True, num_cpus=int(os.getenv("NSLOTS")))
        # Prepare and execute remote function calls for all iterations
        futures = [
            process_single_iteration.remote(i, config) for i in range(config["N"])
        ]
        results = ray.get(futures)  # Collect results from the remote calls
        df = pd.concat(results, ignore_index=True, axis=0)  # Concatenate results into a DataFrame
    else:
        # Prepare an empty DataFrame for non-Ray execution
        df = pd.DataFrame()
        # Process each iteration sequentially
        for i in range(config["N"]):
            rowdata = process_single_iteration(i, config)  # Call the function directly
            df = pd.concat([df, rowdata], ignore_index=True, axis=0)  # Concatenate the results

    # Save the metadata about the dataset in a parquet file
    df.to_parquet(
        os.path.join(out_folder, "metadata.pq")
    )