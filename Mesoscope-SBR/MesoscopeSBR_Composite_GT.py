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
This script combines existing ground truths for beads and synthetic vasculature into a single composite ground truth.

The script reads the bead and vasculature ground truth TIFF files, combines them with a weighting factor, normalizes the result linearly, and writes the composite ground truth to a new TIFF file in the output path.

The script is intended to be a temporary solution until the synthetic data generation is migrated to the sbrnet-core package.
"""
import os
from sbrnet_core.utils import full_read_tiff, write_tiff, linear_normalize

# The sbrnet-core package does not have a script to generate the synthetic ground truths.
# This script takes existing ground truths (beads and vasculature) and combines them into one composite ground truth.
# It is planned to migrate synthetic data generation to the sbrnet-core package from legacy MATLAB scripts.

# Define paths for the input and output data
vasc_path = "/ad/eng/research/eng_research_cisl/jalido/sbrnet/data/synthetic_vasculature/16um_diameter"  # Path to synthetic vasculature TIFF files
bead_path = "/ad/eng/research/eng_research_cisl/jalido/sbrnet/data/beads/"  # Path to bead ground truth TIFF files
out_path = "/ad/eng/research/eng_research_cisl/jalido/sbrnet/data/gts/composite/"  # Output path for composite ground truths

# Check if the output path exists; if not, create it
if not os.path.exists(out_path):
    os.makedirs(out_path)

# Loop through a range of indices to process multiple TIFF files
for i in range(500):
    # Read the bead ground truth TIFF file for the current index
    beads = full_read_tiff(os.path.join(bead_path, f"sim_gt_vol_{i}.tif"))
    
    # Read the synthetic vasculature TIFF file for the current index
    vasc = full_read_tiff(os.path.join(vasc_path, f"Lnet_i_{i}.tiff"))
    
    # Combine beads and vasculature with a weighting factor, normalizing the result linearly
    gt = linear_normalize(beads + 0.5 * vasc)
    
    # Write the composite ground truth to a new TIFF file in the output path
    write_tiff(gt, os.path.join(out_path, f"gt_vol_{i}.tiff"))
    
    # Print the current index to track progress
    print(i)