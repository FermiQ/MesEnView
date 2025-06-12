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
Load configuration settings from a YAML file.

This function opens a specified YAML configuration file, reads its contents,
and parses it into a Python dictionary using PyYAML's safe_load function.

Args:
    config_path (str): The file path to the YAML configuration file that needs 
                       to be loaded.

Returns:
    dict: A dictionary representation of the YAML file contents.
"""
import yaml  # Import the yaml module to enable reading and writing YAML files.

def load_config(config_path: str):
    """
    Load configuration settings from a YAML file.

    This function opens a specified YAML configuration file, reads its contents,
    and parses it into a Python dictionary using PyYAML's safe_load function.
    
    Parameters:
    config_path (str): The file path to the YAML configuration file that needs 
                       to be loaded.

    Returns:
    dict: A dictionary representation of the YAML file contents.
    """
    # Open the specified file in read mode
    with open(config_path, "r") as f:
        # Use yaml.safe_load to parse the YAML file into a Python dictionary
        config = yaml.safe_load(f)

    # Return the loaded configuration as a dictionary
    return config