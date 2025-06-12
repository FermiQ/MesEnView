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
This module defines various constants used throughout the SBRNet project.

The constants include:
- `CM2_SIZE`: The size of the camera matrix in pixels.
- `NUM_VIEWS`: The number of camera views (angles) to be used.
- `FOCUS_LOC`: An array of focal locations (in pixels) for each view.
- `NUM_SLICES`: The number of slices in the refocus volume.
- `view_combos`: A list of combinations of views to be used for rendering/refocusing.
"""
import numpy as np  # Importing the NumPy library for numerical operations
from typing import List  # Importing List type for type hints

# Constants defining the size of the camera matrix in pixels
CM2_SIZE = [2076, 3088]

# Constant defining the number of views (camera angles) to be used
NUM_VIEWS = 9

# Array defining the focal locations (in pixels) for each view
FOCUS_LOC = np.array(
    [
        [406, 909],    # Focal location for view 0
        [407, 1545],   # Focal location for view 1
        [405, 2175],   # Focal location for view 2
        [1037, 911],   # Focal location for view 3
        [1037, 1544],  # Focal location for view 4 (center)
        [1037, 2176],  # Focal location for view 5
        [1675, 911],   # Focal location for view 6
        [1675, 1543],  # Focal location for view 7
        [1675, 2173],  # Focal location for view 8
    ]
)

# Constant defining the number of slices in the refocus volume
NUM_SLICES = 24  

# List defining combinations of views to be used for rendering/refocusing
# Each sublist contains the indices of the views to be combined
# The arrangement corresponds to a grid representation of views:
# 0 | 1 | 2
# ---------
# 3 | 4 | 5
# ---------
# 6 | 7 | 8
view_combos = [
    [4],                            # Focusing only on the center view
    [4, 5],                         # Center view and the right view
    [0, 4],                         # Left view and center view
    [0, 1, 4],                      # Left, top view and center view
    [0, 4, 8],                      # Left view, center view and bottom view
    [1, 3, 5, 7],                   # Top left, center left, center right, and bottom right views
    [0, 2, 6, 8],                   # Left, right, bottom left, and bottom right views
    [1, 3, 4, 5, 7],                # Top, center left, center, center right, and bottom right views
    [0, 2, 4, 6, 8],                # Left, right, center, bottom left, bottom right views
    [0, 2, 4, 5, 6, 8],             # Left, right, center, right center, bottom left, bottom right views
    [0, 2, 3, 4, 5, 6, 8],          # Left, right, top left, center, right center, bottom left, bottom right views
    [0, 1, 2, 3, 5, 6, 7, 8],       # All views except the center view
    [0, 1, 3, 4, 5, 6, 7, 8],       # All views except the top right view (aberrated)
    [0, 2, 3, 4, 5, 6, 7, 8],       # All views except the top middle view
]