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

from abc import ABCMeta, abstractmethod
import pickle

import numpy as np

class FitterInterface:
    """
    Interface that ensures consistency and compatability between all old and new implementations of data classes, fitters and psfs.
    Classes implementing this interafce define the fitting procedure. They combine image data and a psf model to do this (and also optimizer???).
    For example, the procedure for fitting a psf for a single-channel experiment can be very different from one for a multi-channel experiment.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def objective(self, variables: list) -> list:
        """
        Defines the objective that is optimized. In general, calculates the loss from forward images and real images
        and return the loss and its graient.
        """
        raise NotImplementedError("You need to implement a 'objective' method in your fitter class.")

    @abstractmethod
    def learn_psf(self, variables: list=None) -> list:
        """
        Is called by the user and defines the procedure of the psf learning.
        Returns a list containing the results, e.g., a psf object, the final postioins, intensities and backgrounds.
        """
        raise NotImplementedError("You need to implement a 'learn_psf' method in your psf class.")

    def save(self, filename: str):
        """
        Save object to file.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(filename: str):
        """
        Load object from file.
        """
        with open(filename, "rb") as f:
            self = pickle.load(f)
        return self