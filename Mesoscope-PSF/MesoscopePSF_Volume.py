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

import numpy as np
import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter
from .PSFInterface_file import PSFInterface
from ..data_representation.PreprocessedImageDataInterface_file import PreprocessedImageDataInterface
from ..loss_functions import mse_real
from .. import utilities as im
from .. import imagetools as nip

class PSFVolumeBased(PSFInterface):
    """
    PSF class that uses a 3D volume to describe the PSF.
    Should only be used with single-channel data.
    """
    def __init__(self, options = None) -> None:
        self.parameters = None
        self.data = None
        self.bead_kernel = None
        self.options = options
        self.default_loss_func = mse_real
        return

    def calc_initials(self, data: PreprocessedImageDataInterface, start_time=None):
        """
        Provides initial values for the optimizable varibales for the fitter class.
        """
        self.data = data
        _, rois, _, _ = self.data.get_image_data()

        init_positions = np.zeros((rois.shape[0], len(rois.shape)-1))
        init_backgrounds = np.array(np.min(gaussian_filter(rois, [0, 2, 2, 2]), axis=(-3, -2, -1), keepdims=True))
        init_intensities = np.sum(rois - init_backgrounds, axis=(-2, -1), keepdims=True)
        init_intensities = np.mean(init_intensities,axis=1,keepdims=True)
        
        N = rois.shape[0]
        Nz = rois.shape[-3]
        self.calpupilfield('scalar',Nz)
        self.gen_bead_kernel(isVolume=True)

        #self.weight = np.array([np.median(init_intensities)*1, 10, 0.1, 0.1],dtype=np.float32)
        #weight = [5e4,20] + list(np.array([0.1,0.2])/np.median(init_intensities)*2e4)
        init_backgrounds[init_backgrounds<0.1] = 0.1
        bgmean = np.median(init_backgrounds)
        wI = np.lib.scimath.sqrt(np.median(init_intensities))
        weight = [1000*wI,bgmean] + list(np.array([1,1])*40/wI)
        self.weight = np.array(weight,dtype=np.float32)
        init_psf_model = np.zeros(rois[0].shape)+0.002/self.weight[3]
        init_backgrounds = np.ones((N,1,1,1),dtype = np.float32)*np.median(init_backgrounds,axis=0, keepdims=True) / self.weight[1]
        gxy = np.zeros((N,2),dtype=np.float32) 
        gI = np.ones((N,Nz,1,1),dtype = np.float32)*init_intensities
        #gI = np.ones((N,Nz,1,1),dtype = np.float32)*np.mean(init_intensities,keepdims=True)
        self.varinfo = [dict(type='Nfit',id=0),
                   dict(type='Nfit',id=0),
                   dict(type='Nfit',id=0),
                   dict(type='shared'),
                   dict(type='Nfit',id=0)]
        
        if self.options.model.var_photon:
            init_Intensity = gI/self.weight[0]
        else:
            init_Intensity = init_intensities / self.weight[0]
        return [init_positions.astype(np.float32),
                init_backgrounds.astype(np.float32),
                init_Intensity.astype(np.float32),
                init_psf_model.astype(np.float32),
                gxy],start_time
        
    def calc_forward_images(self, variables):
        """
        Calculate forward images from the current guess of the variables.
        Shifting is done by Fourier transform and applying a phase ramp.
        """

        pos, backgrounds, intensities, I_model, gxy = variables

        I_model = tf.complex(I_model,0.0)
        I_otfs = im.fft3d(I_model*self.weight[3])*self.bead_kernel #*tf.complex(intensities*self.weight[0],0.0) 
        pos = tf.complex(tf.reshape(pos,pos.shape+(1,1,1)),0.0)
        I_res = im.ifft3d(I_otfs*self.phaseRamp(pos))

        psf_fit = tf.math.real(I_res)
        if self.options.model.estimate_drift:
            gxy = gxy*self.weight[2]
            psf_shift = self.applyDrfit(psf_fit,gxy)
            forward_images = psf_shift*intensities*self.weight[0] + backgrounds*self.weight[1]
        else:
            forward_images = psf_fit*intensities*self.weight[0] + backgrounds*self.weight[1]

        return forward_images

    def postprocess(self, variables):
        """
        Applies postprocessing to the optimized variables. In this case calculates
        real positions in the image from the positions in the roi. Also, normalizes
        psf and adapts intensities and background accordingly.
        """
        positions, backgrounds, intensities, I_model,gxy = variables
        
        I_model = I_model*self.weight[3]
        I_model = I_model.astype(np.complex64)
        I_model_bead = np.real(im.ifft3d(im.fft3d(I_model)*self.bead_kernel))
        I_model = I_model.astype(np.float32)
        z_center = (I_model.shape[-3] - 1) // 2
        images, _, centers, _ = self.data.get_image_data()
        centers_with_z = np.concatenate((np.full((centers.shape[0], 1), z_center), centers[:,-2:]), axis=1)

        global_positions = centers_with_z - positions
            
        return [global_positions.astype(np.float32),
                backgrounds*self.weight[1], # already correct
                intensities*self.weight[0], # already correct
                I_model_bead,
                I_model,
                gxy*self.weight[2],
                np.flip(I_model,axis=-3),
                variables] # already correct

    def res2dict(self,res):
        res_dict = dict(pos=res[0],
                        I_model_bead =res[3],
                        I_model = res[4],
                        bg=np.squeeze(res[1]),
                        intensity=np.squeeze(res[2]),
                        drift_rate=res[5],
                        I_model_reverse = res[6],
                        offset=np.min(res[4]),
                        cor_all = self.data.centers_all,
                        cor = self.data.centers)

        return res_dict