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
Extracts regions of interest (ROIs) around the local maxima in the input image after applying Gaussian filtering.

Parameters:
    im (np.ndarray): The image to extract ROIs from.
    ROIsize (tuple): Multidimensional size of the ROI to extract around each maximum. If fewer dimensions are provided, the others will not be extracted.
    sigma (float or list): Size of the Gaussian filter kernel; if None, no filtering is applied.
    threshold_rel (float): Relative threshold to extract peaks.
    alternateImg (np.ndarray): An alternate image to extract from instead of `im`.
    kernel (tuple): Size of the kernel used for local maximum detection.
    borderDist (tuple): Minimum distance to keep around maxima.
    FOV (tuple): Field of view parameters to filter detected coordinates.

Returns:
    tuple: (stacked n-dimensional ROIs and their center coordinates).
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndimage
from scipy.spatial import distance
from scipy import cluster as cluster
import warnings
import numbers


def extractMultiPeaks(im, ROIsize, sigma=None, threshold_rel=None, alternateImg=None, kernel=(3,3,3), borderDist=None, FOV=None):
    """
    Extracts regions of interest (ROIs) around the local maxima in the input image 
    after applying Gaussian filtering.

    :param im: np.ndarray, The image to extract ROIs from.
    :param ROIsize: tuple, Multidimensional size of the ROI to extract around each maximum. 
                    If fewer dimensions are provided, the others will not be extracted.
    :param sigma: float or list, Size of the Gaussian filter kernel; if None, no filtering is applied.
    :param threshold_rel: float, Relative threshold to extract peaks.
    :param alternateImg: np.ndarray, An alternate image to extract from instead of `im`.
    :param kernel: tuple, Size of the kernel used for local maximum detection.
    :param borderDist: tuple, Minimum distance to keep around maxima.
    :param FOV: tuple, Field of view parameters to filter detected coordinates.
    :return: tuple, (stacked n-dimensional ROIs and their center coordinates).
    """
    # Apply Gaussian filter if sigma is provided and non-zero
    if sigma is not None and np.linalg.norm(sigma) > 0:
        im2 = gaussian_filter(im, sigma)
    else:
        im2 = im
    
    # Find local maxima coordinates in the filtered image
    coordinates = localMax(im2, threshold_rel=threshold_rel, kernel=kernel)
    coordinates = np.array(coordinates)
    
    # Filter coordinates based on border distance if provided
    if coordinates.size > 0:
        if borderDist is not None:        
            borderDist = np.array(borderDist)
            inBorder = np.all(coordinates - borderDist >= 0, axis=1) & np.all(im.shape - coordinates - borderDist >= 0, axis=1)
            coordinates = coordinates[inBorder, :]
        
        # Filter coordinates based on the field of view if provided
        if FOV is not None:        
            fov = np.array(FOV)
            coord_r = (coordinates[:, -1] - fov[1]) ** 2 + (coordinates[:, -2] - fov[0]) ** 2
            inFov = coord_r < (fov[2] ** 2)
            coordinates = coordinates[inFov, :]

    # Use an alternate image if provided
    if alternateImg is not None:
        im = alternateImg
    
    # Round coordinates to integers for indexing
    centers = np.round(coordinates).astype(np.int32)
    
    # Adjust the dimensions of centers based on ROIsize
    if len(ROIsize) < centers.shape[-1]:
        centers = centers[:, -len(ROIsize):]
    
    # Extract ROIs if any coordinates are valid
    if coordinates.size > 0:
        ROIs = multiROIExtract(im, centers, ROIsize=ROIsize)
    else:
        ROIs = None
    
    return ROIs, centers


def extractMultiPeaks_smlm(im, ROIsize, sigma=None, threshold_rel=None, alternateImg=None, kernel=(3,3,3), borderDist=None, min_dist=None, FOV=None):
    """
    Extracts ROIs around the local maxima in the input image using a specific method 
    suited for single-molecule localization microscopy (SMLM).

    :param im: np.ndarray, The image to extract ROIs from.
    :param ROIsize: tuple, Multidimensional size of the ROI to extract around each maximum.
    :param sigma: float or list, Size of the Gaussian filter kernel; if None, no filtering is applied.
    :param threshold_rel: float, Relative threshold to extract peaks.
    :param alternateImg: np.ndarray, An alternate image to extract from instead of `im`.
    :param kernel: tuple, Size of the kernel used for local maximum detection.
    :param borderDist: tuple, Minimum distance to keep around maxima.
    :param min_dist: float, Minimum distance to maintain between detected maxima.
    :param FOV: tuple, Field of view parameters to filter detected coordinates.
    :return: tuple, (stacked n-dimensional ROIs and their center coordinates).
    """
    # Apply Gaussian filter with a specific method to enhance SMLM data
    if sigma is not None and np.linalg.norm(sigma) > 0:
        im2 = gaussian_filter(im, list(np.array(sigma) * 0.75)) - gaussian_filter(im, sigma)
    else:
        im2 = im
    
    # Find local maxima coordinates in the filtered image
    coordinates = localMax(im2, threshold_rel=threshold_rel, kernel=kernel)
    coordinates = np.array(coordinates)
    
    if coordinates.size > 0:
        # Filter coordinates for border distance
        if borderDist is not None:        
            borderDist = np.array(borderDist)
            inBorder = np.all(coordinates - borderDist >= 0, axis=1) & np.all(im.shape - coordinates - borderDist >= 0, axis=1)
            coordinates = coordinates[inBorder, :]
        
        # Filter coordinates for field of view
        if FOV is not None:        
            fov = np.array(FOV)
            coord_r = (coordinates[:, -1] - fov[1]) ** 2 + (coordinates[:, -2] - fov[0]) ** 2
            inFov = coord_r < (fov[2] ** 2)
            coordinates = coordinates[inFov, :]
    
    # Use an alternate image if provided
    if alternateImg is not None:
        im = alternateImg
    
    # Round coordinates to integers for indexing
    centers = np.round(coordinates).astype(np.int32)
    
    # Adjust dimensions based on ROIsize
    if len(ROIsize) < centers.shape[-1]:
        centers = centers[:, -len(ROIsize):]
    
    # Extract ROIs if any coordinates are valid
    if coordinates.size > 0:
        ROIs = multiROIExtract_smlm(im, centers, ROIsize=ROIsize)
    else:
        ROIs = None
    
    return ROIs, centers
    

def localMax(img, threshold_rel=None, kernel=(3,3,3)):
    """
    Identifies the local maxima in an image based on a specified threshold.

    :param img: np.ndarray, The input image to search for local maxima.
    :param threshold_rel: float, Relative threshold to filter maxima based on intensity.
    :param kernel: tuple, Size of the kernel used for local maximum detection.
    :return: list, Coordinates of the detected local maxima.
    """
    # Apply maximum filter to identify local maxima
    imgMax = ndimage.maximum_filter(img, size=kernel)
    imgMax = (imgMax == img) * img  # Keep only the local maxima values
    mask = imgMax == img
    
    # Apply threshold if specified
    if threshold_rel is not None:
        thresh = np.quantile(img[mask], 1 - 1e-4) * threshold_rel
        labels, num_labels = ndimage.label(imgMax > thresh)
    else:
        labels, num_labels = ndimage.label(imgMax)

    # Get the positions of the maxima
    coords = ndimage.measurements.center_of_mass(img, labels=labels, index=np.arange(1, num_labels + 1))

    return coords


def multiROIExtract(im, centers, ROIsize):
    """
    Extracts multiple ROIs from the image based on a list of center coordinates.

    :param im: np.ndarray, The image to extract ROIs from.
    :param centers: list or np.ndarray, List of center coordinates for ROIs.
    :param ROIsize: tuple, Multidimensional size of the ROI to extract.
    :return: np.ndarray, Stacked extractions of ROIs.
    """
    listOfROIs = []
    for centerpos in centers:
        # Adjust center position for dimensions
        if len(centerpos) > im.ndim:
            centerpos = centerpos[-im.ndim:]
        if len(ROIsize) < len(centerpos):
            centerpos = centerpos[-len(ROIsize):]

        # Extract the ROI using the specified center position
        myROI = extract(im, ROIsize=ROIsize, centerpos=centerpos)
        listOfROIs.append(myROI)
    
    return np.stack(listOfROIs)


def multiROIExtract_smlm(im, centers, ROIsize):
    """
    Extracts multiple ROIs specifically for SMLM data based on center coordinates.

    :param im: np.ndarray, The image to extract ROIs from.
    :param centers: list or np.ndarray, List of center coordinates for ROIs.
    :param ROIsize: tuple, Multidimensional size of the ROI to extract.
    :return: np.ndarray, Stacked extractions of ROIs.
    """
    listOfROIs = []
    for centerpos in centers:
        # Extract the ROI using the specified center position
        myROI = im[ROIcoords(centerpos, ROIsize, im.ndim)]
        listOfROIs.append(myROI)
    
    return np.stack(listOfROIs)


def combine_close_cor(centers, min_dist):
    """
    Combines close coordinates based on a minimum distance.

    :param centers: np.ndarray, Array of center coordinates.
    :param min_dist: float, Minimum distance to maintain between combined coordinates.
    :return: np.ndarray, New coordinates with combined close centers.
    """
    # Calculate pairwise distances between centers
    dis = distance.pdist(centers)
    link = cluster.hierarchy.linkage(dis, 'complete')
    Tc = cluster.hierarchy.fcluster(link, t=min_dist, criterion='distance')
    
    cor = np.zeros((np.max(Tc), 2), dtype=np.int32)
    for t in range(0, np.max(Tc)):
        maskT = (Tc == (t + 1))
        # Combine centers if there are multiple close ones
        if np.sum(maskT) > 1:
            cor[t] = np.mean(centers[maskT], axis=0)
        else:
            cor[t] = centers[maskT]

    return cor


def extract(img, ROIsize=None, centerpos=None, PadValue=0.0, checkComplex=True):
    """
    Extracts a portion of an n-dimensional array based on specified ROI size and center position.

    :param img: np.ndarray, Input image from which to extract the ROI.
    :param ROIsize: tuple, Size of the region of interest to extract.
    :param centerpos: tuple, Center of the ROI in the source image.
    :param PadValue: float, Value to assign to the padded area; if None, no padding is performed.
    :param checkComplex: bool, Flag to check if the input image has complex values.
    :return: np.ndarray, The extracted ROI from the input image.
    """
    # Check if the image is complex
    if checkComplex:
        if np.iscomplexobj(img):
            raise ValueError(
                "Found complex-valued input image. For Fourier-space extraction use extractFt, which handles the borders or use checkComplex=False as an argument to this function")

    mysize = img.shape

    # If no ROIsize is provided, use the image size
    if ROIsize is None:
        ROIsize = mysize
    else:
        ROIsize = expanddimvec(ROIsize, len(mysize), mysize)

    mycenter = [sd // 2 for sd in mysize]
    # If no center position is provided, use the center of the image
    if centerpos is None:
        centerpos = mycenter
    else:
        centerpos = coordsToPos(expanddimvec(centerpos, img.ndim, othersizes=mycenter), mysize)

    # Extract the ROI using calculated coordinates
    res = img[ROIcoords(centerpos, ROIsize, img.ndim)]
    if PadValue is None:
        return res
    else:
        # Perform padding if needed
        pads = [(max(0, ROIsize[d] // 2 - centerpos[d]), 
                  max(0, centerpos[d] + ROIsize[d] - mysize[d] - ROIsize[d] // 2)) for d in range(img.ndim)]
        resF = np.pad(res, tuple(pads), 'constant', constant_values=PadValue)
        return resF


def expanddimvec(shape, ndims, othersizes=None, trailing=False, value=1):
    """
    Expands an n-tuple to the necessary number of dimensions by inserting leading or trailing dimensions.

    :param shape: tuple, Input shape to expand.
    :param ndims: int, Target number of dimensions.
    :param othersizes: tuple, Sizes to use for expansion if provided.
    :param trailing: bool, If True, append trailing dimensions rather than leading.
    :param value: int, Value to fill in for new dimensions.
    :return: tuple, Expanded shape.
    """
    if shape is None:
        return None
    if isinstance(shape, numbers.Number):
        shape = (shape,)
    else:
        shape = tuple(shape)
    
    # Calculate number of missing dimensions
    missingdims = ndims - len(shape)
    
    if missingdims > 0:
        if othersizes is None:
            if trailing:
                return shape + (missingdims) * (value,)
            else:
                return (missingdims) * (value,) + shape
        else:
            if trailing:
                return shape + tuple(othersizes[-missingdims::])
            else:
                return tuple(othersizes[0:missingdims]) + shape
    else:
        return shape[-ndims:]


def coordsToPos(coords, ashape):
    """
    Converts a coordinate vector to positive numbers using a given shape.

    :param coords: list or np.ndarray, Coordinates that may contain negative values.
    :param ashape: list or tuple, Shape defining the limits for coordinates.
    :return: list, Converted coordinates in the positive range.
    """
    mylen = len(coords)
    assert(mylen == len(ashape))
    return [coords[d] + (coords[d] < 0) * ashape[d] for d in range(mylen)]


def ROIcoords(center, asize, ndim=None):
    """
    Constructs a coordinate vector for array access based on the center and ROI size.

    :param center: list or tuple, Center coordinates for the ROI.
    :param asize: tuple, Size of the ROI to extract.
    :param ndim: int, Total number of dimensions of the array.
    :return: tuple, A tuple of slices for indexing the ROI.
    """
    if ndim is None:
        ndim = len(center)

    slices = []
    
    # Handle leading dimensions if necessary
    if ndim > len(center):
        slices = (ndim - len(center)) * slice(None)
    
    for d in range(ndim - len(center), ndim):  # Only specify the last dimensions
        asp = asize[d]
        if asp < 0:
            raise ValueError("asize has to be >= 0")
        astart = center[d] - asp // 2
        astop = astart + asp
        slices.append(slice(max(astart, 0), max(astop, 0)))

    return tuple(slices)


def expanddim(img, ndims, trailing=None):
    """
    Expands an n-dimensional image by inserting leading dimensions.

    :param img: np.ndarray, Input image to expand.
    :param ndims: int, Number of dimensions to expand to.
    :param trailing: bool, If True, append trailing dimensions rather than leading.
    :return: np.ndarray, Reshaped image with added dimensions.
    """
    if trailing is None:
        trailing = ndims < 0

    if ndims < 0:
        ndims = -ndims
    res = np.reshape(img, expanddimvec(img.shape, ndims, None, trailing))

    return res


def unifysize(mysize):
    """
    Converts various types of input sizes into a list.

    :param mysize: any, Input size which could be a list, tuple, or ndarray.
    :return: list, A list representation of the size.
    """
    if isinstance(mysize, list) or isinstance(mysize, tuple) or isinstance(mysize, np.ndarray):
        return list(mysize)
    else:
        return list(mysize.shape)


def ones(s, dtype=None, order='C', ax=None):
    """
    Creates an array of ones with the specified shape.

    :param s: tuple or np.ndarray, Shape of the array to create.
    :param dtype: data-type, Optional; desired data-type for the array.
    :param order: {'C', 'F'}, Optional; order in which the array will be stored in memory.
    :param ax: int or None, Optional; specifies the dimension to expand to.
    :return: np.ndarray, Array of ones with the specified shape.
    """
    if isnp(s):
        s = s.shape
    res = np.ones(s, dtype, order)
    if ax is not None:
        res = castdim(res, wanteddim=ax)
    return res


def isnp(animg):
    """
    Checks if the input is a numpy ndarray.

    :param animg: any, Input to be checked.
    :return: bool, True if the input is a numpy ndarray, False otherwise.
    """
    return isinstance(animg, np.ndarray)


def castdim(img, ndims=None, wanteddim=0):
    """
        expands a 1D image to the necessary number of dimension casting the dimension to a wanted one
        it orients a vector along the -wanteddim direction
        ----------
        img: input image to expand
        ndims: number of dimensions to expand to
        wanteddim: number that the one-D axis should end up in (default:0)
    """
    return np.reshape(img, castdimvec(img.shape, ndims, wanteddim))

def castdimvec(mysize, ndims=None, wanteddim=0):
    """
        expands a shape tuple to the necessary number of dimension casting the dimension to a wanted one
        ----------
        img: input image to expand
        ndims: number of dimensions to expand to. If None, wanteddim is used to determine the maximal size of dims
        wanteddim: number that the one-D axis should end up in (default:0)

        see also:
        expanddimvec
    """
    mysize = tuple(mysize)
    if ndims is None:
        if wanteddim >= 0:
            ndims = wanteddim + 1
        else:
            ndims = - wanteddim
    if wanteddim<0:
        wanteddim = ndims+wanteddim
    if wanteddim+len(mysize) > ndims:
        raise ValueError("castdim: ndims is smaller than requested total size including the object to place.")
    newshape = wanteddim*(1,)+mysize+(ndims-wanteddim-len(mysize))*(1,)
    return newshape




def zeros(s, dtype=None, order='C',  ax=None):
    if isnp(s):
        s = s.shape
    res = np.zeros(s, dtype, order)
    if ax is not None:
        res = castdim(res, wanteddim=ax)
    return res

def dimToPositive(dimpos,ndims):
    """
        converts a dimension position to a positive number using a given length.

        dimpos: dimension to adress
        ndims: total number of dimensions

    """
    return dimpos+(dimpos<0)*ndims *ndims 