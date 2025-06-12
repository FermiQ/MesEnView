%{
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
%}

%% cm2_forward_gpu
%
% Performs a convolution operation on a 3D input array with corresponding Point
% Spread Functions (PSFs) using GPU acceleration.
%
% SYNTAX:
%   y = cm2_forward_gpu(x, psfs, verbose)
%
% INPUTS:
%   x (3D array): Input data of size (M, N, K), where (M, N) are the dimensions
%                 of each 2D slice and K is the number of slices.
%   psfs (3D array): PSFs corresponding to each slice in x, of the same size
%                    as x.
%   verbose (boolean): If true, provides progress updates during processing.
%
% OUTPUTS:
%   y (2D array): Resulting 2D array after convolution operations, aggregated
%                 over all slices.
function y = cm2_forward_gpu(x, psfs, verbose)
    % cm2_forward_gpu performs a convolution operation on a 3D input array
    % with corresponding Point Spread Functions (PSFs) using GPU acceleration.
    %
    % Parameters:
    %   x (3D array): Input data of size (M, N, K), where (M, N) are the 
    %                 dimensions of each 2D slice and K is the number of slices.
    %   psfs (3D array): PSFs corresponding to each slice in x, of the same 
    %                    size as x.
    %   verbose (boolean): If true, provides progress updates during processing.
    %
    % Returns:
    %   y (2D array): Resulting 2D array after convolution operations, 
    %                  aggregated over all slices.

    % Define a function to perform a 2D forward Fast Fourier Transform (FFT)
    F2D = @(x) fftshift(fft2(ifftshift(x)));
    
    % Define a function to perform a 2D inverse Fast Fourier Transform (FFT)
    Ft2D = @(x) fftshift(ifft2(ifftshift(x)));
    
    % Define a function to pad a 2D array to enhance processing
    pad2d = @(x) padarray(x, 0.5 * size(x));
    
    % Define a function to crop a 2D array to extract the central region
    crop2d = @(x) x(1 + size(x, 1) / 4:size(x, 1) / 4 * 3, ...
                    1 + size(x, 2) / 4:size(x, 2) / 4 * 3);
    
    % Define a function to perform 2D convolution with cropping
    conv2d = @(obj, psf) crop2d(real(Ft2D(F2D(pad2d(obj)) .* F2D(pad2d(psf)))));
    
    % Initialize the output array y to zeros with the same size as the first 
    % two dimensions of x.
    y = zeros(size(x, 1), size(x, 2));
    
    % Loop through each slice of the input data
    for i = 1:size(x, 3)
        % Perform convolution on the current slice and add to the output
        y = y + conv2d(gpuArray(x(:, :, i)), gpuArray(psfs(:, :, i)));
        
        % If verbose mode is enabled, display the current slice index
        if verbose
            disp(i);
        end
    end
end