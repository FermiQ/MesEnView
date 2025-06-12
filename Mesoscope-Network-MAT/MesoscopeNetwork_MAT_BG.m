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


%% bg_removal
% This function performs background removal from an input image using
% morphological opening. It computes the background and then subtracts it
% from the original image to enhance the foreground features.
%
% @param image A grayscale or binary image from which the background is to be removed.
%              The input image is expected to be a 2D matrix.
% @param kernel_size A scalar value that defines the size of the structuring element
%                    (disk shape) used in the morphological operation. It controls the
%                    degree of background smoothing; larger values result in more
%                    aggressive background removal.
% @return img The resulting image after background subtraction. This image
%             highlights the foreground objects by removing the estimated
%             background.
% @return bg The estimated background image obtained using morphological opening.
%             This image represents the smoothed version of the original image.
function [img,bg] = bg_removal(image, kernel_size)
% bg_removal - This function performs background removal from an input image
% using morphological opening. It computes the background and then subtracts
% it from the original image to enhance the foreground features.
%
% Syntax: [img, bg] = bg_removal(image, kernel_size)
%
% Inputs:
%   image      - A grayscale or binary image from which the background is to be removed.
%                The input image is expected to be a 2D matrix.
%   kernel_size - A scalar value that defines the size of the structuring element
%                (disk shape) used in the morphological operation. It controls the
%                degree of background smoothing; larger values result in more
%                aggressive background removal.
%
% Outputs:
%   img        - The resulting image after background subtraction. This image
%                highlights the foreground objects by removing the estimated
%                background.
%   bg         - The estimated background image obtained using morphological opening.
%                This image represents the smoothed version of the original image.
%
% Example: 
%   [foreground, background] = bg_removal(input_image, 5);

% Perform morphological opening on the input image using a disk-shaped structuring element.
% The structuring element has a radius defined by kernel_size.
bg = imopen(image, strel('disk', kernel_size, 8));

% Subtract the estimated background from the original image to obtain the foreground.
img = imsubtract(image, bg);
end