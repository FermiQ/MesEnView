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

%% enhance_contrast_high_mat
% Enhances the contrast of an input matrix by clipping high intensity values.
%
% Syntax:
%   [output, val_max] = enhance_contrast_high_mat(data, high_threshold)
%
% Inputs:
%   data (numeric matrix or array): The original intensity values to be processed.
%   high_threshold (scalar, optional): The cumulative distribution function (CDF) threshold for clipping high intensity values. Defaults to 0.9999 if not provided.
%
% Outputs:
%   output (numeric matrix): The modified matrix with high intensity values clipped to enhance contrast.
%   val_max (scalar): The maximum intensity level that was clipped during the enhancement process.
function [output, val_max] = enhance_contrast_high_mat(data, high_threshold)
% enhance_contrast_high_mat - Enhances the contrast of an input matrix by clipping high intensity values.
%
% Syntax: [output, val_max] = enhance_contrast_high_mat(data, high_threshold)
%
% Inputs:
%   data - A numeric matrix or array containing the original intensity values, which will be processed 
%          to enhance contrast. The data is converted to double precision for accurate calculations.
%   high_threshold - A scalar value representing the cumulative distribution function (CDF) threshold 
%                    for clipping high intensity values. If not provided, defaults to 0.9999.
%
% Outputs:
%   output - A numeric matrix of the same size as the input 'data', where high intensity values have 
%            been clipped to enhance contrast.
%   val_max - A scalar value representing the maximum intensity level that was clipped during the 
%             enhancement process.

% Convert the input data to double precision for accurate calculations
data = double(data);

% Check if only one input argument was provided; if so, set the default high threshold
if nargin == 1
    high_threshold = 0.9999; % Default threshold for high intensity clipping
end

% Create a histogram of the flattened data with a specified number of bins (2^14)
hist_obj = histogram(data(:), 2^14);

% Extract the edges and counts of the histogram bins
bin_edges = hist_obj.BinEdges; % The edges of the histogram bins
bin_counts = hist_obj.BinCounts; % The counts of pixels in each bin

% Calculate the probability density function (PDF) from the bin counts
pdf = bin_counts ./ sum(bin_counts(:)); 

% Compute the cumulative distribution function (CDF) from the PDF
cdf = cumsum(pdf); 

% Find the index of the bin where the CDF is closest to the high threshold
[~, idx] = min(abs(cdf - high_threshold));

% Calculate the high clipping value as the average of the bin edge at the found index and the next one
high_clip_val = 0.5 * (bin_edges(idx) + bin_edges(idx + 1));

% Clip the data values that are greater than or equal to the high clipping value
data(data >= high_clip_val) = high_clip_val;

% Store the high clipping value to output
val_max = high_clip_val; 

% Set the output to the modified data
output = data; 

% Close all histogram figures (if any were opened)
close all; 
end