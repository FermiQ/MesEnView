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

%% read_tif_to_mat
% Reads a multi-page TIFF file and stores its pages in a 3D matrix.
%
% Syntax:
%   y = read_tif_to_mat(filename, verbose)
%
% Inputs:
%   filename - A string representing the path to the TIFF file that needs to be read.
%   verbose - A boolean flag that determines whether to display progress information
%             while reading the pages of the TIFF file.
%
% Outputs:
%   y - A 3D matrix where each slice along the third dimension corresponds to a
%       different page (or frame) of the TIFF file.
% read_tif_to_mat - Reads a multi-page TIFF file and stores its pages in a 3D matrix
%
% Syntax: 
%   y = read_tif_to_mat(filename, verbose)
%
% Inputs:
%   filename - A string representing the path to the TIFF file that needs to be read.
%              The function utilizes this path to access the file and read its content.
%
%   verbose - A boolean flag (true/false) that determines whether to display progress 
%             information to the command window while reading the pages of the TIFF file. 
%             If true, the function will output the current page number and the total 
%             number of pages being processed.
%
% Outputs:
%   y - A 3D matrix where each slice along the third dimension corresponds to a different 
%       page (or frame) of the TIFF file. The dimensions of y will be [height, width, numPages],
%       where height and width are the dimensions of each page/image in the TIFF file, 
%       and numPages is the total number of pages in the file.

function y = read_tif_to_mat(filename, verbose)

% Get the total number of pages in the TIFF file by using imfinfo, which returns
% metadata information about the file. The size of the first dimension indicates 
% the number of images (or pages) in the TIFF.
nz = size(imfinfo(filename), 1);

% Loop through each page of the TIFF file, from 1 to the total number of pages (nz).
for i = 1:nz
   % Read the i-th page of the TIFF file using imread and assign it to the 3D matrix y.
   % Each slice of y along the third dimension corresponds to an individual page.
   y(:,:,i) = imread(filename, i); 
   
   % If the verbose flag is true, display the current page number and the total number of pages.
   if verbose
       disp([num2str(i), '/', num2str(nz)]);
   end
end

end