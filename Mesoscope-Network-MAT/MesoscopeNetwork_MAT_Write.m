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


%% write_mat_to_tif
% Save a 3D matrix as a multi-page TIFF file without compression.
%
% Parameters:
%   mat (3D array): A 3D matrix where each slice along the third dimension
%                   represents a separate image that will be saved as a 
%                   page in the TIFF file.
%   filename (string): The desired name of the output TIFF file, including 
%                      the file extension (e.g., 'output.tif').
% Function to save a 3D matrix as a multi-page TIFF file without compression.
% 
% Parameters:
%   mat (3D array): A 3D matrix where each slice along the third dimension
%                   represents a separate image that will be saved as a 
%                   page in the TIFF file.
%   filename (string): The desired name of the output TIFF file, including 
%                      the file extension (e.g., 'output.tif').
%
% The function uses the imwrite function to write the images to a TIFF file.
% The first slice of the matrix is written to the file, and subsequent
% slices are appended to create a multi-page TIFF.

function write_mat_to_tif(mat, filename)

    % Save the first slice of the 3D matrix to the specified filename as a TIFF file
    % with no compression.
    imwrite(mat(:,:,1), filename, 'compression', 'none');
    
    % Get the size of the third dimension (number of slices) of the matrix.
    [~, ~, nz] = size(mat);
    
    % Loop through each slice starting from the second slice to the last.
    for i = 2:nz
        % Append each subsequent slice to the TIFF file without compression.
        imwrite(mat(:,:,i), filename, 'compression', 'none', 'writemode', 'append'); 
    end

end