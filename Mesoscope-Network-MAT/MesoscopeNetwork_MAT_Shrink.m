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


%% average_shrink
% Reduces the size of the input matrix by averaging blocks of a specified size.
%
% Syntax:
%   output = average_shrink(input, shrink_size)
%
% Inputs:
%   input (numeric array) - A 2D matrix to be shrunk. The function will average
%                          its values over non-overlapping blocks.
%   shrink_size (integer) - The size of each block over which the average will
%                          be calculated. It should be a divisor of both the
%                          number of rows and columns of the input matrix.
%
% Output:
%   output (numeric array) - A 2D matrix that contains the averaged values of
%                           the input matrix, reduced in size according to the
%                           shrink_size parameter.
function output = average_shrink(input, shrink_size)
% average_shrink - Reduces the size of the input matrix by averaging 
% blocks of a specified size.
%
% Syntax: output = average_shrink(input, shrink_size)
%
% Inputs:
%   input      - A 2D matrix (numeric array) to be shrunk. The function 
%                will average its values over non-overlapping blocks.
%   shrink_size - A positive integer that defines the size of each block 
%                 over which the average will be calculated. It should 
%                 be a divisor of both the number of rows and columns of 
%                 the input matrix (after padding if necessary).
%
% Output:
%   output     - A 2D matrix that contains the averaged values of the 
%                input matrix, reduced in size according to the 
%                shrink_size parameter.

% Get the size of the input matrix
[rows,cols] = size(input);

% Check if the rows and columns of the input matrix are divisible by 
% the shrink_size. If not, pad the input matrix to make it divisible.
if mod(rows,shrink_size) || mod(cols,shrink_size)
    input = padarray(input,...
        [shrink_size-mod(rows,shrink_size),...
        shrink_size-mod(cols,shrink_size)],'post');
end

% Update the size of the padded input matrix
[rows,cols] = size(input);

% Create a averaging kernel of size shrink_size x shrink_size
kernel = ones(shrink_size);
% Normalize the kernel so that the sum of its elements equals 1
kernel = kernel./sum(kernel(:));

% Apply 2D convolution to the input matrix with the averaging kernel
output = conv2(input,kernel);

% Select every shrink_size-th element from both dimensions to create the 
% downsampled output
output = output(shrink_size:shrink_size:rows,shrink_size:shrink_size:cols);

% The following block of code is commented out. It provides an alternative
% method to achieve the same functionality using a nested loop approach.
% This method calculates the mean of each block manually.
% output = zeros(rows/shrink_size, cols/shrink_size);
% for i = 1:size(output,1)
%     for j = 1:size(output,2)
%         block = input((i-1)*shrink_size+1:...
%             (i-1)*shrink_size+shrink_size,...
%             (j-1)*shrink_size+1:...
%             (j-1)*shrink_size+shrink_size);
%         output(i,j) = mean(block(:));
%     end
% end

end