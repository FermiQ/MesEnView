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

%% useful functions
% Convert a matrix to a column vector
% @param x Input matrix to be converted to a column vector
% @return VEC The input matrix converted to a column vector
VEC = @(x) x(:);

% Clip the values of x to be within the range [vmin, vmax]
% @param x Input array to be clipped
% @param vmin Minimum value to clip to
% @param vmax Maximum value to clip to
% @return The input array x clipped to the range [vmin, vmax]
clip = @(x, vmin, vmax) max(min(x, vmax), vmin);

% Perform a 2D Fast Fourier Transform (FFT) with zero-centering
% @param x Input 2D array to be transformed
% @return The 2D FFT of the input array x with zero-centering
F2D = @(x) fftshift(fft2(ifftshift(x)));

% Perform the inverse 2D Fast Fourier Transform (IFFT) with zero-centering
% @param x Input 2D array to be transformed back
% @return The 2D IFFT of the input array x with zero-centering
Ft2D = @(x) fftshift(ifft2(ifftshift(x)));

% Pad a 2D array by adding zeros around it
% @param x Input 2D array to be padded
% @return The input array x padded with zeros
pad2d = @(x) padarray(x,0.5*size(x));

% Crop a 2D array to the central region, effectively reducing its size
% @param x Input 2D array to be cropped
% @return The central region of the input array x
crop2d = @(x) x(1+size(x,1)/4:size(x,1)/4*3,1+size(x,2)/4:size(x,2)/4*3);

% Convolve 2D object with a point spread function (PSF) and crop the result
% @param obj 2D object array to be convolved
% @param psf 2D point spread function to convolve with
% @return The result of convolving the object with the PSF, cropped to the central region
conv2d = @(obj,psf) crop2d(real(Ft2D(F2D(pad2d(obj)).*F2D(pad2d(psf)))));

% Normalize a matrix to have unit norm
% @param x Input array to be normalized
% @return The input array x normalized to have unit norm
unit_norm = @(x) x./norm(x(:));

% Compute the autocorrelation of an input matrix x
% @param x Input array for which autocorrelation is to be computed
% @return The autocorrelation of the input array x
auto_corr = @(x) crop2d(Ft2D(F2D(pad2d(unit_norm(x))).*conj(F2D(pad2d(unit_norm(x))))));

% Compute the cross-correlation between two matrices x and y
% @param x First input array
% @param y Second input array
% @return The cross-correlation between the input arrays x and y
x_corr = @(x,y) crop2d(Ft2D(F2D(pad2d(unit_norm(x))).*conj(F2D(pad2d(unit_norm(y))))));

% Normalize a matrix x to the range [0, 1]
% @param x Input array to be normalized
% @return The input array x normalized to the range [0, 1]
linear_normalize = @(x) (x - min(x(:)))./(max(x(:))-min(x(:)));

%% useful functions
% Convert a matrix to a column vector
VEC = @(x) x(:); 

% Clip the values of x to be within the range [vmin, vmax]
% x: input array to be clipped
% vmin: minimum value to clip to
% vmax: maximum value to clip to
clip = @(x, vmin, vmax) max(min(x, vmax), vmin);

% Perform a 2D Fast Fourier Transform (FFT) with zero-centering
% x: input 2D array to be transformed
F2D = @(x) fftshift(fft2(ifftshift(x)));

% Perform the inverse 2D Fast Fourier Transform (IFFT) with zero-centering
% x: input 2D array to be transformed back
Ft2D = @(x) fftshift(ifft2(ifftshift(x)));

% Pad a 2D array by adding zeros around it
% x: input 2D array to be padded
pad2d = @(x) padarray(x,0.5*size(x)); 

% Crop a 2D array to the central region, effectively reducing its size
% x: input 2D array to be cropped
crop2d = @(x) x(1+size(x,1)/4:size(x,1)/4*3,1+size(x,2)/4:size(x,2)/4*3);

% Convolve 2D object with a point spread function (PSF) and crop the result
% obj: 2D object array to be convolved
% psf: 2D point spread function to convolve with
conv2d = @(obj,psf) crop2d(real(Ft2D(F2D(pad2d(obj)).*F2D(pad2d(psf)))));

% Normalize a matrix to have unit norm
% x: input array to be normalized
unit_norm = @(x) x./norm(x(:)); 

% Compute the autocorrelation of an input matrix x
% x: input array for which autocorrelation is to be computed
auto_corr = @(x) crop2d(Ft2D(F2D(pad2d(unit_norm(x))).*conj(F2D(pad2d(unit_norm(x))))));

% Compute the cross-correlation between two matrices x and y
% x: first input array
% y: second input array
x_corr = @(x,y) crop2d(Ft2D(F2D(pad2d(unit_norm(x))).*conj(F2D(pad2d(unit_norm(y))))));

% Normalize a matrix x to the range [0, 1]
% x: input array to be normalized
linear_normalize = @(x) (x - min(x(:)))./(max(x(:))-min(x(:)));

%% load 3D PSFs data
load('downsampled_3d_psfs.mat'); % Load downsampled 3D point spread function (PSF) data

%% down sample to further reduce the scale
% Extract a specific slice range from the PSFs for further processing
psfs = psfs_ds(:,:,81:320); 
clear psfs_ds % Clear the original data to save memory

% Initialize a temporary array for storing processed PSFs
psfs_tmp = zeros(324,432,16);
for i = 0:15
    % Sum slices of the PSFs within the specified index range and average them
    tmp = psfs(:,:,i*15+1:i*15+15);
    tmp = sum(tmp,3); % Sum across the third dimension
    tmp = average_shrink(tmp,3); % Reduce the size of the PSF
    psfs_tmp(:,:,i+1) = tmp; % Store the result in the temporary array
end
psfs = psfs_tmp; % Update PSFs with the processed ones
clear psfs_tmp % Clear the temporary variable

% Prepare PSF reconstruction by removing small background values
[rows,cols,depth] = size(psfs); % Get dimensions of the PSFs
psfs_recon = psfs; % Initialize reconstruction PSFs from original PSFs
psfs_recon(psfs_recon<=1.5e-4) = 0; % Set small values to zero

% Normalize the PSFs and the reconstruction PSFs
for i = 1:depth
    tmp = psfs(:,:,i);
    tmp = tmp ./ sum(tmp(:)); % Normalize current PSF
    psfs(:,:,i) = tmp; % Update the normalized PSF
    tmp = psfs_recon(:,:,i);
    tmp = tmp ./ sum(tmp(:)); % Normalize the reconstruction PSF
    psfs_recon(:,:,i) = tmp; % Update the normalized reconstruction PSF
end

%% synthesize a single 2D measurement by depth-wise convolution between
% PSFs and objects (here it is a volume of 1.5 pixel radius spheres)
% Create a meshgrid for full field of view (FOV) coordinates
[xx,yy] = meshgrid([-cols/2:1:cols/2-1], [-rows/2:1:rows/2-1]); 
% Create coordinates for a small sphere volume
[sx,sy,sz] = meshgrid([-17:17].*(1/5), [-17:17].*(1/5), [-1:1]); 
num_particles = 200; % Number of particles to simulate
current_radius = 1.5; % Radius of the sphere in pixels

% Create a dense sphere volume for convolution
sphere = zeros(7*5,7*5,3); % Initialize a 3D array for the sphere
sphere(sqrt(sx.^2+sy.^2+sz.^2)<=current_radius) = 1; % Define the sphere based on the radius

% Shrink the generated sphere to make it smoother
sphere2 = zeros(7,7,3);
for i = 1:3
    sphere2(:,:,i) = average_shrink(sphere(:,:,i),5); % Apply average shrink to each slice
end
sphere = sphere2; % Update the sphere with the shrunk version
sphere = sum(sphere,3); % Sum across the third dimension to create a 2D projection
clear sphere2 % Clear the temporary variable

% Initialize variables for ground truth (GT) volume and locations
gt_volume = zeros(rows,cols,depth); % Initialize GT volume
gt_locations = zeros(num_particles,4); % Initialize GT locations array
std_lum = 0.3; % Standard deviation for random luminosity

% Randomly place particles in the GT volume
for i = 1:num_particles
    rr = randi([7,rows-7]); % Random row index
    cc = randi([7,cols-7]); % Random column index
    
    % Ensure particles are within a certain distance from the center
    while true 
        rr = randi([7,rows-7]);
        cc = randi([7,cols-7]);
        if sqrt(xx(rr,cc).^2 + yy(rr,cc).^2) <= 100
            break
        end
    end

    zz = randi([2, depth-2]); % Random depth index
    gt_locations(i,1) = rr; % Store row location
    gt_locations(i,2) = cc; % Store column location
    gt_locations(i,3) = zz; % Store depth location

    tmp_lum = 1 + std_lum*randn(1); % Generate random luminosity
    tmp_lum = clip(tmp_lum, 0,1.5); % Clip luminosity to valid range
    gt_locations(i,4) = tmp_lum; % Store luminosity
    gt_volume(rr-3:rr+3,cc-3:cc+3,zz) = gt_volume(rr-3:rr+3,cc-3:cc+3,zz) + tmp_lum.*sphere; % Add the sphere to the GT volume
end

% Synthesis of measurements from ground truth volume using forward model
y_part1 = gather(cm2_forward_gpu(gt_volume,psfs,false)); % Perform forward simulation using GPU
% y_part2 = crop2d(gather(cm2_forward_bg_gpu(gt_volume_bg,psfs,false))); % Background part (commented out)

% y = y_part1 + factor*y_part2; % Combine signal with background (commented out)

y = y_part1; % Use the first part of the measurement

% Normalize the measurements to the range [0, 1]
y = y./max(y(:)); 

% Simulate Poisson noise in the measurements
y = poissrnd(y.*1000)/1000; 

% Add Gaussian noise to the measurements
noise_std = 0.02; % Standard deviation of the noise
y = y + noise_std*randn(size(y)); % Add noise to measurements
y = clip(y,0,1); % Clip the noisy measurements to valid range

%% reconstruction
% Convert PSFs and measurements to single precision for processing
psfs_recon = single(psfs_recon); 
y = single(y); 
y_bgsub = single(bg_removal(y,4)); % Perform background removal

% Set parameters for the reconstruction algorithm
para = [];
para.mu1 = 1; % Parameter for regularization
para.mu2 = 1; % Parameter for regularization
para.mu3 = 1; % Parameter for another regularization
para.mu4 = 1; % Parameter for another regularization
para.clip_min = 0; % Minimum clipping value
para.clip_max = 100; % Maximum clipping value
para.color = jet(256); % Color map for visualization
para.tau_l1 = 0.03; % Step size for L1 regularization
para.tau_tv = 0.01; % Step size for total variation regularization
para.rtol = 1.5; % Relative tolerance for convergence
para.mu_ratio = 1.1; % Ratio for adjusting mu
para.display_flag = 1; % Flag to enable display during processing
para.termination_ratio = 0.01; % Termination criteria based on ratio
para.plateau_tolerance = 4; % Tolerance for plateau in optimization
para.maxiter = 128; % Maximum number of iterations for reconstruction
para.img_save_period = 80; % Save results every given iterations
para.img_save_path = ''; % Path for saving images (empty: do not save)
xhat = ADMM_LSI_deconv_3D(y_bgsub,psfs_recon,para); % Perform 3D deconvolution using ADMM

%% save gt vol as tif
% Prepare the reconstructed volume for saving as TIF
xhat(xhat<=0) = 0; % Set negative values to zero
xhat = xhat(rows/2+1:rows/2+rows,cols/2+1:cols/2+cols,end:-1:1); % Adjust the output volume to match the ground truth dimensions
xhat = uint8(255*linear_normalize(xhat)); % Normalize and convert to uint8 for saving 
write_mat_to_tif(xhat, 'xhat_cropped.tif'); % Save the reconstructed volume to TIF
write_mat_to_tif(uint8(255*linear_normalize(gt_volume)), 'gt_vol.tif'); % Save the ground truth volume to TIF