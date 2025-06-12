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


%% ADMM_LSI_deconv_3D
% This function applies the Alternating Direction Method of Multipliers (ADMM)
% to solve the 3D deconvolution problem with 3D total variation (TV)
% and l1 regularization. The goal is to minimize the cost function:
% 0.5*|CHx - y|^2 + tau*|x|_1 + non_negative_indicator{x}
%
% Inputs:
%   y      - The measured data (size: [rows, cols, layers])
%   psf    - The point spread function (size: [rows, cols, layers])
%   para   - A structure containing parameters for the ADMM algorithm
%   varargin - Optional custom display region parameters (if provided)
%
% Outputs:
%   output  - The estimated object after the ADMM process
function [output] = ADMM_LSI_deconv_3D(y, psf, para, varargin)
%% ADMM_LSI_deconv_3D - A debug version of a 3D CM2 model ADMM solver
% This function applies the Alternating Direction Method of Multipliers (ADMM) 
% to solve the 3D deconvolution problem with 3D total variation (TV) 
% and l1 regularization. The goal is to minimize the cost function:
% 0.5*|CHx - y|^2 + tau*|x|_1 + non_negative_indicator{x}
%
% Inputs:
%   y      - The measured data (size: [rows, cols, layers])
%   psf    - The point spread function (size: [rows, cols, layers])
%   para   - A structure containing parameters for the ADMM algorithm
%   varargin - Optional custom display region parameters (if provided)
%
% Outputs:
%   output  - The estimated object after the ADMM process

% Convert input data to single precision
y = single(y);
psf = single(psf);

% Extract relevant parameters from the para structure
[rows, cols, layers] = size(psf); % Get dimensions of PSF
mu1 = para.mu1; % Lagrange multiplier for data fidelity term
mu2 = para.mu2; % Lagrange multiplier for total variation term
mu3 = para.mu3; % Lagrange multiplier for l1 term
mu4 = para.mu4; % Lagrange multiplier for non-negativity
tau_l1 = para.tau_l1; % Regularization parameter for l1 norm
tau_tv = para.tau_tv; % Regularization parameter for total variation
maxiter = para.maxiter; % Maximum number of iterations
termination_ratio = para.termination_ratio; % Termination criteria based on evolution ratio
plateau_tolerence = para.plateau_tolerence; % Tolerance for plateau convergence
color = para.color; % Color map for display
clip_min = para.clip_min; % Minimum clipping value
clip_max = para.clip_max; % Maximum clipping value

% Additional parameters
rtol = para.rtol; % Relative tolerance for parameter update
mu_ratio = para.mu_ratio; % Ratio for updating multipliers
img_save_period = para.img_save_period; % Interval for saving images
img_save_path = para.img_save_path; % Path for saving images

% Check for optional arguments for custom display region
if length(varargin) == 3
    custom_display_region_flag = true; % Flag for custom display region
    display_row_start = varargin{1}; % Starting row for display
    display_col_start = varargin{2}; % Starting column for display
    display_width = varargin{3}; % Width of the display region
else
    custom_display_region_flag = false; % Default to no custom region
end

%% Define operators for FFT and cropping
F = @(x) fftshift(fft2(ifftshift(x))); % 2D FFT operator
Ft = @(x) fftshift(ifft2(ifftshift(x))); % Inverse 2D FFT operator
F3D = @(x) fftshift(fftn(ifftshift(x))); % 3D FFT operator
Ft3D = @(x) fftshift(ifftn(ifftshift(x))); % Inverse 3D FFT operator
C2D = @(x) x(1+rows/2:rows+rows/2, 1+cols/2:cols+cols/2); % Cropping operator for 2D
CT2D = @(x) padarray(x, [rows/2, cols/2]); % Padding operator for 2D
clip = @(x, vmin, vmax) max(min(x, vmax), vmin); % Clipping function
VEC = @(x) x(:); % Vectorization function
linear_normalize = @(x) (x - min(x(:))) ./ (max(x(:)) - min(x(:))); % Linear normalization

%% ADMM algorithm initialization
Hs = F3D(pad3d(psf)); % Frequency representation of the PSF
Hs_conj = conj(Hs); % Conjugate of the PSF in frequency domain
H = @(x) real(Ft3D(F3D(x) .* Hs)); % Forward operator
HT = @(x) real(Ft3D(F3D(x) .* Hs_conj)); % Adjoint operator
HTH = abs(Hs .* Hs_conj); % H^H * H term for updating x

% Initialize variables for ADMM
xt = zeros(2*rows, 2*cols, layers, 'single'); % Initialization of x
gamma1 = zeros(2*rows, 2*cols, layers, 'single'); % Dual variable for data fidelity
gamma3 = zeros(2*rows, 2*cols, layers, 'single'); % Dual variable for l1 constraint
gamma4 = zeros(2*rows, 2*cols, layers, 'single'); % Dual variable for non-negativity
CTy = CT3D(y, layers); % Pad y for 3D processing

% Generate 3D Laplacian for total variation regularization
PsiTPsi = generate_laplacian_3D(rows, cols, layers);
gamma2_1 = zeros(2*rows-1, 2*cols, layers, 'single'); % Dual variable for TV in x-direction
gamma2_2 = zeros(2*rows, 2*cols-1, layers, 'single'); % Dual variable for TV in y-direction
gamma2_3 = zeros(2*rows, 2*cols, layers-1, 'single'); % Dual variable for TV in z-direction

% Define the proximal operator for the total variation
PsiT = @(P1, P2, P3) cat(1, P1(1,:,:), diff(P1, 1, 1), -P1(end,:,:)) + ...
                   cat(2, P2(:,1,:), diff(P2, 1, 2), -P2(:,end,:)) + ...
                   cat(3, P3(:,:,1), diff(P3, 1, 3), -P3(:,:,end));
Psi = @(x) deal(-diff(x, 1, 1), -diff(x, 1, 2), -diff(x, 1, 3)); % Gradient operator for TV

% Initialize TV-related variables
[ut1, ut2, ut3] = Psi(zeros(2*rows, 2*cols, layers, 'single')); % Initialization for TV
Psixt1 = ut1; % Proximal variable for TV in x-direction
Psixt2 = ut2; % Proximal variable for TV in y-direction
Psixt3 = ut3; % Proximal variable for TV in z-direction

% Precompute masks for updates
x_mask = 1 ./ (mu1 * HTH + mu2 * PsiTPsi + mu3 + mu4); % Denominator for x updates
v_mask = 1 ./ (CT3D(ones(size(y)), layers) + mu1); % Denominator for v updates

% Initialize iteration variables
iter = 0;
Hxtp = zeros(2*rows, 2*cols, layers, 'single'); % Previous estimate of Hx

% Display initialization (if enabled)
if para.display_flag
    f1 = figure('rend', 'painters', 'pos', [50 50 1500 900]); % Create a figure for display
end

next_iteration = 1; % Flag for continuing iterations
num_plateaus = 0; % Counter for consecutive plateau iterations

% Main ADMM loop
while (next_iteration) && (iter <= maxiter) 
   iter = iter + 1; % Increment iteration count
   Hxt = Hxtp; % Store previous Hx estimate
   vtp = v_mask .* (mu1 * Hxt + gamma1 + CTy); % Update v
   wtp = clip(xt + gamma3 / mu3, clip_min, clip_max); % Update w with clipping
   ztp = wthresh(gamma4 / mu4 + xt, 's', tau_l1 / mu4); % Soft thresholding for z
   [ut1, ut2, ut3] = soft_thres_3d(Psixt1 + gamma2_1 / mu2, Psixt2 + gamma2_2 / mu2, Psixt3 + gamma2_3 / mu2, tau_tv / mu2); % Update TV variables

   % Compute contributions to x update
   tmp_part1 = mu1 * HT(vtp) - HT(gamma1); 
   tmp_part2 = mu2 * PsiT(ut1 - gamma2_1 / mu2, ut2 - gamma2_2 / mu2, ut3 - gamma2_3 / mu2); 
   tmp_part3 = mu3 * wtp - gamma3; 
   tmp_part4 = mu4 * ztp - gamma4; 

   % Update x
   xtp_numerator = tmp_part1 + tmp_part2 + tmp_part3 + tmp_part4; 
   xtp = real(Ft3D(F3D(xtp_numerator) .* x_mask)); % Compute new x estimate

   % Update dual variables
   Hxtp = H(xtp); % Update Hx estimate
   gamma1 = gamma1 + mu1 * (Hxtp - vtp); % Update dual variable for data fidelity

   % Update dual variables for TV
   [Psixt1, Psixt2, Psixt3] = Psi(xtp); 
   gamma2_1 = gamma2_1 + mu2 * (Psixt1 - ut1);
   gamma2_2 = gamma2_2 + mu2 * (Psixt2 - ut2);
   gamma2_3 = gamma2_3 + mu2 * (Psixt3 - ut3);

   % Update dual variables for l1 and non-negativity
   gamma3 = gamma3 + mu3 * (xtp - wtp);
   gamma4 = gamma4 + mu4 * (xtp - ztp);

   % Residual calculations for convergence checks
   primal_residual_mu1 = norm(VEC(Hxtp - vtp)); % Primal residual for mu1
   dual_residual_mu1 = mu1 * norm(VEC(Hxt - Hxtp)); % Dual residual for mu1
   [mu1, mu1_update] = ADMM_update_param(mu1, rtol, mu_ratio, primal_residual_mu1, dual_residual_mu1); % Update mu1

   primal_residual_mu2_1 = norm(VEC(Psixt1 - ut1)); % Primal residual for mu2 (x)
   primal_residual_mu2_2 = norm(VEC(Psixt2 - ut2)); % Primal residual for mu2 (y)
   primal_residual_mu2_3 = norm(VEC(Psixt3 - ut3)); % Primal residual for mu2 (z)
   primal_residual_mu2 = norm([primal_residual_mu2_1, primal_residual_mu2_2, primal_residual_mu2_3]); % Overall primal residual for mu2

   [Psixt1_last, Psixt2_last, Psixt3_last] = Psi(xt); % Get previous Psi variables for dual residuals
   dual_residual_mu2_1 = mu2 * norm(VEC(Psixt1_last - Psixt1)); % Dual residual for mu2 (x)
   dual_residual_mu2_2 = mu2 * norm(VEC(Psixt2_last - Psixt2)); % Dual residual for mu2 (y)
   dual_residual_mu2_3 = mu2 * norm(VEC(Psixt3_last - Psixt3)); % Dual residual for mu2 (z)
   dual_residual_mu2 = norm([dual_residual_mu2_1, dual_residual_mu2_2, dual_residual_mu2_3]); % Overall dual residual for mu2
   [mu2, mu2_update] = ADMM_update_param(mu2, rtol, mu_ratio, primal_residual_mu2, dual_residual_mu2); % Update mu2

   primal_residual_mu3 = norm(VEC(xtp - wtp)); % Primal residual for mu3
   dual_residual_mu3 = mu3 * norm(VEC(xt - xtp)); % Dual residual for mu3
   [mu3, mu3_update] = ADMM_update_param(mu3, rtol, mu_ratio, primal_residual_mu3, dual_residual_mu3); % Update mu3

   primal_residual_mu4 = norm(VEC(xtp - ztp)); % Primal residual for mu4
   dual_residual_mu4 = mu4 * norm(VEC(xt - xtp)); % Dual residual for mu4
   [mu4, mu4_update] = ADMM_update_param(mu4, rtol, mu_ratio, primal_residual_mu4, dual_residual_mu4); % Update mu4

   % Check if any mu values were updated
   if mu1_update || mu2_update || mu3_update || mu4_update
       mu_update = 1; % Indicate that mu values were updated
   else
       mu_update = 0; % No updates
   end

   % Check termination condition: consecutive plateaus and mu not updated
   xt_last = xt; % Store last estimate of x
   xt = xtp; % Update x
   evolution_ratio_of_the_iteration = compute_evolution_ratio(xt, xt_last); % Compute evolution ratio
   if (evolution_ratio_of_the_iteration <= termination_ratio) && (mu_update == 0)
       num_plateaus = num_plateaus + 1; % Increase plateau count
   else
       num_plateaus = 0; % Reset plateau count
   end

   % If consecutive plateaus exceed tolerance, stop the iterations
   if num_plateaus >= plateau_tolerence
       next_iteration = 0; % Terminate iterations
   end

   % Display iteration progress
   disp(['iteration: ', num2str(iter), ', evo ratio: ', num2str(evolution_ratio_of_the_iteration), ', consecutive plateaus: ', num2str(num_plateaus)]);
   if next_iteration
      disp('continue next iteration...'); 
   else
      disp('terminating...');
      write_mat_to_tif(uint8(255 * linear_normalize(xt)), [img_save_path, '_final_iter_', num2str(iter), '.tif']); % Save final result
   end

   % Save intermediate results at specified intervals
   if mod(iter, img_save_period) == 0
       write_mat_to_tif(uint8(255 * linear_normalize(xt)), [img_save_path, '_iter_', num2str(iter), '.tif']); % Save intermediate result
   end

   % Update masks if any mu has been updated
   if mu_update
       disp(['mu updated. mu1: ', num2str(round(mu1, 3)), ', mu2: ', num2str(round(mu2, 3)),...
           ', mu3: ', num2str(round(mu3, 3)), ', mu4: ', num2str(round(mu4, 3))]);
       x_mask = 1 ./ (mu1 * HTH + mu2 * PsiTPsi + mu3 + mu4); % Update denominator for x
       v_mask = 1 ./ (CT3D(ones(size(y)), layers) + mu1); % Update denominator for v
   end
   
   % Display results if enabled
   if para.display_flag
       img2display = max(xt, [], 3); % Prepare image for display
       
       figure(f1); % Switch to the display figure
       if custom_display_region_flag
           subplot(1, 2, 1), imagesc(img2display(display_row_start:display_row_start + display_width - 1,...
               display_col_start:display_col_start + display_width - 1)),...
               colormap(color); axis image; colorbar; title(iter);
       else
           subplot(1, 2, 1), imagesc(img2display), colormap(color); axis image; colorbar; title(iter);
       end
       % Calculate residuals and costs for plotting
       residual = abs(crop3d(H(xt)) - y); % Compute residual
       dterm = 0.5 * sum(residual(:).^2); % Data fidelity term
       [tv_x, tv_y, tv_z] = Psi(xt); % Total variation components
       tv_x = cat(1, tv_x, zeros(1, 2*cols, layers, 'single')); % Pad for consistency
       tv_y = cat(2, tv_y, zeros(2*rows, 1, layers, 'single'));
       tv_z = cat(3, tv_z, zeros(2*rows, 2*cols, 1, 'single'));
       tvterm = tau_tv * sum(sqrt(tv_x(:).^2 + tv_y(:).^2 + tv_z(:).^2)); % Total variation term
       l1term = tau_l1 * sum(abs(xt(:))); % l1 norm term
       cost = dterm + tvterm + l1term; % Total cost
       
       % Plot costs over iterations
       subplot(1, 2, 2), plot(iter, log10(cost), 'bo'), grid on, hold on;...
           plot(iter, log10(dterm), 'ro'), hold on;...
           plot(iter, log10(tvterm), 'go'), hold on;...
           plot(iter, log10(l1term), 'mo'), hold on;...
           title('log axis: blue: cost; red: data fidelity; green: tv; purple: l1');
       drawnow; % Update display
   end

% Calculate and log various loss components during the optimization process
loss_cvy = crop3d(vtp) - y; % Compute the convex loss by subtracting the actual output y from the cropped vtp
loss_cvy = 0.5 * sum(VEC(loss_cvy.^2)); % Calculate the total convex loss using L2 norm (squared error)

% Calculate total variation (TV) loss, weighted by tau_tv
loss_tau_tv = tau_tv * (sum(abs(VEC(ut1))) + sum(abs(VEC(ut2))) + sum(abs(VEC(ut3)))); 

% Calculate L1 loss, weighted by tau_l1
loss_tau_l1 = tau_l1 * sum(abs(VEC(ztp))); 

% Calculate loss for the parameter mu1
loss_mu1 = (Hxtp - vtp + gamma1 / mu1); % Compute the residual loss for mu1
loss_mu1 = 0.5 * mu1 * sum(VEC(loss_mu1.^2)); % Calculate the weighted loss for mu1

% Compute losses for mu2 parameters
loss_mu2_1 = Psixt1 - ut1 + gamma2_1/mu2; % Residual loss for first mu2 parameter
loss_mu2_2 = Psixt2 - ut2 + gamma2_2/mu2; % Residual loss for second mu2 parameter
loss_mu2_3 = Psixt3 - ut3 + gamma2_3/mu2; % Residual loss for third mu2 parameter
loss_mu2 = 0.5 * mu2 * (sum(VEC(loss_mu2_1.^2)) + sum(VEC(loss_mu2_2.^2)) + sum(VEC(loss_mu2_3.^2))); % Total loss for mu2

% Calculate losses for the parameters mu3 and mu4
loss_mu3 = xtp - wtp + gamma3/mu3; % Loss for mu3
loss_mu3 = 0.5 * mu3 * sum(VEC(loss_mu3.^2)); % Weighted loss for mu3
loss_mu4 = xtp - ztp + gamma4/mu4; % Loss for mu4
loss_mu4 = 0.5 * mu4 * sum(VEC(loss_mu4.^2)); % Weighted loss for mu4

% Calculate the total loss by summing all individual losses
loss_tot = loss_cvy + loss_tau_tv + loss_tau_l1 + loss_mu1 + loss_mu2 + loss_mu3 + loss_mu4; 

% Display the current iteration details and loss values rounded to 3 decimal places
disp(['iter: ',num2str(iter),', cvy: ',num2str(round(loss_cvy,3)), ', tau_tv: ',num2str(round(loss_tau_tv,3))...
    , ', tau_l1: ',num2str(round(loss_tau_l1,3))...
    ,', mu1: ',  num2str(round(loss_mu1,3)) ,', mu1: ',  num2str(round(loss_mu1,3))...
    , ', mu3: ',  num2str(round(loss_mu3,3)), ', mu4: ',  num2str(round(loss_mu4,3))...
    , ', total: ',  num2str(round(loss_tot,3))]);
end
end

% Function to update the parameter mu based on the residual and update ratio
function [mu_out, mu_update] = ADMM_update_param(mu,resid_tol,mu_ratio,r,s)
    % Adjust mu based on the comparison of residual r and tolerance multiplied by s
    if r > resid_tol*s % If residual is greater than the tolerance threshold, increase mu
        mu_out = mu*mu_ratio; % Update mu by multiplying with the ratio
        mu_update = 1; % Indicate an increase in mu
    elseif r*resid_tol < s % If residual is significantly smaller
        mu_out = mu/mu_ratio; % Decrease mu
        mu_update = -1; % Indicate a decrease in mu
    else
        mu_out = mu; % No change to mu
        mu_update = 0; % Indicate no change
    end
end

% Function to apply soft thresholding in 3D
function [varargout] =  soft_thres_3d(v,h,z,tau)
    % Calculate the magnitude of the 3D input components
    mag = sqrt(cat(1,v,zeros(1,size(v,2),size(v,3),'single')).^2 + ...
                cat(2,h,zeros(size(h,1),1,size(h,3),'single')).^2 + ...
                cat(3,z, zeros(size(z,1),size(z,2),1,'single')).^2);
    
    % Apply soft thresholding to the magnitude with threshold tau
    magt = wthresh(mag,'s',tau);
    
    % Calculate multiplier based on the thresholded magnitude
    mmult = magt./mag; 
    mmult(mag==0) = 0; % Avoid division by zero by nullifying zero magnitudes
    
    % Output the thresholded components
    varargout{1} = v.*mmult(1:end-1,:,:); % Thresholded v component
    varargout{2} = h.*mmult(:,1:end-1,:); % Thresholded h component
    varargout{3} = z.*mmult(:,:,1:end-1); % Thresholded z component
end

% Function to generate a 3D Laplacian kernel
function PsiTPsi = generate_laplacian_3D(rows,cols,layers)
    F3D = @(x) fftshift(fftn(ifftshift(x))); % 3D FFT function
    
    % Initialize a Laplacian matrix for 3D
    lapl = zeros(2*rows,2*cols, layers,'single'); % Use single precision for efficiency
    lapl(rows+1,cols+1, layers/2+1) = 6; % Set the central element for Laplacian
    
    % Set the neighboring elements for the Laplacian kernel
    lapl(rows+1,cols+2, layers/2+1) = -1;
    lapl(rows+2,cols+1, layers/2+1) = -1;
    lapl(rows,cols+1, layers/2+1) = -1;
    lapl(rows+1,cols, layers/2+1) = -1;
    lapl(rows+1,cols+1, layers/2+2) = -1;
    lapl(rows+1,cols+1, layers/2) = -1;
    
    % Compute the power spectrum of the Laplacian kernel
    PsiTPsi = abs(F3D(lapl));   
end

% Function to crop a 3D array to a specified region
function output = crop3d(input)
    [r,c,z] = size(input); % Get dimensions of the input
    output = input(r/4+1:r*3/4,c/4+1:c*3/4,z/2); % Crop to the center region
end

% Function to pad a 2D input to a 3D measurement space
function output = CT3D(input,tot_layers) 
    [r,c] = size(input); % Get dimensions of the input
    output = padarray(input,[r/2, c/2]); % Pad the input to make it larger
    output = cat(3,zeros(r*2, c*2, tot_layers/2-1,'single'), output, zeros(r*2,c*2,tot_layers/2,'single')); % Add layers of zeros for the 3D structure
end

% Function to pad a 3D PSF stack
function output = pad3d(input) 
    [r,c,~] = size(input); % Get the size of the input
    output = padarray(input,[r/2,c/2,0]); % Pad the input in the first two dimensions
end

% Function to perform 3D convolution using FFT
function output = conv3d(obj,psf)
    F3D = @(x) fftshift(fftn(ifftshift(x))); % Define a 3D FFT function
    Ft3D = @(x) fftshift(ifftn(ifftshift(x))); % Define a 3D inverse FFT function
    % Perform convolution in the frequency domain and crop to get the final output
    output = crop3d(real(Ft3D(F3D(pad3d(obj)).*F3D(pad3d(psf))))); 
end

% Function to compute the evolution ratio between two 3D inputs
function evolution_ratio = compute_evolution_ratio(xt,xtm1)
    % Calculate the evolution ratio by computing the norm difference 
    evolution_ratio = norm(xt(:) - xtm1(:)) / norm(xtm1(:)); % Normalize the difference by the norm of the previous state
end