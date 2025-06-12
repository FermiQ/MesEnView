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

import matplotlib.pyplot as plt  # Importing matplotlib for plotting
from matplotlib import gridspec  # Importing gridspec to create complex subplot layouts
import numpy as np  # Importing numpy for numerical operations

def showlearnedparam(f, p):
    """
    Displays learned parameters from the fitting process for the specified channel type.

    Parameters:
    f : object
        Contains data regarding the results of the analysis.
    p : object
        Contains parameters such as channel type that dictate how the data should be interpreted.

    - If the channel type is 'single', it extracts the corresponding values from f.rois and f.res.
    - For '4pi' channel type, it calculates the intensity phase and magnitude.
    - It then plots the X, Y, and Z positions of the beads, the intensities, backgrounds, and drift rates.
    """
    # Extract relevant data based on channel type
    if p.channeltype == 'single':
        cor = f.rois.cor  # Correction values
        pos = f.res.pos  # Positions of the beads
        photon = f.res.intensity.transpose()  # Photon counts (transposed for plotting)
        bg = f.res.bg  # Background values
        drift = f.res.drift_rate  # Drift rates
    else:
        cor = f.rois.cor[0]  # Use first channel correction values
        pos = f.res.channel0.pos  # Positions for channel 0
        photon = f.res.channel0.intensity.transpose()  # Intensity for channel 0
        bg = f.res.channel0.bg  # Background for channel 0
        drift = f.res.channel0.drift_rate  # Drift for channel 0

    # Special handling for 4pi channel type
    if p.channeltype == '4pi':
        phi = np.angle(f.res.channel0.intensity)  # Calculate the phase of the intensity
        photon = np.abs(f.res.channel0.intensity.transpose())  # Use absolute intensity for plotting

    # Create a figure with a specific size
    fig = plt.figure(figsize=[16, 8])
    
    # Create a grid for subplots
    spec = gridspec.GridSpec(ncols=4, nrows=2,
                             width_ratios=[3, 3, 3, 3], wspace=0.4,
                             hspace=0.3, height_ratios=[4, 4])
    
    # Plotting the positions of the beads in X, Y, and Z
    ax = fig.add_subplot(spec[0])
    plt.plot(pos[:, 2] - cor[:, 1])  # X positions after correction
    plt.xlabel('bead number')
    plt.ylabel('x (pixel)')

    ax = fig.add_subplot(spec[1])
    plt.plot(pos[:, 1] - cor[:, 0])  # Y positions after correction
    plt.xlabel('bead number')
    plt.ylabel('y (pixel)')

    ax = fig.add_subplot(spec[2])
    plt.plot(pos[:, 0])  # Z positions
    plt.xlabel('bead number')
    plt.ylabel('z (pixel)')

    # If channel type is 4pi, plot the phase
    if p.channeltype == '4pi':
        ax = fig.add_subplot(spec[3])
        plt.plot(phi)  # Plot the phase values
        ax.set_xlabel('bead number')
        ax.set_ylabel('phi (radian)')

    ax = fig.add_subplot(spec[4])
    plt.plot(photon)  # Plot the photon counts
    if len(photon.shape) > 1:
        plt.xlabel('z slice')  # If multiple slices, label accordingly
        plt.legend(['bead 1'])
    else:
        plt.xlabel('bead number')  # For single slice
    plt.ylabel('photon')

    ax = fig.add_subplot(spec[5])
    plt.plot(bg)  # Plot background values
    plt.xlabel('bead number')
    plt.ylabel('background')

    ax = fig.add_subplot(spec[6])
    plt.plot(drift)  # Plot drift rates
    plt.xlabel('bead number')
    plt.ylabel('drift per z slice (pixel)')
    plt.legend(['x', 'y'])

    plt.show()  # Display all plots

    return  # End of function

def showlearnedparam_insitu(f, p):
    """
    Displays learned parameters from the fitting process for the specified channel type in situ.

    Parameters:
    f : object
        Contains data regarding the results of the analysis.
    p : object
        Contains parameters such as channel type that dictate how the data should be interpreted.
    
    - Similar to showlearnedparam but may have different handling specific to in situ conditions.
    """
    # Extract relevant data based on channel type
    if p.channeltype == 'single':
        cor = f.rois.cor  # Correction values
        pos = f.res.pos  # Positions of the beads
        photon = f.res.intensity  # Photon counts
        bg = f.res.bg  # Background values
    else:
        cor = f.rois.cor[0]  # Use first channel correction values
        pos = f.res.channel0.pos  # Positions for channel 0
        photon = f.res.channel0.intensity  # Intensity for channel 0
        bg = f.res.channel0.bg  # Background for channel 0

    # Special handling for 4pi channel type
    if p.channeltype == '4pi':
        phi = np.angle(f.res.channel0.intensity)  # Calculate the phase of the intensity
        photon = np.abs(f.res.channel0.intensity)  # Use absolute intensity for plotting

    # Create a figure with a specific size
    fig = plt.figure(figsize=[16, 8])

    # Create a grid for subplots
    spec = gridspec.GridSpec(ncols=4, nrows=2,
                             width_ratios=[3, 3, 3, 3], wspace=0.4,
                             hspace=0.3, height_ratios=[4, 4])
    
    # Plotting the positions of the beads in X, Y, and Z
    ax = fig.add_subplot(spec[0])
    plt.plot(pos[:, 2] - cor[:, 1], '.')  # X positions after correction
    plt.xlabel('emitter number')
    plt.ylabel('x (pixel)')

    ax = fig.add_subplot(spec[1])
    plt.plot(pos[:, 1] - cor[:, 0], '.')  # Y positions after correction
    plt.xlabel('emitter number')
    plt.ylabel('y (pixel)')

    ax = fig.add_subplot(spec[2])
    plt.plot(pos[:, 0], '.')  # Z positions
    plt.xlabel('emitter number')
    plt.ylabel('z (pixel)')

    # If channel type is 4pi, plot the phase
    if p.channeltype == '4pi':
        ax = fig.add_subplot(spec[3])
        plt.plot(phi, '.')  # Plot the phase values
        ax.set_xlabel('emitter number')
        ax.set_ylabel('phi (radian)')
        ax = fig.add_subplot(spec[6])
        plt.plot(pos[:, 0], phi, '.')  # Position vs phase
        ax.set_xlabel('z (pixel)')
        ax.set_ylabel('phi (radian)')

    ax = fig.add_subplot(spec[4])
    plt.plot(photon, '.')  # Plot the photon counts
    plt.xlabel('emitter number')
    plt.ylabel('photon')

    ax = fig.add_subplot(spec[5])
    plt.plot(bg, '.')  # Plot background values
    plt.xlabel('emitter number')
    plt.ylabel('background')

    return  # End of function

def showpupil(f, p, index=None):
    """
    Displays the pupil function (magnitude and phase) based on the channel type and optional index.

    Parameters:
    f : object
        Contains data regarding the results of the analysis.
    p : object
        Contains parameters such as channel type that dictate how the data should be interpreted.
    index : int, optional
        Index to select a specific pupil function for multi-channel data.

    - Pupil functions can be visualized for single, multi, or 4pi channel types.
    """
    # Handling for single channel type
    if p.channeltype == 'single':
        fig = plt.figure(figsize=[12, 5])
        if index is None:
            pupil = f.res.pupil  # Obtain pupil data without index
        else:
            pupil = f.res.pupil[index]  # Obtain pupil data for specified index

        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(np.abs(pupil), interpolation='nearest')  # Pupil magnitude
        plt.title('pupil magnitude')
        plt.colorbar()  # Show color bar

        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(np.angle(pupil), interpolation='nearest')  # Pupil phase
        plt.title('pupil phase')
        plt.colorbar()  # Show color bar

    # Handling for multi-channel type
    elif p.channeltype == 'multi':
        Nchannel = f.rois.psf_data.shape[0]  # Number of channels
        fig = plt.figure(figsize=[5 * Nchannel, 4])
        fig1 = plt.figure(figsize=[5 * Nchannel, 4])
        
        # Loop through each channel and plot pupil functions
        for i in range(0, Nchannel):
            if index is None:
                pupil = f.res['channel'+str(i)].pupil  # Obtain pupil data for each channel
            else:
                pupil = f.res['channel'+str(i)].pupil[index]  # Obtain pupil data for specific index

            ax = fig.add_subplot(1, Nchannel, i + 1)
            pupil_mag = np.abs(pupil)  # Calculate magnitude
            h = ax.imshow(pupil_mag, interpolation='nearest')  # Display pupil magnitude
            ax.axis('off')  # Hide axis
            ax.set_title('pupil magnitude ' + str(i))
            fig.colorbar(h, ax=ax)  # Color bar for magnitude

            ax1 = fig1.add_subplot(1, Nchannel, i + 1)
            pupil_phase = np.angle(pupil)  # Calculate phase
            h1 = ax1.imshow(pupil_phase, interpolation='nearest')  # Display pupil phase
            ax1.axis('off')  # Hide axis
            ax1.set_title('pupil phase ' + str(i))
            fig1.colorbar(h1, ax=ax1)  # Color bar for phase

    # Handling for 4pi channel type
    elif p.channeltype == '4pi':
        Nchannel = f.rois.psf_data.shape[0]  # Number of channels
        fig = plt.figure(figsize=[20, 8])
        
        # Loop through each channel and plot top and bottom pupil functions
        for i in range(0, Nchannel):
            ax = fig.add_subplot(2, 4, i + 1)
            pupil_mag = np.abs(f.res['channel'+str(i)].pupil1)  # Top pupil magnitude
            plt.imshow(pupil_mag, interpolation='nearest')  # Display pupil magnitude
            plt.axis('off')  # Hide axis
            plt.title('top pupil magnitude ' + str(i))
            plt.colorbar()  # Color bar

            ax = fig.add_subplot(2, 4, i + 5)
            pupil_mag = np.abs(f.res['channel'+str(i)].pupil2)  # Bottom pupil magnitude
            plt.imshow(pupil_mag, interpolation='nearest')  # Display pupil magnitude
            plt.axis('off')  # Hide axis
            plt.title('bottom pupil magnitude ' + str(i))
            plt.colorbar()  # Color bar

        # Plot phases similarly
        fig = plt.figure(figsize=[20, 8])
        for i in range(0, Nchannel):
            ax = fig.add_subplot(2, 4, i + 1)
            pupil_phase = np.angle(f.res['channel'+str(i)].pupil1)  # Top pupil phase
            plt.imshow(pupil_phase, interpolation='nearest')  # Display pupil phase
            plt.axis('off')  # Hide axis
            plt.title('top pupil phase ' + str(i))
            plt.colorbar()  # Color bar

            ax = fig.add_subplot(2, 4, i + 5)
            pupil_phase = np.angle(f.res['channel'+str(i)].pupil2)  # Bottom pupil phase
            plt.imshow(pupil_phase, interpolation='nearest')  # Display pupil phase
            plt.axis('off')  # Hide axis
            plt.title('bottom pupil phase ' + str(i))
            plt.colorbar()  # Color bar

    return  # End of function

def showzernike(f, p, index=None):
    """
    Displays Zernike coefficients and their corresponding pupil functions for the specified channel type.

    Parameters:
    f : object
        Contains data regarding the results of the analysis.
    p : object
        Contains parameters such as channel type that dictate how the data should be interpreted.
    index : int, optional
        Index to specify which Zernike coefficients to visualize.

    - Depending on the channel type, it visualizes both magnitude and phase components of Zernike polynomials.
    """
    n_max = p.option.model.n_max  # Maximum order for Zernike polynomials
    Nk = (n_max + 1) * (n_max + 2) // 2  # Total number of Zernike coefficients

    # Indices for specific Zernike modes to display
    indz = np.array([4, 5, 6, 7, 10, 21])
    textstr = [None] * 6  # Prepare the text strings for display
    textstr[0] = r'$\mathrm{D \ astigmatism}$'
    textstr[1] = r'$\mathrm{astigmatism}$'
    textstr[2] = r'$\mathrm{V \ coma}$'
    textstr[3] = r'$\mathrm{H \ coma}$'
    textstr[4] = r'$\mathrm{spherical}$'
    textstr[5] = r'$\mathrm{2nd \ spherical}$'

    # Create a mask of valid Zernike indices
    mask = indz < Nk
    Nzk = np.sum(mask)  # Number of valid Zernike modes

    # Handling for single channel type
    if p.channeltype == 'single':
        fig = plt.figure(figsize=[10, 8])
        if index is None:
            zcoeff = f.res.zernike_coeff  # Obtain Zernike coefficients
        else:
            zcoeff = f.res.zernike_coeff[:, index]  # Obtain coefficients for specified index

        # Determine the aperture based on pupil shape
        if len(f.res.pupil.shape) > 2:
            aperture = np.float32(np.abs(f.res.pupil[0]) > 0.0)  # For 3D pupil
        else:
            aperture = np.float32(np.abs(f.res.pupil) > 0.0)  # For 2D pupil

        ax = fig.add_subplot(2, 1, 1)
        ax.plot(zcoeff.transpose(), '.-')  # Plot Zernike coefficients
        ax.plot(indz[mask], zcoeff[1, indz[mask]], 'ko', markersize=6, mfc='none')  # Highlight specific modes
        plt.xlabel('zernike polynomial')
        plt.ylabel('coefficient')
        plt.legend(['pupil magnitude', 'pupil phase'])  # Legend for the plot

        ax1 = fig.add_subplot(2, 1, 2)
        tstr = ''
        for i in range(0, Nzk):
            tstr = '\n'.join((tstr, textstr[i] + '=%.2f' % (zcoeff[1][indz[i]], )))  # Create display text
        tstr = tstr[1:]  # Remove leading newline
        bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)  # Text box style
        ax1.text(0.03, 0.9, tstr, fontsize=12, bbox=bbox,
                 transform=ax1.transAxes, horizontalalignment='left', verticalalignment='top')
        ax1.set_axis_off()  # Hide axis for this plot

        Zk = f.res.zernike_polynomial  # Get Zernike polynomial data

        # Compute the pupil functions using Zernike coefficients and the aperture
        pupil_mag = np.sum(Zk * zcoeff[0].reshape((-1, 1, 1)), axis=0) * aperture  # Magnitude
        pupil_phase = np.sum(Zk[4:] * zcoeff[1][4:].reshape((-1, 1, 1)), axis=0) * aperture  # Phase

        # Create a figure for magnitude and phase
        fig = plt.figure(figsize=[12, 5])
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(pupil_mag, interpolation='nearest')  # Display pupil magnitude
        plt.colorbar()  # Show color bar
        plt.title('pupil magnitude', fontsize=20)  # Title for magnitude plot
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(pupil_phase, interpolation='nearest')  # Display pupil phase
        plt.colorbar()  # Show color bar
        plt.title('pupil phase', fontsize=20)  # Title for phase plot

    # Handling for multi-channel type
    elif p.channeltype == 'multi':
        # Get the number of channels from the PSF data shape
        Nchannel = f.rois.psf_data.shape[0]
        # Create a figure for plotting
        fig = plt.figure(figsize=[12, 6])
        # Create subplots for pupil magnitude and phase
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax5 = fig.add_subplot(2, 2, 3)
        # Create figures for pupil magnitude and phase
        fig1 = plt.figure(figsize=[5 * Nchannel, 4])
        fig2 = plt.figure(figsize=[5 * Nchannel, 4])
        # Get Zernike polynomial data for channel 0
        Zk = f.res.channel0.zernike_polynomial
        
        # Loop over each channel
        for i in range(0, Nchannel):
            # Select Zernike coefficients based on index, if provided
            if index is None:
                zcoeff = f.res['channel' + str(i)].zernike_coeff
            else:
                zcoeff = f.res['channel' + str(i)].zernike_coeff[:, index]
            
            # Determine the aperture based on the pupil shape
            if len(f.res['channel' + str(i)].pupil.shape) > 2:
                aperture = np.float32(np.abs(f.res['channel' + str(i)].pupil[0]) > 0.0)
            else:
                aperture = np.float32(np.abs(f.res['channel' + str(i)].pupil) > 0.0)

            # Plot the Zernike coefficients for pupil magnitude and phase
            line, = ax1.plot(zcoeff[0], '.-')
            ax2.plot(zcoeff[1], '.-')
            ax1.set_xlabel('zernike polynomial')
            ax1.set_ylabel('coefficient')
            ax1.set_title('pupil magnitude')
            ax2.set_title('pupil phase')
            line.set_label('channel ' + str(i))
            ax1.legend()
            # Mark the coefficients based on the provided index and mask
            if i == Nchannel - 1:
                ax1.plot(indz[mask], zcoeff[0][indz[mask]], 'ko', markersize=6, mfc='none')
                ax2.plot(indz[mask], zcoeff[1][indz[mask]], 'ko', markersize=6, mfc='none')

            # Append information to textstr for later display
            for k in range(0, Nzk):
                textstr[k] = '\n'.join((textstr[k], r'$\mathrm{ch}%d=%.2f$' % (i, zcoeff[1][indz[k]],),))

            # Create a subplot for pupil magnitude
            ax3 = fig1.add_subplot(1, Nchannel, i + 1)
            # Calculate pupil magnitude using Zernike coefficients
            pupil_mag = np.sum(Zk * zcoeff[0].reshape((-1, 1, 1)), axis=0) * aperture
            h = ax3.imshow(pupil_mag, interpolation='nearest')
            ax3.axis('off')
            ax3.set_title('pupil magnitude ' + str(i), fontsize=20)
            fig1.colorbar(h, ax=ax3)

            # Create a subplot for pupil phase
            ax4 = fig2.add_subplot(1, Nchannel, i + 1)
            # Calculate pupil phase using Zernike coefficients
            pupil_phase = np.sum(Zk[4:] * zcoeff[1][4:].reshape((-1, 1, 1)), axis=0) * aperture
            h1 = ax4.imshow(pupil_phase, interpolation='nearest')
            ax4.axis('off')
            ax4.set_title('pupil phase ' + str(i), fontsize=20)
            fig2.colorbar(h1, ax=ax4)
        
        # Create a bounding box for text display
        bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
        # Display the appended information in ax5
        for k in range(0, len(textstr)):
            ax5.text(0.01 + k * 0.35, 0.9, textstr[k], fontsize=12, bbox=bbox,
                     transform=ax5.transAxes, horizontalalignment='left', verticalalignment='top')
        ax5.set_axis_off()
    
    # Handling for 4pi channel type
    elif p.channeltype == '4pi':
        # Get the number of channels from the PSF data shape
        Nchannel = f.rois.psf_data.shape[0]
        # Create a figure for plotting
        fig = plt.figure(figsize=[12, 10])
        # Create subplots for pupil magnitude and phase
        ax1 = fig.add_subplot(3, 2, 1)
        ax2 = fig.add_subplot(3, 2, 2)
        ax3 = fig.add_subplot(3, 2, 3)
        ax4 = fig.add_subplot(3, 2, 4)
        ax5 = fig.add_subplot(3, 2, 5)

        # Loop through the Zernike coefficients for each channel
        for k in range(0, Nzk):
            textstr[k] = '\n'.join((textstr[k], r'$\mathrm{upper}=%.2f$' % (f.res.channel0.zernike_coeff_phase[0][indz[k]],),
                                    r'$\mathrm{lower}=%.2f$' % (f.res.channel0.zernike_coeff_phase[1][indz[k]],)))

        # Loop over each channel
        for i in range(0, Nchannel):
            # Get the Zernike coefficients for magnitude and phase
            zcoeff_mag = f.res['channel' + str(i)].zernike_coeff_mag
            zcoeff_phase = f.res['channel' + str(i)].zernike_coeff_phase

            # Plot the upper pupil magnitude
            line, = ax1.plot(zcoeff_mag[0], '.-')    
            ax2.plot(zcoeff_phase[0], '.-')
            ax2.set_ylim((-0.6, 0.6))
            ax3.plot(zcoeff_mag[1], '.-')
            ax4.plot(zcoeff_phase[1], '.-')
            ax4.set_ylim((-0.6, 0.6))
            ax3.set_xlabel('zernike polynomial')
            ax3.set_ylabel('coefficient')
            ax1.set_title('upper pupil magnitude')
            ax2.set_title('upper pupil phase')
            ax3.set_title('lower pupil magnitude')
            ax4.set_title('lower pupil phase')
            line.set_label('channel ' + str(i))
            ax1.legend()
            # Mark the coefficients based on the provided index and mask
            if i == Nchannel - 1:
                ax1.plot(indz[mask], zcoeff_mag[0][indz[mask]], 'ko', markersize=6, mfc='none')
                ax2.plot(indz[mask], zcoeff_phase[0][indz[mask]], 'ko', markersize=6, mfc='none')
                ax3.plot(indz[mask], zcoeff_mag[1][indz[mask]], 'ko', markersize=6, mfc='none')
                ax4.plot(indz[mask], zcoeff_phase[1][indz[mask]], 'ko', markersize=6, mfc='none')

        # Create a bounding box for text display
        bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
        # Display the appended information in ax5
        for k in range(0, len(textstr)):
            ax5.text(0.01 + k * 0.35, 0.9, textstr[k], fontsize=12, bbox=bbox,
                     transform=ax5.transAxes, horizontalalignment='left', verticalalignment='top')
        ax5.set_axis_off()

        # Calculate the aperture based on pupil data
        aperture = np.float32(np.abs(f.res.channel0.pupil1) > 0.0)
        # Get Zernike polynomial data for channel 0
        Zk = f.res.channel0.zernike_polynomial
        fig = plt.figure(figsize=[20, 8])

        # Create plots for upper pupil magnitude
        for i in range(0, Nchannel):
            ax = fig.add_subplot(2, 4, i + 1)
            pupil_mag = np.sum(Zk * f.res['channel' + str(i)].zernike_coeff_mag[0].reshape((-1, 1, 1)), axis=0) * aperture
            plt.imshow(pupil_mag, interpolation='nearest')
            plt.axis('off')
            plt.title('upper pupil magnitude ' + str(i), fontsize=20)
            plt.colorbar()
            ax = fig.add_subplot(2, 4, i + 5)
            pupil_mag = np.sum(Zk * f.res['channel' + str(i)].zernike_coeff_mag[1].reshape((-1, 1, 1)), axis=0) * aperture
            plt.imshow(pupil_mag, interpolation='nearest')
            plt.axis('off')
            plt.title('lower pupil magnitude ' + str(i), fontsize=20)
            plt.colorbar()
        
        fig = plt.figure(figsize=[20, 8])
        # Create plots for upper pupil phase
        for i in range(0, Nchannel):
            ax = fig.add_subplot(2, 4, i + 1)
            pupil_phase = np.sum(Zk[4:] * f.res['channel' + str(i)].zernike_coeff_phase[0][4:].reshape((-1, 1, 1)), axis=0) * aperture
            plt.imshow(pupil_phase, interpolation='nearest')
            plt.axis('off')
            plt.title('upper pupil phase ' + str(i), fontsize=20)
            plt.colorbar()
            ax = fig.add_subplot(2, 4, i + 5)
            pupil_phase = np.sum(Zk[4:] * f.res['channel' + str(i)].zernike_coeff_phase[1][4:].reshape((-1, 1, 1)), axis=0) * aperture
            plt.imshow(pupil_phase, interpolation='nearest')
            plt.axis('off')
            plt.title('lower pupil phase ' + str(i), fontsize=20)
            plt.colorbar()
        # Show all figures created
        plt.show()
    return


def showzernikemap(f, p, index=None):
    """Display the Zernike map based on the channel type.

    Args:
        f: The data structure containing the results and PSF data.
        p: An object containing parameters including channel type.
        index: An optional index for selecting specific Zernike coefficients.
    """
    # Handling for single-channel type
    if p.channeltype == 'single':
        zmap = f.res.zernike_map  # Zernike map
        zcoeff = f.res.zernike_coeff  # Zernike coefficients
        pupil = f.res.pupil  # Pupil function
        Zk = f.res.zernike_polynomial  # Zernike polynomial
        # Call the zernikemap function to visualize the data
        zernikemap(f, index, zmap, zcoeff, pupil, Zk)

    # Handling for multi-channel type
    if p.channeltype == 'multi':
        Nchannel = f.rois.psf_data.shape[0]  # Get number of channels
        # Loop over each channel
        for i in range(0, Nchannel):
            print('channel ' + str(i))  # Print current channel number
            zmap = f.res['channel' + str(i)].zernike_map  # Zernike map for current channel
            zcoeff = f.res['channel' + str(i)].zernike_coeff  # Zernike coefficients for current channel
            pupil = f.res['channel' + str(i)].pupil  # Pupil function for current channel
            Zk = f.res['channel' + str(i)].zernike_polynomial  # Zernike polynomial for current channel
            # Call the zernikemap function to visualize the data
            zernikemap(f, index, zmap, zcoeff, pupil, Zk)

def zernikemap(f, index, zmap, zcoeff, pupil, Zk):
    """Visualize the Zernike map and coefficients.

    Args:
        f: The data structure containing the results.
        index: Indices for plotting specific Zernike coefficients.
        zmap: The Zernike map to be displayed.
        zcoeff: The Zernike coefficients.
        pupil: The pupil function used in the calculations.
        Zk: The Zernike polynomial data.
    """
    # Default index if none is provided
    if index is None:
        index = [4, 5, 6, 7, 10, 11, 12, 15, 16, 21]
        mask = np.array(index) < (zcoeff.shape[-1] - 1)  # Create a mask based on the shape of coefficients
        index = np.array(index)[mask]  # Filter the index based on mask

    # Create a figure for coefficient plotting
    fig = plt.figure(figsize=[16, 4])
    ax = fig.add_subplot(1, 2, 1)
    plt.plot(zcoeff[0].transpose(), 'k', alpha=0.1)  # Plot the magnitude coefficients
    plt.plot(index, zcoeff[0, 0, index], 'ro')  # Highlight specific indices
    plt.xlabel('zernike polynomial')
    plt.ylabel('coefficient')
    plt.title('pupil magnitude')
    plt.legend(['bead 1'])

    ax = fig.add_subplot(1, 2, 2)
    plt.plot(zcoeff[1].transpose(), 'k', alpha=0.1)  # Plot the phase coefficients
    plt.plot(index, zcoeff[1, 0, index], 'ro')  # Highlight specific indices
    plt.xlabel('zernike polynomial')
    plt.ylabel('coefficient')
    plt.title('pupil phase')
    plt.legend(['bead 1'])

    # Determine the aperture based on the shape of the pupil
    if len(pupil.shape) > 2:
        aperture = np.float32(np.abs(pupil[0]) > 0.0)
    else:
        aperture = np.float32(np.abs(pupil) > 0.0)
    imsz = np.array(f.rois.image_size)  # Get image size from data structure
    
    # Calculate scale for mapping
    scale = (imsz[-2:] - 1) / (np.array(zmap.shape[-2:]) - 1)

    N = len(index)  # Number of indices
    Nx = 4  # Number of columns
    Ny = N // Nx + 1  # Number of rows
    # Create a grid specification for subplots
    fig = plt.figure(figsize=[4.5 * Nx, 7 * Ny])
    spec = gridspec.GridSpec(ncols=Nx, nrows=2 * Ny,
                             width_ratios=list(np.ones(Nx)), wspace=0.1,
                             hspace=0.2, height_ratios=list(np.ones(2 * Ny)))

    # Define names for aberrations based on index
    abername = [''] * np.max([zcoeff.shape[-1], 22])
    abername[3] = 'defocus'
    abername[4] = 'D astigmatism'
    abername[5] = 'astigmatism'
    abername[6] = 'V coma'
    abername[7] = 'H coma'
    abername[10] = 'spherical'
    abername[11] = '2nd ast.'
    abername[12] = '2nd D ast.'
    abername[15] = '2nd H coma'
    abername[16] = '2nd V coma'
    abername[21] = '2nd spherical'

    # Loop over indices to create Zernike map visualizations
    for i, id in enumerate(index):
        j = i // Nx  # Determine the row
        ax = fig.add_subplot(spec[i + j * Nx])
        plt.imshow(zmap[1, id], cmap='twilight', interpolation='nearest')  # Display the Zernike map
        plt.axis('off')
        plt.title('(' + str(id) + ') ' + abername[id], fontsize=16)  # Set title with aberration name
        plt.colorbar()  # Add color bar for reference

        ax = fig.add_subplot(spec[i + (j + 1) * Nx])
        plt.imshow(Zk[id] * aperture, cmap='viridis', interpolation='nearest')  # Display the pupil function
        plt.axis('off')
        plt.colorbar()  # Add color bar for reference
    plt.show()  # Show all figures created

def showpsfvsdata(f, p, index):
    """Compare PSF data with fitted PSF data and plot correlation.

    Args:
        f: The data structure containing the PSF data and fit.
        p: An object containing parameters including channel type.
        index: Index of the specific PSF data to compare.
    """
    psf_data = f.rois.psf_data  # PSF data from results
    psf_fit = f.rois.psf_fit  # Fitted PSF data
    
    # Handling for single-channel type
    if p.channeltype == 'single':
        im1 = psf_data[index]  # Retrieve PSF data for the specified index
        im2 = psf_fit[index]  # Retrieve fitted PSF data for the specified index
        psfcompare(im1, im2, p.pixel_size.z)  # Compare and visualize the PSF data

    # Handling for multi-channel type
    else:
        Nchannel = psf_data.shape[0]  # Number of channels
        # Loop over each channel for comparison
        for ch in range(0, Nchannel):
            if p.channeltype == '4pi':
                im1 = psf_data[ch, index, 0]  # Retrieve PSF data for the specified index
                im2 = psf_fit[ch, index, 0]
            else:
                im1 = psf_data[ch, index]
                im2 = psf_fit[ch, index]
            print('channel ' + str(ch))  # Output the channel being processed
            psfcompare(im1, im2, p.pixel_size.z)  # Compare PSF data for each channel

    # Attempt to retrieve correlation data, with fallback
    try:
        cor = f.res.cor  # Try to get correlation from results
    except:
        cor = f.res.channel1.cor  # Fallback to channel 1 correlation

    imsz = f.rois.image_size  # Get image size for plotting
    fig = plt.figure(figsize=[4, 4])  # Create a new figure for plotting
    # Plot correlation data
    plt.plot(cor[index, -1], cor[index, -2], 'ro')  # Plot the correlation point
    plt.xlim(0, imsz[-1])  # Set x-axis limits
    plt.ylim(0, imsz[-2])  # Set y-axis limits
    plt.xlabel('x (pixel)')  # Label for x-axis
    plt.ylabel('y (pixel)')  # Label for y-axis
    plt.legend(['bead' + str(index)])  # Legend for the plot
    plt.show()  # Display the plot
    return  # End of function


def psfcompare(im1, im2, pz):
    """Compare two PSF images and display them side by side.

    Args:
        im1: The first PSF image to compare.
        im2: The second PSF image to compare.
        pz: The pixel size in z direction.
    """
    Nz = im1.shape[0]  # Number of z slices in the PSF image
    zrange = np.linspace(-Nz/2 + 0.5, Nz/2 - 0.5, Nz) * pz  # Calculate z range based on pixel size
    zind = range(0, Nz, 4)  # Indices for the z slices to display
    cc = im1.shape[-1] // 2  # Central index for xz slice
    N = len(zind) + 1  # Total number of subplots

    fig = plt.figure(figsize=[3 * N, 6])  # Create a figure for displaying images
    for i, id in enumerate(zind):
        ax = fig.add_subplot(2, N, i + 1)  # Create subplot for first image
        plt.imshow(im1[id], cmap='twilight')  # Display the first PSF image
        plt.axis('off')  # Hide axis
        plt.title(str(np.round(zrange[id], 2)) + r'$\ \mu$m', fontsize=30)  # Title with z position

        ax = fig.add_subplot(2, N, i + 1 + N)  # Create subplot for second image
        plt.imshow(im2[id], cmap='twilight')  # Display the second PSF image
        plt.axis('off')  # Hide axis

    # Display xz slices for both images
    ax = fig.add_subplot(2, N, N)  # Create xz subplot for first image
    plt.imshow(im1[:, cc], cmap='twilight')  # Display xz slice of first image
    plt.axis('off')  # Hide axis
    plt.title('xz', fontsize=30)  # Title for xz slice
    plt.colorbar()  # Show colorbar for reference

    ax = fig.add_subplot(2, N, 2 * N)  # Create xz subplot for second image
    plt.imshow(im2[:, cc], cmap='twilight')  # Display xz slice of second image
    plt.axis('off')  # Hide axis
    plt.colorbar()  # Show colorbar for reference

    plt.show()  # Display the comparison plot
    return  # End of function


def showpsfvsdata_insitu(f, p):
    """Compare average PSF data with fitted PSF data in situ.

    Args:
        f: The data structure containing the PSF data and fit.
        p: An object containing parameters including channel type.
    """
    # Handling for single-channel type
    if p.channeltype == 'single':
        rois = f.rois.psf_data  # Get PSF data
        I_model = f.res.I_model  # Get fitted model PSF data
        zf = f.res.pos[:, 0]  # Get z positions
        Nz = I_model.shape[0]  # Number of z slices
        edge = np.real(f.res.zoffset) + range(0, Nz + 1)  # Calculate edges for digitization
        ind = np.digitize(zf, edge.flatten())  # Digitize z positions into bins
        rois_avg = np.zeros(I_model.shape)  # Initialize array for average PSF data
        
        # Average PSF data for each z slice
        for ii in range(1, Nz + 1):
            mask = (ind == ii)  # Create a mask for the current bin
            if sum(mask) > 0:
                rois_avg[ii - 1] = np.mean(rois[mask], axis=0)  # Average over the bin

        psfcompare(rois_avg, I_model, p.pixel_size.z)  # Compare averaged PSF with model PSF

    # Handling for multi-channel type
    else:
        Nchannel = f.rois.psf_data.shape[0]  # Get number of channels
        zoffset = f.res.channel0.zoffset  # Get z offset for the first channel
        for ch in range(0, Nchannel):
            rois = f.rois.psf_data[ch]  # Get PSF data for the current channel
            I_model = f.res['channel' + str(ch)].I_model  # Get fitted model for the channel
            if p.channeltype == '4pi':
                I_model = f.res['channel' + str(ch)].psf_model  # Use PSF model for 4pi type
            
            zf = f.res.channel0.pos[:, 0]  # Get z positions from the first channel
            Nz = I_model.shape[0]  # Get number of z slices
            edge = np.real(zoffset) + range(0, Nz + 1)  # Calculate edges for digitization
            ind = np.digitize(zf, edge.flatten())  # Digitize z positions into bins
            rois_avg = np.zeros(I_model.shape)  # Initialize array for average PSF data
            
            # Average PSF data for each z slice
            for ii in range(1, Nz + 1):
                mask = (ind == ii)  # Create a mask for the current bin
                if sum(mask) > 0:
                    rois_avg[ii - 1] = np.mean(rois[mask], axis=0)  # Average over the bin
            print('channel ' + str(ch))  # Output the channel being processed
            psfcompare(rois_avg, I_model, p.pixel_size.z)  # Compare averaged PSF with model PSF for the channel

    return  # End of function


def showlocalization(f, p):
    """Display localization results and biases.

    Args:
        f: The data structure containing localization results.
        p: An object containing parameters including channel type.
    """
    loc = f.locres.loc  # Retrieve localization results
    plotlocbias(loc, p)  # Plot localization bias based on results
    # If fine localization data is available, plot it as well
    if hasattr(f.locres, 'loc_FD'):
        loc = f.locres.loc_FD  # Retrieve fine localization results
        plotlocbias(loc, p)  # Plot the bias for fine localization

    return  # End of function


def plotlocbias(loc, p):
    """Plot localization bias as a function of z slice.

    Args:
        loc: The localization data containing x, y, z positions.
        p: An object containing parameters including pixel sizes.
    """
    Nz = loc.z.shape[1]  # Number of z slices
    fig = plt.figure(figsize=[16, 4])  # Create figure for bias plots
    spec = gridspec.GridSpec(ncols=3, nrows=1,
                             width_ratios=[3, 3, 3], wspace=0.3,
                             hspace=0.3, height_ratios=[1])  # Grid specification for subplots
    
    # Plot x bias vs z slice
    ax = fig.add_subplot(spec[0])  
    plt.plot(loc.x.transpose() * p.pixel_size.x * 1e3, 'k', alpha=0.1)  # Plot all x biases
    plt.plot(loc.x[0] * 0.0, 'r')  # Plot average x bias
    ax.set_xlabel('z slice')  # Label x-axis
    ax.set_ylabel('x bias (nm)')  # Label y-axis
    
    # Plot y bias vs z slice
    ax = fig.add_subplot(spec[1])
    plt.plot(loc.y.transpose() * p.pixel_size.y * 1e3, 'k', alpha=0.1)  # Plot all y biases
    plt.plot(loc.y[0] * 0.0, 'r')  # Plot average y bias
    ax.set_xlabel('z slice')  # Label x-axis
    ax.set_ylabel('y bias (nm)')  # Label y-axis
    
    # Plot z bias vs z slice
    ax = fig.add_subplot(spec[2])
    bias_z = (loc.z - np.linspace(0, Nz - 1, Nz)) * p.pixel_size.z * 1e3  # Calculate z bias
    plt.plot(bias_z.transpose(), 'k', alpha=0.1)  # Plot all z biases
    plt.plot(loc.z[0] * 0.0, 'r')  # Plot average z bias
    ax.set_xlabel('z slice')  # Label x-axis
    ax.set_ylabel('z bias (nm)')  # Label y-axis
    # Set y-axis limits based on quantiles for better visualization
    plt.ylim([np.maximum(np.quantile(bias_z[:, 2:-2], 0.001), -300), np.minimum(np.quantile(bias_z[:, 2:-2], 0.999), 300)])
    plt.show()  # Display the bias plots
    return  # End of function


def showtransform(f):
    """Show transformation results between reference and target positions.

    Args:
        f: The data structure containing transformation results.
    """
    Nchannel = f.rois.psf_data.shape[0]  # Get number of channels
    ref_pos = f.res.channel0.pos  # Get reference positions from channel 0
    dxy = f.res.xyshift  # Get xy shift for transformations
    fig = plt.figure(figsize=[5 * Nchannel, 10])  # Create figure for displaying transformations
    spec = gridspec.GridSpec(ncols=Nchannel, nrows=2,
                             width_ratios=list(np.ones(Nchannel)), wspace=0.3,
                             hspace=0.2, height_ratios=[1, 1])  # Grid specification for subplots

    # Prepare reference positions for transformations
    cor_ref = np.concatenate((ref_pos[:, 1:], np.ones((ref_pos.shape[0], 1))), axis=1)

    # Loop through each channel for transformation visualization
    for i in range(1, Nchannel):
        pos = f.res['channel' + str(i)].pos  # Get positions for the current channel
        # Prepare target transformations based on the number of channels
        if Nchannel < 3:
            cor_target = np.matmul(cor_ref - f.res.imgcenter, f.res.T)[..., :-1] + f.res.imgcenter[:-1]
        else:
            cor_target = np.matmul(cor_ref - f.res.imgcenter, f.res.T[i - 1])[..., :-1] + f.res.imgcenter[:-1]

        ax = fig.add_subplot(spec[i])  # Create subplot for reference positions
        plt.plot(ref_pos[:, -1], ref_pos[:, -2], '.')  # Plot reference positions
        plt.plot(pos[:, -1] - dxy[i][-1], pos[:, -2] - dxy[i][-2], 'o', markersize=8, mfc='none')  # Plot target positions
        plt.plot(f.res.imgcenter[1], f.res.imgcenter[0], '*')  # Plot image center
        ax.set_xlabel('x (pixel)')  # Label x-axis
        ax.set_ylabel('y (pixel)')  # Label y-axis
        plt.title('channel' + str(i))  # Title for current channel

        ax1 = fig.add_subplot(spec[Nchannel + i])  # Create subplot for transformed positions
        plt.plot(cor_target[:, -1], cor_target[:, -2], '.')  # Plot transformed reference positions
        plt.plot(pos[:, -1], pos[:, -2], 'o', markersize=8, mfc='none')  # Plot actual target positions
        plt.plot(f.res.imgcenter[1], f.res.imgcenter[0], '*')  # Plot image center
        ax1.set_xlabel('x (pixel)')  # Label x-axis
        ax1.set_ylabel('y (pixel)')  # Label y-axis

    ax.legend(['ref', 'target', 'center'])  # Legend for reference plot
    ax1.legend(['ref_trans', 'target', 'center'])  # Legend for transformed plot


def showpsf(f, p):
    """Display PSF images based on the channel type.

    Args:
        f: The data structure containing PSF results.
        p: An object containing parameters including channel type.
    """
    # Handling for single-channel type
    if p.channeltype == 'single':
        im1 = f.res.I_model  # Get fitted model PSF data
        psfdisp(im1, p.pixel_size.z)  # Display PSF data

    # Handling for multi-channel type
    else:
        Nchannel = f.rois.psf_data.shape[0]  # Get number of channels
        for ch in range(0, Nchannel):
            if p.channeltype == '4pi':
                im1 = f.res['channel' + str(ch)].psf_model  # Get PSF model for 4pi type
            else:
                im1 = f.res['channel' + str(ch)].I_model  # Get fitted model for other types
            print('channel ' + str(ch))  # Output the channel being processed
            psfdisp(im1, p.pixel_size.z)  # Display PSF data

    return  # End of function


def psfdisp(im1, pz):
    """Display a single PSF image and its slices.

    Args:
        im1: The PSF image to display.
        pz: The pixel size in the z direction.
    """
    Nz = im1.shape[0]  # Number of z slices in the PSF image
    zrange = np.linspace(-Nz/2 + 0.5, Nz/2 - 0.5, Nz) * pz  # Calculate z range

    zind = range(0, Nz, 4)  # Indices for z slices to display
    cc = im1.shape[-1] // 2  # Central index for xz slice
    N = len(zind) + 1  # Total number of subplots

    fig = plt.figure(figsize=[3 * N, 3])  # Create a figure for displaying images
    for i, id in enumerate(zind):
        ax = fig.add_subplot(1, N, i + 1)  # Create a subplot for the PSF image
        plt.imshow(im1[id], cmap='twilight')  # Display the PSF image
        plt.title(str(np.round(zrange[id], 2)) + r'$\ \mu$m', fontsize=30)  # Title with z position
        plt.axis('off')  # Hide axes

    ax = fig.add_subplot(1, N, N)  # Create a subplot for xz slice
    plt.imshow(im1[:, cc], cmap='twilight')  # Display xz slice of the PSF image
    plt.axis('off')  # Hide axes
    plt.title('xz', fontsize=30)  # Title for xz slice
    plt.colorbar()  # Show colorbar for reference
    plt.show()  # Display the PSF image
    return  # End of function

def showcoord(f, p):
    """
    Displays the coordinates of points in a plot based on the specified channel type.
    
    Parameters:
    f : object
        An object that contains the results and the necessary data for plotting.
    p : object
        An object that contains the channel type. It has an attribute 'channeltype' 
        which determines if the data is single-channel or multi-channel.
    """
    
    # Check if the channel type is 'single'
    if p.channeltype == 'single':
        # Create a figure for single channel plotting with a specified size
        fig = plt.figure(figsize=[5, 5])
        
        # Retrieve the coordinates of the results
        cor = f.res.cor
        cor_all = f.res.cor_all
        
        # Plot all coordinates as dots
        plt.plot(cor_all[:, -1], cor_all[:, -2], '.')
        # Plot selected coordinates as hollow circles
        plt.plot(cor[:, -1], cor[:, -2], 'o', markersize=8, mfc='none')
        
        # Set labels for x and y axes
        plt.xlabel('x (pixel)')
        plt.ylabel('y (pixel)')
        
        # Create a legend for the plot
        plt.legend(['all', 'selected'])
    else:
        # If channel type is not single, determine the number of channels
        Nchannel = f.rois.psf_data.shape[0]
        
        # Create a figure with a width proportional to the number of channels
        fig = plt.figure(figsize=[5 * Nchannel, 5])
        # Define a grid for subplots based on the number of channels
        spec = gridspec.GridSpec(ncols=Nchannel, nrows=1,
                                 width_ratios=list(np.ones(Nchannel)), wspace=0.3,
                                 hspace=0.2, height_ratios=[1])

        # Loop through each channel to plot the coordinates
        for i in range(0, Nchannel):
            # Retrieve coordinates for the specific channel
            cor = f.res['channel' + str(i)].cor
            cor_all = f.res['channel' + str(i)].cor_all
            
            # Create a subplot for the current channel
            ax = fig.add_subplot(spec[i])
            # Plot all coordinates as dots for the channel
            plt.plot(cor_all[:, -1], cor_all[:, -2], '.')
            # Plot selected coordinates as hollow circles for the channel
            plt.plot(cor[:, -1], cor[:, -2], 'o', markersize=8, mfc='none')
            
            # Set labels for the axes of the subplot
            ax.set_xlabel('x (pixel)')
            ax.set_ylabel('y (pixel)')
            # Set the title for the current channel
            plt.title('channel' + str(i))

        # Create a legend for the subplots
        ax.legend(['all', 'selected'])