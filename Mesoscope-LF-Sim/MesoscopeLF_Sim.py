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
The script provides a Tkinter-based GUI application for calculating and visualizing
parameters related to a Light-Field Computational Mesoscope. It includes functions
for calculating Equivalent Index of Spherical (EIS) numbers, Focal Magnification (FM),
resolution, field of view (FOV), and depth of field (DOF) based on user inputs. The
application allows users to input various parameters such as numerical aperture, MLA
pitch, pixel size, wavelength, relay magnification, MO focal length, and MLA focal
length. It also provides plotting capabilities using matplotlib to visualize the
calculated parameters. The GUI includes options to save plots, calculate focal lengths,
and display information about the application.
"""
"""
Calculates the EIS (Equivalent Index of Spherical) number and FM (Focal Magnification) values based on the provided input parameters.

Parameters:
x (float): The x-coordinate values for calculations.
y (float): The y-coordinate values for calculations.
nmax (int): The maximum number of iterations for calculations.

Returns:
EIS (array): An array of EIS values calculated based on the input parameters.
FM (array): An array of FM values calculated based on the input parameters.

def EISnum(x, y, nmax):
    # Function implementation
    pass

The Tkinter application for a Fourier Light-Field Microscope (FLFM) parameter. It includes various input fields for different parameters such as numerical aperture, MLA pitch, pixel size, wavelength, relay magnification, MO focal length, and MLA focal length. Imports the tkinter library, which provides a set of tools for creating graphical user interface (GUI) applications in Python. The tkinter library includes a wide range of widgets and tools for building desktop applications, including windows, buttons, menus, and more. It is a cross-platform library, allowing the same code to run on multiple operating systems.
"""
import tkinter  # Import the tkinter library for creating GUI applications
import time  # Import the time library for time-related functions
from tkinter import filedialog  # Import filedialog for file save dialog
from tkinter import simpledialog  # Import simpledialog for simple dialog boxes
import os  # Import os for operating system-dependent functionality
import matplotlib.pyplot as plt  # Import matplotlib for plotting graphs
import numpy as np  # Import numpy for numerical operations
from pandas import DataFrame  # Import DataFrame from pandas for data manipulation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Import FigureCanvasTkAgg for embedding matplotlib in Tkinter
from matplotlib.figure import Figure  # Import Figure to create a new figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)  # Import additional TkAgg functionality
from PIL import Image  # Import Image for image handling
from tkinter import ttk  # Import ttk for themed widgets

# Set the plotting style to 'seaborn-white'
plt.style.use('seaborn-white')

def SaveIm():
    """Function to save the current figure as a PNG file.
    
    Prompts the user to select a location and filename to save the figure.
    """
    answer = filedialog.asksaveasfilename(parent=master, initialdir=os.getcwd(), 
                                            title='Select folder and name', filetypes=types)
    # Save the figure as a PNG file with a DPI of 300
    fig1.savefig(answer + '.png', dpi=300)

def refresh(self, self1):
    """Clears the contents of the provided axis or figure.
    
    Parameters:
    self : matplotlib axis or figure
        The axis or figure to be cleared.
    self1 : matplotlib axis or figure
        The axis or figure to be refreshed.
    """
    self1.clear()

def EISnum(x, y, nmax):
    """Calculates the EIS (Equivalent Index of Spherical) number and FM (Focal Magnification) values.
    
    Parameters:
    x : float
        The x-coordinate values for calculations.
    y : float
        The y-coordinate values for calculations.
    nmax : int
        The maximum number of iterations for calculations.
    
    Returns:
    EIS : array
        An array of EIS values calculated based on the input parameters.
    FM : array
        An array of FM values calculated based on the input parameters.
    """
    numa = float(NA.get())  # Get numerical aperture from input
    pit = float(pitch.get())  # Get pitch from input
    M = float(Mag.get())  # Get magnification from input
    N = 2 * M * y * np.tan(np.arcsin(numa)) / (pit)  # Calculate N based on formulas

    N = np.trunc(N)  # Truncate N to find integral values

    FM = np.zeros(int(max(N)) - int(min(N)))  # Initialize FM array
    EIS = np.zeros(int(max(N)) - int(min(N)))  # Initialize EIS array

    n = 0  # Counter for valid N values
    s = 0  # Index for FM and EIS arrays
    for x in range(0, nmax - 1):
        con = N[x + 1] - N[x]  # Calculate the difference between consecutive N values
        
        if con != 0:  # If the difference is not zero
            n += 1  # Increment the valid N counter
            
            EIS[s] = int(min(N)) + n  # Store the EIS value
            FM[s] = y[x]  # Store the FM value
            
            s += 1  # Move to the next index in FM and EIS arrays
    return EIS, FM  # Return EIS and FM arrays

def res(x, y):
    """Calculates the resolution based on input parameters.
    
    Parameters:
    x : float
        The x-coordinate for resolution calculation.
    y : float
        The y-coordinate for resolution calculation.
    
    Returns:
    float
        Calculated resolution based on input parameters.
    """
    pixel = float(pix.get())  # Get pixel size from input
    lamb = float(wav.get())  # Get wavelength from input
    numa = float(NA.get())  # Get numerical aperture from input
    pit = float(pitch.get())  # Get pitch from input
    M = float(Mag.get())  # Get magnification from input
    return 2 * pixel * y / x * M + M * lamb * 10**(-3) * y * 10**(3) * np.tan(np.arcsin(numa)) / (pit * 10**(3) * numa)

def FOV(x, y):
    """Calculates the field of view based on input parameters.
    
    Parameters:
    x : float
        The x-coordinate for FOV calculation.
    y : float
        The y-coordinate for FOV calculation.
    
    Returns:
    float
        Calculated field of view based on input parameters.
    """
    pixel = float(pix.get())  # Get pixel size from input
    lamb = float(wav.get())  # Get wavelength from input
    numa = float(NA.get())  # Get numerical aperture from input
    pit = float(pitch.get())  # Get pitch from input
    M = float(Mag.get())  # Get magnification from input
    return pit * y / x * M  # Return field of view

def DOF(x, y):
    """Calculates the depth of field based on input parameters.
    
    Parameters:
    x : float
        The x-coordinate for DOF calculation.
    y : float
        The y-coordinate for DOF calculation.
    
    Returns:
    float
        Calculated depth of field based on input parameters.
    """
    pixel = float(pix.get())  # Get pixel size from input
    lamb = float(wav.get())  # Get wavelength from input
    numa = float(NA.get())  # Get numerical aperture from input
    pit = float(pitch.get())  # Get pitch from input
    M = float(Mag.get())  # Get magnification from input
    return M**2 * pixel**2 * y * 10**(3) * np.tan(np.arcsin(numa)) / (pit * 10**(3) * numa * x / y) + M**2 * lamb * 10**(-3) * (2 * y * 10**(3) * np.tan(np.arcsin(numa)) / (pit * 10**(3) * numa))**(2)

def plot():
    """Plots the graphs for Lateral resolution, FOV, and DOF based on user inputs.
    
    This function generates three subplots on a single canvas and updates the displayed figure.
    """
    refresh(canvas1, fig1)  # Refresh the canvas

    comap = Combo.get()  # Get colormap selection from the combobox
    
    nmax = 100  # Set the maximum number of points to plot
    
    wa = float(fmo.get())  # Get focal magnification value
    wb = float(fml.get())  # Get MLA focal length value
    xlimit = float(xaxis.get())  # Get x-axis upper limit value
    mlimit = float(maxis.get())  # Get x-axis lower limit value
    colm = int(col.get())  # Get colormap divisions value

    x = np.linspace(mlimit, xlimit, nmax)  # Create an array of x values
    y = np.linspace(mlimit, xlimit, nmax)  # Create an array of y values
    
    X, Y = np.meshgrid(x, y)  # Create a meshgrid for 2D plotting
    Z1 = res(X, Y)  # Calculate resolution values
    Z2 = FOV(X, Y)  # Calculate FOV values
    Z3 = DOF(X, Y)  # Calculate DOF values

    EIS, FM = EISnum(x, y, nmax)  # Calculate EIS and FM arrays
    
    # Adding the first subplot for Lateral resolution
    plt = fig1.add_subplot(131)
    limits = [mlimit, xlimit, mlimit, xlimit]  # Set axis limits
    plt.axis(limits)  # Apply limits to the axis
    plt.set_ylabel("MO focal length (mm)")  # Set y-axis label
    plt.set_xlabel("Lens array focal length (mm)")  # Set x-axis label
    plt.set_title("Lateral resolution (" + chr(956) + "m)")  # Set title
    plt.set_aspect('equal', 'box')  # Maintain aspect ratio
    
    # Create contour plot for resolution
    cont1 = plt.contourf(X, Y, Z1, colm, cmap=comap)
    fig1.colorbar(cont1, shrink=0.70)  # Add colorbar for the contour plot

    re = res(wb, wa)  # Calculate resolution for specific points
    re = "{:.2f}".format(re)  # Format the resolution value
    if i.get() == 1:  # Check if the first checkbox is selected
        plt.plot([mlimit, xlimit], [wa, wa], color='white', linewidth=1, linestyle='dotted')  # Add a horizontal line
        plt.text(wb, wa, str(re), color='white', fontsize=9)  # Label the line with the resolution value
        
    if f.get() == 1:  # Check if the second checkbox is selected
        plt.plot([wb, wb], [mlimit, xlimit], color='white', linewidth=1, linestyle='dotted')  # Add a vertical line

    for x in range(int(min(EIS) - 1), int(max(EIS))):  # Iterate through EIS values
        plt.plot([xlimit, mlimit], [FM[x - int(min(EIS))], FM[x - int(min(EIS))]], color='white', linewidth=1, linestyle='solid')  # Plot solid lines
        plt.text(round((xlimit - mlimit) * 2 / 3 + mlimit), FM[x - int(min(EIS))] + 0.6 / max(EIS), 'N=' + str(int(EIS[x - int(min(EIS))])), color='white')  # Label the lines with EIS values

    clist = cont1.collections[:]  # Get all contour collections for future use
    
    # Adding the second subplot for FOV
    plt2 = fig1.add_subplot(132)
    plt2.axis(limits)  # Set axis limits
    plt2.set_ylabel("MO focal length (mm)")  # Set y-axis label
    plt2.set_xlabel("Lens array focal length (mm)")  # Set x-axis label
    plt2.set_title("FOV (mm)")  # Set title
    plt2.set_aspect('equal', 'box')  # Maintain aspect ratio
    
    # Create contour plot for FOV
    cont2 = plt2.contourf(X, Y, Z2, colm, cmap=comap)
    fig1.colorbar(cont2, shrink=0.70)  # Add colorbar for the contour plot

    field = FOV(wb, wa)  # Calculate FOV for specific points
    field = "{:.2f}".format(field)  # Format the FOV value
    
    if i.get() == 1:  # Check if the first checkbox is selected
        plt2.plot([mlimit, xlimit], [wa, wa], color='white', linewidth=1, linestyle='dotted')  # Add a horizontal line
        plt2.text(wb, wa, str(field), color='white', fontsize=9)  # Label the line with the FOV value
        
    if f.get() == 1:  # Check if the second checkbox is selected
        plt2.plot([wb, wb], [mlimit, xlimit], color='white', linewidth=1, linestyle='dotted')  # Add a vertical line

    # Adding the third subplot for DOF
    plt3 = fig1.add_subplot(133)
    plt3.axis(limits)  # Set axis limits
    plt3.set_ylabel("MO focal length (mm)")  # Set y-axis label
    plt3.set_xlabel("Lens array focal length (mm)")  # Set x-axis label
    plt3.set_title("DOF (" + chr(956) + "m)")  # Set title
    plt3.set_aspect('equal', 'box')  # Maintain aspect ratio
    
    # Create contour plot for DOF
    cont3 = plt3.contourf(X, Y, Z3, colm, cmap=comap)
    fig1.colorbar(cont3, shrink=0.70)  # Add colorbar for the contour plot

    do = DOF(wb, wa)  # Calculate DOF for specific points
    do = "{:.0f}".format(do)  # Format the DOF value
    
    if i.get() == 1:  # Check if the first checkbox is selected
        plt3.plot([mlimit, xlimit], [wa, wa], color='white', linewidth=1, linestyle='dotted')  # Add a horizontal line
        plt3.text(wb, wa, str(do), color='white', fontsize=9)  # Label the line with the DOF value
        
    if f.get() == 1:  # Check if the second checkbox is selected
        plt3.plot([wb, wb], [mlimit, xlimit], color='white', linewidth=1, linestyle='dotted')  # Add a vertical line
        
    canvas1.draw()  # Redraw the canvas to update the plots

def show_values1():
    """Handles input from the first focal length input box.
    
    This function retrieves the value from the first focal length input and processes it if needed.
    """
    wa = float(fmo.get()) / 100  # Get the first focal length value and convert to mm

def show_values2():
    """Handles input from the second focal length input box.
    
    This function retrieves the value from the second focal length input and processes it if needed.
    """
    wb = float(fml.get()) / 100  # Get the second focal length value and convert to mm

def New_page():
    """Creates a new popup window for focal length calculations.
    
    This function sets up a new Tkinter window to collect MO parameters and calculate focal lengths.
    """
    popup = tkinter.Tk()  # Create a new Tkinter window
    popup.title("Focal length calculator")  # Set window title
    popup.geometry("200x150+700+70")  # Set window size and position
    Param = tkinter.Label(popup, text='MO parameters:', font=('Helvetica', 8, 'bold'))  # Create a label for MO parameters
    Param.place(x=50, y=20)  # Position the label

    varm = tkinter.DoubleVar(value=20)  # Create a DoubleVar for magnification with a default value
    et_m = tkinter.Label(popup, text="Magnification:")  # Create a label for magnification input
    et_m.place(x=5, y=50, width=150)  # Position the label
    m = tkinter.Spinbox(popup, from_=1, to=100, increment=1, textvariable=varm)  # Create a Spinbox for magnification input
    m.place(x=125, y=50, width=40)  # Position the Spinbox

    vlist = ["Olympus", "Nikon", "Zeiss", "Leica"]  # List of MO companies
    Param = tkinter.Label(popup, text='MO company:')  # Create a label for MO company selection
    Param.place(x=20, y=80)  # Position the label
    Combo = ttk.Combobox(popup, values=vlist, width=8)  # Create a combobox for company selection
    Combo.set("Olympus")  # Set default value
    Combo.place(x=105, y=80)  # Position the combobox

    def focal():
        """Calculates focal length based on the selected MO company and magnification.
        
        This function retrieves the selected company and magnification, calculates the focal length,
        and displays it in the popup window.
        """
        brand = Combo.get()  # Get the selected brand from the combobox

        # Set focal length based on selected brand
        if brand == "Nikon":
            f = 200
        if brand == "Olympus":
            f = 180
        if brand == "Zeiss":
            f = 165
        if brand == "Leica":
            f = 200

        fmo = f / float(m.get())  # Calculate focal length
        et_re = tkinter.Label(popup, text="fmo=" + str(fmo) + " mm")  # Create a label to display the result
        et_re.config(font=("Arial", 10, 'bold'))  # Configure font for the label
        et_re.config(fg="#0000FF")  # Set text color to blue
        et_re.config(bg="yellow")  # Set background color to yellow
        et_re.place(x=95, y=110, width=100)  # Position the label

    # Create a button to calculate focal length
    cal = tkinter.Button(popup, text='Calculate', command=focal, height=1, width=9)
    cal.place(x=10, y=110)  # Position the button
    
    popup.update()  # Update the popup window
    popup.deiconify()  # Show the popup window
    popup.mainloop()  # Start the main loop for the popup window
    
import tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk

def exitProgram():
    """
    Closes the Tkinter application by destroying the main window.
    """
    master.destroy()

# Create the main application window
master = tkinter.Tk()

# Set the title of the window
master.title('FLFM parameters')

# Set the size and position of the window
master.geometry("1500x500+0+0")

# Create a figure for plotting with specified size and resolution
fig1 = Figure(figsize=(12, 4), dpi=100)

# Define file types for a file dialog (currently unused)
types = [('all types', '.*'), ('text files', '.txt')]

# Create a canvas for the figure and place it in the Tkinter window
canvas1 = FigureCanvasTkAgg(fig1, master)
canvas1.get_tk_widget().pack(side=tkinter.RIGHT)

# Label for Fourier Light-Field Microscope parameters
Param = tkinter.Label(master, text='Fourier Light-Field microscope parameters:', font=('Helvetica', 8, 'bold'))
Param.place(x=50, y=20)

# Numerical Aperture (NA) input
var = tkinter.DoubleVar(value=0.2)
et_NA = tkinter.Label(text="MO numerical aperture:")
et_NA.place(x=50, y=50, width=150)
NA = tkinter.Spinbox(from_=0.05, to=0.95, increment=0.05, textvariable=var)
NA.place(x=205, y=50, width=50)

# MLA pitch input
var = tkinter.DoubleVar(value=1)
et_pitch = tkinter.Label(text="MLA pitch (mm):")
et_pitch.place(x=50, y=80, width=150)
pitch = tkinter.Spinbox(from_=0.1, to=3, increment=0.1, textvariable=var)
pitch.place(x=205, y=80, width=50)

# Pixel size input
var = tkinter.DoubleVar(value=1.5)
et_pix = tkinter.Label(text="Pixel size (" + chr(956) + "m):")
et_pix.place(x=50, y=110, width=150)
pix = tkinter.Spinbox(from_=0.9, to=7, increment=0.1, textvariable=var)
pix.place(x=205, y=110, width=50)

# Wavelength input
var = tkinter.DoubleVar(value=550)
et_wav = tkinter.Label(text="Wavelength (nm):")
et_wav.place(x=50, y=140, width=150)
wav = tkinter.Spinbox(from_=400, to=700, increment=1, textvariable=var)
wav.place(x=205, y=140, width=50)

# Relay magnification input
var = tkinter.DoubleVar(value=1)
et_mag = tkinter.Label(text="Relay magnification:")
et_mag.place(x=50, y=170, width=150)
Mag = tkinter.Spinbox(from_=0.1, to=10, increment=0.1, textvariable=var)
Mag.place(x=205, y=170, width=50)

# Button to plot the data
plot_button = tkinter.Button(master, 
                             command=plot,  # Function to be called when button is clicked
                             height=1, 
                             width=12,
                             text="Plot")
plot_button.place(x=500, y=10)

# Button to save plots
Sav = tkinter.Button(master, text='Save plots', command=SaveIm, height=1, width=12)
Sav.place(x=600, y=10)

# Button to open Fmo calculator
FMO = tkinter.Button(master, text='Fmo calculator', command=New_page, height=1, width=12)
FMO.place(x=700, y=10)

# Button to exit the application
Exit = tkinter.Button(master, text='Exit', command=exitProgram, height=1, width=12)
Exit.place(x=800, y=10)

# Label for Plot parameters
Param = tkinter.Label(master, text='Plot parameters:', font=('Helvetica', 8, 'bold'))
Param.place(x=50, y=220)

# Axes upper limit input
var = tkinter.DoubleVar(value=12)
et_xaxis = tkinter.Label(text="Axes upper limit (mm):")
et_xaxis.place(x=50, y=250, width=150)
xaxis = tkinter.Spinbox(from_=0.1, to=20, increment=0.1, textvariable=var)
xaxis.place(x=205, y=250, width=50)

# Axes lower limit input
var = tkinter.DoubleVar(value=2)
et_maxis = tkinter.Label(text="Axes lower limit (mm):")
et_maxis.place(x=50, y=280, width=150)
maxis = tkinter.Spinbox(from_=0.1, to=20, increment=0.1, textvariable=var)
maxis.place(x=205, y=280, width=50)

# List of colormaps for plotting
vlist = ["jet", "hot", "plasma", "hsv", "gray", "inferno", "tab20c"]
Param = tkinter.Label(master, text='Choose colormap:')
Param.place(x=62, y=310)

# Dropdown for selecting colormap
Combo = ttk.Combobox(master, values=vlist, width=6)
Combo.set("jet")  # Default value
Combo.place(x=205, y=310)

# Colormap divisions input
var = tkinter.DoubleVar(value=30)
et_col = tkinter.Label(text="Colormap divisions:")
et_col.place(x=45, y=340, width=150)
col = tkinter.Spinbox(from_=1, to=200, increment=1, textvariable=var)
col.place(x=205, y=340, width=50)

# MO focal length input
var = tkinter.DoubleVar(value=3.2)
et_fmo = tkinter.Label(text="MO focal length (mm):")
et_fmo.place(x=50, y=370, width=150)
fmo = tkinter.Spinbox(from_=1, to=30, increment=0.5, textvariable=var)
fmo.place(x=205, y=370, width=50)

# Checkbox for an option related to MO focal length
i = tkinter.IntVar()
c = tkinter.Checkbutton(master, variable=i)
c.place(x=32, y=368)

# MLA focal length input
var = tkinter.DoubleVar(value=6)
et_fml = tkinter.Label(text="MLA focal length (mm):")
et_fml.place(x=50, y=400, width=150)
fml = tkinter.Spinbox(from_=1, to=30, increment=0.5, textvariable=var)
fml.place(x=205, y=400, width=50)

# Checkbox for an option related to MLA focal length
f = tkinter.IntVar()
l = tkinter.Checkbutton(master, variable=f)
l.place(x=32, y=398)

def About():
    """
    Opens a new window displaying information from 'about.txt'.
    """
    f = open("about.txt", "r")
    filewin = tkinter.Toplevel(master)
    button = tkinter.Label(filewin, text=f.read(), justify='left')
    button.pack()

# Create a menu bar with a Help menu
menubar = tkinter.Menu(master)
helpmenu = tkinter.Menu(menubar, tearoff=0)
menubar.add_cascade(label="Help", menu=helpmenu)
helpmenu.add_command(label="About...", command=About)

# Configure the main window to use the created menu bar
master.config(menu=menubar)

# Update and show the main window
master.update()
master.deiconify()

# Start the Tkinter main event loop
master.mainloop()
