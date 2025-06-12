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

from abc import ABCMeta, abstractmethod
import time

import numpy as np
import scipy as sp
import scipy.optimize as optimize
import tensorflow as tf
import sys
import tkinter as tk
from tkinter import messagebox as mbox

class OptimizerABC:
    """
    Abstract base class for optimizers. It ensures consistency and compatibility between Fitters and Optimizers.
    Core function is 'minimize' which is called by the fitter. The rest is handled by the optimizer.
    Allows to use different TensorFlow optimizers and also the L-BFGS-B optimizer from scipy.
    Is basically a wrapper around those optimizers to call them similarly in the Fitter.
    Defines an interface for other optimizers (basically the minimize function) that self-made optimizers must fulfill.
    """

    __metaclass__ = ABCMeta  # Set the class as an abstract class

    def __init__(self, maxiter, options, kwargs) -> None:
        """
        Initializes the optimizer with given parameters.

        Parameters:
        maxiter (int): The maximum number of iterations for the optimization process.
        options (dict): Configuration options to be used by the optimizer.
        kwargs (dict): Additional parameters that can be passed to the optimizer.
        """
        self.maxiter = maxiter  # Set maximum iterations

        # Print step size for output logs - calculates number of steps to print
        self.print_step_size = np.max((np.round(self.maxiter / 10).astype(np.int32),20))
        self.print_width = len(str(self.print_step_size))  # Width for formatting output

        self.history = [['step', 'time', 'loss']]  # Initialize history of optimization

        self.objective = None  # Placeholder for objective function
        self.variables = None  # Placeholder for variables to optimize
        self.weight = None  # Placeholder for weights (if applicable)
        self.rate = 1.1  # Rate for adaptive learning (specific to certain optimizers)
        self.mu = 1  # Parameter for some optimization methods (e.g., L-BFGS)
        self.opt = self.create_actual_optimizer(options, kwargs)  # Instantiate the actual optimizer

    @abstractmethod
    def create_actual_optimizer(self, options):
        """
        Here the actual underlying optimizer should be created.
        
        Parameters:
        options (dict): Configuration options for creating the optimizer.

        Raises:
        NotImplementedError: If the method is not implemented in the derived class.
        """
        raise NotImplementedError("You need to implement a 'create_actual_optimizer' method in your optimizer class.")

    def minimize(self, objective, variables, pbar):
        """
        Adapts the given variables in a way that minimizes the given objective.
        Returns the new state of the variables after the optimization.

        Parameters:
        objective (callable): The objective function to minimize.
        variables (list): The list of variables to optimize.
        pbar (object): Progress bar object for tracking optimization progress.

        Returns:
        list: Updated variables after optimization.
        """
        variables = [tf.Variable(variable) for variable in variables]  # Convert variables to TensorFlow Variables

        for step in range(self.maxiter):  # Loop over the number of iterations
            start = time.time()  # Record start time for iteration
            with tf.GradientTape() as tape:  # Track gradients
                tape.watch(variables)  # Watch the variables for gradient computation
                loss = objective(variables)  # Compute the loss using the objective function
            pbar.update(1)  # Update progress bar
            #self.write_output(step, loss)  # Output loss to console (commented out)
            pbar.set_description("current loss %f" % loss)  # Update description of progress bar
            gradients = tape.gradient(loss, variables)  # Compute gradients of loss
            self.opt.apply_gradients(zip(gradients, variables))  # Apply gradients to variables

            self.update_history(step+1, time.time()-start, loss.numpy())  # Update history with step data

        #self.write_output(self.maxiter, objective(variables), True)  # Final output (commented out)

        result_variables = [variable.numpy() for variable in variables]  # Convert variables back to numpy arrays

        return result_variables  # Return updated variables

    def objective_wrapper_for_optimizer(self):
        """
        Wrapper around the actual objective. Needed since TensorFlow optimizer
        can only optimize a function that takes no arguments and returns a loss.
        """
        return self.objective(self.variables)  # Call the objective function with the current variables

    def write_output(self, step, loss, do_anyway=False):
        """
        Writes output to the console in a nicely formatted way.
        Used to show user the progress of the optimization.

        Parameters:
        step (int): The current step of the optimization.
        loss (float): The current loss value.
        do_anyway (bool): Flag to force output regardless of step count (default is False).
        """
        # TODO: one could calculate an estimate how long the optimization still takes
        self.print_step_size =  np.max((np.round(self.maxiter / 10).astype(np.int32),20))  # Update print step size
        if (step % self.print_step_size == 0) or do_anyway:  # Check if it's time to print output
            #tf.print(f"[{step:5}/{self.maxiter}]  loss={loss:>8.2f} ")
            tf.print("step:", step, "loss:", loss)  # Print current step and loss
        return

    def update_history(self, step, time, loss):
        """
        Save information of each iteration to the history.
        This history can be later used to analyze the efficiency of the optimization.

        Parameters:
        step (int): Current optimization step.
        time (float): Time taken for the current optimization step.
        loss (float): Loss value at the current step.
        """
        self.history.append([step, time, loss])  # Append current step data to history
        return

class Adadelta(OptimizerABC):
    """
    Wrapper around TensorFlow's Adadelta optimizer.
    """
    def __init__(self, maxiter, learning_rate=0.001, rho=0.95, epsilon=1e-07,
                name='Adadelta', **kwargs) -> None:
        """
        Initializes the Adadelta optimizer with the specified parameters.

        Parameters:
        maxiter (int): Maximum number of iterations for optimization.
        learning_rate (float): Learning rate for the optimizer.
        rho (float): Decay rate for moving average of squared gradients.
        epsilon (float): Small constant for numerical stability.
        name (str): Name of the optimizer instance.
        kwargs (dict): Additional parameters for optimizer configuration.
        """
        options = locals().copy()  # Copy local variable options
        del options['self']  # Remove self reference
        del options['maxiter']  # Remove maxiter reference
        del options['kwargs']  # Remove kwargs reference
        del options['__class__']  # Remove class reference
        super().__init__(maxiter, options, kwargs)  # Call parent constructor

    def create_actual_optimizer(self, options, kwargs):
        """
        Create the actual Adadelta optimizer with given options.

        Parameters:
        options (dict): Configuration options for the optimizer.
        kwargs (dict): Additional parameters for the optimizer.

        Returns:
        object: Instance of the Adadelta optimizer.
        """
        return tf.optimizers.Adadelta(**options, **kwargs)  # Instantiate and return the Adadelta optimizer

class Adagrad(OptimizerABC):
    """
    Wrapper around TensorFlow's Adagrad optimizer.
    """
    def __init__(self, maxiter, learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07,
                name='Adagrad', **kwargs) -> None:
        """
        Initializes the Adagrad optimizer with the specified parameters.

        Parameters:
        maxiter (int): Maximum number of iterations for optimization.
        learning_rate (float): Learning rate for the optimizer.
        initial_accumulator_value (float): Initial value for accumulators.
        epsilon (float): Small constant for numerical stability.
        name (str): Name of the optimizer instance.
        kwargs (dict): Additional parameters for optimizer configuration.
        """
        options = locals().copy()  # Copy local variable options
        del options['self']  # Remove self reference
        del options['maxiter']  # Remove maxiter reference
        del options['kwargs']  # Remove kwargs reference
        del options['__class__']  # Remove class reference
        super().__init__(maxiter, options, kwargs)  # Call parent constructor

    def create_actual_optimizer(self, options, kwargs):
        """
        Create the actual Adagrad optimizer with given options.

        Parameters:
        options (dict): Configuration options for the optimizer.
        kwargs (dict): Additional parameters for the optimizer.

        Returns:
        object: Instance of the Adagrad optimizer.
        """
        return tf.optimizers.Adagrad(**options, **kwargs)  # Instantiate and return the Adagrad optimizer

class Adam(OptimizerABC):
    """
    Wrapper around TensorFlow's Adam optimizer.
    """
    def __init__(self, maxiter, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
                name='Adam', **kwargs) -> None:
        """
        Initializes the Adam optimizer with the specified parameters.

        Parameters:
        maxiter (int): Maximum number of iterations for optimization.
        learning_rate (float): Learning rate for the optimizer.
        beta_1 (float): Exponential decay rate for the first moment estimates.
        beta_2 (float): Exponential decay rate for the second moment estimates.
        epsilon (float): Small constant for numerical stability.
        amsgrad (bool): Whether to use the AMSGrad variant of Adam.
        name (str): Name of the optimizer instance.
        kwargs (dict): Additional parameters for optimizer configuration.
        """
        options = locals().copy()  # Copy local variable options
        del options['self']  # Remove self reference
        del options['maxiter']  # Remove maxiter reference
        del options['kwargs']  # Remove kwargs reference
        del options['__class__']  # Remove class reference
        super().__init__(maxiter, options, kwargs)  # Call parent constructor

    def create_actual_optimizer(self, options, kwargs):
        """
        Create the actual Adam optimizer with given options.

        Parameters:
        options (dict): Configuration options for the optimizer.
        kwargs (dict): Additional parameters for the optimizer.

        Returns:
        object: Instance of the Adam optimizer.
        """
        return tf.optimizers.Adam(**options, **kwargs)  # Instantiate and return the Adam optimizer

class Adamax(OptimizerABC):
    """
    Wrapper around TensorFlow's Adamax optimizer.
    """
    def __init__(self, maxiter, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, 
                name='Adamax', **kwargs) -> None:
        """
        Initializes the Adamax optimizer with the specified parameters.

        Parameters:
        maxiter (int): Maximum number of iterations for optimization.
        learning_rate (float): Learning rate for the optimizer.
        beta_1 (float): Exponential decay rate for the first moment estimates.
        beta_2 (float): Exponential decay rate for the second moment estimates.
        epsilon (float): Small constant for numerical stability.
        name (str): Name of the optimizer instance.
        kwargs (dict): Additional parameters for optimizer configuration.
        """
        options = locals().copy()  # Copy local variable options
        del options['self']  # Remove self reference
        del options['maxiter']  # Remove maxiter reference
        del options['kwargs']  # Remove kwargs reference
        del options['__class__']  # Remove class reference
        super().__init__(maxiter, options, kwargs)  # Call parent constructor

    def create_actual_optimizer(self, options, kwargs):
        """
        Create the actual Adamax optimizer with given options.

        Parameters:
        options (dict): Configuration options for the optimizer.
        kwargs (dict): Additional parameters for the optimizer.

        Returns:
        object: Instance of the Adamax optimizer.
        """
        return tf.optimizers.Adamax(**options, **kwargs)  # Instantiate and return the Adamax optimizer

class Ftrl(OptimizerABC):
    """
    Wrapper around TensorFlow's Ftrl optimizer.
    """
    def __init__(self, maxiter, learning_rate=0.001, learning_rate_power=-0.5, initial_accumulator_value=0.1,
                l1_regularization_strength=0.0, l2_regularization_strength=0.0,
                name='Ftrl', l2_shrinkage_regularization_strength=0.0, beta=0.0, **kwargs) -> None:
        """
        Initializes the Ftrl optimizer with the specified parameters.

        Parameters:
        maxiter (int): Maximum number of iterations for optimization.
        learning_rate (float): Learning rate for the optimizer.
        learning_rate_power (float): Power for decay of the learning rate.
        initial_accumulator_value (float): Initial value for accumulators.
        l1_regularization_strength (float): Strength of L1 regularization.
        l2_regularization_strength (float): Strength of L2 regularization.
        name (str): Name of the optimizer instance.
        l2_shrinkage_regularization_strength (float): Strength of L2 shrinkage regularization.
        beta (float): Regularization parameter for FTRL.
        kwargs (dict): Additional parameters for optimizer configuration.
        """
        options = locals().copy()  # Copy local variable options
        del options['self']  # Remove self reference
        del options['maxiter']  # Remove maxiter reference
        del options['kwargs']  # Remove kwargs reference
        del options['__class__']  # Remove class reference
        super().__init__(maxiter, options, kwargs)  # Call parent constructor

    def create_actual_optimizer(self, options, kwargs):
        """
        Create the actual Ftrl optimizer with given options.

        Parameters:
        options (dict): Configuration options for the optimizer.
        kwargs (dict): Additional parameters for the optimizer.

        Returns:
        object: Instance of the Ftrl optimizer.
        """
        return tf.optimizers.Ftrl(**options, **kwargs)  # Instantiate and return the Ftrl optimizer

class Nadam(OptimizerABC):
    """
    Wrapper around TensorFlow's Nadam optimizer.
    """
    def __init__(self, maxiter, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                name='Nadam', **kwargs) -> None:
        """
        Initializes the Nadam optimizer with the specified parameters.

        Parameters:
        maxiter (int): Maximum number of iterations for optimization.
        learning_rate (float): Learning rate for the optimizer.
        beta_1 (float): Exponential decay rate for the first moment estimates.
        beta_2 (float): Exponential decay rate for the second moment estimates.
        epsilon (float): Small constant for numerical stability.
        name (str): Name of the optimizer instance.
        kwargs (dict): Additional parameters for optimizer configuration.
        """
        options = locals().copy()  # Copy local variable options
        del options['self']  # Remove self reference
        del options['maxiter']  # Remove maxiter reference
        del options['kwargs']  # Remove kwargs reference
        del options['__class__']  # Remove class reference
        super().__init__(maxiter, options, kwargs)  # Call parent constructor

    def create_actual_optimizer(self, options, kwargs):
        """
        Create the actual Nadam optimizer with given options.

        Parameters:
        options (dict): Configuration options for the optimizer.
        kwargs (dict): Additional parameters for the optimizer.

        Returns:
        object: Instance of the Nadam optimizer.
        """
        return tf.optimizers.Nadam(**options, **kwargs)  # Instantiate and return the Nadam optimizer

class RMSprop(OptimizerABC):
    """
    Wrapper around TensorFlow's RMSprop optimizer.
    """
    def __init__(self, maxiter, learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,
                name='RMSprop', **kwargs) -> None:
        """
        Initializes the RMSprop optimizer with the specified parameters.

        Parameters:
        maxiter (int): Maximum number of iterations for optimization.
        learning_rate (float): Learning rate for the optimizer.
        rho (float): Decay rate for moving average of squared gradients.
        momentum (float): Momentum factor.
        epsilon (float): Small constant for numerical stability.
        centered (bool): If True, compute centered RMSProp, otherwise compute standard RMSProp.
        name (str): Name of the optimizer instance.
        kwargs (dict): Additional parameters for optimizer configuration.
        """
        options = locals().copy()  # Copy local variable options
        del options['self']  # Remove self reference
        del options['maxiter']  # Remove maxiter reference
        del options['kwargs']  # Remove kwargs reference
        del options['__class__']  # Remove class reference
        super().__init__(maxiter, options, kwargs)  # Call parent constructor

    def create_actual_optimizer(self, options, kwargs):
        """
        Create the actual RMSprop optimizer with given options.

        Parameters:
        options (dict): Configuration options for the optimizer.
        kwargs (dict): Additional parameters for the optimizer.

        Returns:
        object: Instance of the RMSprop optimizer.
        """
        return tf.optimizers.RMSprop(**options, **kwargs)  # Instantiate and return the RMSprop optimizer

class SGD(OptimizerABC):
    """
    Wrapper around TensorFlow's Stochastic Gradient Descent (SGD) optimizer.
    This class allows for setting various parameters for the SGD optimizer.
    """
    
    def __init__(self, maxiter, learning_rate=0.01, momentum=0.0, nesterov=False,
                 name='SGD', **kwargs) -> None:
        """
        Initialize the SGD optimizer.
        
        Parameters:
        - maxiter: int
            The maximum number of iterations for the optimization process.
        - learning_rate: float, default=0.01
            The step size at each iteration while moving toward a minimum.
        - momentum: float, default=0.0
            The momentum factor, which helps to accelerate SGD in the relevant direction.
        - nesterov: bool, default=False
            Whether to enable Nesterov momentum.
        - name: str, default='SGD'
            Name of the optimizer.
        - **kwargs: additional keyword arguments
            Additional options to be passed to the optimizer.
        """
        options = locals().copy()  # Copying local variables to options
        del options['self']  # Removing 'self' from options
        del options['maxiter']  # Removing 'maxiter' as it's handled by the parent class
        del options['kwargs']  # Removing 'kwargs' for clean inheritance
        del options['__class__']  # Removing class reference
        super().__init__(maxiter, options, kwargs)  # Initializing parent class

    def create_actual_optimizer(self, options, kwargs):
        """
        Create the actual TensorFlow optimizer using the provided options and kwargs.

        Parameters:
        - options: dict
            Dictionary of options for the optimizer.
        - kwargs: dict
            Additional keyword arguments for the optimizer.

        Returns:
        - tf.optimizers.SGD
            The created SGD optimizer instance.
        """
        return tf.optimizers.SGD(**options, **kwargs)  # Create and return SGD optimizer


class L_BFGS_B(OptimizerABC):
    """
    Wrapper around SciPy's L-BFGS-B optimizer.
    Note: There isn't a built-in L-BFGS-B optimizer available in TensorFlow.
    """

    def __init__(self, maxiter, gtol=1e-10, **kwargs) -> None:
        """
        Initialize the L-BFGS-B optimizer.

        Parameters:
        - maxiter: int
            The maximum number of iterations for the optimization process.
        - gtol: float, default=1e-10
            The gradient norm must be less than gtol before the optimizer exits.
        - **kwargs: additional keyword arguments
            Additional options to be passed to the optimizer.
        """
        options = locals().copy()  # Copying local variables to options
        del options['self']  # Removing 'self' from options
        del options['kwargs']  # Removing 'kwargs' for clean inheritance
        del options['__class__']  # Removing class reference
        super().__init__(maxiter, options, kwargs)  # Initializing parent class

        self.step = 0  # Counter for optimization steps
        self.status = None  # To store final output from the optimization

        self.shapes = []  # To store shapes of input variables
        self.lengths = []  # To store lengths of input variables
        self.dtypes = []  # To store data types of input variables

    def create_actual_optimizer(self, options, kwargs):
        """
        Placeholder method to comply with the abstract base class.
        This method adapts the options to fit the SciPy API, as it only provides a function.

        Parameters:
        - options: dict
            Dictionary of options for the optimizer.
        - kwargs: dict
            Additional keyword arguments for the optimizer.

        Returns:
        - None
        """
        self.options = {**options, **kwargs}  # Merge options and kwargs
        return None  # Since SciPy does not provide an optimizer object

    def minimize(self, objective, variables, varinfo, pbar):
        """
        Perform minimization of the given objective function.

        Parameters:
        - objective: callable
            The objective function to minimize.
        - variables: list
            The list of variables to optimize.
        - varinfo: list
            Information about the variables being optimized.
        - pbar: object
            Progress bar instance for tracking progress.

        Returns:
        - list
            The updated state of the variables after optimization.
        """
        self.objective = objective  # Store the objective function
        self.step = 0  # Reset step count
        init_var = self.flatten_variables(variables)  # Flatten variables for optimization
        self.options['maxiter'] = self.maxiter  # Set max iterations in options
        start_time = pbar.postfix[-1]['time']  # Get the current time from progress bar
        self.mu = 1  # Initialize mu for the optimization process
        
        # Execute the optimization using SciPy's minimize function
        result = optimize.minimize(fun=self.objective_wrapper_for_optimizer, 
                                    x0=init_var, 
                                    args=(varinfo, pbar, start_time), 
                                    jac=True, 
                                    method='L-BFGS-B', 
                                    options=self.options)
        
        self.status = result  # Store the result of optimization

        result_var = result.x  # Extract optimized variable values
        variables = self.reshape_variables_np(result_var)  # Reshape optimized variables
        return variables  # Return the reshaped variables

    def flatten_variables(self, variables):
        """
        Flatten and concatenate a list of variables into a single vector.
        This is necessary because the L-BFGS-B optimizer can only handle vectors.
        The shapes and lengths of variables are saved for later reconstruction.

        Parameters:
        - variables: list
            The list of variables to flatten.

        Returns:
        - np.ndarray
            A single concatenated vector of all variables.
        """
        flat_variables = []  # List to hold flattened variables
        self.shapes = []  # Reset shapes list
        self.lengths = []  # Reset lengths list
        self.dtypes = []  # Reset data types list

        for variable in variables:  # Iterate through each variable
            shape = variable.shape  # Get the shape of the variable
            self.shapes.append(shape)  # Save the shape
            self.lengths.append(np.product(shape))  # Save the total length
            self.dtypes.append(variable.dtype)  # Save the data type
            flat_variables.append(variable.flatten())  # Flatten the variable

        return np.concatenate(flat_variables)  # Concatenate and return flattened variables

    def reshape_variables_np(self, var):
        """
        Reshape the current state of the variables from the optimizer's current guess.

        Parameters:
        - var: np.ndarray
            The current guess of the variables in vector form.

        Returns:
        - list
            The list of reshaped variables.
        """
        variables = []  # List to hold reshaped variables
        idx_count = 0  # Index to track position in flat variable array

        for i, (shape, length, dtype) in enumerate(zip(self.shapes, self.lengths, self.dtypes)):
            variable = var[idx_count: idx_count + length]  # Extract segment for current variable
            variables.append(np.reshape(variable, shape).astype(dtype))  # Reshape back to original shape
            idx_count += length  # Update index count

        return variables  # Return the reshaped variables

    def reshape_variables_tf(self, var):
        """
        Reshape the current state of the variables from the optimizer's current guess using TensorFlow.

        Parameters:
        - var: tf.Tensor
            The current guess of the variables in vector form.

        Returns:
        - list
            The list of reshaped TensorFlow variables.
        """
        variables = [None] * len(self.shapes)  # Preallocate list for variables
        idx_count = 0  # Index to track position in flat variable array

        for i, (shape, length, dtype) in enumerate(zip(self.shapes, self.lengths, self.dtypes)):
            variable = var[idx_count: idx_count + length]  # Extract segment for current variable
            variables[i] = tf.cast(tf.reshape(variable, shape), dtype)  # Reshape and cast to original dtype
            idx_count += length  # Update index count

        return variables  # Return the reshaped TensorFlow variables

    def objective_wrapper_for_optimizer(self, var, varinfo, pbar, start_time):
        """
        Wrapper around the actual objective for compatibility with SciPy's optimizer.
        Converts TensorFlow variables to NumPy with double precision (float64).

        Parameters:
        - var: np.ndarray
            The current guess of the variables in vector form.
        - varinfo: list
            Information about the variables being optimized.
        - pbar: object
            Progress bar instance for tracking progress.
        - start_time: float
            Start time for performance tracking.

        Returns:
        - list
            A list containing the loss value and gradients.
        """
        return [np.real(tensor.numpy()).astype(np.float64) for tensor in self.objective_wrapper_for_gradient(
            tf.cast(var, tf.float32), varinfo, pbar, start_time)]  # Run gradient calculation and convert to float64

    def objective_wrapper_for_gradient(self, var, varinfo, pbar, start_time=None):
        """
        Wrapper around the actual objective that calculates the gradient.

        Parameters:
        - var: tf.Tensor
            The current guess of the variables.
        - varinfo: list
            Information about the variables being optimized.
        - pbar: object
            Progress bar instance for tracking progress.
        - start_time: float, optional
            Start time for performance tracking.

        Returns:
        - tuple
            A tuple containing the loss and gradient vector.
        """
        loss = 0.0  # Initialize loss
        Np = len(self.shapes)  # Number of variables
        Nfit = self.shapes[0][0]  # Fitting dimension from the first variable shape
        start = time.time()  # Record the start time for this step
        grad = [None] * Np  # Initialize gradient list
        batchsize = self.batch_size  # Batch size for processing
        variables = self.reshape_variables_tf(var)  # Reshape variables for TensorFlow processing
        ind = list(np.int32(np.linspace(0, Nfit, Nfit // batchsize + 2)))  # Create index for batching
        var1 = [None] * Np  # Preallocate variable storage

        for i in range(len(ind) - 1):  # Loop through batches
            for k in range(Np):  # Loop through all variables
                if varinfo[k]['type'] == 'Nfit':
                    # Handle fitting type variables differently based on their ID
                    if varinfo[k]['id'] == 0:
                        var1[k] = variables[k][ind[i]:ind[i + 1]]  # Select appropriate slice
                    elif varinfo[k]['id'] == 1:
                        var1[k] = variables[k][:, ind[i]:ind[i + 1]]
                else:
                    var1[k] = variables[k]  # Directly assign non-fitting type variables
                    if i == 0:
                        grad[k] = 0.0  # Initialize gradient for non-fitting variables

            with tf.GradientTape() as tape:  # Start recording gradients
                tape.watch(var1)  # Watch the current variables
                loss1 = self.objective(var1, self.mu, ind[i:i + 2])  # Calculate loss for this batch
            w1 = var1[0].shape[0] / Nfit  # Weight for this batch
            loss += loss1 * w1  # Update total loss
            grad1 = tape.gradient(loss1, var1)  # Compute gradients for this batch

            for k in range(Np):  # Loop through all variables
                if grad1[k] is None:
                    grad1[k] = var1[k] * 0  # If gradient is None, set to zero

            for k in range(Np):  # Consolidate gradients
                if varinfo[k]['type'] == 'Nfit':
                    if grad[k] is None:
                        grad[k] = grad1[k]  # Initialize gradient if None
                    else:
                        grad[k] = tf.concat((grad[k], grad1[k]), axis=varinfo[k]['id'])  # Concatenate gradients
                else:
                    grad[k] += grad1[k] * w1  # Accumulate gradients for non-fitting variables

        # Update progress bar with current loss and time
        pbar.postfix[-1]['loss'] = loss
        pbar.postfix[-1]['time'] = start_time + pbar._time() - pbar.start_t
        pbar.update(1)  # Update the progress bar
        self.update_history(self.step + 1, time.time() - start, loss.numpy())  # Record history of optimization steps
        self.step += 1  # Increment step count

        # Combine gradients into a single vector
        gradvec = tf.reshape(grad[0], [-1])  # Flatten the first gradient
        for g in grad[1:]:  # Concatenate remaining gradients
            gradvec = tf.concat((gradvec, tf.reshape(g, [-1])), axis=0)

        self.mu *= self.rate  # Update mu based on the learning rate
        self.mu = np.min([1e7, self.mu])  # Cap mu to a maximum value
        return loss, gradvec  # Return loss and gradient vector

    def objective_wrapper_for_gradient_copy(self, var, pbar, start_time=None):
        """
        Another wrapper around the actual objective. Needed for gradient calculation
        via TensorFlow's GradientTape. This is an alternative approach to allow gradient computation.

        Parameters:
        - var: tf.Tensor
            The current guess of the variables.
        - pbar: object
            Progress bar instance for tracking progress.
        - start_time: float, optional
            Start time for performance tracking.

        Returns:
        - tuple
            A tuple containing the loss and gradient.
        """
        start = time.time()  # Start the timer
        with tf.GradientTape() as tape:  # Start recording gradients
            tape.watch(var)  # Watch the current variable
            variables = self.reshape_variables_tf(var)  # Reshape for TensorFlow use
            loss = self.objective(variables)  # Compute the loss

        # Update progress bar with current loss and time
        pbar.postfix[-1]['loss'] = loss
        pbar.postfix[-1]['time'] = start_time + pbar._time() - pbar.start_t
        pbar.update(1)  # Update the progress bar
        self.update_history(self.step + 1, time.time() - start, loss.numpy())  # Log history
        self.step += 1  # Increment step count
        
        grad = tape.gradient(loss, var)  # Compute gradient of loss with respect to the variables
        return loss, grad  # Return loss and gradient