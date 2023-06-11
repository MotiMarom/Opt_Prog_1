# Moti Marom
# ID 025372830

# test_unconstrained_min.py
import numpy as np
from unconstrained_min import unconstrained_minimization
from utils import plot_contour
from examples import example_func_quad_1
from examples import example_func_quad_2
from examples import example_func_quad_3
from examples import example_func_rosenbrock
from examples import example_func_linear
from examples import example_func_nonquad

# test params
max_iter = 1000
step_tol = 1e-8
obj_tol = 1e-12

# Get the requested function to analyze from user
print('Hello!')
print('Please pick a function for analysis from the following:')
print('1-circle, 2-ellipse, 3-shifted ellipse, 4-Rosenbrock, 5-linear, 6-nonquad')
function_index = input('type a single number between [1, 6]:')
function_index = int(function_index)

if function_index == 1:
    print('You chose 1: circle')
    func_name = 'circle'
    func2min = example_func_quad_1
elif function_index == 2:
    func_name = 'ellipse'
    print('You chose 2: ellipse')
    func2min = example_func_quad_2
elif function_index == 3:
    func_name = 'shifted ellipse'
    print('You chose 3: shifted ellipse')
    func2min = example_func_quad_3
elif function_index == 4:
    func_name = 'Rosenbrock'
    print('You chose 4: Rosenbrock')
    func2min = example_func_rosenbrock
elif function_index == 5:
    func_name = 'linear'
    print('You chose 5: linear')
    func2min = example_func_linear
elif function_index == 6:
    func_name = 'nonquad'
    print('You chose 6: nonquad')
    func2min = example_func_nonquad
else:
    print("You chose {} where it should be an integer number between 1-6. please rerun and try again."
          .format(function_index))

if function_index in range(1, 7):

    # Gradient Descent ('gd')
    method = 'gd'
    x0 = np.array([8, 6], dtype=np.float64)
    results_gd = \
        unconstrained_minimization(func2min, x0, max_iter, obj_tol, step_tol, method)

    # Newton Descent ('newton')
    method = 'newton'
    x0 = np.array([8, 6], dtype=np.float64)
    if function_index == 5:
        # Linear leads to singularity
        results_newton = []
    else:
        results_newton = \
            unconstrained_minimization(func2min, x0, max_iter, obj_tol, step_tol, method)

    # BFGS Descent ('bfgs')
    method = 'bfgs'
    x0 = np.array([8, 6], dtype=np.float64)
    if function_index == 5:
        # Linear leads to singularity
        results_bfgs = []
    else:
        results_bfgs = \
            unconstrained_minimization(func2min, x0, max_iter, obj_tol, step_tol, method)

    # SR1 Descent ('sr1')
    method = 'sr1'
    x0 = np.array([8, 6], dtype=np.float64)
    if function_index == 5:
        # Linear leads to singularity
        results_sr1 = []
    else:
        results_sr1 = \
            unconstrained_minimization(func2min, x0, max_iter, obj_tol, step_tol, method)

    # plot all methods tracks on top of the contour
    plot_contour(func2min, func_name, results_gd, results_newton, results_bfgs, results_sr1)

    print('End of {} analysis'.format(func_name))

print('End of running.')

