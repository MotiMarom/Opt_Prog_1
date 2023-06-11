# Moti Marom
# ID 025372830

# utils.py

import numpy as np
import matplotlib.pyplot as plt

def plot_contour(obj_func, func_name, results_gd, results_newton, results_bfgs, results_sr1):

    # prints
    # Meshgrid between -10 and 10
    x0 = np.linspace(-10., 10., 100)
    x1 = np.linspace(-10., 10., 100)
    mesh_x0, mesh_x1 = np.meshgrid(x0, x1)
    n_rows, n_cols = mesh_x0.shape
    func_mesh_x = np.array(np.zeros((n_rows, n_cols)), dtype=np.float64)

    for r in range(n_rows):
        for c in range(n_cols):
            x_in_to_obj_f = np.array([mesh_x0[r, c], mesh_x1[r, c]], dtype=np.float64)
            func_x, g_NA, h_NA = obj_func(x_in_to_obj_f, en_hessian=False)
            func_mesh_x[r, c] = func_x

    # Plot
    fig = plt.figure(figsize=(20, 20))
    ax = plt.axes(projection='3d')
    ax.contour3D(mesh_x0, mesh_x1, func_mesh_x, 60, cmap='viridis')

    # all methods
    if results_gd:
        ax.scatter(results_gd[0][0], results_gd[0][1], results_gd[1], marker='o', color='red', linewidth=10)
        ax.plot(results_gd[2][:, 0], results_gd[2][:, 1], results_gd[3], color='red', linewidth=3, label=results_gd[5])

    if results_newton:
        ax.scatter(results_newton[0][0], results_newton[0][1], results_newton[1], marker='o', color='green', linewidth=10)
        ax.plot(results_newton[2][:, 0], results_newton[2][:, 1], results_newton[3], color='green', linewidth=3, label=results_newton[5])

    if results_bfgs:
        ax.scatter(results_bfgs[0][0], results_bfgs[0][1], results_bfgs[1], marker='o', color='blue', linewidth=10)
        ax.plot(results_bfgs[2][:, 0], results_bfgs[2][:, 1], results_bfgs[3], color='blue', linewidth=3, label=results_bfgs[5])

    if results_sr1:
        ax.scatter(results_sr1[0][0], results_sr1[0][1], results_sr1[1], marker='o', color='black', linewidth=10)
        ax.plot(results_sr1[2][:, 0], results_sr1[2][:, 1], results_sr1[3], color='black', linewidth=3, label=results_sr1[5])

    ax.set_title('$f(x) = f$({}) track using all minimization method'.format(func_name))
    ax.set_xlabel('$x_{0}$')
    ax.set_ylabel('$x_{1}$')
    ax.set_zlabel('$f(x) = f$({})'.format(func_name))
    ax.view_init(40, 20)

    plt.legend()
    plt.show()

