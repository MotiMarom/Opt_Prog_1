# Moti Marom
# ID 025372830

# unconstrained_min.py

import numpy as np

def wolfe_backtrack(func2min, xi, step_direct):
    """
    Wolfe backtracking condition.
    
    Input:
    - func2min: objective function to minimize    
    - x: current position.
    - p: step direction

    Output:
    - alpha: i.e., step length

    """    
    # Init 
    # make sure 0 < c1 < c2 < 1
    alpha = 1.0
    c_wolfe_1 = 0.0001
    c_wolfe_2 = 0.9 # not needed 
    alpha_factor = 0.25
    hessian_flag = False
    max_iter = 1000
    alpha_ok = True
    
    # current position
    p = step_direct
    func_x, grad_x, NA = func2min(xi, hessian_flag)
    grad_proj_step = np.dot(grad_x, p)

    # next position
    x_next = xi - alpha*p
    func_x_next, grad_x_next, NA = func2min(x_next, hessian_flag)
    # wolfe #1 threshold
    d_grad_proj_step = alpha * c_wolfe_1 * grad_proj_step    
    # wolfe #2 threshold    
    #next_step_grad_proj = np.dot(grad_x_next, p)

    i_iter = 0
    #while (((func_x_next > func_x + d_grad_proj_step) or # wolfe 1
    #        (next_step_grad_proj < c_wolfe_2 * grad_proj_step)) and # wolfe 2
    #       (i_iter <= max_iter)):
    while ((func_x_next > func_x - d_grad_proj_step) and # wolfe 1
           (i_iter <= max_iter)):
        # search for alpha
        i_iter += 1
        alpha *= alpha_factor
        x_next = xi - alpha*p
        func_x_next, grad_x_next, NA = func2min(x_next, hessian_flag)
        d_grad_proj_step = alpha * c_wolfe_1 * grad_proj_step
        #print(" Backtrack loop({}): f(x) = {},  f(x+ap) = {} :"
        #      .format(i_iter, func_x, func_x_next))

    print(" Backtrack: alpha = {}, f(x) = {},  f(x+ap) = {}, dgrad(x) = {} :"
          .format(alpha, func_x, func_x_next, d_grad_proj_step))

    if i_iter > max_iter:
        alpha_ok = False
        
    return alpha, alpha_ok
    
def newton_descent(func2min, x0, max_iter, obj_tol, param_tol):
    """
    Newton descent method algorithm.
    
    For inputs & outputs description please refer to gradient descent function.

    """
    # Init success flag
    success_flag = True
    
    # Enable Hessian
    hessian_flag = True
    
    # Insert the initial position
    xi = x0
    
    # Init path records
    dim_x = len(xi)
    x_track = np.array(np.zeros((max_iter, dim_x)), dtype=np.float64)
    f_track = np.array(np.zeros(max_iter), dtype=np.float64)
    
    for i in range(max_iter):
        # calculate f(xi) and grad(xi):
        func_x, grad_x, hessian_x = func2min(xi, hessian_flag)
        
        # append to track records
        x_track[i, :] = xi
        f_track[i] = func_x
        
        # print current iteration status
        print("Newton: iter = {},  x = {}, f(x) = {} :".format(i+1, xi, func_x))
        
        # check for possible termination
        if i >= 1:
            dx = x_track[i-1] - x_track[i]
            if np.all(dx <= param_tol):
                print("Newton termination: small dx =", dx)
                break
            
            df = f_track[i-1] - f_track[i]
            if np.all(df <= obj_tol):
                print("Newton termination: small df =", df)
                break
            
        # find Newton direction using equation solver:
        newton_step = np.linalg.solve(hessian_x, grad_x)

        # find step length using wolfe backtrack condition
        alpha, alpha_ok = wolfe_backtrack(func2min, xi, newton_step)
        
        if alpha_ok == False:
            break
        
        # take 1 step towards opposite of current gradient
        xi -= alpha * newton_step

                
    if (i >= max_iter-1) or (alpha_ok == False):
        success_flag = False
        
    # Final status    
    print("Newton final: iter = {},  x = {}, f(x) = {}, OK = {} :"
          .format(i+1, xi, func_x, success_flag))
    
    # Results
    x_track = x_track[:(i+1), :]
    f_track = f_track[:(i+1)]
    final_x = xi
    final_fx = func_x
    
    return final_x, final_fx, x_track, f_track, success_flag

    
def gradient_descent(func2min, x0, max_iter, obj_tol, param_tol):
    """
    Gradient descent method algorithm.

    Input:
    - func2min: objective function to minimize    
    - x0: starting point.
    - max_iter: maximum allowed number of iterations.
    - obj_tol: numeric tolerance for successful termination in terms of small enough 
               change in objective function values, between two consecutive iterations.
    - param_tol: numeric tolerance for successful termination in terms of small enough 
                 distance between two consecutive iterations iteration locations.

    Returns:
    - final_x: final position, x.
    - final_fx: final f(x).
    - x_track: position x record through the process.
    - f_track: f(x) record through the process.
    
    """
    # Init success flag
    success_flag = True
    
    # No need for Hessian in gradient descent
    hessian_flag = False
    
    # Insert the initial position
    xi = x0
    
    # Init path records
    dim_x = len(xi)
    x_track = np.array(np.zeros((max_iter, dim_x)), dtype=np.float64)
    f_track = np.array(np.zeros(max_iter), dtype=np.float64)
    
    for i in range(max_iter):
        # calculate f(xi) and grad(xi):
        func_x, grad_x, NA = func2min(xi, hessian_flag)
        
        # append to track records
        x_track[i, :] = xi
        f_track[i] = func_x

        # print current iteration status
        print("GD: iter = {},  x = {}, f(x) = {} :".format(i+1, xi, func_x))
        
        # check for possible termination
        if i >= 1:
            dx = x_track[i-1] - x_track[i]
            if np.all(dx <= param_tol):
                print("GD termination: small dx =", dx)
                break
            
            df = f_track[i-1] - f_track[i]
            if np.all(df <= obj_tol):
                print("GD termination: small df =", df)
                break
            
        # find step length using wolfe backtrack condition
        # where the step_direction is the current gradient
        step_direction = grad_x
        alpha, alpha_ok = wolfe_backtrack(func2min, xi, step_direction)
        
        # take 1 step towards opposite of current gradient
        xi -= alpha * grad_x
        
        
    if (i >= max_iter-1) or (alpha_ok == False):
        success_flag = False
        
    # Final status    
    print("GD final: iter = {},  x = {}, f(x) = {}, OK = {}:"
          .format(i+1, xi, func_x, success_flag))
    
    # Results
    x_track = x_track[:(i+1), :]
    f_track = f_track[:(i+1)]
    final_x = xi
    final_fx = func_x
    
    return final_x, final_fx, x_track, f_track, success_flag
    

def bfgs_descent(func2min, x0, max_iter, obj_tol, param_tol):
    """
    BFGS quasi-newton descent method algorithm.
    
    For inputs & outputs description please refer to gradient descent function.

    """
    # Init success flag
    success_flag = True
    
    # Enable Hessian
    hessian_flag = True
    
    # Insert the initial position
    xi = x0

    # initial computation of: f(xi), grad(xi) and the hessain(xi) :
    func_x, grad_x, hessian_x = func2min(xi, hessian_flag)
    
    # Init path records
    dim_x = len(xi)
    x_track = np.array(np.zeros((max_iter, dim_x)), dtype=np.float64)
    f_track = np.array(np.zeros(max_iter), dtype=np.float64)

    # First position
    x_track[0, :] = xi
    f_track[0] = func_x
    g_prev = grad_x
    print("BFGS: iter = {},  x = {}, f(x) = {} :".format(1, xi, func_x))

    # Disable Hessian (as from now on, I use approximation for the hessian)
    hessian_flag = False
    
    for i in range(1, max_iter):
        # step direction using equation solver:
        bfgs_step = np.linalg.solve(hessian_x, grad_x)
        
        # find step length using wolfe backtrack condition
        alpha, alpha_ok = wolfe_backtrack(func2min, xi, bfgs_step)
        
        if alpha_ok == False:
            break
        
        # take 1 step towards opposite of current gradient
        xi -= alpha * bfgs_step
        
        # calculate f(xi) and grad(xi):
        func_x, grad_x, NA = func2min(xi, hessian_flag)
        
        # append to track records
        x_track[i, :] = xi
        f_track[i] = func_x
        
        # print current iteration status
        print("BFGS: iter = {},  x = {}, f(x) = {} :".format(i+1, xi, func_x))
        
        # check for possible termination
        if i >= 1:
            dx = x_track[i-1] - x_track[i]
            if np.all(dx <= param_tol):
                print("BFGS termination: small dx =", dx)
                break
            
            df = f_track[i-1] - f_track[i]
            if np.all(df <= obj_tol):
                print("BFGS termination: small df =", df)
                break
            
        # calculate position and gradient differentials
        si = x_track[i] - x_track[i-1] # i.e., Sk=X(k+1)-X(k)
        yi = grad_x - g_prev # i.e., Yk=Grad(X(k+1)) - Grad(X(k))
        g_prev = grad_x # keep the updated gradient for the next iteration
        
        # calculate an approximation for the hessian for the next iteration
        term_2_nom = np.matmul(np.outer(np.matmul(hessian_x, si), si), hessian_x)
        term_2_den = np.inner(np.matmul(si, hessian_x), si)
        if np.all(term_2_nom == 0) or np.all(term_2_den == 0):
            term_2 = np.array(np.zeros(hessian_x.shape), dtype=np.float64)
        else:
            term_2 = -1 * np.divide(term_2_nom, term_2_den)
        
        term_3_nom = np.outer(yi, yi)
        term_3_den = np.inner(yi, si)
        if np.all(term_3_nom == 0) or np.all(term_3_den == 0):
            term_3 = np.array(np.zeros(hessian_x.shape), dtype=np.float64)
        else:
            term_3 = np.divide(term_3_nom, term_3_den)
        
        hessian_x = np.sum([hessian_x, term_2, term_3], axis=0)
                    
                
    if (i >= max_iter-1) or (alpha_ok == False):
        success_flag = False
        
    # Final status    
    print("BFGS final: iter = {},  x = {}, f(x) = {}, OK = {} :"
          .format(i+1, xi, func_x, success_flag))
    
    # Results
    x_track = x_track[:(i+1), :]
    f_track = f_track[:(i+1)]
    final_x = xi
    final_fx = func_x
    
    return final_x, final_fx, x_track, f_track, success_flag


def sr1_descent(func2min, x0, max_iter, obj_tol, param_tol):
    """
    SR1 quasi-newton descent method algorithm.
    
    For inputs & outputs description please refer to gradient descent function.

    """
    # Init success flag
    success_flag = True
    
    # Enable Hessian
    hessian_flag = True
    
    # Insert the initial position
    xi = x0

    # initial computation of: f(xi), grad(xi) and the hessain(xi) :
    func_x, grad_x, hessian_x = func2min(xi, hessian_flag)
    
    # Init path records
    dim_x = len(xi)
    x_track = np.array(np.zeros((max_iter, dim_x)), dtype=np.float64)
    f_track = np.array(np.zeros(max_iter), dtype=np.float64)

    # First position
    x_track[0, :] = xi
    f_track[0] = func_x
    g_prev = grad_x
    print("SR1: iter = {},  x = {}, f(x) = {} :".format(1, xi, func_x))

    # Disable Hessian (as from now on, I use approximation for the hessian)
    hessian_flag = False
    
    for i in range(1, max_iter):
        # step direction using equation solver:
        sr1_step = np.linalg.solve(hessian_x, grad_x)

        # find step length using wolfe backtrack condition
        alpha, alpha_ok = wolfe_backtrack(func2min, xi, sr1_step)
        
        if alpha_ok == False:
            break
        
        # take 1 step towards opposite of current gradient
        xi -= alpha * sr1_step
        
        # calculate f(xi) and grad(xi):
        func_x, grad_x, NA = func2min(xi, hessian_flag)
        
        # append to track records
        x_track[i, :] = xi
        f_track[i] = func_x
        
        # print current iteration status
        print("SR1: iter = {},  x = {}, f(x) = {} :".format(i+1, xi, func_x))
        
        # check for possible termination
        if i >= 1:
            dx = x_track[i-1] - x_track[i]
            if np.all(dx <= param_tol):
                print("SR1 termination: small dx =", dx)
                break
            
            df = f_track[i-1] - f_track[i]
            if np.all(df <= obj_tol):
                print("SR1 termination: small df =", df)
                break
            
        # calculate position and gradient differentials
        si = x_track[i] - x_track[i-1] # i.e., Sk=X(k+1)-X(k)
        yi = grad_x - g_prev # i.e., Yk=Grad(X(k+1)) - Grad(X(k))
        g_prev = grad_x # keep the updated gradient for the next iteration
        
        # calculate an approximation for the hessian for the next iteration
        grad_err = yi - np.matmul(hessian_x, si)
        term_2_nom = np.outer(grad_err, grad_err)
        term_2_den = np.inner(grad_err, si)
        if np.all(term_2_nom == 0) or np.all(term_2_den == 0):
            term_2 = np.array(np.zeros(hessian_x.shape), dtype=np.float64)
        else:
            term_2 = np.divide(term_2_nom, term_2_den)

        hessian_x = np.sum([hessian_x, term_2], axis=0)


    if (i >= max_iter-1) or (alpha_ok == False):
        success_flag = False
        
    # Final status    
    print("SR1 final: iter = {},  x = {}, f(x) = {}, OK = {} :"
          .format(i+1, xi, func_x, success_flag))
    
    # Results
    x_track = x_track[:(i+1), :]
    f_track = f_track[:(i+1)]
    final_x = xi
    final_fx = func_x
    
    return final_x, final_fx, x_track, f_track, success_flag


    
def unconstrained_minimization(func2min, x0, max_iter, obj_tol, param_tol, method):

    """
    Input:
    - method_func: function argument to support various minimization methods.
                   Supported methods are the following:
                   Gradient descent, Newton, BFGS and SR1.
    - f: function minimized.
    - x0: starting point.
    - max_iter: maximum allowed number of iterations.
    - obj_tol: numeric tolerance for successful termination in terms of small enough 
               change in objective function values, between two consecutive iterations.
    - param_tol: numeric tolerance for successful termination in terms of small enough 
                 distance between two consecutive iterations iteration locations.

    Returns:
    - final location
    - final objective value
    - success/failure Boolean flag
    """
    print('The chosen method is =', method)
    
    if method == 'gd':
        final_x, final_fx, x_track, f_track, success_flag = gradient_descent(func2min, x0, max_iter, obj_tol, param_tol)
    elif method == 'newton':   
        final_x, final_fx, x_track, f_track, success_flag = newton_descent(func2min, x0, max_iter, obj_tol, param_tol)
    elif method == 'bfgs':   
        final_x, final_fx, x_track, f_track, success_flag = bfgs_descent(func2min, x0, max_iter, obj_tol, param_tol)
    elif method == 'sr1':   
        final_x, final_fx, x_track, f_track, success_flag = sr1_descent(func2min, x0, max_iter, obj_tol, param_tol)
    else:
        final_x = None
        final_fx = None
        x_track = None
        f_track = None
        success_flag = False
        print('The chosen method does not fit. Please make you use on of the following strings: gd, newton, bfgs or sr1')
        

    return (final_x, final_fx, x_track, f_track, success_flag, method)


    
    
    
    
    
    
    
    
    