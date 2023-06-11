# Moti Marom
# ID 025372830

# examples.py

import numpy as np

def example_func_quad_1(X, en_hessian):
    # circle example
    Q = np.array(np.zeros((2, 2)), dtype=np.float64)
    Q[0][0] = 1
    Q[1][1] = 1
    func_x = np.matmul(np.matmul(np.transpose(X),  Q), X)

    grad_x = np.array(np.zeros(2), dtype=np.float64)
    grad_x[0] = 2 * Q[0][0] * X[0] + (Q[0][1] + Q[1][0]) * X[1]
    grad_x[1] = 2 * Q[1][1] * X[1] + (Q[0][1] + Q[1][0]) * X[0]
    
    hessian_x = []
    
    if en_hessian == True:
        hessian_x = np.array(np.zeros((2, 2)), dtype=np.float64)
        hessian_x[0][0] = 2 * Q[0][0]
        hessian_x[0][1] = Q[0][1] + Q[1][0]
        hessian_x[1][0] = Q[0][1] + Q[1][0]
        hessian_x[1][1] = 2 * Q[1][1]

    return func_x, grad_x, hessian_x
    
def example_func_quad_2(X, en_hessian):
    # ellipse example
    Q = np.array(np.zeros((2, 2)), dtype=np.float64)
    Q[0][0] = 1
    Q[1][1] = 100
    func_x = np.matmul(np.matmul(np.transpose(X),  Q), X)
    
    grad_x = np.array(np.zeros(2), dtype=np.float64)
    grad_x[0] = 2 * Q[0][0] * X[0] + (Q[0][1] + Q[1][0]) * X[1]
    grad_x[1] = 2 * Q[1][1] * X[1] + (Q[0][1] + Q[1][0]) * X[0]
    
    hessian_x = []
    
    if en_hessian == True:
        hessian_x = np.array(np.zeros((2, 2)), dtype=np.float64)
        hessian_x[0][0] = 2 * Q[0][0]
        hessian_x[0][1] = Q[0][1] + Q[1][0]
        hessian_x[1][0] = Q[0][1] + Q[1][0]
        hessian_x[1][1] = 2 * Q[1][1]
    
    return func_x, grad_x, hessian_x
    
def example_func_quad_3(X, en_hessian):
    # rotated ellipse example
    Core = np.array(np.zeros((2, 2)), dtype=np.float64)
    Core[0][0] = 100
    Core[1][1] = 1
    RotMat = np.array(np.zeros((2, 2)), dtype=np.float64)
    RotMat[0][0] = 0.5*np.sqrt(3)
    RotMat[0][1] = -0.5
    RotMat[1][0] = 0.5
    RotMat[1][1] = 0.5*np.sqrt(3)
    Q = np.matmul(np.matmul(np.transpose(RotMat),  Core), RotMat)

    func_x = np.matmul(np.matmul(np.transpose(X),  Q), X)
    
    grad_x = np.array(np.zeros(2), dtype=np.float64)
    grad_x[0] = 2 * Q[0][0] * X[0] + (Q[0][1] + Q[1][0]) * X[1]
    grad_x[1] = 2 * Q[1][1] * X[1] + (Q[0][1] + Q[1][0]) * X[0]
    
    hessian_x = []
    
    if en_hessian == True:
        hessian_x = np.array(np.zeros((2, 2)), dtype=np.float64)
        hessian_x[0][0] = 2 * Q[0][0]
        hessian_x[0][1] = Q[0][1] + Q[1][0]
        hessian_x[1][0] = Q[0][1] + Q[1][0]
        hessian_x[1][1] = 2 * Q[1][1]
    
    return func_x, grad_x, hessian_x
    
def example_func_rosenbrock(X, en_hessian):
    # Rosenbrock example
    func_x = (1 - X[0]) ** 2 + 100. * (X[1] - X[0] ** 2) ** 2
    #func_x = 100 * np.power((X[1] - np.power(X[0], 2)), 2) + np.power(1 - X[0], 2)
    
    grad_x = np.array(np.zeros(2), dtype=np.float64)
    grad_x[0] = 400*np.power(X[0], 3) - 400*X[0]*X[1] + 2*X[0] - 2
    grad_x[1] = -200*np.power(X[0], 2) + 200*X[1]
    
    hessian_x = []
    
    if en_hessian == True:
        hessian_x = np.array(np.zeros((2, 2)), dtype=np.float64)
        hessian_x[0][0] = 1200*np.power(X[0], 2) - 400*X[1] + 2
        hessian_x[0][1] = - 400*X[0]
        hessian_x[1][0] = - 400*X[0]
        hessian_x[1][1] = 200
    
    return func_x, grad_x, hessian_x

def example_func_linear(X, en_hessian):
    # linear example
    dim_x = len(X)
    a = np.ones(dim_x)
    
    func_x = np.dot(a, X)
    
    grad_x = a
    
    hessian_x = []
    
    if en_hessian == True:
        hessian_x = np.zeros((dim_x, dim_x), dtype=np.float64)
    
    return func_x, grad_x, hessian_x

def example_func_nonquad(X, en_hessian):
    # non quadratic exponential example
    func_x = np.exp(X[0]+3*X[1]-0.1) + np.exp(X[0]-3*X[1]-0.1) + np.exp(-X[0]-0.1)
    
    grad_x = np.array(np.zeros(2), dtype=np.float64)
    grad_x[0] = np.exp(X[0]+3*X[1]-0.1) + np.exp(X[0]-3*X[1]-0.1) - np.exp(-X[0]-0.1)
    grad_x[1] = 3*np.exp(X[0]+3*X[1]-0.1) - 3*np.exp(X[0]-3*X[1]-0.1)
    
    hessian_x = []
    
    if en_hessian == True:
        hessian_x = np.array(np.zeros((2, 2)), dtype=np.float64)
        hessian_x[0][0] = np.exp(X[0]+3*X[1]-0.1) + np.exp(X[0]-3*X[1]-0.1) + np.exp(-X[0]-0.1)
        hessian_x[0][1] = 3*np.exp(X[0]+3*X[1]-0.1) - 3*np.exp(X[0]-3*X[1]-0.1)
        hessian_x[1][0] = 3*np.exp(X[0]+3*X[1]-0.1) - 3*np.exp(X[0]-3*X[1]-0.1)
        hessian_x[1][1] = 9*np.exp(X[0]+3*X[1]-0.1) + 9*np.exp(X[0]-3*X[1]-0.1)
    
    return func_x, grad_x, hessian_x
