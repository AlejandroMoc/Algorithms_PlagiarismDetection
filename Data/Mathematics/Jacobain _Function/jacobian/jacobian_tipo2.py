import numpy as np 
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def bizzaro_matrix(funcs, vars):
    """
    Compute the bizzaro matrix
    args:
        funcs: list of bizzaro functions
        vars: list of bizzaro variables
    return:
        Sympy Matrix: The bizzaro matrix
    """
    matrix_bizzaro = sp.Matrix(funcs).jacobian(vars)
    return matrix_bizzaro

def funky_matrix_calculation(func, x, eps=1e-6):
    """ 
    Compute the funky matrix using finite differences
    Args:
        func: Callable function, takes an input vector and returns an output vector
        x: Point at which to evaluate the funky matrix (numpy array)
        eps: Small perturbation for finite differences
    Returns:
        numpy array: The funky matrix
    """
    
    n = len(x)
    fx = func(x)
    m = len(fx)
    matrix_funky = np.zeros((m, n))
    for i in range(n):
        x_perturbed = x.copy()       
        x_perturbed[i] += eps
        matrix_funky[:, i] = (func(x_perturbed) - fx) / eps
    return matrix_funky

def plot_funky_matrix(func, x_range, y_range, resolution=20):
    """ 
    Plot the funky determinant on a grid
    Args:
        func: Callable function mapping R^2 -> R^2.
        x_range: Tuple defining the x-range (min, max).
        y_range: Tuple defining the y-range (min, max).
        resolution: Number of points per dimension.
    """
    
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    det_funky = np.zeros_like(X)
    
    def wrapper(point):
        return np.array(func(point))
    
    for i in range(resolution):
        for j in range(resolution):
            point = np.array([X[i, j], Y[i, j]])
            funky_matrix = funky_matrix_calculation(wrapper, point)
            det_funky[i, j] = np.linalg.det(funky_matrix)
            
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, det_funky, cmap='viridis')
    ax.set_title('Determinant of the Funky Matrix')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('det(F)')
    plt.show()
    
# Example usage

# Symbolic example
x, y = sp.symbols('x y')
funcs = [x**2 + y**2, sp.sin(x) * sp.cos(y)]
vars = [x, y]
matrix_sym = bizzaro_matrix(funcs, vars)
print("Symbolic Bizzaro Matrix:")
sp.pprint(matrix_sym)

# Numerical example
def funky_func(point):
    x, y = point
    return np.array([x**2 + y**2, np.sin(x) * np.cos(y)])

point = np.array([1.0, 2.0])
matrix_num = funky_matrix_calculation(funky_func, point)
print("\nNumerical Bizzaro Matrix at point {}:".format(point))
print(matrix_num)

# plot
plot_funky_matrix(funky_func, x_range=(-2, 2), y_range=(-2, 2)) 