import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def symbolic_jacobian_equivalent(func_list, var_list):
    jacobian_matrix = sp.Matrix(func_list).jacobian(var_list)
    return jacobian_matrix

def numerical_jacobian_equivalent(func, point, eps=1e-6):
    n = len(point)
    fx = func(point)
    m = len(fx)
    jacobian_matrix = np.zeros((m, n))
    i = 0
    while i < n:
        point_perturbed = point.copy()
        point_perturbed[i] += eps
        jacobian_matrix[:, i] = (func(point_perturbed) - fx) / eps
        i += 1
    return jacobian_matrix

def plot_jacobian_equivalent(func, x_range, y_range, resolution=20):
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    det_jacobian = np.zeros_like(X)

    def wrapper(point):
        return np.array(func(point))

    i = 0
    while i < resolution:
        j = 0
        while j < resolution:
            point = np.array([X[i, j], Y[i, j]])
            jacobian_matrix = numerical_jacobian_equivalent(wrapper, point)
            det_jacobian[i, j] = np.linalg.det(jacobian_matrix)
            j += 1
        i += 1

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, det_jacobian, cmap='viridis')
    ax.set_title('Determinant of the Jacobian')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('det(J)')
    plt.show()

# Example usage

# Symbolic example
x, y = sp.symbols('x y')
func_list = [x**2 + y**2, sp.sin(x) * sp.cos(y)]
var_list = [x, y]
jacobian_sym = symbolic_jacobian_equivalent(func_list, var_list)
print("Symbolic Jacobian:")
sp.pprint(jacobian_sym)

# Numerical example
def example_function(point):
    x, y = point
    return np.array([x**2 + y**2, np.sin(x) * np.cos(y)])

point_example = np.array([1.0, 2.0])
jacobian_num = numerical_jacobian_equivalent(example_function, point_example)
print("\nNumerical Jacobian at point {}:".format(point_example))
print(jacobian_num)

# Plot
plot_jacobian_equivalent(example_function, x_range=(-2, 2), y_range=(-2, 2))