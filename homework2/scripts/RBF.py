import numpy as np
import matplotlib.pyplot as plt


def RBF_comparison(coord_template, coord_controlpoints, points, generate_new_points_flag, plot_flag=False):
    """
    @param  coord_template: starting coordinates of the mesh
    @param  coord_controlpoints: coordinates where to put each control point
    @param  points: Points to transform
    @param  generate_new_points_flag: flag to generate new points for the shape
    @param  plot_flag: flag for the plot
    @return : transformed points
   """
    x, y = np.array(coord_template[:, 0]), np.array(coord_template[:, 1])
    x_control, y_control = np.array(coord_controlpoints[:, 0]), np.array(coord_controlpoints[:, 1])

    def radial_basis(x, y, eps):
        return np.exp(-eps ** 2 * (x ** 2 + y ** 2))

    # control points
    coordinates = np.column_stack((x, y))
    n_control_point = coordinates.shape[0]

    eps = 4.9  # windsock // cactus // cw

    S = np.zeros((n_control_point, n_control_point))
    Delta = np.zeros((n_control_point, 2))

    for i in range(n_control_point):
        for j in range(n_control_point):
            S[i, j] = radial_basis(coordinates[i][0] - coordinates[j][0], coordinates[i][1] - coordinates[j][1], eps)

    # displacements
    coordinates_new = np.column_stack((x_control, y_control))

    Delta = coordinates_new - coordinates

    W = np.linalg.solve(S, Delta)

    if generate_new_points_flag:
        x.resize(points.shape[0] + 3)
        y.resize(points.shape[0] + 3)
        x[3:] = points[:, 0]
        y[3:] = points[:, 1]

    phi_x = x
    phi_y = y

    for i in range(n_control_point):
        # print(i)
        phi_x = phi_x + W[i, 0] * radial_basis(x - coordinates[i][0], y - coordinates[i][1], eps)
        phi_y = phi_y + W[i, 1] * radial_basis(x - coordinates[i][0], y - coordinates[i][1], eps)

    if plot_flag:
        plt.figure()
        plt.plot(np.array(coord_controlpoints[:, 0]), np.array(coord_controlpoints[:, 1]), 'bo-')
        plt.plot(phi_x, phi_y, 'r-', linewidth=2)
        plt.legend(["Original mesh", "RBF"])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("RBF method for the boundaries")
        plt.show()

    phi = np.zeros(shape=(phi_x.shape[0], 2))
    phi[:, 0] = phi_x
    phi[:, 1] = phi_y

    return phi
