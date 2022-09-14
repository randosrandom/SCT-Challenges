import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import control_points_generation as cntr
import Mesh as mesh
import copy


def x_transform(coordinates, scale):
    """
    @param coordinates: Points coordinates
    @param scale: scale to reduce/increment the x_value of the points
    @return: transformed coordinates
    """
    coordinates_x_trans = np.array(coordinates)
    coordinates_x_trans[:, 0] *= scale
    return coordinates_x_trans


def y_transform(coordinates, scale, bounds, centering=False):
    """
    @param coordinates: cooridnates of the points
    @param scale: scale to reduce/increment the mesh
    @param bounds:
    @param centering:
    @return: transformed coordinates
    """
    if scale < 1:
        print("There may be crossings if scale < 1! Check the shape after the transformation!")

    coordinates_y_trans = np.array(coordinates)
    x_max_idx = list(coordinates[:, 0] == np.max(coordinates[:, 0])).index(True)

    mean = 0.5

    if centering:
        coordinates_y_trans[:, 1] = coordinates_y_trans[:, 1] - mean

    x_bound_min = bounds[0]
    x_bound_max = bounds[1]

    trans_list = (coordinates[:, 0] > x_bound_min) & (coordinates[:, 0] < x_bound_max)

    for i in range(x_max_idx):
        if trans_list[i]:
            coordinates_y_trans[i, 1] = y_helper(np.array(coordinates_y_trans[i, 1]), scale, True)

    for i in range(x_max_idx, coordinates.shape[0]):
        if trans_list[i]:
            coordinates_y_trans[i, 1] = y_helper(np.array(coordinates_y_trans[i, 1]), scale, False)

    # for i in range(x_max_idx):
    #   for j in range(x_max_idx, coordinates.shape[0]):
    #        if coordinates_y_trans[j][1] > coordinates_y_trans[i][1]:
    #            if np.abs(coordinates_y_trans[i][0] - coordinates_y_trans[j][0]) < 3e-2:
    #                raise ValueError("Crossing happended!")

    if centering:
        coordinates_y_trans[:, 1] = coordinates_y_trans[:, 1] + mean

    return coordinates_y_trans


def y_helper(y, scale, upper_part=True):
    """
    @param y: initial coordinates
    @param scale: scare to increment/recude the lower part of the mesh
    @param upper_part: flag
    @return: modified coordinates
    """
    if upper_part:
        y_pos_scale_idxs = (y >= 0)
        y_neg_scale_idxs = (y < 0)
    else:
        y_pos_scale_idxs = (y < 0)
        y_neg_scale_idxs = (y >= 0)

    y[y_pos_scale_idxs] *= scale
    y[y_neg_scale_idxs] = 2 * y[y_neg_scale_idxs] - y[y_neg_scale_idxs] * scale

    return y


def rotation_transform(coordinates, theta, x_ref=0.75, flag_second_part=True):
    """
    @param coordinates: initial coordinates
    @param theta: angle
    @param x_ref: x_min where to start the rotation
    @param flag_second_part:
    @return:
    """
    if flag_second_part:
        rot_idxs = (coordinates[:, 0] > x_ref)
        coordinates_theta = np.array(coordinates[rot_idxs])
        return_array = np.array(coordinates)
        coeff = 0
        for i in range(coordinates_theta.shape[0]):
            coeff = (coordinates_theta[i, 0] - x_ref) / np.max(coordinates_theta[:, 0])
            phi_1 = lambda x, y: np.cos(theta * coeff) * (x) - np.sin(theta * coeff) * (y)
            phi_2 = lambda x, y: np.sin(theta * coeff) * (x) + np.cos(theta * coeff) * (y)

            coordinates_theta[i, 0] = phi_1(coordinates[rot_idxs][i, 0], coordinates[rot_idxs][i, 1])
            coordinates_theta[i, 1] = phi_2(coordinates[rot_idxs][i, 0], coordinates[rot_idxs][i, 1])

        j = 0
        for i in list(np.where(rot_idxs)[0]):
            return_array[i] = coordinates_theta[j]
            j = j + 1

    else:
        rot_idxs = (coordinates[:, 0] < x_ref) & (coordinates[:, 0] > 0)
        coordinates_theta = np.array(coordinates[rot_idxs])
        return_array = np.array(coordinates)
        coeff = 0

        for i in range(coordinates_theta.shape[0]):
            coeff = (np.abs(coordinates_theta[i, 0] - x_ref)) / np.max(coordinates_theta[:, 0])
            phi_1 = lambda x, y: np.cos(theta * coeff) * (x) - np.sin(theta * coeff) * (y)
            phi_2 = lambda x, y: np.sin(theta * coeff) * (x) + np.cos(theta * coeff) * (y)

            coordinates_theta[i, 0] = phi_1(coordinates[rot_idxs][i, 0], coordinates[rot_idxs][i, 1])
            coordinates_theta[i, 1] = phi_2(coordinates[rot_idxs][i, 0], coordinates[rot_idxs][i, 1])

        j = 0
        for i in list(np.where(rot_idxs)[0]):
            return_array[i] = coordinates_theta[j]
            j = j + 1

    # Potrebbe sistemare questa cosa
    for i in range(return_array.shape[0]):
        if return_array[i, 0] < 0:
            return_array[i, 0] = 0.

    return return_array


def plot_transformation(template, new_coord):
    """
    FUnction to plot the transformation
    @param template: initial coordinates
    @param new_coord: transformed coordinates
    """
    plt.plot(template[:, 0], template[:, 1])
    plt.plot(new_coord[:, 0], new_coord[:, 1])
    plt.legend(["Template", "New"])
    plt.show()


def noise(coordinates):
    coordinates_new = np.array(coordinates)
    for i in range(coordinates_new.shape[0]):
        if coordinates_new[i, 0] > 0.01:
            coordinates_new[i, 0] += np.random.normal(-0.02, 0.02)
            coordinates_new[i, 1] += np.random.normal(-0.02, 0.02)
    return coordinates_new


def write_transformations(new_coord, shape_type, shape_type_trans):
    """
    @param new_coord: coordinates of the mesh
    @param shape_type: shape_type of the mesh
    @param shape_type_trans: type of transformation on the mesh
    """
    file = cntr.write_file_csv(new_coord, shape_type, shape_type_trans)
    new_mesh = mesh.Mesh(f'transformations_csv/{shape_type}/{shape_type}_{shape_type_trans}.csv',
                         shape_type + "_" + shape_type_trans)
    coord_mesh = new_mesh.create_mesh()
    new_mesh.write_msh(True, shape_type)


def serial_rotation(original_mesh, shape_type):
    """
    Function to do lots of transformation one after the other
    """
    print("+++++++++++++++++++++++++++++++++++++")
    print("        Serial Transformation        ")
    print("+++++++++++++++++++++++++++++++++++++")

    x1 = [np.pi / j for j in range(-14, -10, 2)]
    xx1 = [np.pi / j for j in range(10, 14, 2)]
    x1.extend(xx1)

    # NOISE
    for i in range(4):
        new_coord = noise(original_mesh.bound_points)
        write_transformations(np.array(new_coord), shape_type, f"noise_{i}")

    # ROTAZIONE
    for i in x1:
        new_coord = rotation_transform(original_mesh.bound_points, i, shape_type)
        write_transformations(np.array(new_coord), shape_type, f"rot_({round(i, 3)})")

    # TRASLAZIONE
    x2 = [i for i in np.linspace(0.5, 1, 4)]
    xx2 = [x2[i] + 0.5 for i in range(len(x2))]
    x2.extend(xx2)

    for i in x2:
        new_coord = x_transform(original_mesh.bound_points, i)
        write_transformations(np.array(new_coord), shape_type, f"x_tr_({round(i, 3)})")

    # RANDOM
    for i in x2:
        new_coord = x_transform(original_mesh.bound_points, i)
        for j in x1:
            new_new_coord = rotation_transform(new_coord, j, 0.75 * i)
            write_transformations(np.array(new_new_coord), shape_type, f"x_tr_({round(i, 3)})_rot_({round(j, 3)})")


# cauli : 1.8 > i[0] > 1.16325382e-18 ; i[1] < 0.6
# CW : if 1 > i[0] and i[1] < 0.45 or 1.6 > i[0] > 1 and i[1] < 0.1
def prova_trasform(coord, scale):
    """
    @param coord: coordinates of the mesh
    @param scale: scale to change the lower part of the mesh
    @return: transformed coordinates
    """
    coordinate = np.array(coord)
    new = np.array([i for i in coordinate[3:, :] if 1.8 > i[0] > 1.16325382e-18 and i[1] < 0.6])
    new[:, 1] = new[:, 1] * (1 / scale)
    coordinate[coordinate.shape[0] - new.shape[0]:, :] = new
    print(new[-1:new.shape[0], :])
    return coordinate


def cut(Meshtype, x_max = 0.6):
    """
    @param Meshtype: mesh to cut, ONLY THE CONSTRUCTOR
    """
    new_Mesh = copy.deepcopy(Meshtype)
    new_Mesh.create_mesh(True, x_max)

def angle_indicator(sub_file_name, x_max_for_angle = 0.6, x_min_for_angle = 0.): # finite differences with penalization
        
    coord_file = pd.read_csv(sub_file_name)
    x, y = np.array(coord_file.iloc[:, 0]), np.array(coord_file.iloc[:, 1])
    
    x_max_index = list(x == np.max(x)).index(True)
    
    x = x[x_max_index:]
    y = y[x_max_index:]
    
    points_idx = (x < x_max_for_angle) & (x > x_min_for_angle)
    
    x = x[points_idx]
    y = y[points_idx]
    
    return np.sum(np.diff(y) / np.diff(x) * (x_max_for_angle -x[1:])) / len(np.diff(y)) 