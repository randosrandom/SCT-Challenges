"""!
@file Exercise1.py

@brief Exercise on PCA-based shape models.

@author Stefano Pagani <stefano.pagani@polimi.it>.

@date 2022

@section Course: Scientific computing tools for advanced mathematical modelling.
"""

# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
GENERATING CONTROL POINTS : SCRIPT
"""


def readCoord(file_n, plot_flag=False):
    """
    @param file_n: file to read in order to get the boundaries of the mesh
    @param plot_flag: True if you want to plot the boundaries of the mesh
    @return:
    """
    coord_file = pd.read_csv(file_n)
    x, y = np.array(coord_file.iloc[:, 0]), np.array(coord_file.iloc[:, 1])

    if plot_flag:
        plt.scatter(coord_file['x'], coord_file['y'], s=0.5)
        plt.title("Boundaries of the template")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
    return x, y


def find_control_points_for_template(x, y, shape_type, plot_flag=False):
    num_cp_vert = 3  # num control point on vertical line
    subsampling_rate = 5

    fix_idxs = (x != 0.)

    x_t_mob = x[fix_idxs]
    xx_t = x_t_mob[::subsampling_rate]

    y_t_mob = y[fix_idxs]
    yy_t = y_t_mob[::subsampling_rate]

    x_init = np.zeros(num_cp_vert)
    y_init = np.array(np.linspace(0, 1, num_cp_vert))

    x_t_final = np.append(x_init, xx_t)
    y_t_final = np.append(y_init, yy_t)

    df = pd.DataFrame({"x_template_final": x_t_final, "y_template_final": y_t_final})
    df.to_csv("Control_points/All_cp/All_points_template.csv", index=False)
    if plot_flag:
        plt.plot(df[str("x_template_final")], df[str("y_template_final")], 'o-')
        plt.title("Control points for the template")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
    return x_t_final, y_t_final


def read_file_points(name_file, shape_type, list_by_hand, plot_flag=False):
    """
    From a GMSH FILE
    @param name_file: file where to read the control points
    @param shape_type: shape_type of the mesh we want to implement : cactus,windsock,cavolfiore
    @return: none
    """
    file = open(name_file, 'r')
    Lines = file.readlines()

    num_cp_vert = 3
    x_init = np.zeros(num_cp_vert)
    y_init = np.array(np.linspace(0, 1, num_cp_vert))

    x = []
    y = []

    [x.append(i) for i in x_init]  # add initiali points
    [y.append(j) for j in y_init]  # add initial points

    for line in Lines[::2]:
        s = line.split()
        x.append(float(s[2][1:-1]))
        y.append(float(s[3][:-1]))

    print(f"--- Writing csv file of Control_points for : {shape_type}")
    df = pd.DataFrame({str("x_" + shape_type): x, str("y_" + shape_type): y})
    df.to_csv(str("Control_points/All_cp/All_control_points_" + shape_type + ".csv"), index=False)

    if plot_flag:
        plt.plot(df["x_" + shape_type], df["y_" + shape_type], 'o-')
        plt.title("Control points for the : " + shape_type)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    df_subsample = subsample_control_points(df, shape_type, list_by_hand)
    return df_subsample


def subsample_control_points(df, shape_type, list_by_hand, plot_flag=False, flag_cut = False):
    x_temp, y_temp = np.array(df.iloc[:, 0]), np.array(df.iloc[:, 1])

    x = x_temp[list_by_hand]
    y = y_temp[list_by_hand]

    df = pd.DataFrame({str("x_" + shape_type): x, str("y_" + shape_type): y})
    df.to_csv(str("Control_points/cp/Control_points_" + shape_type + ".csv"), index=False)

    if plot_flag:
        plt.plot(x_temp, y_temp, '-')
        plt.plot(df["x_" + shape_type], df["y_" + shape_type], 'o')
        plt.title("Control points for the : " + shape_type)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
    return df


def write_file_csv(coord, shape_type='', shape_type_trns=''):
    """
    From a GMSH FILE
    @param name_file: file where to read the control points
    @param shape_type: mesh shape_type
    @param tipo di trasfomrazion
    @return: none
    """
    if shape_type_trns != '':
        print(f"--- Writing csv file for transformed boundaries of : {shape_type}")
        df = pd.DataFrame({str("x_" + shape_type): coord[:, 0], str("y_" + shape_type): coord[:, 1]})
        df.to_csv(str(f"transformations_csv/{shape_type}/{shape_type}_{shape_type_trns}.csv"), index=False)
    else:
        print(f"--- Writing csv file for boundaries of : {shape_type}")
        df = pd.DataFrame({str("x_" + shape_type): coord[:, 0], str("y_" + shape_type): coord[:, 1]})
        df.to_csv(str("Control_points/subsample/subsample_" + shape_type + ".csv"), index=False)
