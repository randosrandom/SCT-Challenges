"""
# TODO:
1) Change control_points_generation: add a different function for saving.
2) Fix the saving for template
"""
###########################################################################
# DEBUG VARIABLE NEEDED FOR TEMPORARY CODE
DEBUG = True
FAIR_COMPARISON = True
SAME_AREA = True  # if true shapes don't have same maximum x-coordinate; if false shapes have same maximum x-coordinate
TRANSFORM_SHAPES = True
###########################################################################

# imports
import control_points_generation as cntr
import RBF as rbf
import pandas as pd
import Mesh as mesh
import trasformations as trns
import numpy as np

"""
MAIN
"""

# Definition of the points for the TEMPLATE
file_template = 'complete_boundary/boundary_points_template.csv'
x_template, y_template = cntr.readCoord(file_template, False)
all_points = np.column_stack((x_template, y_template))[(x_template > 0)]  # positive points
x_template_final, y_template_final = cntr.find_control_points_for_template(x_template, y_template, "template", False)
indexing = [i for i in range(len(x_template_final))]
template_with_all_cp = pd.DataFrame(
    {"x_template": x_template_final, "y_template": y_template_final})

print("------------------------------")
print("  Generating Control Points   ")
print("------------------------------")
# In this part of the code we generate the control points for each shape :

name_file1 = 'GMSH_points/punti_cactus.txt'
name_file2 = 'GMSH_points/punti_windsock.txt'
name_file3 = 'GMSH_points/punti_cw.txt'
name_file4 = 'GMSH_points/punti_cauli.txt'

list_by_hand = [0, 1, 2, 4, 7, 10, 12, 14, 15, 16, 17, 18, 20, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 35, 36, 37,
                40, 42, 44, 45, 48, 50, 52, 54]
if DEBUG:
    df_template_subsample = cntr.subsample_control_points(template_with_all_cp, "template", list_by_hand)
    df_cactus_subsample = cntr.read_file_points(name_file1, 'cactus', list_by_hand)
    df_windsock_subsample = cntr.read_file_points(name_file2, 'windsock', list_by_hand)
    df_cw_subsample = cntr.read_file_points(name_file3, 'cw', list_by_hand)
    df_cauli_subsample = cntr.read_file_points(name_file4, 'cauli', list_by_hand)

# The code needs to be fixed. Indeed there is no point in rigenerating template control points
# each time, and reading statically the control points for the shape.
# It is also dangerous to add subsampling in this way!

if TRANSFORM_SHAPES:
    # for cw
    temp = np.column_stack((df_cw_subsample.iloc[:, 0], df_cw_subsample.iloc[:, 1]))
    # temp = trns.rotation_transform(temp, -np.pi/12, "cw", 1.8, False)
    temp = trns.rotation_transform(temp, np.pi / 12, 1.1, True)

    bounds = [1, 1.7]
    scale = 0.95
    df_cw_subsample = pd.DataFrame(trns.y_transform(temp, scale, bounds, True))

"""
Ora possiamo generare le rispettive mesh : sfruttiamo le mesh generate dal RBF !!
"""

print("------------------------------")
print("    Generating boundaries     ")
print("------------------------------")

coord_template = rbf.RBF_comparison(df_template_subsample.iloc, df_template_subsample.iloc, all_points,
                                    generate_new_points_flag=True)
cntr.write_file_csv(coord_template, 'template')
coord_cw = rbf.RBF_comparison(df_template_subsample.iloc, df_cw_subsample.iloc, all_points,
                              generate_new_points_flag=True)
cntr.write_file_csv(coord_cw, 'cw')
coord_windsock = rbf.RBF_comparison(df_template_subsample.iloc, df_windsock_subsample.iloc, all_points,
                                    generate_new_points_flag=True)
cntr.write_file_csv(coord_windsock, 'windsock')
coord_cactus = rbf.RBF_comparison(df_template_subsample.iloc, df_cactus_subsample.iloc, all_points,
                                  generate_new_points_flag=True)
cntr.write_file_csv(coord_cactus, 'cactus')
coord_cauli = rbf.RBF_comparison(df_template_subsample.iloc, df_cauli_subsample.iloc, all_points,
                                 generate_new_points_flag=True)
cntr.write_file_csv(coord_cauli, 'cauli')

"""
FAIR COMPARISON
"""

print("------------------------------")
print("      Fair comparison         ")
print("------------------------------")

# Try to normalized Area of the shapes, by imposing same x-length in the four morphologies

if FAIR_COMPARISON:

    x_common = 1.8  # common_x

    if SAME_AREA:
        # ADJUSTMENTS TO HAVE EQUAL AREA
        cw_adjust = -0.25  # !!!
        cauli_adjust = 0.1
        cactus_adjust = 0.1
        windsock_adjust = 0.0
    else:
        cw_adjust = 0.  # !!!
        cauli_adjust = 0.
        cactus_adjust = 0.
        windsock_adjust = 0.

    x_max_cw = np.max(coord_cw[:, 0])
    fair_coord_cw = trns.x_transform(coord_cw, (x_common + cw_adjust) / x_max_cw)
    cntr.write_file_csv(fair_coord_cw, 'cw')

    x_max_windsock = np.max(coord_windsock[:, 0])
    fair_coord_windsock = trns.x_transform(coord_windsock, (x_common + windsock_adjust) / x_max_windsock)
    cntr.write_file_csv(fair_coord_windsock, 'windsock')

    x_max_cactus = np.max(coord_cactus[:, 0])
    fair_coord_cactus = trns.x_transform(coord_cactus, (x_common + cactus_adjust) / x_max_cactus)
    cntr.write_file_csv(fair_coord_cactus, 'cactus')

    x_max_cauli = np.max(coord_cauli[:, 0])
    fair_coord_cauli = trns.x_transform(coord_cauli, (x_common + cauli_adjust) / x_max_cauli)
    cntr.write_file_csv(fair_coord_cauli, 'cauli')

print("------------------------------")
print("      Creating meshes         ")
print("------------------------------")

template_mesh = mesh.Mesh('Control_points/subsample/subsample_template.csv', 'template')
template_mesh.create_mesh()

cactus_mesh = mesh.Mesh('Control_points/subsample/subsample_cactus.csv', 'cactus')
cactus_mesh.create_mesh()

windsock_mesh = mesh.Mesh('Control_points/subsample/subsample_windsock.csv', 'windsock')
windsock_mesh.create_mesh()

cw_mesh = mesh.Mesh('Control_points/subsample/subsample_cw.csv', 'cw')
cw_mesh.create_mesh()

cauli_mesh = mesh.Mesh('Control_points/subsample/subsample_cauli.csv', 'cauli')
cauli_mesh.create_mesh()

"""
TRANSFORMATION FOR THE MESHES
"""

print("------------------------------")
print("      Transformations         ")
print("------------------------------")

# Transformations

# Cut for the meshes :

trns.cut(cauli_mesh)
trns.cut(cw_mesh)
trns.cut(windsock_mesh)
trns.cut(cactus_mesh)

# ANGLE INDICATOR

sub_file_name_1 = 'Control_points/subsample/subsample_cactus.csv'
sub_file_name_2 = 'Control_points/subsample/subsample_windsock.csv'
sub_file_name_3 = 'Control_points/subsample/subsample_cw.csv'
sub_file_name_4 = 'Control_points/subsample/subsample_cauli.csv'

angle_cactus = trns.angle_indicator(sub_file_name_1)
angle_windsock = trns.angle_indicator(sub_file_name_2)
angle_cw = trns.angle_indicator(sub_file_name_3)
angle_cauli = trns.angle_indicator(sub_file_name_4)
