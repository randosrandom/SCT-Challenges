# imports
import control_points_generation as cntr
import RBF as rbf
import pandas as pd
import Mesh as mesh
import trasformations as trns
import numpy as np


# MACROS
SUBSAMPLE = True
FAIR_COMPARISON = True
SAME_AREA = True  # if true shapes don't have same maximum x-coordinate; if false shapes have same maximum x-coordinate
TRANSFORM_SHAPES = True

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
if SUBSAMPLE:
    df_template_subsample = cntr.subsample_control_points(template_with_all_cp, "template", list_by_hand)
    df_cactus_subsample = cntr.read_file_points(name_file1, 'cactus', list_by_hand)
    df_windsock_subsample = cntr.read_file_points(name_file2, 'windsock', list_by_hand)
    df_cw_subsample = cntr.read_file_points(name_file3, 'cw', list_by_hand)
    df_cauli_subsample = cntr.read_file_points(name_file4, 'cauli', list_by_hand)

# tranform shapes in order to prove that funnel and slope are the most determinant
if TRANSFORM_SHAPES:
    # for cactus
    bounds = [0, 3]
    temp_cactus = np.column_stack((df_cactus_subsample.iloc[:, 0], df_cactus_subsample.iloc[:, 1]))
    temp_cactus = trns.y_transform(temp_cactus, 0.9, bounds, True)
    df_cactus_subsample = pd.DataFrame(temp_cactus)

    
    # for cw
    bounds = [0, 2.2]
    temp_cw = np.column_stack((df_cw_subsample.iloc[:, 0], df_cw_subsample.iloc[:, 1]))
    temp_cw = trns.y_transform(temp_cw, 0.75 , bounds, True)
    temp_cw = trns.rotation_transform(temp_cw, np.pi / 24 , 1, False)
    df_cw_subsample = pd.DataFrame(temp_cw)

    
    #for windsock
    bounds = [0, 3]
    temp_windsock = np.column_stack((df_windsock_subsample.iloc[:, 0], df_windsock_subsample.iloc[:, 1]))
    temp_windsock = trns.rotation_transform(temp_windsock, - np.pi / 36, 1, False)
    temp_windsock = trns.y_transform(temp_windsock, 1.1, bounds, True)
    df_windsock_subsample = pd.DataFrame(temp_windsock)

    
    #for cauli
    bounds = [0.25, 1.3]
    temp_cauli = np.column_stack((df_cauli_subsample.iloc[:, 0], df_cauli_subsample.iloc[:, 1]))
    temp_cauli = trns.rotation_transform(temp_cauli, - np.pi / 12 , 1.5, False)
    temp_cauli = trns.y_transform(temp_cauli, 1.3, bounds, True)
    df_cauli_subsample = pd.DataFrame(temp_cauli)

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
        cactus_adjust = 0.32
        cw_adjust =  0.11
        windsock_adjust =  - 0.17
        cauli_adjust = - 0.27
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

x_max = 0.6

trns.cut(cauli_mesh, x_max = x_max)
trns.cut(cw_mesh, x_max = x_max)
trns.cut(windsock_mesh, x_max = x_max)
trns.cut(cactus_mesh, x_max = x_max)

# ANGLE INDICATOR

sub_file_name_1 = 'Control_points/subsample/subsample_cactus.csv'
sub_file_name_2 = 'Control_points/subsample/subsample_windsock.csv'
sub_file_name_3 = 'Control_points/subsample/subsample_cw.csv'
sub_file_name_4 = 'Control_points/subsample/subsample_cauli.csv'

angle_cactus = trns.angle_indicator(sub_file_name_1, x_max)
angle_windsock = trns.angle_indicator(sub_file_name_2, x_max)
angle_cw = trns.angle_indicator(sub_file_name_3, x_max)
angle_cauli = trns.angle_indicator(sub_file_name_4, x_max)
          

          