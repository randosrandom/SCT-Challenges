import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import meshpy.triangle as triangle


class Mesh:
    # The init method
    def __init__(self, file, nome):
        """
        @param file: file with the boundary points
        @param nome: shape_type of mesh
        """
        # Instance Variable
        self.file = file
        self.shape_type = nome
        self.points = []
        self.attr = []
        self.elements = []
        self.bound_points = []
        self.facets = []
        self.markers = []

    def __repr__(self):
        s = "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n" + \
            "                     MESH PROPERTIES : " + f"{self.shape_type}                    \n" + \
            "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n" + \
            "Shape of the mesh :  Triangles \n" + \
            "Number of points :  " + str(np.array(self.points).shape[0]) + "\n" + \
            "Number of elements :  " + str(np.array(self.elements).shape[0]) + "\n" + \
            "Number of boundary points :  " + str(np.array(self.facets).shape[0])
        return s

    def create_mesh(self, flag_cut=False, x_max = 0.6):
        """
        @param flag_cut: flag to cut the mesh
        """

        def round_trip_connect(start, end):
            """
            @param start: starting point
            @param end: ending point
            @return: connection
            """
            return [(i, i + 1) for i in range(start, end)] + [(end, start)]

        def refinement_func(tri_points, area):
            """
            Function to refine the mesh
            """
            max_area = 0.001
            return bool(area > max_area)

        df = pd.read_csv(self.file)
        x = np.array(df.iloc[:, 0])
        y = np.array(df.iloc[:, 1])

        if flag_cut:
            lun = df.iloc[:, 0].shape[0]
            lista = []
            for idx in range(lun):
                if df.iloc[idx, 0] > x_max:
                    lista.append(idx)
            x = np.delete(x, lista)
            y = np.delete(y, lista)

        self.bound_points = np.zeros(shape=(len(x), len(y)))
        self.bound_points[:, 0] = x
        self.bound_points[:, 1] = y

        points = [(self.bound_points[i, 0], self.bound_points[i, 1]) for i in range(self.bound_points.shape[0])]
        self.facets = facets = round_trip_connect(0, len(points) - 1)
        self.markers = [1 for i in range(len(facets))]

        info = triangle.MeshInfo()
        info.set_points(points)
        info.set_facets(facets, facet_markers=self.markers)
        # build the triangles
        mesh = triangle.build(info, refinement_func=refinement_func)

        self.points = np.array(mesh.points)
        self.elements = np.array(mesh.elements)
        self.attr = np.array(mesh.point_markers)
        self.facets = np.array(mesh.facets)

        self.write_msh(flag_cut)


    def plot_mesh(self):
        """
        Function to plot the mesh
        """
        n = np.size(self.attr)
        outer_nodes = [i for i in range(n) if self.attr[i] == 1]
        plt.triplot(self.points[:, 0], self.points[:, 1], self.elements)
        plt.plot(self.points[outer_nodes, 0], self.points[outer_nodes, 1], "r.")
        plt.title(str("Mesh for : " + self.shape_type))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def write_msh(self, flag=False, flag_trans=False, shape_type=''):
        """
        Function to write the .msh file
        @param flag: flag to impose cut on the mesh
        @param flag_trans: flag to impose transformation on the mesh
        @param shape_type: shape_type of the mesh
        """
        if flag:
            print(f"--- Writing the .msh file for : {self.shape_type} cut ---")
            file_to_write = str(f"geometries/new_meshes/cut/{self.shape_type}.msh")
        elif flag_trans:
            print(f"--- Writing the .msh file for : {self.shape_type} trans ---")
            file_to_write = str(f"geometries/new_meshes/transformation/{shape_type}/{self.shape_type}.msh")
        else:
            print(f"--- Writing the .msh file for : {self.shape_type} ---")
            file_to_write = str(f"geometries/new_meshes/{self.shape_type}.msh")

        with open(file_to_write, 'w') as file:
            file.write("$MeshFormat\n")
            file.write("2.2 0 8\n")
            file.write("$EndMeshFormat\n")
            file.write("$Nodes\n")
            file.write(str(self.points.shape[0]) + '\n')
            var1_2 = []
            for i in range(len(self.points)):
                s = str(i + 1) + " " + str(self.points[i][0]) + " " + str(self.points[i][1]) + " 0" + '\n'
                file.write(s)

                if self.attr[i] == 1:
                    if self.points[i][0] < 0.0001:
                        var1_2.append("2 2")
                    else:
                        var1_2.append("1 1")
                else:
                    var1_2.append("2 1")
            file.write("$EndNodes\n")
            file.write("$Elements\n")
            file.write(str(self.facets.shape[0] + self.elements.shape[0]) + '\n')

            j = 0
            for i in range(len(self.facets)):
                s = str(i + 1) + " 1 2 " + str(var1_2[self.facets[i][0]]) + " " + str(
                    self.facets[i][0] + 1) + " " + str(self.facets[i][1] + 1) + '\n'
                file.write(s)
                j = j + 1

            for i in range(len(self.elements)):
                s = str(j + i + 1) + " 2 2 2 1 " + str(self.elements[i][0] + 1) + " " + str(
                    self.elements[i][1] + 1) + " " + str(self.elements[i][2] + 1) + '\n'
                file.write(s)

            file.write("$EndElements")

            file.close()

