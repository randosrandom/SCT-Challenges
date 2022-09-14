import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb


def import_mesh(filename):

    def create_paths():

        load_dir = "../mesh/"
        save_dir = load_dir + filename
        filepath = load_dir + filename + ".msh"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return save_dir, filepath

    def create_files(save_dir, filepath):

        with open(filepath) as f:

            sec_dim = [int(x) for x in f.readline().split()]
            nodes_coords    = np.zeros((sec_dim[0], 3))
            triangles_nodes = np.zeros((sec_dim[1], 3), dtype=int)
            edges_nodes     = np.zeros((sec_dim[2], 3), dtype=int)

            for i in range(sec_dim[0]):
                line = f.readline().split()
                node_type = 1 if int(line[2]) != 0 else 0
                nodes_coords[i] = [float(line[0]), float(line[1]), node_type]
            for i in range(sec_dim[1]):
                line = f.readline().split()
                triangles_nodes[i] = [int(line[0])-1, int(line[1])-1, int(line[2])-1]
            for i in range(sec_dim[2]):
                line = f.readline().split()
                edges_nodes[i] = [int(line[0])-1, int(line[1])-1, int(line[2])]

            np.save(save_dir + "/coords.npy",    nodes_coords)
            np.save(save_dir + "/triangles.npy", triangles_nodes)
            np.save(save_dir + "/edges.npy",     edges_nodes)

            return save_dir
    
    return create_files(*create_paths())

def build_graph(folder):

    def add_nodes(graph):
        coords = np.load(folder+"/coords.npy")
        for i, v in enumerate(coords):
            graph.add_node(i, x=v[0], y=v[1], n=v[2])
    def add_edges(graph):
        triangles = np.load(folder+"/triangles.npy")
        for triangle in triangles:
            graph.add_edge(triangle[0], triangle[1])
            graph.add_edge(triangle[1], triangle[2])
            graph.add_edge(triangle[2], triangle[0])

    graph = nx.Graph()
    add_nodes(graph)
    add_edges(graph)

    return graph

def build_adjacency(graph, folder):
    n = graph.number_of_nodes()
    adj = np.zeros((n,n), dtype=int)
    for i in range(n):
        for idx in graph.adj[i].keys():
            adj[i,idx] = 1
            adj[idx,i] = 1
    np.save(folder+"/adjacency.npy", adj)
    return adj

def import_features(filepath):

    with open(filepath, 'r') as f:

        n_nodes, time_steps = [int(n) for n in f.readline().split()]

        feature_arr = np.zeros((n_nodes, time_steps))

        for i in range(time_steps):

            for j in range(n_nodes):
                feature_arr[j][i] = np.float(f.readline()) 

            check = f.readline()

            if(check != '\n'):
                raise ValueError('Expected EOF while reading file {} at the end of time_step {}, but read {}'.format(filepath, i, check))

    return feature_arr

# Work if graph is small
def plot_graph(graph):
    plt.figure()
    nx.draw(graph, with_labels=True, font_weight='bold', node_color = "limegreen")
    plt.show()

def plot_mesh(graph, plot_nodes=False):
    plt.figure()

    edges = list(graph.edges())

    for i in range(len(edges)):
        x = []
        y = []
        n1 = edges[i][0]
        n2 = edges[i][1]
        x.append(graph.nodes.data()[n1]['x'])
        y.append(graph.nodes.data()[n1]['y'])
        x.append(graph.nodes.data()[n2]['x'])
        y.append(graph.nodes.data()[n2]['y'])
        plt.plot(x, y, marker=' ', c='b', linewidth='0.5')

    if plot_nodes:
        for i in range(graph.number_of_nodes()):
            x_coord = graph.nodes.data()[i]['x']
            y_coord = graph.nodes.data()[i]['y']
            plt.text(x_coord, y_coord, str(i), c='r', fontsize='xx-small')

    plt.axis('equal')
    plt.show()

def main():

    filename = "Th"
    folder = import_mesh(filename)
    graph = build_graph(folder)
    adj = build_adjacency(graph, folder)
    print(adj)
    #plot_graph(graph)
    plot_mesh(graph)
    return 0

if __name__ == "__main__":
    if not os.getcwd()[-3:] == "src":
        os.chdir(os.path.join(os.getcwd(),"src"))
    main()
