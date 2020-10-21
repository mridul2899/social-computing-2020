import snap
Rnd = snap.TRnd(42)
Rnd.Randomize()

import sys
import os
import numpy as np

def print_nodes_and_edges(G):
    """
    Prints the number of nodes and edges in the subgraph G
    """

    print("Number of nodes:", G.GetNodes())
    print("Number of edges:", G.GetEdges())


def print_degrees(G):
    """
    Prints the number of nodes having degree=7 in the graph
    Also prints the nodes with the highest degree in the graph
    """

    print('Number of nodes with degree=7:', snap.CntDegNodes(G, 7))

    highest_deg = 0
    highest_deg_list = []
    for node in G.Nodes():
        degree = node.GetDeg()

        if degree > highest_deg:
            highest_deg = degree
            highest_deg_list = []
            highest_deg_list.append(node.GetId())
        elif degree == highest_deg:
            highest_deg_list.append(node.GetId())

    print('Node id(s) with highest degree:', end=' ')
    for node in highest_deg_list[: -1]:
        print(node, end=',')
    print(highest_deg_list[-1])


def degree_distribution_plot(G):
    """
    Saves the degree distribution plot of the subgraph G
    The file is saved in the directory './plots/deg_dist_<subgraph_name>.png'
    """

    snap.PlotOutDegDistr(G, sys.argv[-1], f"Degree Distribution in {sys.argv[-1]}")

    try:
        os.mkdir('./plots')
    except:
        pass

    os.rename(f'outDeg.{sys.argv[-1]}.png', f'./plots/deg_dist_{sys.argv[-1]}.png')
    os.remove(f'outDeg.{sys.argv[-1]}.plt')
    os.remove(f'outDeg.{sys.argv[-1]}.tab')


def print_full_diameter(G):
    """
    Prints full diameter by sampling 10, 100, 1000 nodes in subgraph G
    Also prints mean and variance of the full diameters obtained
    """

    d10 = snap.GetBfsFullDiam(G, 10)
    d100 = snap.GetBfsFullDiam(G, 100)
    d1000 = snap.GetBfsFullDiam(G, 1000)
    array = np.array([d10, d100, d1000])
    mean = round(np.mean(array), 4)
    variance = round(np.var(array), 4)

    print("Approximate full diameter by sampling 10 nodes:", d10)
    print("Approximate full diameter by sampling 100 nodes:", d100)
    print("Approximate full diameter by sampling 1000 nodes:", d1000)
    print(f"Approximate full diameter (mean and variance): {mean},{variance}")


def print_effective_diameter(G):
    """
    Prints the approximate effective diameter by sampling 10, 100, 1000 nodes in subgraph G
    Also prints mean and variance of approximate effective diameters obtained
    """

    d10 = snap.GetBfsEffDiam(G, 10)
    d100 = snap.GetBfsEffDiam(G, 100)
    d1000 = snap.GetBfsEffDiam(G, 1000)
    array = np.array([d10, d100, d1000])
    mean = round(np.mean(array), 4)
    variance = round(np.var(array), 4)

    print("Approximate effective diameter by sampling 10 nodes:", round(d10, 4))
    print("Approximate effective diameter by sampling 100 nodes:", round(d100, 4))
    print("Approximate effective diameter by sampling 1000 nodes:", round(d1000, 4))
    print(f"Approximate effective diameter (mean and variance): {mean},{variance}")


def shortest_path_distribution_plot(G):
    """
    Saves the shortest path distribution plot of the subgraph G
    The file is saved in the directory './plots/shortest_path_<subgraph_name>.png'
    """

    snap.PlotShortPathDistr(G, sys.argv[-1], f"Shortest Path Distribution in {sys.argv[-1]}")

    try:
        os.mkdir('./plots')
    except:
        pass

    os.rename(f'diam.{sys.argv[-1]}.png', f'./plots/shortest_path_{sys.argv[-1]}.png')
    os.remove(f'diam.{sys.argv[-1]}.plt')
    os.remove(f'diam.{sys.argv[-1]}.tab')


def print_components(G):
    """
    Prints the fraction of nodes in the largest component of subgraph G
    Also prints the number of edge bridges and articulation points
    """

    print("Fraction of nodes in largest connected component:", round(snap.GetMxWccSz(G), 4))

    EdgeV = snap.TIntPrV()
    snap.GetEdgeBridges(G, EdgeV)
    print("Number of edge bridges:", EdgeV.Len())

    ArtNIdV = snap.TIntV()
    snap.GetArtPoints(G, ArtNIdV)
    print("Number of articulation points:", ArtNIdV.Len())


def connected_components_plot(G):
    """
    Saves the connected components' size distribution plot of the subgraph G
    The file is saved in the directory './plots/connected_comp_<subgraph_name>.png'
    """

    snap.PlotWccDistr(G, sys.argv[-1], f"Connected Components' Size Distribution in {sys.argv[-1]}")

    try:
        os.mkdir('./plots')
    except:
        pass

    os.rename(f'wcc.{sys.argv[-1]}.png', f'./plots/connected_comp_{sys.argv[-1]}.png')
    os.remove(f'wcc.{sys.argv[-1]}.plt')
    os.remove(f'wcc.{sys.argv[-1]}.tab')


def print_connectivity_clustering(G):
    """
    Prints the average clustering coefficient, number of triads in subgraph G
    Also prints clustering coefficient and number of triads for random nodes
    Also prints the number of edges that participate in at least one triad
    """

    GraphClustCoeff = snap.GetClustCf(G)
    print("Average clustering coefficient:", round(GraphClustCoeff, 4))

    print("Number of triads:", snap.GetTriads(G))

    NId = G.GetRndNId()
    print(f'Clustering coefficient of random node {NId}:', round(snap.GetNodeClustCf(G, NId)))

    NId = G.GetRndNId()
    print(f'Number of triads random node {NId} participates:', snap.GetNodeTriads(G, NId))

    print('Number of edges that participate in at least one triad:', snap.GetTriadEdges(G))


def clustering_coefficient_plot(G):
    """
    Saves the clustering coefficient distribution plot of the subgraph G
    The file is saved in the directory './plots/clustering_coeff_<subgraph_name>.png'
    """

    snap.PlotClustCf(G, sys.argv[-1], f"Clustering Coefficient Distribution in {sys.argv[-1]}")

    try:
        os.mkdir('./plots')
    except:
        pass

    os.rename(f'ccf.{sys.argv[-1]}.png', f'./plots/clustering_coeff_{sys.argv[-1]}.png')
    os.remove(f'ccf.{sys.argv[-1]}.plt')
    os.remove(f'ccf.{sys.argv[-1]}.tab')



if __name__ == "__main__":
    path = f'./subgraphs/{sys.argv[-1]}'
    G = snap.LoadEdgeList(snap.PUNGraph, path, 0, 1, "\t")

    print_nodes_and_edges(G)

    print_degrees(G)

    degree_distribution_plot(G)

    print_full_diameter(G)

    print_effective_diameter(G)

    shortest_path_distribution_plot(G)

    print_components(G)

    connected_components_plot(G)

    print_connectivity_clustering(G)

    clustering_coefficient_plot(G)