import snap

def generate_graph(edge_list_file):
    """
    returns an undirected graph generated using SNAP from edge list text file
    """

    # open edge list text file and read its contents
    file = open(edge_list_file, 'r')
    lines = file.readlines()
    file.close()

    # add nodes and edges
    G = snap.TUNGraph().New()
    for line in lines:
        node1, node2 = [int(a) for a in line.strip().split()]
        try:
            G.AddNode(node1)
        except:
            pass
        try:
            G.AddNode(node2)
        except:
            pass
        G.AddEdge(node1, node2)

    # return the created graph
    return G

def compute_closeness_centralities(G):
    """
    returns sorted closeness centrality values for all nodes in the graph G
    """

    # compute closeness centrality values using SNAP
    snap_closeness = []
    for node in snap.Nodes(G):
        value = snap.GetClosenessCentr(G, node.GetId())
        snap_closeness.append((node.GetId(), value))

    # sort the closeness values in reverse order
    snap_closeness.sort(reverse=True, key=lambda x: x[1])

    # return the closeness centrality values
    return snap_closeness

def compute_betweenness_centralities(G):
    """
    returns sorted betweenness centrality values for all nodes in the graph G
    uses node fraction of 0.8 to speed up computation of betweenness values
    """

    # compute betweenness centrality values using SNAP
    num_nodes = G.GetNodes()
    snap_betweenness = []
    Nodes = snap.TIntFltH()
    Edges = snap.TIntPrFltH()
    snap.GetBetweennessCentr(G, Nodes, Edges, 0.8)
    for node in Nodes:
        snap_betweenness.append((node, 2 * Nodes[node] / ((num_nodes - 1) * (num_nodes - 2))))

    # sort the betweenness values in descending order
    snap_betweenness.sort(reverse=True, key=lambda x: x[1])

    # return the betweenness centrality values
    return snap_betweenness

def compute_pageranks(G):
    """
    returns sorted closeness centrality values for all nodes in the graph G
    epsilon value is set to 1e-8 to check for convergence
    damping factor, alpha, is 0.8
    max iterations = 100
    """

    # compute PageRank values using SNAP
    snap_pagerank = []
    PRankH = snap.TIntFltH()
    snap.GetPageRank(G, PRankH, 0.8, 1e-8)
    for node in PRankH:
        snap_pagerank.append((node, PRankH[node]))

    # sort the PageRank values in descending order
    snap_pagerank.sort(reverse=True, key=lambda x: x[1])

    # return PageRank values
    return snap_pagerank

def find_overlaps(G, snap_closeness, snap_betweenness, snap_pagerank):
    """
    Prints the number of overlaps in top 100 nodes for all centrality values

    """
    # open centrality files
    closeness = open('centralities/closeness.txt', 'r')
    betweenness = open('centralities/betweenness.txt', 'r')
    pagerank = open('centralities/pagerank.txt', 'r')

    # create empty sets
    nodes_close = set()
    nodes_between = set()
    nodes_pr = set()
    snap_nodes_close = set()
    snap_nodes_between = set()
    snap_nodes_pr = set()

    # add top 100 nodes to sets for all centralities (with or without snap)
    for i in range(100):
        nodes_close.add(int(closeness.readline().strip().split()[0]))
        nodes_between.add(int(betweenness.readline().strip().split()[0]))
        nodes_pr.add(int(pagerank.readline().strip().split()[0]))
        snap_nodes_close.add(snap_closeness[i][0])
        snap_nodes_between.add(snap_betweenness[i][0])
        snap_nodes_pr.add(snap_pagerank[i][0])

    # print number of overlaps for each centrality
    print(f'#overlaps for Closeness Centrality: {len(nodes_close.intersection(snap_nodes_close))}')
    print(f'#overlaps for Betweenness Centrality: {len(nodes_between.intersection(snap_nodes_between))}')
    print(f'#overlaps for PageRank Centrality: {len(nodes_pr.intersection(snap_nodes_pr))}')

    # close the files
    closeness.close()
    betweenness.close()
    pagerank.close()

if __name__ == "__main__":
    edge_list_file = 'facebook_combined.txt'
    G = generate_graph(edge_list_file)

    snap_closeness = compute_closeness_centralities(G)

    snap_betweenness = compute_betweenness_centralities(G)

    snap_pagerank = compute_pageranks(G)

    find_overlaps(G, snap_closeness, snap_betweenness, snap_pagerank)
