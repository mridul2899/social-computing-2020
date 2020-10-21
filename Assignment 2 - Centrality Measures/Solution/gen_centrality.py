import os

def create_centralities_directory():
    """
    creates a directory for storing centralities text files if not present
    """

    try:
        os.mkdir('centralities')
    except:
        pass


def generate_graph(edge_list_file):
    """
    creates list of nodes, adjacency list for from edge list text file
    returns the total number of nodes, nodes list and adjacency list
    """

    # open and read contents of text file, close the file
    file = open(edge_list_file, 'r')
    lines = file.readlines()
    file.close()

    # create a list of nodes
    nodes = set()
    for line in lines:
        node1, node2 = [int(a) for a in line.strip().split()]
        nodes.add(node1)
        nodes.add(node2)
    nodes = list(nodes)
    nodes.sort()
    num_nodes = len(nodes)

    # create adjacency list
    adjacency_list = [[] for node in nodes]
    for line in lines:
        node1, node2 = [int(a) for a in line.strip().split()]
        adjacency_list[node1].append(node2)
        adjacency_list[node2].append(node1)

    # return number of nodes, list of nodes and adjacency list
    return num_nodes, nodes, adjacency_list

def compute_closeness_centralities(num_nodes, nodes, adjacency_list):
    """
    computes and stores closeness centrality values for all nodes in the graph
    """

    # Compute sum of shortest distances to and from all nodes using BFS
    distance_sum = [0 for node in nodes]
    for node in nodes:
        found = [0 for i in range(num_nodes)]
        found[node] = 1
        to_visit_nodes = []
        to_visit_nodes.append((node, 0))
        while len(to_visit_nodes) != 0:
            current_node, distance = to_visit_nodes.pop(0)
            for neighbor in adjacency_list[current_node]:
                if found[neighbor] == 0:
                    to_visit_nodes.append((neighbor, distance + 1))
                    distance_sum[node] += (distance + 1)
                    found[neighbor] = 1

    # Compute closeness centrality values
    closeness_centralities = []
    for node, value in enumerate(distance_sum):
        closeness_centralities.append((node, value))
    closeness_centralities.sort(key=lambda x: x[1])

    # Save the closeness centrality values in a text file
    closeness = open('centralities/closeness.txt', 'w')
    for (node, value) in closeness_centralities:
        if value != 0:
            closeness.write(f'{node} {round((num_nodes - 1) / value, 6)}\n')
        else:
            closeness.write(f'{node} {0}\n')
    closeness.close()

def compute_betweenness_centralities(num_nodes, nodes, adjacency_list):
    """
    computes and stores betweenness centrality values for all nodes in the graph
    uses O(V.E) Brandes' Algorithm for undirected graphs to compute values
    """

    # compute betweenness centralities using Brandes' Algorithm
    betweenness_centralities = [0 for node in nodes]
    for node in nodes:
        stack = []
        paths = [[] for n in nodes]
        number_of_paths = [0 for n in nodes]
        number_of_paths[node] = 1
        distances = [-1 for n in nodes]
        distances[node] = 0
        queue = []

        queue.append(node)
        while len(queue) > 0:
            vertex = queue.pop(0)
            stack.append(vertex)
            for node2 in adjacency_list[vertex]:
                if distances[node2] < 0:
                    queue.append(node2)
                    distances[node2] = distances[vertex] + 1
                if distances[node2] == distances[vertex] + 1:
                    number_of_paths[node2] += number_of_paths[vertex]
                    paths[node2].append(vertex)

        values = [0 for n in nodes]
        while len(stack) > 0:
            node2 = stack.pop()
            for vertex in paths[node2]:
                values[vertex] += (number_of_paths[vertex] / number_of_paths[node2]) * (1 + values[node2])
            if node2 != node:
                betweenness_centralities[node2] += values[node2]

    # normalize and sort betweenness centrality values
    betweenness_centralities_nodes = []
    for i in range(len(betweenness_centralities)):
        betweenness_centralities[i] /= ((num_nodes - 1) * (num_nodes - 2))
        betweenness_centralities_nodes.append((i, betweenness_centralities[i]))
    betweenness_centralities_nodes.sort(reverse=True, key=lambda x: x[1])

    # store betweenness centralities values in a text file
    betweenness = open('centralities/betweenness.txt', 'w')
    for node, value in betweenness_centralities_nodes:
        betweenness.write(f'{node} ')
        betweenness.write('%.6f' % round(value, 6))
        betweenness.write('\n')
    betweenness.close()

def compute_biased_pageranks(nodes, adjacency_list):
    """
    computes and stores biased pagerank values for all nodes in the graph
    biasing is done using a subset of all nodes with IDs divisible by 4
    uses standard power-iteration method for computation for 100 epochs max
    """

    # compute number of nodes divisible by 4
    div_4 = 0
    for node in nodes:
        if node % 4 == 0:
            div_4 += 1

    # initialise biased vectors and variables
    d = [(1 / div_4) if node % 4 == 0 else 0 for node in nodes]
    page_ranks = d[:]
    new_page_ranks = [0 for i in page_ranks]
    alpha = 0.8

    # reiterate for a maximum of 100 epochs for PageRank computation
    for i in range(100):
        # compute new PageRanks
        for node in nodes:
            t = 0
            for node2 in adjacency_list[node]:
                t = t + page_ranks[node2] / len(adjacency_list[node2])
            new_page_ranks[node] = alpha * t + (1 - alpha) * d[node]

        # L1 normalise PageRanks
        sum_l1 = sum(new_page_ranks)
        for j in range(len(new_page_ranks)):
            new_page_ranks[j] /= sum_l1

        # check convergence with epsilon = 1e-8, break loop if converged
        loop_again = False
        epsilon=1e-8
        for j in range(len(page_ranks)):
            if abs(page_ranks[j] - new_page_ranks[j]) > epsilon:
                loop_again = True
                break
        if not loop_again:
            # print(i + 1, "iterations")
            break

        # update PageRanks
        page_ranks = new_page_ranks[:]

    # sort by PageRank values
    for node, rank in enumerate(page_ranks):
        page_ranks[node] = (node, rank)
    page_ranks.sort(reverse=True, key=lambda x: x[1])

    # write the PageRank values to a text file
    pagerank = open('centralities/pagerank.txt', 'w')
    for node, rank in page_ranks:
        pagerank.write(f'{node} ')
        pagerank.write('%.6f\n' % round(rank, 6))
    pagerank.close()

if __name__ == "__main__":
    create_centralities_directory()

    edge_list_file = 'facebook_combined.txt'
    num_nodes, nodes, adjacency_list = generate_graph(edge_list_file)

    compute_closeness_centralities(num_nodes, nodes, adjacency_list)

    compute_betweenness_centralities(num_nodes, nodes, adjacency_list)

    compute_biased_pageranks(nodes, adjacency_list)
