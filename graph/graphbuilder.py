import numpy as np
import networkx as nx


# get random regular graph
def regular(degree,nodes):
    while True:
        try:
            print('==>begin to generate Regular Graph')
            sequence = np.ones(nodes)*degree
            # the degree distribution of graph is equal to sequence
            RG = nx.random_degree_sequence_graph(sequence,tries=1000000)            
            if nx.is_connected(RG):
                matrix = np.array(nx.adjacency_matrix(RG).todense())
                print('==>success to generate Regular Graph')
                return matrix
        except:
            print('==>loss to generate Regular Graph')


# Get regualr graph by hand
# Each node is connected to degree//2 nodes on either side of it
def first_regular_graph(degree, nodes):

    assert degree%2==0 and nodes%2==0
    matrix = np.zeros([nodes,nodes])
    for line in range(matrix.shape[0]):
        for D in range(-degree//2,degree//2+1):
            matrix[line, (line+D)%nodes] = 1
    matrix = matrix - np.diag(np.ones(nodes))

    return matrix
