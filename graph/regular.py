import numpy as np
import networkx as nx


def regular(degree,nodes):
    while True:
        try:
            print('==>begin to generate Regular Graph')
            sequence = np.ones(nodes)*degree
            RG = nx.random_degree_sequence_graph(sequence,tries=1000000)            
            if nx.is_connected(RG):
                matrix = np.array(nx.adjacency_matrix(RG).todense())
                print('==>success to generate Regular Graph')
                return matrix
        except:
            print('==>loss to generate Regular Graph')


def first_regular_graph(degree, nodes):
    assert degree%2==0 and nodes%2==0
    matrix = np.zeros([nodes,nodes])
    for line in range(matrix.shape[0]):
        for D in range(-degree//2,degree//2+1):
            matrix[line, (line+D)%nodes] = 1
    matrix = matrix - np.diag(np.ones(nodes))
    return matrix


def last_regular_graph(degree, nodes):
    assert degree%2==1 and nodes%2==0
    degree = nodes - degree - 1
    matrix = np.zeros([nodes,nodes])
    for line in range(matrix.shape[0]):
        for D in range(-degree//2,degree//2+1):
            matrix[line, (line+D)%nodes] = 1
    matrix = matrix - np.diag(np.ones(nodes))
    matrix = np.ones([nodes,nodes]) - matrix - np.diag(np.ones(nodes))
    return matrix
