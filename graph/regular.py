import networkx as nx
import matplotlib.pyplot as plt
import random
from random import choice
import numpy as np
import tqdm

# 生成具有脉冲分布的图
def regular(degree,nodes):
    while True:
        try:
            print('==>begin to generate Regular Graph')
            sequence = np.ones(nodes)*degree
            # 确保度值平均度值是neighbors
            RG = nx.random_degree_sequence_graph(sequence,tries=1000000)            
            if nx.is_connected(RG):
                matrix = np.array(nx.adjacency_matrix(RG).todense())
                print('==>success to generate Regular Graph')
                return matrix
        except:
            print('==>loss to generate Regular Graph')


# 生成具有脉冲分布的图,手动构建
def first_regular_graph(degree, nodes):
    # 必须保证都是偶数,不然没法构建这种图
    assert degree%2==0 and nodes%2==0
    matrix = np.zeros([nodes,nodes])
    for line in range(matrix.shape[0]):
        for D in range(-degree//2,degree//2+1):
            matrix[line, (line+D)%nodes] = 1
    matrix = matrix - np.diag(np.ones(nodes))
    return matrix
def last_regular_graph(degree, nodes):
    # 必须保证都是偶数,不然没法构建这种图
    assert degree%2==1 and nodes%2==0
    degree = nodes - degree - 1
    matrix = np.zeros([nodes,nodes])
    for line in range(matrix.shape[0]):
        for D in range(-degree//2,degree//2+1):
            matrix[line, (line+D)%nodes] = 1
    matrix = matrix - np.diag(np.ones(nodes))
    matrix = np.ones([nodes,nodes]) - matrix - np.diag(np.ones(nodes))
    return matrix
# matrix = last_regular_graph(3,8)
# print(np.sum(matrix,axis=-1))
# print(matrix)