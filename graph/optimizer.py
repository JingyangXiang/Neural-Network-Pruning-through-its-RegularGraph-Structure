import networkx as nx
from random import choice
import numpy as np
import tqdm

from args import args
from graph.graphbuilder import regular


def GraphGenerator(degree,nodes,matrix=None):
    if matrix is None:
        print('===>Fist generate the matrix')
        matrix = regular(degree, nodes)

    matrix = PathOptimizer(matrix)
    return matrix


def compute_average_degree(matrix):
    ave_degree = np.sum(matrix)/np.square(matrix.shape[0])

    return ave_degree


def judge(source1,source2,target1,target2):
    # Four nodes should be different
    if source1 == source2:
        return False
    if target1 == target2:
        return False
    if source1 == target2:
        return False
    if source2 == target1:
        return False
    return True


def PathOptimizer(matrix):

    interval_down = args.interval_down
    interval_up = args.interval_up
    nodes = matrix.shape[0]
    assert matrix.shape[0]==matrix.shape[1] and len(matrix.shape)==2
    graph = nx.from_numpy_matrix(matrix)

    pbar = range(int(nodes *(nodes-1)))
    path_before = nx.average_shortest_path_length(graph)

    for _ in tqdm.tqdm(enumerate(pbar), ascii=True, total=len(pbar)):
        # choice four nodes
        source1 = choice(np.array(graph.nodes()))
        source2 = choice(np.array(graph.nodes()))
        target1 = choice(np.array(graph[source1]))
        target2 = choice(np.array(graph[source2]))

        # make sure four nodes is different
        state_judge = judge(source1,source2,target1,target2)
        # when a pair of node is same, go to next step
        if not state_judge:
            continue

        if graph.has_edge(source1,target2) or graph.has_edge(source2,target1):
            continue

        former_graph = graph.copy()
        # rewire n1<->n2, n3<->n4 to n1<->n4, n2<->n3
        graph.remove_edges_from([[source1,target1],[source2,target2]])
        graph.add_edges_from([[source1,target2],[source2,target1]])

        # graph should keep connected
        if nx.is_connected(graph):
            path_after = nx.average_shortest_path_length(graph)
            if args.search_direction.lower() == 'random':
                if not interval_down < np.abs(path_after - path_before) < interval_up:
                    graph = former_graph
                else:
                    break
            elif args.search_direction.lower() == 'min':
                if path_after > path_before:
                    graph = former_graph
                elif (path_before - path_after)>args.interval_up:
                    graph = former_graph
                elif args.interval_down<np.abs(path_after-path_before)<args.interval_up:
                    if not args.extreme:
                        return np.array(nx.adjacency_matrix(graph).todense())
            elif args.search_direction.lower() == 'max':
                if path_after < path_before:
                    graph = former_graph
                elif (path_after - path_before)>args.interval_up:
                    graph = former_graph
                elif args.interval_down<np.abs(path_after-path_before)<args.interval_up:
                    if not args.extreme:
                        return np.array(nx.adjacency_matrix(graph).todense())
        else:
            # if graph(n+1) is not connected, return graph(n)
            graph.add_edges_from([[source1,target1],[source2,target2]])
            graph.remove_edges_from([[source1,target2],[source2,target1]])

    return np.array(nx.adjacency_matrix(graph).todense())
