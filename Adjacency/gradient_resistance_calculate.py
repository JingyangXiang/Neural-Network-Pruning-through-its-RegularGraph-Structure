import os
import time
import pathlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def gradient_infuence_cal(matrix):
    assert matrix.shape[0] == matrix.shape[1]
    nodes = matrix.shape[0]
    # record one nodes need how many steps could influence whole graph
    steps = []
    # record when nodes influeces whole graph, how many nodes it influences totally
    influences = []

    for node in range(nodes):
        temp = 0.
        influence = 0
        index = 0
        node_connect = np.zeros(nodes,dtype=int)
        node_connect[node] = 1
        while influence!=nodes:
            temp += influence
            node_connect = np.sum(matrix[node_connect==1], axis=0).reshape(-1)
            node_connect = np.array(node_connect!=0)
            influence = np.sum(node_connect)
            index = index + 1
            if index > 100:
                break
        steps.append(index)
        influences.append(temp)

    return np.max(steps), np.mean(steps), influences, steps


def write_result_to_csv(path_to_csv,**kwargs):
    results = pathlib.Path(path_to_csv) / f"{nodes}_{int(degree)}_information.csv"

    if not results.exists():
        results.write_text(
            "Date Finished, "
            "Nodes, "
            "Degree, "
            "Path, "
            "Average Shortest Path, "
            "Cluster, "
            "Tranvity, "
            "Diameter, "
            "GradientInfluenceMax, "
            "GradientInfluenceMean, "
            "RelationMean, "
            "Relation\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{nodes}, "
                "{degree}, "
                "{path}, "
                "{average_shortest_path}, "
                "{clustering}, "
                "{tranvity}, "
                "{diameter}, "
                "{step_max}, "
                "{step_mean}, "
                "{relation_mean}, "
                "{relation}\n"
            ).format(now=now, **kwargs)
        )


nodes = 64
layers_num = 15
channels = 64
NodesPath = f'{nodes}'
Nodes_Degrees_path = [os.path.join(NodesPath, degree) for degree in os.listdir(NodesPath)]

path_to_csv = 'matrix'
if not os.path.exists(path_to_csv):
    os.mkdir(path_to_csv)

for degree in Nodes_Degrees_path:
    matrixs = [os.path.join(degree, matrix) for matrix in os.listdir(degree)]
    steps_collecter = []
    channels_collecter = []
    for matrix_path in matrixs:
        if os.path.isfile(matrix_path) and 'npy' in matrix_path:
            matrix = np.load(matrix_path)

            node = nodes
            degree = np.sum(matrix)//node
            graph = nx.from_numpy_matrix(matrix)
            average_shortest_path = nx.average_shortest_path_length(graph)
            clustering = nx.average_clustering(graph)
            tranvity = nx.transitivity(graph)
            diameter = nx.diameter(graph)

            step_max, step_mean, influences, steps = gradient_infuence_cal(matrix)

            relation = []
            channels_collect = []
            for index, (influence, collect) in enumerate(zip(influences, steps)):
                relation.append(f'{collect}:{(layers_num - collect)*channels + influence}')
                channels_collect.append((layers_num - collect)*channels + influence)

            write_result_to_csv(
                path_to_csv=path_to_csv,
                path=matrix_path,
                clustering=clustering,
                tranvity=tranvity,
                average_shortest_path=average_shortest_path,
                degree=int(np.sum(matrix)//matrix.shape[0]),
                nodes=matrix.shape[0],
                step_max=step_max,
                step_mean=step_mean,
                diameter=diameter,
                relation_mean=np.mean(channels_collect),
                relation=relation
            )
            steps_collecter.extend(steps)
            channels_collecter.extend(channels_collect)

    plt.scatter(steps_collecter, channels_collecter)
    plt.xlabel('steps')
    plt.ylabel('channels')
    plt.savefig(f'matrix//{nodes}_{int(degree)}.pdf')
    plt.close()
