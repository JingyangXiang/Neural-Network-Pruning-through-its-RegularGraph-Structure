import networkx as nx
import matplotlib.pyplot as plt
import random
from random import choice
import numpy as np
import tqdm
from graph.build_graph import regular, first_regular_graph
import os
from args import args

def GraphGenerator(degree,nodes,matrix=None):
    # 根据指定的图生成器生成所需要的邻接矩阵
    # 建议初始图给定
    if matrix is None:
        # 如果给的matrix是None,就表示模型是第一次搜索,那么肯定是需要先生成一个可以用的
        # 第二次就来就直接优化了
        # args.matrix是全局变量,讲道理来说幅值一次就全局改变了
        print('===>Fist generate the matrix')
        matrix = regular(degree, nodes)
    # 优化平均最短路径
    matrix = PathOptimizer(matrix)
    return matrix



def compute_average_degree(matrix): 
    # 在此认为模型的最大度值是等于节点个数的
    ave_degree = np.sum(matrix)/np.square(matrix.shape[0])
    return ave_degree


def judge(source1,source2,target1,target2):
    # 判断一下四个节点是不相同的
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
    # interval_dowm是搜索间隙,如果是单步搜索就转图的话,为了控制节点的分布,要优化超过interval_dowm之后才可以return
    # 根据矩阵生成图
    interval_down = args.interval_down
    interval_up = args.interval_up
    nodes = matrix.shape[0]
    assert matrix.shape[0]==matrix.shape[1] and len(matrix.shape)==2
    graph = nx.from_numpy_matrix(matrix)
    # 记录开始的平均最短路径   
    path_begin = nx.average_shortest_path_length(graph)
    # tqdm的进度条,比较好看
    pbar = range(int(nodes *(nodes-1)))
    # 利用for循环进行搜索
    # 记录开始的平均最短路径
    path_before = nx.average_shortest_path_length(graph)

    for _ in tqdm.tqdm(enumerate(pbar), ascii=True, total=len(pbar)):

        # 选择四个节点并且进行条件判断
        source1 = choice(np.array(graph.nodes()))
        source2 = choice(np.array(graph.nodes()))
        target1 = choice(np.array(graph[source1]))
        target2 = choice(np.array(graph[source2]))
        # 判断四个节点是否有相同的
        state_judge = judge(source1,source2,target1,target2)
        # 如果有节点是相同的就跳过
        if not state_judge:
            continue
        # 如果之前的起点和现在的终点已经有边了,那么跳过
        if graph.has_edge(source1,target2) or graph.has_edge(source2,target1):
            continue
        # 保存之前的那一份模型
        former_graph = graph.copy()
        # 对网络进行删边和重新连边
        graph.remove_edges_from([[source1,target1],[source2,target2]])
        graph.add_edges_from([[source1,target2],[source2,target1]])
        # 如果重连边之后网络还是连通的,就比较连边之后的结果和连边之前的结果
        if nx.is_connected(graph):
            path_after = nx.average_shortest_path_length(graph)
            if args.search_direction.lower() == 'random':
                if not interval_down < np.abs(path_after - path_before) < interval_up:
                    # print(path_after)
                    graph = former_graph
                else:
                    break
                # 如果是随机的话,就随机变换边,然后一口气生成
            elif args.search_direction.lower() == 'min':
                # 因为路径要往小了搜,如果之后的路径比之前的路径要大,那么就返回回去,继续下一步搜索
                # print(f'path before:{path_before}  path after: {path_after}')
                if path_after > path_before:
                    graph = former_graph
                elif (path_before - path_after)>args.interval_up:
                    graph = former_graph
                elif args.interval_down<np.abs(path_after-path_before)<args.interval_up:
                    if not args.extreme:
                        return np.array(nx.adjacency_matrix(graph).todense())
            elif args.search_direction.lower() == 'max':
                # 因为路径要往大了搜,如果之后的路径比之前的路径要小,那么就返回回去,继续下一步搜索
                if path_after < path_before:
                    graph = former_graph
                elif (path_after - path_before)>args.interval_up:
                    graph = former_graph
                elif args.interval_down<np.abs(path_after-path_before)<args.interval_up:
                    if not args.extreme:
                        return np.array(nx.adjacency_matrix(graph).todense())
        else:
            # 如果重新连边之后图不是连通的把删边和连边回退回去
            graph.add_edges_from([[source1,target1],[source2,target2]])
            graph.remove_edges_from([[source1,target2],[source2,target1]])
    # for循环结束之后返回邻接矩阵
    return np.array(nx.adjacency_matrix(graph).todense())














