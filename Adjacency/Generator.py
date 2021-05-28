import networkx as nx
import matplotlib.pyplot as plt
import time
from functools import cmp_to_key
import numpy as np
from graph.regular import  first_regular_graph,last_regular_graph
from graph.optimizer import PathOptimizer
import os
from args import args
import pathlib





def GraphGenerator(degree,nodes,matrix=None,**kwargs):
    # 根据指定的图生成器生成所需要的邻接矩阵
    if matrix is None:
        # 如果给的matrix是None,就表示模型是第一次搜索,那么肯定是需要先生成一个可以用的
        # 第二次就来就直接优化了
        # args.matrix是全局变量,讲道理来说幅值一次就全局改变了
        print('===>Fist generate the matrix')
        matrix = first_regular_graph(degree, nodes)
    # 优化平均最短路径
    matrix = PathOptimizer(matrix)
    return matrix

def get_path(path):
    global i
    i = 0
    while os.path.exists(os.path.join(path, f'matrix-{i}.npy')):
        i = i + 1
    return os.path.join(path, f'matrix-{i}')

def compare(x1,x2):
    assert os.path.isfile(x1)
    assert os.path.isfile(x2)
    m1 = np.load(x1)
    m2 = np.load(x2)
    g1 = nx.from_numpy_matrix(m1)
    g2 = nx.from_numpy_matrix(m2)
    p1 = nx.average_shortest_path_length(g1)
    p2 = nx.average_shortest_path_length(g2)
    if p1>p2:
        return 1
    else:
        return -1

def write_result_to_csv(path_to_csv,**kwargs):
    print(kwargs)
    results = pathlib.Path(path_to_csv) / f"{args.nodes}_{degree}_information.csv"

    if not results.exists():
        results.write_text(
            "Date Finished, "
            "Nodes, "
            "Degree, "
            "Dir, "
            "Path, "
            "Cluster, "
            "Tranvity\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{nodes}, "
                "{degree}, "
                "{dir}, "
                "{path}, "
                "{clustering}, "
                "{tranvity}\n"
            ).format(now=now, **kwargs)
        )




MIN = eval(input('Input the min degree: '))
MAX = eval(input('Input the MAX degree: '))

num_split = 6
args.interval_up = 0.2
args.interval_down = 0.05
for degree in range(MIN,MAX+1,2):
    args.iter_num = max(270 - int(2.5 * num_split * (degree - 4)), 120)
    files = os.path.join('Adjacency', f'{args.nodes}/{degree}')
    if not os.path.exists(files):
        os.makedirs(files)
    if len(os.listdir(files))<args.iter_num:
        for split in range(num_split):
            # 生成的图分度值是奇数还是偶数
            if degree%2==0:
                matrix = first_regular_graph(degree,args.nodes)
                args.search_direction = 'min'
            elif degree%2==1:
                matrix = last_regular_graph(degree, args.nodes)
                args.search_direction = 'max'
            # 生成以后先朝着变大的方向搜索100次图,然后再往小了搜索
            # args.search_direction = 'max'
            # args.extreme = True
            # graph = nx.from_numpy_matrix(matrix)
            # print(f'max search: {nx.average_shortest_path_length(graph)}')
            # matrix = GraphGenerator(degree, args.nodes, matrix)
            # 按照一定的间隔往小了搜索,为了形成比较均匀的分布
            for index in range(args.iter_num//num_split):
                np.save(get_path(files), matrix)
                graph = nx.from_numpy_matrix(matrix)
                print(f'{index}: {nx.average_shortest_path_length(graph)}')
                matrix = GraphGenerator(degree, args.nodes, matrix)

    # 排序-->重命名-->展示分布
    dirs = os.listdir(files)
    if not os.path.isfile(os.path.join(files, f'M0.npy')):
        for i in range(len(dirs)):
            dirs[i] = os.path.join(files, dirs[i])
        dirs_sorted = sorted(dirs,key=cmp_to_key(compare))
        for index, dir in enumerate(dirs_sorted):
            os.rename(dir, os.path.join(files, f'M{index}.npy'))

    dirs = os.listdir(files)

    for i in range(len(dirs)):
        if '.npy' in dirs[i]:
            dirs[i] = os.path.join(files, dirs[i])

    dirs_sorted = sorted(dirs,key=cmp_to_key(compare))
    for dir in dirs_sorted:
        # 记录属性
        matrix = np.load(f'{dir}')
        graph = nx.from_numpy_matrix(matrix)
        path = nx.average_shortest_path_length(graph)
        path = nx.average_shortest_path_length(graph)
        clustering = nx.average_clustering(graph)
        tranvity = nx.transitivity(graph)
        write_result_to_csv(
            path_to_csv=files,
            dir=dir,
            path=path,
            clustering=clustering,
            tranvity=tranvity,
            degree=np.sum(matrix)//matrix.shape[0],
            nodes=matrix.shape[0]
        )











