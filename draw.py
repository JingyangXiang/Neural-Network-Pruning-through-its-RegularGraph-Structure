import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os



def draw_graph(path):
    matrix = np.load(path)

    G = nx.from_numpy_matrix(matrix)
    # print(f'==>{nx.average_shortest_path_length(G)}')
    nx.draw(G,
            pos = nx.shell_layout(G), # pos 指的是布局,主要有spring_layout,random_layout,circle_layout,shell_layout
            node_color = 'b',   # node_color指节点颜色,有rbykw,同理edge_color
            edge_color = 'r',
            with_labels = False,  # with_labels指节点是否显示名字
            font_size =18,  # font_size表示字体大小,font_color表示字的颜色
            node_size =25)  # font_size表示字体大小,font_color表示字的颜色

    plt.savefig(path.replace('npy', 'png'))
    # nx.write_gexf(G, 'network.gexf')  # gexf格式文件可以导入gephi中进行分析
    # plt.show()
    plt.close()


path = 'Adjacency'
dirs = os.listdir(path)
for index in range(len(dirs)):
    dirs[index] = os.path.join(path, dirs[index])
for dir in dirs:
    dir1s = os.listdir(dir)
    for index in range(len(dir1s)):
        dir1s[index] = os.path.join(dir,dir1s[index])
    print(dir1s)
    for dir1 in dir1s:
        dir2s = os.listdir(dir1)
        for index in range(len(dir2s)):
            dir2s[index] = os.path.join(dir1, dir2s[index])
        print(dir2s)
        for dir2 in dir2s:
            if os.path.isfile(dir2) and 'npy' in dir2:
                draw_graph(dir2)

    # ['10', '12', '14', '16', '18', '20', '22', '24', '26', '28', '30', '4', '6', '8']


