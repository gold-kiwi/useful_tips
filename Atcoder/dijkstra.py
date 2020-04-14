import numpy as np
import heapq
import time

def dijkstra(graph,node_count,start_index,end_index):
    INF = np.sum(graph) + 1
    node = [INF] * node_count
    conf = [0] * node_count
    root = [-1] * node_count

    node[start_index] = 0
    while sum(conf) < node_count:
        min_node_dex = -1
        min_node_val = INF
        for i in range(node_count):
            if conf[i] == 0:
                if node[i] < min_node_val:
                    min_node_val = node[i]
                    min_node_dex = i
        conf[min_node_dex] = 1
        for i in range(node_count):
            if conf[i] == 0:
                if g[min_node_dex][i] > 0:
                    new_dist = node[min_node_dex] + g[min_node_dex][i]
                    if new_dist < node[i]:
                        node[i] = new_dist
                        root[i] = min_node_dex

    minimum_root = []
    minimum_root.append(end_index)
    dex = end_index
    while True:
        if dex == start_index:
            break
        minimum_root.append(root[dex])
        dex = root[dex]
    minimum_root.reverse()
    
    return node[end_index] , minimum_root


def dijkstra_heapq(graph,node_count,start_index,end_index):
    INF = np.sum(graph) + 1
    node = [[INF,i] for i in range(node_count)]
    const_node = [[INF,-1] for i in range(node_count)]

    node[start_index][0] = 0
    heapq.heapify(node)

    while len(node) > 0:
        sub_dist,min_node_dex = heapq.heappop(node)
        const_node[min_node_dex][0] = sub_dist

        for i in range(len(node)):
            if g[min_node_dex][node[i][1]] > 0:
                    new_dist = sub_dist + g[min_node_dex][node[i][1]]
                    if new_dist < node[i][0]:
                        node[i][0] = new_dist
                        const_node[node[i][1]][1] = min_node_dex

    minimum_root = []
    minimum_root.append(end_index)
    dex = end_index
    while True:
        if dex == start_index:
            break
        minimum_root.append(const_node[dex][1])
        dex = const_node[dex][1]
    minimum_root.reverse()
    
    return const_node[end_index][0] , minimum_root



node_count = 9
g = np.zeros((node_count,node_count),dtype=np.int64)
s_i = 0
e_i = 8


'''
g[0][1] = 5
g[0][2] = 4
g[1][2] = 2
g[1][4] = 7
g[1][3] = 6
g[2][1] = 2
g[2][4] = 9
g[3][4] = 1
g[4][3] = 2
'''

'''
g[0][1] = 1
g[0][2] = 4
g[0][3] = 5
g[1][0] = 1
g[1][2] = 2 
g[2][0] = 4
g[2][1] = 2
g[2][4] = 6
g[2][5] = 5 
g[3][0] = 5
g[3][5] = 2
g[4][2] = 6
g[4][6] = 2
g[5][2] = 5
g[5][3] = 2
g[5][6] = 3
g[5][7] = 1
g[6][4] = 2
g[6][5] = 3
g[6][7] = 4
g[7][5] = 1
g[7][6] = 4
'''


g[0][1]=1
g[0][2]=2
g[0][3]=3
g[0][4]=5
g[1][0]=1
g[1][2]=3
g[2][1]=3
g[2][0]=2
g[2][3]=2
g[2][5]=3
g[3][0]=3
g[3][2]=2
g[3][5]=1
g[4][0]=5
g[4][5]=4
g[4][6]=4
g[5][2]=3
g[5][3]=1
g[5][4]=4
g[5][8]=5
g[5][7]=7
g[6][4]=4
g[6][8]=7
g[7][5]=7
g[7][8]=2
g[8][7]=2
g[8][5]=5
g[8][6]=7

start_index = 0
end_index = 8
start = time.time()
min_dist,min_root = dijkstra(g,node_count,start_index,end_index)
elapsed_time = time.time() - start
print ("elapsed_time:%1.10f" % elapsed_time)
print(min_dist,min_root)

start = time.time()

min_dist,min_root = dijkstra_heapq(g,node_count,start_index,end_index)
elapsed_time = time.time() - start
print ("elapsed_time:%1.10f" % elapsed_time)

print(min_dist,min_root)


'''
a = [[1,0], [6,1], [8,2], [0,3], [-1,4]]
heapq.heapify(a)  # リストを優先度付きキューへ
print(a)

print(heapq.heappop(a))  # 最小値の取り出し
print(a)

heapq.heappush(a, [-2,5])  # 要素の挿入
print(a)
'''