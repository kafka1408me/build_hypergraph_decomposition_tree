import hypernetx as hnx
import matplotlib.pyplot as plt

empty_set = set()

count_nodes = 16

hyperedges2 = {1:['v1', 'v2', 'v3', 'v4', 'v5', 'v6'],
              2:['v6', 'v7', 'v8'],
              3:['v5', 'v7', 'v9'],
              4:['v8', 'v9', 'v10', 'v11', 'v12', 'v13'],
              5:['v13', 'v14', 'v15'],
              6:['v15', 'v16'],
              7:['v14', 'v16']}

hyperedges = {edge - 1:[int(node_str[1:]) - 1 for node_str in node] for edge, node in hyperedges2.items()}

print('starting hyperedges:\n', hyperedges)

def createKoenigGraphFromHyperedges(my_hyperedges):
#    koenig_graph = {i:[] for i in range(count_nodes)}
    koenig = dict()
    for edge, _nodes in my_hyperedges.items():
        for _node in _nodes:
            if not _node in koenig:
                koenig[_node] = []
            koenig[_node].append(edge)
    return koenig

#accounted_nodes = [0 for _ in range(count_nodes)]

koenig_graph = createKoenigGraphFromHyperedges(hyperedges)


hypergraph = hnx.Hypergraph(hyperedges2)

fig, ax = plt.subplots(1, 1)

hnx.draw(
            hypergraph,
            ax=ax,
        #    edge_labels=edge_labels,
            node_labels_kwargs={
                'fontsize': 8
            }
        )

print('starting koening_graph:\n', koenig_graph)

def GrahamAlgorithm(koenig_graph, hyperedges):
    print("Graham start !!!")
    count_deleted = 999
    while count_deleted:
        count_deleted = 0
        deleted_nodes = []
        # удаление висячих вершин
        for node, edges_for_node in koenig_graph.items():
            if len(edges_for_node) < 2:
                deleted_nodes.append(node)
                count_deleted += 1
        for del_node in deleted_nodes:
            for edge in koenig_graph[del_node]:
                hyperedges[edge].remove(del_node)
            koenig_graph.pop(del_node)
        print(deleted_nodes, 'were deleted')
        # удаление вложенных ребер
        edges = list(hyperedges.keys())
        count_edges = len(edges)
        del_edges = []
        print("koening now:", koenig_graph)
        print("hyperedges now:", hyperedges)
        for edge in edges[:-1]:
            if edge in del_edges:
                continue
            cur_edge = set(hyperedges[edge])
            if cur_edge == empty_set:
                del_edges.append(edge)
                print(cur_edge, 'is empty!')
                continue
            for _edge in range(edge + 1, count_edges):
                other_edge = hyperedges[_edge]
                if cur_edge.issubset(other_edge):
                    print(f'1)edge{edge} del')
                    del_edges.append(edge)
                    break
                elif set(other_edge).issubset(cur_edge):
                    print(f'2)edge{_edge} del')
                    del_edges.append(_edge)
        count_deleted += len(del_edges)
        for del_edge in del_edges:
            for node in hyperedges[del_edge]:
                koenig_graph[node].remove(del_edge)
            hyperedges.pop(del_edge)
        print("koening now:", koenig_graph)
        print("hyperedges now:", hyperedges)
        print('*************************')


print('------ new koening_graph:\n', koenig_graph)
print('------ new hyperedges:\n', hyperedges)

GrahamAlgorithm(koenig_graph=koenig_graph, hyperedges=hyperedges)

edges = hyperedges.keys()


def findPathes(stack_way, checked_set, finding_edges, passed_edges = {}):
  #  print(f'findPathes. stack: {stack_way}')
    edge = stack_way[-1]
    nodes_for_edge_set = set(hyperedges[edge]) - checked_set
    for node_for_edge in nodes_for_edge_set:
        neighboring_edges = [_edge for _edge in koenig_graph[node_for_edge] if not (_edge in passed_edges) and not(_edge in stack_way) and _edge != edge]
        for neighbour_edge in neighboring_edges:
            if neighbour_edge in finding_edges:
                finding_edges.remove(neighbour_edge)
                if not finding_edges:
                    return
            stack_way.append(neighbour_edge)
            findPathes(stack_way, checked_set, finding_edges, passed_edges)
            if not finding_edges:
                return
    passed_edges.add(edge)
    stack_way.pop()

accounted_intersections = set()   # учтенные множества соединений ребер
articulation_set        = set()    # найденные точки сочленения

# удаление множеств сочленения
for edge in edges:
    nodes_for_edge = hyperedges[edge]
    nodes_for_edge_set = set(nodes_for_edge)
    for node_for_edge in nodes_for_edge:
        neighboring_edges = [_edge for _edge in koenig_graph[node_for_edge] if _edge > edge]
        for neighbour_edge in neighboring_edges:
            intersection_set = nodes_for_edge_set & set(hyperedges[neighbour_edge])
            if intersection_set in accounted_intersections:
                continue
            accounted_intersections |= intersection_set
         #   print(f'@@@@@@@@@ intersection between edges {edge} and {neighbour_edge}', intersection_set)
            finding_edges = set(koenig_graph[node_for_edge]) - {edge}
            passed_edges = set()
            findPathes(stack_way=[edge], checked_set=intersection_set, finding_edges=finding_edges, passed_edges=passed_edges)
          #  print(f'finding edges: {finding_edges}\npassed_edges: {passed_edges}')
            if finding_edges != empty_set:
                print(f'Найдено множество сочленения: {intersection_set}')
                articulation_set |= intersection_set
            pass

print(f'Множество сочленений: {articulation_set}')

# Если есть(!) точки сочленения, то
for point_articulation in articulation_set:
    for edge in koenig_graph[point_articulation]:
        hyperedges[edge].remove(point_articulation)
    del koenig_graph[point_articulation]


print('------ после удаления сочленений koening_graph:\n', koenig_graph)
print('------ после удаления сочленений hyperedges:\n', hyperedges)

# выделение блоков

def findPathes(stack_way, passed_edges):
  #  print(f'findPathes. stack: {stack_way}')
    edge = stack_way[-1]
    nodes_for_edge_set = set(hyperedges[edge])
    for node_for_edge in nodes_for_edge_set:
        neighboring_edges = [_edge for _edge in koenig_graph[node_for_edge] if not (_edge in passed_edges) and not(_edge in stack_way) and _edge != edge]
        for neighbour_edge in neighboring_edges:
            stack_way.append(neighbour_edge)
            findPathes(stack_way, passed_edges)
    passed_edges.add(edge)
    stack_way.pop()

my_edges = set(hyperedges.keys())
blocks = []

while my_edges != empty_set:
    block = set()
    findPathes([my_edges.pop()], block)
    blocks.append(block)
    my_edges -= block

print(f'Выделенные блоки: {blocks}')

blocks_koenig_graph = []
blocks_hyperedges   = []

# Применение алгоритма Грэхема в каждому блоку
for block in blocks:
    block_hyperedges = dict()
    for edge in block:
        block_hyperedges[edge] = hyperedges[edge]
    block_koening = createKoenigGraphFromHyperedges(my_hyperedges=block_hyperedges)
    GrahamAlgorithm(koenig_graph=block_koening, hyperedges=block_hyperedges)
    blocks_hyperedges.append(block_hyperedges)
    blocks_koenig_graph.append(block_koening)
    print(f'#####Для блока:\nkoening: {block_koening}\nhyperedges: {block_hyperedges}')


def createL2Graph(hyperedges: dict, koening_graph: dict)->dict:
    l2 = dict()
    for node, edges in koening_graph.items():
        node_neighbours = []
        for edge in edges:
            node_neighbours += hyperedges[edge]
        node_neighbours = set(node_neighbours) - {node}
        l2[node] = node_neighbours
    return l2

l2 = createL2Graph( blocks_hyperedges[0],blocks_koenig_graph[0])

# Является ли гиперграф триангулированным ?
def isGraphTriangulated(l2_graph: dict) -> bool:
    l2 = l2_graph.copy()
    count_nodes = len(l2)

    # Поиск и удаление симплициальных вершин
    while count_nodes:
        nodes = l2.keys()
        pass

    return count_nodes == 0

# Поиск ситуации 1



plt.show()







