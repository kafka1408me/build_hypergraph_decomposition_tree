import hypernetx as hnx
import matplotlib.pyplot as plt
import networkx as nx
import copy
from random import shuffle

empty_set = set()

# count_nodes = 16
#
# hyperedges = {1:['v1', 'v2', 'v3', 'v4', 'v5', 'v6'],
#               2:['v6', 'v7', 'v8'],
#               3:['v5', 'v7', 'v9'],
#               4:['v8', 'v9', 'v10', 'v11', 'v12', 'v13'],
#               5:['v13', 'v14', 'v15'],
#               6:['v15', 'v16'],
#               7:['v14', 'v16']}

# graph_size = 60
# number_of_edges = 30
#
# def flatten(t):
#     return [item for sublist in t for item in sublist]
#
#
# nodes = [f"v{i}" for i in range(1, graph_size + 1)]
# shuffled_nodes = nodes
# shuffle(shuffled_nodes)
# simple_edges = [[shuffled_nodes[i], shuffled_nodes[i + 1]]
#                 for i in range(len(shuffled_nodes) - 1)]
# shuffle(simple_edges)
#
# hyperedge_volume = int(len(simple_edges) / number_of_edges)
# hyperedges = [simple_edges[hyperedge_volume * i: hyperedge_volume *
#                                                  (i + 1)] for i in range(number_of_edges)]
# hyperedges[number_of_edges -
#            1].append(flatten(simple_edges[hyperedge_volume * number_of_edges:]))
# hyperedges = [flatten(i) for i in hyperedges]
# hyperedges = {
#     i + 1: list(set(hyperedges[i]))
#     for i in range(len(hyperedges))
# }

# Ациклические гиперграф
# hyperedges = {1:['v2', 'v4'],
#               2:['v4', 'v6'],
#               3:['v1', 'v8'],
#               4:['v1', 'v7'],
#               5:['v3', 'v5', 'v6', 'v7']}


def buildHypergraphDecompositionTree(_hyperedges):
    hyperedges = {edge - 1:sorted([int(node_str[1:]) - 1 for node_str in node]) for edge, node in _hyperedges.items()}
    print('starting hyperedges:\n', hyperedges)

    max_edge_number = max(hyperedges.keys())

    def createKoenigGraphFromHyperedges(my_hyperedges):
        #    koenig_graph = {i:[] for i in range(count_nodes)}
        koenig = dict()
        for edge, _nodes in my_hyperedges.items():
            for _node in _nodes:
                if not _node in koenig:
                    koenig[_node] = []
                koenig[_node].append(edge)
        return koenig

    koenig_graph = createKoenigGraphFromHyperedges(hyperedges)

    original_hyperedges = copy.deepcopy(hyperedges)

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
        #    print('edges:', edges)
            count_edges = len(edges)
            del_edges = []
            print("koening now:", koenig_graph)
            print("hyperedges now:", hyperedges)
            for edge_num in range(count_edges):
                edge = edges[edge_num]
                print('& edge:',edge)
                if edge in del_edges:
            #        print(f'edge {edge} in del_edges; continue')
                    continue
                cur_edge = set(hyperedges[edge])
                if cur_edge == empty_set:
                    del_edges.append(edge)
                    print(cur_edge, 'is empty!')
                    continue
                for _edge_num in range(edge_num+ 1, count_edges):
                    _edge = edges[_edge_num]
                    other_edge = hyperedges[_edge]
            #        print('edge:',edge,'; _edge:', other_edge)
                    if cur_edge.issubset(other_edge):
                        print(f'1)edge {edge} del')
                        del_edges.append(edge)
                        break
                    elif set(other_edge).issubset(cur_edge):
                        print(f'2)edge {_edge} del')
                        del_edges.append(_edge)
            count_deleted += len(del_edges)
            for del_edge in del_edges:
                for node in hyperedges[del_edge]:
                    koenig_graph[node].remove(del_edge)
                hyperedges.pop(del_edge)
            print("koening now:", koenig_graph)
            print("hyperedges now:", hyperedges)
            print('*************************')
            pass
        pass


    print('------ new koening_graph:\n', koenig_graph)
    print('------ new hyperedges:\n', hyperedges)

    # Шаг I CTDA

    GrahamAlgorithm(koenig_graph=koenig_graph, hyperedges=hyperedges)

    # Шаг 2 CTDA (если гиперграф не М-ациклический)
    if len(koenig_graph) != 0:
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

        blocks_koenig_graph = []
        blocks_hyperedges = []

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

        h_add = copy.deepcopy(original_hyperedges)

        for i in range(len(blocks_hyperedges)):
            l2 = createL2Graph( blocks_hyperedges[i],blocks_koenig_graph[i])
            #isGraphTriangulated(l2)

            edge_list = []
            for node, neihbour_nodes in l2.items():
                for neihbour_node in neihbour_nodes:
                    if neihbour_node < node:
                        continue
                    edge_list.append((node, neihbour_node))

            print('edge_list:', edge_list)

            g = nx.Graph(edge_list)
            is_triangulated_graph = nx.algorithms.is_chordal(g)
            if is_triangulated_graph:
                triangulared_graph = g
            else:
                triangulared_graph, a = nx.algorithms.complete_to_chordal_graph(g)
            max_cliques = nx.algorithms.clique.find_cliques(triangulared_graph)
            print('triangulated:', list(triangulared_graph.edges))

            for i, max_clique in enumerate(max_cliques):
                clique_set = set(max_clique)
                is_edge_exist = False
                for edge, nodes in h_add .items():
                    if set(nodes) == clique_set:
                        is_edge_exist = True
                        break
                if not is_edge_exist:
                    max_edge_number += 1
                    h_add [max_edge_number] = max_clique

        print('original_heperedges + max_cliques =', h_add)
    else:
        print('hypergraph is M-acyclic !!!')
        h_add = original_hyperedges

    # Этап III CTDA
    lh = []
    nodes_sets = [set(nodes) for edge, nodes in h_add.items()]
    for i in range(max_edge_number):
        for j in range(i + 1, max_edge_number + 1):
            intersection = nodes_sets[i].intersection(nodes_sets[j])
            if intersection != empty_set:
                lh.append((-len(intersection), (i,j)))

    print('lh =', lh)
    print('len(lh)', len(lh))

    p = []

    def MakeSet(x):
        p[x] = 0

    def FindSet(x):
        return x if x == p[x] else FindSet(p[x])

    def UnionSet(u, v):
        p[u] = p[v]

    def CruskalAlgorithm(lh):
        MST = []
        for edge in range(max_edge_number + 1):
            p.append(edge)
        lh = sorted(lh, key=lambda element: element[0])
        for element in lh:
            edge = element[1]
            uRep = FindSet(edge[0])
            vRep = FindSet(edge[1])
            if uRep != vRep:
                MST.append(edge)
                UnionSet(uRep, vRep)

        return MST

    decomposition_tree = CruskalAlgorithm(lh)

    print('*** decomposition tree ***\n', decomposition_tree)

    labaled_edges = {}

    decomposition_tree_width = -1
    for i in range(max_edge_number + 1):
        label = ''
        nodes = sorted(h_add[i])
        for node in nodes:
            label += 'v' + str(node + 1)
        labaled_edges[i] = label
        count_nodes = len(nodes)
        if count_nodes > decomposition_tree_width:
            decomposition_tree_width = count_nodes
    decomposition_tree_width -= 1

    labeled_decomposition_tree = [(labaled_edges[edge[0]], labaled_edges[edge[1]]) for edge in decomposition_tree]
    # Возвращаем дерево декомпозиции и ширину дерева декомпозиции
    return nx.Graph(labeled_decomposition_tree), decomposition_tree_width




# Тест
# graph_decomposition_tree = buildHypergraphDecompositionTree(hyperedges)
#
# hypergraph = hnx.Hypergraph(hyperedges)
#
# fig, ax = plt.subplots(1, 2)
#
# hnx.draw(
#     hypergraph,
#     ax=ax[0],
#     node_labels_kwargs={
#         'fontsize': 8
#     }
# )
#
# nx.draw_kamada_kawai(
#         graph_decomposition_tree,
#         ax=ax[1],
#         with_labels = True
#         )
#
# plt.show()



