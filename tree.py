class Tree:
    def __init__(self):
        self.threshold_value = None
        self.feature = None
        self.nodes = [None, None]

    def __getitem__(self, item):
        if type(item) == type(int()):
            return self.nodes[item]
        elif type(item) == type(list()):
            if len(item) > 1:
                return self.__getitem__(item[0]).__getitem__(item[1:])
            else:
                return self.__getitem__(item[0])

    def __setitem__(self, item, value):
        if type(item) == type(int()):
            self.nodes[item] = value
        elif type(item) == type(list()):
            if len(item) > 1:
                self[item[0]][item[1:]] = value
            else:
                self[item[0]] = value

    def _eval(self, object):
        node = self.nodes[0] if object[self.feature] >= self.threshold_value else self.nodes[1]
        return node._eval(object) if type(Tree()) == type(node) else node

    def _get_graph(self, iterator=0):
        graph = [[iterator, self.feature, self.threshold_value], []]
        for child_index, child in enumerate(self.nodes):
            if type(Tree()) == type(child):
                if child_index == 0:
                    graph[1].append(child._get_graph(iterator + 1))
                else:
                    if type(self.nodes[0]) == type(Tree()):
                        graph[1].append(child._get_graph(iterator + 2 + self.nodes[0].check_descendants()))
                    else:
                        graph[1].append(child._get_graph(iterator + 2))
            else:
                if child_index == 0:
                    graph[1].append([iterator + 1, child])
                else:
                    if type(self.nodes[0]) == type(Tree()):
                        graph[1].append([iterator + 2 + self.nodes[0].check_descendants(), child])
                    else:
                        graph[1].append([iterator + 2, child])

        return graph

    def eval(self, object_list):
        return [self._eval(object) for object in object_list]

    def show(self):
        def _get_edges(array):
            edges = []
            for child in array[1]:
                if isinstance(child[0], list):
                    [edges.append(edge) for edge in _get_edges(child)]
                    edges.append((array[0][0], child[0][0]))
                else:
                    edges.append((array[0][0], child[0]))

            return edges

        def _get_labels_loop(array):
            edges = []
            for child in array[1]:
                if isinstance(child[0], list):
                    edges.append(child[0][1:])
                    [edges.append(label) for label in _get_labels_loop(child)]
                else:
                    edges.append(child[1:])

            return edges

        def _get_labels(array):
            return [array[0][1:], *_get_labels_loop(array)]

        def _get_heights_loop(array, iterator=1):
            edges = []
            for child in array[1]:
                if isinstance(child[0], list):
                    edges.append(iterator)
                    [edges.append(label) for label in _get_heights_loop(child, iterator + 1)]
                else:
                    edges.append(iterator)

            return edges

        def _get_heights(array):
            return [0, *_get_heights_loop(array)]

        import networkx as nx
        import matplotlib.pyplot as plt

        edges = _get_edges(self._get_graph())
        labels = _get_labels(self._get_graph())
        y_pos = _get_heights(self._get_graph())
        x_temp = [[i - (y_pos.count(level) / 2.) for i in range(y_pos.count(level))] for level in range(max(y_pos) + 1)]
        x_pos = [-x_temp[i].pop() for i in y_pos]

        labels = {i: "[%s] > %s" % (label[0], label[1]) if len(label) > 1 else "%s" % label[0] for i, label in
                  enumerate(labels)}

        graph = nx.Graph()
        graph.add_nodes_from(range(self.check_nodes()))

        pos = [(x_pos[node], -y_pos[node]) for node in range(self.check_nodes())]
        nx.draw_networkx_nodes(graph, pos, node_size=3000, node_color='#A0CBE2')
        nx.draw_networkx_edges(graph, pos, edges)
        nx.draw_networkx_labels(graph, pos, labels, 8)
        plt.show()

    def check_descendants(self):
        return sum([child.check_descendants() if type(Tree()) == type(child) else 0 for child in self.nodes]) + 2

    def check_height(self):
        return max([child.check_height() if type(Tree()) == type(child) else 1 for child in self.nodes]) + 1

    def check_depth(self):
        return self.check_height() - 1

    def check_nodes(self):
        return self.check_descendants() + 1


def generate_random_tree(features, terminal_node_callback, max_depth=10, iteration=0):
    from random import random, choice
    tree = None

    if random() > (iteration * 1. / (max_depth - 1)):
        tree = Tree()
        tree.feature = choice(range(features))
        tree.threshold_value = random() * 2 - 1
        for j in range(2):
            tree.nodes[j] = generate_random_tree(features, terminal_node_callback, max_depth, iteration + 1)
    else:
        tree = terminal_node_callback()

    return tree


def crossover_trees(tree1, tree2, co_factor=(0.5, 0.5)):
    from random import random, choice
    from copy import deepcopy as dc
    tree = tree1

    path1 = [choice(range(2))]
    path2 = [choice(range(2))]

    while type(tree1[path1]) == type(Tree()) and random() > co_factor[0]:
        path1.append(choice(range(2)))

    while type(tree2[path2]) == type(Tree()) and random() > co_factor[1]:
        path2.append(choice(range(2)))

    tree[path1] = dc(tree2[path2])

    return tree


def _random_prune(tree, terminal_node_callback, prune_factor=0.33):
    from random import random, choice

    path = []

    if tree.check_descendants() > 2:
        while len(path) < 1:
            path = [choice(range(2))]

            while type(tree[path]) == type(Tree()) and random() > prune_factor:
                path.append(choice(range(2)))

            if type(tree[path]) != type(Tree()):
                path = path[:-1]

        tree[path] = terminal_node_callback()

    return tree


def _random_terminal_mutation(tree, terminal_node_callback):
    from random import random, choice

    path = []

    while len(path) < 1:
        path = [choice(range(2))]

        while type(tree[path]) == type(Tree()):
            path.append(choice(range(2)))

    tree[path] = terminal_node_callback()

    return tree


def _random_branch_mutation(tree, features, branch_mutation_factor=0.66):
    from random import random, choice

    path = []

    if tree.check_descendants() > 2:
        while len(path) < 1:
            path = [choice(range(2))]

            while type(tree[path]) == type(Tree()) and random() > branch_mutation_factor:
                path.append(choice(range(2)))

            if type(tree[path]) != type(Tree()):
                path = path[:-1]

        if random() > 0.5:
            tree[path].feature = choice(range(features))
        else:
            tree[path].threshold_value = random() * 2 - 1

    return tree


def tree_mutation(tree, features, terminal_node_callback, f1=0.33, f2=0.33):
    from random import random

    if f1 + f2 < 1:
        roll = random()

        if roll > f1 + f2:
            return _random_prune(tree, terminal_node_callback)
        elif roll > f1:
            return _random_branch_mutation(tree, features)
        else:
            return _random_terminal_mutation(tree, terminal_node_callback)

    else:
        raise Exception('Factors sum to 1 or more')
