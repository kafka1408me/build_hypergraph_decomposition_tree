import sys
import hypernetx as hnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from random import randint, shuffle
from PyQt5.QtGui import QIntValidator, QValidator
from PyQt5.QtWidgets import QApplication, QLabel, QMessageBox, QWidget, QMainWindow, QHBoxLayout, QVBoxLayout, \
    QLineEdit, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


def flatten(t):
    return [item for sublist in t for item in sublist]


class MainWidget(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout()
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setSpacing(5)

        self.canvas = MplCanvas(self, 5, 4)

        self.graph_size_input = Input()
        self.graph_size_input.setTitle("Количество вершин")
        self.graph_size_input.setText("12")
        self.graph_size_input.setValidator(QIntValidator())

        self.number_of_edges_input = Input()
        self.number_of_edges_input.setTitle("Количество гиперрёбер")
        self.number_of_edges_input.setText("4")
        self.number_of_edges_input.setValidator(QIntValidator())

        generate_button = QPushButton('Сгенерировать', self)
        generate_button.clicked.connect(self.generate)

        self.start_node_input = Input()
        self.start_node_input.setTitle("Стартовая вершина")
        self.start_node_input.setText("1")
        self.start_node_input.setValidator(QIntValidator())

        self.end_node_input = Input()
        self.end_node_input.setTitle("Конечная вершина")
        self.end_node_input.setText("5")
        self.end_node_input.setValidator(QIntValidator())

        run_button = QPushButton('Найти кратчайший путь', self)
        run_button.clicked.connect(self.run)

        self.result_label = QLabel()
        self.result_label.setWordWrap(True)

        main_layout.addWidget(self.canvas)
        main_layout.setStretch(0, 10)
        main_layout.addItem(sidebar_layout)

        sidebar_layout.addWidget(self.graph_size_input)
        sidebar_layout.addWidget(self.number_of_edges_input)
        sidebar_layout.addWidget(generate_button)
        sidebar_layout.addWidget(self.start_node_input)
        sidebar_layout.addWidget(self.end_node_input)
        sidebar_layout.addWidget(run_button)
        sidebar_layout.addWidget(self.result_label)
        sidebar_layout.addStretch(10)

        self.setLayout(main_layout)

    def run(self):
        self.result_label.setText("")

        graph = self.graph
        nodes = list(graph.nodes)

        try:
            start = int(self.start_node_input.text())
        except ValueError:
            self.showError("Введите стартовую вершину")
            return
        if not (f"v{start}" in nodes):
            self.showError("Стартовая вершина не найдена в графе")
            return

        try:
            end = int(self.end_node_input.text())
        except ValueError:
            self.showError("Введите конечную вершину")
            return
        if not (f"v{end}" in nodes):
            self.showError("Конечная вершина не найдена в графе")
            return

        start = f"v{start}"
        end = f"v{end}"

        hyperedges = self.hyperedges.copy()
        for i in graph.edges:
            hyperedge = graph.edges[i]
            hyperedges[i] = {
                "nodes": hyperedges[i],
                "weight": hyperedge.weight
            }

        print("\nnodes", nodes)
        print("hyperedges", hyperedges)
        print("start node", start)
        print("end node", end)
        print('-----------')

        dist = {node: 1000000 for node in nodes}
        routes = {node: [] for node in nodes}

        dist[start] = 0
        queue = [node for node in nodes]
        seen = set()
        while len(queue) > 0:
            min_dist = sys.maxsize
            min_node = None
            for node in queue:
                if dist[node] < min_dist and node not in seen:
                    min_dist = dist[node]
                    min_node = node

            queue.remove(min_node)
            seen.add(min_node)
            connections = [(i, hyperedges[i]["nodes"], hyperedges[i]["weight"])
                           for i in hyperedges if min_node in hyperedges[i]["nodes"]]
            print(min_node)
            print(connections)

            for (edge, neighbors, weight) in connections:
                for neighbor in neighbors:
                    tot_dist = weight + min_dist
                    if tot_dist < dist[neighbor]:
                        dist[neighbor] = tot_dist
                        routes[neighbor] = list(routes[min_node])
                        routes[neighbor].append(edge)

        print('-----------')
        print("distances", dist)
        print("routes", routes)
        self.result_label.setText(
            f"Кратчайшее расстояние от {start} до {end} равно {dist[end]}\nМаршрут: {routes[end]}")

    def generate(self):
        self.result_label.setText("")

        try:
            graph_size = int(self.graph_size_input.text())
        except ValueError:
            self.showError("Введите размер графа")
            return

        try:
            number_of_edges = int(self.number_of_edges_input.text())
        except ValueError:
            self.showError("Введите количество гиперрёбер")
            return

        nodes = [f"v{i}" for i in range(1, graph_size + 1)]
        shuffled_nodes = nodes
        shuffle(shuffled_nodes)
        simple_edges = [[shuffled_nodes[i], shuffled_nodes[i + 1]]
                        for i in range(len(shuffled_nodes) - 1)]
        shuffle(simple_edges)

        hyperedge_volume = int(len(simple_edges) / number_of_edges)
        hyperedges = [simple_edges[hyperedge_volume * i: hyperedge_volume *
                                                         (i + 1)] for i in range(number_of_edges)]
        hyperedges[number_of_edges -
                   1].append(flatten(simple_edges[hyperedge_volume * number_of_edges:]))
        hyperedges = [flatten(i) for i in hyperedges]
        hyperedges = {
            i: list(set(hyperedges[i]))
            for i in range(len(hyperedges))
        }
        print("\n", hyperedges)

        self.graph = hnx.Hypergraph(hyperedges)
        self.hyperedges = hyperedges
        for i in range(len(self.graph.edges)):
            self.graph.edges[i].weight = randint(1, 100)
        self.draw()

    def draw(self):
        self.canvas.clear()
        edge_labels = {i: f"{i}:{self.graph.edges[i].weight}"
                       for i in range(len(self.graph.edges))}
        hnx.draw(
            self.graph,
            ax=self.canvas.axes,
            edge_labels=edge_labels,
            node_labels_kwargs={
                'fontsize': 8
            }
        )

        self.canvas.draw()

    def showError(self, text: str):
        error_dialog = QMessageBox()
        error_dialog.setText(text)
        error_dialog.adjustSize()
        error_dialog.exec()


class Input(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.title = QLabel()
        self.le = QLineEdit()

        layout.addWidget(self.title)
        layout.addWidget(self.le)

    def text(self) -> str:
        return self.le.text()

    def setText(self, text: str):
        self.le.setText(text)

    def setTitle(self, text: str):
        self.title.setText(text)

    def setValidator(self, validator: QValidator):
        self.le.setValidator(validator)


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

    def clear(self):
        self.axes.cla()


class Window(QMainWindow):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setGeometry(600, 600, 1000, 800)
        self.setWindowTitle('Нахождение кратчайшего пути в гиперграфе')

        mainWidget = MainWidget()
        mainWidget.generate()

        self.setCentralWidget(mainWidget)
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Window()
    sys.exit(app.exec())
