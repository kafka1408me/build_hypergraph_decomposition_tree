import sys
import hypernetx as hnx
import networkx as nx
#import numpy as np
import math
import matplotlib.pyplot as plt

from random import randint, shuffle
from PyQt5.QtGui import QIntValidator, QValidator
from PyQt5.QtWidgets import QApplication, QLabel, QMessageBox, QWidget, QMainWindow, QHBoxLayout, QVBoxLayout, \
    QLineEdit, QPushButton, QSplitter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import QSize

import graph_decomposition


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

        run_button = QPushButton('Найти дерево декомпозиции гиперграфа', self)
        run_button.clicked.connect(self.run)

        self.result_label = QLabel()
        self.result_label.setWordWrap(True)

        self.decomposition_tree_canvas = MplCanvas(self, 6, 4)

        self.splitter = QSplitter()

        self.splitter.addWidget(self.canvas)
        self.splitter.addWidget(self.decomposition_tree_canvas )

        main_layout.addWidget(self.splitter)
        main_layout.setStretch(0, 10)
        main_layout.addItem(sidebar_layout)

        sidebar_layout.addWidget(self.graph_size_input)
        sidebar_layout.addWidget(self.number_of_edges_input)
        sidebar_layout.addWidget(generate_button)
        sidebar_layout.addWidget(run_button)
        sidebar_layout.addWidget(self.result_label)
        sidebar_layout.addStretch(10)

        self.setLayout(main_layout)

    def run(self):
        self.result_label.setText("")

        self.graph_decomposition_tree, decomposition_tree_width = \
            graph_decomposition.buildHypergraphDecompositionTree(self.hyperedges)

        self.result_label.setText(f"<font color='blue'>Ширина построенного дерева декомпозиции: {decomposition_tree_width}</font>")

        self.decomposition_tree_canvas.show()
        self.drawDecompositionTree()


    def generate(self):
        self.decomposition_tree_canvas.hide()
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
            i + 1: list(set(hyperedges[i]))
            for i in range(len(hyperedges))
        }
        print("\n", hyperedges)

        self.graph = hnx.Hypergraph(hyperedges)
        self.hyperedges = hyperedges
        self.drawHypergraph()

    def drawHypergraph(self):
        self.canvas.clear()
        hnx.draw(
            self.graph,
            ax=self.canvas.axes,
            node_labels_kwargs={
                'fontsize': 8
            }
        )
        self.canvas.draw()

    def drawDecompositionTree(self):
        self.decomposition_tree_canvas.clear()
        nx.draw_kamada_kawai(
            self.graph_decomposition_tree,
            ax=self.decomposition_tree_canvas.axes,
            with_labels=True
        )
        self.decomposition_tree_canvas.draw()

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
        self.setMinimumWidth(150)

    def clear(self):
        self.axes.cla()



class Window(QMainWindow):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setGeometry(600, 600, 1000, 800)
        self.setWindowTitle('Построение дерева декомпозиции гиперграфа')

        mainWidget = MainWidget()
        mainWidget.generate()

        self.setCentralWidget(mainWidget)
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Window()
    sys.exit(app.exec())
