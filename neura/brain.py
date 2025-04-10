import numpy as np
import networkx as nx
import random

from neuratron import Model

def generate_brain_(n_nodes:int, n_conns_max:int, inner_neuratron_shape:tuple=(20, 20)):
    B = nx.Graph()
    nodes = [Model(shape=inner_neuratron_shape, lr=0.001) for i in range(n_nodes-1)]
    nodes.append(Model(shape=inner_neuratron_shape, lr=0.001, is_output=True))

    for node in nodes:
        for other in random.sample(nodes, k=n_conns_max):
            if node != other:
                B.add_edge(node, other)

    return B, nodes[0], nodes[-1]

def feed_forward(B, input_node:Model, X:np.ndarray):
    outputs = []
    input_node.is_processed = True

    for neighbor in B.neighbors(input_node):
        output = input_node.feed_forward(X, 10)
        
        if not neighbor.is_processed:
            outputs.append(feed_forward(B, neighbor, output))

        elif neighbor.is_output:
            outputs.append(neighbor.feed_forward(output, 1))

    return outputs

def backpropagation(B, output_node:Model, grad=None):
    grad = output_node.backward_pass(grad)
    for neighbor in B.neighbors(output_node):
        new_grad = neighbor.backward_pass(grad)
        
        if neighbor.is_processed:
            backpropagation(B, neighbor, new_grad)

        neighbor.is_processed = False
