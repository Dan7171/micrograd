
from engine import Value
from nn import MLP
import random
import numpy as np
from graphviz import Digraph

def trace(root):
    # builds sets of nodes and edges in the graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for c in v._children:
                edges.add((c,v))
                build(c)
    build(root)
    return nodes, edges
    
def draw_dot(root):
    dot = Digraph(format='svg',graph_attr={'rankdir':'LR'}) # left to right
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        # print(n)
        # print(n.label)
        val_node_label = '' if not len(n.label) else f'{n.label}, '    
        val_node_label += f'v:{n.data:.2f}, ∂(L)/∂(v):{n.grad:.2f}'
        print(val_node_label)
        dot.node(name=uid,label=val_node_label,shape='record')
        if n._op:
            # if the node is a result of some operation, create entering op node to it
            dot.node(name=uid+n._op, label=n._op) # create op node
            dot.edge(uid+n._op, uid) # edge from op symbol to op result
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)),str(id(n2))+n2._op) # edge from previous value to next op symbol

    return dot


# Simple example
a = Value(3.0); a.label='a'
b = a**2; b.label = 'b'
y_pred = b + 5; y_pred.label='y^'
y_true = Value(15.0); y_true.label='y'
diff =  y_pred - y_true ; diff.label='diff'
L = diff**2; L.label='L'
# v,e = trace(L)
# draw_dot(L) 
L.backward()

