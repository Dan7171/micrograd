# Α α, Β β, Γ γ, Δ δ, Ε ε, Ζ ζ, Η η, Θ θ, Ι ι, Κ κ, Λ λ, Μ μ, Ν ν, Ξ ξ, Ο ο, Π π, Ρ ρ, Σ σ ς, Τ τ, Υ υ, Φ φ, Χ χ, Ψ ψ, Ω ω
import math
import numpy as np

class Value:
    def __init__(self, data, op='', children=(), backward:callable=lambda:None): 
        assert isinstance(data, (int, float))
        self.data = data
        self.grad = 0.0 # ∂(Loss)/∂(self.data). An entry in Loss gradient   https://en.wikipedia.org/wiki/Gradient. Respecting torch api
        self._op = op # the operation that generated the value (if exists. only for visualizations)
        self.label = '' # (optional node name (label). only for visualizations)
        self._children = children # the Values that participated in self._op (will be children in DAG during backward pass) 
        self._backward = backward # Function to apply in backward pass. self._backward must compute and set child.grad =∂(Loss)/∂(child) for each child of self. 
    
    def __repr__(self):
        return f"Value({self.data})"
    
    def __add__(self, other):
        """
        return a new node for self + other 
        """
        other = other if isinstance(other,Value) else Value(other)
        out_data = self.data + other.data 
        out_node = Value(out_data, '+', children=(self, other)) # new node was born in the computational graph
        
        def out_backward():
            """
            Backward function of the "Out" node.
            """
            dl_do = out_node.grad # ∂(Loss)/∂(out)
            for c in out_node._children: # self and other  
                do_dc = 1.0 #  ∂(out)/∂(child) (here 1 because ∂(self+other)/∂(self)=∂(self+other)/∂(other) = 1)
                c.grad = dl_do * do_dc # set ∂(Loss)/∂(child) from chain rule 
        
        out_node._backward = out_backward
        return out_node 


    def __mul__(self, other):
        """
        return a new Value node of self.data * 
        """
        other = other if isinstance(other,Value) else Value(other)
        out_data = self.data * other.data  
        out_node = Value(out_data, '*', children=(self, other)) # new node was born in the computational graph

        def out_backward():
            """
            Backward function of the "Out" node.
            """
            dl_do = out_node.grad 
            c1, c2 = out_node._children
            
            do_dc1 = c2.data # ∂(c1c2)/∂c1 = c2
            c1.grad += dl_do * do_dc1 
            
            do_dc2 = c1.data # ∂(c1c2)/∂c2 = c1
            c2.grad += dl_do * do_dc2 
        
        out_node._backward = out_backward
        return out_node 

    def __rmul__(self,other):
        """
        Enables Multiplication (__mul__) of both self*other and other*self (a fallback if __mul__ fails) 
        """
        return self * other

    def __pow__(self, other):
        """
        x**(const)
        """
        # other = other if isinstance(other, Value) else Value(other)

        assert type(other) in [int, float]
        out_data = self.data ** other
        out_node = Value(out_data, op=f'**{other:.2f}',children=(self,))
        def out_backward():
            """
            Backward function of the "Out" node.
            """
            dl_do = out_node.grad # ∂(Loss)/∂(Out)
            c = out_node._children[0] 
            do_dc = other * c.data**(other - 1) # ∂(Out)/∂(child)
            c.grad += dl_do * do_dc # ∂(Loss)/∂(child) = ∂(Loss)/∂(Out) * ∂(Out)/∂(child) by chain rule 
        
        out_node._backward = out_backward
        return out_node

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1 # use __pow__ and __mul__
    
    def __sub__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        return self +  (-1.0) * other # use __add__

    def __exp__(self):
        """
        returns a new Value instance (a node) with val = e^(self.data) 
        """
        out_data = math.exp(self.data) # e^x  
        out_node = Value(out_data, f'e^x', children=(self,)) 

        def out_backward():
            """
            Backward function of the "Out" node.
            """
            dl_do = out_node.grad # ∂(Loss)/∂(out)
            do_dc = out_node.data # ∂(e&c)/∂(c) == ∂(out)/∂(c) = e^c 
            c = self._children[0] # c is the only child of e^c 
            c.grad += dl_do * do_dc # ∂(loss)/∂(child)  (from chain rule) 
        out_node._backward = out_backward
        return out_node
    


    def relu(self):
        """
        Relu(x) = max(x,0)
        """
        out_data = max(0,self.data)
        out_node = Value(out_data, 'ReLU', children=(self,))

        def out_backward():
            dl_do = out_node.grad
            c = self._children[0] # x
            do_dc = float(c.data > 0) # # ∂(Relu(x)/∂(x)) = 1 if x > 0 else 0 
            c.grad = dl_do * do_dc # chain rule 
        out_node._backward = out_backward
        return out_node


    def backward(self):
        """
        Accumulates new gradients in the network. 
        loss_node: node which holds the network's loss. 
        """
        topo_sort = topological_sort(self) # starting from the loss node, ending in input nodes (input coordinates)
        # print("Topo sort:")
        # for n in topo_sort:
        #     print(n, n.label)
        
        self.grad = 1.0 # d(loss)/d(loss) = 1 
        # print('Start backprop')
        for node in topo_sort:
            node._backward() # computing gradients for node's children using the chain rule (d(loss)/d(child) = d(loss)/d(node) * d(node)/d(child))


def topological_sort(root)->list:
    """
    params: root is the root of the dag
    return a topological sort (a list of nodes from the dag where each node is (somewhere) to the left of its children)
    """

    reversed_topo = []
    reversed_topo_set = set() # (visited set) for efficient checking if node is already exists added to reversed_topo
    
    def build_topo_rec(node):
        if node in reversed_topo_set:
            return # because we dont want to add any node more than once to the result
        for ch in node._children:
            build_topo_rec(ch)
        
        reversed_topo_set.add(node)
        reversed_topo.append(node)
        
        
    build_topo_rec(root)
    reversed_topo.reverse()
    return reversed_topo 

