# Α α, Β β, Γ γ, Δ δ, Ε ε, Ζ ζ, Η η, Θ θ, Ι ι, Κ κ, Λ λ, Μ μ, Ν ν, Ξ ξ, Ο ο, Π π, Ρ ρ, Σ σ ς, Τ τ, Υ υ, Φ φ, Χ χ, Ψ ψ, Ω ω
import math
class Value:
    def __init__(self, data, op='', label='', children=[], backward:callable=lambda:None): 
        assert isinstance(data, (int, float))
        self.data = data
        self.grad = 0.0 # ∂(Loss)/∂(self.data). An entry in Loss gradient   https://en.wikipedia.org/wiki/Gradient. Respecting torch api
        self._op = op # the operation that generated the value (if exists. only for visualizations)
        self._label = label # (optional node name (label). only for visualizations)
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
        
        def out_backward():
            """
            Compute and set ∂(Loss)/∂(child) for each child of self.
            ∂(Loss)/∂(child) = ∂(Loss)/∂(self) * ∂(self)/∂(child) 
            """
            dL_dself = self.grad # ∂(Loss)/∂(self). guaranteed to be known during backprop due to topological sort
            for c in self._children: 
                dself_dc = 1.0 #  ∂(self)/∂(child)
                c.grad = dL_dself * dself_dc # set ∂(Loss)/∂(child) from chain rule (∂(Loss)/∂(child) = ∂(Loss)/∂(self) * ∂(self)/∂(child) )

        out_node = Value(out_data, '+', children=(self, other), backward=out_backward) # new node was born in the computational graph
        return out_node 

    def __mul__(self, other):
        """
        return a new Value node of self.data * 
        """
        other = other if isinstance(other,Value) else Value(other)
        out_data = self.data * other.data  
        
        def out_backward():
            """
            Compute and set ∂(Loss)/∂(child) for each child of self.
            ∂(Loss)/∂(child) = ∂(Loss)/∂(self) * ∂(self)/∂(child) 
            """
            dL_dself = self.grad # ∂(Loss)/∂(self). guaranteed to be known during backprop due to topological sort
            c1, c2 = self._children
            dself_dc1 = c2.data # ∂(c1c2)/∂c1 = c2
            c1.grad = dL_dself * dself_dc1 
            dself_dc2 = c1.data # ∂(c1c2)/∂c2 = c1
            c2.grad = dL_dself * dself_dc2 
            
        out_node = Value(out_data, '*', children=(self, other), backward=out_backward) # new node was born in the computational graph
        return out_node 

    def __pow__(self, other):
        # other = other if isinstance(other, Value) else Value(other)
        out_data = self.data ** other.data
        def out_backward():
            dL_dself = self.grad # ∂(Loss)/∂(self)
            

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1 # use __pow__ and __mul__
    
    def __sub__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        return self + (-other) # use __add__

    def __exp__(self):
        """
        return a new Value node of self.data * 
        """
        other = other if isinstance(other,Value) else Value(other)
        out_data = math.exp(self.data) # e^x  
        
        def out_backward():
            """
            Compute and set ∂(Loss)/∂(child) for each child of self.
            ∂(Loss)/∂(child) = ∂(Loss)/∂(self) * ∂(self)/∂(child) 
            """
            dL_dself = self.grad # ∂(Loss)/∂(self). guaranteed to be known during backprop due to topological sort
            dself_dc = self.data # ∂(e^x)/∂(x) = e^x 
            c = self._children[0] # only one child
            c.grad += dL_dself * dself_dc # chain rule 

        out_node = Value(out_data, 'exp', children=(self,), backward=out_backward) # new node was born in the computational graph
        return out_node
    

  
    
    
    def __div__(self):
        """
        return a new Value node of self.data * 
        """
        out_data = math.exp(self.data) # e^x  
        
        def out_backward():
            """
            Compute and set ∂(Loss)/∂(child) for each child of self.
            ∂(Loss)/∂(child) = ∂(Loss)/∂(self) * ∂(self)/∂(child) 
            """
            dL_dself = self.grad # ∂(Loss)/∂(self). guaranteed to be known during backprop due to topological sort
            dself_dc = self.data # ∂(e^x)/∂(x) = e^x 
            c = self._children[0] # only one child
            c.grad += dL_dself * dself_dc # chain rule 

        out_node = Value(out_data, '^', children=[self], backward=out_backward) # new node was born in the computational graph
        return out_node

    def relu(self):
        """
        Relu(x) = max(x,0)
        """
        out_data = math.max(0,self.val)
        def out_backward():
            c = self._children[0] # x
            c.grad = float(c.data > 0) # ∂(Relu(x)/∂(x)) = 1 if x > 0 else 0 
        out_node = Value(out_data, 'Relu', children=[self], backward=out_backward)
        return out_node



def backward(loss_node):
    """
    Accumulates new gradients in the network. 
    loss_node: node which holds the network's loss. 
    """
    loss_node.grad = 1.0 # d(loss)/d(loss) = 1 
    topo_sort = topological_sort(loss_node) # starting from the loss node, ending in input nodes (input coordinates)
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
        for ch in node.children:
            build_topo_rec(ch)
        
        reversed_topo_set.add(node)
        reversed_topo.append(node)
        
        
    build_topo_rec(root)
    topo = reversed_topo.reversed()
    return topo 

