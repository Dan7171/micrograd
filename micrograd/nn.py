
from engine import Value
import random
import numpy as np

class Neuron:
    """A single neuron, performing <x^T,w>+b"""
    def __init__(self, in_features:int):
        """

        σ: non linearity function
        """
        self.w = [Value(random.uniform(-1,1)) for _ in range(in_features)]
        self.b = Value(random.uniform(-1,1))
        
    def __call__(self,x) -> Value:
        return self.b + np.sum([xi*wi for (xi, wi) in zip(x,self.w)])
    
    
    def parameters(self):
        param_list = []
        for wi in self.w:
            param_list += wi
        param_list += b
        return param_list

class Layer: # 
    """A layer of Neurons (nn.Linear in torch) """
    def __init__(self, in_features :int, out_features:int, non_lin:str):
        """
        in_features: Previous Layer Width. Number of inputs for a neuron in the layer (number of inputs to the layer, the width of previous layer) 
        out_features: This Layer Width. (number of neurons in this layer). 
        """
        self.neurons = [Neuron(in_features) for i in range(out_features)]


    def parameters(self):        
        param_list = []
        for neuron in self.neurons:
            param_list.extend(neuron.parameters())
        return param_list

    def __call__(self,x) -> list[Value]:
        output = [None for _ in self.neurons]
        for i,ni in enumerate(self.neurons):
            output[i] = ni(x) # output of the i-th neuron in the layer
            
        return output

class ReLU:
    """
    element-wise ReLU on input vector 
    """
    def __init__(self):
        pass
    
    def __call__(self,x:list[Value]):
        output = [None for _ in range(len(x))]
        for i,xi in enumerate(x):
            output[i] = xi.relu() # Relu(xi)
        return output
        
class MLP:
    """ Multi Layer Perceptron (Fully Connected FF NN)"""
    def __init__(self, in_features:int, layers_features:list[int], non_lin:'relu'):
        """
        in_features: input dimension of the network
        layers_features: list of dimension of each layer (number of neurons)
        example: nin=5, nouts=[4,5,2] 
                *       *
                *   *   *
                *   *   *   *
                *   *   *   *
                *   *   *
                
        """
        nn_dims = [in_features] + layers_features
        self.layers = [Layer(nn_dims[i], nn_dims[i+1], non_lin) for i in range(len(nn_dims)-1)]
        self.non_lin = [self._get_non_lin(non_lin) for _ in range(len(layers_features)-1 )]
    
    def _get_non_lin(self, non_lin_type):
        match non_lin_type:
            case 'relu':
                return ReLU()

    def parameters(self):        
        param_list = []
        for layer in self.layers:
            param_list.extend(layer.parameters())
        return param_list
    
    def __call__(self,x):
        """
        Forward pass
        """
        for i,layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                σi = self.non_lin[i] # non linearity function (vector to vector) 
                x = σi(x) # non linearity on all layers except the last.    
        return x
        
if __name__ == '__main__':
    random.seed(42)
    nn = MLP(3,[2,1],'relu')
    

    X_train = [
        [1.0, 2.0, 3.0],
        [0.0, 1.0, 0.0],
        [0.5, 1.0, 3.0],
            
    ]
    y_train = [1.0, 2.0, 3.0] 

    n_epoch = 3
    lr = 0.01
    for ep in range(n_epoch):
        loss = 0
        for x,y in zip(X_train,y_train):
            y_pred = nn(x)
         
            loss += (y_pred[0] - y)**2
        
        for p in nn.parameters():
            p.grad = 0 

        loss.backward() # get d(L)/d(p) for each parameter p in the network 
        
        for p in nn.parameters():
            p.value -= lr * p.grad # Update network params: p -= lr * d(L)/d(p)
