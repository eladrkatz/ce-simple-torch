from functools import reduce
import math
import random
from typing import Literal, Optional
from graphviz import Digraph
from PIL import Image
import io

Operator = Literal["+", "*", "-", "tanh"]

class Value:
    def __init__(self, data: float, label: str):
        self.data = data
        self.label = label
        self.op:Optional[Operator] = None
        self.children: list[Value] = []
        self.grad = 0


def op(operator: Operator, operands: list[Value]):
    d = 0
    match operator:
        case "+":
            d = reduce(lambda a,b: a+b, [o.data for o in operands], 0)
        case "*":
            d = operands[0].data * operands[1].data
        case "-":
            d = operands[0].data - operands[1].data
        case "tanh":
            d = math.tanh(operands[0].data)
    
    v = Value(d, f"{operator}")
    v.op = operator
    v.children = operands
    
    return v

def draw_value_recursive(v: Value, graph: Digraph, visited: list[Value]):
    if v in visited:
        return
    visited.append(v)
    fillcolor = 'white'
    
    if "tanh" in v.label or "relu" in v.label:
        fillcolor = 'lightgreen'
    
    if "*" in v.label or "+" in v.label or "-" in v.label:
        fillcolor = "lightcyan"
    
    if "x" in v.label:
        fillcolor = "lightcoral"
    
    if "w" in v.label:
        fillcolor = "lightyellow"
    
    graph.node(
        name=str(id(v)),
        label=f"{{{v.label} | data {v.data:.4f} | grad {v.grad:.4f}}}",
        # label=f"{{{v.label} | data {v.data:.4f} }}",
        shape="record",
        style="filled",
        fillcolor=fillcolor,
    )

    for c in v.children:
        draw_value_recursive(c, graph, visited)
        graph.edge(str(id(c)), str(id(v)))

def draw_value(v: Value):
    Image.MAX_IMAGE_PIXELS = None  # Remove the limit entirely
    graph = Digraph(format="svg", graph_attr={"rankdir": "LR"})  # LR = left to right
    draw_value_recursive(v, graph, [])
    png_bytes = graph.pipe(format="png")
    img = Image.open(io.BytesIO(png_bytes))
    img.show()  

def _zero_grad_recursive(v: Value):
    v.grad = 0 
    for c in v.children:
        _zero_grad_recursive(c)


def calc_grad(v: Value):
    _zero_grad_recursive(v)
    v.grad = 1
    _calc_grad_recursive(v, [])
 
def _calc_grad_recursive(v: Value, visited: list[Value]):
    if v in visited:
        return
    visited.append(v)   

    match v.op:
        case "+":
            for c in v.children:
                c.grad += v.grad
        case "*":
            v.children[0].grad += v.grad * v.children[1].data
            v.children[1].grad += v.grad * v.children[0].data
        case "-":
            v.children[0].grad += v.grad
            v.children[1].grad += -v.grad
        case "tanh":
            v.children[0].grad += v.grad * (1 - v.data**2)
    
    for c in v.children:
        _calc_grad_recursive(c, visited)     
            
def show_grad():
    # n = x1w1+ x2*w2 + x3*w3 + b
    x1 = Value(1, 'x1'); x2 = Value(2, 'x2'); x3 = Value(3, 'x3'); w1 = Value(4, 'w1'); w2 = Value(5, 'w2'); w3 = Value(6, 'w3'); b = Value(2, 'b')

    mult1 = op("*", [x1, w1])
    mult2 = op("*", [x2, w2])
    mult3 = op("*", [x3, w3])
    
    n = op("+", [mult1, mult2, mult3, b])
    
    # n = x1*w1+ x2*w2 + x3*w3 + b
    
    #print(n.data)
    
    
    calc_grad(n)
    
    draw_value(n)

    # x2 += 0.001
    # # grad of x2 = 5

    # n2 = x1*w1+ x2*w2 + x3*w3 + b
    
    # print(n)
    # print((n2 - n) / 0.001)

class Neuron:
    def __init__(self, n: int):
        self.b = Value(random.uniform(-1,1), 'b')
        self.ws = [Value(random.uniform(-1,1), f"w{i}") for i in range(n)]

    def run(self, xs: list[Value]):
        assert len(xs) == len(self.ws)
        mults = [op("*", [x,w]) for x,w in zip(xs, self.ws)]
        all = op("+", mults +[self.b])
        
        out = op("tanh", [all])
        return out    

class Layer:
    def __init__(self, n_input:int, n:int):
        self.neurons = [Neuron(n_input) for i in range(n)]

    def run(self, xs: list[Value]):
        out = [neuron.run(xs) for neuron in  self.neurons]
        return out
    

class NN:
    def __init__(self, n_input: int, layers: list[int]):
        all = [n_input] + layers
        self.layers = [Layer(all[i], all[i+1]) for i in range(len(layers))]

    def run(self, xs: list[Value]):
        for l in self.layers:
            xs = l.run(xs)
        
        # for simplicity
        assert len(xs) ==1
        return xs[0]
    

def show_nn():
    # n = Neuron(3)
    
    xs = [Value(d, f"x{i}") for i, d in enumerate([1,2,3])]
    
    # o = n.run(xs)
    
    # calc_grad(o)
    
    # draw_value(o)
    
    nn = NN(3, [4, 4, 1])
    
    o = nn.run(xs)
    
    calc_grad(o)
    draw_value(o)
    
    
def train_nn():
    xs = [
        [Value(d, f"x{i}") for (i,d) in enumerate([1,2,3])],
        [Value(d, f"x{i}") for (i,d) in enumerate([4,2,1])],
        [Value(d, f"x{i}") for (i,d) in enumerate([1,52,3])],
        [Value(d, f"x{i}") for (i,d) in enumerate([13,2,32])]
    ]
    
    y_real = [Value(d, "y") for d in [0.1, 0, 1, 1]]

    nn = NN(3, [4, 4, 1])
    
    for i in range(10000):
        y_pred = [nn.run(x) for x in xs]
        
        losses = [op("-", [y1,y2]) for y1,y2 in zip(y_real, y_pred)]
        
        losses_sq = [op("*", [l, l]) for l in losses]
        
        loss = op("+", losses_sq)
        
        calc_grad(loss)
        
        # draw_value(loss)
        print([y.data for y in y_pred])
        # print(loss.data)
        
        params:list[Value] = []
        for l in nn.layers:
            for n in l.neurons:
                params += n.ws
                params += [n.b]

        for p in params:
            p.data -= p.grad * 0.01 # learning rate
    
    
    
    
    # calc_grad(o)
    # draw_value(o)

def main():
    # show_grad()
    
    # show_nn()

    train_nn()

if __name__ == "__main__":
    main()
