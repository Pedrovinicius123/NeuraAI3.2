import torch
import torch.nn as nn

from neuratron import Neuratron
from sklearn.metrics import r2_score

loss = nn.MSELoss()


class MLNeuratron(nn.Module):
    def __init__(self, *neuratrons, lr:float=0.001, inner_shape:int=20):
        super().__init__(self)

        self.neuratrons = neuratrons
        self.lr = lr
        self.inner_shape = inner_shape
        

    def forward(self, X:np.ndarray, rotule:str):
        forward = torch.from_numpy(X)
        
        for layer in self.neuratrons:
            forward = layer(forward, rotule)

        return forward

    def fit(self, X:np.ndarray, y:np.ndarray, epochs:int=1000, rotule:str):
        for i in range(epochs):
            out = self.forward(X)
            grad = None

            print(f"LOSS: {loss(out, Y)}  R2_SCORE: {r2_score(out.detach().numpy(), Y.detach().numpy())}" )

            for i, neuratron in enumerate(reversed(neuratrons)):
                if i == 0:
                    grad = neuratron.backward(rotule, out=out, Y=y)
                    continue

                grad = neuratron.backward(rotule, grad=grad)           
