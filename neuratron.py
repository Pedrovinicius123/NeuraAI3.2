import torch
import torch.nn as nn
import torch.optim as optim
        
class Neuratron(nn.Module):
    def __init__(self, shape:tuple):
        super().__init__(self)
        self.alloc_mtrx = nn.Linear(*shape)
        self.alloc_mtrx.requires_grad = False
        self.alloc_map = {}
        self.not_allocated = [] 

    def allocate_memory(self, shape:tuple, rotule:str):
        allocated = None
        if not any(self.alloc_map):
            allocated = self.alloc_mtrx[:shape[0], :shape[1]]
            self.not_allocated.append(self.allocated[shape[0]:, :shape[1]])
            self.not_allocated.append(self.allocated[:shape[0], shape[1]:])
            self.not_allocated.append(self.allocated[shape[0]:, shape[1]:])

        else:
            for i, alloc in enumerate(self.not_allocated):
                if not(shape[0] > alloc.shape[0] or shape[1] > alloc.shape[1]):
                    allocated = alloc[:shape[0], :shape[1]]
                    self.not_allocated.pop(i)
                    self.not_allocated.append(alloc[shape[0]:, :shape[1]])
                    self.not_allocated.append(alloc[:shape[0], shape[1]:])
                    self.not_allocated.append(alloc[shape[0]: shape[1]:])

                    break

        allocated.requires_grad = False
        self.alloc_map[rotule] = allocated

    def forward(self, X:np.ndarray, rotule:str):
        self.X = torch.from_numpy(X)
        if rotule not in self.alloc_map:
            self.allocate_memory(*X.shape, rotule)

        return self.alloc_map[rotule](self.X)

    def backward(self, rotule:str, out:np.ndarray=None, Y:np.ndarray=None, grad=None):
        if grad is None and out and Y:
            self.alloc_map[rotule].requires_grad = True

            loss = (2/n) * (out - Y)
            loss.backward()

            self.alloc_map[rotule].weight -= self.lr * self.alloc_map[rotule].weight.grad
            self.alloc_map[rotule].bias -= self.lr * self.alloc_map[rotule].bias.grad
            self.alloc_map[rotule].requires_grad = False

            return self.X.grad

        elif grad:
            self.alloc_map[rotule].requires_grad = True
            self.alloc_map[rotule].backward(gradient=grad)

            self.alloc_map[rotule].weight -= self.lr * self.alloc_map[rotule].weight.grad
            self.alloc_map[rotule].bias -= self.lr * self.alloc_map[rotule].bias.grad

            self.alloc_map[rotule].requires_grad = False
            return self.X.grad
            