import numpy as np
import time

class Model:
    def __init__(self, shape:tuple=(100,100), lr:float=0.001, is_output:bool=False):
        self.shape = shape
        self.lr = lr
        self.is_output = is_output
        self.is_processed = False

        self.total = np.random.rand(*shape)
        self.total_bias = np.sum(self.total, axis=0)

        self.using_weight = self.total
        self.using_bias = self.total_bias

    def feed_forward(self, X, output_shape:int):
        self.X = X
        input_shape = X.shape[1]

        self.using_weight = self.total[:input_shape, :output_shape]
        self.using_bias = self.total_bias[:output_shape]
        self.output = np.dot(X, self.using_weight) + self.using_bias
        
        return self.output

    def backward_pass(self, grad=None):
        if grad is None:
            grad = 2 * np.mean(-self.output, axis=0)
            X_refited = np.array([np.sum(self.X, axis=0)])
            grad = X_refited.T * grad

        else:
            print(grad.shape, self.using_weight.shape, self.X.shape)
            g = np.dot(self.X, self.using_weight)
            grad = np.dot(g, grad)
        
        grad = np.clip(grad, -10, 10)
        self.using_weight -= np.dot(self.using_weight.T, grad) * self.lr
        self.using_bias = np.dot(self.using_bias, grad) * self.lr if len(grad.shape) > 1 else self.using_bias * grad * self.lr

        self.total[:self.using_weight.shape[0], :self.using_weight.shape[1]] = self.using_weight
        self.total_bias[:self.using_bias.shape[0]] = self.using_bias

        grad = np.dot(grad, self.using_weight)
        return grad
