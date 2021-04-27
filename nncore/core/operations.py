import numpy as np
import abc

class Operation(abc.ABC):

    @abc.abstractmethod
    def f(X):
        """
        Forward propagation step
        """    

    @abc.abstractmethod
    def bprop(X):
        """
        Back propagation step (derivate)
        """


# Activation Functions
class Sigmoid(Operation):
    def f(X):
        return 1 / (1 + np.exp(-X))

    def bprop(X):
        return Sigmoid.f(X)*(1 - Sigmoid.f(X))

class Softplus(Operation):
    def f(X):
        return np.log(1 + np.exp(X))
    
    def bprop(X):
        return Sigmoid.f(X)

class Tanh(Operation):
    def f(X):
        return (np.exp(X) - np.exp(-X))/(np.exp(X) + np.exp(-X))

    def bprop(X):
        return 1 - np.power(Tanh.f(X), 2)

class Relu(Operation):
    def f(X):
        return np.maximum(0, X)
    def bprop(X):
        R = np.zeros(X.shape)
        R[X>=0] = 1
        return R

class LeakyRelu(Operation):
    def f(X):
        return np.maximum(0.01*X, X)

    def bprop(X):
        R = np.ones(X.shape)*0.01
        R[X>=0] = 1
        return R