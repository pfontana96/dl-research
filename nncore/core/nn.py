import numpy as np
import abc
from operations import *

def normalizeBatch(Z):
    """
    Normalizes batch Z:
        Z' =  (Z - mean)/standard_deviation
    """
    _, m = Z.shape
    delta = 1e-8
    mean = np.sum(Z, axis=0)/m
    std = np.sqrt(delta + np.sum(np.square(Z - mean), axis=0)/m)
    return (Z-mean)/std

class Layer(object):
    def __init__(self, n_units, activation=Sigmoid):
        assert issubclass(activation, Operation)
        self.n = n_units
        self.activation = activation
    
    def __setActivation(activation):
        assert issubclass(activation, Operation)
        self.activation = activation

# Artificial Neural Network based on Gradient Descent and its derivations
class ANN(object):
    """
    Artificial Neural Network
    """
    def __init__(self, n_inputs, n_outputs, **kwargs):
        input_act = kwargs.get('act_in', Sigmoid)
        output_act = kwargs.get('act_out', Sigmoid)
        self.layers = [Layer(n_inputs, input_act), Layer(n_outputs, output_act)]

    def addHiddenLayer(self, n_units, activation):
        m = self.layers[-2].n
        self.layers.insert(-1, Layer(n_units, activation))            

    def initParameters(self):
        self.W = []
        self.b = []

        for l in range(1, len(self.layers)):
            n = self.layers[l].n
            m = self.layers[l-1].n
            self.W.append(np.random.rand(n, m))
            self.b.append(np.zeros((n, 1)))

    def fwdProp(self, X, training=True):
        Z = np.copy(X)
        Z_list = [] # List of linear regressions results
        A_list = [Z] # List of activations results
        if __debug__:
            print("~~~~~~~~~ FWD ~~~~~~~~~")
        for l in range(len(self.W)):
            if __debug__:
                print("Layer {}\r\nW shape: {}, b shape: {}, Z shape: {}".format(l+1, self.W[l].shape, self.b[l].shape, Z.shape))
            Z = self.W[l].dot(Z) + self.b[l] # Linear Regression
            Z = normalizeBatch(Z) # Batch Normalization
            A = self.layers[l].activation.f(Z) # Activation
            Z_list.append(Z)
            A_list.append(A)

        if training:
            return (Z_list, A_list)
        else:
            return A_list[-1]
        
    @staticmethod
    def getCost(Y_predicted, Y):
        """
        Cross-Entropy for logistic regression
        """
        delta = 1e-47 # threshold to avoid log(0)
        _, m = Y.shape
        if __debug__:
            print("~~~~~~~~~ COST ~~~~~~~~~")
            print("m: {}".format(m))
            print("Max and min values predicted: {} & {}".format(np.max(Y_predicted), np.min(Y_predicted)))
            print("Max and min values true: {} & {}".format(np.max(Y), np.min(Y)))
        return -np.sum(Y * np.log(Y_predicted + delta) + (1 - Y) * np.log(1 - Y_predicted + delta))/m
        

    def backProp(self, Z_list, A_list, Y):
        """
        Back propagation: Returns the derivatives of the parameters
        """
        if __debug__:
            print("~~~~~~~~~ BWD ~~~~~~~~~")
        
        dW = []
        db = []
        _, m = Z_list[0].shape
        
        dA = -Y/A_list[-1] + (1 - Y)/(1 - A_list[-1])
        for i in reversed(range(len(self.W))):
            dZ = dA * self.layers[i].activation.bprop(Z_list[i])
            dW.insert(0, A_list[-1].dot(dZ.T)/m)
            db.insert(0, np.sum(dZ, axis=1, keepdims=True)/m)
            if __debug__:
                print("Layer {}\r\ndZ:{} | dW:{}  | db:{}".format(i, dZ.shape, dW[0].shape, db[0].shape))
            dA = self.W[i].T.dot(dZ)
        # For logistic regression, the derivate of the cross-entropy
        # equals A[N] - Y 
        # dA = A_list[-1] - Y    
        # for i in reversed(range(len(self.W))):
        #     dZ = dA * self.layers[i].activation.bprop(Z_list[i])
        #     dW.insert(0, dZ.dot(A_list[i].T)/m) # As A_list[0] = X then A_list[i] is actually A_list[l-1]
        #     db.insert(0, np.sum(dZ, axis=1, keepdims=True)/m)
        #     dA = self.W[i].T.dot(dZ)
        


        # deltas = [None] * len(self.W)
        # deltas[-1] = (A_list[-1] - Y)*self.layers[-1].activation.bprop(Z_list[-1])
        # for i in reversed(range(len(deltas)-1)):
        #     deltas[i] = (self.W[i+1].T @ deltas[i+1])*self.layers[i].activation.bprop(Z_list[i])
        # db = [(d @ np.ones((m, 1)))/m for d in deltas]
        # dW = [(d @ A_list[i].T)/m for i,d in enumerate(deltas)]

        return (dW, db)
    
    def update(self, dW, db, learning_rate):
        
        for l in range(len(self.W)):
            self.W[l] -= learning_rate*dW[l]
            self.b[l] -= learning_rate*db[l]

    def train(self, X, Y, epochs, epsilon=0.01):
        cost = []
        for epoch in range(epochs):
            Z_list, A_list = self.fwdProp(X)
            cost.append(ANN.getCost(A_list[-1], Y))
            dW, db = self.backProp(Z_list, A_list, Y)
            self.update(dW, db, epsilon)
            if (epoch+1)%100 == 0:
                print("({ep}/{tot}) Cost: {c}".format(ep=epoch+1, tot=epochs, c=cost[-1]))
        return cost
    
    def predict(self, X):
        return np.array(np.around(self.fwdProp(X, training=False)), dtype=int)

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    
    # Load data
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    X_train = X_train.T
    X_test = X_test.T
    y_train  = y_train.reshape(1, -1)
    y_test  = y_test.reshape(1, -1)

    print("Shape X_train: {} || Shape y_train: {}".format(X_train.shape, y_train.shape))
    # Training NN
    nn = ANN(30, 1, act_in=Sigmoid)
    nn.addHiddenLayer(8, Sigmoid)
    nn.addHiddenLayer(15, Sigmoid)
    nn.initParameters()
    epochs = 1000
    learning_rate = 0.1

    cost = nn.train(X_train, y_train, epochs, learning_rate)

    # Test
    y_pred = nn.predict(X_test)
    missclass = np.count_nonzero(np.abs(y_test - y_pred))
    print("Test error: {error} %".format(error=100*missclass/len(y_pred)))

    plt.plot(np.array(range(1, epochs+1)), cost)
    plt.show()