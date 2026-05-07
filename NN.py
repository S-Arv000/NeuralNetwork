import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 0.1, seed = 50):
        """
        (# input features, # hidden neursons, # output neurons)
        
        """
        rng = np.random.default_rng(seed)

        # Weight of inpurt to hidden
        # input_size inputs to hidden_size neurons
        # W1 = input_size x hidden_size
        self.W1 = rng.normal(
            loc = 0,
            scale = 2/np.sqrt(input_size), 
            size = (input_size, hidden_size)
            )
        
        # bias 1 
        # b1 ia 1 x hidden_size
        self.b1 = np.zeros((1, hidden_size))

        # Weight of hidden to output
        # hidden_size hiddern neurons to output_size neurons
        # W2 = hidden_size x output_size
        self.W2 = rng.normal( 
            loc = 0,
            scale = 2/np.sqrt(hidden_size),
            size = (hidden_size, output_size)
             )
        
        # bias 2 
        # b2 is 1 x output_size
        self.b2 = np.zeros((1, output_size))

        self.learning_rate = learning_rate

    # TRANSFORMATIONS

    def ReLU(self, x):
        
        return np.maximum(0, x)
    
    def ReLU_derivative(self, x):
        
        return np.where(x>0, 1, 0)
    
    def sigmoid(self, x):
        
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        i = self.sigmoid(x)
        
        return i * (1 - i)
    
    # FORWARD PROPOGATION 
    def forward(self, X):
        # map input to hidden
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.ReLU(self.Z1)

        # map hidden to output
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.sigmoid(self.Z2)

        return self.A2

    # BACKWARD PROPOGATION
    def backward(self, X, y_true):

        m = X.shape[0] # numper of inputs

        # - output to hidden -
        # Gradient loss w.t.r output Z2
        dZ2 = (self.A2 - y_true)/ m 

        # Gradient loss w.t.r W2 and b2
        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis = 0 , keepdims = True)

        # Gradient loss w.t.r hidden layer A1
        dA1 = dZ2 @ self.W2.T

        # - hidden to input -
        # Gradient loss w.t.r hidden layer Z1
        dZ1 = dA1 * self.ReLU_derivative(self.Z1)

        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis = 0, keepdims = True)

        return dW2, db2, dW1, db1  

    # LOSS FINCTION
    def compute_loss(self, y_true, y_predicted):
        
        Loss = -np.mean((y_true * np.log(y_predicted)) + (1 - y_true) * np.log(1 - y_predicted))
        
        return Loss

    # Training loop
    def train(self, X, y_true, epochs = 10000, track_value = 1000):
        # forward -> calc. loss -> backward -> update -> forward -> ......

        self.loss_values = []

        for epoch in range(epochs + 1):
            
            y_predicted = self.forward(X)

            loss = self.compute_loss(y_true, self.A2)

            dW2, db2, dW1, db1 = self.backward(X, y_true)

            self.W2 = self.W2 - self.learning_rate * dW2
            self.b2 = self.b2 - self.learning_rate * db2
            self.W1 = self.W1 - self.learning_rate * dW1
            self.b1 = self.b1 - self.learning_rate * db1


            
            # track loss
            if epoch % track_value == 0:
                print(f"Epoch: {epoch}, Loss: {loss}")
                self.loss_values.append(loss)


    
    def predict_prob(self, X):

        prob = self.forward(X)

        return prob
    
    def prob_to_labels(self, X):

        prob = self.predict_prob(X)

        label = np.where(prob > 0.5, 1, 0)

        return label 
    
    def accuracy(self, X, y_true): 

        y_predicted = self.prob_to_labels(X)
        accuracy = np.mean(y_predicted == y_true)

        return accuracy


