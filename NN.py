from networkx import difference
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layer_dims, activations, learning_rate = 0.1, seed = 50):
        """        
    
        Layer_dims = [input_size, Hidden_size1, hidden_size2,...., output_size]

        activations = [hidden1_activation, hiddne2_activation....]
        example: activations = ["relu", "sigmoid", "tanh"]
                    ^ no other activations yet
        
        len(activations) = len(layer_dims) - 1 

        use cache for backprob.          
        """
    
        self.layer_dimensions = layer_dims
        self.activations = activations
        self.learning_rate = learning_rate
        
        self.weights = []
        self.biases = []
        
        self.loss_values = []

        rng = np.random.default_rng(seed)

        for i in range(len(layer_dims) - 1):
            input_size = layer_dims[i]
            output_size = layer_dims[i + 1]

            activation = activations[i]

        # Weight of inpurt to hidden
        # input_size inputs to hidden_size neurons
            W = rng.normal(
                loc = 0,
                scale = 2/np.sqrt(input_size), 
                size = (input_size, output_size)
                )
            
            self.weights.append(W)

            # bias  
            # b is 1 x hidden_size
            self.biases.append(np.zeros((1, output_size)))
            


    #----------------- TRANSFORMATIONS-----------------------

    def ReLU(self, x):
        
        return np.maximum(0, x)
    
    def ReLU_derivative(self, x):
        
        return np.where(x>0, 1, 0)
    
    def sigmoid(self, x):
        
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        x = self.sigmoid(x)
        
        return x * (1 - x)
    
    def tanh(self, x):

        return np.tanh(x)
    
    def tanh_derivative(self, x):
        x = self.tanh(x)

        return 1-x**2
    


    def activation(self, z, func):

        if func == "relu":
            return self.ReLU(z)
        elif func == "sigmoid":
            return self.sigmoid(z)
        elif func == "tanh":
            return self.tanh(z)
        else:
            raise ValueError(f"Unsupported activation: {func}")
    
    def activation_derivative(self, z, func):

        if func == "relu":
            return self.ReLU_derivative(z)
        elif func == "sigmoid":
            return self.sigmoid_derivative(z)
        elif func == "tanh":
            return self.tanh_derivative(z)
        else:
            raise ValueError(f"Unsupported activation: {func}")

    #------------------------------------------------------------------

    # FORWARD PROPOGATION 
    def forward(self, X):

        # first A is inputs
        A = X

        self.cache = []
        # map input to hidden

        for i in range(len(self.layer_dimensions)-1):
            W = self.weights[i]
            b = self.biases[i]
            activation_function = self.activations[i]

            A_prev = A

            Z = A_prev @ W + b
            A = self.activation(z = Z, func = activation_function)

            # update cache
            self.cache.append({"A_prev" : A_prev, 
                               "W" : W, 
                               "b" : b, 
                               "Z": Z, 
                               "A": A,
                               "activation_function": activation_function})
            
        return A

    # BACKWARD PROPOGATION
    def backward(self, X, y_true):

        m = y_true.shape[0] 
        gradient_W = [None] * (len(self.layer_dimensions) - 1)
        gradient_b = [None] * (len(self.layer_dimensions) - 1)

        A_out = self.forward(X)

        # output to hideen
        dZ = (A_out - y_true) / m

        for i in reversed(range(len(self.weights))):
            cache = self.cache[i]
            A_prev = cache["A_prev"]
            W = cache["W"]
            b = cache["b"]
            Z = cache["Z"]
            A = cache["A"]
            activation_function = cache["activation_function"]

            # hidden to hidden to hidden to ..... to input
            dZi = dZ * self.activation_derivative(Z, cache["activation_function"])
            dWi = A_prev.T @ dZi
            dbi = np.sum(dZi, axis = 0, keepdims = True)

            gradient_W[i] = dWi
            gradient_b[i] = dbi

            # if in hiddne layer, update dZ for next iteration 
            if i > 0:
                dA_prev = dZi @ W.T
                dZ = dA_prev * self.activation_derivative(self.cache[i-1]["Z"], self.cache[i-1]["activation_function"])


        return gradient_W, gradient_b 

    # LOSS FINCTION
    def compute_loss(self, y_true, y_predicted):
        
        Loss = -np.mean((y_true * np.log(y_predicted)) + (1 - y_true) * np.log(1 - y_predicted))
        
        return Loss

    # Gradient check
    def gradient_check(self, X, y_true, epsilon = 1e-5):

        self.forward(X)
        gradient_W, gradient_b = self.backward(X, y_true)

        # Check wegith W1
        original_value = self.weights[0][0, 0]

        self.weights[0][0, 0] = original_value + epsilon
        positive_loss = self.compute_loss(y_true, self.forward(X))

        self.weights[0][0,0] = original_value - epsilon
        negative_loss = self.compute_loss(y_true, self.forward(X))

        self.weights[0][0, 0] = original_value

        numerical_gradient = (positive_loss - negative_loss) / (2* epsilon)
        backdrop_gradient = gradient_W[0][0, 0] # type: ignore
        difference = (numerical_gradient - backdrop_gradient)

        return abs(difference)

    # Training loop
    def update_parameters(self, gradient_W, gradient_b):

        for i in range(len(self.weights)): 
            self.weights[i] = self.weights[i] - self.learning_rate * gradient_W[i]
            self.biases[i] = self.biases[i] - self.learning_rate * gradient_b[i]

        return self.weights, self.biases

    def train(self, X, y_true, epochs = 10000, track_value = 1000):
        # forward -> calc. loss -> backward -> update -> forward -> ......

        self.loss_values = []

        for epoch in range(epochs + 1):
            
            y_predicted = self.forward(X)

            loss = self.compute_loss(y_true, y_predicted)

            gradient_W, gradient_b = self.backward(X, y_true)

            self.update_parameters(gradient_W, gradient_b)

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


