import numpy as np
import matplotlib.pyplot as plt
from NN import NeuralNetwork

# data
X = np.array(
            [
              [0, 0, 0],
              [1, 0, 0], 
              [0, 1, 0],
              [0, 0 , 1],
              [1, 1, 0], 
              [1, 0, 1], 
              [0, 1, 1], 
              [1, 1, 1], 
            ]
        )

# outputs

y = np.array(
            [
              [0],
              [0], 
              [0], 
              [0],
              [1], 
              [1], 
              [1], 
              [0],
            ]
        )

layer_dims = [3, 8, 8, 4, 1]

activations = ["relu", "sigmoid", "tanh", "sigmoid"]

nn = NeuralNetwork(layer_dims = layer_dims, activations = activations, learning_rate = 0.5, seed = 30)

nn.train(X, y, epochs = 10000)

print("Probablities:", nn.predict_prob(X))

print("Predicted labels:", nn.prob_to_labels(X))

print("Actual Data:", y)

print("Accuracy:", nn.accuracy(X,y))

print("Gradient Check:", nn.gradient_check(X, y))

print("Loss Plot:")
plt.plot(
        range(0, len(nn.loss_values)), 
        nn.loss_values, 
        color = "red"
        )
plt.xlabel("Epoch (per 10000)")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

