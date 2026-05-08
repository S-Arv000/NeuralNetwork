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

# 

nn = NeuralNetwork(input_size = 3, hidden_size = 8, output_size = 1, learning_rate = 0.1, seed = 30)

nn.forward(X)

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
plt.xlabel("Epoch (per 1000)")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

