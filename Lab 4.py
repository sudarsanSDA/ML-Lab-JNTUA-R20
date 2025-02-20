import numpy as np
import pandas as pd

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Load the data
data = pd.read_csv("training_data_task4.csv")

# Input features and target labels
X = data[["Input1", "Input2"]].values
y = data["Output"].values.reshape(-1, 1)

# Initialize network parameters
input_layer_neurons = 2  # Number of input neurons
hidden_layer_neurons = 2  # Number of hidden neurons
output_neurons = 1  # Number of output neurons

# Weights and biases initialization
np.random.seed(42)
hidden_weights = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
hidden_bias = np.random.uniform(size=(1, hidden_layer_neurons))
output_weights = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
output_bias = np.random.uniform(size=(1, output_neurons))

# Hyperparameters
learning_rate = 0.1
epochs = 100000

# Training the neural network
for epoch in range(epochs):
    # Forward pass
    hidden_layer_activation = np.dot(X, hidden_weights) + hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, output_weights) + output_bias
    predicted_output = sigmoid(output_layer_activation)

    # Backward pass
    error = y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    hidden_weights += X.T.dot(d_hidden_layer) * learning_rate
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    # Print the error every 1000 epochs
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch + 1}, Error: {np.mean(np.abs(error))}")

# Final predicted output
print("\nFinal Predicted Output:")
print(np.round(predicted_output, 2))
