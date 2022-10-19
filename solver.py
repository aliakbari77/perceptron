from audioop import bias
import numpy as np

def hardlim(n):
    if (n.any() >= 0):
        return 1
    return 0

def hardlims(n):
    if (n >= 0):
        return 1
    return -1

a = lambda w, p, b: hardlim(np.dot(w, p) + b) 

weights = []
inputs = []
biases = []
outputs = []
errors = []
targets = [1, 0]

w_0 = np.array([1, 2])
b_0 = np.array([1])

p1 = np.array([[1], [2]])
p2 = np.array([[-1], [1]])

weights.append(w_0)
inputs.append(p1)
inputs.append(p2)
biases.append(b_0)

numberOfIterations = len(inputs)

for i in range(0, numberOfIterations):
    output = a(weights[i], inputs[i], biases[i])
    outputs.append(output)

    error = targets[i] - outputs[i]
    errors.append(error)

    newWeight = weights[i] + (errors[i]*inputs[i]).transpose()
    newBias = biases[i] + errors[i]

    weights.append(newWeight)
    biases.append(newBias)

print(weights, errors)  





