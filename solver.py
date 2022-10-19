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
targets = [1, 0, 3]

w0 = np.array([1, 2])
b0 = np.array([1])

p1 = np.array([[1], [2]])
p2 = np.array([[-1], [1]])
p3 = np.array([[0], [-1]])

weights.append(w0)
inputs.append(p1)
inputs.append(p2)
inputs.append(p3)
biases.append(b0)

numberOfIterations = len(inputs)

# TODO: define MSE to stop the algorithm
# HINT: may the disicion boundary not
# classify correctly at one scope

MSE = 1

while MSE:
    for i in range(0, numberOfIterations):
        output = a(weights[i], inputs[i], biases[i])
        outputs.append(output)

        error = targets[i] - outputs[i]
        errors.append(error)

        newWeight = weights[i] + (errors[i]*inputs[i]).transpose()
        newBias = biases[i] + errors[i]

        weights.append(newWeight)
        biases.append(newBias)

print(weights, biases)  





