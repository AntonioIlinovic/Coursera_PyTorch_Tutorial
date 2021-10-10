import numpy as np
import matplotlib.pyplot as plt
import torch


# The class for plotting

class plot_diagram():

    # Constructor
    def __init__(self, X, Y, w, stop, go=False):
        start = w.data
        self.error = []
        self.parameter = []
        self.X = X.numpy()
        self.Y = Y.numpy()
        self.parameter_values = torch.arange(start, stop)
        self.Loss_function = [criterion(forward(X), Y) for w.data in self.parameter_values]
        w.data = start

    # Executor
    def __call__(self, Yhat, w, error, n):
        self.error.append(error)
        self.parameter.append(w.data)
        plt.subplot(212)
        plt.plot(self.X, Yhat.detach().numpy())
        plt.plot(self.X, self.Y, 'ro')
        plt.xlabel("A")
        plt.ylim(-20, 20)
        plt.subplot(211)
        plt.title("Data Space (top) Estimated Line (bottom) Iteration " + str(n))
        plt.plot(self.parameter_values.numpy(), torch.tensor(self.Loss_function).numpy())
        plt.plot(self.parameter, self.error, 'ro')
        plt.xlabel("B")
        plt.figure()
        plt.show()

    # Destructor
    def __del__(self):
        plt.close('all')


# Create the f(X) with a slope of -3
X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = -3 * X

# Plot the line with blue
plt.plot(X.numpy(), f.numpy(), label='f')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

Y = f + 0.1 * torch.randn(X.size())

# Plot the data points (with noise)
plt.plot(X.numpy(), Y.numpy(), "rx", label=Y)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

w = torch.tensor(-10.0, requires_grad=True)
# Create Learning Rate and an empty list to record the loss for each iteration
lr = 0.1
LOSS = []


# Create forward function for prediction
def forward(x):
    return w * x


# Create the MSE function to evaluate the result
def criterion(Yhat, y):
    return torch.mean((Yhat - y) ** 2)


gradient_plot = plot_diagram(X, Y, w, stop=10)

# Define a function for train the model
def train_model(iter):
    for epoch in range(iter):
        # Predict the values
        Yhat = forward(X)

        # Calculate the loss of this iteration
        loss = criterion(Yhat, Y)

        # Plot the diagram for us to have a better idea
        gradient_plot(Yhat, w, loss.item(), epoch)

        # store the loss to list
        LOSS.append(loss.item())

        # backward pass: compute gradient of the loss with respect to all the learnable parameters
        loss.backward()

        # update parameters
        w.data = w.data - lr * w.grad.data

        # zero the gradients before running the backward pass
        w.grad.data.zero_()


train_model(4)

plt.plot(LOSS)
plt.tight_layout()
plt.xlabel("Epoch/Iterations")
plt.ylabel("Cost")
plt.show()
