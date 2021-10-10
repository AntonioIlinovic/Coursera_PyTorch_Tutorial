import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import torch
from torch.utils.data import Dataset, DataLoader


# class for plotting the diagrams
class PlotErrorSurfaces(object):

    # Constructor
    def __init__(self, w_range, b_range, X, Y, n_samples=30, go=True):
        W = np.linspace(-w_range, w_range, n_samples)
        B = np.linspace(-b_range, b_range, n_samples)
        w, b = np.meshgrid(W, B)
        Z = np.zeros((n_samples, n_samples))
        count1 = 0
        self.y = Y.numpy()
        self.x = X.numpy()
        min_Z = None
        min_Z_coordinates_values = None
        for w1, b1 in zip(w, b):
            count2 = 0
            for w2, b2 in zip(w1, b1):
                Z_count1_count2 = np.mean((self.y - (w2 * self.x + b2)) ** 2)
                Z[count1, count2] = Z_count1_count2
                if min_Z is None or Z_count1_count2 < min_Z:
                    min_Z = Z_count1_count2
                    min_Z_coordinates_values = (w2, b2)

                count2 += 1
            count1 += 1
        self.Z = Z
        self.w = w
        self.b = b
        self.W = []
        self.B = []
        self.LOSS = []
        self.n = 0
        if go:
            plt.figure("Loss Surface", figsize=(7.5, 5))
            plt.axes(projection='3d').plot_surface(self.w, self.b, self.Z, rstride=1, cstride=1, cmap='viridis',
                                                   edgecolor='none', alpha=0.8)
            plt.title('Loss Surface')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.plot(min_Z_coordinates_values[0], min_Z_coordinates_values[1], min_Z, 'bo')
            plt.show()
            plt.figure("Loss Surface Contour")
            plt.title('Loss Surface Contour')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.contour(self.w, self.b, self.Z)
            plt.plot(min_Z_coordinates_values[0], min_Z_coordinates_values[1], 'bo')
            plt.show()

    # Setter
    def set_para_loss(self, W, B, loss):
        self.n = self.n + 1
        self.W.append(W)
        self.B.append(B)
        self.LOSS.append(loss)

    # Plot diagram
    def final_plot(self):
        ax = plt.axes(projection='3d')
        ax.plot_wireframe(self.w, self.b, self.Z)
        ax.scatter(self.W, self.B, self.LOSS, c='r', marker='x', s=200, alpha=1)
        plt.figure()
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c='r', marker='x')
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()

    # Plot diagram
    def plot_ps(self):
        plt.subplot(121)
        plt.ylim()
        plt.plot(self.x, self.y, 'ro', label="training points")
        plt.plot(self.x, self.W[-1] * self.x + self.B[-1], label="estimated line")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Data Space Iteration: ' + str(self.n))
        plt.legend()
        plt.subplot(122)
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c='r', marker='x')
        plt.title('Loss Surface Contour')
        plt.xlabel('w')
        plt.ylabel('b')
        plt.legend()
        plt.show()


torch.manual_seed(1)

X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = 5 * X - 3
Y = f + 0.1 * torch.randn(X.size())

# plot the lina and the data
plt.figure("The function line and the data")
plt.plot(X.numpy(), Y.numpy(), 'rx', label='y')
plt.plot(X.numpy(), f.numpy(), label='f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

get_surface = PlotErrorSurfaces(10, 10, X, Y, n_samples=30, go=False)


def forward(x):
    return w * x + b


def criterion(yhat, y):
    return torch.mean((yhat - y) ** 2)


# Define the function for training model
plt.figure("Training epochs")
w = torch.tensor(-15.0, requires_grad=True)
b = torch.tensor(-10.0, requires_grad=True)
lr = 0.1
LOSS_BGD = []


def train_model_BGD(epochs):
    for epoch in range(epochs):
        yhat = forward(X)
        loss = criterion(yhat, Y)
        LOSS_BGD.append(loss.item())
        get_surface.set_para_loss(w.item(), b.item(), loss.item())
        #get_surface.plot_ps()
        loss.backward()
        w.data = w.data - lr * w.grad.data
        b.data = b.data - lr * b.grad.data
        w.grad.data.zero_()
        b.grad.data.zero_()


train_model_BGD(10)

############ Stochastic Gradient Descent (SGD) with Dataset DataLoader

class Data(Dataset):

    def __init__(self):
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        self.y = 5 * X - 3
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


dataset = Data()
trainloader = DataLoader(dataset=dataset, batch_size=1)

# Create a plot_error_surfaces object.
get_surface = PlotErrorSurfaces(20, 20, X, Y, 30, go=False)

# Define train_model_SGD function
w = torch.tensor(-15.0, requires_grad=True)
b = torch.tensor(-10.0, requires_grad=True)
LOSS_SGD = []
lr = 0.1


def train_model_SGD(epochs):
    for epoch in range(epochs):
        Yhat = forward(X)
        get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), criterion(Yhat, Y).tolist())
        #get_surface.plot_ps()
        LOSS_SGD.append(criterion(forward(X), Y).tolist())
        for x, y in trainloader:
            yhat = forward(x)
            loss = criterion(yhat, y)
            get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())
            loss.backward()
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data
            w.grad.data.zero_()
            b.grad.data.zero_()
        #get_surface.plot_ps()
train_model_SGD(10)

##### Mini Batch Gradient Descent: Batch Size Equals 5
get_surface = PlotErrorSurfaces(15, 13, X, Y, 30, go = False)

dataset = Data()
trainloader = DataLoader(dataset = dataset, batch_size = 5)

# Define train_model_Mini5 function
w = torch.tensor(-15.0, requires_grad = True)
b = torch.tensor(-10.0, requires_grad = True)
LOSS_MINI5 = []
lr = 0.1

def train_model_Mini5(epochs):
    for epoch in range(epochs):
        Yhat = forward(X)
        get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), criterion(Yhat, Y).tolist())
        #get_surface.plot_ps()
        LOSS_MINI5.append(criterion(forward(X), Y).tolist())
        for x, y in trainloader:
            yhat = forward(x)
            loss = criterion(yhat, y)
            get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())
            loss.backward()
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data
            w.grad.data.zero_()
            b.grad.data.zero_()

# Run train_model_Mini5 with 10 iterations.
train_model_Mini5(10)



### Mini Batch Gradient Descent: Batch Size Equals 10
get_surface = PlotErrorSurfaces(15, 13, X, Y, 30, go = False)

dataset = Data()
trainloader = DataLoader(dataset = dataset, batch_size = 10)


# Define train_model_Mini5 function
w = torch.tensor(-15.0, requires_grad = True)
b = torch.tensor(-10.0, requires_grad = True)
LOSS_MINI10 = []
lr = 0.1

def train_model_Mini10(epochs):
    for epoch in range(epochs):
        Yhat = forward(X)
        get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), criterion(Yhat, Y).tolist())
        #get_surface.plot_ps()
        LOSS_MINI10.append(criterion(forward(X),Y).tolist())
        for x, y in trainloader:
            yhat = forward(x)
            loss = criterion(yhat, y)
            get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())
            loss.backward()
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data
            w.grad.data.zero_()
            b.grad.data.zero_()

# Run train_model_Mini5 with 10 iterations.
train_model_Mini10(10)


# Plot out the LOSS for each method

plt.plot(LOSS_BGD,label = "Batch Gradient Descent")
plt.plot(LOSS_SGD,label = "Stochastic Gradient Descent")
plt.plot(LOSS_MINI5,label = "Mini-Batch Gradient Descent, Batch size: 5")
plt.plot(LOSS_MINI10,label = "Mini-Batch Gradient Descent, Batch size: 10")
plt.legend()
plt.show()