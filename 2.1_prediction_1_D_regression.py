import torch

# Define w = 2 and b = -1 for y = wx + b
b = torch.tensor(-1.0, requires_grad=True)
w = torch.tensor(2.0, requires_grad=True)


# Function forward(x) for prediction
def forward(x):
    y_hat = w * x + b
    return y_hat


x = torch.tensor([[1.0],
                  [2.0],
                  [3.0]])
y_hat = forward(x)
print("The prediction is: ", y_hat)

from torch.nn import Linear

torch.manual_seed(0)

# Create Linear Regression Model, and print out the parameters
lr = Linear(in_features=1, out_features=1, bias=True)
print(f"Parameters w and b: {list(lr.parameters())}")

print(f"Python dictionary: {lr.state_dict()}")
print(f"keys: {lr.state_dict().keys()}")
print(f"values: {lr.state_dict().values()}")

print(f"weight: {lr.weight}")
print(f"bias: {lr.bias}")

x = torch.tensor([[1.0],
                  [2.0]])
yhat = lr(x)
print(f"The prediction: {yhat}")

# Build Custom Modules
from torch import nn


# Customize Linear Regression Class
class LR(nn.Module):

    def __init__(self, input_size, output_size):
        # Inherit from parent
        super(LR, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out


lr = LR(input_size=1, output_size=1)
print(f"The parameters:\n{list(lr.parameters())}")
print(f"Linear model: {lr.linear}")


x = torch.tensor([[1.0]])
yhat = lr(x)
print(f"the prediction is: {yhat}")

print("Python dictionary: ", lr.state_dict())
print("keys: ",lr.state_dict().keys())
print("values: ",lr.state_dict().values())