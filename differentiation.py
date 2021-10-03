import torch
import numpy as np
import matplotlib.pyplot as plt


# # Create a tensor x
#
# x = torch.tensor(3.0, requires_grad = True)
# print("The tensor x: ", x)
#
# # Create a tensor y according to y = x^2
#
# y = x**2 + 2*x + 1
# print("The result of y = x^2: ", y)
#
# print(y.backward())
# print(x.grad)
#
#
# print("\n\n\n")
#
# u = torch.tensor(1., requires_grad=True)
# v = torch.tensor(2., requires_grad=True)
# f = u*v + u**2
# f.backward()
# print(f"u grad = {u.grad}\nv grad = {v.grad}")
#
#
# print("\n\n\n")

# Calculate the derivative with multiple values

# x = torch.linspace(-10, 10, 10, requires_grad = True)
# Y = x ** 2
# y = torch.sum(x ** 2)
#
# y.backward()
#
# plt.plot(x.detach().numpy(), Y.detach().numpy(), label = 'function')
# plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label = 'derivative')
# plt.xlabel('x')
# plt.legend()
# plt.show()


# Take the derivative of Relu with respect to multiple value. Plot out the function and its derivative

x = torch.linspace(-10, 10, 10000, requires_grad = True)
Y = torch.relu(x)
y = Y.sum()
y.backward()
plt.plot(x.detach().numpy(), Y.detach().numpy(), label = 'function')
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label = 'derivative')
plt.xlabel('x')
plt.legend()
plt.show()