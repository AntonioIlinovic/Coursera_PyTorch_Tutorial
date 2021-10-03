import matplotlib
import torch
import numpy as np
import matplotlib.pyplot as plt

x = torch.linspace(0, 2*np.pi, 100)
y = torch.sin(x)

plt.plot(x, y)
plt.plot(x.numpy(), y.numpy())
plt.show()
