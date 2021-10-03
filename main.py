import torch
import numpy as np

float_tensor = torch.FloatTensor([0, 1, 2, 3, 4])
print(f"tensor created with #FloatTensor has type: float_tensor.type() = {float_tensor.type()}\n")

tensor_a = torch.tensor([0, 1, 2, 3, 4])
print(f"tensor_a type before converting type: {tensor_a.type()}")
tensor_a = tensor_a.type(torch.FloatTensor)
print(f"tensor_a type after converting:  tensor_a.type() = {tensor_a.type()}\n")


one_D_tensor = torch.tensor([1, 2, 3, 4, 5])
print(f"one_D_tensor.size() = {one_D_tensor.size()}")
print(f"one_D_tensor.ndimension() = {one_D_tensor.ndimension()}\n")

tensor_b = torch.Tensor([1, 2, 3, 4, 5])
print(f"tensor_b before converting view:\nshape = {tensor_b.shape}\nndimension = {tensor_b.ndimension()}\n")
tensor_b_col = tensor_b.view(5, 1)
# tensor_b_col = tensor_b.view(-1, 1)
print(f"tensor_b_col after converting view:\nshape = {tensor_b_col.shape}\nndimension = {tensor_b_col.ndimension()}\n")

tensor_six_elem = torch.Tensor([0, 1, 2, 3, 4, 5])
# tensor_six_elem_col = tensor_six_elem.view(6, 1)
tensor_six_elem_col = tensor_six_elem.view(-1, 1)

numpy_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
torch_tensor = torch.from_numpy(numpy_array)
back_to_numpy = torch_tensor.numpy()

new_tensor = torch.tensor([5, 2, 6, 1])
print(new_tensor[0])
print(new_tensor[0].item())
print(new_tensor[-1])


