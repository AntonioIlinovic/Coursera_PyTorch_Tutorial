import torch
from torch.utils.data import Dataset
torch.manual_seed(1)

# Define class for dataset
class toy_set(Dataset):

    # Constructor with default values
    def __init__(self, length=100, transform=None):
        self.len = length
        self.x = 2 * torch.ones(length, 2)
        self.y = torch.ones(length, 1)
        self.transform = transform

    # Getter
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    # Get Length
    def __len__(self):
        return self.len


# Create transform class add_mult
class add_mult(object):

    # Constructor
    def __init__(self, addx=1, muly=2):
        self.addx = addx
        self.muly = muly

    # Executor
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x + self.addx
        y = y * self.muly
        sample = x, y
        return sample

# # Create Dataset Object. Find out the value on index 1. Find out the length of Dataset Object.
# our_dataset = toy_set()
# print("Our toy_set object: ", our_dataset)
# print("Value on index 0 of our toy_set object: ", our_dataset[0])
# print("Our toy_set length: ", len(our_dataset))
#
#
# # Use loop to print out first 3 elements in dataset
#
# for i in range(3):
#     x, y=our_dataset[i]
#     print("index: ", i, '; x:', x, '; y:', y)
#
#
# for x,y in our_dataset:
#     print(' x:', x, 'y:', y)


# Create an add_mult transform object, and a toy_set object
a_m = add_mult()
data_set = toy_set()

# # Use loop to print out first 10 elements in dataset
# for i in range(10):
#     x, y = data_set[i]
#     print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
#     x_, y_ = a_m(data_set[i])
#     print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)

cust_data_set = toy_set(transform=a_m)
# Use loop to print out first 10 elements in dataset

for i in range(10):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = cust_data_set[i]
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)


from torchvision import transforms


class mult(object):

    def __init__(self, mult=100):
        self.mult = mult

    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x * self.mult
        y = y * self.mult
        sample = x, y
        return sample

# Combine the add_mult() and mult()
data_transform = transforms.Compose([add_mult(), mult()])
print("The combination of transforms (Compose): ", data_transform)

print(data_transform(data_set[0]))
















