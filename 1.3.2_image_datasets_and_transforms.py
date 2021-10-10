import os

import matplotlib.pylab as plt
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

torch.manual_seed(0)


def show_data(data_sample, shape=(28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title("y = " + data_sample[1])
    plt.show()


# Read CSV file from the URL and print out the first five samples
directory = "./resources/data"
csv_file = "index.csv"
csv_path = os.path.join(directory, csv_file)

data_name = pd.read_csv(csv_path)
print(data_name.head())

# Get the value on location row 0, column 1 (Notice that index starts at 0)
# remember this dataset has only 100 samples to make the download faster
print("File name: ", data_name.iloc[0, 1])

# Get the value on Location row 0, column 0
print("y: ", data_name.iloc[0, 0])

# Print out the file name and the class number of the element on row 1
print("File name: ", data_name.iloc[1, 1])
print("class or y: ", data_name.iloc[1, 0])

# Printout the total number of rows in training dataset
print("The number of rows: ", data_name.shape[0])

# Combine the directory path with file name
image_name = data_name.iloc[1, 1]
image_path = os.path.join(directory, image_name)
print(image_path)

# Plot the second training image
image = Image.open(image_path)
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title(data_name.iloc[1, 0])
plt.show()


# Create your own dataset object
class MyDataset(Dataset):

    def __init__(self, csv_file, data_dir, transform=None):
        # Image directory
        self.data_dir = data_dir

        # The transform is going to be used on image
        self.transform = transform

        data_dir_csv_file = os.path.join(self.data_dir, csv_file)

        # Load the CSV file contains image info
        self.data_name = pd.read_csv(data_dir_csv_file)

        # Number of images in dataset
        self.len = self.data_name.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # Image file path
        image_name = os.path.join(self.data_dir, data_name.iloc[index, 1])

        # Open image file
        image = Image.open(image_name)

        # The class label for the image
        image_label = data_name.iloc[index, 0]

        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, image_label


# Create the dataset object
dataset = MyDataset(csv_file=csv_file, data_dir=directory)

sample_image = dataset[0][0]
sample_image_label = dataset[0][1]

plt.imshow(sample_image, cmap="gray")
plt.title(sample_image_label)
plt.show()


# Combine two transforms: crop and convert to tensor. Apply the compose to MNIST dataset
crop_tensor_data_transform = transforms.Compose([transforms.CenterCrop(20), transforms.ToTensor()])
dataset = MyDataset(csv_file=csv_file, data_dir=directory, transform=crop_tensor_data_transform)
print("The shape of the first element tensor: ", dataset[0][0].shape)

sample_image, sample_label = dataset[0]
show_data(dataset[0], (20, 20))

vertical_flip_horizontal_flip_transform = \
    transforms.Compose(
        [transforms.RandomVerticalFlip(p=1), transforms.RandomHorizontalFlip(p=1), transforms.ToTensor()])

practice_dataset = MyDataset(csv_file=csv_file, data_dir=directory, transform=vertical_flip_horizontal_flip_transform)

show_data(practice_dataset[0])
