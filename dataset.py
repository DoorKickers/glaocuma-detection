import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from model import Autoencoder, DenoisingAutoencoder
from torchvision.datasets import ImageFolder

transform = Compose([
   Resize((224, 224)),  # Resize the images to a specific size
   ToTensor(),  # Convert the images to tensors
])
train_dir = "/root/workspace_remote/data/RIM-ONE_DL_images/partitioned_by_hospital/training_set"
test_dir = "/root/workspace_remote/data/RIM-ONE_DL_images/partitioned_by_hospital/test_set"
train_dataset = ImageFolder(root = train_dir, transform = transform)
test_dataset = ImageFolder(root = test_dir, transform = transform)
extended_dataset = []

def rotate_images(image, num_rotations):
    rotated_images = []
    for angle in range(0, 360, num_rotations):
        rotated_image = transforms.functional.rotate(image, angle)
        rotated_images.append(rotated_image)
    return rotated_images

for image, label in train_dataset:
    rotated_images = rotate_images(image, 60)
    extended_dataset.extend([(rotated_image, label) for rotated_image in rotated_images])

torch.save(extended_dataset, 'ex_train_dataset.pth')
torch.save(test_dataset, 'test_dataset.pth')
