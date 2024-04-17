import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Grayscale
from torch.utils.data import DataLoader
# from model import Autoencoder, DenoisingAutoencoder
from torchvision.datasets import ImageFolder

def create_valid_dataset():
    data_dir = "/root/workspace_remote/data/RIM-ONE_DL_images/partitioned_by_hospital/training_set"
    
# Define the mean and standard deviation of the dataset
mean = [0.5, 0.5, 0.5]  # Replace with the actual mean of your dataset
std = [0.5, 0.5, 0.5]  # Replace with the actual standard deviation of your dataset

transform = Compose([
    # Grayscale(num_output_channels=1),
    Resize((224, 224)),  # Resize the images to a specific size
    ToTensor(),  # Convert the images to tensors
    # transforms.Normalize(mean, std)
])
transform2 = transforms.Compose([
    # transforms.RandomRotation((-30, 30)),  # 随机旋转 -30度~30度
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomVerticalFlip(),  # 随机竖直翻转
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.2)),  # 随机缩放 0.8倍~1.2倍并随机裁剪到指定大小
    transforms.ToTensor(),  # 转换为张量
])

train_dir = "/root/workspace_remote/data/RIM-ONE_DL_images/partitioned_by_hospital/training_set"
test_dir = "/root/workspace_remote/data/RIM-ONE_DL_images/partitioned_by_hospital/test_set"
train_dataset = ImageFolder(root = train_dir, transform = transform)
test_dataset = ImageFolder(root = test_dir, transform = transform)

def compute_blur_score2(image):
    # print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return blur_score

for i in range(40):
    img_tensor, _ = test_dataset[i]
    img = transforms.ToPILImage()(img_tensor)
    blur = compute_blur_score2(np.array(img))
    name = f"dataset_example/img_{blur}.jpg"
    img.save(name)

# torch.save(test_dataset, 'gray_test_dataset_norm.pth')
# torch.save(train_dataset, 'gray_train_dataset_norm.pth')
sys.exit(0)
extended_dataset = []

for image, label in train_dataset:
    image_flip = transforms.functional.vflip(image)
    for angle in range(0, 360, 90) :
        extended_dataset.append((transforms.functional.rotate(image, angle), label))
        extended_dataset.append((transforms.functional.rotate(image_flip, angle), label))

for i in range(6, 10):
    pil_image = transforms.ToPILImage()(extended_dataset[i][0])
    path = f"dataset_example/{i}.jpg"
    pil_image.save(path)
    


def rotate_images(image, num_rotations):
    rotated_images = []
    for angle in range(0, 360, num_rotations):
        rotated_image = transforms.functional.rotate(image, angle)
        rotated_images.append(rotated_image)
    return rotated_images

print(len(extended_dataset))

torch.save(extended_dataset, 'accredited_extended_dataset_norm.pth')

