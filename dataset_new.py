import sys
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

# torch.save(extended_dataset, 'accredited_extended_dataset.pth')

