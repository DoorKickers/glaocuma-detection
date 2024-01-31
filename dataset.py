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

class AugmentedImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(AugmentedImageFolder, self).__init__(root, transform, target_transform)
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            augmented_samples = [self.transform(sample) for _ in range(5)]  # 创建5个增强样本
            return augmented_samples, target

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


transform = Compose([
   Resize((224, 224)),  # Resize the images to a specific size
   ToTensor(),  # Convert the images to tensors
])
transform2 = transforms.Compose([
    transforms.RandomRotation((-30, 30)),  # 随机旋转 -30度~30度
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomVerticalFlip(),  # 随机竖直翻转
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.2)),  # 随机缩放 0.8倍~1.2倍并随机裁剪到指定大小
    transforms.ToTensor(),  # 转换为张量
])

train_dir = "/root/workspace_remote/data/RIM-ONE_DL_images/partitioned_by_hospital/training_set"
test_dir = "/root/workspace_remote/data/RIM-ONE_DL_images/partitioned_by_hospital/test_set"
# train_dataset = ImageFolder(root = train_dir, transform = transform2)
test_dataset = ImageFolder(root = test_dir, transform = transform)
train_dataset = AugmentedImageFolder(root=train_dir, transform=None)
augmented_dataset = []
for image, label in train_dataset:
    augmented_dataset.append((transform(image), label))
    for _ in range(5):
        augmented_dataset.append((transform2(image), label))
        
# image, _ = train_dataset[0]
# image = transforms.functional.to_pil_image(image)
# image.save('preview_transform_dataset.jpg')

torch.save(augmented_dataset, 'trans_train_dataset.pth')
sys.exit(0)
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

image, _ = extended_dataset[0]
image = transforms.functional.to_pil_image(image)
image.save('preview_transform_dataset.jpg')

# torch.save(extended_dataset, 'ex_train_dataset.pth')
# torch.save(test_dataset, 'test_dataset.pth')
