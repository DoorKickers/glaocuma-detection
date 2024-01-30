import random
import numpy as np
import torch
import torch.fft as fft
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision import transforms
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import cv2
from torch.optim.lr_scheduler import StepLR
from PIL import Image

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# prefix_dir = "../data/RIM-ONE_DL_images/partitioned_randomly/"
# training_dir = prefix_dir + "/training_set"
# test_dir = prefix_dir + "/test_set"
training_dir = "../data/RIM-ONE_DL_images/partitioned_by_hospital/training_set"
test_dir = "../data/RIM-ONE_DL_images/partitioned_by_hospital/test_set"
data_dir = "../data/RIMONE-db-r2/"

def tensor_to_image(tensor):
    # 反转换标准化
    transform = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=[1/0.5, 1/0.5, 1/0.5]),
        transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1, 1, 1]),
        transforms.ToPILImage()
    ])
    image = transform(tensor)
    return image

transform = Compose([
   Resize(224),  # Resize the images to a specific size
   # ToTensor(),  # Convert the images to tensors
   # Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

training_dataset = ImageFolder(root = training_dir, transform = transform)
test_dataset = ImageFolder(root = test_dir, transform = transform)
noised_dataset = []


def add_noise(image, noise_type="gaussian", mean=0, std=0.1):
    if noise_type == "gaussian":
        h, w = image.shape[:2]
        noise = np.random.normal(mean, std, (h, w, 3))
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 1)
        return noisy_image

for image, label in training_dataset:
    # print(image.size)
    image_ = np.array(image)
    label = torch.tensor(label)
    print(image_.shape)
    # print(image_)
    # image_ = cv2.resize(image_, (224, 224))
    # print(image_)
    # print(image_.shape)
    noised_image = add_noise(image_)
    noised_image = torch.from_numpy(noised_image)
    noised_dataset.extend((noised_image, label))


device = torch.device("cuda")

# full_dataset = ImageFolder(root = data_dir, transform = None)

# train_ratio = 0.7
# test_ratio = 0.3

# 计算划分的样本数量
# train_size = int(train_ratio * len(full_dataset))
# test_size = len(full_dataset) - train_size

# 使用random_split函数划分数据集
# training_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

autoencoder = Autoencoder().to(device)

criterion_autoencoder = nn.MSELoss()

optimizer_autoencoder = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

epoches = 1000

log_interval = 1

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)  # cpu
torch.cuda.manual_seed(seed)  # gpu
torch.cuda.manual_seed_all(seed)  # all gpus

batch_size = 64
shuffle = True

training_dataLoader = DataLoader(noised_dataset, batch_size = batch_size, shuffle = False)
test_dataLoader = DataLoader(test_dataset, batch_size = len(test_dataset), shuffle = False)

# print(training_dataLoader.shape)

for epoch in range(1, epoches) :
    autoencoder.train()

    # Get the current GPU memory usage
    current_memory = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
    print(f"Current GPU memory usage: {current_memory:.2f} MB")
    for batch_idx, (images, labels) in enumerate(training_dataLoader):
        #print(images.size())
        optimizer_autoencoder.zero_grad()

        images = images.to(device)
        labels = labels.to(device)

        reconstructed_images = autoencoder(images)

        origin_images = training_dataset[batch_idx]

        autoencoder_loss = criterion_autoencoder(reconstructed_images, origin_images)
        # loss = F.nll_loss(outputs, labels)

        autoencoder_loss.backward(retain_graph=True)

        optimizer_autoencoder.step()

        #print(batch_idx)
        if batch_idx % log_interval == 0:
            print(f"autoencoder loss: {autoencoder_loss.item():.4f}")
    
