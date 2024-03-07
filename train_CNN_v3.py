import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time
import numpy
import sys
import random
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from model import Autoencoder, DenoisingAutoencoder, ResNet50_normal, VGG19
from torchvision.datasets import ImageFolder
from PIL import Image
from gradCAM import GradCAM
import torch.nn.functional as F
import torchvision.models as models
import torch.nn.init as init


def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64

train_dataset = torch.load('accredited_extended_dataset.pth')
test_dataset = torch.load('test_dataset.pth')
# valid_dataset = torch.load('accredited_val_dataset.pth')
valid_dataset = torch.load('ex_train_dataset.pth')

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, len(test_dataset), shuffle=False)
valid_loader = DataLoader(valid_dataset, len(valid_dataset), shuffle=False)

model = VGG19(2).to(device)

print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

glo_mx = 0.0