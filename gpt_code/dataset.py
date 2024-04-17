import os
import torch
from torchvision import datasets, transforms

# 数据转换
data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

# 加载训练集和测试集图像
train_dir = "/root/workspace_remote/data/RIM-ONE_DL_images/partitioned_by_hospital/training_set"
test_dir = "/root/workspace_remote/data/RIM-ONE_DL_images/partitioned_by_hospital/test_set"

train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transform)

# 提取绿色通道
def extract_green_channel(dataset):
    green_channel_data = []
    for image, _ in dataset:
        green_channel = image[1]  # 提取绿色通道
        green_channel = green_channel.unsqueeze(0)
        green_channel_data.append((green_channel, _))
    return green_channel_data

# 提取训练集和测试集的绿色通道数据
train_green_channel = extract_green_channel(train_dataset)
test_green_channel = extract_green_channel(test_dataset)
for i in range(10):
    img, _ = train_green_channel[i]
    pilImage = transforms.ToPILImage()(img)
    path = f"samples/img{i}.jpg"
    pilImage.save(path)

# 保存训练集和测试集的绿色通道数据为.pth文件
torch.save(train_green_channel, 'train_green_channel.pth')
torch.save(test_green_channel, 'test_green_channel.pth')
