import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
import torch.nn.functional as F

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

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(1000, 2)  # 添加一个全连接层
        self.fc1 = nn.Linear(1000, 1000)
        self.fc2 = nn.Linear(1000, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc3 = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)
        # for param in self.resnet18.parameters():
            # param.requires_grad = False

    def forward(self, x):
        # x = self.resnet18.conv1(x)
        # x = self.resnet18.bn1(x)
        # x = self.resnet18.relu(x)
        # x = self.resnet18.maxpool(x)

        # x = self.resnet18.layer1(x)
        # x = self.resnet18.layer2(x)
        # x = self.resnet18.layer3(x)
        # x = self.resnet18.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc3(x)
        x = self.resnet18(x)
        # x = self.pool(x)
        # x = self.fc3(x)
        # x = self.dropout(x)
        x = self.fc(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        # x = self.softmax(x)
        return x

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        # for param in self.resnet50.parameters():
            # param.requires_grad = False

    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # x = F.log_softmax(x, dim = 1)
        # x = nn.softmax(x, dim = 1)
        return x


class ResNet50_max(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50_max, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = F.log_softmax(x, dim = 1)
        return x


# model = ResNet50_max()
# print(model)

