import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms
import random

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

class ResNet50_normal(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50_normal, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(2048, num_classes)
        # self.fc = nn.Linear(2048, num_classes)
        # for param in self.resnet50.parameters():
            # param.requires_grad = False

    def forward(self, x):
        x = self.resnet50(x)
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

class EdgeVGG19(nn.Module):
    def __init__(self, num_classes):
        super(EdgeVGG19, self).__init__()
        
        # 加载预训练的VGG19模型
        vgg19 = models.vgg19(pretrained=True)
        
        # 提取卷积层和全连接层
        self.edge = vgg19.features[:9]
        self.remain = vgg19.features[9:]
        self.avgpool = vgg19.avgpool
        num_features = self.vgg19.classifier[6].in_features
        self.vgg19.classifier[6] = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        edge = self.edge(x)
        x = x * 0.3
        edge = edge * 0.7
        concatenated = torch.cat((x, edge), dim = 1)
        concatenated
        x = self.classifier(x)

        # 将边缘细节特征与全连接层输出进行加权结合
        x = self.custom_weighted_layer(x) * 0.3 + edge * 0.7
        
        return x
# model = ResNet50_max()
# print(model)

class VGG19(nn.Module):
    def __init__(self, num_classes):
        super(VGG19, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True)
        num_features = self.vgg19.classifier[6].in_features
        self.vgg19.classifier[6] = nn.Linear(num_features, num_classes)
    def forward(self, x):
        x = self.vgg19(x)
        return x

class VGG19_with_STN(nn.Module):
    def __init__(self, num_classes):
        super(VGG19_with_STN, self).__init__()
         # 空间变换参数局部化网络(Localization Network)
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        # 回归参数预测网络(Regressor Network)
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 52 * 52, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)  # 3x2参数矩阵(2个平移参数，3个缩放/旋转参数)
        )
        self.vgg19 = models.vgg19(pretrained=True)
        num_features = self.vgg19.classifier[6].in_features
        self.vgg19.classifier[6] = nn.Linear(num_features, num_classes)
    def forward(self, x):
        # 提取特征
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 52 * 52)  # 将特征展平
        
        # 预测变换参数
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)  # reshape为3x2矩阵
        
        # 执行空间变换
        grid = F.affine_grid(theta, x.size())  # 根据变换参数生成采样网格
        x = F.grid_sample(x, grid)  # 对输入图像进行采样
        x = self.vgg19(x)
        return x

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()

        # Spatial transformer localization network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            # 224 224 -> 224 - 6 = 218
            nn.MaxPool2d(2, stride=2),
            # 109 109
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            # 105
            nn.MaxPool2d(2, stride=2),
            # 52
            nn.ReLU(True),

            nn.Conv2d(10, 12, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            # 25

            nn.Conv2d(12, 14, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            # 10

            nn.Conv2d(14, 16, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            # 2


        )

        # Regressor for the transformation parameters
        self.fc_loc = nn.Sequential(
            nn.Linear(16 * 2 * 2, 32),
            nn.ReLU(True),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        # Spatial transformer network forward function
        x = self.localization(x)
        x = x.view(-1, 16 * 2 * 2)
        x = self.fc_loc(x)
        return x


class double_VGG19(nn.Module):
    def __init__(self, num_classes):
        super(double_VGG19, self).__init__()
        self.vgg19_first = models.vgg19(pretrained=True)
        self.relu1 = nn.ReLU(True)
        self.relu2 = nn.ReLU(True)
        self.stn = STN()
        self.vgg19_second = models.vgg19(pretrained=True)
        num_features = self.vgg19_second.classifier[6].in_features
        self.vgg19_second.classifier[6] = nn.Linear(num_features, num_classes)
        self.vgg19_first.classifier[6] = nn.Linear(num_features, 3)
        self.X = torch.arange(224.0, device='cuda').unsqueeze(1).expand(224, 224)
        self.Y = torch.arange(224.0, device='cuda').unsqueeze(0).expand(224, 224)
        torch.autograd.set_detect_anomaly(True)
        self.is_train = True
    def freeze_first(self):
        for param in self.vgg19_first.parameters():
            param.requires_grad = False

    def forward(self, x):
        y = self.vgg19_first(x)
        # y = self.stn(x)
        # y = self.relu1(y)
        # y = -y + 1
        # y = self.relu2(y)
        y = torch.sigmoid(y)
        # print(y)
        idx = random.randint(0, x.size()[0] - 1)
        if self.is_train == True:
            pil_image = transforms.ToPILImage()(x[idx])
            path = f"circle/example1.jpg"
            pil_image.save(path)

        # 1. norm test set to be maximum
        # 2. freeze first vgg19 after some epoch
        if self.is_train == True:
            # channel_mean = x.mean(dim=(1, 2))  # 计算每个通道的均值
            # for c in range(x.size(0)):
            #     x[c][x[c] < channel_mean[c]] = 0  # 将小于均值的元素赋值为0
            #     # 计算每个位置的最大值
            # # max_values, _ = torch.max(x, dim=0)
            # # 使用广播将最大值复制到每个通道的相应位置
            # # x = max_values.unsqueeze(0).expand_as(x)
            # # return tensor
            for i in range(x.size()[0]):
                # continue
                # flag = random.randint(0, 1)
                # if flag == 0:
                    # continue
                y[i][0] = y[i][0] * 224
                y[i][1] = y[i][1] * 224
                # x_factor = random.uniform(108, 116)
                # y_factor = random.uniform(108, 116)
                t_X = self.X.clone()
                t_Y = self.Y.clone()
                t_XX = t_X - y[i][0]
                t_YY = t_Y - y[i][1]
                # t_XX = t_X - x_factor
                # t_YY = t_Y - y_factor
                t_XXX = torch.mul(t_XX, t_XX.detach())
                t_YYY = torch.mul(t_YY, t_YY.detach())
                dis = torch.sqrt(torch.add(t_XXX, t_YYY))
                dis = 224 - dis
                dis = dis / 224
                # dis = dis + y[i][2]
                dis = dis + 1
                dis = torch.log2(dis) 

                # range_factor = random.uniform(0.6, 0.7)
                mask = (dis >= y[i][2])
                mx = torch.max(x[i])
                # min_k_factor = torch.min(x[i])
                min_k_factor = 1.0
                max_k_factor = 1.0 / (mx + 0.01)
                # max_k_factor = 0.7
                if min_k_factor > max_k_factor:
                    t = min_k_factor
                    min_k_factor = max_k_factor
                    max_k_factor = t
                k_factor = random.uniform(min_k_factor, max_k_factor)
                # print(f"k_factor : {k_factor}")
                dis.masked_fill_(mask, k_factor)
                # dis = dis * 2.5
                # print(dis / 224)
                # dis = dis * 
                # print(dis)
                # dis_ = torch.min(dis, torch.tensor(1.0))
                dis = dis.unsqueeze(0)
                x[i] = torch.mul(x[i], dis.detach())
                x[i] = torch.min(x[i], torch.tensor(1.0))
        # else:
            # for i in range(x.size()[0]):
                # mx = torch.max(x[i])
                # factor = 1.0 / (mx + 0.01)
                # x[i] = x[i] * factor
                # x[i] = torch.min(x[i], torch.tensor(1.0))

        # c, h, w = x.size()[1:]
        # print(f"X : {y[0][0]}, Y : {y[0][1]}")
        # print(f"factor : {y[0][0]}")
        # j = torch.arange(h, dtype=torch.float32).unsqueeze(1).unsqueeze(2).unsqueeze(0)
        # k = torch.arange(w, dtype=torch.float32).unsqueeze(0).unsqueeze(2).unsqueeze(0)
        # diff_j = j - y[:, 0, 0, 0].unsqueeze(1).unsqueeze(1)
        # diff_k = k - y[:, 1, 0, 0].unsqueeze(1).unsqueeze(1)
        # squared_diff = diff_j * diff_j + diff_k * diff_k
        # scaling_factor = torch.sqrt(squared_diff) * y[:, 2, 0, 0].unsqueeze(1).unsqueeze(1)
        # x *= scaling_factor
        if self.is_train == True:
            pil_image = transforms.ToPILImage()(x[idx])
            path = f"circle/example2.jpg"
            pil_image.save(path)
        x = self.vgg19_second(x)
        return x


class sphereAttention(nn.Module):
    def __init__(self, input_size, input_channel):
        super(sphereAttention, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(input_channel, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

class double_res50(nn.Module):
    def __init__(self, num_classes):
        super(double_res50, self).__init__()
        self.res50_first = models.vgg19(pretrained=True)
        self.res50_second = models.vgg19(pretrained=True)
        self.res50_first.fc = nn.Linear(2048, 3)
        self.res50_second.fc = nn.Linear(2048, num_classes)
        self.X = torch.arange(224.0, device='cuda').unsqueeze(1).expand(224, 224)
        self.Y = torch.arange(224.0, device='cuda').unsqueeze(0).expand(224, 224)
        torch.autograd.set_detect_anomaly(True)
        self.is_train = True
        self.bn = nn.BatchNorm2d(64)
    def forward(self, x):
        y = self.res50_first(x)
        # y = self.bn(y)
        y = torch.sigmoid(y)
        idx = random.randint(0, x.size()[0] - 1)
        if self.is_train == False:
            pil_image = transforms.ToPILImage()(x[idx])
            path = f"circle_res/example1.jpg"
            pil_image.save(path)

        for i in range(x.size()[0]):
            y[i][0] = y[i][0] * 224
            y[i][1] = y[i][1] * 224
            t_X = self.X.clone()
            t_Y = self.Y.clone()
            t_XX = t_X - y[i][0]
            t_YY = t_Y - y[i][1]
            t_XXX = torch.mul(t_XX, t_XX.detach())
            t_YYY = torch.mul(t_YY, t_YY.detach())
            dis = torch.sqrt(torch.add(t_XXX, t_YYY))
            dis = 224 - dis
            dis = dis / 224
            dis = dis + 1
            dis = torch.log2(dis)
            mask = (dis >= y[i][2] * 1.45)
            dis.masked_fill_(mask, 1)
            # dis = dis * 2.5
            # print(dis / 224)
            # dis = dis * 
            # print(dis)
            dis_ = torch.min(dis, torch.tensor(1.0))
            dis_ = dis_.unsqueeze(0)
            x[i] = torch.mul(x[i], dis_.detach())
        # c, h, w = x.size()[1:]
        print(f"X : {y[0][0]}, Y : {y[0][1]}")
        # j = torch.arange(h, dtype=torch.float32).unsqueeze(1).unsqueeze(2).unsqueeze(0)
        # k = torch.arange(w, dtype=torch.float32).unsqueeze(0).unsqueeze(2).unsqueeze(0)
        # diff_j = j - y[:, 0, 0, 0].unsqueeze(1).unsqueeze(1)
        # diff_k = k - y[:, 1, 0, 0].unsqueeze(1).unsqueeze(1)
        # squared_diff = diff_j * diff_j + diff_k * diff_k
        # scaling_factor = torch.sqrt(squared_diff) * y[:, 2, 0, 0].unsqueeze(1).unsqueeze(1)
        # x *= scaling_factor
        if self.is_train == False:
            pil_image = transforms.ToPILImage()(x[idx])
            path = f"circle_res/example2.jpg"
            pil_image.save(path)
        x = self.res50_second(x)
        return x



class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channels = channels
        self.reduction = reduction

        # Spatial attention layers
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()
        )

        # Channel attention layers
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Spatial attention
        spatial_attention = self.spatial_attention(x)
        x = x * spatial_attention

        # Channel attention
        channel_attention = self.channel_attention(x)
        x = x * channel_attention

        return x


class CBVGG(nn.Module):
    def __init__(self, num_classes):
        super(CBVGG, self).__init__()
        self.vgg19 = models.vgg19(pretrained = True)
        # Modify the last fully connected layer for binary classification
        num_features = self.vgg19.classifier[6].in_features
        self.vgg19.classifier[6] = nn.Linear(num_features, num_classes)

        # Add CBAM module after each convolutional layer
        self.add_cbam(self.vgg19.features)

    def add_cbam(self, module):
        if isinstance(module, nn.Conv2d):
            module.add_module('cbam', CBAM(module.out_channels))
        elif isinstance(module, nn.Module):
            for child in module.children():
                self.add_cbam(child)



    def forward(self, x):
        x = self.vgg19(x)
        return x


class ViT(nn.Module):
    def __init__(self, num_classes):
        super(ViT, self).__init__()
        self.transformer = nn.Transformer(d_model=768, nhead=12, num_encoder_layers=12)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.transformer(x, x)
        x = x.mean(dim=1)  # 对图像的每个patch取平均
        x = self.fc(x)
        return x
