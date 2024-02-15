import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# 定义Grad-CAM函数
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.feature_maps = None
        self.gradient = None
        self.model.eval()

        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0].detach()

        target_layer = self.model.resnet.layer4[-1]  # 最后一个layer4的基本块
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_image, target_class, alpha=0.1, device = torch.device("cpu")):
        input_tensor = transforms.ToTensor()(input_image)
        input_batch = input_tensor.unsqueeze(0).to(device)

        # 前向传播
        logits = self.model(input_batch)
        self.model.zero_grad()

        print(logits)

        softmax_output = torch.exp(logits)
        print(softmax_output)

        # 反向传播
        one_hot = torch.zeros_like(logits)
        one_hot[0][target_class] = 1
        logits.backward(gradient=one_hot)

        # 获取特征图梯度
        gradients = torch.mean(self.gradient, dim=(2, 3), keepdim=True)
        # 计算特征图权重
        weights = torch.mean(self.feature_maps * gradients, dim=1, keepdim=True)
        weights = nn.functional.relu(weights)

        # 将特征图权重和对应的特征图相乘并求和
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        cam = nn.functional.relu(cam)

        # 归一化热力图
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)

        # 将热力图从Tensor转换为NumPy数组
        heatmap = cam[0][0].cpu().numpy()

        # 将热力图调整为与输入图像相同的大小
        heatmap = cv2.resize(heatmap, (input_image.size[0], input_image.size[1]))
        heatmap = np.uint8(255 * heatmap)

        # 将热力图应用于原始图像
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 将输入图像转换为NumPy数组
        input_image = np.array(input_image)

        # 将热力图叠加在原始图像上
        overlay = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        output_image = cv2.addWeighted(overlay, 1 - alpha, heatmap, alpha, 0)

        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

        return output_image