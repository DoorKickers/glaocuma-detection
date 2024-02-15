import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from model import Autoencoder, DenoisingAutoencoder, ResNet50_max
from torchvision.datasets import ImageFolder
from PIL import Image
from gradCAM import GradCAM

def test_denoise_CNN():
    # 加载训练好的模型参数
    device = torch.device("cuda")
    #model = Autoencoder()
    #model.load_state_dict(torch.load('autoencoder.pth'))
    model = DenoisingAutoencoder().to(device)
    model.load_state_dict(torch.load('s_autoencoder_0.0001.pth'))
    model.eval()

    # 加载和预处理测试图片
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # image = Image.open('/root/workspace_remote/data/RIM-ONE_DL_images/partitioned_by_hospital/training_set/glaucoma/r2_Im365.png')
    image = Image.open('/root/workspace_remote/data/RIM-ONE_DL_images/partitioned_by_hospital/test_set/glaucoma/r2_Im257.png')

    image.save('origin_image.jpg')

    # 将图片转为RGB模式，并进行预处理转换
    image_rgb = image.convert('RGB')
    image_tensor = transform(image_rgb).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # 添加高斯噪声
    noisy_image = image_tensor + 0.0 * torch.randn(image_tensor.size()).to(device)

    # 将图片输入模型进行降噪
    denoised_image = model(noisy_image)

    # 将结果转为PIL图像并保存
    denoised_image = transforms.ToPILImage()(denoised_image.squeeze(0))
    noisy_image = transforms.ToPILImage()(noisy_image.squeeze(0))

    denoised_image.save('denoised_image.jpg')
    noisy_image.save('noisy_image.jpg')

def test_grad_CAM():
    model = ResNet50_max()
    model.load_state_dict(torch.load('ResNet50max.pth'))
    image = Image.open('/root/workspace_remote/data/RIM-ONE_DL_images/partitioned_by_hospital/training_set/glaucoma/r2_Im354.png') 
    image.save('beforemap.jpg')
    grad_cam = GradCAM(model)
    target_class = 0
    heatmap = grad_cam.generate_heatmap(image, target_class, alpha=0.15)
    output_path = 'heatmap.jpg'
    heatmap = Image.fromarray(heatmap)
    heatmap.save(output_path)
    print("saved")

def test_transform():
    image = Image.open('/root/workspace_remote/data/RIM-ONE_DL_images/partitioned_by_hospital/test_set/glaucoma/r2_Im257.png') 
    print(image.getpixel((0, 0)))

    image = transforms.ToTensor()(image)
    print(image)

    image= transforms.ToPILImage()(image)


    print(image.getpixel((0, 0)))


test_grad_CAM()



