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
from torchvision.datasets import ImageFolder
from PIL import Image
import torch.nn.functional as F
import torchvision.models as models
import torch.nn.init as init

seed = 12124
torch.manual_seed(seed)
random.seed(seed)
numpy.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
learning_rate = 1e-3
num_epochs = 200
log_interval = 1

#train_dataset = torch.load('accredited_extended_dataset.pth')
train_dataset = torch.load('train_dataset.pth')
test_dataset = torch.load('val_dataset.pth')
# valid_dataset = torch.load('accredited_val_dataset.pth')
# valid_dataset = torch.load('ex_train_dataset.pth')

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=4)
test_loader = DataLoader(test_dataset, len(test_dataset), shuffle=False, num_workers=4)
# valid_loader = DataLoader(valid_dataset, len(valid_dataset), shuffle=False, num_workers=4)

VGGmodel = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device)
num_features = VGGmodel.classifier[6].in_features
VGGmodel.classifier[6] = nn.Linear(num_features, 2).to(device)

print(VGGmodel)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(VGGmodel.parameters(), lr=2e-5)

# 使用学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

glo_mx = 0.0
weight_decay = 1e-5

def train(model, device, train_loader, optimizer, epoch, scheduler, use_scheduler):
    model.train()
    total_images = 0
    if use_scheduler == True:
        scheduler.step()
    train_correct = 0
    train_loss = 0
    def random_block_mask(images_tensor, block_size):
        _, _, h, w = images_tensor.size()
        top = torch.randint(0, h - block_size, (images_tensor.size(0),), device=images_tensor.device)
        left = torch.randint(0, w - block_size, (images_tensor.size(0),), device=images_tensor.device)
        for i in range(images_tensor.size(0)):
            images_tensor[i, :, top[i]:top[i]+block_size, left[i]:left[i]+block_size] = 0.0
        return images_tensor
    image_transform = transforms.Compose([
        transforms.RandomRotation(30),  # 随机旋转角度范围为 -15 到 15 度
        # transforms.RandomResizedCrop(224, scale=(0.8, 1.2))  # 随机缩放大小在 0.8 到 1.2 之间
            # transforms.RandomRotation((-30, 30)),  # 随机旋转 -30度~30度

        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomVerticalFlip(),  # 随机竖直翻转
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.2)),  # 随机缩放 0.8倍~1.2倍并随机裁剪到指定大小
    ])

    for batch_idx, (images, labels) in enumerate(train_loader):
        total_images += len(images)
        images = images.to(device)
        origin_images = images
        for i in range(images.size(0)):
            images[i] = image_transform(images[i])
        # images = random_block_mask(images, 20)
        labels = labels.to(device)

        outputs = model(images)
        debug = 0
        if debug == True:
            print(outputs)
        
        loss = criterion(outputs, labels)

        l2_reg = torch.tensor(0., device=device)
        for param in model.parameters():
            l2_reg += torch.norm(param)

        use_l2_reg = False
        if use_l2_reg == True:
            loss += weight_decay * l2_reg

        images = origin_images
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, total_images, len(train_dataset),
            100. * total_images / len(train_dataset), loss.item()))
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()

    train_loss /= len(train_dataset)
    train_accuracy = train_correct / len(train_dataset)
    print(f"Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f}")

def eval(model, device, test_loader, is_validation):
    model.eval()
    # model.train()
    global glo_mx
    correct = 0
    total = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            # images = images + 0.003 * torch.randn(images.size()).to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            tp = torch.sum((predicted == 1) & (labels == 1)).item()
            tn = torch.sum((predicted == 0) & (labels == 0)).item()
            fp = torch.sum((predicted == 1) & (labels == 0)).item()
            fn = torch.sum((predicted == 0) & (labels == 1)).item()
    accuracy = correct / total
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    if is_validation == False and accuracy > glo_mx and accuracy > 0.40:
        glo_mx = accuracy
        model_name = "VGG_model/max_accuracy_{}.pth".format(accuracy)
        # model_name = "ResNet50max.pth"
        cam_name = "ResNet50_cam_{}.jpg".format(accuracy)
        # test_cam(cam_name)
        save = 0
        if save == True:
            torch.save(model.state_dict(), model_name) 
    if is_validation == False:
        print(f"Accuracy on test set: {accuracy:.2%}")
    else:
        print(f"Accuracy on validation test: {accuracy:.2%}")
    print(f"Val Sensitivity: {sensitivity:.4f} - Val Specificity: {specificity:.4f}")



start_time = time.time()

def train_single():
    for epoch in range(num_epochs):
        train(VGGmodel, device, train_loader, optimizer, epoch, scheduler, False)
        # eval(VGGmodel, device, valid_loader, True)
        eval(VGGmodel, device, test_loader, False)
        elapsed_time = time.time() - start_time
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")
        print(f"Max Accuracy on test set: {glo_mx:.2%}")

def fine_tuning(epochs_first, epochs_second):
    # Freeze the parameters in the feature extraction part
    for param in VGGmodel.features.parameters():
        param.requires_grad = False
    for epoch in range(epochs_first):
        train(VGGmodel, device, train_loader, optimizer, epoch, scheduler, False)
        eval(VGGmodel, device, test_loader, False)
        elapsed_time = time.time() - start_time
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")
        print(f"Max Accuracy on test set: {glo_mx:.2%}")
    for param in VGGmodel.features.parameters():
        param.requires_grad = True
    for param_group in optimizer.param_groups:
        param_group['lr'] = 1e-5
    for epoch in range(epochs_second):
        train(VGGmodel, device, train_loader, optimizer, epoch, scheduler, False)
        eval(VGGmodel, device, test_loader, False)
        elapsed_time = time.time() - start_time
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")
        print(f"Max Accuracy on test set: {glo_mx:.2%}") 

train_single()
torch.save(VGGmodel.state_dict(), 'Discriminator_weight.pth')