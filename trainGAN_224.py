import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time
import sys
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from model import Autoencoder, DenoisingAutoencoder, ResNet18, ResNet50, ResNet50_max, ResNet50_normal
from torchvision.datasets import ImageFolder
from PIL import Image
from gradCAM import GradCAM
import torch.nn.functional as F
import torchvision.models as models

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
learning_rate = 1e-3
num_epochs = 200
log_interval = 1


# train_dataset = torch.load('origin_train_dataset.pth')
# train_dataset = torch.load('ex_train_dataset.pth')
train_dataset = torch.load('accredited_extended_dataset.pth')
# train_dataset = torch.load('accredited_extended_dataset.pth')
test_dataset = torch.load('test_dataset.pth')
valid_dataset = torch.load('ex_train_dataset.pth')

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, len(test_dataset), shuffle=False)
valid_loader = DataLoader(valid_dataset, len(valid_dataset), shuffle=False)



# model = Autoencoder().to(device)
# model.load_state_dict(torch.load('autoencoder.pth'))
# CNNmodel = ResNet18(2).to(device)
# CNNmodel = ResNet50_max(2).to(device)
CNNmodel = ResNet50_normal(2).to(device)
VGGmodel = models.vgg19(pretrained = True).to(device)
# print(VGGmodel)
# sys.exit(0)
num_features = VGGmodel.classifier[6].in_features
VGGmodel.classifier[6] = nn.Linear(num_features, 2).to(device)

print(VGGmodel)
criterion = nn.CrossEntropyLoss()
# criterion = nn.NLLLoss().to(device)
# for param in VGGmodel.features.parameters():
#   param.requires_grad = False
# optimizer = optim.Adam(CNNmodel.parameters(), lr=1e-3)
optimizer = optim.RMSprop(VGGmodel.parameters(), lr=1e-5)

# 使用学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

AE = DenoisingAutoencoder().to(device)
AE.load_state_dict(torch.load('s_autoencoder_0.0001.pth'))
AE.eval()
# CNNmodel.load_state_dict(torch.load('ResNet18_max_accuracy_0.8620689655172413.pth'))

glo_mx = 0.0

weight_decay = 1e-4;

def train(model, device, train_loader, optimizer, epoch, scheduler):
    model.train()
    total_images = 0
    # scheduler.step()
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
        transforms.RandomRotation(80),  # 随机旋转角度范围为 -15 到 15 度
        # transforms.RandomResizedCrop(224, scale=(0.8, 1.2))  # 随机缩放大小在 0.8 到 1.2 之间
    ])

    for batch_idx, (images, labels) in enumerate(train_loader):
        total_images += len(images)
        images = images.to(device)
        # if batch_idx % 3 == 0:
            # images = random_block_mask(images, 50)
        # images = random_block_mask(images, 25)
        # for i in range(images.size(0)):
            # images[i] = image_transform(images[i]).to(device)
        # images = images + 0.003 * torch.randn(images.size()).to(device)
        # images = AE(images)
        labels = labels.to(device)

        outputs = model(images)
        # print(outputs)
        # loss = F.nll_loss(outputs, labels)
        loss = criterion(outputs, labels)
        l2_reg = torch.tensor(0., device=device)
        for param in model.parameters():
            l2_reg += torch.norm(param)

        loss += weight_decay * l2_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, total_images, len(train_dataset),
            100. * total_images / len(train_dataset), loss.item()))
        train_loss += loss.item() * images.size(0)
        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        _, predicted = torch.max(outputs.data, 1)
        # print(_)
        train_correct += (predicted == labels).sum().item()

    train_loss /= len(train_dataset)
    train_accuracy = train_correct / len(train_dataset)
    print(f"Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f}")

def eval(model, device, test_loader):
    model.eval()
    # model.train()
    criterion = nn.MSELoss()
    global glo_mx
    pre_mx = 0.8276
    # weights = model.fc.weight.data
    # features = model.layer4
    def test_cam(path):
        image = Image.open('/root/workspace_remote/data/RIM-ONE_DL_images/partitioned_by_hospital/training_set/glaucoma/r2_Im365.png') 
        grad_cam = GradCAM(model)
        target_class = 0
        heatmap = grad_cam.generate_heatmap(image, target_class, alpha=0.2, device=torch.device("cuda"))
        heatmap = Image.fromarray(heatmap)
        heatmap.save(path)
        print("saved")
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
            ## images = images + 0.0make_unique05 * torch.randn(images.size()).to(device)
            # print(images.shape)
            # images = AE(images)
            # image, _ = test_loader.dataset[0]
            # image_tensor = image.unsqueeze(0)
            # image_tensor = image_tensor.to(device)
            # features_output = features(image_tensor)
            # output = model(image_tensor)
            # class_score = output[0]
            # cam = torch.matmul(weights, features_output.squeeze())
            # cam = nn.functional.relu(cam)
            # cam = cam.unsqueeze(0)
            # cam = nn.functional.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
            # cam = cam.squeeze()
            # cam = cam - cam.min()
            # cam = cam / cam.max()


            # image_noisy = images[0]
            # image = transforms.functional.to_pil_image(image)
            # image_noisy = transforms.functional.to_pil_image(image_noisy.squeeze(0))
            # image.save('image.jpg')
            # image_noisy.save('image_noisy.jpg')
            # image_tensor = image.unsqueeze(0)
            # image_tensor = image_tensor.to(device)
            # image_ae = AE(image_tensor)
            # temp_loss = criterion(image_tensor, image_ae)
            # print(f"loss = {temp_loss.item()}")
            # image = transforms.functional.to_pil_image(image)
            # image_ae = transforms.functional.to_pil_image(image_ae.squeeze(0))
            # image.save('image.jpg')
            # image_ae.save('image_ae.jpg')
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
    if accuracy > glo_mx and accuracy > 0.40:
        glo_mx = accuracy
        # model_name = "ResNet50_max_accuracy_{}.pth".format(accuracy)
        # model_name = "ResNet50max.pth"
        cam_name = "ResNet50_cam_{}.jpg".format(accuracy)
        # test_cam(cam_name)

        # torch.save(CNNmodel.state_dict(), model_name) 
    print(f"Accuracy on test set: {accuracy:.2%}")
    print(f"Val Sensitivity: {sensitivity:.4f} - Val Specificity: {specificity:.4f}")



start_time = time.time()
# last_part_epochs = 10
# together_epochs = 200
# for param in CNNmodel.resnet50.parameters():
#     param.requires_grad = False
# for epoch in range(last_part_epochs):
#     train(CNNmodel, device, train_loader, optimizer, epoch, scheduler)
#     eval(CNNmodel, device, test_loader)
#     elapsed_time = time.time() - start_time
#     print(f"Elapsed Time: {elapsed_time:.2f} seconds")

# for param in CNNmodel.resnet50.parameters():
#     param.requires_grad = True

# optimizer = optim.Adam(CNNmodel.parameters(), lr=learning_rate)
# optimizer = optim.RMSprop(CNNmodel.parameters(), lr=1e-5)

# # 使用学习率调度器
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

# for epoch in range(together_epochs):
#     train(CNNmodel, device, train_loader, optimizer, epoch, scheduler)
#     eval(CNNmodel, device, test_loader)
#     elapsed_time = time.time() - start_time
#     print(f"Elapsed Time: {elapsed_time:.2f} seconds")
#     print(f"Max Accuracy on test set: {glo_mx:.2%}")

# sys.exit(0)
# eval(CNNmodel, device, test_loader)
def train_single():
    for epoch in range(num_epochs):
        train(VGGmodel, device, train_loader, optimizer, epoch, scheduler)
        eval(VGGmodel, device, test_loader)
        elapsed_time = time.time() - start_time
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")
        print(f"Max Accuracy on test set: {glo_mx:.2%}")
def fine_tuning(epochs_first, epochs_second):
    # Freeze the parameters in the feature extraction part
    for param in VGGmodel.features.parameters():
        param.requires_grad = False
    for epoch in range(epochs_first):
        train(VGGmodel, device, train_loader, optimizer, epoch, scheduler)
        eval(VGGmodel, device, test_loader)
        elapsed_time = time.time() - start_time
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")
        print(f"Max Accuracy on test set: {glo_mx:.2%}")
    for param in VGGmodel.features.parameters():
        param.requires_grad = True
    for param_group in optimizer.param_groups:
        param_group['lr'] = 2e-5
    for epoch in range(epochs_second):
        train(VGGmodel, device, train_loader, optimizer, epoch, scheduler)
        eval(VGGmodel, device, test_loader)
        elapsed_time = time.time() - start_time
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")
        print(f"Max Accuracy on test set: {glo_mx:.2%}") 

def fine_tuning_res(epochs_first, epochs_second):
    # Freeze the parameters in the feature extraction part
    for param in CNNmodel.resnet50.parameters():
        param.requires_grad = False
    for param in CNNmodel.resnet50.fc.parameters():
        param.requires_grad = True
    for m in CNNmodel.modules():
          if isinstance(m, nn.BatchNorm2d):
              m.weight.requires_grad = True
              m.bias.requires_grad = True
    for epoch in range(epochs_first):
        train(CNNmodel, device, train_loader, optimizer, epoch, scheduler)
        eval(CNNmodel, device, test_loader)
        elapsed_time = time.time() - start_time
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")
        print(f"Max Accuracy on test set: {glo_mx:.2%}")
    for param in CNNmodel.resnet50.parameters():
        param.requires_grad = True
    for param_group in optimizer.param_groups:
        param_group['lr'] = 2e-5
    for epoch in range(epochs_second):
        train(CNNmodel, device, train_loader, optimizer, epoch, scheduler)
        eval(CNNmodel, device, test_loader)
        elapsed_time = time.time() - start_time
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")
        print(f"Max Accuracy on test set: {glo_mx:.2%}") 
# 原则 所有的参数改变 执行不同的代码 一定要修改调用而不是去注释定义
# fine_tuning(10, 20)
for param in CNNmodel.parameters():
    param.requires_grad = True
train_single()
# fine_tuning_res(10, 20)
# torch.save(model.state_dict(), 'autoencoder.pth')
# torch.save(CNNmodel.state_dict(), 'ResNet18.pth')