import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from model import Autoencoder, DenoisingAutoencoder, ResNet18
from torchvision.datasets import ImageFolder

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
learning_rate = 1e-3
num_epochs = 400
log_interval = 1


train_dataset = torch.load('ex_train_dataset.pth')
test_dataset = torch.load('test_dataset.pth')

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, len(test_dataset), shuffle=False)



# model = Autoencoder().to(device)
# model.load_state_dict(torch.load('autoencoder.pth'))
CNNmodel = ResNet18(2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(CNNmodel.parameters(), lr=learning_rate)

# 使用学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

AE = DenoisingAutoencoder().to(device)
AE.load_state_dict(torch.load('s_autoencoder_0.0001.pth'))
AE.eval()

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_images = 0
    scheduler.step()
    train_correct = 0
    train_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        total_images += len(images)
        images = images.to(device)
        images = images + 0.005 * torch.randn(images.size()).to(device)
        # images = AE(images)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

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
        train_correct += (predicted == labels).sum().item()

    train_loss /= len(train_dataset)
    train_accuracy = train_correct / len(train_dataset)
    print(f"Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f}")

def eval(model, device, test_loader):
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        correct = 0
        total = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for images, labels in test_loader:
            images = images.to(device)
            images = images + 0.005 * torch.randn(images.size()).to(device)
            # print(images.shape)
            # images = AE(images)
            image, _ = test_loader.dataset[0]
            # image_tensor = image.unsqueeze(0)
            # image_tensor = image_tensor.to(device)
            image_noisy = images[0]
            image = transforms.functional.to_pil_image(image)
            image_noisy = transforms.functional.to_pil_image(image_noisy.squeeze(0))
            image.save('image.jpg')
            image_noisy.save('image_noisy.jpg')
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
        print(f"Accuracy on test set: {accuracy:.2%}")
        print(f"Val Sensitivity: {sensitivity:.4f} - Val Specificity: {specificity:.4f}")


start_time = time.time()
for epoch in range(num_epochs):
    train(CNNmodel, device, train_loader, optimizer, epoch)
    eval(CNNmodel, device, test_loader)
    elapsed_time = time.time() - start_time
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")

# torch.save(model.state_dict(), 'autoencoder.pth')
torch.save(CNNmodel.state_dict(), 'ResNet18.pth')