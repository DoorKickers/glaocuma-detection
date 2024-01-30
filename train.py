import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from model import Autoencoder, DenoisingAutoencoder
from torchvision.datasets import ImageFolder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
learning_rate = 1e-3
num_epochs = 400
log_interval = 1


transform = Compose([
   Resize((224, 224)),  # Resize the images to a specific size
   ToTensor(),  # Convert the images to tensors
])
train_dir = "/root/workspace_remote/data/RIM-ONE_DL_images/partitioned_by_hospital/training_set"
train_dataset = ImageFolder(root = train_dir, transform = transform)
extended_dataset = []

def rotate_images(image, num_rotations):
    rotated_images = []
    for angle in range(0, 360, num_rotations):
        rotated_image = transforms.functional.rotate(image, angle)
        rotated_images.append(rotated_image)
    return rotated_images

for image, label in train_dataset:
    rotated_images = rotate_images(image, 60)
    extended_dataset.extend([(rotated_image, label) for rotated_image in rotated_images])

train_loader = DataLoader(extended_dataset, batch_size = batch_size, shuffle = True)

# model = Autoencoder().to(device)
# model.load_state_dict(torch.load('autoencoder.pth'))
model = DenoisingAutoencoder().to(device)
model.load_state_dict(torch.load('s_autoencoder.pth'))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_images = 0
    for batch_idx, (img, _) in enumerate(train_loader):
        total_images += len(img)
        noisy_img = img + 0.05 * torch.randn(img.size())

        img = img.to(device)
        noisy_img = noisy_img.to(device)

        output = model(noisy_img)
        loss = criterion(output, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, total_images, len(extended_dataset),
            100. * total_images / len(extended_dataset), loss.item()))
        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

start_time = time.time()
for epoch in range(num_epochs):
    train(model, device, train_loader, optimizer, epoch)
    elapsed_time = time.time() - start_time
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")

# torch.save(model.state_dict(), 'autoencoder.pth')
torch.save(model.state_dict(), 's_autoencoder.pth')