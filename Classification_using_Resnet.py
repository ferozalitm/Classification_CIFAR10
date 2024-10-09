
# Classification using Resnet
# Dataset: CIFAR10 
# Using BatchNorm + Data Augmentation to improve test per by regularization (train 78, test 77)
# Step lr to further improve test performance.

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Hyperparameters
no_classes = 10
batch_size = 256

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

train_dataset = torchvision.datasets.CIFAR10(root='data/',
                                     train=True,
                                     transform=transform,
                                     download=True)

test_dataset = torchvision.datasets.CIFAR10(root='data/',
                                     train=False,
                                     transform=transforms.ToTensor(),
                                     download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

# Finding image size and channel depth
# train_dataset[0][0].shape  -> torch.Size([3, 32, 32])
ip_image_size = train_dataset[0][0].shape[1]
print(f'image_size: {ip_image_size}')
ip_image_ch = train_dataset[0][0].shape[0]
print(f'ip_image_ch: {ip_image_ch}')
print(ip_image_ch)

no_batches_train = len(train_loader)
no_batches_tst = len(test_loader)
print(f"No_batches train: {no_batches_train}")
print(f"No_batches test: {no_batches_tst}")

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
model = ResNet(ResidualBlock, [2, 2, 2]).to(device)


# Build model.
# model = ConvNet(ip_image_size).to(device)
# breakpoint()

# Build optimizer.
learning_rate = 0.01
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Build loss.
criterion = nn.CrossEntropyLoss()

no_epochs = 150
first_pass = True
epoch_all = []
accuracy_test_all = []
accuracy_train_all = []
loss_test_all = []
loss_train_all = []

curr_lr = learning_rate
for epoch in range(no_epochs):

    # Training
    batch_idx = 0
    total_loss_train = 0
    total_correct_train = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass.
        pred = model(images)

        # print(pred.shape)
        # breakpoint()

        # Compute loss.
        loss = criterion(pred, labels)
        if epoch == 0 and first_pass == True:
            print(f'Initial {epoch} loss: ', loss.item())
            first_pass = False

        # Compute gradients.
        optimizer.zero_grad()
        loss.backward()

        # 1-step gradient descent.
        optimizer.step()

        # calculating train loss
        total_loss_train += loss.item()
        total_correct_train += torch.sum(labels == torch.argmax(pred, dim=1)).item()

        if epoch == 0 and (batch_idx+1) % 10 == 0:
            print(f"Train Batch:{batch_idx}/{no_batches_train}, loss: {loss}, total_loss: {total_loss_train}")

            
    # Decay learning rate
    if (epoch+1) % 50 == 0:
        curr_lr /= 10
        update_lr(optimizer, curr_lr)

    print(f'Train Epoch:{epoch}, Average Train loss:{total_loss_train/no_batches_train}, Average Train accuracy:{total_correct_train/len(train_dataset)*100.} ', )

    # Testing after each epoch
    model.eval()
    with torch.no_grad():

        total_loss_test = 0
        total_correct_test = 0

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass.
            pred = model(images)

            # Compute test loss.
            loss = criterion(pred, labels)
            total_loss_test += loss.item()
            total_correct_test += torch.sum(labels == torch.argmax(pred, dim=1)).item()
            # print(f"test Batch:{batch_idx}/{len(test_loader)}, loss: {loss}, total_loss: {total_loss_test}")

        print(f'Test Epoch:{epoch}, Average Test loss: {total_loss_test/no_batches_tst}, Average Test accuracy: {total_correct_test/len(test_dataset)*100.}', )
    

    # PLotting train and test curves
    # breakpoint()
    epoch_all.append(epoch)
    accuracy_train = total_correct_train/len(train_dataset)*100.
    accuracy_train_all.append(accuracy_train)
    accuracy_test = total_correct_test/len(test_dataset)*100.
    accuracy_test_all.append(accuracy_test)
    loss_test_all.append(total_loss_test/no_batches_tst)
    loss_train_all.append(total_loss_train/no_batches_train)

    plt.clf()
    plt.plot(epoch_all, accuracy_train_all, marker = 'o', mec = 'g', label='Average Train accuracy')
    plt.plot(epoch_all, accuracy_test_all, marker = 'o', mec = 'r', label='Average Test accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
    plt.savefig('v6a_BN_Aug_Resnet_StepLR0_01_ep150_accuracy.png')

    plt.clf()
    plt.plot(epoch_all, loss_train_all, marker = 'o', mec = 'g', label='Average Train loss')
    plt.plot(epoch_all, loss_test_all, marker = 'o', mec = 'r', label='Average Test loss')
    plt.legend()
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig('v6a_BN_Aug_Resnet_StepLR0_01_ep150_loss.png')

    model.train()
