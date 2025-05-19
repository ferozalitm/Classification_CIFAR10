# Author: T M Feroz Ali
#       Classification using CNNs (3 layer)
#       Dataset: CIFAR10 
#       Using BatchNorm to improce accuracy N/w(3->32->64) (train 98, test 70)
#       Using BatchNorm + Data Augmentation to improve test per by regularization (train 78, test 77)
#       Train  accuracy: ~82.75
#       Test accuracy: ~81.5

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

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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

# Build a fully connected layer and forward pass
class ConvNet(nn.Module):
    def __init__(self, ip_image_size):
        super().__init__()
        self.op_layer_size = ip_image_size

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride=2))
        self.op_layer_size = (self.op_layer_size - 5)//1 + 1
        self.op_layer_size = (self.op_layer_size - 3)//2 + 1        

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride=2))
        self.op_layer_size = (self.op_layer_size - 5)//1 + 1
        self.op_layer_size = (self.op_layer_size - 3)//2 + 1
        
        # print(self.op_layer_size)
        self.fc_layer_size = self.op_layer_size*self.op_layer_size*64
        print(f'op_layer_size:{self.op_layer_size}, fc_layer_size:{self.fc_layer_size}')
        
        self.fc = nn.Linear(in_features=self.fc_layer_size, out_features=no_classes)      

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(-1, self.fc_layer_size ) # Better way: x = x.view(x.size(0), -1)  
        x = self.fc(x)
        return x

# Build model.
model = ConvNet(ip_image_size).to(device)
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
    plt.savefig('v5b_BN_Aug_StepLR50_accuracy.png')

    plt.clf()
    plt.plot(epoch_all, loss_train_all, marker = 'o', mec = 'g', label='Average Train loss')
    plt.plot(epoch_all, loss_test_all, marker = 'o', mec = 'r', label='Average Test loss')
    plt.legend()
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig('v5b_BN_Aug_StepLR50_loss.png')

    model.train()
