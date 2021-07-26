# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 11:39:21 2021

@author: rapha
"""
import os
import torch
import torchvision

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10
import torch.nn.functional as F

batch_size = 64
epochs = 100
device = "cuda"

class AllCNNC(nn.Module):
    def __init__(self):
        super(AllCNNC, self).__init__()
        
        self.conv_1 = nn.Conv2d(3, 96, (3,3),
                                 stride = 1, padding = 1, dilation = 1, 
                                 groups = 1, bias = True,
                                 padding_mode = "zeros")
        self.conv_2 = nn.Conv2d(96, 96, (3,3),
                                 stride = 1, padding = 1, dilation = 1, 
                                 groups = 1, bias = True,
                                 padding_mode = "zeros")
        self.conv_3 = nn.Conv2d(96, 96, (3,3),
                                 stride = 2, padding = 1, dilation = 1, 
                                 groups = 1, bias = True,
                                 padding_mode = "zeros")

        self.conv_4 = nn.Conv2d(96, 192, (3,3),
                                 stride = 1, padding = 1, dilation = 1, 
                                 groups = 1, bias = True,
                                 padding_mode = "zeros")
        self.conv_5 = nn.Conv2d(192, 192, (3,3),
                                 stride = 1, padding = 1, dilation = 1, 
                                 groups = 1, bias = True,
                                 padding_mode = "zeros")
        self.conv_6 = nn.Conv2d(192, 192, (3,3),
                                 stride = 2, padding = 1, dilation = 1, 
                                 groups = 1, bias = True,
                                 padding_mode = "zeros")

        self.conv_7 = nn.Conv2d(192, 192, (3,3),
                                 stride = 1, padding = 0, dilation = 1, 
                                 groups = 1, bias = True,
                                 padding_mode = "zeros")
        self.conv_8 = nn.Conv2d(192, 192, (3,3),
                                 stride = 1, padding = 1, dilation = 1, 
                                 groups = 1, bias = True,
                                 padding_mode = "zeros")
        self.conv_9 = nn.Conv2d(192, 10, (3,3),
                                 stride = 1, padding = 1, dilation = 1, 
                                 groups = 1, bias = True,
                                 padding_mode = "zeros")
        
        self.avg_layer = nn.AvgPool2d(kernel_size = (6,6), 
                                       stride = (1,1), padding = 0)
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = F.relu(self.conv_4(x))
        x = F.relu(self.conv_5(x))
        x = F.relu(self.conv_6(x))
        x = F.relu(self.conv_7(x))
        x = F.relu(self.conv_8(x))
        x = F.relu(self.conv_9(x))
        res = F.softmax(self.flatten(self.avg_layer(x)))
        return res

class AllCNNC_short_kernel(nn.Module):
    def __init__(self):
        super(AllCNNC_short_kernel, self).__init__()
        
        self.conv_1 = nn.Conv2d(3, 32, (5,5),
                                 stride = 2, padding = 2, dilation = 1, 
                                 groups = 1, bias = True,
                                 padding_mode = "zeros")
        self.conv_2 = nn.Conv2d(32, 64, (5,5),
                                 stride = 2, padding = 2, dilation = 1, 
                                 groups = 1, bias = True,
                                 padding_mode = "zeros")
        self.conv_3 = nn.Conv2d(64, 64, (8,8),
                                 stride = 1, padding = 0, dilation = 1, 
                                 groups = 1, bias = True,
                                 padding_mode = "zeros")

        self.conv_4 = nn.Conv2d(64, 10, (1,1),
                                 stride = 1, padding = 0, dilation = 1, 
                                 groups = 1, bias = True,
                                 padding_mode = "zeros")
        
        self.avg_layer = nn.AvgPool2d(kernel_size = (1,1), 
                                       stride = (1,1), padding = 0)
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = F.relu(self.conv_4(x))
        res = F.softmax(self.flatten(self.avg_layer(x)))
        return res
if __name__ == "__main__":
    network_name = "pure_short"
    seed = 10
    
    # chose between: original, pure, pure_short
    torch.manual_seed(seed)
    import random
    random.seed(seed)    

    if device == "cuda":
                    
        torch.set_default_tensor_type('torch.cuda.FloatTensor')        
    else:
        torch.set_default_tensor_type('torch.FloatTensor') 
    
           
    model = AllCNNC_short_kernel().to(device)
    from torchsummary import summary
    
    #print(summary(model, (3,32,32)))
    
    print('### Training on CIFAR10')
    data_dir = './data'
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset_total = torchvision.datasets.CIFAR10(root=data_dir,
                                                  train=True,
                                                download=True,
                                                transform=transform)
    g_cuda = torch.Generator(device=device)
    trainset, valset = torch.utils.data.random_split(trainset_total,
                                                     [45000, 5000], generator = g_cuda)
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                           download=True,
                                           transform=transform)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=batch_size,
                                              shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    test_losses = []
    train_losses = []
    accuracies = []
    for e in range(epochs):
        model.train()
        train_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions ,targets)
            loss.backward()
            optimizer.step()
            train_loss += loss
        train_losses.append(train_loss/i)
        test_loss = 0.0
        total = 0.0
        correct = 0.0
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs).detach()
            loss = criterion(predictions,targets)
            test_loss += loss
            # calculate accuracy
            _, predicted = torch.max(predictions.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
        test_losses.append(test_loss/i)
        accuracies.append(100*correct/total)
        
        print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
        print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {test_loss / len(test_loader)}')
    
    # save
    torch.save(model.state_dict(), "./weights_backprop.pth")
    model.load_state_dict(torch.load("./weights_backprop.pth"))
    model.eval()
    
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(figsize = (10,8))
    axs.plot(test_losses, label = "test loss")
    axs.plot(train_losses, label = "train loss")
    axs.set_title("train_test_loss")
    axs.legend()
    fig.savefig("./training_backprop.png")
    
    fig, axs = plt.subplots(figsize = (10,8))
    axs.plot(accuracies, label = "accuracies")
    axs.set_title("accuracies")
    axs.legend()
    fig.savefig("./training_accuracies.png")
