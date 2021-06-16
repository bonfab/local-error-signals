from utils.models import load_best_model_from_exp_dir

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch import nn, optim
import matplotlib.pyplot as plt

from evaluation.utils import track_activations
import evaluation.intrinsic_dimension as id
import evaluation.rsa as rsa
import evaluation.utils as utils
import evaluation.logs as logs
#import wandb



class Evaluation():

    def __init__(self, model_name, model, model_path):

        self.model_name = model_name
        self.model = model 
        self.model_path = model_path

        self.transform = transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.batch_size = 100
        
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size,
                                        shuffle=True, num_workers=2)

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size,
                                         shuffle=False, num_workers=2)
        
        self.classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

    def plot_ide(self, ide_layers):
        
        label = 'Intrinsic Dimension'
        xs = [] 
        ys = []
        for name, value in ide_layers.items():
            if "conv" in name:
                xs.append(name)
                ys.append(value)

        fig, ax = plt.subplots()

        #ax.plot(xs[:550], ys1[:550], label=algorithm1a, c='r')
        ax.plot(xs, ys, label=label, c='b')

        #ax2.plot(xs, ys2, label=algorithm1b, c='g')
        #ax2.plot(xs[:550], zs2[:550], label=algorithm2b, c='g')
        ax.set(xlabel='layer', ylabel='dimension')
        #ax2.set(xlabel='iteration', ylabel='percentage right')
        ax.legend(loc='upper left')
        #ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.9))
        ax.grid()
        #ax2.grid()
        save_path = 'ide_local_loss.png'
        plt.savefig(save_path)
        plt.show()

    def plot_rsa(self, rsa_layers):


        matrix = np.zeros((32,32))
        for i in range(32):
            for j in range(i,32):
                matrix[i][j] = self.cos_between(rsa_layers[i], rsa_layers[j])
                matrix[j][i] = matrix[i][j]

        plt.imshow(matrix, cmap="bwr", origin="lower")
        plt.colorbar()
        plt.grid("off")
        plt.xlabel("Network activations")
        plt.ylabel("Network activations")
        save_path = 'rsa_matrix.png'
        plt.savefig(save_path)
        plt.show()

    
    def evaluate(self):

        print('Load the weights...')


        self.model.load_state_dict(torch.load(self.model_path))

        running_loss = 0.0
        total = 0.0
        correct = 0.0

        activations = {}
        ide_layers = {}
        input_rdms = {}
        rsa_layers = {}

        #wandb.init(entity="NI-Project", project="local-error")

        print('Starting evaluation')

        trackingflag = utils.TrackingFlag(True, self.model_name, None, None)
        for name in list(self.model.named_modules())[1:]:
            ide_layers[name[0]] = 0.
            rsa_layers[name[0]] = 0.

        for i, data in enumerate(self.trainloader, 0):

            activations_, handles = utils.track_activations(list(self.model.named_modules())[1:], trackingflag)

            inputs, labels = data
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            acc = 100.*correct/total

            activations.update(activations_)
            activations = utils.flatten_activations(activations)

            for handle in handles:
                handle.remove()

            for name, acts in activations.items():
                mean, _ = id.computeID(acts)
                ide_layers[name[2]] += mean
                ide_layers[name[2]] = ide_layers[name[2]]/2
                #rsa_layers[name[2]] = (rsa.correlation_matrix(acts))

            activations = {}
            running_loss += loss.item()

            running_loss = 0.0
            

            return self.plot(ide_layers)
                


        
        print('Finished evaluation')



model = load_best_model_from_exp_dir("../2021-06-16_10-01-31/6")

agent = Evaluation(model_name = "AllCNN-normal", model = model,  model_path='./cifar_net_15.pth')

agent.evaluate()


    

