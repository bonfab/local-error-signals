from collections import OrderedDict

from omegaconf import OmegaConf

from models.local_loss_net import LocalLossNet
from models.local_loss_blocks import LocalLossBlock
from src.utils.configuration import set_seed
from utils.data import get_datasets
from utils.models import load_best_model_from_exp_dir

import numpy as np
import torch
import pandas as pd
import torchvision
from torchvision import transforms
from torch import nn, optim
import matplotlib.pyplot as plt

from evaluation.utils import track_activations
import evaluation.intrinsic_dimension as id
import evaluation.rsa as rsa
import evaluation.utils as utils
import evaluation.logs as logs
# import wandb


class Evaluation:

    def __init__(self, cfg, model, data_set):

        self.cfg = cfg
        self.model = model
        self.model.eval()
        if isinstance(self.model, LocalLossNet) or isinstance(self.model, LocalLossBlock):
            self.model.local_loss_eval()

        self.num_classes = cfg.num_classes
        self.batch_size = cfg.batch_size

        self.data_set = data_set
        self.data_loader = torch.utils.data.DataLoader(self.data_set, batch_size=self.batch_size,
                                                       shuffle=True, num_workers=cfg.data_loader_workers,
                                                       worker_init_fn=lambda worker_id: np.random.seed(
                                                           self.cfg.seed + worker_id)
                                                       )

        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse',
                        'ship', 'truck')

        self.criterion = nn.CrossEntropyLoss()

    def plot_ide(self, ide_layers):

        label = 'Intrinsic Dimension'
        xs = []
        ys = []
        for name, value in ide_layers.items():
            # if "conv" in name:
            xs.append(name)
            ys.append(value)

        fig, ax = plt.subplots()

        # ax.plot(xs[:550], ys1[:550], label=algorithm1a, c='r')
        ax.plot(xs, ys, label=label, c='b')

        # ax2.plot(xs, ys2, label=algorithm1b, c='g')
        # ax2.plot(xs[:550], zs2[:550], label=algorithm2b, c='g')
        ax.set(xlabel='layer', ylabel='dimension')
        # ax2.set(xlabel='iteration', ylabel='percentage right')
        ax.legend(loc='upper left')
        # ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.9))
        ax.grid()
        # ax2.grid()
        save_path = 'ide_local_loss.png'
        plt.savefig(save_path)
        plt.show()

    def plot_rsa(self, rsa_layers):

        matrix = np.zeros((32, 32))
        for i in range(32):
            for j in range(i, 32):
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

        set_seed(self.cfg.seed)

        print('Load the weights...')

        # self.model.load_state_dict(torch.load(self.model_path))

        running_loss = 0.0
        total = 0.0
        correct = 0.0

        activations = OrderedDict()
        ide_layers = OrderedDict()
        input_rdms = OrderedDict()
        rsa_layers = OrderedDict()

        # wandb.init(entity="NI-Project", project="local-error")

        print('Starting evaluation')

        if isinstance(self.model, LocalLossNet):
            named_modules = self.model.get_base_inference_layers()
        else:
            named_modules = list(self.model.named_modules())[1:]

        trackingflag = utils.TrackingFlag(True, None, None, None)
        for name in named_modules:
            ide_layers[name[0]] = 0.
            rsa_layers[name[0]] = 0.

        with torch.no_grad():
            for i, data in enumerate(self.data_loader, 0):

                activations_, handles = utils.track_activations(named_modules, trackingflag)

                inputs, labels = data
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                acc = 100. * correct / total

                activations.update(activations_)
                activations = utils.flatten_activations(activations)

                for handle in handles:
                    handle.remove()

                for name, acts in activations.items():
                    mean, _ = id.computeID(acts, verbose=False)
                    ide_layers[name[2]] += mean
                    ide_layers[name[2]] = ide_layers[name[2]] / 2
                    # rsa_layers[name[2]] = (rsa.correlation_matrix(acts))

                activations = {}
                running_loss += loss.item()

                running_loss = 0.0

                ide_layers_df = pd.DataFrame({
                    'layer': ide_layers.keys(),
                    'dimension': ide_layers.values()
                })
                ide_layers_df.to_csv('./internal_dimensions.csv', index=False)

                print('Finished evaluation')

                if self.cfg.plot:
                    self.plot_ide(ide_layers)

                return


if __name__ == "__main__":
    cfg = OmegaConf.create({
        "data": {
            "name": "CIFAR-10"
        },
        "model_dir": "../2021-06-18_12-50-30/3",
        "batch_size": 100,
        "num_classes": 10,
        "seed": 1234,
        "data_loader_workers": 3,
        "plot": True
    })

    train_set, _ = get_datasets(cfg.data, "../data")
    model = load_best_model_from_exp_dir("../2021-06-18_12-50-30/3")

    agent = Evaluation(cfg, model=model, data_set=train_set)

    agent.evaluate()
