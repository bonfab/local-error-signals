from collections import OrderedDict

from omegaconf import OmegaConf

from models.local_loss_net import LocalLossNet
from models.local_loss_blocks import LocalLossBlock
from utils.configuration import set_seed
from utils.data import get_datasets
from utils.models import load_best_model_from_exp_dir

import numpy as np
import torch
import pandas as pd
import torchvision
from torchvision import transforms
from torch import nn, optim
import matplotlib.pyplot as plt
import seaborn as sns

from evaluation.utils import track_activations
import representation_analysis_tools.intrinsic_dimension as id
import representation_analysis_tools.rsa as rsa
import representation_analysis_tools.utils as utils
import representation_analysis_tools.centered_kernel_alignment as cka


class Evaluation:

    def __init__(self, cfg, models, model_names, data_set, data_set_name="CIFAR-10"):

        self.cfg = cfg
        self.models = [(model, name) for model, name in zip(models, model_names)]

        self.num_classes = cfg.num_classes
        self.batch_size = cfg.batch_size

        self.data_set = data_set
        self.data_set_name = data_set_name
        set_seed(self.cfg.seed)
        self.data_loader = torch.utils.data.DataLoader(self.data_set, batch_size=self.batch_size,
                                                       shuffle=True, num_workers=0,
                                                       worker_init_fn=lambda worker_id: np.random.seed(
                                                           self.cfg.seed + worker_id)
                                                       )
        self.criterion = nn.CrossEntropyLoss()

        self.activations = {}
        self.named_modules = {}
        for model in self.models:

            model[0].eval()
            if isinstance(model[0], LocalLossNet) or isinstance(model[0], LocalLossBlock):
                model[0].local_loss_eval()

            self.activations[model] = OrderedDict()
            if isinstance(model[0], LocalLossNet):
                self.named_modules[model] = model[0].get_base_inference_layers()
            else:
                self.named_modules[model] = list(model[0].named_modules())[1:]

    def plot_ide(self, ide_layers, model_name):

        sns.set_style('whitegrid')
        sns.set(rc={"figure.figsize": (10, 6)})
        dim_plot = sns.lineplot(data=ide_layers, x='layer', y='dimension', linewidth=2)
        dim_plot.set(title='Intrinsic Dimension of Network Layers')
        plt.xticks(rotation=25)
        if self.cfg.show_plot:
            plt.show()
        fig = dim_plot.get_figure()
        fig.savefig(f"ide_{model_name}")

    def ide_analysis(self):

        for model in self.models:
            ide_layers = utils.compute_from_activations(self.activations[model], id.computeID, nres=self.cfg.ide.nres,
                                                        fraction=self.cfg.ide.fraction, verbose=False)

            ide_layers_df = pd.DataFrame({
                'layer': [layer for (m, d, layer, e) in ide_layers.keys()],
                'dimension': [mean for (mean, variance) in ide_layers.values()],
                'variance': [variance for (mean, variance) in ide_layers.values()]
            })

            #ide_layers_df.to_csv(self.cfg.ide.csv_save_path, index=False)

            if self.cfg.plot:
                self.plot_ide(ide_layers_df, model[1])

            del ide_layers_df, ide_layers

    def plot_rdms(self, corr_dist, model_name1, model_name2):

        fig, ax = plt.subplots(1, 1)
        img = ax.imshow(corr_dist[self.data_set_name][1])
        x_label_list = [x[2] for x in corr_dist[self.data_set_name][0]]

        ax.set_xticks([x for x in range(len(corr_dist[self.data_set_name][0]))])
        ax.set_yticks([x for x in range(len(corr_dist[self.data_set_name][0]))])
        ax.set_xticklabels(x_label_list)
        ax.set_yticklabels(x_label_list)

        plt.xticks(rotation=45)
        fig.colorbar(img)
        save_path = 'correlation_matrix_{}_{}.png'.format(model_name1, model_name2)
        plt.savefig(save_path)
        plt.show()

    def rmds_analysis(self):

        for i, model1 in enumerate(self.models):
            for j in range(i+1, len(self.models)):
                model2 = self.models[j]
                input_rdms = rsa.input_rdms_from_activations(self.activations[model1])
                input_rdms.update(rsa.input_rdms_from_activations(self.activations[model2]))
                input_rdms = utils.separate_data_names(input_rdms)
                corr_dist = rsa.corr_dist_of_input_rdms(input_rdms)
                del input_rdms
                # mdm_embedding = utils.repr_dist_embedding(corr_dist)
                if self.cfg.plot:
                    self.plot_rdms(corr_dist, model1[1], model2[1])

                del corr_dist

    def plot_cka(self, linear_cka_dist_mat, model_name1, model_name2):

        fig, ax = plt.subplots(1, 1)
        img = ax.imshow(linear_cka_dist_mat[self.data_set_name][1])
        x_label_list = [x[2] for x in linear_cka_dist_mat[self.data_set_name][0]]

        ax.set_xticks([x for x in range(len(linear_cka_dist_mat[self.data_set_name][0]))])
        ax.set_yticks([x for x in range(len(linear_cka_dist_mat[self.data_set_name][0]))])
        ax.set_xticklabels(x_label_list)
        ax.set_yticklabels(x_label_list)

        plt.xticks(rotation=45)
        fig.colorbar(img)
        save_path = 'linear_cka_matrix_{}_{}.png'.format(model_name1, model_name2)
        plt.savefig(save_path)
        plt.show()

    def cka_outer_analysis(self):

        for i, model1 in enumerate(self.models):
            for j in range(i+1, len(self.models)):
                model2 = self.models[j]
                outer_prod_triu_arrays = cka.outer_product_triu_array_from_activations(self.activations[model1])
                outer_prod_triu_arrays.update(cka.outer_product_triu_array_from_activations(self.activations[model2]))
                outer_prod_triu_arrays_seprated = utils.separate_data_names(outer_prod_triu_arrays)
                del outer_prod_triu_arrays
                linear_cka_dist_mat = cka.matrix_of_linear_cka(outer_prod_triu_arrays_seprated)
                del outer_prod_triu_arrays_seprated
                # linear_cka_embedding = utils.repr_dist_embedding(linear_cka_dist_mat)

                if self.cfg.plot:
                    self.plot_cka(linear_cka_dist_mat, model1[1], model2[1])
                del linear_cka_dist_mat

    def evaluate(self):

        set_seed(self.cfg.seed)

        running_loss = 0.0
        total = 0.0
        correct = 0.0

        print('Starting evaluation')

        for i, (inputs, labels) in enumerate(self.data_loader, 0):

            for model in self.models:
                tracking_flag = utils.TrackingFlag(True, model[1], self.data_set_name, None)
                activations_, handles = utils.track_activations(self.named_modules[model], tracking_flag)

                with torch.no_grad():
                    outputs = model[0](inputs)
                loss = self.criterion(outputs, labels)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                acc = 100. * correct / total
                running_loss += loss.item()

                self.activations[model].update(activations_)
                self.activations[model] = utils.flatten_activations(self.activations[model])

                for handle in handles:
                    handle.remove()

            if self.cfg.ide.calculate:
                self.ide_analysis()

            if self.cfg.rsa.calculate:
                self.rmds_analysis()

            if self.cfg.cka.calculate:
                self.cka_outer_analysis()

            print('Finished evaluation')

            return


if __name__ == "__main__":
    cfg = OmegaConf.create({
        "data": {
            "name": "CIFAR-10"
        },
        "model_dirs": ["../2021-06-18_12-50-30/0",
                       "../2021-06-18_12-50-30/2",
                       "../2021-06-18_12-50-30/3"],
        "batch_size": 200,
        "num_classes": 10,
        "seed": 1234,
        "plot": True,
        "show_plot": True,
        "ide": {
            "calculate": True,
            "nres": 3,
            "fraction": 0.5,
            "csv_save_path": './internal_dimensions.csv',
            "plot_save_path": 'internal_dimensions.png'
        },
        "rsa": {
            "calculate": True
        },
        "cka": {
            "calculate": True
        }

    })

    train_set, _ = get_datasets(cfg.data, "../data")
    models = []
    names = []
    for dirc in cfg.model_dirs:
        model, params = load_best_model_from_exp_dir(dirc)
        models.append(model)
        if params.model.loss.backprop:
            name = params.model.name + "-backprop"
        else:
            name = params.model.name + "-" + params.model.loss.loss_sup
        names.append(name)

    agent = Evaluation(cfg, models=models, model_names=names, data_set=train_set, data_set_name=cfg.data.name)
    agent.evaluate()
