import os.path
from collections import OrderedDict
from itertools import chain

import hydra
from omegaconf import OmegaConf

from models.local_loss_net import LocalLossNet
from models.local_loss_blocks import LocalLossBlock
from models import AllCNN
from utils.configuration import set_seed
from utils.data import get_datasets
from utils.models import load_best_model_from_exp_dir

import numpy as np
import torch
import pandas as pd
from torch import nn, optim
import matplotlib.pyplot as plt
import seaborn as sns

import representation_analysis_tools.intrinsic_dimension as id
import representation_analysis_tools.rsa as rsa
import representation_analysis_tools.utils as utils
import representation_analysis_tools.centered_kernel_alignment as cka


class Evaluation:

    def __init__(self, cfg, models, model_names, data_set, data_set_name="CIFAR-10"):

        self.cfg = cfg
        self.models_names = list(zip(models, model_names))

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
        for model in self.models_names:

            model[0].eval()
            if isinstance(model[0], LocalLossNet) or isinstance(model[0], LocalLossBlock):
                model[0].local_loss_eval()

            self.activations[model] = OrderedDict()
            if isinstance(model[0], LocalLossNet):
                self.named_modules[model] = model[0].get_base_inference_layers()
                if isinstance(model[0], AllCNN):
                    for i in range(len(self.named_modules[model])-1):
                        self.named_modules[model][i] = (f"Conv {i+1}", self.named_modules[model][i][1])
                    self.named_modules[model][-1] = ("AvgPool", self.named_modules[model][-1][1])
            else:
                self.named_modules[model] = list(model[0].named_modules())[1:]

    def plot_ide(self, ide_layers, model_name):

        sns.set_style('whitegrid')
        sns.set(rc={"figure.figsize": (10, 6)})
        dim_plot = sns.lineplot(data=ide_layers, x='Layer', y='Dimension', ci="sd", linewidth=2)
        dim_plot.set(title='Intrinsic Dimension of Network Layers')
        plt.xticks(rotation=20)
        if self.cfg.show_plot:
            plt.show()
        fig = dim_plot.get_figure()
        fig.savefig(os.path.join(self.cfg.save_dir, "ide", f"ide_{model_name}.png"))

    def plot_together(self, list_ide_layers):
        total_df = list_ide_layers[0]
        total_df['Model'] = self.models_names[0][1]
        for i in range(1, len(list_ide_layers)):
            list_ide_layers[i]['Model'] = self.models_names[i][1]
            total_df = total_df.append(list_ide_layers[i])
        sns.set(rc={"figure.figsize": (10, 6)})
        dim_plot = sns.lineplot(data=total_df, x='Layer', y='Dimension', ci="sd", hue='Model', linewidth=2)
        dim_plot.set(title='Intrinsic Dimension of Network Layers')
        dim_plot.set_xlim(left=0, right=list_ide_layers[0].nunique()['Layer']-1)
        plt.xticks(rotation=20)
        plt.tight_layout()
        if self.cfg.show_plot:
            plt.show()
        fig = dim_plot.get_figure()
        fig.savefig(os.path.join(self.cfg.save_dir, "ide", f"ide.png"))

    def ide_analysis(self):

        save_dir = os.path.join(self.cfg.save_dir, "ide")
        os.makedirs(save_dir, exist_ok=True)
        ide_layers_dfs = []
        for i, model in enumerate(self.models_names):
            ide_layers = utils.compute_from_activations(self.activations[model], id.computeID_unagg,
                                                        nres=self.cfg.ide.nres,
                                                        fraction=self.cfg.ide.fraction, verbose=False)

            ide_layers_dfs.append(pd.DataFrame({
                'Layer': list(
                    chain.from_iterable([[layer] * len(ids) for ((m, d, layer, e), ids) in ide_layers.items()])),
                'Dimension': list(chain.from_iterable([ids for ids in ide_layers.values()]))
            }))
            ide_layers_dfs[-1]['Model'] = model[1]
            ide_layers_dfs[i].to_csv(os.path.join(self.cfg.save_dir, "ide", f"{model[1]}-ide.csv"), index=False)
            if self.cfg.plot and not self.cfg.ide.plot_together:
                self.plot_ide(ide_layers_dfs[i], model[1])

        if self.cfg.plot:
            if self.cfg.ide.plot_together:
                self.plot_together(ide_layers_dfs)

    def plot_rdms_deprecated(self, corr_dist, model_name1, model_name2):
        fig, ax = plt.subplots(1, 1)
        img = ax.imshow(corr_dist[self.data_set_name][1])
        x_label_list = [x[2] for x in corr_dist[self.data_set_name][0]]

        ax.set_xticks([x for x in range(len(corr_dist[self.data_set_name][0]))])
        ax.set_yticks([x for x in range(len(corr_dist[self.data_set_name][0]))])
        ax.set_xticklabels(x_label_list)
        ax.set_yticklabels(x_label_list)
        print(x_label_list)

        plt.xticks(rotation=45)
        fig.colorbar(img)
        save_path = os.path.join(self.cfg.save_dir, "rdms",
                                 'correlation_matrix_{}_{}.png'.format(model_name1, model_name2))
        plt.savefig(save_path)
        plt.show()

    def plot_rdms(self, corr_df, model_name1, model_name2):

        mask = np.zeros_like(corr_df)
        mask[np.triu_indices_from(mask)] = True
        mask = np.logical_xor(mask, np.identity(mask.shape[0]))
        submask = np.zeros((int(mask.shape[0] / 2), int(mask.shape[1] / 2)))
        submask[np.triu_indices_from(submask)] = True
        submask = np.logical_xor(submask, np.identity(submask.shape[0]))
        mask[int(mask.shape[0]/2):, :int(mask.shape[1]/2)] = submask
        #print(mask)
        ax = sns.heatmap(corr_df, mask=mask)
        ax.set_title(f"Correlation of layers:\n(A) {model_name1} - (B) {model_name2}")
        plt.tight_layout()
        plt.show()
        save_path = os.path.join(self.cfg.save_dir, "rdms",
                                 'correlation_matrix_{}_{}.png'.format(model_name1, model_name2))
        plt.savefig(save_path)

    def make_dist_df(self, dist_matrix):
        names = []
        for info in dist_matrix[self.data_set_name][0]:
            if info[0] not in names:
                names.append(info[0])
        layers = ["(A) " + info[2] if info[0] == names[0] else "(B) " + info[2]
                  for info in dist_matrix[self.data_set_name][0]]
        return pd.DataFrame(dist_matrix[self.data_set_name][1], layers, layers)

    def rmds_analysis(self):

        save_dir = os.path.join(self.cfg.save_dir, "rdms")
        os.makedirs(save_dir, exist_ok=True)
        for i, model1 in enumerate(self.models_names):
            for j in range(i + 1, len(self.models_names)):
                model2 = self.models_names[j]
                input_rdms = rsa.input_rdms_from_activations(self.activations[model1])
                input_rdms.update(rsa.input_rdms_from_activations(self.activations[model2]))
                input_rdms = utils.separate_data_names(input_rdms)
                corr_dist = rsa.corr_dist_of_input_rdms(input_rdms)
                del input_rdms

                corr_df = self.make_dist_df(corr_dist)
                if self.cfg.plot:
                    self.plot_rdms(corr_df, model1[1], model2[1])

                del corr_dist

    def plot_cka_deprecated(self, linear_cka_dist_mat, model_name1, model_name2):

        fig, ax = plt.subplots(1, 1)
        img = ax.imshow(linear_cka_dist_mat[self.data_set_name][1])
        x_label_list = [x[2] for x in linear_cka_dist_mat[self.data_set_name][0]]

        ax.set_xticks([x for x in range(len(linear_cka_dist_mat[self.data_set_name][0]))])
        ax.set_yticks([x for x in range(len(linear_cka_dist_mat[self.data_set_name][0]))])
        ax.set_xticklabels(x_label_list)
        ax.set_yticklabels(x_label_list)

        plt.xticks(rotation=45)
        fig.colorbar(img)
        save_path = os.path.join(self.cfg.save_dir, "cka",
                                 'linear_cka_matrix_{}_{}.png'.format(model_name1, model_name2))
        plt.savefig(save_path)
        plt.show()

    def plot_cka(self, cka_dist_df, model_name1, model_name2):
        mask = np.zeros_like(cka_dist_df)
        mask[np.triu_indices_from(mask)] = True
        mask = np.logical_xor(mask, np.identity(mask.shape[0]))
        submask = np.zeros((int(mask.shape[0] / 2), int(mask.shape[1] / 2)))
        submask[np.triu_indices_from(submask)] = True
        submask = np.logical_xor(submask, np.identity(submask.shape[0]))
        mask[int(mask.shape[0] / 2):, :int(mask.shape[1] / 2)] = submask
        # print(mask)
        ax = sns.heatmap(cka_dist_df, mask=mask)
        ax.set_title(f"CKA of layers:\n(A) {model_name1} - (B) {model_name2}")
        plt.tight_layout()
        plt.show()
        save_path = os.path.join(self.cfg.save_dir, "cka",
                                 'linear_cka_matrix_{}_{}.png'.format(model_name1, model_name2))
        plt.savefig(save_path)


    def cka_outer_analysis(self):

        save_dir = os.path.join(self.cfg.save_dir, "cka")
        os.makedirs(save_dir, exist_ok=True)
        for i, model1 in enumerate(self.models_names):
            for j in range(i + 1, len(self.models_names)):
                model2 = self.models_names[j]
                outer_prod_triu_arrays = cka.outer_product_triu_array_from_activations(self.activations[model1])
                outer_prod_triu_arrays.update(cka.outer_product_triu_array_from_activations(self.activations[model2]))
                outer_prod_triu_arrays_seprated = utils.separate_data_names(outer_prod_triu_arrays)
                del outer_prod_triu_arrays
                linear_cka_dist_mat = cka.matrix_of_linear_cka(outer_prod_triu_arrays_seprated)
                del outer_prod_triu_arrays_seprated
                # linear_cka_embedding = utils.repr_dist_embedding(linear_cka_dist_mat)
                cka_dist_df = self.make_dist_df(linear_cka_dist_mat)
                if self.cfg.plot:
                    self.plot_cka(cka_dist_df, model1[1], model2[1])
                del linear_cka_dist_mat

    def evaluate(self):

        set_seed(self.cfg.seed)

        running_loss = 0.0
        total = 0.0
        correct = 0.0

        print('Starting evaluation')

        for i, (inputs, labels) in enumerate(self.data_loader, 0):

            for model in self.models_names:
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

            if self.cfg.rmds.calculate:
                self.rmds_analysis()

            if self.cfg.cka.calculate:
                self.cka_outer_analysis()

            print('Finished evaluation')

            return


@hydra.main(config_path="../configs/analysis", config_name="config.yaml")
def main(cfg: OmegaConf):
    OmegaConf.set_struct(cfg, False)
    train_set, _ = get_datasets(cfg.dataset, cfg.data_dir)
    models = []
    names = ["Full Backprop", "Prediction Local Loss", "Prediction & Similarity Local Loss"]
    for direc in cfg.evaluation.model_dirs:
        model, params = load_best_model_from_exp_dir(direc)
        models.append(model)
        if len(names) < 1:
            if params.model.loss.backprop:
                name = params.model.name + "-backprop"
            else:
                name = params.model.name + "-" + params.model.loss.loss_sup
            names.append(name)

    agent = Evaluation(cfg.evaluation, models=models, model_names=names, data_set=train_set,
                       data_set_name=cfg.dataset.name)
    agent.evaluate()


if __name__ == "__main__":
    main()
