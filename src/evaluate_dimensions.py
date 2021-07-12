import os.path
import re
from collections import OrderedDict
from itertools import chain

import hydra
from omegaconf import OmegaConf

from models.local_loss_net import LocalLossNet
from models.local_loss_blocks import LocalLossBlock
from models import AllCNN
from theoretical_framework_for_target_propagation.lib.conv_networks_AllCNN import DDTPPureConvAllCNNC, \
    DDTPPureShortCNNC_kernelmod
from theoretical_framework_for_target_propagation.AllCNNC_backprop import AllCNNC_short_kernel
from theoretical_framework_for_target_propagation.allCNNC_main_last import args as args_original
from theoretical_framework_for_target_propagation.allCNNC_main_last import \
    load_network_w_weights as load_network_w_weights_original
from theoretical_framework_for_target_propagation.allCNNC_main_pureshort import args as args_short
from theoretical_framework_for_target_propagation.allCNNC_main_pureshort import \
    load_network_w_weights as load_network_w_weights_short
from utils.configuration import set_seed
from utils.data import get_datasets
from utils.models import load_best_model_from_exp_dir

import numpy as np
import torch
import pandas as pd
from torch import nn
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

        def target_prop_filter_modules(modules):
            filtered_modules = modules[2::2]
            filtered_modules.append(modules[-2])
            return filtered_modules

        def filter_flatten(modules):
            to_delete = []
            for j in range(len(modules)):
                if modules[j][0].lower().__contains__("flatten"):
                    to_delete.append(j)
            for j in to_delete:
                del modules[j]
            return modules

        for model in self.models_names:
            model[0].eval()
            if isinstance(model[0], LocalLossNet) or isinstance(model[0], LocalLossBlock):
                model[0].local_loss_eval()
            self.activations[model] = OrderedDict()

            if isinstance(model[0], LocalLossNet):
                self.named_modules[model] = model[0].get_base_inference_layers()
            elif isinstance(model[0], AllCNNC_short_kernel):
                self.named_modules[model] = filter_flatten((list(model[0].named_modules())[1:]))
            elif isinstance(model[0], (DDTPPureConvAllCNNC, DDTPPureShortCNNC_kernelmod)):
                self.named_modules[model] = filter_flatten(
                    target_prop_filter_modules((list(model[0].named_modules())[1:])))
            else:
                self.named_modules[model] = list(model[0].named_modules())[1:]

            if isinstance(model[0], (AllCNN, DDTPPureShortCNNC_kernelmod, DDTPPureConvAllCNNC, AllCNNC_short_kernel)):
                for i in range(len(self.named_modules[model]) - 1):
                    self.named_modules[model][i] = (f"C {i + 1}", self.named_modules[model][i][1])
                self.named_modules[model][-1] = ("Pool", self.named_modules[model][-1][1])

    def make_save_dirs(self, name):
        save_dir = os.path.join(self.cfg.save_dir, name)
        plot_dir = os.path.join(save_dir, "plots")
        csv_dir = os.path.join(save_dir, "csvs")
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)

        return csv_dir, plot_dir

    def plot_ide(self, ide_layers, model_name, save_dir):

        plt.close()
        sns.set(rc={"figure.figsize": (11, 5)})
        sns.set_style("whitegrid", {'axes.grid': False})
        dim_plot = sns.lineplot(data=ide_layers, x='Layer', y='Dimension', ci="sd", linewidth=2)
        plt.suptitle('Intrinsic Dimension of Network Layers', weight='bold')
        dim_plot.set(xlim=(0, ide_layers.Layer.nunique()-1))
        if self.cfg.show_plot:
            plt.show()
        fig = dim_plot.get_figure()
        fig.savefig(os.path.join(save_dir, f"ide_{model_name}.png"))

    def plot_together(self, list_ide_layers, save_dir, plot_pool=True, split_multiple=True):

        plt.close()
        sns.set_style('whitegrid')
        total_dfs = {}
        for i in range(len(list_ide_layers)):
            list_ide_layers[i]['Model'] = self.models_names[i][1]
            if split_multiple:
                nunique = list_ide_layers[i].Layer.nunique()
                if nunique in total_dfs.keys():
                    total_dfs[nunique] = total_dfs[nunique].append(list_ide_layers[i])
                else:
                    total_dfs[nunique] = list_ide_layers[i]
            else:
                if i > 0:
                    total_dfs[0].append(list_ide_layers[i])
                else:
                    total_dfs[0] = list_ide_layers[i]

        if not plot_pool:
            for key, df in total_dfs.items():
                total_dfs[key] = df[~df.Layer.str.contains("Pool")]
                total_dfs[key] = df[~df.Layer.str.contains("pool")]

        print(total_dfs)
        fig, axes = plt.subplots(1, len(total_dfs.values()))
        sns.set(rc={"figure.figsize": (10, 6)})
        for i, df in enumerate(total_dfs.values()):
            if i % 2 == 1:
                palette = "colorblind"
            else:
                palette = "Set2"
            dim_plot = sns.lineplot(ax=axes[i], data=df, x='Layer', y='Dimension', ci="sd", hue='Model',
                                        linewidth=2, palette=palette)
            axes[i].set_xlim(left=0, right=df.Layer.nunique() - 1)
            axes[i].legend(loc="lower left")
            axes[i].grid(False)
            if i > 0:
                axes[i].set_ylabel(None)
                axes[i].set_yticklabels([])
        plt.suptitle('Intrinsic Dimension of Network Layers', weight='bold')
        plt.tight_layout()
        if self.cfg.show_plot:
            plt.show()
        fig = dim_plot.get_figure()
        fig.savefig(os.path.join(save_dir, f"ide.png"))

    def ide_analysis(self):

        csv_dir, plot_dir = self.make_save_dirs("ide")
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
            ide_layers_dfs[i].to_csv(os.path.join(csv_dir, f"{model[1]}_ide.csv"), index=False)
            if self.cfg.plot:
                self.plot_ide(ide_layers_dfs[i], model[1], plot_dir)

        if self.cfg.plot and self.cfg.ide.plot_together:
            self.plot_together(ide_layers_dfs, plot_dir)

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

    def plot_rdms(self, corr_df, model_name1, model_name2, save_dir):

        plt.close()
        mask = np.zeros_like(corr_df)
        mask[np.triu_indices_from(mask)] = True
        mask = np.logical_xor(mask, np.identity(mask.shape[0]))
        """submask = np.zeros((int(mask.shape[0] / 2), int(mask.shape[1] / 2)))
        submask[np.triu_indices_from(submask)] = True
        submask = np.logical_xor(submask, np.identity(submask.shape[0]))
        mask[int(mask.shape[0] / 2):, :int(mask.shape[1] / 2)] = submask"""
        # print(mask)
        ax = sns.heatmap(corr_df, mask=mask)
        #ax.set_title(f"(A) {model_name1} - (B) {model_name2}")
        plt.suptitle(f"Correlation of layers\n{model_name1} - {model_name2}", weight="bold")
        plt.tight_layout()
        plt.show()
        save_path = os.path.join(save_dir,
                                 '{}_{}_rdms.png'.format(model_name1, model_name2))
        plt.savefig(save_path)

    def make_dist_df(self, dist_matrix, model_name1, model_name2):
        names = []
        ann1 = re.search("\(([A-Za-z0-9]+)\)", model_name1)
        ann2 = re.search("\(([A-Za-z0-9]+)\)", model_name2)
        if ann1 is not None and ann2 is not None:
            ann1 = ann1.group(1)
            ann2 = ann2.group(1)
        else:
            ann1 = "A"
            ann2 = "B"
        for info in dist_matrix[self.data_set_name][0]:
            if info[0] not in names:
                names.append(info[0])
        layers = [f"({ann1}) " + info[2] if info[0] == names[0] else f"({ann2}) " + info[2]
                  for info in dist_matrix[self.data_set_name][0]]
        return pd.DataFrame(dist_matrix[self.data_set_name][1], layers, layers)

    def rmds_analysis(self):

        csv_dir, plot_dir = self.make_save_dirs("rdms")

        for i, model1 in enumerate(self.models_names):
            for j in range(i + 1, len(self.models_names)):
                model2 = self.models_names[j]
                input_rdms = rsa.input_rdms_from_activations(self.activations[model1])
                input_rdms.update(rsa.input_rdms_from_activations(self.activations[model2]))
                input_rdms = utils.separate_data_names(input_rdms)
                corr_dist = rsa.corr_dist_of_input_rdms(input_rdms)
                del input_rdms

                corr_df = self.make_dist_df(corr_dist, model1[1], model2[1])
                corr_df.to_csv(os.path.join(csv_dir, f"{model1[1]}_{model2[1]}_rdms.csv"), index=False)
                if self.cfg.plot:
                    self.plot_rdms(corr_df, model1[1], model2[1], plot_dir)

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

    def plot_cka(self, cka_dist_df, model_name1, model_name2, save_dir):

        plt.close()
        mask = np.zeros_like(cka_dist_df)
        mask[np.triu_indices_from(mask)] = True
        mask = np.logical_xor(mask, np.identity(mask.shape[0]))
        # submask = np.zeros((int(mask.shape[0] / 2), int(mask.shape[1] / 2)))
        # submask[np.triu_indices_from(submask)] = True
        # submask = np.logical_xor(submask, np.identity(submask.shape[0]))
        # mask[int(mask.shape[0] / 2):, :int(mask.shape[1] / 2)] = submask
        # print(mask)
        ax = sns.heatmap(cka_dist_df, mask=mask)
        plt.suptitle(f"CKA of layers:\n{model_name1} - {model_name2}", weight="bold")
        plt.tight_layout()
        plt.show()
        save_path = os.path.join(save_dir,
                                 '{}_{}_cka.png'.format(model_name1, model_name2))
        plt.savefig(save_path)

    def cka_outer_analysis(self):

        csv_dir, plot_dir = self.make_save_dirs("cka")

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
                cka_dist_df = self.make_dist_df(linear_cka_dist_mat, model1[1], model2[1])
                cka_dist_df.to_csv(os.path.join(csv_dir, f"{model1[1]}_{model2[1]}_cka.csv"))
                if self.cfg.plot:
                    self.plot_cka(cka_dist_df, model1[1], model2[1], plot_dir)
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
    names = [
        "Full Backprop (A1)",
        "Prediction & Similarity Local Loss (B)",
        "Full Backprop Short (A2)",
        "Target Propagation Short (C)"
    ]
    for path in cfg.models.model_paths.local_loss:
        model, params = load_best_model_from_exp_dir(path)
        models.append(model)
        if len(names) < 1:
            if params.model.loss.backprop:
                name = params.model.name + "-backprop"
            else:
                name = params.model.name + "-" + params.model.loss.loss_sup
            names.append(name)
    if cfg.models.model_paths.target_prop.all_cnn_short_backprop is not None:
        for i, path in enumerate(cfg.models.model_paths.target_prop.all_cnn_short_backprop):
            model = AllCNNC_short_kernel()
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            models.append(model)
            if len(names) < 1:
                names.append(f"short-back-prop-{i}")

    if cfg.models.model_paths.target_prop.all_cnn_original is not None:
        for i, path in enumerate(cfg.models.model_paths.target_prop.all_cnn_original):
            args_original["no_cuda"] = True
            model = load_network_w_weights_original(args_original, path)
            models.append(model)
            if len(names) < 1:
                names.append(f"target-prop-original-{i}")
    if cfg.models.model_paths.target_prop.all_cnn_short is not None:
        for i, path in enumerate(cfg.models.model_paths.target_prop.all_cnn_short):
            args_short["no_cuda"] = True
            model = load_network_w_weights_short(args_short, path)
            models.append(model)
            if len(names) < 1:
                names.append(f"target-prop-short-{i}")

    agent = Evaluation(cfg.evaluation, models=models, model_names=names, data_set=train_set,
                       data_set_name=cfg.dataset.name)
    agent.evaluate()


if __name__ == "__main__":
    main()
