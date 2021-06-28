import wandb
import os
import numpy as np
from pathlib import Path
from representation_analysis_tools.lazydict import RootPathLazyDictionary, CallWithPath
from cloudpickle import pickle
from functools import partial


def activation_kind_to_npy_file_name(actKind):
    return actKind.__str__().lower().replace('(', '_').replace(')', '').replace(', ', '_').replace('"', '_').replace("'", '_').replace('=', '_')+'.npy'


def loss_acc_log(use_wandb):
    def loss_acc_log_(name, loss, acc, step_name=None, step=None):
        log_d = {f"{name} Loss": loss, f"{name} Accuracy": acc}
        if step_name:
            log_d.update({f"{step_name}": step})
        #wandb.log(log_d)
    return loss_acc_log_ if use_wandb else lambda *args, **kwargs: None


def repr_sim_log(use_wandb):
    def repr_sim_log_(repr_kind, repr_dict, step_name=None, step=None):
        log_d = {}
        for data_name, repr_sim in repr_dict.items():
            log_d.update({f"{repr_kind} {data_name}": repr_sim})
        
        if step_name:
            log_d.update({f"{step_name}": step})
        #wandb.log(log_d)
    return repr_sim_log_ if use_wandb else lambda *args, **kwargs: None


def log_data(data, path, root_path=Path("./logs")):
    npy_files_path = Path(str(path.with_suffix(''))+'_npy_files')
    os.makedirs(root_path/npy_files_path, exist_ok=True)
    dict_to_npy = RootPathLazyDictionary()

    for actKind, value in data.items():
        npy_path = npy_files_path/activation_kind_to_npy_file_name(actKind)
        if not os.path.exists(root_path/npy_path):
            np.save(root_path/npy_path, value)
        dict_to_npy[actKind] = CallWithPath(partial(np.load, mmap_mode="c"), npy_path)

    dict_to_npy['root_path'] = root_path

    if os.path.exists(root_path/path):
        dict_to_npy_old = load_log_data(path, root_path=root_path)
        dict_to_npy.update(dict_to_npy_old)

    with open(root_path/path, 'wb') as handle:
        pickle.dump(dict_to_npy, handle)


def load_log_data(path, root_path=Path("./logs")):
    path = root_path/path
    with open(path, 'rb') as handle:
        dict_to_npy = pickle.load(handle)
    dict_to_npy['root_path'] = root_path
    return dict_to_npy


def log_similarity_metric(data, name, model_name, path=Path("similarity_metrics"), **kwargs):
    name_ = name + ".pickle"
    log_data(data, path/model_name/name_, **kwargs)


def load_similarity_metric(name, model_name, path=Path("similarity_metrics"), **kwargs):
    name_ = name + ".pickle"
    return load_log_data(path/model_name/name_, **kwargs)


def log_distance_matrices(data, name, model_name='', path=Path("distance_matrices"), root_path=Path("./logs")):
    name_ = name + ".pickle"
    folder_path = root_path/path/model_name
    os.makedirs(folder_path, exist_ok=True)
    
    with open(folder_path/name_, 'wb') as handle:
        pickle.dump(data, handle)


def load_distance_matrices(name, model_name='', path=Path("distance_matrices"), root_path=Path("./logs")):
    name_ = name + ".pickle"
    with open(root_path/path/model_name/name_, 'rb') as handle:
        return pickle.load(handle)
