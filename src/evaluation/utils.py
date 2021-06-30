import collections
import collections.abc
import torch
import warnings
import numpy as np
from functools import partial
from sklearn.manifold import MDS
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics.pairwise import euclidean_distances

# from umap import UMAP

ActivationKind = collections.namedtuple(
    'ActivationKind', ['model_name', 'data_name', 'layer_name', 'epoch'])


class TrackingFlag():
    def __init__(self, active, model_name, data_name, epoch):
        self.active = active
        self.model_name = model_name
        self.data_name = data_name
        self.epoch = epoch


def activations_labels():
    return collections.defaultdict(list)


def track_activations_labels(activations_labels, targets, tf):
    act_kind = ActivationKind(tf.model_name, tf.data_name, None, tf.epoch)

    if not isinstance(targets, list):
        targets = [targets]

    targets_np = []
    for target in targets:
        targets_np.append(target.detach().clone().cpu().numpy())

    activations_labels[act_kind].append(np.array(targets_np))


# TODO: implement RSA as forward hook
# a dictionary that keeps saving the activations as they come
def track_activations_(tf, differentiable=False):
    activations = collections.defaultdict(list)

    def save_activation(name, mod, _, out):
        if tf.active:
            act_kind = ActivationKind(tf.model_name, tf.data_name, name, tf.epoch)
            activations[act_kind].append(out.clone().detach().cpu())

    def save_activation_diff(name, mod, inp, out):
        _ = mod
        _ = inp
        if tf.active:
            activations[name].append(out.clone())

    return activations, save_activation if not differentiable else save_activation_diff


def track_activations(named_modules, tf, differentiable=False):
    activations, save_activation = track_activations_(tf, differentiable=differentiable)
    handles = []

    # NOTE: gradient passes through forward_hook
    for name, m in named_modules:
        handles.append(m.register_forward_hook(partial(save_activation, name)))

    return activations, handles


def track_sample_wise_grads(named_modules, tf):
    pass  # autograd_hacks.add_hooks_from_list(named_modules)


def flatten_activations(activations):
    # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
    activations = {
        name: torch.cat(outputs, 0).numpy()
        for name, outputs in activations.items()
    }
    return activations


def compute_from_activations(activations, metric_fn, already_computed_layers=None, **kwargs):
    computed_layers = collections.OrderedDict()

    if not isinstance(activations.get(list(activations.keys())[0]), np.ndarray):
        activations = flatten_activations(activations)
    for name, acts in activations.items():
        computed_layers[name] = metric_fn(acts, **kwargs)

    if already_computed_layers is not None:
        already_computed_layers.update(computed_layers)
    else:
        already_computed_layers = computed_layers

    return already_computed_layers


def separate_data_names(computed_metric):
    computed_metric_separate = {}
    for act_kind, computed_metric_inner in computed_metric.items():
        if act_kind.data_name not in computed_metric_separate:
            computed_metric_separate[act_kind.data_name] = {}

        computed_metric_separate[act_kind.data_name].update({act_kind: computed_metric_inner})

    return computed_metric_separate


def matrix_of_dist(dist_fn, features):
    if isinstance(features, dict) and isinstance(
            list(features.keys())[0], str):
        mat_of_dist_dict = {}
        for data_name, inner_features in features.items():
            mat_of_dist_dict[data_name] = dist_fn(inner_features)
        return mat_of_dist_dict

    return dist_fn(features)


def repr_dist_embedding(repr_dist, embedding='mds', specific_mds_dimensions=[0, 1], calc_mds_emb_error=False):
    def repr_dist_embedding_(repr_dist, embedding):
        if embedding == 'mds':
            # proportion of variance explained by embedding
            repr_dist_centered = KernelCenterer().fit_transform(np.square(repr_dist) * (-0.5))
            eigvals = np.linalg.eigvalsh(repr_dist_centered)
            eigvals.sort()
            eigvals = eigvals[::-1]
            print(
                'Proportion of variance explained by first, second, (third) and both (, all three) dimension(s): ({}, {}, ({}) = {}, ({}))'
                    .format(
                    np.sum(eigvals[0]) / np.sum(eigvals),
                    np.sum(eigvals[1]) / np.sum(eigvals),
                    np.sum(eigvals[2]) / np.sum(eigvals),
                    np.sum(eigvals[:2]) / np.sum(eigvals),
                    np.sum(eigvals[:3]) / np.sum(eigvals)))

            try:
                if specific_mds_dimensions is None: raise ValueError("Use scikit MDS.")
                warnings.filterwarnings("error")
                eigvals, eigvec = np.linalg.eigh(repr_dist_centered)
                eigvals, eigvec = eigvals[::-1], eigvec[:, ::-1]
                eigvals[np.isclose(eigvals, 0)] = 0
                if len(np.intersect1d(specific_mds_dimensions, np.where(eigvals < 0.))) > 0:
                    print("You want to embed in dimensions with negative eigenvalues.")
                else:
                    eigvals = np.clip(eigvals, 0., None, out=eigvals)
                eigvals_sqr = np.sqrt(eigvals, out=eigvals)
                eigvec[:] *= eigvals_sqr
                mdm_transformed = eigvec[:, specific_mds_dimensions] @ np.diag(eigvals_sqr[specific_mds_dimensions])
            except Exception as e:
                warnings.resetwarnings()

                if isinstance(e, RuntimeWarning): print("Use scikit MDS as classical variant fails.")

                embedding = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
                mdm_transformed = embedding.fit_transform(repr_dist)

            if calc_mds_emb_error:
                # TODO: seems to be not correct because it depends on scale of embedding

                print("Relative error of embedding")
                # http://www.mbfys.ru.nl/~robvdw/CNP04/LAB_ASSIGMENTS/LAB05_CN05/MATLAB2007b/stats/html/cmdscaledemo.html
                # maxrelerr = max(abs(D - pdist(Y(:,1:2)))) / max(D)
                edists = euclidean_distances(mdm_transformed, mdm_transformed)
                maxrelerr = np.max(np.abs(repr_dist - edists)) / repr_dist.max()
                print("maxrelerr:", maxrelerr)
                meanabserr = np.mean(np.abs(repr_dist - edists))
                print("meanabserr:", meanabserr)
                meanrelerr = np.mean(np.abs((repr_dist - edists) / (repr_dist + 1e-9)))
                print("meanrelerr:", meanrelerr)

                # Eigenspectrum plot
                # plt.plot(eigvals)
                # plt.show()

                # is the B in http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/tutorials/mvahtmlnode99.html
        elif embedding == 'umap':
            mdm_transformed = UMAP(n_neighbors=50, random_state=42, metric='precomputed').fit_transform(
                repr_dist)
        else:
            raise ValueError()

        return mdm_transformed

    if isinstance(repr_dist, dict):
        mdm_transformed_dict = {}
        for data_name, inner_repr_dist in repr_dist.items():
            mdm_transformed_dict[data_name] = (inner_repr_dist[0], repr_dist_embedding_(inner_repr_dist[1], embedding))
        return mdm_transformed_dict

    return repr_dist_embedding_(repr_dist, embedding)


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def cos_similarity(A, B):
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)

    return A.dot(B)
