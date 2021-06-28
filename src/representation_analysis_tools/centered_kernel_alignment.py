"""
from paper: Similarity of Neural Network Representations Revisited (Kornblith 2019), https://arxiv.org/abs/1905.00414
"""
import warnings
import numpy as np
import representation_analysis_tools.utils as utils
from scipy.spatial.distance import pdist

# check if ipywidgets is installed before importing tqdm.auto
# to ensure it won't fail and a progress bar is displayed
import importlib
if importlib.util.find_spec('ipywidgets') is not None:
    from tqdm.auto import tqdm
else:
    from tqdm import tqdm

NP_EPS = np.finfo(np.float32).eps

def matrix_of_linear_cka(features, from_outer_product_triu_array=True, verbose=True, scipy=False):
    def matrix_of_linear_cka_(features):
        from_dict = False
        act_kinds = None
        if isinstance(features, dict):
            from_dict = True
            act_kinds = list(features.keys())
            features = list(features.values())

        assert isinstance(features, list), f"features should be a list or a dict. is: {type(features)}"

        if not scipy:
            if from_outer_product_triu_array:
                linear_cka_mat = linear_cka_from_triu_array_batch(np.array(features))
            else:
                linear_cka_mat = np.zeros((len(features), len(features)))
                outer_iter = range(0, len(features))
                if verbose:
                    outer_iter = tqdm(outer_iter)
                for i in outer_iter:
                    inner_iter = range(i, len(features))
                    if verbose:
                        inner_iter = tqdm(inner_iter)
                    for j in inner_iter:
                        linear_cka_mat[i, j] = linear_cka(features[i], features[j])

                diagm = np.zeros_like(linear_cka_mat)
                np.fill_diagonal(diagm, np.diag(linear_cka_mat))
                linear_cka_mat = linear_cka_mat + linear_cka_mat.T - diagm
        else:
            if from_outer_product_triu_array:
                linear_cka_mat = linear_cka_from_triu_array_batch(np.array(features))
            else:
                linear_cka_mat = pdist(features, metric=linear_cka)

        if linear_cka_mat.shape == (2, 2):
            return linear_cka_mat[0, 1]

        linear_cka_mat = (linear_cka_mat + linear_cka_mat.T)/2.0

        assert np.allclose(np.diag(linear_cka_mat), 1.)
        linear_cka_mat[np.diag_indices_from(linear_cka_mat)] = 1.0

        assert np.allclose(linear_cka_mat[linear_cka_mat > 1.], 1.0)
        linear_cka_mat[linear_cka_mat > 1.] = 1.

        # from http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/tutorials/mvahtmlnode99.html
        # http://www.minerazzi.com/tutorials/distance-similarity-tutorial.pdf
        def is_pos_def(A):
            if np.allclose(A, A.T):
                try:
                    np.linalg.cholesky(A)
                    return True
                except np.linalg.LinAlgError:
                    print("Distance matrix not only positive eigenvalues.")
                    return False
            else:
                print("Distance matrix not symmetric.")
                return False

        _ = is_pos_def(linear_cka_mat) or warnings.warn(
            "Similarity matrix not positive semidefinite. Transformation questionable."
        )

        linear_cka_mat_dist = np.sqrt(2. - 2 * linear_cka_mat)
        if from_dict:
            return (act_kinds, linear_cka_mat_dist)

        return linear_cka_mat_dist

    return utils.matrix_of_dist(matrix_of_linear_cka_, features)


def linear_cka(x, y):
    # assumes centered means along colums (=samples over features -> zero-mean features)
    x_, y_ = (x - x.mean(axis=0)), (y - y.mean(axis=0))
    return (np.linalg.norm(y_.T @ x_)**2) / (np.linalg.norm(x_.T @ x_) * np.linalg.norm(y_.T @ y_) + NP_EPS)


def cka_feature_to_outer_product(x):
    x_ = (x - x.mean(axis=0))
    return x_ @ x_.T


def cka_feature_to_outer_product_triu_array(x):
    x_ = (x - x.mean(axis=0))
    xxt = x_ @ x_.T
    return xxt[np.triu_indices_from(xxt)]


def outer_prod_from_triu_array(xtriu):
    n = int(0.5 * (np.sqrt(8 * xtriu.shape[0] +1) -1))
    xxt = np.zeros((n, n))
    xxt[np.triu_indices_from(xxt)] = xtriu
    diagm = np.zeros_like(xxt)
    np.fill_diagonal(diagm, np.diag(xxt))
    xxt += xxt.T
    xxt -= diagm
    return xxt


def linear_cka_from_outer_product(xxt, yyt):
    # np.trace(yyt @ xxt) == np.dot(yyt.T.reshape(-1), xxt.reshape(-1)), but 'dot' is much faster
    return (np.dot(yyt.T.reshape(-1), xxt.reshape(-1)))/(np.linalg.norm(xxt) * np.linalg.norm(yyt) + NP_EPS)


def outer_product_triu_array_from_activations(activations, op_triu_dict=None):
    return utils.compute_from_activations(
        activations,
        cka_feature_to_outer_product_triu_array,
        already_computed_layers=op_triu_dict)


def linear_cka_from_triu_array(xtriu, ytriu, first_ind=None, second_ind=None, stored_norms=None):
    # nominator = pass
    n = int(0.5 * (np.sqrt(8 * xtriu.shape[0] +1) -1))
    triu_inds = np.triu_indices(n)
    diaginds = np.where(triu_inds[0] == triu_inds[1])

    if first_ind is not None or second_ind is not None:
        assert isinstance(
        stored_norms,
        np.ndarray), "If norms should be cached, then provide a stored_norms array"

    if first_ind and stored_norms[first_ind] >= 0:
        norm_first = stored_norms[first_ind]
    else:
        norm_first = np.sqrt(np.linalg.norm(xtriu)**2 * 2 - np.linalg.norm(xtriu[diaginds])**2)
        if first_ind:
            stored_norms[first_ind] = norm_first

    if second_ind and stored_norms[second_ind] >= 0:
        norm_second = stored_norms[second_ind]
    else:
        norm_second = np.sqrt(np.linalg.norm(ytriu)**2 * 2 - np.linalg.norm(ytriu[diaginds])**2)
        if second_ind:
            stored_norms[second_ind] = norm_second

    nominator = np.dot(xtriu, ytriu) * 2 - np.dot(xtriu[diaginds], ytriu[diaginds])
    return nominator / (norm_first * norm_second + NP_EPS)


def linear_cka_from_triu_array_batch(triu_array):
    n = int(0.5 * (np.sqrt(8 * triu_array.shape[1] +1) -1))
    triu_inds = np.triu_indices(n)
    diaginds = np.where(triu_inds[0] == triu_inds[1])

    diag_vals = triu_array[:, diaginds][:, 0, :]
    diag_dot = diag_vals @ diag_vals.T
    diag_dot = (diag_dot + diag_dot.T) / 2.
    nominator = (triu_array @ triu_array.T) * 2 - diag_dot
    norms = np.sqrt(np.linalg.norm(triu_array, axis=1)**2 *2 - np.linalg.norm(triu_array[:, diaginds][:, 0, :], axis=1)**2)
    denominator = norms[:, None] @ norms[:, None].T
    denominator = (denominator + denominator.T)/2.0
    return nominator / (denominator + NP_EPS)