import torch
import scipy.stats
# from sodeep import sodeep as losses
from representation_analysis_tools.blackbox_ranker import TrueRanker, rank_normalised

# TODO: try sodeep (https://github.com/technicolor-research/sodeep)


def cov(x, y):
    m = 0.5*(x+y)
    st = torch.stack([x-m, y-m])
    return st.T @ st


# do not use this! makes no sense i guess
# typically one has to provide the covariance matrix (which would describe the distribution)
# but what is the distribution of pearson correlation of randomly picked batches.
def mahalanobis_dist_sq(x, y):
    return ((x-y)[None, :] @ torch.inverse(cov(x, y)) @ (x-y)[:, None]).item()


@torch.jit.script
def pearsonr_vec(x, y):
    EPS = 1e-7
    xcentered = x - torch.mean(x)
    ycentered = y - torch.mean(y)

    r = torch.dot(xcentered, ycentered) * torch.rsqrt(
        torch.sum(xcentered**2) + EPS) * torch.rsqrt(torch.sum(ycentered**2) + EPS)

    return r


@torch.jit.script
def pearsonr_batch(x, upper_triu=torch.tensor(True)):
    EPS = 1e-7
    xf = torch.flatten(x, start_dim=1)
    xf_c = xf - torch.mean(xf, dim=1, keepdim=True)
    xf_c_std = xf_c * torch.rsqrt(torch.sum(xf_c ** 2, dim=1, keepdim=True) + EPS)
    r = xf_c_std @ torch.transpose(xf_c_std, 1, 0)

    if upper_triu:
        upper_triu_indices = torch.triu_indices(r.size()[0], r.size()[1], offset=1).unbind()
        return r[upper_triu_indices[0], upper_triu_indices[1]]
    return r


# TODO: very slow, try softrank (google): even slower
@torch.jit.script
def sigmoid_ranks(xf, inv_smoothness=torch.tensor(50.)):
    EPS = 1e-7
    # std = torch.std(xf) + EPS
    std = 1.

    indicator_mat = []
    for i in range(xf.size()[0]):
        indicator_mat.append(xf - torch.roll(xf, i))
    indicator_mat = torch.stack(indicator_mat, dim=1)
    indicator_mat_sigm = torch.sigmoid(inv_smoothness*(indicator_mat/std)) + 0.5
    indicator_mat_ = indicator_mat_sigm
    return indicator_mat_.sum(dim=1)


def ranks(xf):
    return torch.unique(xf, sorted=True, return_inverse=True)[1].float()


def representation_similarity(x, y, repr_kind="rsa"):
    repr_kind_options = ("rsa", "rsa_no_ranks", "cka", "rv2")
    assert repr_kind in repr_kind_options, f"Invalid kind. Options: {repr_kind_options}"

    # just to be save 
    x = x.detach()
    y = y.detach()

    if repr_kind == "rsa":
        return scipy.stats.spearmanr(pearsonr_batch(x).numpy(), pearsonr_batch(y).numpy())
    if repr_kind == "rsa_no_ranks":
        return 
    if repr_kind == "cka":
        return linear_cka(x, y).item()
    if repr_kind == "rv2":
        return rv2(x, y).item()


def representation_similarity_dist(x, y, repr_kind="rsa", diff_wrt_y=False, nn_sorter_ckp_path='./sodeep/weights/best_model_gruc.pth.tar', lambda_val=0.5):
    repr_kind_options = ("rsa", "rsa_nn", "rsa_sigmoid", "rsa_no_ranks", "cka", "rv2")
    assert repr_kind in repr_kind_options, f"Invalid kind. Options: {repr_kind_options}"

    if not diff_wrt_y:
        y = y.detach()

    # minimizing mse loss is equivalent to maximizing correlation
    if repr_kind == "rsa":
        return torch.nn.functional.mse_loss(TrueRanker.apply(pearsonr_batch(x)[None, :], lambda_val),
                            rank_normalised(pearsonr_batch(y)[None, :]) if not diff_wrt_y else TrueRanker.apply(pearsonr_batch(y)[None, :], lambda_val)
                            )
        # return 1. - pearsonr_vec(TrueRanker.apply(pearsonr_batch(x), lambda_val),
        #                     rank_normalised(pearsonr_batch(y)))
    # if repr_kind == "rsa_nn":
    #     if not hasattr(representation_similarity_dist, 'spearman_loss_fn'):
    #         representation_similarity_dist.spearman_loss_fn = losses.SpearmanLoss(*losses.load_sorter(nn_sorter_ckp_path))
    #         representation_similarity_dist.spearman_loss_fn.sorter.to(x.device)
    #     return representation_similarity_dist.spearman_loss_fn(pearsonr_batch(x), pearsonr_batch(y))
    if repr_kind == "rsa_sigmoid":
        return torch.nn.functional.mse_loss(sigmoid_ranks(pearsonr_batch(x)),
                            ranks(pearsonr_batch(y)) if not diff_wrt_y else sigmoid_ranks(pearsonr_batch(y))
                            )
        # return 1. - pearsonr_vec(sigmoid_ranks(pearsonr_batch(x)),
        #                     ranks(pearsonr_batch(y)))
    if repr_kind == "rsa_no_ranks":
        return torch.nn.functional.mse_loss(pearsonr_batch(x), pearsonr_batch(y))
    if repr_kind == "cka":
        return 1. - linear_cka(x, y)
    if repr_kind == "rv2":
        return 1. - rv2(x, y)
    
    return None


@torch.jit.script
def linear_cka(x, y):
    EPS = 1e-7
    xf = torch.flatten(x, start_dim=1)
    xf_c = xf - torch.mean(xf, dim=0, keepdim=True)
    yf = torch.flatten(y, start_dim=1)
    yf_c = yf - torch.mean(yf, dim=0, keepdim=True)

    return torch.norm(torch.transpose(yf_c, 0, 1) @ xf_c)**2 / (
        torch.norm(torch.transpose(xf_c, 0, 1) @ xf_c + EPS) *
        torch.norm(torch.transpose(yf_c, 0, 1) @ yf_c + EPS))


@torch.jit.script
def rv2(x, y):
    '''
    original paper: Matrix correlations for high-dimensional data: the modified RV-coefficient (Smilde, Kiers, Bijlsma, Rubingh, & Van Erk, 2009)
    neural network similarity paper: The effect of task and training on intermediate representations in convolutional neural networks revealed with modified RV similarity analysis (Thompson, Bengio, Schönwiesner, 2019)
    
    is between -1 and 1,
    cite from original paper:
    The interpretation of RV2 =−1 is that the association between the rows of X is proportional to the association between the rows of Y but with a negative sign (equivalent to a negative Pearson correlation)
    '''
    EPS = 1e-7
    xf = torch.flatten(x, start_dim=1)
    xf_c = xf - torch.mean(xf, dim=0, keepdim=True)
    yf = torch.flatten(y, start_dim=1)
    yf_c = yf - torch.mean(yf, dim=0, keepdim=True)
    xxt = xf_c @ torch.transpose(xf_c, 1, 0)
    xxttilde = xxt - torch.diag(xxt)
    xxttilde_f = torch.flatten(xxttilde)
    
    yyt = yf_c @ torch.transpose(yf_c, 1, 0)
    yyttilde = yyt - torch.diag(yyt)
    yyttilde_f = torch.flatten(yyttilde)

    return torch.dot(xxttilde_f, yyttilde_f) * torch.rsqrt(torch.dot(xxttilde_f, xxttilde_f) + EPS) * torch.rsqrt(torch.dot(yyttilde_f, yyttilde_f) + EPS)



if __name__ == "__main__":
    import numpy as np
    from scipy.stats import spearmanr

    m1 = torch.randn(10, 100) * 10.
    m2 = torch.randn(10, 150) * 10.

    v1 = representation_similarity_dist(m1, m1, repr_kind='rsa').item()
    assert np.isclose(v1, 0., rtol=1e-01, atol=1e-2), f"Should be one as input is equivalent. Is: {v1}"

    v2 = representation_similarity_dist(m1, m2, repr_kind='rsa').item()
    v2_ = representation_similarity_dist(m2, m1, repr_kind='rsa').item()

    corr_m1, corr_m2 = np.corrcoef(m1), np.corrcoef(m2)
    v2_np = 1. - spearmanr(corr_m1[np.triu_indices_from(corr_m1, k=1)].reshape(-1), corr_m2[np.triu_indices_from(corr_m2, k=1)].reshape(-1))[0]

    assert np.isclose(v2, v2_, rtol=1e-01), f"RSA should be symmetric. Is: ({v2}, {v2_})"
    # assert np.isclose(v2, v2_np, rtol=2e-01), f"Pytorch != Scipy+Numpy. Is: ({v2}, {v2_np})"
