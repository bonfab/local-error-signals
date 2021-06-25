import numpy as np
import scipy.stats
import utils

# check if ipywidgets is installed before importing tqdm.auto
# to ensure it won't fail and a progress bar is displayed
import importlib
if importlib.util.find_spec('ipywidgets') is not None:
    from tqdm.auto import tqdm
else:
    from tqdm import tqdm


NP_EPS = np.finfo(np.float32).eps


def corr_dist_of_input_rdms(input_rdms):
    def corr_dist_of_input_rdms_(input_rdms):
        from_dict = False
        act_kinds = None
        if isinstance(input_rdms, dict):
            from_dict = True
            act_kinds = list(input_rdms.keys())
            input_rdms = np.stack(list(input_rdms.values()))

        input_rdms = np.array(input_rdms)
        input_rdms = input_rdms.reshape(input_rdms.shape[0], -1)
        corr_dist_of_input_rdms, _ = scipy.stats.spearmanr(input_rdms.T)

        if isinstance(corr_dist_of_input_rdms, float):
            return corr_dist_of_input_rdms

        # return 1. - corr_dist_of_input_rdms

        # from http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/tutorials/mvahtmlnode99.html
        # http://www.minerazzi.com/tutorials/distance-similarity-tutorial.pdf
        def is_pos_def(A):
            if np.allclose(A, A.T):
                try:
                    np.linalg.cholesky(A)
                    return True
                except np.linalg.LinAlgError:
                    return False
            else:
                return False

        assert is_pos_def(
            corr_dist_of_input_rdms
        ), "Similarity matrix not positive semidefinite. Transformation questionable."
        corr_dist = np.sqrt(2. - 2 * corr_dist_of_input_rdms)
        if from_dict:
            return (act_kinds, corr_dist)

        return corr_dist

    return utils.matrix_of_dist(corr_dist_of_input_rdms_, input_rdms)


# pearson correlation is invariant to scale and location, a + bX (what is described as undesirable in the Hinton paper)
def correlation_matrix(activations, SPLITROWS=100):
    # number of rows in one chunk

    activations = np.reshape(activations, (activations.shape[0], -1))
    numrows = activations.shape[0]

    # subtract means form the input data
    activations -= np.mean(activations, axis=1)[:, None]

    # normalize the data
    activations /= (np.sqrt(np.sum(activations * activations, axis=1))[:, None] + NP_EPS)

    # reserve the resulting table onto HDD
    # res = np.memmap("/tmp/mydata.dat", 'float64', mode='w+', shape=(numrows, numrows))
    res = np.zeros((numrows, numrows))

    if SPLITROWS:
        assert numrows % SPLITROWS == 0, f"SPLITROWS {SPLITROWS} has to be equal divider of numrows: {numrows}"
    else:
        SPLITROWS = numrows

    for r in tqdm(range(0, numrows, SPLITROWS), desc='outer'):
        for c in tqdm(range(r, numrows, SPLITROWS), desc='inner'):
            r1 = r + SPLITROWS
            c1 = c + SPLITROWS
            chunk1 = activations[r:r1]
            chunk2 = activations[c:c1]
            res[r:r1, c:c1] = np.dot(chunk1, chunk2.T)

    del activations

    # triu without diagonal is okay because input rdm is then used for spearman correlation, 
    # which would assign the diagonal elements
    # the same fractional rank
    return res[np.triu_indices_from(res, 1)]


def input_rdms_from_activations(activations, input_rdms=None, SPLITROWS=None):
    return utils.compute_from_activations(
        activations,
        lambda acts: 1. - correlation_matrix(acts, SPLITROWS=SPLITROWS),
        already_computed_layers=input_rdms)