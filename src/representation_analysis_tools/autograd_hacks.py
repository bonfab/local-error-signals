"""
Library for extracting interesting quantites from autograd, see README.md

Not thread-safe because of module-level variables

Notation:
o: number of output classes (exact Hessian), number of Hessian samples (sampled Hessian)
n: batch-size
do: output dimension (output channels for convolution)
di: input dimension (input channels for convolution)
Hi: per-example Hessian of matmul, shaped as matrix of [dim, dim], indices have been row-vectorized
Hi_bias: per-example Hessian of bias
Oh, Ow: output height, output width (convolution)
Kh, Kw: kernel height, kernel width (convolution)

Jb: batch output Jacobian of matmul, output sensitivity for example,class pair, [o, n, ....]
Jb_bias: as above, but for bias

A, activations: inputs into current layer
B, backprops: backprop values (aka Lop aka Jacobian-vector product) observed at current layer


@original source: https://github.com/cybertronai/autograd-hacks/blob/master/autograd_hacks.py
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import collections

import numpy as np
import scipy.linalg

from sklearn.metrics.pairwise import cosine_similarity

from representation_analysis_tools import utils
from representation_analysis_tools.utils import ActivationKind
from copy import deepcopy

from numba import jit

_supported_layers = ['Linear', 'Conv2d']  # Supported layer class types
_hooks_disabled: bool = False           # work-around for https://github.com/pytorch/pytorch/issues/25723
_enforce_fresh_backprop: bool = False   # global switch to catch double backprop errors on Hessian computation


def add_hooks_from_list(list_of_modules) -> None:
    """
    Adds hooks to model to save activations and backprop values.

    The hooks will
    1. save activations into param.activations during forward pass
    2. append backprops to params.backprops_list during backward pass.

    Call "remove_hooks(model)" to disable this.

    Args:
        model:
    """

    global _hooks_disabled
    _hooks_disabled = False

    handles = []
    # TODO these here are usually tuples and sequentials -> find a way to get just the linear layer
    for name, layer in list_of_modules:
        if "task" in name:
            # these don't make sense to look at, as they are the start of the backprop, all receiving the same value bc. it is the gradient w.r.t. to the mean of the output which is 1/len for all values
            continue

        if _layer_type(layer) in _supported_layers:
            # handles.append(layer.register_forward_hook(_capture_activations))
            handles.append(layer.register_backward_hook(_capture_backprops))
        elif _layer_type(layer) == "Sequential":
            handles.append(layer.register_backward_hook(_capture_backprops))
        else:
            print(f"Unsupported layer: {_layer_type(layer)}")

    return handles


def add_hooks(model: nn.Module) -> None:
    """
    Adds hooks to model to save activations and backprop values.

    The hooks will
    1. save activations into param.activations during forward pass
    2. append backprops to params.backprops_list during backward pass.

    Call "remove_hooks(model)" to disable this.

    Args:
        model:
    """

    global _hooks_disabled
    _hooks_disabled = False

    handles = []
    for layer in model.modules():
        if _layer_type(layer) in _supported_layers:
            handles.append(layer.register_forward_hook(_capture_activations))
            handles.append(layer.register_backward_hook(_capture_backprops))

    model.__dict__.setdefault('autograd_hacks_hooks', []).extend(handles)


def remove_hooks(model: nn.Module) -> None:
    """
    Remove hooks added by add_hooks(model)
    """

    assert model == 0, "not working, remove this after fix to https://github.com/pytorch/pytorch/issues/25723"

    if not hasattr(model, 'autograd_hacks_hooks'):
        print("Warning, asked to remove hooks, but no hooks found")
    else:
        for handle in model.autograd_hacks_hooks:
            handle.remove()
        del model.autograd_hacks_hooks

def remove_hooks_from_list(hook_list) -> None:
    """
    Remove hooks added by add_hooks_from_list
    """
    for handle in hook_list:
        handle.remove()


def disable_hooks() -> None:
    """
    Globally disable all hooks installed by this library.
    """

    global _hooks_disabled
    _hooks_disabled = True


def enable_hooks() -> None:
    """the opposite of disable_hooks()"""

    global _hooks_disabled
    _hooks_disabled = False


def is_supported(layer: nn.Module) -> bool:
    """Check if this layer is supported"""

    return _layer_type(layer) in _supported_layers


def _layer_type(layer: nn.Module) -> str:
    return layer.__class__.__name__


def _capture_activations(layer: nn.Module, input: List[torch.Tensor], output: torch.Tensor):
    """Save activations into layer.activations in forward pass"""

    if _hooks_disabled:
        return
    assert _layer_type(layer) in _supported_layers, "Hook installed on unsupported layer, this shouldn't happen"
    setattr(layer, "activations", input[0].detach())


def _capture_backprops(layer: nn.Module, _input, output):
    """Append backprop to layer.backprops_list in backward pass."""
    global _enforce_fresh_backprop

    if _hooks_disabled:
        return

    if _enforce_fresh_backprop:
        assert not hasattr(layer, 'backprops_list'), "Seeing result of previous backprop, use clear_backprops(model) to clear"
        _enforce_fresh_backprop = False

    if not hasattr(layer, 'backprops_list'):
        setattr(layer, 'backprops_list', [])
    layer.backprops_list.append(output[0].detach())


def add_dummy_backprops_from_list(list_of_modules):
    for name, layer in list_of_modules:
        if "task" in name:
            # these don't make sense to look at, as they are the start of the backprop, all receiving the same value bc. it is the gradient w.r.t. to the mean of the output which is 1/len for all values
            continue
        if _layer_type(layer) in _supported_layers:
            if not hasattr(layer, 'backprops_list'):
                setattr(layer, 'backprops_list', [])
            layer.backprops_list.append(None)
        elif _layer_type(layer) == "Sequential":
            if not hasattr(layer, 'backprops_list'):
                setattr(layer, 'backprops_list', [])
            layer.backprops_list.append(None)

def join_backprops_over_classes_from_list(list_of_modules, number_of_classes):
    for name, layer in list_of_modules:
        if "task" in name:
            # these don't make sense to look at, as they are the start of the backprop, all receiving the same value bc. it is the gradient w.r.t. to the mean of the output which is 1/len for all values
            continue

        assert hasattr(layer, 'backprops_list'), f"Layer: {layer} should have a backprops_list."
        assert len(layer.backprops_list) >= number_of_classes, f"Length of backprops_list: {len(layer.backprops_list)} smaller than number_of_classes: {number_of_classes}"

        # what can happen
        # when its the zeroth task: only 0-number_of_classes-1 elements, 
        # for each next task: shift by one 
        
        # take the last number_of_classes elements and concat them
        new_backprops_list_item = torch.stack(layer.backprops_list[-number_of_classes:], dim=1)
        del layer.backprops_list[-number_of_classes:]
        layer.backprops_list.append(new_backprops_list_item)

def dummy_backprops_for_task_i_from_list(list_of_modules, task_i):
    for name, layer in list_of_modules:
        if "task" in name:
            # these don't make sense to look at, as they are the start of the backprop, all receiving the same value bc. it is the gradient w.r.t. to the mean of the output which is 1/len for all values
            continue
        if _layer_type(layer) in _supported_layers:
            if not hasattr(layer, 'backprops_list'):
                setattr(layer, 'backprops_list', [])
            if len(layer.backprops_list) < task_i+1:
                layer.backprops_list.append(None)
        elif _layer_type(layer) == "Sequential":
            if not hasattr(layer, 'backprops_list'):
                setattr(layer, 'backprops_list', [])
            if len(layer.backprops_list) < task_i+1:
                layer.backprops_list.append(None)


def clear_backprops(model: nn.Module) -> None:
    """Delete layer.backprops_list in every layer."""
    for layer in model.modules():
        if hasattr(layer, 'backprops_list'):
            del layer.backprops_list


def clear_backprops_from_list(list_of_modules) -> None:
    """Delete layer.backprops_list in every layer."""
    for name, layer in list_of_modules:
        if "task" in name:
            # these don't make sense to look at, as they are the start of the backprop, all receiving the same value bc. it is the gradient w.r.t. to the mean of the output which is 1/len for all values
            continue
        if _layer_type(layer) in _supported_layers:
            if hasattr(layer, 'backprops_list'):
                del layer.backprops_list
        elif _layer_type(layer) == "Sequential":
            if hasattr(layer, 'backprops_list'):
                del layer.backprops_list
        # TODO maybe also delete the grad attribute


def compute_grad1(model: nn.Module, loss_type: str = 'mean') -> None:
    """
    Compute per-example gradients and save them under 'param.grad1'. Must be called after loss.backprop()

    Args:
        model:
        loss_type: either "mean" or "sum" depending whether backpropped loss was averaged or summed over batch
    """

    assert loss_type in ('sum', 'mean')
    for layer in model.modules():
        layer_type = _layer_type(layer)
        if layer_type not in _supported_layers:
            continue
        assert hasattr(layer, 'activations'), "No activations detected, run forward after add_hooks(model)"
        assert hasattr(layer, 'backprops_list'), "No backprops detected, run backward after add_hooks(model)"
        assert len(layer.backprops_list) == 1, "Multiple backprops detected, make sure to call clear_backprops(model)"

        A = layer.activations
        n = A.shape[0]
        if loss_type == 'mean':
            B = layer.backprops_list[0] * n
        else:  # loss_type == 'sum':
            B = layer.backprops_list[0]

        if layer_type == 'Linear':
            setattr(layer.weight, 'grad1', torch.einsum('ni,nj->nij', B, A))
            if layer.bias is not None:
                setattr(layer.bias, 'grad1', B)

        elif layer_type == 'Conv2d':
            A = torch.nn.functional.unfold(A, layer.kernel_size)
            B = B.reshape(n, -1, A.shape[-1])
            grad1 = torch.einsum('ijk,ilk->ijl', B, A)
            shape = [n] + list(layer.weight.shape)
            setattr(layer.weight, 'grad1', grad1.reshape(shape))
            if layer.bias is not None:
                setattr(layer.bias, 'grad1', torch.sum(B, dim=2))


# TODO probably don't need that anymore -> I just care about the grads w.r.t. activations
def compute_grad1_from_list(list_of_modules, loss_type: str = 'mean', differentiable=False, ignore_bias=True) -> None:
    """
    Compute per-example gradients and save them under 'param.grad1'. Must be called after loss.backprop()

    Args:
        model:
        loss_type: either "mean" or "sum" depending whether backpropped loss was averaged or summed over batch
    """

    assert loss_type in ('sum', 'mean')    
    for name, layer in list_of_modules:
        if "task" in name:
            # these don't make sense to look at, as they are the start of the backprop, all receiving the same value bc. it is the gradient w.r.t. to the mean of the output which is 1/len for all values
            continue
        layer_type = _layer_type(layer)
        if layer_type is "Sequential":
            layer = layer[0]
            layer_type = _layer_type(layer)
        elif layer_type not in _supported_layers:
            continue
        assert hasattr(layer, 'activations'), "No activations detected, run forward after add_hooks(model)"
        assert hasattr(layer, 'backprops_list'), "No backprops detected, run backward after add_hooks(model)"
        
        for i in range(len(layer.backprops_list)):
        # assert len(layer.backprops_list) == 1, "Multiple backprops detected, make sure to call clear_backprops(model)"

            i_ = '_' + str(i)

            if layer.backprops_list[i] is None:
                # add dummy grad1
                setattr(layer.weight, f"grad1{i_}", None)
                continue

            A = layer.activations
            n = A.shape[0]
            if loss_type == 'mean':
                B = layer.backprops_list[i] * n
            else:  # loss_type == 'sum':
                B = layer.backprops_list[i]

            if layer_type == 'Linear':
                setattr(layer.weight, f'grad1{i_}', 
                    torch.einsum('ni,nj->nij', B, A).detach().cpu() if not differentiable else torch.einsum('ni,nj->nij', B, A))
                if not ignore_bias and layer.bias is not None:
                    setattr(layer.bias, f'grad1{i_}', B.detach().cpu() if not differentiable else B)

            elif layer_type == 'Conv2d':
                A = torch.nn.functional.unfold(A, layer.kernel_size)
                B = B.reshape(n, -1, A.shape[-1])
                grad1 = torch.einsum('ijk,ilk->ijl', B, A)
                shape = [n] + list(layer.weight.shape)
                setattr(layer.weight, f'grad1{i_}', grad1.reshape(shape).detach().cpu() if not differentiable else grad1.reshape(shape))
                if not ignore_bias and layer.bias is not None:
                    setattr(layer.bias, f'grad1{i_}', torch.sum(B, dim=2).detach().cpu() if not differentiable else torch.sum(B, dim=2))

@jit(nopython=True)
def compute_scalar_products_from_grads(grads_x, grads_y):
    c_x = grads_x @ grads_x.T
    c_y = grads_y @ grads_y.T
    k = grads_x @ grads_y.T

    # make them symmetric
    c_x = np.maximum(c_x, c_x.transpose())
    c_y = np.maximum(c_y, c_y.transpose())
    return c_x, c_y, k

@jit
def compute_sqrtms(c_x, c_y, eps=3.* np.finfo(np.float32).eps):
    sqrt_c_x = scipy.linalg.sqrtm(c_x + eps * np.eye(c_x.shape[0], dtype=np.float32)).real.astype(np.float32)
    sqrt_c_y = scipy.linalg.sqrtm(c_y + eps * np.eye(c_x.shape[1], dtype=np.float32)).real.astype(np.float32)
    return sqrt_c_x, sqrt_c_y

@jit(nopython=True)
def compute_normalized_kernel_matrix_(sqrt_c_x, sqrt_c_y, k):
    inv_sqrt_c_x = np.linalg.inv(sqrt_c_x)
    inv_sqrt_c_y = np.linalg.inv(sqrt_c_y)
    normed_k = inv_sqrt_c_x @ k @ inv_sqrt_c_y
    return normed_k

@jit
def gradient_similarity_kernel(grads):
    m, n = grads.shape[0], grads.shape[0]
    dm = np.empty((m * (m - 1)) // 2, dtype=np.float32)
    k_ = 0
    for i in range(0, m - 1):
        for j in range(i + 1, m):
            c_x, c_y, k = compute_scalar_products_from_grads(grads[i, :, :], grads[j, :, :])
            sqrt_c_x, sqrt_c_y = compute_sqrtms(c_x, c_y)
            normed_k = compute_normalized_kernel_matrix_(sqrt_c_x, sqrt_c_y, k)
            dm[k_] = normed_k.trace() / normed_k.shape[0]
            k_ = k_ + 1

    return dm

def grad_dms_from_grads(grads, grad_rdms=None):
    return utils.compute_from_activations(
        grads,
        lambda grads_: 1. - gradient_similarity_kernel(grads_),
        already_computed_layers=grad_rdms)

#TODO accumulate from tracked_modules into a dict
# name is given, but remove _act
# ignore_bias = True
# have multiple dicts: one for output 0, one for output 1, one for output 0+1, one for output [0 concat 1]

#TODO handle skip_tasks by adding the correct grad_x but with the value None

def create_grads_dict():
    return collections.defaultdict(list)

def grads_to_dicts_from_list(grads_dict, list_of_modules, tracking_flag, ignore_bias=True):
    assert ignore_bias, "ignore_bias=False is not supported."

    def act_kind_from_(layer_name, grads_kind_name):
        # layer_name_ = layer_name.strip("_act")
        layer_name_ = layer_name
        return ActivationKind(tracking_flag.model_name, tracking_flag.data_name, layer_name_+'_'+grads_kind_name, tracking_flag.epoch)

    for name, layer in list_of_modules:
        if "task" in name:
            # these don't make sense to look at, as they are the start of the backprop, all receiving the same value bc. it is the gradient w.r.t. to the mean of the output which is 1/len for all values
            continue

        layer_ = None
        if _layer_type(layer) in _supported_layers:
            layer_ = layer
        elif _layer_type(layer) == "Sequential":
            layer_ = layer

        assert hasattr(layer_, "backprops_list"), "Tracked layers should have backprops_list attribute!"

        if layer_.backprops_list[0] is not None:
            grads_dict[act_kind_from_(name, "grads_wrt_output_0")].append(deepcopy(layer_.backprops_list[0]).cpu())
        if layer_.backprops_list[1] is not None:
            grads_dict[act_kind_from_(name, "grads_wrt_output_1")].append(deepcopy(layer_.backprops_list[1]).cpu())

        if layer_.backprops_list[0] is not None and layer_.backprops_list[1] is not None:
            grads_dict[act_kind_from_(name, "grads_wrt_output_0_1_sum")].append(deepcopy(layer_.backprops_list[0] + layer_.backprops_list[1]).cpu())
            grads_dict[act_kind_from_(name, "grads_wrt_output_0_1_concat")].append(deepcopy(torch.cat((layer_.backprops_list[0], layer_.backprops_list[1]), dim=1)).cpu())

    # delete all grad1 attributes
    # TODO check if this really works...
    for name, layer in list_of_modules:
        if "task" in name:
            # these don't make sense to look at, as they are the start of the backprop, all receiving the same value bc. it is the gradient w.r.t. to the mean of the output which is 1/len for all values
            continue

        layer_ = None
        if _layer_type(layer) in _supported_layers:
            layer_ = layer
        elif _layer_type(layer) == "Sequential":
            layer_ = layer

        if hasattr(layer_, "backprops_list"):
            del layer_.backprops_list

def compute_hess(model: nn.Module,) -> None:
    """Save Hessian under param.hess for each param in the model"""

    for layer in model.modules():
        layer_type = _layer_type(layer)
        if layer_type not in _supported_layers:
            continue
        assert hasattr(layer, 'activations'), "No activations detected, run forward after add_hooks(model)"
        assert hasattr(layer, 'backprops_list'), "No backprops detected, run backward after add_hooks(model)"

        if layer_type == 'Linear':
            A = layer.activations
            B = torch.stack(layer.backprops_list)

            n = A.shape[0]
            o = B.shape[0]

            A = torch.stack([A] * o)
            Jb = torch.einsum("oni,onj->onij", B, A).reshape(n*o,  -1)
            H = torch.einsum('ni,nj->ij', Jb, Jb) / n

            setattr(layer.weight, 'hess', H)

            if layer.bias is not None:
                setattr(layer.bias, 'hess', torch.einsum('oni,onj->ij', B, B)/n)

        elif layer_type == 'Conv2d':
            Kh, Kw = layer.kernel_size
            di, do = layer.in_channels, layer.out_channels

            A = layer.activations.detach()
            A = torch.nn.functional.unfold(A, (Kh, Kw))       # n, di * Kh * Kw, Oh * Ow
            n = A.shape[0]
            B = torch.stack([Bt.reshape(n, do, -1) for Bt in layer.backprops_list])  # o, n, do, Oh*Ow
            o = B.shape[0]

            A = torch.stack([A] * o)                          # o, n, di * Kh * Kw, Oh*Ow
            Jb = torch.einsum('onij,onkj->onik', B, A)        # o, n, do, di * Kh * Kw

            Hi = torch.einsum('onij,onkl->nijkl', Jb, Jb)     # n, do, di*Kh*Kw, do, di*Kh*Kw
            Jb_bias = torch.einsum('onij->oni', B)
            Hi_bias = torch.einsum('oni,onj->nij', Jb_bias, Jb_bias)

            setattr(layer.weight, 'hess', Hi.mean(dim=0))
            if layer.bias is not None:
                setattr(layer.bias, 'hess', Hi_bias.mean(dim=0))


def backprop_hess(output: torch.Tensor, hess_type: str) -> None:
    """
    Call backprop 1 or more times to get values needed for Hessian computation.

    Args:
        output: prediction of neural network (ie, input of nn.CrossEntropyLoss())
        hess_type: type of Hessian propagation, "CrossEntropy" results in exact Hessian for CrossEntropy

    Returns:

    """

    assert hess_type in ('LeastSquares', 'CrossEntropy')
    global _enforce_fresh_backprop
    n, o = output.shape

    _enforce_fresh_backprop = True

    if hess_type == 'CrossEntropy':
        batch = F.softmax(output, dim=1)

        mask = torch.eye(o).expand(n, o, o)
        diag_part = batch.unsqueeze(2).expand(n, o, o) * mask
        outer_prod_part = torch.einsum('ij,ik->ijk', batch, batch)
        hess = diag_part - outer_prod_part
        assert hess.shape == (n, o, o)

        for i in range(n):
            hess[i, :, :] = symsqrt(hess[i, :, :])
        hess = hess.transpose(0, 1)

    elif hess_type == 'LeastSquares':
        hess = []
        assert len(output.shape) == 2
        batch_size, output_size = output.shape

        id_mat = torch.eye(output_size)
        for out_idx in range(output_size):
            hess.append(torch.stack([id_mat[out_idx]] * batch_size))

    for o in range(o):
        output.backward(hess[o], retain_graph=True)


def symsqrt(a, cond=None, return_rank=False, dtype=torch.float32):
    """Symmetric square root of a positive semi-definite matrix.
    See https://github.com/pytorch/pytorch/issues/25481"""

    s, u = torch.symeig(a, eigenvectors=True)
    cond_dict = {torch.float32: 1e3 * 1.1920929e-07, torch.float64: 1E6 * 2.220446049250313e-16}

    if cond in [None, -1]:
        cond = cond_dict[dtype]

    above_cutoff = (abs(s) > cond * torch.max(abs(s)))

    psigma_diag = torch.sqrt(s[above_cutoff])
    u = u[:, above_cutoff]

    B = u @ torch.diag(psigma_diag) @ u.t()
    if return_rank:
        return B, len(psigma_diag)
    else:
        return B
