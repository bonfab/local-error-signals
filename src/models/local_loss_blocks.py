import abc
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from optimizers.sam import SAM
import utils.models as utils
#from utils.models import similarity_matrix, LinearFAFunction, LinearFA, find_zero_grads


class LocalLossBlock(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.local_eval = False

    def print_grads(self):
        for name, module in self.named_modules():
            module.register_backward_hook(utils.find_zero_grads(name))

    def select_optimizer(self):
        if self.args.backprop:
            return

        if self.args.sam:
            if self.args.optim == 'sgd':
                self.optimizer = SAM(self.parameters(), optim.SGD, lr=0, weight_decay=self.args.weight_decay,
                                     momentum=self.args.momentum)
            elif self.args.optim == 'adam' or self.args.optim == 'amsgrad':
                self.optimizer = SAM(self.parameters(), optim.Adam, lr=0, weight_decay=self.args.weight_decay,
                                     amsgrad=self.args.optim == 'amsgrad')
            if self.args.exponential_lr_scheduler:
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer.base_optimizer,
                                                                        self.args.exponential_lr_gamma)

        elif self.args.optim == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=0, weight_decay=self.args.weight_decay,
                                       momentum=self.args.momentum)
        elif self.args.optim == 'adam' or self.args.optim == 'amsgrad':
            self.optimizer = optim.Adam(self.parameters(), lr=0, weight_decay=self.args.weight_decay,
                                        amsgrad=self.args.optim == 'amsgrad')
            if self.args.exponential_lr_scheduler:
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.args.exponential_lr_gamma)

    def lr_scheduler_step(self):
        if not self.args.backprop and self.args.exponential_lr_scheduler:
            self.scheduler.step()

    def local_loss_eval(self):
        self.local_eval = True

    def local_loss_train(self):
        self.local_eval = False

    def clear_stats(self):
        if not self.no_print_stats:
            self.loss_sim = 0.0
            self.loss_pred = 0.0
            self.correct = 0
            self.examples = 0

    def print_stats(self):
        if not self.args.backprop:
            stats = '{}, loss_sim={:.4f}, loss_pred={:.4f}, error={:.3f}%, num_examples={}\n'.format(
                self.encoder,
                self.loss_sim / self.examples,
                self.loss_pred / self.examples,
                100.0 * float(self.examples - self.correct) / self.examples,
                self.examples)
            return stats
        else:
            return ''

    def set_learning_rate(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def optim_zero_grad(self):
        self.optimizer.zero_grad()

    def optim_step(self):
        self.optimizer.step()

    def calc_loss_unsup(self, x, Rh, h):

        if self.args.loss_unsup == 'sim':
            Rx = utils.similarity_matrix(x, self.args.no_similarity_std).detach()
            loss_unsup = F.mse_loss(Rh, Rx)
        elif self.args.loss_unsup == 'recon' and not self.first_layer:
            x_hat = self.nonlin(self.decoder_x(h))
            loss_unsup = F.mse_loss(x_hat, x.detach())
        else:
            if self.args.gpus:
                loss_unsup = torch.cuda.FloatTensor([0])
            else:
                loss_unsup = torch.FloatTensor([0])
        return loss_unsup

    @abc.abstractmethod
    def calc_loss_sup(self, Rh, h, y, y_onehot):
        return

    @abc.abstractmethod
    def forward_pass(self, x):
        return

    @abc.abstractmethod
    def calculate_similarity_matrix(self, h):
        return

    def calc_combined_loss(self, x, Rh, h, y, y_onehot):
        loss_unsup = self.calc_loss_unsup(x, Rh, h)
        loss_sup = self.calc_loss_sup(Rh, h, y, y_onehot)
        return self.args.alpha * loss_unsup + (1 - self.args.alpha) * loss_sup

    def step_sam(self, x, Rh, h, y, y_onehot):

        # Combine unsupervised and supervised loss
        loss = self.calc_combined_loss(x, Rh, h, y, y_onehot)
        # print(f'loss 1 {loss.item()}')

        # Single-step back-propagation
        if self.training:
            loss.backward(retain_graph=False)

        # First optimizer step in this layer and detach computational graph
        if self.training and not self.args.no_detach:
            self.optimizer.first_step(zero_grad=True)

        _, h = self.forward_pass(x)
        Rh = self.calculate_similarity_matrix(h)

        # Combine unsupervised and supervised loss
        loss = self.calc_combined_loss(x, Rh, h, y, y_onehot)
        # print(f'loss 2 {loss.item()}')

        # Single-step back-propagation
        if self.training:
            loss.backward(retain_graph=self.args.no_detach)

        # Second optimizer step in this layer and detach computational graph
        if self.training and not self.args.no_detach:
            self.optimizer.second_step(zero_grad=True)

        return loss


class LocalLossBlockLinear(LocalLossBlock):
    '''A module containing nn.Linear -> nn.BatchNorm1d -> nn.ReLU -> nn.Dropout
       The block can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.

    Args:
        num_in (int): Number of input features to linear layer.
        num_out (int): Number of output features from linear layer.
        num_classes (int): Number of classes (used in local prediction loss).
        first_layer (bool): True if this is the first layer in the network (used in local reconstruction loss).
        dropout (float): Dropout rate, if None, read from args.dropout.
        batchnorm (bool): True if to use batchnorm, if None, read from args.no_batch_norm.
    '''

    def __init__(self, args, num_in, num_out, num_classes, nonlin="relu", first_layer=False, dropout=0, batchnorm=None,
                 print_stats=True):
        super().__init__(args)

        self.no_print_stats = not print_stats
        self.num_classes = num_classes
        self.first_layer = first_layer
        self.dropout_p = dropout
        self.batchnorm = not args.no_batch_norm if batchnorm is None else batchnorm
        self.encoder = nn.Linear(num_in, num_out, bias=True)

        if not args.backprop and args.loss_unsup == 'recon':
            self.decoder_x = nn.Linear(num_out, num_in, bias=True)
        if not args.backprop and (args.loss_sup == 'pred' or args.loss_sup == 'predsim'):
            if args.bio:
                self.decoder_y = utils.LinearFA(num_out, args.target_proj_size)
            else:
                self.decoder_y = nn.Linear(num_out, num_classes)
            self.decoder_y.weight.data.zero_()
        if not args.backprop and args.bio:
            self.proj_y = nn.Linear(num_classes, args.target_proj_size, bias=False)
        if not args.backprop and not args.bio and (
                args.loss_unsup == 'sim' or args.loss_sup == 'sim' or args.loss_sup == 'predsim'):
            self.linear_loss = nn.Linear(num_out, num_out, bias=False)
        if self.batchnorm:
            self.bn = torch.nn.BatchNorm1d(num_out)
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)
        if nonlin == 'relu':
            self.nonlin = nn.ReLU(inplace=True)
        elif nonlin == 'leakyrelu':
            self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        if self.dropout_p > 0:
            self.dropout = torch.nn.Dropout(p=self.dropout_p, inplace=False)
        self.select_optimizer()
        self.clear_stats()

    def calc_loss_sup(self, Rh, h, y, y_onehot):
        if self.args.loss_sup == 'sim':
            if self.args.bio:
                Ry = utils.similarity_matrix(self.proj_y(y_onehot), self.args.no_similarity_std).detach()
            else:
                Ry = utils.similarity_matrix(y_onehot, self.args.no_similarity_std).detach()
            loss_sup = F.mse_loss(Rh, Ry)
            if not self.no_print_stats:
                self.loss_sim += loss_sup.item() * h.size(0)
                self.examples += h.size(0)
        elif self.args.loss_sup == 'pred':
            y_hat_local = self.decoder_y(h.view(h.size(0), -1))
            if self.args.bio:
                float_type = torch.cuda.FloatTensor if self.args.gpus else torch.FloatTensor
                y_onehot_pred = self.proj_y(y_onehot).gt(0).type(float_type).detach()
                loss_sup = F.binary_cross_entropy_with_logits(y_hat_local, y_onehot_pred)
            else:
                loss_sup = F.cross_entropy(y_hat_local, y.detach())
            if not self.no_print_stats:
                self.loss_pred += loss_sup.item() * h.size(0)
                self.correct += y_hat_local.max(1)[1].eq(y).cpu().sum()
                self.examples += h.size(0)
        elif self.args.loss_sup == 'predsim':
            y_hat_local = self.decoder_y(h.view(h.size(0), -1))
            if self.args.bio:
                Ry = utils.similarity_matrix(self.proj_y(y_onehot), self.args.no_similarity_std).detach()
                float_type = torch.cuda.FloatTensor if self.args.gpus else torch.FloatTensor
                y_onehot_pred = self.proj_y(y_onehot).gt(0).type(float_type).detach()
                loss_pred = (1 - self.args.beta) * F.binary_cross_entropy_with_logits(y_hat_local, y_onehot_pred)
            else:
                Ry = utils.similarity_matrix(y_onehot, self.args.no_similarity_std).detach()
                loss_pred = (1 - self.args.beta) * F.cross_entropy(y_hat_local, y.detach())
            loss_sim = self.args.beta * F.mse_loss(Rh, Ry)
            loss_sup = loss_pred + loss_sim
            if not self.no_print_stats:
                self.loss_pred += loss_pred.item() * h.size(0)
                self.loss_sim += loss_sim.item() * h.size(0)
                self.correct += y_hat_local.max(1)[1].eq(y).cpu().sum()
                self.examples += h.size(0)

        return loss_sup

    def forward_pass(self, x):

        h = self.encoder(x)

        # Add batchnorm and nonlinearity
        if self.batchnorm:
            h = self.bn(h)
        h = self.nonlin(h)

        # Save return value and add dropout
        h_return = h
        if self.dropout_p > 0:
            h_return = self.dropout(h_return)

        return h_return, h

    def calculate_similarity_matrix(self, h):
        Rh = 0

        # Calculate hidden layer similarity matrix
        if self.args.loss_unsup == 'sim' or self.args.loss_sup == 'sim' or self.args.loss_sup == 'predsim':
            if self.args.bio:
                h_loss = h
            else:
                h_loss = self.linear_loss(h)
            Rh = utils.similarity_matrix(h_loss, self.args.no_similarity_std)

        return Rh

    def forward(self, x, y=None, y_onehot=None):

        assert not (y is None or y_onehot is None) or self.local_eval

        h_return, h = self.forward_pass(x)

        if not self.local_eval:
            # Calculate local loss and update weights
            if (self.training or not self.no_print_stats) and not self.args.backprop:

                Rh = self.calculate_similarity_matrix(h)

                if self.args.sam:
                    loss = self.step_sam(x, Rh, h, y, y_onehot)
                    if self.training and not self.args.no_detach:
                        h_return.detach_()
                else:
                    # Combine unsupervised and supervised loss
                    loss = self.calc_combined_loss(x, Rh, h, y, y_onehot)

                    # Single-step back-propagation
                    if self.training:
                        loss.backward(retain_graph=self.args.no_detach)

                    # Update weights in this layer and detach computational graph
                    if self.training and not self.args.no_detach:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        h_return.detach_()

                loss = loss.item()
            else:
                loss = 0.0

            return h_return, loss

        else:
            return h_return


class LocalLossBlockConv(LocalLossBlock):
    '''
    A block containing nn.Conv2d -> nn.BatchNorm2d -> nn.ReLU -> nn.Dropou2d
    The block can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.

    Args:
        ch_in (int): Number of input features maps.
        ch_out (int): Number of output features maps.
        kernel_size (int): Kernel size in Conv2d.
        stride (int): Stride in Conv2d.
        padding (int): Padding in Conv2d.
        num_classes (int): Number of classes (used in local prediction loss).
        dim_out (int): Feature map height/width for input (and output).
        first_layer (bool): True if this is the first layer in the network (used in local reconstruction loss).
        dropout (float): Dropout rate
        bias (bool): True if to use trainable bias.
        pre_act (bool): True if to apply layer order nn.BatchNorm2d -> nn.ReLU -> nn.Dropou2d -> nn.Conv2d (used for PreActResNet).
        post_act (bool): True if to apply layer order nn.Conv2d -> nn.BatchNorm2d -> nn.ReLU -> nn.Dropou2d.
    '''

    def __init__(self, args, ch_in, ch_out, kernel_size, stride, padding, num_classes, dim_out, nonlin='relu',
                 first_layer=False,
                 dropout=0, bias=None, pre_act=False, post_act=True):
        super().__init__(args)
        self.no_print_stats = not self.args.print_stats
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.num_classes = num_classes
        self.first_layer = first_layer
        self.dropout_p = dropout
        self.bias = True if bias is None else bias
        self.pre_act = pre_act
        self.post_act = post_act
        self.encoder = nn.Conv2d(ch_in, ch_out, kernel_size, stride=stride, padding=padding, bias=self.bias)

        if not self.args.backprop and self.args.loss_unsup == 'recon':
            self.decoder_x = nn.ConvTranspose2d(ch_out, ch_in, kernel_size, stride=stride, padding=padding)
        if self.args.bio or (not self.args.backprop and (args.loss_sup == 'pred' or args.loss_sup == 'predsim')):
            # Resolve average-pooling kernel size in order for flattened dim to match args.dim_in_decoder
            ks_h, ks_w = 1, 1
            dim_out_h, dim_out_w = dim_out, dim_out
            dim_in_decoder = ch_out * dim_out_h * dim_out_w
            while dim_in_decoder > args.dim_in_decoder and ks_h < dim_out:
                ks_h *= 2
                dim_out_h = math.ceil(dim_out / ks_h)
                dim_in_decoder = ch_out * dim_out_h * dim_out_w
                if dim_in_decoder > args.dim_in_decoder:
                    ks_w *= 2
                    dim_out_w = math.ceil(dim_out / ks_w)
                    dim_in_decoder = ch_out * dim_out_h * dim_out_w
            if ks_h > 1 or ks_w > 1:
                pad_h = (ks_h * (dim_out_h - dim_out // ks_h)) // 2
                pad_w = (ks_w * (dim_out_w - dim_out // ks_w)) // 2
                self.avg_pool = nn.AvgPool2d((ks_h, ks_w), padding=(pad_h, pad_w))
            else:
                self.avg_pool = None
        if not args.backprop and (args.loss_sup == 'pred' or args.loss_sup == 'predsim'):
            if args.bio:
                self.decoder_y = utils.LinearFA(dim_in_decoder, args.target_proj_size)
            else:
                self.decoder_y = nn.Linear(dim_in_decoder, num_classes)
            self.decoder_y.weight.data.zero_()
        if not args.backprop and args.bio:
            self.proj_y = nn.Linear(num_classes, args.target_proj_size, bias=False)
        if not args.backprop and (args.loss_unsup == 'sim' or args.loss_sup == 'sim' or args.loss_sup == 'predsim'):
            self.conv_loss = nn.Conv2d(ch_out, ch_out, 3, stride=1, padding=1, bias=False)
        if not args.no_batch_norm:
            if pre_act:
                self.bn_pre = torch.nn.BatchNorm2d(ch_in)
            if not (pre_act and args.backprop):
                self.bn = torch.nn.BatchNorm2d(ch_out)
                nn.init.constant_(self.bn.weight, 1)
                nn.init.constant_(self.bn.bias, 0)
        if nonlin == 'relu':
            self.nonlin = nn.ReLU(inplace=True)
        elif nonlin == 'leakyrelu':
            self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        if self.dropout_p > 0:
            self.dropout = torch.nn.Dropout2d(p=self.dropout_p, inplace=False)
        self.select_optimizer()
        # self.print_grads()
        self.clear_stats()

    def calc_loss_sup(self, Rh, h, y, y_onehot):
        if self.args.loss_sup == 'sim':
            if self.args.bio:
                Ry = utils.similarity_matrix(self.proj_y(y_onehot), self.args.no_similarity_std).detach()
            else:
                Ry = utils.similarity_matrix(y_onehot, self.args.no_similarity_std).detach()
            loss_sup = F.mse_loss(Rh, Ry)
            if not self.no_print_stats:
                self.loss_sim += loss_sup.item() * h.size(0)
                self.examples += h.size(0)
        elif self.args.loss_sup == 'pred':
            if self.avg_pool is not None:
                h = self.avg_pool(h)
            y_hat_local = self.decoder_y(h.view(h.size(0), -1))
            if self.args.bio:
                float_type = torch.cuda.FloatTensor if self.args.gpus else torch.FloatTensor
                y_onehot_pred = self.proj_y(y_onehot).gt(0).type(float_type).detach()
                loss_sup = F.binary_cross_entropy_with_logits(y_hat_local, y_onehot_pred)
            else:
                loss_sup = F.cross_entropy(y_hat_local, y.detach())
            if not self.no_print_stats:
                self.loss_pred += loss_sup.item() * h.size(0)
                self.correct += y_hat_local.max(1)[1].eq(y).cpu().sum()
                self.examples += h.size(0)
        elif self.args.loss_sup == 'predsim':
            if self.avg_pool is not None:
                h = self.avg_pool(h)
            y_hat_local = self.decoder_y(h.view(h.size(0), -1))
            if self.args.bio:
                Ry = utils.similarity_matrix(self.proj_y(y_onehot), self.args.no_similarity_std).detach()
                float_type = torch.cuda.FloatTensor if self.args.gpus else torch.FloatTensor
                y_onehot_pred = self.proj_y(y_onehot).gt(0).type(float_type).detach()
                loss_pred = (1 - self.args.beta) * F.binary_cross_entropy_with_logits(y_hat_local, y_onehot_pred)
            else:
                Ry = utils.similarity_matrix(y_onehot, self.args.no_similarity_std).detach()
                loss_pred = (1 - self.args.beta) * F.cross_entropy(y_hat_local, y.detach())
            loss_sim = self.args.beta * F.mse_loss(Rh, Ry)
            loss_sup = loss_pred + loss_sim
            if not self.no_print_stats:
                self.loss_pred += loss_pred.item() * h.size(0)
                self.loss_sim += loss_sim.item() * h.size(0)
                self.correct += y_hat_local.max(1)[1].eq(y).cpu().sum()
                self.examples += h.size(0)

        return loss_sup

    def forward_pass(self, x, x_shortcut=None):

        # If pre-activation, apply batchnorm->nonlin->dropout
        if self.pre_act:
            if not self.args.no_batch_norm:
                x = self.bn_pre(x)
            x = self.nonlin(x)
            if self.dropout_p > 0:
                x = self.dropout(x)

        # The convolutional transformation
        h = self.encoder(x)

        # If post-activation, apply batchnorm
        if self.post_act and not self.args.no_batch_norm:
            h = self.bn(h)

        # Add shortcut branch (used in residual networks)
        if x_shortcut is not None:
            h = h + x_shortcut

        # If post-activation, add nonlinearity
        if self.post_act:
            h = self.nonlin(h)

        # Save return value and add dropout
        h_return = h

        if self.post_act and self.dropout_p > 0:
            h_return = self.dropout(h_return)

        # Add batchnorm and nonlinearity if not done already
        if not self.post_act and (not self.no_print_stats or self.training) and not self.args.backprop:
            if not self.args.no_batch_norm:
                h = self.bn(h)
            h = self.nonlin(h)

        return h_return, h

    def calculate_similarity_matrix(self, h):

        Rh = 0

        # Calculate hidden feature similarity matrix
        if self.args.loss_unsup == 'sim' or self.args.loss_sup == 'sim' or self.args.loss_sup == 'predsim':
            if self.args.bio:
                h_loss = h
                if self.avg_pool is not None:
                    h_loss = self.avg_pool(h_loss)
            else:
                h_loss = self.conv_loss(h)
            Rh = utils.similarity_matrix(h_loss, self.args.no_similarity_std)

        return Rh

    def forward(self, x, y=None, y_onehot=None, x_shortcut=None):

        assert not (y is None or y_onehot is None) or self.local_eval

        h_return, h = self.forward_pass(x, x_shortcut)

        if not self.local_eval:
            # Calculate local loss and update weights
            if (not self.no_print_stats or self.training) and not self.args.backprop:

                Rh = self.calculate_similarity_matrix(h)

                if self.args.sam:
                    loss = self.step_sam(x, Rh, h, y, y_onehot)
                    if self.training and not self.args.no_detach:
                        h_return.detach_()
                else:
                    # Combine unsupervised and supervised loss
                    loss = self.calc_combined_loss(x, Rh, h, y, y_onehot)
                    # Single-step back-propagation
                    if self.training:
                        if math.isnan(loss):
                            print(loss)
                        loss.backward(retain_graph=self.args.no_detach)

                    # Update weights in this layer and detach computational graph
                    if self.training and not self.args.no_detach:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        h_return.detach_()

                loss = loss.item()
            else:
                loss = 0.0
            return h_return, loss

        else:
            return h_return
