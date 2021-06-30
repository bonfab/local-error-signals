import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from models.local_loss_net import LocalLossNet
from optimizers.sam import SAM
from utils.data import to_one_hot
from models.local_loss_blocks import LocalLossBlockLinear, LocalLossBlockConv
from utils.logging import get_logger, get_csv_logger, retire_logger
from utils.models import count_parameters


class Trainer:

    def __init__(self, cfg, model, train_set, valid_set, logger=None):
        self.cfg = cfg
        self.model = model
        self.logger = logger
        if logger is None:
            self.logger = get_logger(__name__, logging.INFO)
        self.csv_logger = None
        self.logger.info(model.__str__())
        self.logger.info(f'Model has {count_parameters(model)} parameters influenced by global loss')

        self.train_set = train_set
        self.valid_set = valid_set


        self.model.set_learning_rate(cfg.lr)
        self.select_optimizer()

        if cfg.gpus:
            self.model.cuda()

        self.train_loader = self.get_loader(train_set)
        self.valid_loader = self.get_loader(valid_set, shuffle=False)

    def get_loader(self, data_set, shuffle=True):
        kwargs = {'pin_memory': True} if self.cfg.gpus else {}
        return torch.utils.data.DataLoader(
            data_set,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.data_loader_workers,
            worker_init_fn=lambda worker_id: np.random.seed(
                 self.cfg.seed + worker_id),
            **kwargs)

    def select_optimizer(self):
        if self.cfg.sam:
            if self.cfg.sam:
                if self.cfg.optim == 'sgd':
                    self.optimizer = SAM(self.model.parameters(), optim.SGD, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
                                         momentum=self.cfg.momentum)
                elif self.cfg.optim == 'adam' or self.cfg.optim == 'amsgrad':
                    self.optimizer = SAM(self.model.parameters(), optim.Adam, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
                                         amsgrad=self.cfg.optim == 'amsgrad')
                if self.cfg.exponential_lr_scheduler:
                    self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer.base_optimizer,
                                                                            self.cfg.exponential_lr_gamma)
        elif self.cfg.optim == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
                                       momentum=self.cfg.momentum)
        elif self.cfg.optim == 'adam' or self.cfg.optim == 'amsgrad':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
                                        amsgrad=self.cfg.optim == 'amsgrad')
        else:
            raise ValueError(f'Unknown optimizer {self.cfg.optim}')

    def trainiter(self):

        ''' Train model on train set'''
        self.model.train()
        correct = 0
        loss_total_local = 0
        loss_total_global = 0

        # Add progress bar
        if self.cfg.progress_bar:
            pbar = tqdm(total=len(self.train_loader))

        # Clear layerwise statistics
        if not self.cfg.no_print_stats:
            for m in self.model.modules():
                if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv):
                    m.clear_stats()

        # Loop train set
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.cfg.gpus:
                data, target = data.cuda(), target.cuda()
            target_ = target
            target_onehot = to_one_hot(target, self.cfg.num_classes)
            if self.cfg.gpus:
                target_onehot = target_onehot.cuda()

            # Clear accumulated gradient
            self.optimizer.zero_grad()
            self.model.optim_zero_grad()

            output, loss = self.model(data, target, target_onehot)
            loss_total_local += loss * data.size(0)
            loss = F.cross_entropy(output, target)
            if self.cfg.loss_sup == 'predsim' and not self.cfg.backprop:
                loss *= (1 - self.cfg.beta)
            loss_total_global += loss.item() * data.size(0)

            # Backward pass and optimizer step
            # For local loss functions, this will only affect output layer
            loss.backward()
            if self.cfg.sam:
                self.optimizer.first_step(zero_grad=True)
                self.model.local_loss_eval()
                F.cross_entropy(self.model(data), target).backward()
                self.optimizer.second_step()
                self.model.local_loss_train()
            else:
                self.optimizer.step()

            # If special option for no detaching is set, update weights also in hidden layers
            if self.cfg.no_detach:
                self.model.optim_step()

            pred = output.max(1)[1]  # get the index of the max log-probability
            #print(pred, target)
            correct += pred.eq(target_).cpu().sum()

            # Update progress bar
            if self.cfg.progress_bar:
                pbar.set_postfix(loss=loss.item(), refresh=False)
                pbar.update()

            #break

        if self.cfg.progress_bar:
            pbar.close()

        loss_average_local = loss_total_local / len(self.train_loader.dataset)
        loss_average_global = loss_total_global / len(self.train_loader.dataset)
        error_percent = 100 - 100.0 * float(correct) / len(self.train_loader.dataset)
        """string_print = 'Train epoch={}, lr={:.2e}, loss_local={:.4f}, loss_global={:.4f}, error={:.3f}%, mem={:.0f}MiB, max_mem={:.0f}MiB\n'.format(
            epoch,
            lr, 
            loss_average_local,
            loss_average_global,
            error_percent,
            torch.cuda.memory_allocated()/1e6,
            torch.cuda.max_memory_allocated()/1e6)
        if not args.no_print_stats:
            for m in model.modules():
                if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv):
                    string_print += m.print_stats() 
        print(string_print)"""

        return loss_average_local, loss_average_global, 100-error_percent

    def validate(self):
        ''' Run model on validation set '''
        self.model.eval()
        loss_total_local = 0
        valid_loss = 0
        correct = 0

        # Clear layerwise statistics
        if not self.cfg.no_print_stats:
            for m in self.model.modules():
                if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv):
                    m.clear_stats()

        # Loop valid set
        for data, target in self.valid_loader:
            if self.cfg.gpus:
                data, target = data.cuda(), target.cuda()
            target_ = target
            target_onehot = to_one_hot(target, self.cfg.num_classes)
            if self.cfg.gpus:
                target_onehot = target_onehot.cuda()

            with torch.no_grad():
                output, loss = self.model(data, target, target_onehot)
                loss_total_local += loss * data.size(0)
                valid_loss += F.cross_entropy(output, target).item() * data.size(0)
            pred = output.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target_).cpu().sum()

            #break

        loss_average_local = loss_total_local / len(self.train_loader.dataset)
        loss_average = valid_loss / len(self.valid_loader.dataset)
        if self.cfg.loss_sup == 'predsim' and not self.cfg.backprop:
            loss_average *= (1 - self.cfg.beta)
        error_percent = 100 - 100.0 * float(correct) / len(self.valid_loader.dataset)
        """string_print = 'Validate loss_global={:.4f}, error={:.3f}%\n'.format(loss_average, error_percent)
        if not self.cfg.no_print_stats:
            for m in self.model.modules():
                if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv):
                    string_print += m.print_stats()
        print(string_print)"""

        return loss_average_local, loss_average, 100-error_percent

    def fit(self):
        ''' The main training and testing loop '''
        torch.autograd.set_detect_anomaly(True)
        # start_epoch = 1 if checkpoint is None else 1 + checkpoint['epoch']
        self.csv_logger = get_csv_logger(file_path='training_results.csv')
        # TODO: add checkpoint loading
        for epoch in range(self.cfg.epochs):
            # Train and test
            train_loss_local, train_loss_global, train_acc = self.trainiter()
            self.log_results(epoch, train_loss_local, train_loss_global, train_acc, msg='Train Run')
            valid_loss_local, valid_loss_global, valid_acc = self.validate()
            self.log_results(epoch, valid_loss_local, valid_loss_global, valid_acc, msg='Validation Run')
            self.csv_logger.info(f'{epoch},'
                                 f'{train_loss_local},{train_loss_global},{train_acc/100},'
                                 f'{valid_loss_local},{valid_loss_global},{valid_acc/100}')
            if self.cfg.exponential_lr_scheduler:
                self.scheduler.step()
                if isinstance(self.model, LocalLossNet):
                    self.model.lr_scheduler_step()
            self.save_checkpoint(epoch, self.model, self.optimizer)

        retire_logger(self.csv_logger)

    def save_checkpoint(self, epoch, model, optimizer):
        # Check if to save checkpoint
        if self.cfg.checkpointing:
            os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()

            }, os.path.join(self.cfg.checkpoint_dir, f'{epoch}.pt'))

    def log_results(self, epoch, local_loss, global_loss, accuracy, msg=""):
        self.logger.info(f'{msg}\n\t\t'
                         f'epoch: {epoch}\n\t\t'
                         f'local loss: {local_loss:.4f}\n\t\t'
                         f'global loss: {global_loss:.4f}\n\t\t'
                         f'accuracy: {accuracy:.3f}%\n\t\t'
                         f'mem: {torch.cuda.memory_allocated() / 1e6:.0f} MiB\n\t\t'
                         f'max mem: {torch.cuda.max_memory_allocated() / 1e6:.0f} MiB')
