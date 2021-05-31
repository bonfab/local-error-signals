import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from src.utils.data import to_one_hot
from src.models.local_loss_blocks import LocalLossBlockLinear, LocalLossBlockConv
from src.utils.models import count_parameters


class Trainer:

    def __init__(self, cfg, model, train_set, valid_set, logger=None):
        self.cfg = cfg
        self.model = model

        print(model)
        print(f'Model has {count_parameters(model)} parameters influenced by global loss')

        self.train_set = train_set
        self.valid_set = valid_set

        if cfg.optim == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                                       momentum=cfg.momentum)
        elif cfg.optim == 'adam' or cfg.optim == 'amsgrad':
            self.optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                                        amsgrad=cfg.optim == 'amsgrad')
        else:
            raise ValueError(f'Unknown optimizer {cfg.optim}')

        if cfg.gpus:
            self.model.cuda()

        self.train_loader = self.get_loader(train_set)
        self.valid_loader = self.get_loader(valid_set)

    def get_loader(self, data_set):
        kwargs = {'pin_memory': True} if self.cfg.gpus else {}
        return torch.utils.data.DataLoader(
            data_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,
            **kwargs)

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
            self.optimizer.step()

            # If special option for no detaching is set, update weights also in hidden layers
            if self.cfg.no_detach:
                self.model.optim_step()

            pred = output.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target_).cpu().sum()

            # Update progress bar
            if self.cfg.progress_bar:
                pbar.set_postfix(loss=loss.item(), refresh=False)
                pbar.update()

        if self.cfg.progress_bar:
            pbar.close()

        # Format and print debug string
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

        return loss_average_local + loss_average_global, error_percent  # , string_print


    def validate(self):
        ''' Run model on test set '''
        self.model.eval()
        test_loss = 0
        correct = 0

        # Clear layerwise statistics
        if not self.cfg.no_print_stats:
            for m in self.model.modules():
                if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv):
                    m.clear_stats()

        # Loop test set
        for data, target in self.valid_loader:
            if self.cfg.gpus:
                data, target = data.cuda(), target.cuda()
            target_ = target
            target_onehot = to_one_hot(target, self.cfg.num_classes)
            if self.cfg.gpus:
                target_onehot = target_onehot.cuda()

            with torch.no_grad():
                output, _ = self.model(data, target, target_onehot)
                test_loss += F.cross_entropy(output, target).item() * data.size(0)
            pred = output.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target_).cpu().sum()

        # Format and print debug string
        loss_average = test_loss / len(self.valid_loader.dataset)
        if self.cfg.loss_sup == 'predsim' and not self.cfg.backprop:
            loss_average *= (1 - self.cfg.beta)
        error_percent = 100 - 100.0 * float(correct) / len(self.valid_loader.dataset)
        string_print = 'Validate loss_global={:.4f}, error={:.3f}%\n'.format(loss_average, error_percent)
        if not self.cfg.no_print_stats:
            for m in self.model.modules():
                if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv):
                    string_print += m.print_stats()
        print(string_print)

        return loss_average, error_percent, string_print


    def fit(self):
        ''' The main training and testing loop '''
        # start_epoch = 1 if checkpoint is None else 1 + checkpoint['epoch']

        # TODO: add checkpoint loading
        for epoch in range(self.cfg.epochs + 1):
            # Train and test
            train_loss, train_error = self.trainiter()
            print(train_loss, train_error)
            test_loss, test_error= self.validate()
            print(test_loss, test_error)

            """
            # Check if to save checkpoint
            if args.save_dir is not '':
                # Resolve log folder and checkpoint file name
                filename = 'chkp_ep{}_lr{:.2e}_trainloss{:.2f}_testloss{:.2f}_trainerr{:.2f}_testerr{:.2f}.tar'.format(
                        epoch, lr, train_loss, test_loss, train_error, test_error)
                dirname = os.path.join(args.save_dir, args.dataset)
                dirname = os.path.join(dirname, '{}_mult{:.1f}'.format(args.model, args.feat_mult))
                dirname = os.path.join(dirname, '{}_{}x{}_{}_{}_dimdec{}_alpha{}_beta{}_bs{}_cpb{}_drop{}{}_bn{}_{}_wd{}_bp{}_detach{}_lr{:.2e}'.format(
                        args.nonlin, args.num_layers, args.num_hidden, args.loss_sup + '-bio' if args.bio else args.loss_sup, args.loss_unsup, args.dim_in_decoder, args.alpha,
                        args.beta, args.batch_size, args.classes_per_batch, args.dropout, '_cutout{}x{}'.format(args.n_holes, args.length) if args.cutout else '',
                        int(not args.no_batch_norm), args.optim, args.weight_decay, int(args.backprop), int(not args.no_detach), args.lr))
    
                # Create log directory
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                elif epoch==1 and os.path.exists(dirname):
                    # Delete old files
                    for f in os.listdir(dirname):
                        os.remove(os.path.join(dirname, f))
    
                # Add log entry to log file
                with open(os.path.join(dirname, 'log.txt'), 'a') as f:
                    if epoch == 1:
                        f.write('{}\n\n'.format(args))
                        f.write('{}\n\n'.format(model))
                        f.write('{}\n\n'.format(optimizer))
                        f.write('Model {} has {} parameters influenced by global loss\n\n'.format(args.model, count_parameters(model)))
                    f.write(train_print)
                    f.write(test_print)
                    f.write('\n')
                    f.close()
    
                # Save checkpoint for every epoch
                torch.save({
                    'epoch': epoch,
                    'args': args,
                    'state_dict': model.state_dict() if (save_state_dict or epoch==args.epochs) else None,
                    'train_loss': train_error,
                    'train_error': train_error,
                    'test_loss': test_loss,
                    'test_error': test_error,
                }, os.path.join(dirname, filename))
    
                # Save checkpoint for last epoch with state_dict (for resuming)
                torch.save({
                    'epoch': epoch,
                    'args': args,
                    'state_dict': model.state_dict(),
                    'train_loss': train_error,
                    'train_error': train_error,
                    'test_loss': test_loss,
                    'test_error': test_error,
                }, os.path.join(dirname, 'chkp_last_epoch.tar'))"""
