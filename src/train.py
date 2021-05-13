import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def train(epoch, lr):

    ''' Train model on train set'''
    model.train()
    correct = 0
    loss_total_local = 0
    loss_total_global = 0
    
    # Add progress bar
    if args.progress_bar:
        pbar = tqdm(total=len(train_loader))
        
    # Clear layerwise statistics
    if not args.no_print_stats:
        for m in model.modules():
            if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv):
                m.clear_stats()
                
    # Loop train set
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        target_ = target
        target_onehot = to_one_hot(target, num_classes)
        if args.cuda:
            target_onehot = target_onehot.cuda()
  
        # Clear accumulated gradient
        optimizer.zero_grad()
        model.optim_zero_grad()
                    
        output, loss = model(data, target, target_onehot)
        loss_total_local += loss * data.size(0)
        loss = F.cross_entropy(output, target)
        if args.loss_sup == 'predsim' and not args.backprop:
            loss *= (1 - args.beta) 
        loss_total_global += loss.item() * data.size(0)
             
        # Backward pass and optimizer step
        # For local loss functions, this will only affect output layer
        loss.backward()
        optimizer.step()
        
        # If special option for no detaching is set, update weights also in hidden layers
        if args.no_detach:
            model.optim_step()
        
        pred = output.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target_).cpu().sum()
        
        # Update progress bar
        if args.progress_bar:
            pbar.set_postfix(loss=loss.item(), refresh=False)
            pbar.update()
            
    if args.progress_bar:
        pbar.close()
        
    # Format and print debug string
    loss_average_local = loss_total_local / len(train_loader.dataset)
    loss_average_global = loss_total_global / len(train_loader.dataset)
    error_percent = 100 - 100.0 * float(correct) / len(train_loader.dataset)
    string_print = 'Train epoch={}, lr={:.2e}, loss_local={:.4f}, loss_global={:.4f}, error={:.3f}%, mem={:.0f}MiB, max_mem={:.0f}MiB\n'.format(
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
    print(string_print)
    
    return loss_average_local+loss_average_global, error_percent, string_print
    
    
def test(epoch):
    ''' Run model on test set '''
    model.eval()
    test_loss = 0
    correct = 0
    
    # Clear layerwise statistics
    if not args.no_print_stats:
        for m in model.modules():
            if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv):
                m.clear_stats()
    
    # Loop test set
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        target_ = target
        target_onehot = to_one_hot(target, num_classes)
        if args.cuda:
            target_onehot = target_onehot.cuda()
        
        with torch.no_grad():
            output, _ = model(data, target, target_onehot)
            test_loss += F.cross_entropy(output, target).item() * data.size(0)
        pred = output.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target_).cpu().sum()

    # Format and print debug string
    loss_average = test_loss / len(test_loader.dataset)
    if args.loss_sup == 'predsim' and not args.backprop:
        loss_average *= (1 - args.beta)
    error_percent = 100 - 100.0 * float(correct) / len(test_loader.dataset)
    string_print = 'Test loss_global={:.4f}, error={:.3f}%\n'.format(loss_average, error_percent)
    if not args.no_print_stats:
        for m in model.modules():
            if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv):
                string_print += m.print_stats()                
    print(string_print)
    
    return loss_average, error_percent, string_print

''' The main training and testing loop '''
start_epoch = 1 if checkpoint is None else 1 + checkpoint['epoch']
for epoch in range(start_epoch, args.epochs + 1):
    # Decide learning rate
    lr = args.lr * args.lr_decay_fact ** bisect_right(args.lr_decay_milestones, (epoch-1))
    save_state_dict = False
    for ms in args.lr_decay_milestones:
        if (epoch-1) == ms:
            print('Decaying learning rate to {}'.format(lr))
            decay = True
        elif epoch == ms:
            save_state_dict = True

    # Set learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    model.set_learning_rate(lr)
    
    # Check if to remove NClassRandomSampler from train_loader
    if args.classes_per_batch_until_epoch > 0 and epoch > args.classes_per_batch_until_epoch and isinstance(train_loader.sampler, NClassRandomSampler):
        print('Remove NClassRandomSampler from train_loader')
        train_loader = torch.utils.data.DataLoader(dataset_train, sampler = None, batch_size=args.batch_size, shuffle=True, **kwargs)
    
    # Train and test    
    train_loss,train_error,train_print = train(epoch, lr)
    test_loss,test_error,test_print = test(epoch)

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
        }, os.path.join(dirname, 'chkp_last_epoch.tar')) 
