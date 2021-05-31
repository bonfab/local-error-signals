import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def get_datasets(cfg, data_dir, logger=None):
    if logger is not None:
        logger.info(f'selecting dataset {cfg.name}')
    if cfg.name.__contains__("CIFAR-10"):
        return get_CIFAR10_train_test(data_dir)
    else:
        raise ValueError(f"Dataset {cfg.name} not supported")


def get_CIFAR10_train_test(root):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    return trainset, testset


def to_one_hot(y, n_dims=None):
    ''' Take integer tensor y with n dims and convert it to 1-hot representation with n+1 dims. '''
    y_tensor = y.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot


class NClassRandomSampler(torch.utils.data.sampler.Sampler):
    r'''Samples elements such that most batches have N classes per batch.
    Elements are shuffled before each epoch.
    Arguments:
        targets: target class for each example in the dataset
        n_classes_per_batch: the number of classes we want to have per batch
    '''

    def __init__(self, targets, n_classes_per_batch, batch_size):
        self.targets = targets
        self.n_classes = int(np.max(targets))
        self.n_classes_per_batch = n_classes_per_batch
        self.batch_size = batch_size

    def __iter__(self):
        n = self.n_classes_per_batch

        ts = list(self.targets)
        ts_i = list(range(len(self.targets)))

        np.random.shuffle(ts_i)
        # algorithm outline:
        # 1) put n examples in batch
        # 2) fill rest of batch with examples whose class is already in the batch
        while len(ts_i) > 0:
            idxs, ts_i = ts_i[:n], ts_i[n:]  # pop n off the list

            t_slice_set = set([ts[i] for i in idxs])

            # fill up idxs until we have n different classes in it. this should be quick.
            k = 0
            while len(t_slice_set) < 10 and k < n * 10 and k < len(ts_i):
                if ts[ts_i[k]] not in t_slice_set:
                    idxs.append(ts_i.pop(k))
                    t_slice_set = set([ts[i] for i in idxs])
                else:
                    k += 1

            # fill up idxs with indexes whose classes are in t_slice_set.
            j = 0
            while j < len(ts_i) and len(idxs) < self.batch_size:
                if ts[ts_i[j]] in t_slice_set:
                    idxs.append(ts_i.pop(j))  # pop is O(n), can we do better?
                else:
                    j += 1

            if len(idxs) < self.batch_size:
                needed = self.batch_size - len(idxs)
                idxs += ts_i[:needed]
                ts_i = ts_i[needed:]

            for i in idxs:
                yield i

    def __len__(self):
        return len(self.targets)
