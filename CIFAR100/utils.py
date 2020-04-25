import torch
import torchvision
import torchvision.transforms as transforms

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_cifar10(args):

    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader

def get_cifar100(args):
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    train_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader

def get_optim_lr(optimizer):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    return lr


class _GammaScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_gamma', group['gamma'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_gamma' not in group:
                    raise KeyError("param 'initial_gamma' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_gammas = list(map(lambda group: group['initial_gamma'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_gamma(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, gamma in zip(self.optimizer.param_groups, self.get_gamma()):
            param_group['gamma'] = gamma

class StepGamma(_GammaScheduler):
    def __init__(self, optimizer, step_size, gamma=0.5, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super(StepGamma, self).__init__(optimizer, last_epoch)

    def get_gamma(self):
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group['gamma'] for group in self.optimizer.param_groups]
        return [1.0 - (1.0 - group['gamma']) * self.gamma
                for group in self.optimizer.param_groups]


class AdaptiveGamma(_GammaScheduler):
    def __init__(self, optimizer, initial_gamma, adaptation=None, epochs=200, last_epoch=-1):
        self.initial_gamma = initial_gamma
        self.adaptation = adaptation
        self.epochs = epochs
        super(AdaptiveGamma, self).__init__(optimizer, last_epoch)

    def get_gamma(self):
        linear = self.initial_gamma + (1.0 - self.initial_gamma) / (self.epochs - 1) * self.last_epoch
        cos = 1.0 - (1.0 - self.initial_gamma) * (np.cos((self.last_epoch / self.epochs) * np.pi) + 1) * 0.5
        sin = self.initial_gamma + (1.0 - self.initial_gamma) * np.sin((self.last_epoch / self.epochs) * np.pi / 2)
        tuples = [
            ('linear', linear),
            ('cos', cos),
            ('sin', sin),
            (None, self.initial_gamma)
        ]

        find_name = find_name = lambda name: name[0] == self.adaptation
        result = list(filter(find_name, tuples))[0][1]

        return [result for group in self.optimizer.param_groups]
