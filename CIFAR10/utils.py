'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import numpy as np

import torch.nn as nn
import torch.nn.init as init

from torch.optim.optimizer import Optimizer

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_gamma(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['gamma']
    
def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)

# TOTAL_BAR_LENGTH = 65.
# last_time = time.time()
# begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    sys.stdout.flush()

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')

    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

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
        log = (1 - self.initial_gamma) * (1 - np.log((self.last_epoch + 1) / self.epochs) / np.log(1 / self.epochs)) + self.initial_gamma
        tuples = [
            ('linear', linear), 
            ('cos', cos), 
            ('sin', sin),
            ('log', log),
            (None, self.initial_gamma)
        ]
        
        find_name = find_name = lambda name: name[0] == self.adaptation
        result = list(filter(find_name, tuples))[0][1]

        return [result for group in self.optimizer.param_groups]
