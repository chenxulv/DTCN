'''A wrapper class for optimizer '''
import numpy as np
'''A wrapper class for scheduled optimizer '''
import numpy as np


def get_triangular_lr(iteration, stepsize, base_lr, max_lr):
    """
    Given the inputs, calculates the lr that should be
    applicable for this iteration
    """
    cycle = np.floor(1 + iteration / (2 * stepsize))
    x = np.abs(iteration / stepsize - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))
    return lr


class ScheduledOptim_V2():
    '''A simple wrapper class for learning rate scheduling'''
    def __init__(self, optimizer, base_lr=1e-6, max_lr=1e-3, n_warmup_steps=4000):
        self._optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr

        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):

        cycle = np.floor(1 + self.n_steps / (2 * self.n_warmup_steps))

        x = np.abs(self.n_steps / self.n_warmup_steps - 2 * cycle + 1)

        lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))

        return lr

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self._get_lr_scale()
        self.lr = lr

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''
    def __init__(self, optimizer, init_lr=2.0, d_model=256, n_warmup_steps=4000, max_lr=1e-3):
        self._optimizer = optimizer
        self.init_lr = max_lr * (n_warmup_steps**(0.5) * (d_model**(0.5)))  #init_lr
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0
        self.max_lr = max_lr

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model**-0.5) * min(n_steps**(-0.5), n_steps * n_warmup_steps**(-1.5))

        # return (d_model**-0.5) * min(max(n_steps**(-0.5), self.max_lr*0.1*(d_model**0.5)/self.init_lr), n_steps * n_warmup_steps**(-1.5))

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        self.lr = lr

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class ScheduledOptim_V3():
    '''A simple wrapper class for learning rate scheduling'''
    def __init__(self, optimizer, init_lr=1.0, d_model=256, n_warmup_steps=4000):
        self._optimizer = optimizer
        self.init_lr = init_lr
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps

        # if n_steps * n_warmup_steps**(-1.5) < n_steps**(-0.5):
        #     return (d_model**-0.5) * n_steps * n_warmup_steps**(-1.5)
        # else:
        #     return max((d_model**-0.5) * n_steps**(-0.5), 5e-4)
        # return (d_model**-0.5) * min(n_steps**(-0.5), n_steps * n_warmup_steps**(-1.5))

        return (d_model**-0.5) * min(max(n_steps**(-0.5), 5e-4*(d_model**0.5)/self.init_lr), n_steps * n_warmup_steps**(-1.5))

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        self.lr = lr

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


if __name__ == '__main__':
    import matplotlib as mpl
    mpl.use('agg')
    import matplotlib.pyplot as plt

    import torch
    from torch import optim, nn

    model = nn.Linear(10, 20)

    optimizer = ScheduledOptim(optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09), 2, 256, 1000)
    lrs = []
    for i in range(40000):
        optimizer.step_and_update_lr()
        lrs.append(optimizer.lr)

    print(lrs[:10])
    plt.figure(figsize=(16, 9))
    plt.plot([i for i in range(len(lrs))], lrs)
    plt.ylabel('current lr', fontsize=20)
    plt.xlabel('# update step', fontsize=20)
    #plt.savefig('lrs_v2.png')

    optimizer = ScheduledOptim_V2(optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09), base_lr=1e-6, max_lr=1e-3, n_warmup_steps=442)
    lrs = []
    for i in range(10000):
        optimizer.step_and_update_lr()
        lrs.append(optimizer.lr)

    print(lrs[:10])
    #plt.figure(figsize=(16, 9))
    plt.plot([i for i in range(len(lrs))], lrs)
    plt.ylabel('current lr', fontsize=20)
    plt.xlabel('# update step', fontsize=20)
    plt.savefig('lrs_v2.png')

    optimizer = ScheduledOptim_V3(optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09), init_lr=0.5, d_model=256, n_warmup_steps=1000)
    lrs = []
    for i in range(40000):
        optimizer.step_and_update_lr()
        lrs.append(optimizer.lr)

    print(lrs[:10])
    #plt.figure(figsize=(16, 9))
    plt.plot([i for i in range(len(lrs))], lrs)
    plt.ylabel('current lr', fontsize=20)
    plt.xlabel('# update step', fontsize=20)
    plt.savefig('lrs_v3.png')

    pass
