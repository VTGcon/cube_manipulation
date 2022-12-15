import random
import torch
import numpy as np


class ExpirienceReplay:
    def __init__(self, size=10000):
        self.data = []
        self.max_size = size
        self.i = 0

    def add(self, transition):
        if len(self.data) < self.max_size:
            self.data.append(transition)
        else:
            self.data[self.i] = transition
            self.i = (self.i + 1) % self.max_size

    def sample(self, size):
        batch = random.sample(self.data, size)
        ans = list(zip(*batch))
        for i in range(len(ans) - 1):
            ans[i] = torch.FloatTensor(np.array(ans[i]).squeeze(axis=1))
        ans[-1] = torch.tensor((np.array(ans[-1]) * 1).squeeze(axis=1))
        return ans

