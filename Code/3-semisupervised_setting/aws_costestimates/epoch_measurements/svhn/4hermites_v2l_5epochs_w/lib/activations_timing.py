'''
This version of activations file has been used to compute the time and cost of per epoch run on AWS
'''
from scipy.misc import factorial2
import numpy as np
import torch


class Hermite:
    def __init__(self, num_pol=5):
        pass

    def get_initializations(self, num_pol=5, copy_fun='relu'):
        k = []
        if copy_fun == 'relu':
            for n in range(num_pol):
                if n == 0:
                    k.append(1.0 / np.sqrt(2 * np.pi))
                elif n == 1:
                    k.append(1.0 / 2)
                elif n == 2:
                    k.append(1.0 / np.sqrt(4 * np.pi))
                elif n > 2 and n % 2 == 0:
                    c = 1.0 * factorial2(n - 3)**2 / np.sqrt(
                        2 * np.pi * np.math.factorial(n))
                    k.append(c)
                elif n >= 2 and n % 2 != 0:
                    k.append(0.0)
        return k

    def get_vars(self, num_pol=5, copy_fun='relu', seed=1,
                 dtype=torch.float32):
        torch.manual_seed(seed)
        if copy_fun == 'relu':
            k = self.get_initializations(num_pol, copy_fun)
            p = 0.00001 * torch.randn(
                num_pol, requires_grad=False) + torch.tensor(k)
            p_param = torch.nn.Parameter(p)
            return p_param

    def _hermite(self, x, k, num_pol=5):
        return (
            (torch.full_like(x, k[0].item()) * torch.ones_like(x)) +
            (torch.full_like(x, k[1].item()) * x) +
            (torch.full_like(x, k[2].item()) * (x**2 - torch.full_like(x, 1)) /
             torch.full_like(x, 1.4142135623730951)) +
            (torch.full_like(x, k[3].item()) *
             (x**3 - torch.full_like(x, 3) * x) / torch.full_like(
                 x, 2.449489742783178)))

    def hermite(self, x, k, num_pol=5):
        return (
            (torch.full_like(x, k[0].item()) * torch.ones_like(x)) +
            (torch.full_like(x, k[1].item()) * x) +
            (torch.full_like(x, k[2].item()) * (x**2 - 1) / 1.4142135623730951)
            + (torch.full_like(x, k[3].item()) *
               (x**3 - 3 * x) / 2.449489742783178))

    def getVectors(self, x, num_pol=5):
        out = []
        for i in range(num_pol):
            out.append(self.h[i](x).item())
        return out
