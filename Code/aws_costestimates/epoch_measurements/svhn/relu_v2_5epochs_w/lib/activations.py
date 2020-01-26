from scipy.misc import factorial2
import numpy as np
import torch


class Hermite:
    def __init__(self, num_pol=5):
        self.h = []

        def h0(x):
            return torch.ones_like(x)

        self.h.append(h0)

        def h1(x):
            return x

        self.h.append(h1)

        def h2(x):
            return (x**2 - 1) / np.sqrt(np.math.factorial(2))

        self.h.append(h2)

        def h3(x):
            return (x**3 - 3 * x) / np.sqrt(np.math.factorial(3))

        self.h.append(h3)

        def h4(x):
            return (x**4 - 6 * (x**2) + 3) / np.sqrt(np.math.factorial(4))

        self.h.append(h4)

        def h5(x):
            return (x**5 - 10 * x**3 + 15 * x) / np.sqrt(np.math.factorial(5))

        self.h.append(h5)

        def h6(x):
            return (x**6 - 15 * x**4 + 45 * x**2 - 15) / np.sqrt(
                np.math.factorial(6))

        self.h.append(h6)

        def h7(x):
            return (x**7 - 21 * x**5 + 105 * x**3 - 105 * x) / np.sqrt(
                np.math.factorial(7))

        self.h.append(h7)

        def h8(x):
            return (x**8 - 28 * x**6 + 210 * x**4 - 420 * x**2 +
                    105) / np.sqrt(np.math.factorial(8))

        self.h.append(h8)

        def h9(x):
            return (x**9 - 36 * x**7 + 378 * x**5 - 1260 * x**3 +
                    945 * x) / np.sqrt(np.math.factorial(9))

        self.h.append(h9)

        def h10(x):
            return (x**10 - 45 * x**8 + 630 * x**6 - 3150 * x**4 + 4725 * x**2
                    - 945) / np.sqrt(np.math.factorial(10))

        self.h.append(h10)

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

    def hermite(self, x, k, num_pol=5):
        evals = 0.0
        for i in range(num_pol):
            evals += k[i] * self.h[i](x)
        return evals

    def getVectors(self, x, num_pol=5):
        out = []
        for i in range(num_pol):
            out.append(self.h[i](x).item())
        return out
