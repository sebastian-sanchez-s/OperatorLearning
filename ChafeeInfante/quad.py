import numpy as np
import torch


def gaussian_quadrature(a, b, n):
    int_p, int_w = np.polynomial.legendre.leggauss(n)
    mapped_int_p = (a+b)/2 + (b-a)/2 * int_p
    mapped_int_w = int_w * (b-a) / 2

    return torch.tensor(mapped_int_p).float(), torch.tensor(mapped_int_w).float()


def trapezoidal_rule(a, b, n):
    int_p = torch.linspace(a,  b, n).float()
    int_w = torch.ones_like(int_p) * (b-a) / (n-1)
    int_w[[0, -1]] /= 2
    return int_p, int_w
