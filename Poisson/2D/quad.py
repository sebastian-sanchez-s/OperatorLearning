import torch
import numpy as np


def gaussian_quadrature(n):
    int_p_1d, int_w_1d = np.polynomial.legendre.leggauss(n)
    int_p_1d = (1 + torch.tensor(int_p_1d).float()) / 2
    int_w_1d = 0.5*torch.tensor(int_w_1d).float()

    int_p = torch.stack(torch.meshgrid((int_p_1d, int_p_1d), indexing='xy')).reshape(2, -1).float()
    int_w = torch.einsum('i,j->ij', int_w_1d, int_w_1d).ravel().float()

    return int_p, int_w, int_p_1d, int_w_1d



def trapezoidal_rule(n):
    int_p_1d = torch.linspace(0, 1, n)
    int_w_1d = torch.ones(n)
    int_w_1d[[0, -1]] /= 2

    int_p = torch.stack(torch.meshgrid(int_p_1d, int_p_1d, indexing='xy')).view(2, -1)
    int_w = (torch.kron(int_w_1d, int_w_1d) * (1/(n-1))**2).ravel()

    return int_p, int_w, int_p_1d, int_w_1d