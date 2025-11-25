#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn
from pau import PAU

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator


# In[4]:


a, b = 0, 1
npoints = 10

n_elem = npoints-1
area_elem = 1 / (n_elem-1)
int_p_1d = torch.linspace(0, 1, n_elem)
int_w_1d = torch.ones(n_elem)
int_w_1d[[0, -1]] /= 2

int_p = torch.stack(torch.meshgrid((int_p_1d, int_p_1d), indexing='xy')).reshape(2, -1)
int_w = (torch.kron(int_w_1d, int_w_1d) * area_elem**2).ravel()


# In[5]:


def build_net(layers):
    modules = []
    for i in range(len(layers)-1):
        modules.append(nn.Linear(layers[i], layers[i+1]))
        modules.append(PAU(initial_shape='relu'))

    net = nn.Sequential(*modules)
    opt = torch.optim.Adam(net.parameters())

    return net, opt

green_net, _ = build_net([4, 100, 100, 100, 100, 1])

model_data = torch.load('dataset/ds20k/model/100_ds20k_kernel_10qs.model')

green_net.load_state_dict(model_data['net'])
epoch = model_data['epoch']

green_net = torch.compile(green_net)


# In[6]:


def model(x, t):
    '''
        Computes x(t) = int k(t,s) x(s) ds

        t [2]x[points]
        x [bs]x[int_points]
    '''
    result = torch.zeros(x.shape[0], t.shape[1])
    for j, t1t2 in enumerate(t.T):
        t1t2_rep = t1t2.repeat((int_p.shape[1], 1)).T
        Kt1t2s1s2 = green_net(torch.cat((t1t2_rep, int_p)).T).squeeze()

        for i, xi in enumerate(x):
            result[i, j] = (xi * Kt1t2s1s2) @ int_w

    return result


# In[7]:


def loss(U, Upred):
    #  num [bs]
    num = (U - Upred)**2 @ int_w
    #  den [bs]
    den = U**2 @ int_w

    return (num / den).mean()


# In[6]:


NT = 100
t = torch.linspace(a, b, NT)
T = torch.stack(torch.meshgrid(t, t, indexing='xy')).reshape(2, -1)


# In[8]:


def compute_kernel(M: int):
    t = torch.linspace(0, 1, M)
    kernel = torch.zeros(M**4)
    k = 0
    for i in range(M):
        for j in range(M):
            for ii in range(M):
                for jj in range(M):
                    kernel[k] = green_net(torch.tensor([t[i], t[j], t[ii], t[jj]]))
                    k += 1
    return kernel


# In[10]:

M = 25
kernel = compute_kernel(M)

# In[ ]:


kernel = kernel.detach().view(M**2, M**2)


# In[ ]:


plt.figure(figsize=(8, 8))
plt.imshow(kernel, cmap='jet')
plt.axis('off')
plt.colorbar(location='bottom', aspect=15, fraction=0.05, pad=0.04)
plt.savefig(f'figures/epoch_{epoch}_kernel_10qs_matrix_{M}x{M}.png')
plt.show()


# In[19]:


n_elem = M
area_elem = 1 / (n_elem-1)
int_p_1d = torch.linspace(0, 1, n_elem)
int_w_1d = torch.ones(n_elem)
int_w_1d[[0, -1]] /= 2

int_p = torch.stack(torch.meshgrid((int_p_1d, int_p_1d), indexing='xy')).reshape(2, -1)
int_w = (torch.kron(int_w_1d, int_w_1d) * area_elem**2).ravel()


# In[7]:


# N = 12
# t = torch.linspace(0, 1, N)
# T = torch.stack(torch.meshgrid((t, t, t, t), indexing='xy')).reshape(4, -1) 

# kernel = torch.zeros(T.shape[1])

# for i, ts in enumerate(T.T):
    # kernel[i] = green_net(ts)


# In[8]:


# kernel_ = kernel.detach().view(N**2, N**2)


# In[18]:


# plt.figure(figsize=(8, 8))
# plt.imshow(kernel_, cmap='jet')
# plt.axis('off')
# plt.colorbar(location='bottom', aspect=15, fraction=0.05, pad=0.04)
# plt.savefig(f'figures/epoch_{epoch}_kernel_10qs_matrix_{N}x{N}.png')
# plt.show()


# # Kernel2Basis

# In[13]:


U, S, V = torch.linalg.svd(kernel.view(M**2, M**2))


# In[14]:


BASIS = U.T[:8]

for i, b in enumerate(BASIS):
    plt.figure(figsize=(5, 5))
    plt.imshow(b.reshape(M, M), cmap='jet')
    plt.axis('off')
    plt.colorbar(location='right', aspect=15, fraction=0.05, pad=0.09)
    plt.savefig(f'figures/epoch_{epoch}_kernel_10qs_kernel2basis_{i}_resolution_{M}.png', bbox_inches='tight')
    plt.show()


# In[ ]:


#  Gram-Schmidt
V = BASIS
n = len(V)
Q = [None for _ in range(n)]
R = [[0.0 for __ in range(n)] for _ in range(n)]
#
# for i in range(n):
#     R[i][i] = torch.sqrt(V[i] * V[i] @ int_w)
#     Q[i] = V[i] / R[i][i]
#     for j in range(i+1, n):
#         R[i][j] = V[j] * Q[i] @ int_w
#         if abs(R[i][j]) >= 1e-1:
#             V[j] = V[j] - (Q[i] * R[i][j])
# #  End Gram-Schmidt
#
# O_BASIS = torch.stack(Q)
# for i, b in enumerate(O_BASIS):
#     plt.figure(figsize=(3,2))
#     plt.imshow(b.view(M, M))
#     # plt.savefig(f'figures/kernel_basis_orth_{i}.png', bbox_inches='tight')
#     plt.show()
#
# O_EIGV = compute_EIGV(O_BASIS)
#

# In[ ]:




