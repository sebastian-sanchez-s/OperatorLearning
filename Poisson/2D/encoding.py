import torch


def pointwise(y: callable, p):
    ''' p (2, num) -> (num) '''
    return torch.tensor(y(p.T), dtype=torch.float32)


def fourier_coeff(y: callable, n: int, m: int, int_p, int_w):
    ''' int_p (2, num), int_w (num) -> (n*m) '''
    ret = torch.zeros(n*m).float()

    int_y = torch.tensor(y(int_p.T)).float()

    k = 0
    for i in range(1, n):
        for j in range(1, m):
            k1 = torch.tensor(i)
            k2 = torch.tensor(j)
            fun1 = torch.sin(torch.pi * k1 * int_p[0])
            fun2 = torch.sin(torch.pi * k2 * int_p[1])
            num = (fun1 * fun2 * int_y) @ int_w
            den = torch.sqrt((fun1 * fun2)**2 @ int_w)
            ret[k] = num / den
            k += 1
    return ret
