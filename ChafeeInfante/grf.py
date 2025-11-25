import numpy as np

from scipy.fft import dct, idct


def generate_grf(d: int, lengths: list, N: list, k: callable, boundary: str = 'neumann', rng=np.random.default_rng()):
    ''' Generates a Gaussian Random Field over a rectangular region
        in R^d of sides `lengths` with `N` nodes using the kernel `k`

        Parameters:
        - d: dimension where domain is embedded
        - length: list with the lenghts in each dimension
        - N: list the number of nodes in each dimension i.e grid size
        - k: kernel for the covariance. Recieves as input the meshgrid
        in R^d.

        Algorithm base on: 1105.2737
    '''
    assert all(n % 2 == 0 for n in N), "All elements of N must be even."

    match boundary:
        case 'periodic':
            F = lambda x: np.fft.fftn(x)
            Finv = lambda x: np.fft.ifftn(x)
            IndexGen = lambda n,l: np.fft.fftfreq(n,d=l)
        case 'neumann':
            from scipy.fft import dctn, idctn
            F = lambda x: dctn(x)
            Finv = lambda x: idctn(x)
            IndexGen = lambda n,l: np.arange(n)/(n*l)
        case 'dirichlet':
            from scipy.fft import dstn, idstn
            F = lambda x: dstn(x)
            Finv = lambda x: idstn(x)
            IndexGen = lambda n,l: np.arange(n)/(n*l)

    coords = [IndexGen(n, l) for n, l in zip(N, lengths)]
    meshgrid = np.meshgrid(*coords, indexing='ij')

    W = rng.normal(loc=0, scale=1.0/max(N)**d, size=N)
    A = k(meshgrid) * F(W) / np.prod(lengths)

    return Finv(A).real


def RBF(*coords, l=1.0):
    squared_distance = np.sum([coord**2 for coord in coords], axis=0)
    return np.exp(-squared_distance/(2*l**2))
