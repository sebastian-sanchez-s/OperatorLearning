'''
Generate pairs (x,y) such that x solves the diferential equation
    -Delta x = y  on (0,1)^2
           x = 0  over the boundary

y is sampled from a gaussian proccess with radial basis kernel.

generate_dataset.py --nsamples NS --nmesh NM --dirname DIR --subdirname SDIR --length-scale L1 L2 ... Lk
'''

import torch
import numpy as np
import scipy.sparse as sps

from scipy.fft import dct, idct

from pathlib import Path
import argparse

from grf import generate_grf, RBF

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--nsamples', type=int)
parser.add_argument('--nmesh', type=int)
parser.add_argument('--length-scale', nargs='+')
parser.add_argument('--dirname', type=str)
parser.add_argument('--subdirname', type=str)  #  mean to be 'train', 'valid' or 'test'

args = parser.parse_args()

NSAMPLES = args.nsamples
LENGTH_SCALE = [float(l) for l in args.length_scale]
N = int(args.nmesh)

ROOT_PATH = Path('dataset/' + args.dirname)
CURR_PATH = ROOT_PATH / args.subdirname

CURR_PATH.mkdir(exist_ok=True, parents=True)


if __name__ == '__main__':
    '''
        Solve 2D poisson using finite differences with 9 point stencil
        on data sampled from a Gaussian Processes.
    '''
    def generate_pde_data(start, end):
        h = 1/(N-1)
        ones = np.ones(N-2)

        #  9 point stencil
        I = sps.diags(ones, shape=(N-2,N-2))
        Iside = sps.diags([ones, ones],
                         offsets=[-1,1],
                         shape=(N-2, N-2),
                         format='csc')
        Aux_diag = sps.diags([4*ones, -20*ones, 4*ones],
                        offsets=[-1, 0, 1],
                        shape=(N-2, N-2),
                        format='csc')
        Aux_side = sps.diags([ones, 4*ones, ones],
                        offsets=[-1, 0, 1],
                        shape=(N-2, N-2),
                        format='csc')

        A2D = sps.kron(Aux_diag, I) + sps.kron(Iside, Aux_side)

        #  Generate mesh
        dirichlet_nodes = []
        k = 0
        for j in range(N):
            for i in range(N):
                if not( 0 < i < N-1 and 0 < j < N-1 ):
                    dirichlet_nodes.append(k)
                k += 1

        int_nodes = np.setdiff1d(range(N*N), dirichlet_nodes)

        rng = np.random.default_rng(5**start + 11**end)
        if args.subdirname == 'train':
            rng = np.random.default_rng(3**start + 17**end)

        for i in range(start, end):
            #  Generate Gaussian Random Field
            index = i // (NSAMPLES // len(LENGTH_SCALE))
            y = generate_grf(2,
                             [1.0]*2,
                             [N]*2,
                             lambda meshgrid, l=LENGTH_SCALE[index]: RBF(*meshgrid, l=l),
                             boundary='neumann',
                             rng=rng).ravel()

            y -= np.min(y)
            y /= np.max(y)
            y = 0.96*y + 0.02

            #  Solve equation
            x = np.zeros(N*N)
            x[int_nodes] = -sps.linalg.spsolve(A2D, 6*h**2 * y[int_nodes])

            ''' Save data '''
            torch.save(torch.tensor(y).float(), CURR_PATH / f'{i}_Y.pt')
            torch.save(torch.tensor(x).float(), CURR_PATH / f'{i}_X.pt')

    def progress_bar(total, bar_length=25):
        current = 0
        while current < total:
            current = len(list(CURR_PATH.glob('*_X.pt')))
            fraction = current / total

            arrow = int(fraction * bar_length - 1) * '-' + '>'
            padding = int(bar_length - len(arrow)) * ' '

            ending = '\n' if current == total else '\r'

            print(f'Progress: [{arrow}{padding}] {int(fraction*100)}%', end=ending)

    '''
      Distribute the work
    '''
    start = 0
    if CURR_PATH.exists():
        if (CURR_PATH / '0_X.pt').exists():
            start = max([
              int(p.name[:-5]) for p in CURR_PATH.glob('*_X.pt')
            ])
    else:
        CURR_PATH.mkdir(parents=True, exist_ok=True)

    from multiprocessing import Pool, cpu_count

    num_workers = cpu_count()
    samples_per_worker = max(min(10, NSAMPLES // num_workers), 1)

    s = f'''\
    num samples   : {NSAMPLES}
    num workers   : {num_workers}
    load worker   : {samples_per_worker}
    length scale  : {LENGTH_SCALE}
    mesh points   : {N}\
    '''
    print(s)
    with open(CURR_PATH / 'readme_dataset.txt', 'w') as f:
        f.write(s)

    with Pool(processes=num_workers) as pool:
        pool.apply_async(progress_bar, (NSAMPLES,))

        workers = [
            pool.apply_async(generate_pde_data, (i, i+samples_per_worker))
            for i in range(start,
                           NSAMPLES+1-samples_per_worker,
                           samples_per_worker)
        ]

        [worker.get() for worker in workers]
