'''
Generate pairs (x,y) such that x solves the diferential equation
    -Delta x = y  on (0,1)
           x = 0  over t=0, t=1

y is sampled from a gaussian proccess with radial basis kernel.

generate_dataset.py --nsamples NS --nmesh NM --dirname DIR --subdirname SDIR --length-scale L1 L2 ... Lk
'''

import torch
import numpy as np
import scipy.sparse as sps

from scipy.fft import dct, idct

from pathlib import Path
import argparse

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--nsamples')
parser.add_argument('--nmesh')
parser.add_argument('--length-scale', nargs='+')
parser.add_argument('--dirname')
parser.add_argument('--subdirname')  #  mean to be 'train', 'valid' or 'test'

args = parser.parse_args()

NSAMPLES = int(args.nsamples)
LENGTH_SCALE = [float(l) for l in args.length_scale]
N = int(args.nmesh)

ROOT_PATH = Path('dataset/' + args.dirname)
CURR_PATH = ROOT_PATH / args.subdirname

CURR_PATH.mkdir(exist_ok=True, parents=True)


if __name__ == '__main__':
    '''
        Solve 1d poisson using finite differences
    '''
    def generate_pde_data(start, end):
        h = 1/(N-1)
        mesh = np.linspace(0, 1, N)
        A1d = sps.diags([-np.ones(N-2), 2*np.ones(N-2), -np.ones(N-2)],
                        offsets=[-1, 0, 1],
                        shape=(N-2, N-2),
                        format='csc')
        A1d = A1d / (h**2)
        rng = np.random.default_rng()
        for i in range(start, end):
            index = i // (NSAMPLES // len(LENGTH_SCALE))
            white_noise = rng.normal(0, h**2, size=N)
            random_field = idct(np.exp(-mesh**2 / (2*LENGTH_SCALE[index]**2)) * dct(white_noise)).real

            random_field -= np.min(random_field)
            random_field /= np.max(random_field)
            y = 0.96*random_field + 0.02

            x = np.zeros(N)
            x[1:-1] = sps.linalg.spsolve(A1d, y[1:-1])

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
