'''
Generate pairs (x,y) such that x solves the diferential equation
    -x'' + lambda(x^3 - x) = y  on (0,1)
           x = 0  over the boundary

y is sampled from a gaussian proccess with radial basis kernel.

generate_dataset.py --nsamples NS --N N --dirname DIR --subdirname SDIR --length-scale L1 L2 ... Lk
'''
import torch
import numpy as np

from scipy.interpolate import interp1d

from pathlib import Path
import argparse

from grf import generate_grf, RBF
from multiprocessing import Pool, cpu_count

from ode import shoot


# Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--nsamples', type=int)
parser.add_argument('--N', type=int)
parser.add_argument('--length-scale', nargs='+')
parser.add_argument('--dirname', type=str)
parser.add_argument('--subdirname', type=str)  #  mean to be 'train', 'valid' or 'test'

args = parser.parse_args()

NSAMPLES = args.nsamples
LENGTH_SCALE = [float(l) for l in args.length_scale]

ROOT_PATH = Path('dataset/' + args.dirname)
CURR_PATH = ROOT_PATH / args.subdirname

CURR_PATH.mkdir(exist_ok=True, parents=True)


def generate_data(start, end):
    match args.subdirname:
        case 'valid':
            rng = np.random.default_rng(5**start + 11**end)
        case  'train':
            rng = np.random.default_rng(3**start + 17**end)

    N = args.N
    lmbda = 9
    u0, u1 = 0.0, 0.0
    tmin, tmax = 0.0, 1.0
    a0 = 5.0
    dt = (tmax - tmin) / N
    max_iter = 100
    for i in range(start, end):
        index = i // (NSAMPLES // len(LENGTH_SCALE))
        y = generate_grf(1, [1.0], [N],
                         lambda meshgrid, l=LENGTH_SCALE[index]: RBF(*meshgrid, l=l),
                         boundary='neumann',
                         rng=rng)

        y -= np.min(y)
        y /= np.max(y)
        y = lmbda * (0.96*y + 0.02)

        Yfun = interp1d(np.linspace(0, 1, N), y)

        def F(t, U0): return -lmbda*(U0**3 - U0) + Yfun(t)
        def DF(t, U0): return -lmbda*(3*U0**2 - np.ones_like(U0))

        U, _ = shoot(u0, u1, a0, tmin, tmax, dt, F, DF, max_iter)

        ''' Save data '''
        torch.save(torch.tensor(y).float(), CURR_PATH / f'{i}_Y.pt')
        torch.save(torch.tensor(U[:N, 0]).float(), CURR_PATH / f'{i}_X.pt')


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


num_workers = cpu_count()
samples_per_worker = max(min(10, NSAMPLES // num_workers), 1)

s = f'''\
num samples   : {NSAMPLES}
num workers   : {num_workers}
load worker   : {samples_per_worker}
length scale  : {LENGTH_SCALE}
mesh points   : {args.N}\
'''
print(s)
with open(CURR_PATH / 'readme_dataset.txt', 'w') as f:
    f.write(s)

with Pool(processes=num_workers) as pool:
    p1 = pool.apply_async(progress_bar, (NSAMPLES,))

    workers = []
    for i in range(start, NSAMPLES+1-samples_per_worker, samples_per_worker):
        workers.append(pool.apply_async(generate_data, (i, i+samples_per_worker)))

    p1.get()
    for worker in workers:
        worker.get()
