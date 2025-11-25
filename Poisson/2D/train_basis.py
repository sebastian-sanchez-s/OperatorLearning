import signal
import json
import argparse
from pathlib import Path

import torch
from utils import init_sequential

import quad
from scipy.interpolate import RegularGridInterpolator as Interp2D
from encoding import pointwise, fourier_coeff


def __on_interrupt__():
    global train_error_fh, valid_error_fh
    print('Interrupted.')
    save()
    train_error_fh.close()
    valid_error_fh.close()
    quit()


signal.signal(signal.SIGINT, __on_interrupt__)


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int)
parser.add_argument('-n', '--name', type=str)

args = parser.parse_args()

ROOT = Path('model')
ARCH = Path('arch')

with open(ARCH / (args.name + '.json'), 'r') as cfile:
    params = json.load(cfile)

DATAPATH = Path('dataset') / params['ds']

PATH = ROOT / params["name"]
PATH.mkdir(exist_ok=True)


#  Dataset
def load_ds(name):
    '''  name = "train" or "valid"  '''
    global int_w, int_p
    with torch.no_grad():
        ds = PATH / (name + '.ds')
        if ds.exists():
            return torch.load(ds)
        else:
            from math import sqrt

            path = DATAPATH / name
            n_data = len(list(path.glob('*_X.pt')))

            match params['encoder']:
                case 'pointwise':
                    encdim = params['layers-apprx'][0]
                    iq = torch.linspace(0, 1, int(sqrt(encdim)))
                    iq1, iq2 = torch.meshgrid(iq, iq, indexing='xy')
                    q = torch.stack((iq1, iq2)).reshape(2, -1)
                    def encoder(fun): return pointwise(fun, q)
                case 'fourier_coeff':
                    n, m = params['latdim']
                    encdim = n*m
                    def encoder(fun): return fourier_coeff(fun, n, m, int_p, int_w)

            x = torch.zeros(n_data, encdim)
            y = torch.zeros(n_data, int_p.shape[1])

            for i in range(n_data):
                print('Loading', name, 'dataset', f'{i}/{n_data}', end='\r')
                X = torch.load((DATAPATH / name / f'{i}_X.pt')).float()
                Y = torch.load((DATAPATH / name / f'{i}_Y.pt')).float()

                ip = torch.linspace(0, 1, len(Y)).numpy()
                yfun = Interp2D((ip, ip), Y.numpy())
                xfun = Interp2D((ip, ip), X.numpy())

                xenc = pointwise(xfun, int_p.numpy())
                yenc = encoder(yfun)

                x[i, :] = yenc
                y[i, :] = xenc

            torch.save((x, y), ds)
            return (x, y)


def batch(ds, bs):
    x, y = ds
    for i in range(0, len(x), bs):
        yield x[i:i+bs], y[i:i+bs]


def save():
    global curr_epoch
    global A, A_opt, T, T_opt
    torch.save({
        'A': (A.state_dict(), A_opt.state_dict()),
        'T': (T.state_dict(), T_opt.state_dict()),
        'epoch': curr_epoch
    }, PATH / f'{curr_epoch}.model')


#  Model and Loss
def loss(y, ypred):
    global int_w
    num = (y - ypred)**2 @ int_w
    den = y**2 @ int_w
    return (num/den).mean()


def model(x, p):
    ''' x.shape = (batch_size, enc_dim),  p.shape = (num)
    -> (batch_size, latdim)
    '''
    #  A -> (batch_size, lat_dim),  T -> (num, lat_dim)
    a = A(x)
    b = T(p.T)
    return a@b.T


#  Training
def load_train_state():
    global curr_epoch
    global A, A_opt, T, T_opt
    n_epoch = 0
    for a in PATH.glob('*.model'):
        n_epoch = max(n_epoch, int(a.name[:-6]))
    curr_epoch = n_epoch
    if n_epoch > 0:
        data = torch.load(PATH / f'{curr_epoch}.model')
        A.load_state_dict(data['A'][0])
        A_opt.load_state_dict(data['A'][1])
        T.load_state_dict(data['T'][0])
        T_opt.load_state_dict(data['T'][1])


def train(epoch):
    global curr_epoch
    global train_ds, train_bs, train_ds_size, valid_ds, valid_bs, valid_ds_size
    global A, A_opt, T, T_opt
    global train_error_fh, valid_error_fh
    global int_p, int_w

    load_train_state()

    for n_epoch in range(epoch):
        A.train()
        T.train()

        train_error_epoch = 0
        for (x, y) in batch(train_ds, train_bs):
            ypred = model(x, int_p)
            error = loss(y, ypred)
            error.backward()

            A_opt.step()
            T_opt.step()

            if params['trunk-optimizer-scheduler']:
                T_scheduler.step()

            A_opt.zero_grad()
            T_opt.zero_grad()

            train_error_epoch += (error.item() * len(x))

        curr_epoch += 1

        A.eval()
        T.eval()
        with torch.no_grad():
            valid_error_epoch = 0
            for (x, y) in batch(valid_ds, valid_bs):
                ypred = model(x, int_p)
                error = loss(y, ypred).item()
                valid_error_epoch += error * len(x)

            mean_train_error = train_error_epoch / train_ds_size
            mean_valid_error = valid_error_epoch / valid_ds_size

            s = f'epoch: {curr_epoch:6d} ' \
                f'train error: {mean_train_error: .9f} ' \
                f'valid error: {mean_valid_error: .9f}'

            print(s, end='\n' if curr_epoch % params['rf'] == 0 else '\r')

            if curr_epoch % params['bf'] == 0:
                save()

            train_error_fh.write(f'{curr_epoch},{mean_train_error}\n')
            valid_error_fh.write(f'{curr_epoch},{mean_valid_error}\n')


match params['quad-name']:
    case 'gaussian':
        int_p, int_w, _, _ = quad.gaussian_quadrature(params['quad-size'])
    case _:
        int_p, int_w, _, _ = quad.trapezoidal_rule(params['quad-size'])


train_ds = load_ds('train')
valid_ds = load_ds('valid')

train_ds_size = len(train_ds[0])
valid_ds_size = len(valid_ds[0])

train_bs = 2000
valid_bs = 500

train_error_fh = open(PATH / 'train.error', 'a')
valid_error_fh = open(PATH / 'valid.error', 'a')

A = init_sequential(
        params['layers-apprx'],
        finish_with_activation=params['apprx-finish_with_activation'])
T = init_sequential(
        params['layers-trunk'],
        finish_with_activation=params['trunk-finish_with_activation'])

A_opt = torch.optim.Adam(A.parameters())
T_opt = torch.optim.Adam(T.parameters(), lr=1e-4)

match params['trunk-optimizer-scheduler']:
    case 'OneCycleLR':
        T_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                T_opt,
                max_lr=0.01,
                steps_per_epoch=train_ds_size//train_bs,
                epochs=args.epochs)
    case _:
        T_scheduler = None

curr_epoch = 0

save()
train(args.epochs)
save()

train_error_fh.close()
valid_error_fh.close()
