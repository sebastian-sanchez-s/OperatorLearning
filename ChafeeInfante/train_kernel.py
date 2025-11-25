import torch
import torch.nn as nn
from pau import PAU
import json
import signal
from pathlib import Path
import argparse
import quad
from utils import init_sequential
from scipy.interpolate import interp1d


def __on_interrupt__(signum, context):
    global train_error_fh, valid_error_fh
    print('Interrupted.')
    save()
    train_error_fh.close()
    valid_error_fh.close()
    quit()


signal.signal(signal.SIGINT, __on_interrupt__)

#  Setup
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int)
parser.add_argument('-n', '--name', type=str)

args = parser.parse_args()

ROOT = Path('model')
ARCH = Path('arch')

with open(ARCH / (args.name + '.json'), 'r') as cfile:
    params = json.load(cfile)

DATAPATH = Path('dataset') / params['ds']

PATH = ROOT / params['name']
PATH_TRAIN = PATH / 'train'
PATH_VALID = PATH / 'valid'

PATH.mkdir(exist_ok=True, parents=True)
PATH_TRAIN.mkdir(exist_ok=True, parents=True)
PATH_VALID.mkdir(exist_ok=True, parents=True)


#  Dataset
def load_ds(name):
    '''  name = "train" or "valid"  '''
    with torch.no_grad():
        ds = PATH / (name + '.ds')
        if ds.exists():
            return torch.load(ds)
        else:
            path = DATAPATH / name
            n_data = len(list(path.glob('*_X.pt')))

            x = torch.zeros(n_data, len(int_p))
            y = torch.zeros(n_data, len(int_p))
            int_p_np = int_p.numpy()
            for i in range(n_data):
                print('Loading', name, 'dataset', f'{i}/{n_data}', end='\r')
                X = torch.load((DATAPATH / name / f'{i}_X.pt')).float().numpy()
                Y = torch.load((DATAPATH / name / f'{i}_Y.pt')).float().numpy()

                ip = torch.linspace(0, 1, len(Y)).numpy()

                x[i, :] = torch.tensor(interp1d(ip, Y)(int_p_np)).float()
                y[i, :] = torch.tensor(interp1d(ip, X)(int_p_np)).float()

            torch.save((x, y), ds)
            return (x, y)


def batch(ds, bs):
    x, y = ds
    for i in range(0, len(x), bs):
        yield x[i:i+bs], y[i:i+bs]


def save():
    global curr_epoch, K
    if params.get('Psi'):
        torch.save({
            'K': (K.state_dict(), K_opt.state_dict()),
            'Psi': (Psi.state_dict(), Psi_opt.state_dict()),
            'epoch': curr_epoch
        }, PATH / f'{curr_epoch}.model')
    else:
        torch.save({
            'K': (K.state_dict(), K_opt.state_dict()),
            'epoch': curr_epoch
        }, PATH / f'{curr_epoch}.model')


#  Model and Loss
def loss(y, ypred):
    global int_w
    num = (y - ypred)**2 @ int_w
    den = y**2 @ int_w
    return (num/den).mean()


def model_linear(x, p):
    ''' x.shape = (batch_size, p),  p.shape = (num)
    -> (batch_size,)
    For training, p=int_p.

    Computes: x(t) = int K(t,s) x(s) ds
    '''
    #  A -> (batch_size, lat_dim),  T -> (num, lat_dim)
    t1, t2 = torch.meshgrid(p, int_p, indexing='xy')
    t1t2 = torch.stack((t1, t2)).reshape(2, -1)
    Kt1t2 = K(t1t2.T).squeeze().reshape((len(p), len(int_p))).flip(1)
    return torch.einsum('bj,ij,j->bi', x, Kt1t2, int_w)


def model_nonlinear(x, p):
    ''' x.shape = (batch_size, p),  p.shape = (num)
    -> (batch_size,)
    For training, p=int_p.

    Computes: x(t) = int K(t,s) Psi(x(s)) ds
    '''
    #  A -> (batch_size, lat_dim),  T -> (num, lat_dim)
    t1, t2 = torch.meshgrid(p, int_p, indexing='xy')
    t1t2 = torch.stack((t1, t2)).reshape(2, -1)
    Kt1t2 = K(t1t2.T).squeeze().reshape((len(p), len(int_p))).flip(1)
    Nonlinear = Psi(x.unsqueeze(-1)).squeeze()
    return torch.einsum('bj,ij,j->bi', Nonlinear, Kt1t2, int_w)


#  Training
def load_train_state():
    global curr_epoch
    global K, K_opt
    global Psi, Psi_opt
    n_epoch = 0
    for model in PATH.glob('*.model'):
        n_epoch = max(n_epoch, int(model.name[:-6]))
    curr_epoch = n_epoch
    if n_epoch > 0:
        data = torch.load(PATH / f'{curr_epoch}.model')
        K.load_state_dict(data['K'][0])
        K_opt.load_state_dict(data['K'][1])
        if params.get('Psi'):
            Psi.load_state_dict(data['Psi'][0])
            Psi_opt.load_state_dict(data['Psi'][1])


def train(epoch):
    global curr_epoch
    global model
    global train_ds, train_bs, train_ds_size, valid_ds, valid_bs, valid_ds_size
    global train_error_fh, valid_error_fh
    global K, K_opt, K_scheduler, Psi, Psi_opt

    load_train_state()

    for n_epoch in range(epoch):
        K.train()

        train_error_epoch = 0
        for (x, y) in batch(train_ds, train_bs):
            ypred = model(x, int_p)
            error = loss(y, ypred)
            error.backward()

            K_opt.step()
            K_opt.zero_grad()

            if K_scheduler is not None:
                K_scheduler.step()

            if params['Psi']:
                Psi_opt.step()
                Psi_opt.zero_grad()

            train_error_epoch += (error.item() * len(x))

        curr_epoch += 1

        K.eval()
        with torch.no_grad():
            valid_error_epoch = 0
            for (x, y) in batch(valid_ds, valid_bs):
                ypred = model(x, int_p)
                error = loss(y, ypred).item()
                valid_error_epoch += (error * len(x))

            mean_train_error = train_error_epoch / train_ds_size
            mean_valid_error = valid_error_epoch / valid_ds_size

            s = f'epoch: {curr_epoch:6d} ' \
                f'train error: {mean_train_error: 5.9f} ' \
                f'valid error: {mean_valid_error: 5.9f}'

            print(s, end='\n' if curr_epoch % params['rf'] == 0 else '\r')

            if curr_epoch % params['bf'] == 0:
                save()

            train_error_fh.write(f'{curr_epoch},{mean_train_error}\n')
            valid_error_fh.write(f'{curr_epoch},{mean_valid_error}\n')


match params['quad-name']:
    case 'gaussian':
        int_p, int_w = quad.gaussian_quadrature(0, 1, params['quad-size'])
    case 'trapezoidal':
        int_p, int_w = quad.trapezoidal_rule(0, 1, params['quad-size'])
    case _:
        int_p, int_w = quad.trapezoidal_rule(0, 1, params['quad-size'])

train_ds = load_ds('train')
valid_ds = load_ds('valid')

train_ds_size = len(train_ds[0])
valid_ds_size = len(valid_ds[0])

train_bs = 2000
valid_bs = 1000

train_error_fh = open(PATH / 'train.error', 'a')
valid_error_fh = open(PATH / 'valid.error', 'a')

K = init_sequential(
        params['layers'],
        finish_with_activation=params['finish_with_activation'])
K_opt = torch.optim.Adam(K.parameters(), lr=1e-4)

if params['Psi']:
    Psi = init_sequential(
            params['Psi-layers'],
            finish_with_activation=params['Psi-finish_with_activation'])
    Psi_opt = torch.optim.Adam(Psi.parameters(), lr=1e-4)

    model = model_nonlinear
else:
    model = model_linear

match params['optimizer-scheduler']:
    case 'OneCycleLR':
        K_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                K_opt,
                max_lr=0.01,
                steps_per_epoch=train_ds_size//train_bs,
                epochs=args.epochs)
    case _:
        K_scheduler = None

curr_epoch = 0

train(args.epochs)

save()
train_error_fh.close()
valid_error_fh.close()
