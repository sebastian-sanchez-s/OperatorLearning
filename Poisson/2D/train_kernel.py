import torch
from utils import init_sequential
import json
import signal
from pathlib import Path
import argparse
import quad
from scipy.interpolate import RegularGridInterpolator as Interp2D


def __on_interrupt__(signum, frame):
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
PATH.mkdir(exist_ok=True)


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

            x = torch.zeros(n_data, int_p.shape[1])
            y = torch.zeros(n_data, int_p.shape[1])
            int_p_np = int_p.numpy()
            for i in range(n_data):
                print('Loading', name, 'dataset', f'{i}/{n_data}', end='\r')
                X = torch.load((DATAPATH / name / f'{i}_X.pt')).float().numpy()
                Y = torch.load((DATAPATH / name / f'{i}_Y.pt')).float().numpy()

                ip = torch.linspace(0, 1, len(Y)).numpy()

                x[i, :] = torch.tensor(Interp2D((ip, ip), Y)(int_p_np.T)).float()
                y[i, :] = torch.tensor(Interp2D((ip, ip), X)(int_p_np.T)).float()

            print()
            torch.save((x, y), ds)
            return (x, y)


def batch(ds, bs):
    x, y = ds
    for i in range(0, len(x), bs):
        yield x[i:i+bs], y[i:i+bs]


def save():
    global curr_epoch, K
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


def model(x, t):
    ''' Computes x(t1,t2) = int int k((t1,t2), (s1,s2)) x(s1,s2) ds1 ds2
    x is already evaluated in the integration points s1s2,
    p are the points (t1,t2)
    '''
    global int_p, int_w
    from math import sqrt
    n = int(sqrt(t.shape[1]))
    m = int(sqrt(int_p.shape[1]))
    Kinp = torch.meshgrid((t[0, :n], t[0, :n], int_p[0, :m], int_p[0, :m]), indexing='xy')
    Kinp = torch.stack(Kinp).reshape(4, -1)
    Kout = K(Kinp.T).T
    Kout = Kout.reshape((n**2, m**2))
    return (x * int_w) @ Kout.T
    #  x (batch_size, int_num),  p (2, num)
    #  -> (batch_size, num)
    # from math import sqrt
    # n = int(sqrt(p.shape[1]))
    # m = int(sqrt(int_p.shape[1]))

    # Kinp = torch.meshgrid((p[0, :n], p[1, :n], int_p[0, :m], int_p[0, :m]), indexing='xy')
    # Kinp = torch.stack(Kinp).reshape(4, -1)
    # Kout = K(Kinp.T).reshape((n**2, m**2)).flip(0)
    # return torch.einsum('xs,ts,s->xt', x, Kout, int_w)
    # result = torch.zeros(x.shape[0], p.shape[1])
    # for j, s1s2 in enumerate(int_p.T):
    #     s1s2_rep = s1s2.expand(p.shape[1], -1).T
    #     Kt1t2s1s2 = K(torch.cat((p, s1s2_rep)).T).squeeze()
    #
    #     for i, xi in enumerate(x):
    #         result[i, j] += (xi * Kt1t2s1s2) @ int_w
    #
    # return result
    #

#  Training
def load_train_state():
    global curr_epoch
    global K, K_opt
    n_epoch = 0
    for model in PATH.glob('*.model'):
        n_epoch = max(n_epoch, int(model.name[:-6]))
    curr_epoch = n_epoch
    if n_epoch > 0:
        data = torch.load(PATH / f'{curr_epoch}.model')
        K.load_state_dict(data['K'][0])
        K_opt.load_state_dict(data['K'][1])


def train(epoch):
    global curr_epoch
    global train_ds, train_bs, train_ds_size, valid_ds, valid_bs, valid_ds_size
    global train_error_fh, valid_error_fh
    global K, K_opt, K_scheduler

    load_train_state()

    for n_epoch in range(epoch):
        K.train()

        train_error_epoch = 0
        for (x, y) in batch(train_ds, train_bs):
            ypred = model(x, int_p)
            error = loss(y, ypred)
            error.backward()

            K_opt.step()

            if K_scheduler is not None:
                K_scheduler.step()

            K_opt.zero_grad()

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
        int_p, int_w, _, _ = quad.gaussian_quadrature(params['quad-size'])
    case 'trapezoidal':
        int_p, int_w, _, _ = quad.trapezoidal_rule(params['quad-size'])
    case _:
        int_p, int_w, _, _ = quad.trapezoidal_rule(params['quad-size'])

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
save()

train(args.epochs)

save()
train_error_fh.close()
valid_error_fh.close()
