import numpy as np


def rk3(x0: np.array, h: float, t_min: float, t_max: float, f: callable):
    xn = x0
    tn = t_min
    steps = [x0]
    while tn < t_max:
        xi1 = xn
        xi2 = xn + (2/3) * h * f(tn + (2/3)*h, xi1)
        xi3 = xn + (2/3) * h * f(tn + (2/3)*h, xi2)

        xn = xn + h * ((1/4)*f(tn,xi1) + (3/8)*f(tn + (2/3)*h, xi2) + (3/8)*f(tn + (2/3)*h, xi3))
        tn = tn + h
        steps.append(xn)
    return np.array(steps)


def shoot(u0, u1, a0, tmin, tmax, dt, F: callable, DF: callable, max_iter):
    a_next = a0
    n_iter = 0
    while n_iter <= max_iter:
        a_prev = a_next

        #  Solve W
        W0 = np.array([u0, a_prev])
        def wF(t, W): return np.array([W[1], F(t, W[0])])
        W = rk3(W0, dt, tmin, tmax, wF)

        if abs(W[-1, 0] - u1) <= 1e-8:
            return W, a_next

        #  Solve V
        def DFW(t): return DF(0, W[int(t * W.shape[0]), 0])
        def vF(t, V, W=W): return np.array([V[1], DFW(t) * V[0]])

        V0 = np.array([0, 1])
        V = rk3(V0, dt, tmin, tmax, vF)

        #  Update a
        a_next = a_prev - (W[-1, 0]-u1)/V[-1, 0]

        n_iter += 1

    if n_iter >= max_iter:
        raise Exception('Failed to converged.')
