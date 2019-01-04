import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def sigmoid(x):
    sig = 1. / (1. + np.exp(-x))
    return sig

def dsigmoid(x):
    return x * (1. - x)


def forward(X, state_prev, param):
    h_prev, c_prev = state_prev  # State is tuple (h, c)
    # both shape = (num_neu, 1) vector.
    z = np.row_stack((X, h_prev))
    # Construct Gates:
    # Forget Gate:
    f = sigmoid(param["Wf"] @ z + param["bf"])
    # Input Gate:
    i = sigmoid(param["Wi"] @ z + param["bi"])
    # Input Modulation Gate:
    c_bar = np.tanh(param["Wc"] @ z + param["bc"])
    # New Cell State:
    c = f * c_prev + i * c_bar
    # Output Gate:
    o = sigmoid(param["Wo"] @ z + param["bo"])
    # New hidden state.
    h = o * np.tanh(c)
    # Prediction, linear activation.
    v = param["Wv"] @ h + param["bv"]

    y = v  # Linear activation for regression.

    return y, (h, c), (z, f, i, c_bar, o, v)


def backward(target, p, d,
    dh_next, dc_next, c_prev, z, f, i, c_bar, c, o, h, v, y,
    ):
    dv = y - target

    d["Wv"] += dv.T @ h.T
    d["bv"] += dv.T

    dh = p["Wv"].T @ dv.T  # (neu, out) * (out, 1)
    dh += dh_next

    do = dh * np.tanh(c)  # w.r.t. post-activation o
    do = dsigmoid(o) * do  # w.r.t. pre-activation o

    d["Wo"] += do @ z.T  # shape = (neu, neu + in)
    d["bo"] += do  # shape = (neu, 1)

    dc = np.copy(dc_next)
    dc += dh * o * (1. - np.tanh(c) ** 2)  # Add current cell contribution

    dc_bar = dc * i  # w.r.t post-activation c_bar
    dc_bar = (1. - np.tanh(c) ** 2) * dc_bar  # w.r.t pre-activation c_bar
    d["Wc"] += dc_bar @ z.T  # shape (neu, 1) * (1, neu + in) = (neu, neu + in)
    d["bc"] += dc_bar

    di = dc * c_bar  # Post-activation
    di = dsigmoid(i) * di  # Pre-activiation
    d["Wi"] += di @ z.T
    d["bc"] += di

    df = dc * c_prev  # post activation.
    df = dsigmoid(i) * df  # w.r.t. pre act
    d["Wf"] += df @ z.T
    d["bf"] += df

    # dz shape = (neu + in, 1)
    # all W shape = (neu, neu + in)
    dz = (p["Wf"].T @ df
        + p["Wi"].T @ di
        + p["Wc"].T @ dc_bar
        + p["Wo"].T @ do)

    dh_prev = dz[:num_neu, :]  # contribution from h(t-1) to the gross loss via this time period.
    dc_prev = f * dc  # contribution from c(t-1) to the gross loss via this time period.

    return dh_prev, dc_prev


def clear_gradients(grads):
    for var in grads.keys():
        grads[var].fill(0.0)

def clip_gradient(grads):
    for (key, val) in grads.items():
        grads[key] = np.clip(val, -1000.0, 1000.0)


def forward_backward(inputs, targets, h_init, c_init, param, grads):
    # records.
    internals = ("x", "z", "f", "i", "c_bar", "c", "o", "h", "v", "y") 
    rec = {var: {} for var in internals}

    # Values at t - 1, take the initial value.
    rec["h"][-1] = np.copy(h_init)
    rec["c"][-1] = np.copy(c_init)

    loss = 0.0

    # ======== Forward Phase ========
    for t in range(len(inputs)):
        rec["y"][t],\
        (rec["h"][t], rec["c"][t]), \
        (rec["z"][t], rec["f"][t], rec["i"][t], rec["c_bar"][t], rec["o"][t], rec["v"][t]) \
        = forward(
            inputs[t],
            state_prev=(rec["h"][t-1], rec["c"][t-1]),
            param=param
        )

        # Half-MSE
        loss += 0.5 * (targets[t] - rec["y"][t]) ** 2

    loss /= len(inputs)
    clear_gradients(grads)

    # ======== Backward Phase ========

    # For period T, parameters' contributions
    # to gross loss via further cell and state are zero 
    # at the terminal time period. 
    dh_next = np.zeros_like(rec["h"][0])
    dc_next = np.zeros_like(rec["c"][0])

    for t in reversed(range(len(inputs))):
        dh_next, dc_next = \
        backward(
            target=targets[t],
            dh_next=dh_next,
            dc_next=dc_next,
            p=param,
            d=grads,
            c_prev=rec["c"][t-1],
            z=rec["z"][t],
            f=rec["f"][t],
            i=rec["i"][t],
            c_bar=rec["c_bar"][t],
            c=rec["c"][t],
            o=rec["o"][t],
            h=rec["h"][t],
            v=rec["v"][t],
            y=rec["y"][t]
        )
    clip_gradient(grads)

    return grads, loss, rec["y"], (rec["h"][-1], rec["c"][-1])


def grad_desc(p, d, lr):
    for key, grad in d.items():
        p[key] -= lr * grad
    return p, d


if __name__ == "__main__":
    # ==== Model Settings ====
    num_in = 1
    num_out = 1
    num_neu = 8

    # Initialize Parameters
    param = {
        # Forget Gate Parameters:
        "Wf": np.random.randn(num_neu, num_in + num_neu) / num_neu,
        "bf": np.zeros((num_neu, 1)),
        # Input Gate Parameters:
        "Wi": np.random.randn(num_neu, num_in + num_neu) / num_neu,
        "bi": np.zeros((num_neu, 1)),
        # Input Modulation Gate (New Cancadiate Cell Value):
        "Wc": np.random.randn(num_neu, num_in + num_neu) / num_neu,
        "bc": np.zeros((num_neu, 1)),
        # Output Gate Parameters:
        "Wo": np.random.randn(num_neu, num_in + num_neu) / num_neu,
        "bo": np.zeros((num_neu, 1)),
        # Prediction Parameters
        "Wv": np.random.randn(num_out, num_neu) / num_neu,
        "bv": np.zeros((num_out, 1))
    }

    grads = {
        k: np.zeros_like(v)
        for (k, v) in param.items()
    }

    sample_inputs = np.arange(0, 4*np.pi, 0.2)
    sample_targets = np.sin(sample_inputs)

    epochs = int(input("epochs: "))
    for e in range(epochs):
        grads, loss, preds, _ = forward_backward(
            inputs=sample_inputs,
            targets=sample_targets,
            h_init=np.zeros((num_neu, 1)),
            c_init=np.zeros((num_neu, 1)),
            param=param,
            grads=grads
        )
        param, grads = grad_desc(p=param, d=grads, lr=0.003)
        if e % 10 == 0:
            print(f"Epochs={e}, Loss={np.asscalar(loss)}")

    plt.close()
    plt.style.use("seaborn-dark")
    pred = [np.squeeze(x) for x in preds.values()]
    plt.plot(pred)
    plt.plot(sample_targets)
    plt.legend(["predicted", "actual"])
    plt.grid(True)
    plt.show()
