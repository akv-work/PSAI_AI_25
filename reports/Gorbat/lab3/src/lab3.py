import os
import sys

import matplotlib.pyplot as plt
import numpy as np

C0, C1 = -9.0, -8.0
X = np.array([[-9, -9], [-9, -8], [-8, -9], [-8, -8]], float)
Y = np.array([-9, -8, -8, -9], float)
T = (Y - C0) / (C1 - C0)
EE, MAX_EP, INIT = 0.20, 10_000, 0.5
LR = {"sig": 0.5, "relu": 0.05}
SEEDS, FIG = [0, 1, 2, 3, 4], os.path.join(os.path.dirname(__file__), "figures")

sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -500, 500)))
norm = lambda x: (np.asarray(x, float) - C0) / (C1 - C0)
scale = lambda y: C0 + float(y) * (C1 - C0)
cls = lambda v: C1 if abs(v - C1) < abs(v - C0) else C0


def bce(preds):
    p = np.clip(preds, 1e-7, 1 - 1e-7)
    return float(-np.sum(T * np.log(p) + (1 - T) * np.log(1 - p)))


class MLP:
    def __init__(self, seed, hidden):
        rng = np.random.RandomState(seed)
        self.hidden = hidden
        self.W1, self.b1 = rng.randn(2, 2) * INIT, np.zeros(2)
        self.W2, self.b2 = rng.randn(2, 1) * INIT, np.zeros(1)

    def fwd(self, x):
        x = norm(x).ravel()
        z = x @ self.W1 + self.b1
        self.h = np.maximum(z, 0) if self.hidden == "relu" else sig(z)
        self.o = float(sig(self.h @ self.W2 + self.b2).item())
        return self.o

    def step(self, x, t, lr):
        y = self.fwd(x)
        x = norm(x).ravel()
        d = y - t  # BCE + сигмоида на выходе
        self.W2 -= lr * self.h.reshape(-1, 1) * d
        self.b2 -= lr * d
        # ReLU': 1 при h>0, иначе 0; сигмоида': h(1-h)
        dh = (self.W2.ravel() * d) * ((self.h > 0) if self.hidden == "relu" else self.h * (1 - self.h))
        self.W1 -= lr * np.outer(x, dh)
        self.b1 -= lr * dh


def train(seed, hidden):
    net, hist, lr = MLP(seed, hidden), [], LR[hidden]
    for ep in range(1, MAX_EP + 1):
        for x, t in zip(X, T):
            net.step(x, t, lr)
        err = bce(np.array([net.fwd(x) for x in X]))
        hist.append(err)
        if err <= EE:
            return net, ep, err, True, hist
    return net, MAX_EP, err, False, hist


def metrics(net):
    preds, hs = [], []
    for x in X:
        preds.append(net.fwd(x))
        hs.append(net.h.copy())
    preds, hs = np.array(preds), np.vstack(hs)
    real = np.array([scale(p) for p in preds])
    acc = float(np.mean([cls(r) == y for r, y in zip(real, Y)]))
    mae = float(np.mean(np.abs(real - Y)))
    dead = int(np.sum(np.all(hs < 1e-8, axis=0)))
    return preds, real, acc, mae, dead


def run(hidden):
    rows = []
    for s in SEEDS:
        net, ep, err, ok, hist = train(s, hidden)
        _, real, acc, mae, dead = metrics(net)
        rows.append(dict(seed=s, net=net, ep=ep, err=err, ok=ok, hist=hist, acc=acc, mae=mae, dead=dead))
    return rows


def table(name, rows):
    print(f"\n=== {name} ===")
    print(f"{'seed':>5} {'эпохи':>7} {'сход.':>6} {'Es':>10} {'acc':>6} {'MAE':>8} {'мёртв.':>7}")
    ok_ep = []
    for r in rows:
        print(f"{r['seed']:5d} {r['ep']:7d} {'да' if r['ok'] else 'нет':>6} {r['err']:10.4f} "
              f"{r['acc']*100:5.0f}% {r['mae']:8.4f} {r['dead']:7d}")
        if r["ok"]:
            ok_ep.append(r["ep"])
    print(f"сходимость {sum(r['ok'] for r in rows)}/5", end="")
    if ok_ep:
        print(f", эпохи min/max/mean/std = {min(ok_ep)}/{max(ok_ep)}/{np.mean(ok_ep):.1f}/{np.std(ok_ep):.1f}")
    else:
        print()


def plots(a, b):
    os.makedirs(FIG, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    ra, rb = next(r for r in a if r["ok"]), next((r for r in b if r["ok"]), b[0])
    ax[0, 0].plot(ra["hist"], label="А: сигмоида")
    ax[0, 0].plot(rb["hist"], "--", label="В: ReLU")
    ax[0, 0].axhline(EE, ls=":", color="k", label=f"Ee={EE}")
    ax[0, 0].set(xlabel="эпоха", ylabel="Es", title=f"Сходимость (seed А={ra['seed']}, В={rb['seed']})", yscale="log")
    ax[0, 0].legend()
    ax[0, 0].grid(True, alpha=0.3)

    i = np.arange(5)
    ax[0, 1].bar(i - 0.2, [r["ep"] for r in a], 0.4, label="А")
    ax[0, 1].bar(i + 0.2, [r["ep"] for r in b], 0.4, label="В")
    for bars, rows in ((ax[0, 1].patches[:5], a), (ax[0, 1].patches[5:], b)):
        for bar, r in zip(bars, rows):
            if not r["ok"]:
                bar.set_hatch("///")
    ax[0, 1].set(xticks=i, xticklabels=SEEDS, xlabel="seed", ylabel="эпохи", title="Разброс эпох")
    ax[0, 1].legend()

    g = np.linspace(-10, 10, 160)
    aa, bb = np.meshgrid(g, g)
    for axi, rows, title in ((ax[1, 0], a, "А: сигмоида"), (ax[1, 1], b, "В: ReLU")):
        net = next((r["net"] for r in rows if r["ok"]), rows[0]["net"])
        zz = np.vectorize(lambda u, v: scale(net.fwd([u, v])))(aa, bb)
        im = axi.contourf(aa, bb, zz, 20, cmap="RdYlBu", vmin=C0, vmax=C1)
        fig.colorbar(im, ax=axi)
        for x, y in zip(X, Y):
            axi.scatter(*x, c="C0" if y == C0 else "C3", edgecolors="k", s=60)
        axi.set(title=title, xlabel="A", ylabel="B", xlim=(-10, 10), ylim=(-10, 10), aspect="equal")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "lab3.png"), dpi=140)
    plt.close()


def demo(net, name):
    extra = [[-10, -10], [0, 0], [-8.5, -9], [10, 10]]
    print(f"\n--- {name} ---")
    print(f"{'A':>7} {'B':>7} {'y_hat':>8} {'y_real':>9} {'класс':>6}")
    for x in np.vstack([X, extra]):
        y = net.fwd(x)
        print(f"{x[0]:7.1f} {x[1]:7.1f} {y:8.4f} {scale(y):9.4f} {cls(scale(y)):6.0f}")


def main():
    print(f"c0=-9 c1=-8  BCE  Ee={EE}  η_sig={LR['sig']}  η_relu={LR['relu']}")
    a, b = run("sig"), run("relu")
    table("А: сигмоида на всех слоях", a)
    table("В: ReLU скрытый, сигмоида выход", b)
    plots(a, b)
    print(f"график: {FIG}/lab3.png")
    sa, sb = sum(r["ok"] for r in a), sum(r["ok"] for r in b)
    print("\nкратко: сигмоида устойчивее "
          f"({sa}/5 vs {sb}/5). ReLU на удачном seed решает XOR, "
          "но часто умирает скрытый нейрон (h=0) — остаётся 75%. "
          "Многослойность нужна обеим; активация меняет оптимизацию и форму границы, не саму задачу.")
    demo(next(r["net"] for r in a if r["ok"]), "А сигмоида")
    relu_net = next((r["net"] for r in b if r["ok"]), b[0]["net"])
    demo(relu_net, "В ReLU")
    if sys.stdin.isatty():
        net = relu_net
        while True:
            s = input("A B (q): ").strip()
            if s.lower() in ("q", "quit", "exit"):
                break
            try:
                u, v = map(float, s.split())
                if not (-10 <= u <= 10 and -10 <= v <= 10):
                    raise ValueError
                y = net.fwd([u, v])
                print(f"y_hat={y:.4f}  y_real={scale(y):.4f}  класс={cls(scale(y)):g}")
            except ValueError:
                print("два числа из [-10, 10]")


if __name__ == "__main__":
    main()
