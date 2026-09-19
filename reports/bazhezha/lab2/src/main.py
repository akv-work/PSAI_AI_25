import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

C0, C1 = 0.0, -6.0
ALPHA, NMAX = 0.7, 50000
EE_MSE, EE_BCE = 0.01, 0.15
SEEDS = [0, 1, 2, 3, 4]
REP = 0
OUT = Path(__file__).parent / "plots"
RAW = np.array([[0, 0], [0, -6], [-6, 0], [-6, -6]], dtype=float)
EPS = 1e-7


def sigmoid(S):
    S = np.clip(S, -500, 500)
    return 1.0 / (1.0 + np.exp(-S))


def nrm(x):
    return (x - C0) / (C1 - C0)


def den(y):
    return C0 + y * (C1 - C0)


def cls(y):
    return C0 if abs(y - C0) <= abs(y - C1) else C1


X = nrm(RAW)
E = nrm(np.array([0, -6, -6, 0], dtype=float))


class MLP:
    def __init__(self, seed, loss):
        r = np.random.default_rng(seed)
        self.loss = loss
        self.Wh = r.uniform(-0.1, 0.1, (2, 2))
        self.Th = r.uniform(-0.1, 0.1, 2)
        self.Wo = r.uniform(-0.1, 0.1, 2)
        self.To = float(r.uniform(-0.1, 0.1))

    def fwd(self, x):
        self.x = np.asarray(x, float)
        self.h = sigmoid(self.x @ self.Wh - self.Th)
        self.y = float(sigmoid(self.h @ self.Wo - self.To))
        return self.y

    def step(self, x, e):
        y = self.fwd(x)
        dS = (y - e) * y * (1 - y) if self.loss == "mse" else (y - e)
        self.Wo -= ALPHA * dS * self.h
        self.To += ALPHA * dS
        gh = dS * self.Wo
        dFh = self.h * (1 - self.h)
        self.Wh -= ALPHA * np.outer(self.x, gh * dFh)
        self.Th += ALPHA * gh * dFh


def ys(net):
    return np.array([net.fwd(x) for x in X])


def es_mse(net):
    return 0.5 * np.sum((ys(net) - E) ** 2)


def es_bce(net):
    y = np.clip(ys(net), EPS, 1 - EPS)
    return float(-np.sum(E * np.log(y) + (1 - E) * np.log(1 - y)))


def acc(net):
    return sum(cls(den(net.fwd(x))) == cls(den(e)) for x, e in zip(X, E)) / len(E)


def mae(net):
    return float(np.mean(np.abs(den(ys(net)) - den(E))))


def fit(net):
    ee = EE_MSE if net.loss == "mse" else EE_BCE
    loss_fn = es_mse if net.loss == "mse" else es_bce
    hist = []
    for ep in range(1, NMAX + 1):
        for x, e in zip(X, E):
            net.step(x, e)
        s = loss_fn(net)
        hist.append(s)
        if s <= ee:
            return ep, s, True, hist
    return NMAX, loss_fn(net), False, hist


def pred(net, a, b):
    yn = net.fwd(np.array([nrm(a), nrm(b)]))
    yr = den(yn)
    return yn, yr, cls(yr)


def dump(name, net, ep, s, ok):
    print(f"\n{name}  ep={ep}  Es={s:.6f}  stop={'yes' if ok else 'no'}  "
          f"acc={acc(net):.0%}  MAE={mae(net):.4f}")
    for (a, b), e in zip(RAW, E):
        yn, yr, c = pred(net, a, b)
        print(f"  {a:.0f} {b:.0f}  y={yn:.4f}  y_real={yr:.4f}  t={den(e):.0f}  {c:.0f}")


def plots(ha, hb, ra, rb, na, nb):
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ha, label="A MSE", color="C0")
    ax.plot(hb, label="B BCE", color="C1")
    ax.axhline(EE_MSE, color="C0", ls="--", lw=1, label=f"Ee MSE={EE_MSE}")
    ax.axhline(EE_BCE, color="C1", ls="--", lw=1, label=f"Ee BCE={EE_BCE}")
    ax.set(xlabel="эпоха", ylabel="Es", title="Сходимость Es(эпоха), seed=0", yscale="log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(SEEDS))
    w = 0.35
    ca = ["C0" if o else "0.5" for _, _, o in ra]
    cb = ["C1" if o else "0.5" for _, _, o in rb]
    ax.bar(xs - w / 2, [e for e, _, _ in ra], w, color=ca, label="A MSE",
           hatch=["/" if not o else None for _, _, o in ra], edgecolor="k")
    ax.bar(xs + w / 2, [e for e, _, _ in rb], w, color=cb, label="B BCE",
           hatch=["/" if not o else None for _, _, o in rb], edgecolor="k")
    ax.set(xlabel="seed", ylabel="эпохи", title="Эпохи до Es≤Ee (штрих = нет сходимости)",
           xticks=xs, xticklabels=SEEDS)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "epochs_seeds.png", dpi=120)
    plt.close(fig)

    g = np.linspace(-10, 10, 80)
    A, B = np.meshgrid(g, g)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, net, title in ((axes[0], na, "A MSE  ŷ→[c0,c1]"), (axes[1], nb, "B BCE  ŷ→[c0,c1]")):
        Z = np.vectorize(lambda a, b: pred(net, a, b)[1])(A, B)
        im = ax.contourf(A, B, Z, 20, cmap="RdBu")
        fig.colorbar(im, ax=ax, fraction=0.046)
        for (a, b), e in zip(RAW, E):
            ax.scatter(a, b, c="k" if den(e) == C0 else "yellow", s=60, edgecolors="k", zorder=3)
        ax.set(xlabel="A", ylabel="B", title=title, xlim=(-10, 10), ylim=(-10, 10))
    axes[2].bar(["A MSE", "B BCE"], [mae(na), mae(nb)], color=["C0", "C1"])
    axes[2].set(ylabel="MAE [c0,c1]", title="Точность шкалы [c0,c1]")
    fig.tight_layout()
    fig.savefig(OUT / "surface_mae.png", dpi=120)
    plt.close(fig)


def main():
    print("c0=0  c1=-6  MLP 2-2-1  online")
    ra, rb = [], []
    ha = hb = None
    na = nb = None
    for seed in SEEDS:
        a, b = MLP(seed, "mse"), MLP(seed, "bce")
        ea, sa, oa, ha_ = fit(a)
        eb, sb, ob, hb_ = fit(b)
        ra.append((ea, sa, oa))
        rb.append((eb, sb, ob))
        print(f"seed={seed}  A: ep={ea} Es={sa:.4f} stop={oa} acc={acc(a):.0%} MAE={mae(a):.3f}"
              f"  |  B: ep={eb} Es={sb:.4f} stop={ob} acc={acc(b):.0%} MAE={mae(b):.3f}")
        if seed == REP:
            ha, hb, na, nb = ha_, hb_, a, b
            dump("A MSE", a, ea, sa, oa)
            dump("B BCE", b, eb, sb, ob)

    plots(ha, hb, ra, rb, na, nb)
    n_ok_a = sum(o for _, _, o in ra)
    n_ok_b = sum(o for _, _, o in rb)
    print(f"\nустойчивость  A {n_ok_a}/5  B {n_ok_b}/5")
    print(f"графики: {OUT}")

    extra = [(5, -5), (10, -10), (3, -2)]
    print("\nрежим (B BCE): таблица + доп. пары")
    for a, b in list(map(tuple, RAW)) + extra:
        yn, yr, c = pred(nb, a, b)
        tag = "c0" if c == C0 else "c1"
        print(f"  {a:5.1f} {b:5.1f}  y={yn:.4f}  y_real={yr:.4f}  class={c:.0f} ({tag})")

    print("A B из [-10,10] (пусто=выход)")
    while True:
        s = input("> ").strip()
        if not s:
            break
        try:
            a, b = map(float, s.split())
        except ValueError:
            print("нужно: A B")
            continue
        if not (-10 <= a <= 10 and -10 <= b <= 10):
            print("вне диапазона")
            continue
        yn, yr, c = pred(nb, a, b)
        tag = "c0" if c == C0 else "c1"
        print(f"y={yn:.4f}  y_real={yr:.4f}  class={c:.0f} ({tag})")


if __name__ == "__main__":
    main()