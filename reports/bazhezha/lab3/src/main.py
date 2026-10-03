import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

C0, C1 = 0.0, -6.0
NMAX, EE = 50000, 0.15
ETA = {"sig": 0.7, "relu": 0.1}
SEEDS, REP = [0, 1, 2, 3, 4], 1
OUT = Path(__file__).parent / "plots"
RAW = np.array([[0.0, 0], [0, -6], [-6, 0], [-6, -6]])
EPS = 1e-7


def sigmoid(S):
    S = np.clip(S, -500, 500)
    return 1.0 / (1.0 + np.exp(-S))


def relu(S):
    return np.maximum(0.0, S)


def nrm(x):
    return (x - C0) / (C1 - C0)


def den(y):
    return C0 + y * (C1 - C0)


def cls(y):
    return C0 if abs(y - C0) <= abs(y - C1) else C1


X = nrm(RAW)
T = nrm(np.array([0.0, -6, -6, 0]))


class MLP:
    def __init__(self, seed, hidden):
        r = np.random.default_rng(seed)
        self.hidden = hidden
        self.Wh = r.uniform(-0.1, 0.1, (2, 2))
        self.Th = r.uniform(-0.1, 0.1, 2)
        self.Wo = r.uniform(-0.1, 0.1, 2)
        self.To = float(r.uniform(-0.1, 0.1))

    def fwd(self, x):
        self.x = np.asarray(x, float)
        self.Sh = self.x @ self.Wh - self.Th
        self.h = sigmoid(self.Sh) if self.hidden == "sig" else relu(self.Sh)
        self.y = float(sigmoid(self.h @ self.Wo - self.To))
        return self.y

    def step(self, x, e):
        y = self.fwd(x)
        dS = y - e
        a = ETA[self.hidden]
        self.Wo -= a * dS * self.h
        self.To += a * dS
        gh = dS * self.Wo
        dFh = self.h * (1 - self.h) if self.hidden == "sig" else (self.Sh > 0).astype(float)
        self.Wh -= a * np.outer(self.x, gh * dFh)
        self.Th += a * gh * dFh
        self.Wh = np.clip(self.Wh, -10, 10)
        self.Th = np.clip(self.Th, -10, 10)
        self.Wo = np.clip(self.Wo, -10, 10)
        self.To = float(np.clip(self.To, -10, 10))


def ys(net):
    return np.array([net.fwd(x) for x in X])


def es(net):
    y = np.clip(ys(net), EPS, 1 - EPS)
    return float(-np.sum(T * np.log(y) + (1 - T) * np.log(1 - y)))


def acc(net):
    return sum(cls(den(net.fwd(x))) == cls(den(t)) for x, t in zip(X, T)) / len(T)


def mae(net):
    return float(np.mean(np.abs(den(ys(net)) - den(T))))


def dead(net):
    H = np.array([(net.fwd(x), net.h.copy())[1] for x in X])
    return [i for i, z in enumerate(np.max(H, 0) < EPS) if z]


def fit(net):
    hist = []
    for ep in range(1, NMAX + 1):
        for x, t in zip(X, T):
            net.step(x, t)
        s = es(net)
        hist.append(s)
        if s <= EE:
            return ep, s, True, hist
    return NMAX, es(net), False, hist


def pred(net, a, b):
    yn = net.fwd(np.array([nrm(a), nrm(b)]))
    yr = den(yn)
    return yn, yr, cls(yr)


def dump(name, net, ep, s, ok):
    print(f"\n{name}  ep={ep}  Es={s:.6f}  stop={'yes' if ok else 'no'}  "
          f"acc={acc(net):.0%}  MAE={mae(net):.4f}  dead={dead(net)}")
    for (a, b), t in zip(RAW, T):
        yn, yr, c = pred(net, a, b)
        print(f"  {a:.0f} {b:.0f}  y={yn:.4f}  y_real={yr:.4f}  t={den(t):.0f}  {c:.0f}")


def plots(ha, hb, ra, rb, na, nb):
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ha, label="A sigmoid")
    ax.plot(hb, label="B ReLU")
    ax.axhline(EE, ls="--", c="k", lw=1, label=f"Ee={EE}")
    ax.set(xlabel="эпоха", ylabel="Es (BCE)", title="Сходимость, seed=1", yscale="log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    xs, w = np.arange(len(SEEDS)), 0.35
    ca = ["C0" if o else "0.5" for *_, o in ra]
    cb = ["C1" if o else "0.5" for *_, o in rb]
    ax.bar(xs - w / 2, [e for e, *_ in ra], w, color=ca, label="A sigmoid",
           hatch=["/" if not o else None for *_, o in ra], edgecolor="k")
    ax.bar(xs + w / 2, [e for e, *_ in rb], w, color=cb, label="B ReLU",
           hatch=["/" if not o else None for *_, o in rb], edgecolor="k")
    ax.set(xlabel="seed", ylabel="эпохи", title="Эпохи до Es≤Ee (штрих = нет)",
           xticks=xs, xticklabels=SEEDS)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "epochs_seeds.png", dpi=120)
    plt.close(fig)

    g = np.linspace(-10, 10, 80)
    A, B = np.meshgrid(g, g)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, net, title in ((axes[0], na, "A sigmoid"), (axes[1], nb, "B ReLU")):
        Z = np.vectorize(lambda a, b: pred(net, a, b)[1])(A, B)
        im = ax.contourf(A, B, Z, 20, cmap="RdBu")
        fig.colorbar(im, ax=ax, fraction=0.046)
        for (a, b), t in zip(RAW, T):
            ax.scatter(a, b, c="k" if den(t) == C0 else "yellow", s=60, edgecolors="k", zorder=3)
        ax.set(xlabel="A", ylabel="B", title=title, xlim=(-10, 10), ylim=(-10, 10))
    fig.tight_layout()
    fig.savefig(OUT / "surface.png", dpi=120)
    plt.close(fig)


def main():
    print("вариант 1  c0=0  c1=-6  MLP 2-2-1  BCE  A=sigmoid  B=ReLU")
    ra, rb, ha, hb, na, nb = [], [], None, None, None, None
    for seed in SEEDS:
        a, b = MLP(seed, "sig"), MLP(seed, "relu")
        ea, sa, oa, ha_ = fit(a)
        eb, sb, ob, hb_ = fit(b)
        ra.append((ea, sa, oa))
        rb.append((eb, sb, ob))
        print(f"seed={seed}  A: ep={ea} Es={sa:.4f} stop={oa} acc={acc(a):.0%} MAE={mae(a):.3f}"
              f"  |  B: ep={eb} Es={sb:.4f} stop={ob} acc={acc(b):.0%} MAE={mae(b):.3f} dead={dead(b)}")
        if seed == REP:
            ha, hb, na, nb = ha_, hb_, a, b
            dump("A sigmoid", a, ea, sa, oa)
            dump("B ReLU", b, eb, sb, ob)
    plots(ha, hb, ra, rb, na, nb)
    print(f"\nустойчивость  A {sum(o for *_, o in ra)}/5  B {sum(o for *_, o in rb)}/5")
    for name, r in ("A sigmoid", ra), ("B ReLU", rb):
        ok = [e for e, s, o in r if o]
        print(f"  {name}  ok={len(ok)}/5" + (f"  mean_ep={np.mean(ok):.0f}" if ok else ""))
    extra = [(5, -5), (10, -10), (3, -2)]
    print("\nрежим (B ReLU):")
    for a, b in list(map(tuple, RAW)) + extra:
        yn, yr, c = pred(nb, a, b)
        print(f"  {a:5.1f} {b:5.1f}  y={yn:.4f}  y_real={yr:.4f}  class={c:.0f} ({'c0' if c==C0 else 'c1'})")
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
        print(f"y={yn:.4f}  y_real={yr:.4f}  class={c:.0f} ({'c0' if c==C0 else 'c1'})")


if __name__ == "__main__":
    main()
