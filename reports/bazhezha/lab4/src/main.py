import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

A, B, C, D, P, N = 1.0, 0.30, 0.5, 0.70, 6, 4
K_ALL, K_TR, NMAX, EE, ETA = 200, 140, 20000, 1e-4, 0.15
SEEDS = [0, 1, 2, 3, 4]
OUT = Path(__file__).parent / "plots"
EPS = 1e-12


def sigmoid(s):
    s = np.clip(s, -500, 500)
    return 1.0 / (1.0 + np.exp(-s))


def series(n=K_ALL):
    k = np.arange(n, dtype=float)
    return A * np.cos(B * k) + C * np.sin(D * k)


def split_norm(x):
    tr, te = x[:K_TR], x[K_TR:]
    lo, hi = float(tr.min()), float(tr.max())
    nrm = lambda v: (v - lo) / (hi - lo)
    den = lambda v: lo + np.asarray(v) * (hi - lo)
    return tr, te, nrm(x), nrm(tr), nrm(te), lo, hi, den


def windows(x, p):
    X = np.stack([x[i:i + p] for i in range(len(x) - p)])
    return X, x[p:]


def one_step_xy(xn, p, k0, k1):
    X = np.stack([xn[k - p:k] for k in range(k0, k1)])
    return X, xn[k0:k1]


class MLP:
    def __init__(self, p, n, seed):
        r = np.random.default_rng(seed)
        self.Wh, self.Th = r.uniform(-0.2, 0.2, (p, n)), r.uniform(-0.2, 0.2, n)
        self.Wo, self.To = r.uniform(-0.2, 0.2, n), float(r.uniform(-0.2, 0.2))

    def fwd(self, x):
        self.x = np.asarray(x, float)
        self.h = sigmoid(self.x @ self.Wh - self.Th)
        self.y = float(self.h @ self.Wo - self.To)  # линейный выход
        return self.y

    def step(self, x, t):
        y = self.fwd(x)
        d = y - t
        self.Wo -= ETA * d * self.h
        self.To += ETA * d
        gh = d * self.Wo * self.h * (1 - self.h)
        self.Wh -= ETA * np.outer(self.x, gh)
        self.Th += ETA * gh


def es_mean(net, X, T):
    return 0.5 * float(np.mean((np.array([net.fwd(x) for x in X]) - T) ** 2))


def fit(net, X, T, seed, hist=False):
    r = np.random.default_rng(seed)
    h = []
    idx = np.arange(len(X))
    for ep in range(1, NMAX + 1):
        r.shuffle(idx)
        for i in idx:
            net.step(X[i], T[i])
        s = es_mean(net, X, T)
        if hist:
            h.append(s)
        if s <= EE:
            return ep, s, True, h
    return NMAX, es_mean(net, X, T), False, h


def predict(net, X):
    return np.array([net.fwd(x) for x in X])


def mae(y, t):
    return float(np.mean(np.abs(y - t)))


def rmse(y, t):
    return float(np.sqrt(np.mean((y - t) ** 2)))


def smape(y, t):
    return float(100 * np.mean(2 * np.abs(y - t) / (np.abs(y) + np.abs(t) + EPS)))


def metrics(yh, yt, tag):
    print(f"  {tag}  MAE={mae(yh, yt):.4f}  RMSE={rmse(yh, yt):.4f}  sMAPE={smape(yh, yt):.2f}%")


def recursive(net, xn, p, k0, steps):
    w = list(xn[k0 - p:k0])
    out = []
    for _ in range(steps):
        y = net.fwd(np.array(w))
        out.append(y)
        w = w[1:] + [y]
    return np.array(out)


def run_one(p, n, seed, xn, hist=False):
    Xtr, Ttr = windows(xn[:K_TR], p)
    net = MLP(p, n, seed)
    ep, s, ok, h = fit(net, Xtr, Ttr, seed, hist)
    Xte, Tte = one_step_xy(xn, p, K_TR, K_ALL)
    return net, ep, s, ok, h, Xtr, Ttr, Xte, Tte


def study(ps, ns, xn, den):
    rows = []
    for p in ps:
        for n in ns:
            eps, maes, fails = [], [], 0
            for seed in SEEDS:
                net, ep, s, ok, _, _, _, Xte, Tte = run_one(p, n, seed, xn)
                yt, yh = den(Tte), den(predict(net, Xte))
                m = mae(yh, yt)
                eps.append(ep)
                maes.append(m)
                fails += int(not ok)
                print(f"  p={p} N={n} seed={seed}  ep={ep} Es={s:.2e} stop={ok} MAE={m:.4f}")
            rows.append((p, n, np.mean(eps), np.std(eps), np.mean(maes), np.std(maes), fails))
    return rows


def plots(x, den, net, p, Xtr, Ttr, Xte, Tte, h_base, h_p2, h_p12, rec, true30):
    OUT.mkdir(exist_ok=True)
    k = np.arange(K_ALL)
    yhat = np.full(K_ALL, np.nan)
    yhat[p:K_TR] = den(predict(net, Xtr))
    yhat[K_TR:] = den(predict(net, Xte))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(k, x, label="ряд")
    ax.plot(k, yhat, label="1-шаг", lw=1)
    ax.axvline(K_TR - 0.5, c="k", ls="--", label="train/test")
    ax.set(xlabel="k", ylabel="x(k)", title="Ряд и одношаговый прогноз")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "series.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h_base, label=f"p={P} N={N}")
    ax.plot(h_p2, label="p=2")
    ax.plot(h_p12, label=f"p={2 * P}")
    ax.axhline(EE, c="k", ls="--", lw=1)
    ax.set(xlabel="эпоха", ylabel="Es", title="Сходимость", yscale="log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=120)
    plt.close(fig)

    yt, yh = den(Tte), den(predict(net, Xte))
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(yt, yh, s=12)
    lim = [min(yt.min(), yh.min()), max(yt.max(), yh.max())]
    ax.plot(lim, lim, "k--", label="y=x")
    ax.set(xlabel="эталон", ylabel="прогноз", title="Тест: эталон–прогноз")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "scatter.png", dpi=120)
    plt.close(fig)

    err = np.abs(den(rec) - den(true30))
    rng = float(x.max() - x.min())
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(den(true30), label="эталон")
    ax[0].plot(den(rec), label="рекурс.")
    ax[0].set(xlabel="шаг", title="Многошаговый прогноз")
    ax[0].legend()
    ax[1].plot(err)
    ax[1].axhline(0.05 * rng, c="r", ls="--", label="5% размаха")
    ax[1].set(xlabel="шаг", title="|ошибка|")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "recursive.png", dpi=120)
    plt.close(fig)
    hit = np.where(err > 0.05 * rng)[0]
    print(f"размах={rng:.4f}  5%={0.05 * rng:.4f}  порог с шага {int(hit[0] + 1) if len(hit) else '>30'}")


def show(net, xn, x, den, p, k=None, raw=None, tag=""):
    if raw is not None:
        w = np.asarray(raw, float)
        tn = None
        t = None
    else:
        w = x[k - p:k]
        tn, t = xn[k], x[k]
    lo, hi = float(den(0)), float(den(1))
    wn = (np.asarray(w, float) - lo) / (hi - lo)
    yn = net.fwd(wn)
    yr = float(den(yn))
    print(f"{tag} in={np.round(wn, 4).tolist()}  y={yn:.4f}  y_real={yr:.4f}", end="")
    if t is not None:
        print(f"  true={t:.4f}  |e|={abs(yr - t):.4f}")
    else:
        print()


def main():
    x = series()
    tr, te, xn, xntr, xnte, lo, hi, den = split_norm(x)
    print(f"вариант 1  x=a cos(bk)+c sin(dk)  a={A} b={B} c={C} d={D}  p={P} N={N}")
    print(f"train k=0..{K_TR - 1} ({K_TR})  test k={K_TR}..{K_ALL - 1} ({K_ALL - K_TR})")
    print(f"min/max только train: lo={lo:.4f} hi={hi:.4f}")
    print(f"окон train={K_TR - P}  test={K_ALL - K_TR}  η={ETA}  Ee={EE}")
    print("пайплайн: ряд→split→minmax(train)→окно→MLP p-N-1→online+shuffle→Es≤Ee→denorm→MAE/RMSE/sMAPE/рекурс.")
    print("не: batch, сигмоида на выходе, minmax по всему ряду")

    print("\n=== база p=6 N=4 seed=0 ===")
    net, ep, s, ok, h0, Xtr, Ttr, Xte, Tte = run_one(P, N, 0, xn, hist=True)
    print(f"эпохи={ep}  Es={s:.2e}  stop={ok}")
    metrics(den(predict(net, Xtr)), den(Ttr), "train")
    metrics(den(predict(net, Xte)), den(Tte), "test")

    print("\n=== окно p, N=4 ===")
    rp = study([2, 4, P, 2 * P], [N], xn, den)
    print("p | mean_ep±std | mean_MAE±std | fail")
    for p, n, me, se, mm, sm, f in rp:
        print(f"  {p:2d}  {me:.0f}±{se:.0f}  {mm:.4f}±{sm:.4f}  {f}/5")

    print("\n=== скрытый слой N, p=6 ===")
    rn = study([P], [1, 2, N, 2 * N], xn, den)
    print("N | mean_ep±std | mean_MAE±std | fail")
    for p, n, me, se, mm, sm, f in rn:
        print(f"  {n:2d}  {me:.0f}±{se:.0f}  {mm:.4f}±{sm:.4f}  {f}/5")

    all_rows = rp + [r for r in rn if r[1] != N]
    best = min(all_rows, key=lambda r: r[4])
    bp, bn = int(best[0]), int(best[1])
    print(f"\nлучшая по MAE: p={bp} N={bn}  MAE={best[4]:.4f}")
    best_net, *_ = run_one(bp, bn, 0, xn)
    rec = recursive(best_net, xn, bp, K_TR, 30)
    true30 = xn[K_TR:K_TR + 30]

    _, _, _, _, h2, *_ = run_one(2, N, 0, xn, hist=True)
    _, _, _, _, h12, *_ = run_one(2 * P, N, 0, xn, hist=True)
    plots(x, den, net, P, Xtr, Ttr, Xte, Tte, h0, h2, h12, rec, true30)

    print("\n=== шум на train ===")
    rng = np.random.default_rng(0)
    xnz = x.copy()
    xnz[:K_TR] += rng.normal(0, 0.05 * np.std(tr), K_TR)
    trn = xnz[:K_TR]
    lo_n, hi_n = float(trn.min()), float(trn.max())
    xnn = (xnz - lo_n) / (hi_n - lo_n)
    den_n = lambda v, lo=lo_n, hi=hi_n: lo + np.asarray(v) * (hi - lo)
    nnet, epn, sn, okn, _, _, _, Xten, Tten = run_one(P, N, 0, xnn)
    metrics(den_n(predict(nnet, Xten)), den_n(Tten), f"test+noise ep={epn} stop={okn}")

    print("\nрежим:")
    for k in (20, 60, 100, 150, 170, 190):
        show(net, xn, x, den, P, k=k, tag=f"k={k}")
    w_ok = x[20:20 + P].copy()
    w_bad = w_ok.copy()
    w_bad[2] = 10.0
    show(net, xn, x, den, P, raw=w_ok, tag="user")
    show(net, xn, x, den, P, raw=w_bad, tag="выброс")
    print("ввод: k  ИЛИ  p чисел  (пусто=выход)")
    while True:
        s = input("> ").strip()
        if not s:
            break
        parts = s.split()
        try:
            if len(parts) == 1:
                k = int(parts[0])
                if not (P <= k < K_ALL):
                    print("k в [p, 199]")
                    continue
                show(net, xn, x, den, P, k=k, tag=f"k={k}")
            elif len(parts) == P:
                show(net, xn, x, den, P, raw=list(map(float, parts)), tag="user")
            else:
                print(f"нужно k или {P} чисел")
        except ValueError:
            print("ошибка ввода")


if __name__ == "__main__":
    main()
