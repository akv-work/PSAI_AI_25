import os
import sys

import matplotlib.pyplot as plt
import numpy as np

A, B, C, D = 1.5, 0.40, 0.8, 0.90
P_VAR, N_VAR = 6, 5
K_MAX, SPLIT = 199, 140
EE, MAX_EP, LR, INIT = 1e-4, 20_000, 0.08, 0.5
SEEDS = [0, 1, 2, 3, 4]
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def generate(n=K_MAX + 1):
    k = np.arange(n, dtype=float)
    return k, A * np.cos(B * k) + C * np.sin(D * k)


def minmax_fit(x_train):
    lo, hi = float(x_train.min()), float(x_train.max())
    return lo, hi


def nrm(x, lo, hi):
    return (np.asarray(x, float) - lo) / (hi - lo)


def den(y, lo, hi):
    return lo + np.asarray(y, float) * (hi - lo)


def windows(series, p, start, stop):
    """Окна, предсказывающие series[t] для t=start..stop-1; нужны series[t-p:t]."""
    X, T = [], []
    for t in range(start, stop):
        X.append(series[t - p : t])
        T.append(series[t])
    return np.array(X, float), np.array(T, float)


def mae_(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse_(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def smape_(y, p):
    return float(np.mean(2.0 * np.abs(y - p) / (np.abs(y) + np.abs(p) + 1e-12)) * 100.0)


class MLP:
    def __init__(self, p, n, seed):
        rng = np.random.RandomState(seed)
        self.W1, self.b1 = rng.randn(p, n) * INIT, np.zeros(n)
        self.W2, self.b2 = rng.randn(n) * INIT, 0.0

    def fwd(self, x):
        x = np.asarray(x, float).ravel()
        self.h = sig(x @ self.W1 + self.b1)
        self.y = float(self.h @ self.W2 + self.b2)  # линейный выход
        return self.y

    def step(self, x, t):
        y = self.fwd(x)
        d = y - t  # MSE, линейный выход: δ = (ŷ − t)
        self.W2 -= LR * self.h * d
        self.b2 -= LR * d
        dh = (self.W2 * d) * self.h * (1.0 - self.h)
        self.W1 -= LR * np.outer(x, dh)
        self.b1 -= LR * dh

    def predict(self, X):
        return np.array([self.fwd(x) for x in X])


def train(p, n, Xtr, Ttr, seed):
    net, rng, hist = MLP(p, n, seed), np.random.RandomState(seed + 17), []
    m = len(Ttr)
    for ep in range(1, MAX_EP + 1):
        for i in rng.permutation(m):
            net.step(Xtr[i], Ttr[i])
        es = float(np.mean((net.predict(Xtr) - Ttr) ** 2))
        hist.append(es)
        if es <= EE:
            return net, ep, es, True, hist
    return net, MAX_EP, es, False, hist


def metrics_orig(net, Xn, tn, lo, hi):
    pred = den(net.predict(Xn), lo, hi)
    y = den(tn, lo, hi)
    return pred, y, mae_(y, pred), rmse_(y, pred), smape_(y, pred)


def recursive(net, series, lo, hi, k0, steps, p):
    win = list(nrm(series[k0 - p : k0], lo, hi))
    preds = []
    for _ in range(steps):
        y = net.fwd(win)
        preds.append(den(y, lo, hi))
        win = win[1:] + [y]
    return np.array(preds)


def eval_config(p, n, xn, lo, hi, seeds=SEEDS):
    Xtr, Ttr = windows(xn, p, p, SPLIT)
    Xte, Tte = windows(xn, p, SPLIT, len(xn))
    rows = []
    for s in seeds:
        net, ep, es, ok, hist = train(p, n, Xtr, Ttr, s)
        _, _, mae_tr, _, _ = metrics_orig(net, Xtr, Ttr, lo, hi)
        _, _, mae_te, _, _ = metrics_orig(net, Xte, Tte, lo, hi)
        rows.append(dict(seed=s, net=net, ep=ep, es=es, ok=ok, hist=hist, mae_tr=mae_tr, mae_te=mae_te))
    return rows, Xtr, Ttr, Xte, Tte


def summarize(rows):
    eps = np.array([r["ep"] for r in rows], float)
    mae = np.array([r["mae_te"] for r in rows], float)
    n_fail = int(sum(not r["ok"] for r in rows))
    return dict(
        ep_mean=eps.mean(),
        ep_std=eps.std(),
        mae_mean=mae.mean(),
        mae_std=mae.std(),
        n_fail=n_fail,
        rows=rows,
    )


def plot_all(k, x, lo, hi, base, p_cmp, rec_pred, rec_true, k0, steps):
    os.makedirs(FIG, exist_ok=True)
    net = base["net"]
    p = P_VAR
    xn = nrm(x, lo, hi)
    one = np.full_like(x, np.nan)
    for t in range(p, len(x)):
        one[t] = den(net.fwd(xn[t - p : t]), lo, hi)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(k, x, "k-", lw=1, label="ряд")
    ax.plot(k, one, "C0-", lw=1, alpha=0.85, label="одношаговый")
    ax.axvline(SPLIT - 0.5, color="C3", ls="--", label="train/test")
    ax.set(xlabel="k", ylabel="x(k)", title="Ряд и одношаговый прогноз")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "series.png"), dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(base["hist"], label=f"база p={P_VAR}, N={N_VAR}")
    for lab, r in p_cmp:
        ax.plot(r["hist"], ls="--", label=lab)
    ax.axhline(EE, color="k", ls=":", label="Ee")
    ax.set(xlabel="эпоха", ylabel="Es", title="Сходимость", yscale="log")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "conv.png"), dpi=140)
    plt.close()

    Xte, Tte = windows(xn, p, SPLIT, len(x))
    pred_te, y_te, *_ = metrics_orig(net, Xte, Tte, lo, hi)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_te, pred_te, s=18, alpha=0.8)
    lim = [min(y_te.min(), pred_te.min()), max(y_te.max(), pred_te.max())]
    ax.plot(lim, lim, "k--", label="y=x")
    ax.set(xlabel="эталон", ylabel="прогноз", title="Тест: эталон–прогноз", aspect="equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "scatter.png"), dpi=140)
    plt.close()

    rng = float(x.max() - x.min())
    err = np.abs(rec_true - rec_pred)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    h = np.arange(steps)
    ax[0].plot(h, rec_true, "k-", label="эталон")
    ax[0].plot(h, rec_pred, "C0--", label="рекурсия")
    ax[0].set(xlabel="шаг от k=140", ylabel="x", title="Рекурсивный прогноз 30 шагов")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(h, err, "C3-")
    ax[1].axhline(0.05 * rng, color="k", ls=":", label="5% размаха")
    ax[1].set(xlabel="шаг", ylabel="|ошибка|", title="Рост ошибки")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "recursive.png"), dpi=140)
    plt.close()
    over = np.where(err > 0.05 * rng)[0]
    return int(over[0] + 1) if len(over) else None, rng, one


def infer(net, win, lo, hi, true=None):
    wn = nrm(win, lo, hi)
    yn = net.fwd(wn)
    y = float(den(yn, lo, hi))
    print(f"  in_norm={[round(float(v), 4) for v in wn]}")
    print(f"  y_norm={yn:.4f}  y={y:.4f}", end="")
    if true is not None:
        print(f"  x_true={true:.4f}  |e|={abs(y - true):.4f}")
    else:
        print()


def demo_mode(net, x, lo, hi, p):
    print("\n--- режим функционирования ---")
    print("train:")
    for t in (p, 40, 100):
        print(f"k={t}")
        infer(net, x[t - p : t], lo, hi, x[t])
    print("test:")
    for t in (140, 160, 190):
        print(f"k={t}")
        infer(net, x[t - p : t], lo, hi, x[t])
    print("пользовательские:")
    infer(net, x[50 - p : 50], lo, hi, x[50])
    outlier = list(x[50 - p : 50])
    outlier[-1] += 3.0
    print("окно с выбросом (последнее +3):")
    infer(net, outlier, lo, hi, x[50])


def interactive(net, x, lo, hi, p):
    print("\nввод: p чисел ИЛИ k  (q — выход)")
    while True:
        s = input("> ").strip()
        if s.lower() in ("q", "quit", "exit"):
            break
        try:
            vals = list(map(float, s.split()))
            if len(vals) == 1:
                t = int(vals[0])
                if not (p <= t <= K_MAX):
                    raise ValueError
                infer(net, x[t - p : t], lo, hi, x[t])
            elif len(vals) == p:
                infer(net, vals, lo, hi)
            else:
                raise ValueError
        except ValueError:
            print(f"нужно {p} чисел из шкалы ряда или k в [{p}; {K_MAX}]")


def main():
    os.makedirs(FIG, exist_ok=True)
    k, x = generate()
    lo, hi = minmax_fit(x[:SPLIT])
    xn = nrm(x, lo, hi)
    print(f"ЛР4 вар.3  a={A} b={B} c={C} d={D}  p={P_VAR} N={N_VAR}")
    print(f"train k=0..{SPLIT-1} ({SPLIT} т.), test k={SPLIT}..{K_MAX} ({K_MAX-SPLIT+1} т.)")
    print(f"min/max train: {lo:.4f} / {hi:.4f}")
    print(f"окон train={SPLIT-P_VAR}, test={K_MAX-SPLIT+1}; η={LR}, Ee={EE}")
    print(f"тест за [0;1] нормализ.: {(xn[SPLIT:] < 0).sum() + (xn[SPLIT:] > 1).sum()} точек")

    print("\n=== база p=6 N=5 seed=0 ===")
    base_rows, Xtr, Ttr, Xte, Tte = eval_config(P_VAR, N_VAR, xn, lo, hi, seeds=[0])
    base = base_rows[0]
    ptr, ytr, mae_tr, rmse_tr, sm_tr = metrics_orig(base["net"], Xtr, Ttr, lo, hi)
    pte, yte, mae_te, rmse_te, sm_te = metrics_orig(base["net"], Xte, Tte, lo, hi)
    print(f"эпохи={base['ep']} Es={base['es']:.6e} сход={base['ok']}")
    print(f"train MAE={mae_tr:.4f} RMSE={rmse_tr:.4f} sMAPE={sm_tr:.2f}%")
    print(f"test  MAE={mae_te:.4f} RMSE={rmse_te:.4f} sMAPE={sm_te:.2f}%")

    print("\n=== исследование p (N=5, 5 seed) ===")
    p_tab = {}
    p_hists = []
    for p in (2, 4, P_VAR, 2 * P_VAR):
        rows, *_ = eval_config(p, N_VAR, xn, lo, hi)
        p_tab[p] = summarize(rows)
        s = p_tab[p]
        print(f"p={p:2d}  эпохи {s['ep_mean']:.0f}±{s['ep_std']:.0f}  "
              f"MAE_te {s['mae_mean']:.4f}±{s['mae_std']:.4f}  fail {s['n_fail']}/5")
        if p in (2, 2 * P_VAR):
            p_hists.append((f"p={p}", next((r for r in rows if r["ok"]), rows[0])))

    print("\n=== исследование N (p=6, 5 seed) ===")
    n_tab = {}
    for n in (1, 2, N_VAR, 2 * N_VAR):
        rows, *_ = eval_config(P_VAR, n, xn, lo, hi)
        n_tab[n] = summarize(rows)
        s = n_tab[n]
        print(f"N={n:2d}  эпохи {s['ep_mean']:.0f}±{s['ep_std']:.0f}  "
              f"MAE_te {s['mae_mean']:.4f}±{s['mae_std']:.4f}  fail {s['n_fail']}/5")

    best_p, best_n, best_mae, best_row = P_VAR, N_VAR, 1e9, base
    for p, s in p_tab.items():
        if s["mae_mean"] < best_mae:
            best_p, best_n, best_mae, best_row = p, N_VAR, s["mae_mean"], min(s["rows"], key=lambda r: r["mae_te"])
    for n, s in n_tab.items():
        if s["mae_mean"] < best_mae:
            best_p, best_n, best_mae, best_row = P_VAR, n, s["mae_mean"], min(s["rows"], key=lambda r: r["mae_te"])
    print(f"\nлучшая по среднему test MAE: p={best_p} N={best_n} MAE={best_mae:.4f}")

    steps = 30
    rec = recursive(best_row["net"], x, lo, hi, SPLIT, steps, best_p)
    rec_true = x[SPLIT : SPLIT + steps]
    over, rng, _ = plot_all(k, x, lo, hi, base, p_hists, rec, rec_true, SPLIT, steps)
    print(f"рекурсия 30 шагов: MAE={mae_(rec_true, rec):.4f}")
    print(f"ошибка > 5% размаха ({0.05*rng:.4f}) с шага {over}" if over else "5% размаха не превышен")

    rng_n = np.random.RandomState(0)
    x_noisy = x.copy()
    x_noisy[:SPLIT] += 0.05 * rng_n.randn(SPLIT)
    lo_n, hi_n = minmax_fit(x_noisy[:SPLIT])
    xn_n = nrm(x_noisy, lo_n, hi_n)
    noise_rows, *_ = eval_config(P_VAR, N_VAR, xn_n, lo_n, hi_n, seeds=[0])
    Xte_n, Tte_n = windows(nrm(x, lo_n, hi_n), P_VAR, SPLIT, len(x))
    _, _, mae_n, *_ = metrics_orig(noise_rows[0]["net"], Xte_n, Tte_n, lo_n, hi_n)
    print(f"шум σ=0.05 на train: test MAE={mae_n:.4f} (без шума {mae_te:.4f})")

    demo_mode(base["net"], x, lo, hi, P_VAR)
    print(f"\nграфики: {FIG}")
    if sys.stdin.isatty():
        interactive(base["net"], x, lo, hi, P_VAR)


if __name__ == "__main__":
    main()
