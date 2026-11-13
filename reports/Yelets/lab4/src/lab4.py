import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           
import matplotlib.pyplot as plt

A, B, C, D = 0.8, 0.20, 0.4, 0.50
P, N = 8, 4
K = 200                        
TRAIN_END = 140                  
LR = 0.1                        
MAX_EPOCH = 20000
EE = 1e-4                        
SEEDS = [0, 1, 2, 3, 4]
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))

def build_series(n=K):
    k = np.arange(n, dtype=float)
    x = A * np.cos(B * k) + C * np.sin(D * k)
    return k, x

def train_minmax(x_train):
    return float(x_train.min()), float(x_train.max())

def normalize(x, vmin, vmax):
    return (np.asarray(x, float) - vmin) / (vmax - vmin)

def denormalize(y, vmin, vmax):
    return vmin + np.asarray(y, float) * (vmax - vmin)

def sliding_windows(series, p, start, end):
    X, Y = [], []
    for t in range(start, end):
        X.append(series[t - p:t])
        Y.append(series[t])
    return np.asarray(X, float), np.asarray(Y, float)

def calculate_mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))

def calculate_rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def calculate_smape(y, yhat):
    denom = np.abs(y) + np.abs(yhat) + 1e-12
    return float(np.mean(2.0 * np.abs(y - yhat) / denom) * 100.0)

class MLP:
    def __init__(self, p, n_hidden, seed, init_scale=0.1):
        rng = np.random.RandomState(seed)
        self.p = p
        self.n = n_hidden
        self.W1 = rng.randn(p, n_hidden) * init_scale
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.randn(n_hidden) * init_scale
        self.b2 = 0.0

    def forward(self, x):
        x = np.asarray(x, float).ravel()
        z = x @ self.W1 + self.b1
        h = sigmoid(z)
        y = float(h @ self.W2 + self.b2)
        return h, y

    def backward_online(self, x, h, y, target, lr):
        x = np.asarray(x, float).ravel()
        delta_out = y - target                               
        grad_W2 = delta_out * h                             
        grad_b2 = delta_out
        delta_h = (delta_out * self.W2) * h * (1.0 - h)       
        grad_W1 = np.outer(x, delta_h)                     
        grad_b1 = delta_h
        self.W2 -= lr * grad_W2
        self.b2 -= lr * grad_b2
        self.W1 -= lr * grad_W1
        self.b1 -= lr * grad_b1
    def predict(self, X):
        return np.array([self.forward(x)[1] for x in X])

def train_mlp(Xtr, Ytr, p, n_hidden, seed, lr=LR, max_epoch=MAX_EPOCH, ee=EE):
    net = MLP(p, n_hidden, seed)
    rng = np.random.RandomState(seed + 100)
    m = len(Ytr)
    history = []
    for epoch in range(1, max_epoch + 1):
        for i in rng.permutation(m):
            h, y = net.forward(Xtr[i])
            net.backward_online(Xtr[i], h, y, Ytr[i], lr)
        preds = net.predict(Xtr)
        es = float(np.mean((preds - Ytr) ** 2))
        history.append(es)
        if es <= ee:
            return net, epoch, es, True, history
    return net, max_epoch, es, False, history

def eval_metrics(net, Xn, tn, vmin, vmax):
    pred = denormalize(net.predict(Xn), vmin, vmax)
    true = denormalize(tn, vmin, vmax)
    mae_val = calculate_mae(true, pred)
    rmse_val = calculate_rmse(true, pred)
    smape_val = calculate_smape(true, pred)
    return pred, true, mae_val, rmse_val, smape_val

def run_config(p, n_hidden, xn, vmin, vmax, seeds=SEEDS):
    Xtr, Ytr = sliding_windows(xn, p, p, TRAIN_END)
    Xte, Yte = sliding_windows(xn, p, TRAIN_END, len(xn))
    rows = []
    for s in seeds:
        net, ep, es, ok, hist = train_mlp(Xtr, Ytr, p, n_hidden, s)
        mae_tr = eval_metrics(net, Xtr, Ytr, vmin, vmax)[2]
        mae_te = eval_metrics(net, Xte, Yte, vmin, vmax)[2]
        rows.append({"seed": s, "net": net, "ep": ep, "es": es,
                     "ok": ok, "hist": hist,
                     "mae_tr": mae_tr, "mae_te": mae_te})
    return rows, Xtr, Ytr, Xte, Yte

def summarize(results):
    epochs = np.array([r["ep"] for r in results], float)
    test_maes = np.array([r["mae_te"] for r in results], float)
    n_failed = int(np.sum([not r["ok"] for r in results]))
    return {"ep_mean": epochs.mean(), "ep_std": epochs.std(),
        "mae_mean": test_maes.mean(), "mae_std": test_maes.std(),
        "n_fail": n_failed, "rows": results,
    }

def multi_step_forecast(net, series, vmin, vmax, k0, steps, p):
    window = list(normalize(series[k0 - p:k0], vmin, vmax))
    preds = []
    for _ in range(steps):
        _, y_norm = net.forward(window)
        preds.append(float(denormalize(y_norm, vmin, vmax)))
        window = window[1:] + [y_norm]
    return np.array(preds)

def plot_series_with_boundary(k, x, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(k, x, "k-", lw=1, label="ряд")
    ax.axvline(TRAIN_END - 0.5, color="C3", ls="--", label="граница train/test")
    ax.set(xlabel="k", ylabel="x(k)", title="Ряд x(k)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)

def plot_one_step(k, x, net, p, vmin, vmax, path):
    xn = normalize(x, vmin, vmax)
    one = np.full_like(x, np.nan)
    for t in range(p, len(x)):
        _, y_norm = net.forward(xn[t - p:t])
        one[t] = denormalize(y_norm, vmin, vmax)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(k, x, "k-", lw=1, label="ряд")
    ax.plot(k, one, "C0-", lw=1, alpha=0.85, label="одношаговый прогноз")
    ax.axvline(TRAIN_END - 0.5, color="C3", ls="--", label="train/test")
    ax.set(xlabel="k", ylabel="x(k)", title="Ряд и одношаговый прогноз")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)

def plot_convergence(base_hist, p_curves, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(base_hist, label=f"p={P}, N={N}")
    for lab, hist in p_curves:
        ax.plot(hist, ls="--", label=lab)
    ax.axhline(EE, color="k", ls=":", label="Ee")
    ax.set(xlabel="эпоха", ylabel="Es", title="Сходимость", yscale="log")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)

def plot_scatter(net, Xte, Yte, vmin, vmax, path):
    pred, true, *_ = eval_metrics(net, Xte, Yte, vmin, vmax)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true, pred, s=18, alpha=0.8)
    lim = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    ax.plot(lim, lim, "k--", label="y = x")
    ax.set(xlabel="эталон", ylabel="прогноз",
           title="Тест: эталон–прогноз", aspect="equal")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)

def plot_recursive(rec_pred, rec_true, threshold, path):
    err = np.abs(rec_true - rec_pred)
    h = np.arange(1, len(rec_true) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(h, rec_true, "k-", label="эталон")
    ax[0].plot(h, rec_pred, "C0--", label="рекурсия")
    ax[0].set(xlabel="шаг от k=140", ylabel="x",
              title="Рекурсивный прогноз на 30 шагов")
    ax[0].legend(); ax[0].grid(True, alpha=0.3)
    ax[1].plot(h, err, "C3-")
    ax[1].axhline(threshold, color="k", ls=":", label="5% размаха")
    ax[1].set(xlabel="шаг", ylabel="ошибка", title="Рост ошибки")
    ax[1].legend(); ax[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)

def print_forecast(net, window_orig, vmin, vmax, true_val=None):
    wn = normalize(window_orig, vmin, vmax)
    _, y_norm = net.forward(wn)
    y = float(denormalize(y_norm, vmin, vmax))
    print(f"нормализованные входы: [{', '.join(f'{v:.4f}' for v in wn)}]")
    print(f"нормализованный прогноз: {y_norm:.4f}")
    print(f"прогноз (ряд): {y:.4f}")
    if true_val is not None:
        print(f"истинный x(k): {true_val:.4f}")
        print(f"абсолютная ошибка: {abs(y - true_val):.4f}")
    print()

def demo_functioning(net, x, vmin, vmax, p):
    print("\nРежим функционирования: ")
    train_ks = [11, 47, 121]         
    test_ks = [148, 163, 192]
    for kk in train_ks:
        print(f"k={kk} (train)")
        print_forecast(net, x[kk - p:kk], vmin, vmax, true_val=x[kk])
    for kk in test_ks:
        print(f"k={kk} (test)")
        print_forecast(net, x[kk - p:kk], vmin, vmax, true_val=x[kk])

    custom = x[80:80 + p].copy()
    custom[p // 2] += 3.0
    print("окно 1 (с выбросом в середине +3)")
    print_forecast(net, custom, vmin, vmax)

    flat_norm = np.full(p, 0.3)
    flat_orig = denormalize(flat_norm, vmin, vmax)
    print("окно 2 (без выброса)")
    print_forecast(net, flat_orig, vmin, vmax)

def interactive_mode(net, x, vmin, vmax, p):
    print("\nИнтерактивный режим, для выхода введите q")
    while True:
        try:
            s = input(f"Введите номер k ({p}..{K-1}) или {p} чисел в исходной шкале: ").strip()
        except EOFError:
            break
        if s.lower() in ("q", "quit", "exit"):
            break
        parts = s.split()
        if len(parts) == 1:
            try:
                kk = int(parts[0])
            except ValueError:
                print(f"  нужен номер k в [{p}; {K-1}]\n"); continue
            if not (p <= kk < K):
                print(f"  k должен быть в [{p}; {K-1}]\n"); continue
            region = "train" if kk < TRAIN_END else "test"
            print(f"  k={kk} ({region})")
            print_forecast(net, x[kk - p:kk], vmin, vmax, true_val=x[kk])
        elif len(parts) == p:
            try:
                raw = np.array([float(v) for v in parts])
            except ValueError:
                print("  нужны числа\n"); continue
            print_forecast(net, raw, vmin, vmax)
        else:
            print(f"  нужно либо 1 число (k), либо {p} чисел\n")

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    k, x = build_series()
    vmin, vmax = train_minmax(x[:TRAIN_END])
    xn = normalize(x, vmin, vmax)

    print(f"train: k=0..{TRAIN_END - 1} ({TRAIN_END} точек); "
          f"test: k={TRAIN_END}..{K - 1} ({K - TRAIN_END} точек)")
    print(f"min по train: {vmin:.4f}, max по train: {vmax:.4f}")
    n_train_win = TRAIN_END - P
    n_test_win = K - TRAIN_END
    print(f"Окон train: {n_train_win}, окон test: {n_test_win}")

    plot_series_with_boundary(k, x, os.path.join(FIG_DIR, "fig1_series.png"))

    print(f"\nБазовый эксперимент: p={P}, N={N}, seed=0")
    base_rows, Xtr, Ytr, Xte, Yte = run_config(P, N, xn, vmin, vmax, seeds=[0])
    base = base_rows[0]
    _, _, mae_tr, rmse_tr, sm_tr = eval_metrics(base["net"], Xtr, Ytr, vmin, vmax)
    _, _, mae_te, rmse_te, sm_te = eval_metrics(base["net"], Xte, Yte, vmin, vmax)
    print(f"эпохи={base['ep']}  Es={base['es']:.6e}  сошлось={base['ok']}")
    print(f"train: MAE={mae_tr:.4f}  RMSE={rmse_tr:.4f}  sMAPE={sm_tr:.2f}%")
    print(f"test:  MAE={mae_te:.4f}  RMSE={rmse_te:.4f}  sMAPE={sm_te:.2f}%")

    print(f"\nИсследование p: N={N}, 5 seed")
    p_grid = [2, 4, P, 2 * P]
    window_stats, p_curves = {}, []
    for p in p_grid:
        rows, *_ = run_config(p, N, xn, vmin, vmax)
        window_stats[p] = summarize(rows)
        s = window_stats[p]
        print(f"p={p:2d}  эпохи {s['ep_mean']:.0f}±{s['ep_std']:.0f}  "
              f"тест. MAE {s['mae_mean']:.4f}±{s['mae_std']:.4f}  не сошлось {s['n_fail']}/5")
        if p in (2, 2 * P):
            exemplar = next((r for r in rows if r["ok"]), rows[0])
            p_curves.append((f"p={p}", exemplar["hist"]))

    print(f"\nИсследование N: p={P}, 5 seed")
    n_grid = [1, 2, N, 2 * N]
    hidden_stats = {}
    for n_h in n_grid:
        rows, *_ = run_config(P, n_h, xn, vmin, vmax)
        hidden_stats[n_h] = summarize(rows)
        s = hidden_stats[n_h]
        print(f"N={n_h:2d}  эпохи {s['ep_mean']:.0f}±{s['ep_std']:.0f}  "
              f"тест. MAE {s['mae_mean']:.4f}±{s['mae_std']:.4f}  не сошлось {s['n_fail']}/5")

    best = dict(p=P, n=N, mae=mae_te, row=base, kind="base")
    for p, s in window_stats.items():
        if s["mae_mean"] < best["mae"]:
            best.update(p=p, n=N, mae=s["mae_mean"], row=None, kind=f"p={p}")
    for n_h, s in hidden_stats.items():
        if s["mae_mean"] < best["mae"]:
            best.update(p=P, n=n_h, mae=s["mae_mean"], row=None, kind=f"N={n_h}")
    print(f"\nЛучшая по среднему тест. MAE: p={best['p']}, N={best['n']}, "
          f"MAE={best['mae']:.4f}")

    if best["row"] is None:
        source = best["kind"]
        if source.startswith("p="):
            pool = window_stats[best["p"]]["rows"]
        else:
            pool = hidden_stats[best["n"]]["rows"]
    else:
        pool = [base]
    best_row = min(pool, key=lambda r: r["mae_te"])
    best_net = best_row["net"]
    p_best, n_best = best["p"], best["n"]

    plot_convergence(base["hist"], p_curves,
                     os.path.join(FIG_DIR, "fig2_convergence.png"))

    plot_scatter(base["net"], Xte, Yte, vmin, vmax,
                 os.path.join(FIG_DIR, "fig3_scatter.png"))

    plot_one_step(k, x, base["net"], P, vmin, vmax,
                  os.path.join(FIG_DIR, "fig1b_one_step.png"))

    steps = 30
    rec_pred = multi_step_forecast(best_net, x, vmin, vmax, TRAIN_END, steps, p_best)
    rec_true = x[TRAIN_END:TRAIN_END + steps]
    rng_span = float(x[:TRAIN_END].max() - x[:TRAIN_END].min())
    threshold = 0.05 * rng_span
    rec_err = np.abs(rec_true - rec_pred)
    exceed_idx = np.where(rec_err > threshold)[0]
    first_exceed = int(exceed_idx[0] + 1) if len(exceed_idx) else None
    print(f"\nРекурсивный прогноз (p={p_best}, N={n_best}, steps=30): "
          f"MAE={calculate_mae(rec_true, rec_pred):.4f}")
    print(f"размах train={rng_span:.4f}, порог 5%={threshold:.4f}, "
          f"первый шаг превышения: {first_exceed if first_exceed else 'не превышен'}")

    plot_recursive(rec_pred, rec_true, threshold,
                   os.path.join(FIG_DIR, "fig4_recursive.png"))

    demo_functioning(base["net"], x, vmin, vmax, P)
    import sys
    if sys.stdin.isatty():
        interactive_mode(base["net"], x, vmin, vmax, P)
    print(f"\nграфики сохранены в: {FIG_DIR}")

if __name__ == "__main__":
    main()