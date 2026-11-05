import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AMP_COS, FREQ_COS = 3.0, 0.22
AMP_SIN, FREQ_SIN = 1.5, 0.55
WIN, HID = 8, 6
N_POINTS = 200
TRAIN_LIMIT = 140
ETA = 0.1
EPOCH_CAP = 20000
TOL = 1e-4
SEED_LIST = [0, 1, 2, 3, 4]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def act_sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def make_series(n=N_POINTS):
    idx = np.arange(n, dtype=float)
    vals = AMP_COS * np.cos(FREQ_COS * idx) + AMP_SIN * np.sin(FREQ_SIN * idx)
    return idx, vals


def minmax_from_train(part):
    return float(part.min()), float(part.max())


def to_unit(vals, lo, hi):
    return (np.asarray(vals, float) - lo) / (hi - lo)


def from_unit(vals, lo, hi):
    return lo + np.asarray(vals, float) * (hi - lo)


def make_windows(seq, p, left, right):
    feats, targets = [], []
    for t in range(left, right):
        feats.append(seq[t - p:t])
        targets.append(seq[t])
    return np.asarray(feats, float), np.asarray(targets, float)


def metric_mae(true, pred):
    return float(np.mean(np.abs(true - pred)))


def metric_rmse(true, pred):
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def metric_smape(true, pred):
    denom = np.abs(true) + np.abs(pred) + 1e-12
    return float(np.mean(2.0 * np.abs(true - pred) / denom) * 100.0)


class Perceptron:
    def __init__(self, p, n_hidden, seed, scale=0.1):
        rng = np.random.RandomState(seed)
        self.p = p
        self.n = n_hidden
        self.W_in = rng.randn(p, n_hidden) * scale
        self.b_in = np.zeros(n_hidden)
        self.W_out = rng.randn(n_hidden) * scale
        self.b_out = 0.0

    def forward(self, x):
        x = np.asarray(x, float).ravel()
        s = x @ self.W_in + self.b_in
        h = act_sigmoid(s)
        out = float(h @ self.W_out + self.b_out)
        return h, out

    def backward_online(self, x, h, out, target, lr):
        x = np.asarray(x, float).ravel()
        d_out = out - target
        gW_out = d_out * h
        gb_out = d_out
        d_hidden = (d_out * self.W_out) * h * (1.0 - h)
        gW_in = np.outer(x, d_hidden)
        gb_in = d_hidden
        self.W_out -= lr * gW_out
        self.b_out -= lr * gb_out
        self.W_in -= lr * gW_in
        self.b_in -= lr * gb_in

    def predict(self, X):
        return np.array([self.forward(x)[1] for x in X])


def fit_network(Xtr, Ytr, p, n_hidden, seed, lr=ETA, max_epoch=EPOCH_CAP, tol=TOL):
    net = Perceptron(p, n_hidden, seed)
    rng = np.random.RandomState(seed + 100)
    m = len(Ytr)
    hist = []
    for ep in range(1, max_epoch + 1):
        for i in rng.permutation(m):
            h, y = net.forward(Xtr[i])
            net.backward_online(Xtr[i], h, y, Ytr[i], lr)
        preds = net.predict(Xtr)
        es = float(np.mean((preds - Ytr) ** 2))
        hist.append(es)
        if es <= tol:
            return net, ep, es, True, hist
    return net, max_epoch, es, False, hist


def evaluate(net, Xn, tn, lo, hi):
    pred = from_unit(net.predict(Xn), lo, hi)
    true = from_unit(tn, lo, hi)
    return pred, true, metric_mae(true, pred), metric_rmse(true, pred), metric_smape(true, pred)


def run_config(p, n_hidden, xn, lo, hi, seeds=SEED_LIST):
    Xtr, Ytr = make_windows(xn, p, p, TRAIN_LIMIT)
    Xte, Yte = make_windows(xn, p, TRAIN_LIMIT, len(xn))
    rows = []
    for s in seeds:
        net, ep, es, ok, hist = fit_network(Xtr, Ytr, p, n_hidden, s)
        mae_tr = evaluate(net, Xtr, Ytr, lo, hi)[2]
        mae_te = evaluate(net, Xte, Yte, lo, hi)[2]
        rows.append({"seed": s, "net": net, "ep": ep, "es": es,
                     "ok": ok, "hist": hist,
                     "mae_tr": mae_tr, "mae_te": mae_te})
    return rows, Xtr, Ytr, Xte, Yte


def stats(results):
    epochs = np.array([r["ep"] for r in results], float)
    mae_te = np.array([r["mae_te"] for r in results], float)
    n_fail = int(np.sum([not r["ok"] for r in results]))
    return {"ep_mean": epochs.mean(), "ep_std": epochs.std(),
            "mae_mean": mae_te.mean(), "mae_std": mae_te.std(),
            "n_fail": n_fail, "rows": results}


def recursive_forecast(net, series, lo, hi, k_start, steps, p):
    window = list(to_unit(series[k_start - p:k_start], lo, hi))
    result = []
    for _ in range(steps):
        _, y_norm = net.forward(window)
        result.append(float(from_unit(y_norm, lo, hi)))
        window = window[1:] + [y_norm]
    return np.array(result)


def fig_series(idx, vals, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idx, vals, "k-", lw=1, label="ряд")
    ax.axvline(TRAIN_LIMIT - 0.5, color="C3", ls="--", label="граница train/test")
    ax.set(xlabel="k", ylabel="x(k)", title="Ряд x(k)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_one_step(idx, vals, net, p, lo, hi, path):
    xn = to_unit(vals, lo, hi)
    one = np.full_like(vals, np.nan)
    for t in range(p, len(vals)):
        _, y_norm = net.forward(xn[t - p:t])
        one[t] = from_unit(y_norm, lo, hi)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idx, vals, "k-", lw=1, label="ряд")
    ax.plot(idx, one, "C0-", lw=1, alpha=0.85, label="одношаговый прогноз")
    ax.axvline(TRAIN_LIMIT - 0.5, color="C3", ls="--", label="train/test")
    ax.set(xlabel="k", ylabel="x(k)", title="Ряд и одношаговый прогноз")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_convergence(base_hist, extra_curves, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(base_hist, label=f"p={WIN}, N={HID}")
    for lab, hist in extra_curves:
        ax.plot(hist, ls="--", label=lab)
    ax.axhline(TOL, color="k", ls=":", label="Ee")
    ax.set(xlabel="эпоха", ylabel="Es", title="Сходимость", yscale="log")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_scatter(net, Xte, Yte, lo, hi, path):
    pred, true, *_ = evaluate(net, Xte, Yte, lo, hi)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true, pred, s=18, alpha=0.8)
    lim = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    ax.plot(lim, lim, "k--", label="y = x")
    ax.set(xlabel="эталон", ylabel="прогноз",
           title="Тест: эталон–прогноз", aspect="equal")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_recursive(rec_pred, rec_true, threshold, path):
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


def show_forecast(net, window_orig, lo, hi, true_val=None):
    wn = to_unit(window_orig, lo, hi)
    _, y_norm = net.forward(wn)
    y = float(from_unit(y_norm, lo, hi))
    print(f"нормализованные входы: [{', '.join(f'{v:.4f}' for v in wn)}]")
    print(f"нормализованный прогноз: {y_norm:.4f}")
    print(f"прогноз (ряд): {y:.4f}")
    if true_val is not None:
        print(f"истинный x(k): {true_val:.4f}")
        print(f"абсолютная ошибка: {abs(y - true_val):.4f}")
    print()


def demo_mode(net, vals, lo, hi, p):
    print("\nРежим функционирования: ")
    train_ks = [11, 47, 121]
    test_ks = [148, 163, 192]
    for kk in train_ks:
        print(f"k={kk} (train)")
        show_forecast(net, vals[kk - p:kk], lo, hi, true_val=vals[kk])
    for kk in test_ks:
        print(f"k={kk} (test)")
        show_forecast(net, vals[kk - p:kk], lo, hi, true_val=vals[kk])
    custom = vals[80:80 + p].copy()
    custom[p // 2] += 3.0
    print("окно 1 (с выбросом в середине +3)")
    show_forecast(net, custom, lo, hi)
    flat_norm = np.full(p, 0.3)
    flat_orig = from_unit(flat_norm, lo, hi)
    print("окно 2 (без выброса)")
    show_forecast(net, flat_orig, lo, hi)


def interactive_mode(net, vals, lo, hi, p):
    print("\nИнтерактивный режим, для выхода введите q")
    while True:
        try:
            s = input(f"Введите номер k ({p}..{N_POINTS - 1}) или {p} чисел в исходной шкале: ").strip()
        except EOFError:
            break
        if s.lower() in ("q", "quit", "exit"):
            break
        parts = s.split()
        if len(parts) == 1:
            try:
                kk = int(parts[0])
            except ValueError:
                print(f"  нужен номер k в [{p}; {N_POINTS - 1}]\n"); continue
            if not (p <= kk < N_POINTS):
                print(f"  k должен быть в [{p}; {N_POINTS - 1}]\n"); continue
            region = "train" if kk < TRAIN_LIMIT else "test"
            print(f"  k={kk} ({region})")
            show_forecast(net, vals[kk - p:kk], lo, hi, true_val=vals[kk])
        elif len(parts) == p:
            try:
                raw = np.array([float(v) for v in parts])
            except ValueError:
                print("  нужны числа\n"); continue
            show_forecast(net, raw, lo, hi)
        else:
            print(f"  нужно либо 1 число (k), либо {p} чисел\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    idx, vals = make_series()
    lo, hi = minmax_from_train(vals[:TRAIN_LIMIT])
    xn = to_unit(vals, lo, hi)

    print(f"train: k=0..{TRAIN_LIMIT - 1} ({TRAIN_LIMIT} точек); "
          f"test: k={TRAIN_LIMIT}..{N_POINTS - 1} ({N_POINTS - TRAIN_LIMIT} точек)")
    print(f"min по train: {lo:.4f}, max по train: {hi:.4f}")
    print(f"Окон train: {TRAIN_LIMIT - WIN}, окон test: {N_POINTS - TRAIN_LIMIT}")

    fig_series(idx, vals, os.path.join(OUT_DIR, "fig1_series.png"))

    print(f"\nБазовый эксперимент: p={WIN}, N={HID}, seed=0")
    base_rows, Xtr, Ytr, Xte, Yte = run_config(WIN, HID, xn, lo, hi, seeds=[0])
    base = base_rows[0]
    _, _, mae_tr, rmse_tr, sm_tr = evaluate(base["net"], Xtr, Ytr, lo, hi)
    _, _, mae_te, rmse_te, sm_te = evaluate(base["net"], Xte, Yte, lo, hi)
    print(f"эпохи={base['ep']}  Es={base['es']:.6e}  сошлось={base['ok']}")
    print(f"train: MAE={mae_tr:.4f}  RMSE={rmse_tr:.4f}  sMAPE={sm_tr:.2f}%")
    print(f"test:  MAE={mae_te:.4f}  RMSE={rmse_te:.4f}  sMAPE={sm_te:.2f}%")

    print(f"\nИсследование p: N={HID}, 5 seed")
    p_grid = [2, 4, WIN, 2 * WIN]
    win_stats, extra_curves = {}, []
    for p in p_grid:
        rows, *_ = run_config(p, HID, xn, lo, hi)
        win_stats[p] = stats(rows)
        s = win_stats[p]
        print(f"p={p:2d}  эпохи {s['ep_mean']:.0f}±{s['ep_std']:.0f}  "
              f"тест. MAE {s['mae_mean']:.4f}±{s['mae_std']:.4f}  не сошлось {s['n_fail']}/5")
        if p in (2, 2 * WIN):
            exemplar = next((r for r in rows if r["ok"]), rows[0])
            extra_curves.append((f"p={p}", exemplar["hist"]))

    print(f"\nИсследование N: p={WIN}, 5 seed")
    n_grid = [1, 2, HID, 2 * HID]
    hid_stats = {}
    for n_h in n_grid:
        rows, *_ = run_config(WIN, n_h, xn, lo, hi)
        hid_stats[n_h] = stats(rows)
        s = hid_stats[n_h]
        print(f"N={n_h:2d}  эпохи {s['ep_mean']:.0f}±{s['ep_std']:.0f}  "
              f"тест. MAE {s['mae_mean']:.4f}±{s['mae_std']:.4f}  не сошлось {s['n_fail']}/5")

    best = dict(p=WIN, n=HID, mae=mae_te, row=base, kind="base")
    for p, s in win_stats.items():
        if s["mae_mean"] < best["mae"]:
            best.update(p=p, n=HID, mae=s["mae_mean"], row=None, kind=f"p={p}")
    for n_h, s in hid_stats.items():
        if s["mae_mean"] < best["mae"]:
            best.update(p=WIN, n=n_h, mae=s["mae_mean"], row=None, kind=f"N={n_h}")
    print(f"\nЛучшая по среднему тест. MAE: p={best['p']}, N={best['n']}, "
          f"MAE={best['mae']:.4f}")

    if best["row"] is None:
        if best["kind"].startswith("p="):
            pool = win_stats[best["p"]]["rows"]
        else:
            pool = hid_stats[best["n"]]["rows"]
    else:
        pool = [base]
    best_row = min(pool, key=lambda r: r["mae_te"])
    best_net = best_row["net"]
    p_best, n_best = best["p"], best["n"]

    fig_convergence(base["hist"], extra_curves,
                    os.path.join(OUT_DIR, "fig2_convergence.png"))
    fig_scatter(base["net"], Xte, Yte, lo, hi,
                os.path.join(OUT_DIR, "fig3_scatter.png"))
    fig_one_step(idx, vals, base["net"], WIN, lo, hi,
                 os.path.join(OUT_DIR, "fig1b_one_step.png"))

    steps = 30
    rec_pred = recursive_forecast(best_net, vals, lo, hi, TRAIN_LIMIT, steps, p_best)
    rec_true = vals[TRAIN_LIMIT:TRAIN_LIMIT + steps]
    span = float(vals[:TRAIN_LIMIT].max() - vals[:TRAIN_LIMIT].min())
    threshold = 0.05 * span
    rec_err = np.abs(rec_true - rec_pred)
    over = np.where(rec_err > threshold)[0]
    first_over = int(over[0] + 1) if len(over) else None
    print(f"\nРекурсивный прогноз (p={p_best}, N={n_best}, steps=30): "
          f"MAE={metric_mae(rec_true, rec_pred):.4f}")
    print(f"размах train={span:.4f}, порог 5%={threshold:.4f}, "
          f"первый шаг превышения: {first_over if first_over else 'не превышен'}")

    fig_recursive(rec_pred, rec_true, threshold,
                  os.path.join(OUT_DIR, "fig4_recursive.png"))

    demo_mode(base["net"], vals, lo, hi, WIN)
    if sys.stdin.isatty():
        interactive_mode(base["net"], vals, lo, hi, WIN)
    print(f"\nграфики сохранены в: {OUT_DIR}")


if __name__ == "__main__":
    main()