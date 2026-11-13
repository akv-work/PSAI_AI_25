import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A, B, C, D = 1.4, 0.38, 0.7, 0.85
P_VAR, N_VAR = 6, 5
SERIES_LEN = 200
TRAIN_LEN = 140
EE = 1e-4
MAX_EPOCHS = 20000
LR = 0.1
W_INIT = 0.5
SEEDS = [1, 2, 3, 4, 5]
FORECAST_STEPS = 30
THRESHOLD_SHARE = 0.05


def generate_series(n=SERIES_LEN):
    k = np.arange(n)
    return A * np.cos(B * k) + C * np.sin(D * k)


def fit_minmax(train_part):
    return float(train_part.min()), float(train_part.max())


def normalize(x, x_min, x_max):
    return (np.asarray(x, dtype=float) - x_min) / (x_max - x_min)


def denormalize(xn, x_min, x_max):
    return np.asarray(xn, dtype=float) * (x_max - x_min) + x_min


def build_windows(xn, targets, p):
    X = np.array([xn[k - p:k][::-1] for k in targets])
    E = np.array([xn[k] for k in targets])
    return X, E


def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))


class MLP:
    def __init__(self, p, n_hidden, lr, w_init, seed):
        rng = np.random.default_rng(seed)
        self.p = p
        self.n_hidden = n_hidden
        self.lr = lr
        self.W1 = rng.uniform(-w_init, w_init, size=(p, n_hidden))
        self.T1 = rng.uniform(-w_init, w_init, size=(n_hidden,))
        self.W2 = rng.uniform(-w_init, w_init, size=(n_hidden,))
        self.T2 = rng.uniform(-w_init, w_init)

    def forward(self, x):
        y_hidden = sigmoid(x @ self.W1 + self.T1)
        y_out = y_hidden @ self.W2 + self.T2
        return y_hidden, y_out

    def train_step(self, x, e):
        y_hidden, y_out = self.forward(x)
        delta_out = y_out - e
        delta_hidden = delta_out * self.W2 * y_hidden * (1.0 - y_hidden)
        self.W2 -= self.lr * delta_out * y_hidden
        self.T2 -= self.lr * delta_out
        self.W1 -= self.lr * np.outer(x, delta_hidden)
        self.T1 -= self.lr * delta_hidden

    def predict(self, X):
        return sigmoid(X @ self.W1 + self.T1) @ self.W2 + self.T2

    def total_error(self, X, E):
        return float(np.mean(0.5 * (self.predict(X) - E) ** 2))

    def fit(self, X, E, eps, max_epochs, rng):
        order = np.arange(len(X))
        history = []
        es = self.total_error(X, E)
        for epoch in range(1, max_epochs + 1):
            rng.shuffle(order)
            for i in order:
                self.train_step(X[i], E[i])
            es = self.total_error(X, E)
            history.append(es)
            if es <= eps:
                return epoch, es, history
        return max_epochs, es, history


def mae(y, y_hat):
    return float(np.mean(np.abs(y - y_hat)))


def rmse(y, y_hat):
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def smape(y, y_hat):
    denom = np.abs(y) + np.abs(y_hat)
    safe = np.where(denom == 0, 1.0, denom)
    return float(np.mean(np.where(denom == 0, 0.0, 2.0 * np.abs(y - y_hat) / safe)) * 100.0)


def prepare_data(x, p):
    x_min, x_max = fit_minmax(x[:TRAIN_LEN])
    xn = normalize(x, x_min, x_max)
    X_tr, E_tr = build_windows(xn, range(p, TRAIN_LEN), p)
    X_te, E_te = build_windows(xn, range(TRAIN_LEN, len(x)), p)
    return xn, x_min, x_max, X_tr, E_tr, X_te, E_te


def train_one(x, p, n_hidden, seed, lr=LR, eps=EE, max_epochs=MAX_EPOCHS):
    xn, x_min, x_max, X_tr, E_tr, X_te, E_te = prepare_data(x, p)
    model = MLP(p, n_hidden, lr, W_INIT, seed)
    epochs, es, history = model.fit(X_tr, E_tr, eps, max_epochs, np.random.default_rng(seed))
    y_tr = denormalize(E_tr, x_min, x_max)
    y_te = denormalize(E_te, x_min, x_max)
    p_tr = denormalize(model.predict(X_tr), x_min, x_max)
    p_te = denormalize(model.predict(X_te), x_min, x_max)
    return dict(model=model, p=p, n=n_hidden, seed=seed, epochs=epochs, es=es, history=history,
                converged=es <= eps,
                train=dict(mae=mae(y_tr, p_tr), rmse=rmse(y_tr, p_tr), smape=smape(y_tr, p_tr)),
                test=dict(mae=mae(y_te, p_te), rmse=rmse(y_te, p_te), smape=smape(y_te, p_te)),
                y_te=y_te, p_te=p_te, y_tr=y_tr, p_tr=p_tr,
                x_min=x_min, x_max=x_max, n_train=len(X_tr), n_test=len(X_te))


def run_series(x, p, n_hidden, seeds=SEEDS, lr=LR):
    return [train_one(x, p, n_hidden, s, lr=lr) for s in seeds]


def summarize(runs):
    ep = np.array([r["epochs"] for r in runs], dtype=float)
    te = np.array([r["test"]["mae"] for r in runs])
    tr = np.array([r["train"]["mae"] for r in runs])
    return dict(ep_mean=ep.mean(), ep_std=ep.std(), te_mean=te.mean(), te_std=te.std(),
                tr_mean=tr.mean(), failed=sum(not r["converged"] for r in runs))


def print_table(title, label, rows):
    print(f"\n{title}")
    print(f"{label:>8s}{'обуч./тест. примеров':>22s}{'эпохи (ср.±ст.откл.)':>24s}{'MAE test (ср.±ст.откл.)':>26s}{'MAE train (ср.)':>17s}{'не сошлось':>12s}")
    for key, runs in rows:
        s = summarize(runs)
        cnt = f"{runs[0]['n_train']}/{runs[0]['n_test']}"
        print(f"{key:>8}{cnt:>22s}{s['ep_mean']:>16.1f} ± {s['ep_std']:<6.1f}{s['te_mean']:>18.4f} ± {s['te_std']:<7.4f}{s['tr_mean']:>12.4f}{s['failed']:>10d}/5")


def recursive_forecast(model, xn, p, start, steps):
    window = list(xn[start - p:start][::-1])
    preds = []
    for _ in range(steps):
        y = float(model.predict(np.array(window)))
        preds.append(y)
        window = [y] + window[:-1]
    return np.array(preds)


def forecast_window(model, window_chrono, x_min, x_max):
    xn_win = normalize(window_chrono, x_min, x_max)
    y_norm = float(model.predict(xn_win[::-1]))
    return xn_win, y_norm, float(denormalize(y_norm, x_min, x_max))


def show_window(model, x, k_or_values, p, x_min, x_max, label=""):
    if np.isscalar(k_or_values):
        k = int(k_or_values)
        window = x[k - p:k]
        truth = x[k]
    else:
        window = np.asarray(k_or_values, dtype=float)
        k, truth = None, None
    xn_win, y_norm, y_real = forecast_window(model, window, x_min, x_max)
    head = f"{label} k={k}" if k is not None else f"{label}"
    print(f"  {head}")
    print(f"    окно (исходная шкала): {np.round(window, 4).tolist()}")
    print(f"    нормализованные входы: {np.round(xn_win, 4).tolist()}")
    print(f"    прогноз (норм.): {y_norm:.4f}   прогноз (исх.): {y_real:.4f}", end="")
    if truth is not None:
        print(f"   x(k)={truth:.4f}   |ошибка|={abs(truth - y_real):.4f}")
    else:
        print()


def noise_experiment(x, p, n_hidden, noise_share=0.05, epochs_cap=2000):
    rng_noise = np.random.default_rng(123)
    span = x[:TRAIN_LEN].max() - x[:TRAIN_LEN].min()
    x_noisy = x.copy()
    x_noisy[:TRAIN_LEN] += rng_noise.normal(0.0, noise_share * span, size=TRAIN_LEN)
    x_min, x_max = fit_minmax(x_noisy[:TRAIN_LEN])
    xn_noisy = normalize(x_noisy, x_min, x_max)
    xn_clean = normalize(x, x_min, x_max)
    X_tr, E_tr = build_windows(xn_noisy, range(p, TRAIN_LEN), p)
    X_te, E_te = build_windows(xn_clean, range(TRAIN_LEN, len(x)), p)
    X_tr_clean, E_tr_clean = build_windows(xn_clean, range(p, TRAIN_LEN), p)
    res = []
    for s in SEEDS:
        m = MLP(p, n_hidden, LR, W_INIT, s)
        m.fit(X_tr, E_tr, EE, epochs_cap, np.random.default_rng(s))
        tr_noisy_mae = mae(denormalize(E_tr, x_min, x_max), denormalize(m.predict(X_tr), x_min, x_max))
        tr_clean_mae = mae(denormalize(E_tr_clean, x_min, x_max), denormalize(m.predict(X_tr_clean), x_min, x_max))
        te_mae = mae(denormalize(E_te, x_min, x_max), denormalize(m.predict(X_te), x_min, x_max))
        res.append((tr_noisy_mae, tr_clean_mae, te_mae))
    return np.array(res)


def interactive_mode(model, x, p, x_min, x_max):
    print("\nРежим ввода (выход: q)")
    print(f"Введите номер k ({p}..{len(x) - 1}) либо {p} значений ряда через пробел (от старых к новым).")
    while True:
        raw = input("k или окно: ").strip()
        if not raw or raw.lower() == "q":
            break
        try:
            parts = raw.replace(",", " ").split()
            if len(parts) == 1:
                k = int(parts[0])
                if not (p <= k <= len(x) - 1):
                    print(f"  k должен быть в диапазоне {p}..{len(x) - 1}")
                    continue
                show_window(model, x, k, p, x_min, x_max)
            elif len(parts) == p:
                show_window(model, x, [float(v) for v in parts], p, x_min, x_max)
            else:
                print(f"  Нужен один номер k или ровно {p} значений")
        except ValueError:
            print("  Некорректный ввод")


def main():
    x = generate_series()
    k_all = np.arange(len(x))
    p, n = P_VAR, N_VAR

    print(f"Вариант 11: a={A}, b={B}, c={C}, d={D}, p={p}, N={n}")
    print(f"Ряд: x(k) = {A}*cos({B}*k) + {C}*sin({D}*k), k = 0..{len(x) - 1}")
    print(f"min |x(k)| = {np.abs(x).min():.4f}; знак ряда меняется, поэтому MAPE неприменим (деление на ~0), используется sMAPE")

    x_min, x_max = fit_minmax(x[:TRAIN_LEN])
    print(f"\nНормализация по обучающему участку: x_min={x_min:.4f}, x_max={x_max:.4f}")
    xn_all = normalize(x, x_min, x_max)
    print(f"Нормализованный тестовый участок: min={xn_all[TRAIN_LEN:].min():.4f}, max={xn_all[TRAIN_LEN:].max():.4f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(k_all, x, color="#2563eb", linewidth=1.5, label="x(k)")
    ax.axvline(TRAIN_LEN - 0.5, color="#dc2626", linestyle="--", label="граница train/test")
    ax.axvspan(TRAIN_LEN - 0.5, len(x) - 0.5, color="#fde68a", alpha=0.35, label="тестовый участок")
    ax.set_xlabel("k")
    ax.set_ylabel("x(k)")
    ax.set_title("Ряд x(k) = 1.4 cos(0.38k) + 0.7 sin(0.85k)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig("series.png", dpi=150)
    plt.close(fig)

    print("\nПодбор скорости обучения η (p=6, N=5, seed 1..5)")
    print(f"{'η':>6s}{'эпохи (ср.)':>14s}{'не сошлось':>12s}{'MAE test':>12s}")
    for lr in [0.01, 0.03, 0.1, 0.3]:
        rs = run_series(x, p, n, lr=lr)
        s = summarize(rs)
        mark = "  (выбрано)" if lr == LR else ""
        print(f"{lr:>6.2f}{s['ep_mean']:>14.1f}{s['failed']:>10d}/5{s['te_mean']:>12.4f}{mark}")

    print("\nБазовый эксперимент (p=6, N=5, η=%.2f, seed=1)" % LR)
    base = train_one(x, p, n, seed=1)
    print(f"Примеров: обучающих={base['n_train']}, тестовых={base['n_test']}")
    print(f"Эпох: {base['epochs']}, Es={base['es']:.3e}, критерий Es<=1e-4: {'достигнут' if base['converged'] else 'НЕ достигнут'}")
    print(f"{'':8s}{'MAE':>10s}{'RMSE':>10s}{'sMAPE,%':>10s}")
    for name in ("train", "test"):
        m = base[name]
        print(f"{name:8s}{m['mae']:>10.4f}{m['rmse']:>10.4f}{m['smape']:>10.3f}")

    window_ps = [2, 4, P_VAR, 2 * P_VAR]
    runs_p = {pp: run_series(x, pp, n) for pp in window_ps}
    print_table(f"Исследование окна p (N={n}, 5 запусков)", "p", [(pp, runs_p[pp]) for pp in window_ps])

    hidden_ns = [1, 2, N_VAR, 2 * N_VAR]
    runs_n = {nn: run_series(x, p, nn) for nn in hidden_ns}
    print_table(f"Исследование скрытого слоя N (p={p}, 5 запусков)", "N", [(nn, runs_n[nn]) for nn in hidden_ns])

    candidates = [(f"p={pp}, N={n}", runs_p[pp]) for pp in window_ps]
    candidates += [(f"p={p}, N={nn}", runs_n[nn]) for nn in hidden_ns if nn != n]
    best_name, best_runs = min(candidates, key=lambda c: summarize(c[1])["te_mean"])
    best = min(best_runs, key=lambda r: r["test"]["mae"])
    print(f"\nЛучшая по среднему тестовому MAE конфигурация: {best_name} "
          f"(MAE test={summarize(best_runs)['te_mean']:.4f}); для графиков взят запуск seed={best['seed']}")

    pb, nb = best["p"], best["n"]
    xn_b = normalize(x, best["x_min"], best["x_max"])
    preds_n = recursive_forecast(best["model"], xn_b, pb, TRAIN_LEN, FORECAST_STEPS)
    preds = denormalize(preds_n, best["x_min"], best["x_max"])
    truth = x[TRAIN_LEN:TRAIN_LEN + FORECAST_STEPS]
    abs_err = np.abs(truth - preds)
    span_full = float(x.max() - x.min())
    thr = THRESHOLD_SHARE * span_full
    exceed = np.where(abs_err > thr)[0]
    print(f"\nРекурсивный прогноз на {FORECAST_STEPS} шагов с k={TRAIN_LEN} (p={pb}, N={nb})")
    print(f"Размах ряда={span_full:.4f}, порог 5% = {thr:.4f}")
    print(f"{'шаг':>5s}{'k':>6s}{'эталон':>10s}{'прогноз':>10s}{'|ошибка|':>10s}")
    for i in range(FORECAST_STEPS):
        print(f"{i + 1:>5d}{TRAIN_LEN + i:>6d}{truth[i]:>10.4f}{preds[i]:>10.4f}{abs_err[i]:>10.4f}")
    if len(exceed):
        print(f"Ошибка впервые превышает 5% размаха на шаге {exceed[0] + 1} (k={TRAIN_LEN + exceed[0]})")
    else:
        print(f"Ошибка не превышает 5% размаха за все {FORECAST_STEPS} шагов (максимум {abs_err.max():.4f})")
    one_step_te = best["test"]["mae"]
    print(f"Для сравнения: одношаговый MAE на тесте = {one_step_te:.4f}, средняя ошибка рекурсивного прогноза = {abs_err.mean():.4f}")

    full_pred = np.full(len(x), np.nan)
    xn_b_all, xmin_b, xmax_b, Xtr_b, Etr_b, Xte_b, Ete_b = prepare_data(x, pb)
    full_pred[pb:TRAIN_LEN] = denormalize(best["model"].predict(Xtr_b), xmin_b, xmax_b)
    full_pred[TRAIN_LEN:] = denormalize(best["model"].predict(Xte_b), xmin_b, xmax_b)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(k_all, x, color="#2563eb", linewidth=1.5, label="эталон x(k)")
    ax.plot(k_all, full_pred, color="#f97316", linewidth=1.2, linestyle="--", label="одношаговый прогноз сети")
    ax.axvline(TRAIN_LEN - 0.5, color="#dc2626", linestyle=":", label="граница train/test")
    ax.set_xlabel("k")
    ax.set_ylabel("x(k)")
    ax.set_title(f"Одношаговый прогноз на обучающем и тестовом участках (p={pb}, N={nb})")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig("onestep.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(base["history"], label=f"базовая p={p}, N={n}", color="#2563eb", linewidth=1.6)
    for pp, col in ((2, "#dc2626"), (2 * P_VAR, "#16a34a")):
        ax.plot(runs_p[pp][0]["history"], label=f"p={pp}, N={n}", color=col, linewidth=1.4)
    ax.axhline(EE, color="gray", linestyle="--", linewidth=0.9, label="порог Ee=1e-4")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Эпоха (логарифмическая шкала)")
    ax.set_ylabel("Es")
    ax.set_title("Сходимость: базовая конфигурация и две конфигурации из исследования окна")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig("convergence.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(best["y_te"], best["p_te"], s=22, color="#2563eb", edgecolor="black", linewidth=0.4)
    lim = [min(best["y_te"].min(), best["p_te"].min()) - 0.1, max(best["y_te"].max(), best["p_te"].max()) + 0.1]
    ax.plot(lim, lim, color="#dc2626", linestyle="--", label="y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("эталон")
    ax.set_ylabel("прогноз")
    ax.set_title(f"Тестовый участок: эталон vs прогноз (p={pb}, N={nb})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("scatter.png", dpi=150)
    plt.close(fig)

    steps = np.arange(1, FORECAST_STEPS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(steps, truth, color="#2563eb", marker="o", markersize=3, label="эталон")
    axes[0].plot(steps, preds, color="#f97316", marker="s", markersize=3, linestyle="--", label="рекурсивный прогноз")
    axes[0].set_xlabel("шаг прогноза")
    axes[0].set_ylabel("x")
    axes[0].set_title("Рекурсивный прогноз на 30 шагов")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    axes[1].plot(steps, abs_err, color="#dc2626", marker="o", markersize=3, label="|ошибка|")
    axes[1].axhline(thr, color="gray", linestyle="--", label="5% размаха ряда")
    axes[1].set_xlabel("шаг прогноза")
    axes[1].set_ylabel("абсолютная ошибка")
    axes[1].set_title("Рост ошибки по шагам")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig("recursive.png", dpi=150)
    plt.close(fig)

    print("\nЭксперимент с шумом (обучение на ряде + N(0, 5% размаха), тест на чистом ряде, 2000 эпох)")
    clean_te = summarize(run_series(x, p, n))["te_mean"]
    print(f"Без шума (p={p}, N={n}): тестовый MAE = {clean_te:.4f}")
    print(f"{'N':>4s}{'MAE train (с шумом)':>22s}{'MAE train (чистый)':>22s}{'MAE test (чистый)':>20s}")
    for nn in (n, 2 * n):
        r = noise_experiment(x, p, nn)
        print(f"{nn:>4d}{r[:, 0].mean():>22.4f}{r[:, 1].mean():>22.4f}{r[:, 2].mean():>20.4f}")

    model = base["model"]
    print(f"\nРежим функционирования (p={p}, N={n})")
    print("Окна из обучающего участка:")
    for k in (10, 70, 130):
        show_window(model, x, k, p, base["x_min"], base["x_max"], "train")
    print("Окна из тестового участка:")
    for k in (140, 170, 199):
        show_window(model, x, k, p, base["x_min"], base["x_max"], "test")
    print("Пользовательские окна:")
    custom = np.round(x[100:100 + p], 2)
    show_window(model, x, custom, p, base["x_min"], base["x_max"], "пользовательское (округлённый кусок ряда)")
    outlier = x[100:100 + p].copy()
    outlier[3] = 6.0
    show_window(model, x, outlier, p, base["x_min"], base["x_max"], "с выбросом (4-е значение заменено на 6.0)")
    true_next = x[100 + p]
    print(f"    (для сравнения: настоящее следующее значение ряда x({100 + p}) = {true_next:.4f})")

    interactive_mode(model, x, p, base["x_min"], base["x_max"])


if __name__ == "__main__":
    main()