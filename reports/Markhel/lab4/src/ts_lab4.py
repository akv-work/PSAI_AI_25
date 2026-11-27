import json
import os
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

A, B, C, D = 1.2, 0.28, 0.6, 0.65
P_VAR, N_VAR = 6, 4
N_POINTS = 200
K_SPLIT = 140                 # k = 0..139 - обучение, k = 140..199 - тест

LR = 0.3                      # выбрано по результатам подбора (--extra)
INIT = 0.5                    # веса и пороги из U[-INIT; INIT]
EE = 1e-4
MAX_EPOCHS = 20000
SEEDS = [1, 2, 3, 4, 5]
HORIZON = 30

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
EXTRA_FILE = os.path.join(RESULTS_DIR, "extra_stats.json")


# ----------------------------------------------------------------------------------------
# Данные
# ----------------------------------------------------------------------------------------
def generate_series(n=N_POINTS):
    k = np.arange(n)
    return A * np.cos(B * k) + C * np.sin(D * k)


def fit_minmax(train):
    """Параметры нормализации берутся только по обучающему участку."""
    return float(np.min(train)), float(np.max(train))


def normalize(x, x_min, x_max):
    return (np.asarray(x, dtype=float) - x_min) / (x_max - x_min)


def denormalize(x_norm, x_min, x_max):
    return np.asarray(x_norm, dtype=float) * (x_max - x_min) + x_min


def make_windows(series, p, k_from, k_to):
    """Скользящее окно: вход - p значений x(k-p)..x(k-1) (по времени), цель - x(k), k = k_from..k_to-1."""
    k_from = max(k_from, p)
    X = np.array([series[k - p:k] for k in range(k_from, k_to)])
    y = np.array([series[k] for k in range(k_from, k_to)])
    return X, y, np.arange(k_from, k_to)


# ----------------------------------------------------------------------------------------
# Многослойный персептрон p-N-1 (сразу S независимых сетей - по одной на seed)
# ----------------------------------------------------------------------------------------
def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))


def init_net(p, n_hidden, seeds, init=INIT):
    S = len(seeds)
    net = {"W1": np.zeros((S, p, n_hidden)), "b1": np.zeros((S, n_hidden)),
           "W2": np.zeros((S, n_hidden)), "b2": np.zeros(S)}
    for i, s in enumerate(seeds):
        rng = np.random.default_rng(s)
        net["W1"][i] = rng.uniform(-init, init, (p, n_hidden))
        net["b1"][i] = rng.uniform(-init, init, n_hidden)
        net["W2"][i] = rng.uniform(-init, init, n_hidden)
        net["b2"][i] = rng.uniform(-init, init)
    return net


def take(net, idx):
    return {name: value[idx] for name, value in net.items()}


def forward(net, x):
    """Прямой проход: x (S, p) - по одному примеру на каждую из S сетей.
    h_j = σ(Σ_i w1_ij·x_i + b1_j),  y = Σ_j w2_j·h_j + b2 (линейный выход)."""
    h = sigmoid(np.einsum("si,sij->sj", x, net["W1"]) + net["b1"])
    y = np.einsum("sj,sj->s", h, net["W2"]) + net["b2"]
    return h, y


def forward_all(net, X):
    """Прямой проход всей выборки X (L, p) через каждую из S сетей: выход (S, L)."""
    h = sigmoid(np.einsum("li,sij->slj", X, net["W1"]) + net["b1"][:, None, :])
    return np.einsum("slj,sj->sl", h, net["W2"]) + net["b2"][:, None]


def backward(net, x, h, y, d, lr):
    """Обратный проход для одного примера на сеть (онлайн-режим), E = ½(d − y)²:
       δ_out = d − y                         (линейный выход, f'(s) = 1)
       δ_j   = δ_out · w2_j · h_j(1 − h_j)   (сигмоидный скрытый слой)
       W2 += η·δ_out·h,  b2 += η·δ_out,  W1 += η·x ⊗ δ,  b1 += η·δ"""
    d_out = d - y
    d_hid = d_out[:, None] * net["W2"] * h * (1.0 - h)
    net["W2"] += lr * d_out[:, None] * h
    net["b2"] += lr * d_out
    net["W1"] += lr * x[:, :, None] * d_hid[:, None, :]
    net["b1"] += lr * d_hid


def mse(net, X, y):
    return np.mean((forward_all(net, X) - y) ** 2, axis=1)


def train(X, y, n_hidden, seeds, lr=LR, ee=EE, max_epochs=MAX_EPOCHS, noise=0.0):
    """Онлайн-обучение S сетей (по одной на seed); у каждой сети свой порядок примеров в каждой эпохе.
    Es - средняя квадратичная ошибка по всей обучающей выборке после эпохи; остановка при Es <= Ee.
    noise > 0 - к входам при обучении добавляется гауссов шум (эксперимент --extra)."""
    p = X.shape[1]
    net = init_net(p, n_hidden, seeds)
    rngs = [np.random.default_rng(s) for s in seeds]
    noise_rng = np.random.default_rng(12345)
    S, L = len(seeds), len(y)
    epochs = np.full(S, max_epochs)
    active = np.ones(S, dtype=bool)
    history = [[] for _ in range(S)]
    for epoch in range(1, max_epochs + 1):
        idx = np.flatnonzero(active)
        if len(idx) == 0:
            break
        sub = take(net, idx)
        order = np.array([rngs[i].permutation(L) for i in idx])
        for step in range(L):
            x = X[order[:, step]]
            if noise:
                x = x + noise * noise_rng.standard_normal(x.shape)
            h, out = forward(sub, x)
            backward(sub, x, h, out, y[order[:, step]], lr)
        for name in net:
            net[name][idx] = sub[name]
        es = mse(net, X, y)
        for i in idx:
            history[i].append(float(es[i]))
        done = active & (es <= ee)
        epochs[done] = epoch
        active &= ~done
    return net, epochs, mse(net, X, y), history


def single(net, i):
    return {name: value[i:i + 1] for name, value in net.items()}


def predict(net, X):
    """Прогноз одной сети (S = 1) для выборки X: (L, p) -> (L,)."""
    return forward_all(net, np.atleast_2d(X))[0]


# ----------------------------------------------------------------------------------------
# Метрики (в исходной шкале)
# ----------------------------------------------------------------------------------------
def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    return float(100.0 * np.mean(np.abs((y_true - y_pred) / y_true)))


def smape(y_true, y_pred):
    return float(100.0 * np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))))


def metrics(y_true, y_pred):
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred), "sMAPE": smape(y_true, y_pred),
            "MAPE": mape(y_true, y_pred)}


# ----------------------------------------------------------------------------------------
# Эксперименты
# ----------------------------------------------------------------------------------------
class Data:
    def __init__(self):
        self.k = np.arange(N_POINTS)
        self.x = generate_series()
        self.x_min, self.x_max = fit_minmax(self.x[:K_SPLIT])
        self.xn = normalize(self.x, self.x_min, self.x_max)

    def windows(self, p):
        Xtr, ytr, ktr = make_windows(self.xn, p, 0, K_SPLIT)
        Xte, yte, kte = make_windows(self.xn, p, K_SPLIT, N_POINTS)
        return Xtr, ytr, ktr, Xte, yte, kte

    def to_real(self, v):
        return denormalize(v, self.x_min, self.x_max)


def run_config(data, p, n_hidden, seeds=SEEDS, lr=LR, noise=0.0):
    Xtr, ytr, ktr, Xte, yte, kte = data.windows(p)
    net, epochs, es, history = train(Xtr, ytr, n_hidden, seeds, lr=lr, noise=noise)
    runs = []
    for i, s in enumerate(seeds):
        one = single(net, i)
        pred_tr = data.to_real(predict(one, Xtr))
        pred_te = data.to_real(predict(one, Xte))
        runs.append({"seed": s, "net": one, "epochs": int(epochs[i]), "Es": float(es[i]),
                     "converged": bool(es[i] <= EE), "history": history[i],
                     "train": metrics(data.x[ktr], pred_tr), "test": metrics(data.x[kte], pred_te),
                     "pred_tr": pred_tr, "pred_te": pred_te, "ktr": ktr, "kte": kte})
    return {"p": p, "N": n_hidden, "lr": lr, "noise": noise, "runs": runs}


def summary(cfg):
    ep = np.array([r["epochs"] for r in cfg["runs"]])
    te = np.array([r["test"]["MAE"] for r in cfg["runs"]])
    tr = np.array([r["train"]["MAE"] for r in cfg["runs"]])
    return {"p": cfg["p"], "N": cfg["N"], "lr": cfg["lr"], "noise": cfg["noise"],
            "ep_mean": float(ep.mean()), "ep_std": float(ep.std()),
            "ep_min": int(ep.min()), "ep_max": int(ep.max()),
            "mae_mean": float(te.mean()), "mae_std": float(te.std()), "train_mae_mean": float(tr.mean()),
            "failed": int(sum(not r["converged"] for r in cfg["runs"])), "n": len(cfg["runs"])}


def recursive_forecast(net, data, p, k_start=K_SPLIT, horizon=HORIZON):
    """Рекурсивный прогноз: окно начинается с известных значений x(k_start-p..k_start-1),
    каждое новое предсказание добавляется в окно вместо самого старого значения."""
    window = list(data.xn[k_start - p:k_start])
    preds = []
    for _ in range(horizon):
        y = float(predict(net, np.array(window))[0])
        preds.append(y)
        window = window[1:] + [y]
    return data.to_real(np.array(preds))


def print_base(data, run, p, n_hidden):
    print(f"=== Базовый эксперимент: p = {p}, N = {n_hidden}, η = {LR}, seed = {run['seed']} ===")
    print(f"Эпох: {run['epochs']}  |  критерий Es <= {EE:g} достигнут: {'да' if run['converged'] else 'нет'}")
    print(f"Итоговая Es (нормализованная шкала) = {run['Es']:.3e}")
    print(f"{'участок':<10}{'MAE':>9}{'RMSE':>9}{'sMAPE, %':>10}{'MAPE, %':>10}")
    for part in ("train", "test"):
        m = run[part]
        print(f"{'обучение' if part == 'train' else 'тест':<10}{m['MAE']:>9.4f}{m['RMSE']:>9.4f}"
              f"{m['sMAPE']:>10.2f}{m['MAPE']:>10.1f}")
    print()


def print_study(title, rows, key):
    print(f"=== {title} (по {rows[0]['n']} запусков, seed {SEEDS[0]}-{SEEDS[-1]}) ===")
    print(f"{key:>4}{'эпох: среднее ± std':>24}{'мин-макс':>14}{'MAE тест: среднее ± std':>28}"
          f"{'MAE обуч.':>11}{'не достигли Ee':>16}")
    for r in rows:
        print(f"{r[key]:>4}{r['ep_mean']:>15.0f} ± {r['ep_std']:<6.0f}{r['ep_min']:>7}-{r['ep_max']:<6}"
              f"{r['mae_mean']:>19.4f} ± {r['mae_std']:<6.4f}{r['train_mae_mean']:>11.4f}"
              f"{r['failed']:>10} из {r['n']}")
    print()


# ----------------------------------------------------------------------------------------
# Дополнительное исследование (--extra): подбор η и обучение с шумом на входах
# ----------------------------------------------------------------------------------------
EXTRA_LRS = [0.01, 0.03, 0.1, 0.3]
EXTRA_NOISE = [0.0, 0.005, 0.01, 0.02]


def extra_study(data):
    out = {"lr": [], "noise": []}
    for lr in EXTRA_LRS:
        out["lr"].append(summary(run_config(data, P_VAR, N_VAR, lr=lr)))
        print(f"  η = {lr}: готово", flush=True)
    for nz in EXTRA_NOISE:
        out["noise"].append(summary(run_config(data, P_VAR, N_VAR, noise=nz)))
        print(f"  шум σ = {nz}: готово", flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(EXTRA_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


def load_extra():
    if os.path.exists(EXTRA_FILE):
        with open(EXTRA_FILE, encoding="utf-8") as f:
            return json.load(f)
    return None


def print_extra(extra):
    print_study(f"Подбор скорости обучения (p = {P_VAR}, N = {N_VAR})", extra["lr"], "lr")
    print("=== Шум на входах при обучении (σ в нормализованной шкале), p = 6, N = 4, η = 0.3 ===")
    print(f"{'σ':>6}{'эпох: среднее ± std':>24}{'MAE тест':>12}{'MAE обуч.':>11}{'не достигли Ee':>16}")
    for r in extra["noise"]:
        print(f"{r['noise']:>6}{r['ep_mean']:>15.0f} ± {r['ep_std']:<6.0f}{r['mae_mean']:>12.4f}"
              f"{r['train_mae_mean']:>11.4f}{r['failed']:>10} из {r['n']}")
    print()


# ----------------------------------------------------------------------------------------
# Режим функционирования
# ----------------------------------------------------------------------------------------
def functioning(net, data, p, values=None, k=None):
    """Прогноз по окну: либо p значений в исходной шкале, либо номер k (окно x(k-p)..x(k-1) из ряда)."""
    if k is not None:
        values = data.x[k - p:k]
    values = np.asarray(values, dtype=float)
    x_norm = normalize(values, data.x_min, data.x_max)
    y_norm = float(predict(net, x_norm)[0])
    y_real = float(data.to_real(y_norm))
    true = float(data.x[k]) if k is not None and k < N_POINTS else None
    return x_norm, y_norm, y_real, true


def print_functioning(net, data, p, k=None, values=None, label=""):
    x_norm, y_norm, y_real, true = functioning(net, data, p, values=values, k=k)
    head = f"k = {k}" if k is not None else "пользовательское окно"
    print(f"--- {head} {label}".rstrip())
    if k is None:
        print(f"  входы:               {' '.join(f'{v:.3f}' for v in values)}")
    print(f"  нормализованные входы: {' '.join(f'{v:.4f}' for v in x_norm)}")
    print(f"  прогноз (норм.)   = {y_norm:.4f}")
    print(f"  прогноз (исходн.) = {y_real:.4f}")
    if true is not None:
        print(f"  истинное x({k})   = {true:.4f}   |ошибка| = {abs(true - y_real):.4f}")


def interactive_loop(net, data, p):
    print("=== Интерактивный режим ===")
    print(f"Введите номер k ({p}..{N_POINTS - 1}) - окно x(k-{p})..x(k-1) берётся из ряда,")
    print(f"или {p} значений ряда в исходной шкале через пробел (от старого к новому). Выход - 'q'.")
    while True:
        try:
            raw = input("> ").strip()
        except EOFError:
            break
        if not sys.stdin.isatty():
            print(raw)
        if raw.lower() in ("q", "quit", "exit"):
            break
        parts = raw.replace(",", " ").split()
        try:
            nums = [float(v) for v in parts]
        except ValueError:
            print("Не удалось распознать числа.")
            continue
        if len(nums) == 1 and nums[0].is_integer() and p <= nums[0] < N_POINTS:
            print_functioning(net, data, p, k=int(nums[0]))
        elif len(nums) == p:
            print_functioning(net, data, p, values=nums)
        else:
            print(f"Нужно ввести номер k из [{p}; {N_POINTS - 1}] или ровно {p} чисел.")


# ----------------------------------------------------------------------------------------
# Визуализация
# ----------------------------------------------------------------------------------------
COLORS = {"true": "black", "train": "#1f77b4", "test": "#d62728"}


def fig_series(data, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(data.k, data.x, color="black", lw=1.4, marker="o", ms=2.5, label="x(k)")
    ax.axvline(K_SPLIT - 0.5, color="gray", ls="--", lw=1.5)
    ax.axvspan(-0.5, K_SPLIT - 0.5, color="#1f77b4", alpha=0.06, label="обучающий участок (k = 0..139)")
    ax.axvspan(K_SPLIT - 0.5, N_POINTS - 0.5, color="#d62728", alpha=0.06, label="тестовый участок (k = 140..199)")
    for v, txt in ((data.x_min, "min(train)"), (data.x_max, "max(train)")):
        ax.axhline(v, color="#1f77b4", ls=":", lw=1)
        ax.text(N_POINTS - 1, v, txt, ha="right", va="bottom", fontsize=8, color="#1f77b4")
    ax.set_xlim(-1, N_POINTS)
    ax.set_xlabel("k")
    ax.set_ylabel("x(k)")
    ax.set_title(f"Ряд x(k) = {A}·cos({B}·k) + {C}·sin({D}·k)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def fig_one_step(data, run, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(data.k, data.x, color="black", lw=2.2, label="эталон x(k)")
    ax.plot(run["ktr"], run["pred_tr"], color=COLORS["train"], lw=1.2, ls="--", marker=".", ms=4,
            label=f"прогноз на обучении (MAE = {run['train']['MAE']:.4f})")
    ax.plot(run["kte"], run["pred_te"], color=COLORS["test"], lw=1.2, ls="--", marker=".", ms=4,
            label=f"прогноз на тесте (MAE = {run['test']['MAE']:.4f})")
    ax.axvline(K_SPLIT - 0.5, color="gray", ls="--", lw=1.5)
    ax.text(K_SPLIT + 1, ax.get_ylim()[1] * 0.92, "train | test", color="gray")
    ax.set_xlabel("k")
    ax.set_ylabel("x(k)")
    ax.set_title(f"Одношаговый прогноз сети {P_VAR}-{N_VAR}-1 на обучающем и тестовом участках")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def fig_convergence(curves, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, run, color in curves:
        ep = np.arange(1, len(run["history"]) + 1)
        ax.plot(ep, run["history"], color=color, lw=2,
                label=f"{label}: {run['epochs']} эпох" + ("" if run["converged"] else " (Ee не достигнута)"))
    ax.axhline(EE, color="black", ls=":", lw=1.3, label=f"Ee = {EE:g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Номер эпохи (лог. шкала)")
    ax.set_ylabel("Es - средняя ошибка (норм. шкала, лог.)")
    ax.set_title(f"Сходимость при разных размерах окна (N = {N_VAR}, η = {LR}, seed = 1)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def fig_scatter(run, data, path):
    import matplotlib.pyplot as plt
    true = data.x[run["kte"]]
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.scatter(true, run["pred_te"], color=COLORS["test"], s=28, edgecolors="black", linewidths=0.4,
               label="тестовые примеры")
    lim = [min(true.min(), run["pred_te"].min()) - 0.1, max(true.max(), run["pred_te"].max()) + 0.1]
    ax.plot(lim, lim, color="black", lw=1.2, label="y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("эталон x(k)")
    ax.set_ylabel("прогноз сети")
    ax.set_title(f"Тест: эталон - прогноз (MAE = {run['test']['MAE']:.4f})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def fig_recursive(data, rec, one_step, best, threshold, step_bad, path):
    import matplotlib.pyplot as plt
    k = np.arange(K_SPLIT, K_SPLIT + HORIZON)
    true = data.x[k]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(np.arange(K_SPLIT - 20, K_SPLIT + HORIZON), data.x[K_SPLIT - 20:K_SPLIT + HORIZON], color="black",
             lw=2, label="эталон x(k)")
    ax1.plot(k, rec, color="#d62728", lw=1.8, ls="--", marker="o", ms=3.5, label="рекурсивный прогноз")
    ax1.plot(k, one_step, color="#1f77b4", lw=1, ls=":", marker=".", ms=3, label="одношаговый прогноз")
    ax1.axvline(K_SPLIT - 0.5, color="gray", ls="--")
    ax1.set_xlabel("k")
    ax1.set_ylabel("x(k)")
    ax1.set_title(f"Прогноз на {HORIZON} шагов от k = {K_SPLIT} (p = {best['p']}, N = {best['N']})")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    steps = np.arange(1, HORIZON + 1)
    ax2.plot(steps, np.abs(true - rec), color="#d62728", marker="o", ms=3.5, label="рекурсивный")
    ax2.plot(steps, np.abs(true - one_step), color="#1f77b4", marker=".", ms=3, label="одношаговый")
    ax2.axhline(threshold, color="black", ls="--", lw=1.2, label=f"5 % размаха ряда = {threshold:.3f}")
    if step_bad is not None:
        ax2.axvline(step_bad, color="gray", ls=":", label=f"превышение на шаге {step_bad}")
    ax2.set_yscale("log")
    ax2.set_xlabel("номер шага прогноза")
    ax2.set_ylabel("|ошибка| (лог. шкала)")
    ax2.set_title("Абсолютная ошибка по шагам")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def fig_studies(p_rows, n_rows, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
    for ax, rows, key, title in ((axes[0], p_rows, "p", f"Размер окна p (N = {N_VAR})"),
                                 (axes[1], n_rows, "N", f"Число скрытых нейронов N (p = {P_VAR})")):
        x = np.arange(len(rows))
        means = [r["mae_mean"] for r in rows]
        bars = ax.bar(x, means, 0.55, yerr=[r["mae_std"] for r in rows], capsize=5, color="#d62728", alpha=0.8,
                      label="MAE на тесте (среднее ± std)")
        for b, r in zip(bars, rows):
            txt = f"{r['mae_mean']:.4f}" + (f"\nне сошлись:\n{r['failed']} из {r['n']}" if r["failed"] else "")
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() / 2, txt, ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold")
        ax.set_ylim(0, max(means) * 1.15)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{key} = {r[key]}" for r in rows])
        ax.set_ylabel("MAE на тесте (исходная шкала)")
        ax2 = ax.twinx()
        ax2.plot(x, [r["ep_mean"] for r in rows], color="#1f77b4", marker="o", lw=2, label="эпох до остановки (среднее)")
        if key == "p":
            ax2.set_yscale("log")
            ax2.set_ylabel("Эпох (лог. шкала)", color="#1f77b4")
        else:
            ax2.set_ylim(0, max(r["ep_mean"] for r in rows) * 1.3)
            ax2.set_ylabel("Эпох", color="#1f77b4")
        ax.set_title(title)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.savefig(path, dpi=150)


# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    show = "--no-show" not in sys.argv
    if not show:
        import matplotlib
        matplotlib.use("Agg")
    os.makedirs(FIG_DIR, exist_ok=True)

    data = Data()
    print(f"Ряд: x(k) = {A}·cos({B}·k) + {C}·sin({D}·k), k = 0..{N_POINTS - 1};  p = {P_VAR}, N = {N_VAR}")
    print(f"Нормализация по обучающему участку: x_min = {data.x_min:.4f}, x_max = {data.x_max:.4f}")
    print(f"Весь ряд: min = {data.x.min():.4f}, max = {data.x.max():.4f};  тест в норм. шкале: "
          f"[{data.xn[K_SPLIT:].min():.4f}; {data.xn[K_SPLIT:].max():.4f}]")
    Xtr, ytr, _, Xte, yte, _ = data.windows(P_VAR)
    print(f"Скользящее окно p = {P_VAR}: обучающих примеров {len(ytr)} (k = {P_VAR}..{K_SPLIT - 1}), "
          f"тестовых {len(yte)} (k = {K_SPLIT}..{N_POINTS - 1})")
    print(f"η = {LR}, веса из U[-{INIT}; {INIT}], Ee = {EE:g}, максимум {MAX_EPOCHS} эпох")
    print()

    p_values = [2, 4, P_VAR, 2 * P_VAR]
    n_values = [1, 2, N_VAR, 2 * N_VAR]
    configs = {}
    for p in p_values:
        configs[(p, N_VAR)] = run_config(data, p, N_VAR)
    for n in n_values:
        if (P_VAR, n) not in configs:
            configs[(P_VAR, n)] = run_config(data, P_VAR, n)

    base = configs[(P_VAR, N_VAR)]["runs"][0]
    print_base(data, base, P_VAR, N_VAR)

    p_rows = [summary(configs[(p, N_VAR)]) for p in p_values]
    n_rows = [summary(configs[(P_VAR, n)]) for n in n_values]
    print_study(f"Исследование окна p (N = {N_VAR})", p_rows, "p")
    print_study(f"Исследование скрытого слоя N (p = {P_VAR})", n_rows, "N")

    best = min((summary(c) for c in configs.values()), key=lambda r: r["mae_mean"])
    best_cfg = configs[(best["p"], best["N"])]
    best_run = sorted(best_cfg["runs"], key=lambda r: r["test"]["MAE"])[len(best_cfg["runs"]) // 2]
    rec = recursive_forecast(best_run["net"], data, best["p"])
    true = data.x[K_SPLIT:K_SPLIT + HORIZON]
    one_step = best_run["pred_te"][:HORIZON]
    err = np.abs(true - rec)
    span = float(data.x.max() - data.x.min())
    threshold = 0.05 * span
    bad = np.flatnonzero(err > threshold)
    step_bad = int(bad[0]) + 1 if len(bad) else None
    print(f"=== Рекурсивный прогноз на {HORIZON} шагов от k = {K_SPLIT}: лучшая конфигурация p = {best['p']}, "
          f"N = {best['N']} (seed {best_run['seed']}) ===")
    print(f"{'шаг':>4}{'k':>5}{'эталон':>10}{'рекурсивный':>13}{'|ошибка|':>10}{'одношаговый':>13}{'|ошибка|':>10}")
    for s in range(HORIZON):
        print(f"{s + 1:>4}{K_SPLIT + s:>5}{true[s]:>10.4f}{rec[s]:>13.4f}{err[s]:>10.4f}"
              f"{one_step[s]:>13.4f}{abs(true[s] - one_step[s]):>10.4f}")
    print(f"Порог 5 % размаха ряда: 0.05·{span:.4f} = {threshold:.4f};  "
          + (f"превышен впервые на шаге {step_bad}" if step_bad else f"не превышен за {HORIZON} шагов"))
    print(f"MAE рекурсивного прогноза = {err.mean():.4f}, одношагового на тех же k = "
          f"{np.mean(np.abs(true - one_step)):.4f};  макс. |ошибка| рекурсивного = {err.max():.4f}")
    full = N_POINTS - K_SPLIT
    err_full = np.abs(data.x[K_SPLIT:] - recursive_forecast(best_run["net"], data, best["p"], horizon=full))
    bad_full = np.flatnonzero(err_full > threshold)
    print(f"Продолжение рекурсии до конца теста ({full} шагов): макс. |ошибка| = {err_full.max():.4f}, "
          f"MAE = {err_full.mean():.4f}; " + (f"порог 5 % впервые превышен на шаге {bad_full[0] + 1}"
                                             if len(bad_full) else "порог 5 % не превышен"))
    print()

    extra = extra_study(data) if "--extra" in sys.argv else load_extra()
    if extra is not None:
        print_extra(extra)

    paths = {"series": "fig1_series.png", "one_step": "fig2_one_step.png", "convergence": "fig3_convergence.png",
             "scatter": "fig4_scatter.png", "recursive": "fig5_recursive.png", "studies": "fig6_studies.png"}
    paths = {k: os.path.join(FIG_DIR, v) for k, v in paths.items()}
    fig_series(data, paths["series"])
    fig_one_step(data, base, paths["one_step"])
    fig_convergence([(f"p = {P_VAR} (базовая)", base, "#1f77b4"),
                     ("p = 2", configs[(2, N_VAR)]["runs"][0], "#d62728"),
                     (f"p = {2 * P_VAR}", configs[(2 * P_VAR, N_VAR)]["runs"][0], "#2ca02c")],
                    paths["convergence"])
    fig_scatter(base, data, paths["scatter"])
    fig_recursive(data, rec, one_step, best, threshold, step_bad, paths["recursive"])
    fig_studies(p_rows, n_rows, paths["studies"])
    print("Графики сохранены:")
    for v in paths.values():
        print("  ", v)
    print()

    print(f"=== Режим функционирования (базовая сеть p = {P_VAR}, N = {N_VAR}) ===")
    for k in (20, 70, 120):
        print_functioning(base["net"], data, P_VAR, k=k, label="(обучающий участок)")
    for k in (150, 175, 199):
        print_functioning(base["net"], data, P_VAR, k=k, label="(тестовый участок)")
    custom = [round(float(v), 3) for v in generate_series(260)[250:256]]
    print_functioning(base["net"], data, P_VAR, values=custom, label="(значения формулы при k = 250..255; x(256) = "
                      f"{generate_series(257)[256]:.4f})")
    outlier = list(data.x[164:170].round(3))
    outlier[3] = 3.0
    print_functioning(base["net"], data, P_VAR, values=outlier, label="(окно k = 170 с выбросом 3.0 на 4-й позиции; "
                      f"истинное x(170) = {data.x[170]:.4f})")
    print()

    if "--no-input" not in sys.argv:
        interactive_loop(base["net"], data, P_VAR)

    if show:
        import matplotlib.pyplot as plt
        plt.show()
