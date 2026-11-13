import numpy as np
import matplotlib.pyplot as plt

A, B, C, D = 1.1, 0.42, 0.55, 0.95
P_VAR, N_VAR = 6, 4
N_POINTS, N_TRAIN = 200, 140
EE, MAX_EPOCHS, ALPHA = 1e-4, 20000, 0.1
SEEDS = [0, 1, 2, 3, 4]


def target(k):
    return A * np.cos(B * k) + C * np.sin(D * k)


def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-np.clip(s, -500, 500)))


def normalize(v, vmin, vmax):
    return (v - vmin) / (vmax - vmin)


def denormalize(v, vmin, vmax):
    return v * (vmax - vmin) + vmin


def make_dataset(seq, p):
    X = np.array([seq[i: i + p] for i in range(len(seq) - p)])
    e = seq[p:]
    return X, e


def split_temporal(X, e, p):
    n = N_TRAIN - p
    return X[:n], e[:n], X[n:], e[n:]


class MLP:
    def __init__(self, n_in, n_hid, seed=0):
        rng = np.random.RandomState(seed)
        self.W1 = rng.uniform(-0.5, 0.5, (n_in, n_hid))
        self.T1 = rng.uniform(-0.5, 0.5, n_hid)
        self.W2 = rng.uniform(-0.5, 0.5, n_hid)
        self.T2 = float(rng.uniform(-0.5, 0.5))

    def forward(self, xk):
        h = sigmoid(xk @ self.W1 - self.T1)
        yv = float(h @ self.W2 - self.T2)
        return h, yv

    def total_error(self, X_n, e_n):
        # средняя по выборке ошибка Es
        h = sigmoid(X_n @ self.W1 - self.T1)
        out = h @ self.W2 - self.T2
        return float(np.mean(0.5 * (out - e_n) ** 2))

    def predict(self, X_n, y_min, y_max):
        h = sigmoid(X_n @ self.W1 - self.T1)
        return denormalize(h @ self.W2 - self.T2, y_min, y_max)


def train(X_n, e_n, n_in, n_hid, alpha=ALPHA, Ee=EE, max_epochs=MAX_EPOCHS, seed=0):
    net = MLP(n_in, n_hid, seed)
    rng = np.random.RandomState(seed + 1000)
    errors = []
    L = len(e_n)

    for epoch in range(max_epochs):
        for k in rng.permutation(L):
            xk = X_n[k]
            ek = float(e_n[k])

            h, yv = net.forward(xk)

            delta_out = yv - ek
            delta_hid = h * (1.0 - h) * (net.W2 * delta_out)

            net.W2 -= alpha * h * delta_out
            net.T2 += alpha * delta_out
            net.W1 -= alpha * np.outer(xk, delta_hid)
            net.T1 += alpha * delta_hid

        Es = net.total_error(X_n, e_n)
        errors.append(Es)
        if Es <= Ee:
            break

    return net, errors


def mae(t, f):
    return float(np.mean(np.abs(t - f)))


def rmse(t, f):
    return float(np.sqrt(np.mean((t - f) ** 2)))


def smape(t, f):
    return float(np.mean(2 * np.abs(t - f) / (np.abs(t) + np.abs(f))) * 100)


def recursive_forecast(net, y_n, p, start, steps):
    window = list(y_n[start - p:start])
    out = []
    for _ in range(steps):
        yv = net.forward(np.array(window))[1]
        out.append(yv)
        window = window[1:] + [yv]
    return np.array(out)


def predict_window(net, values, y_min, y_max, true_value=None):
    xn = normalize(np.asarray(values, dtype=float), y_min, y_max)
    yn = net.forward(xn)[1]
    y = denormalize(yn, y_min, y_max)
    print("  нормализованные входы:", np.round(xn, 4))
    print(f"  прогноз (норм.): {yn:.4f}")
    print(f"  прогноз (исходная шкала): {y:.4f}")
    if true_value is not None:
        print(f"  истина x(k): {true_value:.4f}, абсолютная ошибка: {abs(true_value - y):.4f}")


def get_data(y_n, p):
    X, e = make_dataset(y_n, p)
    return split_temporal(X, e, p)


def get_run(runs, y_n, p, n_hid, seed):
    key = (p, n_hid, seed)
    if key not in runs:
        X_tr, e_tr, _, _ = get_data(y_n, p)
        runs[key] = train(X_tr, e_tr, p, n_hid, seed=seed)
    return runs[key]


def evaluate(net, y_n, p, y_min, y_max):
    X_tr, e_tr, X_te, e_te = get_data(y_n, p)
    res = {}
    for name, X, e in (("train", X_tr, e_tr), ("test", X_te, e_te)):
        t = denormalize(e, y_min, y_max)
        f = net.predict(X, y_min, y_max)
        res[name] = (mae(t, f), rmse(t, f), smape(t, f))
    return res


def study(runs, y_n, y_min, y_max, title, configs):
    print(f"\n{title}")
    print(f"{'p':>3} {'N':>3} | {'эпохи (ср. ± разброс)':>24} | {'test MAE (ср. ± разброс)':>26} | не достигли Ee")
    rows = {}
    for p, n in configs:
        epochs, maes, fails = [], [], 0
        for s in SEEDS:
            net, errs = get_run(runs, y_n, p, n, s)
            epochs.append(len(errs))
            maes.append(evaluate(net, y_n, p, y_min, y_max)["test"][0])
            fails += errs[-1] > EE
        rows[(p, n)] = np.mean(maes)
        print(f"{p:>3} {n:>3} | {np.mean(epochs):>12.0f} ± {np.std(epochs):<9.0f} | "
              f"{np.mean(maes):>12.5f} ± {np.std(maes):<10.5f} | {fails}/{len(SEEDS)}")
    return rows


def plot_all(runs, y_seq, y_n, y_min, y_max, best):
    k = np.arange(N_POINTS)
    net = get_run(runs, y_n, P_VAR, N_VAR, 0)[0]
    X_tr, _, X_te, e_te = get_data(y_n, P_VAR)
    pred_tr = net.predict(X_tr, y_min, y_max)
    pred_te = net.predict(X_te, y_min, y_max)

    plt.figure(figsize=(10, 4))
    plt.plot(k, y_seq, label="x(k)")
    plt.axvline(N_TRAIN - 0.5, color="r", ls="--", label="граница train/test")
    plt.xlabel("k"); plt.ylabel("x"); plt.legend(); plt.grid(alpha=.3)
    plt.title("Исходный ряд")
    plt.savefig("fig1_series.png", dpi=150, bbox_inches="tight")

    plt.figure(figsize=(10, 4))
    plt.plot(k, y_seq, color="gray", label="эталон")
    plt.plot(k[P_VAR:N_TRAIN], pred_tr, "--", color="tab:blue", label="прогноз (train)")
    plt.plot(k[N_TRAIN:], pred_te, "--", color="tab:orange", label="прогноз (test)")
    plt.axvline(N_TRAIN - 0.5, color="r", ls=":")
    plt.xlabel("k"); plt.ylabel("x"); plt.legend(); plt.grid(alpha=.3)
    plt.title(f"Одношаговый прогноз, p={P_VAR}, N={N_VAR}")
    plt.savefig("fig2_onestep.png", dpi=150, bbox_inches="tight")

    plt.figure(figsize=(8, 4.5))
    for p, n in ((P_VAR, N_VAR), (2, N_VAR), (2 * P_VAR, N_VAR)):
        errs = get_run(runs, y_n, p, n, 0)[1]
        plt.semilogy(errs, label=f"p={p}, N={n}  (эпох={len(errs)})")
    plt.axhline(EE, color="k", ls=":", label=f"Ee={EE}")
    plt.xlabel("эпоха"); plt.ylabel("Es"); plt.legend(); plt.grid(alpha=.3, which="both")
    plt.title("Сходимость обучения")
    plt.savefig("fig3_convergence.png", dpi=150, bbox_inches="tight")

    plt.figure(figsize=(5, 5))
    plt.scatter(denormalize(e_te, y_min, y_max), pred_te, s=12)
    lim = [y_seq.min(), y_seq.max()]
    plt.plot(lim, lim, "r-", label="y = x")
    plt.xlabel("эталон"); plt.ylabel("прогноз"); plt.legend(); plt.grid(alpha=.3)
    plt.title("Тестовый участок")
    plt.savefig("fig4_scatter.png", dpi=150, bbox_inches="tight")

    p_b, n_b = best
    steps = 30
    net_b = get_run(runs, y_n, p_b, n_b, 0)[0]
    rec = denormalize(recursive_forecast(net_b, y_n, p_b, N_TRAIN, steps), y_min, y_max)
    ref = y_seq[N_TRAIN:N_TRAIN + steps]
    err = np.abs(ref - rec)
    thr = 0.05 * (y_seq.max() - y_seq.min())
    over = np.where(err > thr)[0]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(range(1, steps + 1), ref, label="эталон")
    ax[0].plot(range(1, steps + 1), rec, "--", label="рекурсивный прогноз")
    ax[0].set_xlabel("шаг"); ax[0].set_ylabel("x"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(range(1, steps + 1), err, label="|ошибка|")
    ax[1].axhline(thr, color="r", ls="--", label="5% размаха")
    ax[1].set_xlabel("шаг"); ax[1].set_ylabel("абсолютная ошибка")
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle(f"Рекурсивный прогноз, p={p_b}, N={n_b}")
    fig.savefig("fig5_recursive.png", dpi=150, bbox_inches="tight")

    print(f"\nРекурсивный прогноз (p={p_b}, N={n_b})")
    print("  ошибка на шагах 1, 5, 10, 20, 30:", np.round(err[[0, 4, 9, 19, 29]], 5))
    print("  первый шаг с ошибкой > 5% размаха:", over[0] + 1 if len(over) else "не превышена")
    plt.show()


def functioning_demo(net, y_seq, y_min, y_max):
    print("\nРежим функционирования")
    for label, ks in (("train", (20, 60, 100)), ("test", (150, 170, 190))):
        for k in ks:
            print(f"Окно из ряда ({label}), k={k}:")
            predict_window(net, y_seq[k - P_VAR:k], y_min, y_max, y_seq[k])
    print("Пользовательское окно:")
    predict_window(net, [0.9, 0.6, 0.1, -0.4, -0.9, -1.1], y_min, y_max)
    print("Окно с выбросом (k=100, четвёртое значение заменено на 3.0):")
    win = y_seq[100 - P_VAR:100].copy()
    win[3] = 3.0
    predict_window(net, win, y_min, y_max, y_seq[100])


def main():
    y_seq = target(np.arange(N_POINTS))
    y_min, y_max = y_seq[:N_TRAIN].min(), y_seq[:N_TRAIN].max()   # по обучающему участку
    y_n = normalize(y_seq, y_min, y_max)
    runs = {}

    print(f"Базовый эксперимент: p={P_VAR}, N={N_VAR}, alpha={ALPHA}")
    net, errs = get_run(runs, y_n, P_VAR, N_VAR, 0)
    res = evaluate(net, y_n, P_VAR, y_min, y_max)
    print(f"  эпох: {len(errs)}, Es: {errs[-1]:.3e}")
    for name in ("train", "test"):
        m = res[name]
        print(f"  {name}: MAE={m[0]:.5f}  RMSE={m[1]:.5f}  sMAPE={m[2]:.3f}%")

    rows_p = study(runs, y_n, y_min, y_max, "Исследование окна (N=4)",
                   [(p, N_VAR) for p in (2, 4, P_VAR, 2 * P_VAR)])
    rows_n = study(runs, y_n, y_min, y_max, "Исследование скрытого слоя (p=6)",
                   [(P_VAR, n) for n in (1, 2, N_VAR, 2 * N_VAR)])
    best = min({**rows_p, **rows_n}.items(), key=lambda kv: kv[1])[0]
    print(f"\nЛучшая по test MAE конфигурация: p={best[0]}, N={best[1]}")

    functioning_demo(net, y_seq, y_min, y_max)
    plot_all(runs, y_seq, y_n, y_min, y_max, best)


if __name__ == "__main__":
    main()
