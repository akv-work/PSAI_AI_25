import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


A, B, C, D = 2.2, 0.27, 1.1, 0.62     
P_VAR, N_VAR = 8, 5                   
N_POINTS = 200                      
N_TRAIN = 140                         
ETA = 0.1                             
EE = 1e-4                             
MAX_EPOCHS = 20_000
SEEDS = [0, 1, 2, 3, 4]               
INIT_RANGE = 0.3                      
HORIZON = 30                          
OUT = "out"


def generate_series(n=N_POINTS):
    """x_k = a*cos(b*k) + c*sin(d*k), k = 0..n-1."""
    k = np.arange(n)
    return A * np.cos(B * k) + C * np.sin(D * k)


def fit_minmax(x_train):
    """min и max берутся ТОЛЬКО по обучающему участку."""
    return x_train.min(), x_train.max()


def normalize(x, xmin, xmax):
    """x_norm = (x - xmin) / (xmax - xmin)."""
    return (x - xmin) / (xmax - xmin)


def denormalize(xn, xmin, xmax):
    """x = x_norm * (xmax - xmin) + xmin."""
    return xn * (xmax - xmin) + xmin


def make_windows(x, p, k_from, k_to):
    """
    Скользящее окно: для каждого k из [k_from, k_to) вход = (x[k-p], ..., x[k-1]), цель = x[k].
    Возвращает X (m×p), T (m,), номера целей ks.
    """
    ks = np.arange(k_from, k_to)
    X = np.array([x[k - p:k] for k in ks])
    T = x[ks]
    return X, T, ks



def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))


def init_weights(p, n_hidden, rng):
    """Малые случайные веса. W1: p×N, b1: N, v: N, c: скаляр."""
    return {
        "W1": rng.uniform(-INIT_RANGE, INIT_RANGE, (p, n_hidden)),
        "b1": rng.uniform(-INIT_RANGE, INIT_RANGE, n_hidden),
        "v": rng.uniform(-INIT_RANGE, INIT_RANGE, n_hidden),
        "c": float(rng.uniform(-INIT_RANGE, INIT_RANGE)),
    }


def forward(w, x):
    """Прямой проход. x — вектор (p,) или матрица (m×p). Возвращает (h, y)."""
    s = x @ w["W1"] + w["b1"]          # s_j = sum_i W1[i,j]*x_i + b1_j
    h = sigmoid(s)                      # h_j = 1/(1+exp(-s_j))
    y = h @ w["v"] + w["c"]             # y = sum_j v_j*h_j + c   (f(s)=s)
    return h, y


def backward(w, x, h, y, d):
    """
    Обратный проход для E = 1/2 (y-d)^2.
        delta_out = dE/dy_in = y - d                      (f'(s)=1)
        dE/dv_j   = delta_out * h_j,   dE/dc = delta_out
        delta_j   = delta_out * v_j * h_j * (1 - h_j)     (σ'(s)=h(1-h))
        dE/dW1_ij = delta_j * x_i,     dE/db1_j = delta_j
    """
    delta_out = y - d
    delta_h = delta_out * w["v"] * h * (1.0 - h)
    return {
        "W1": np.outer(x, delta_h),
        "b1": delta_h,
        "v": delta_out * h,
        "c": delta_out,
    }


def epoch_error(w, X, T):
    """Es = (1/M) * sum 1/2 (y - d)^2 — средняя по выборке ошибка (нормализованная шкала)."""
    _, y = forward(w, X)
    return float(np.mean(0.5 * (y - T) ** 2))


def train(Xtr, Ttr, p, n_hidden, seed, eta=ETA, ee=EE, max_epochs=MAX_EPOCHS):
    """
    Онлайн-обучение: обновление весов после каждого примера, порядок примеров
    перемешивается в каждой эпохе. Остановка: Es <= ee или max_epochs.
    ee=None — обучение ровно max_epochs эпох без раннего останова.
    """
    rng = np.random.default_rng(seed)
    w = init_weights(p, n_hidden, rng)
    m = len(Xtr)
    hist = []
    converged = False
    for epoch in range(1, max_epochs + 1):
        for i in rng.permutation(m):
            x, d = Xtr[i], Ttr[i]
            h, y = forward(w, x)
            g = backward(w, x, h, y, d)
            for key in w:                       # w <- w - eta * dE/dw
                w[key] = w[key] - eta * g[key]
        es = epoch_error(w, Xtr, Ttr)
        hist.append(es)
        if ee is not None and es <= ee:
            converged = True
            break
    return w, np.array(hist), converged



def mae(y, yh):
    return float(np.mean(np.abs(y - yh)))


def rmse(y, yh):
    return float(np.sqrt(np.mean((y - yh) ** 2)))


def mape(y, yh):
    """MAPE, %. Неприменима, если ряд проходит через ноль (деление на ~0)."""
    return float(100.0 * np.mean(np.abs((y - yh) / y)))


def smape(y, yh, eps=1e-12):
    """sMAPE, %: 100 * mean( 2|y-ŷ| / (|y|+|ŷ|) ). Знаменатель обращается в ноль только если y и ŷ == 0 одновременно."""
    return float(100.0 * np.mean(2.0 * np.abs(y - yh) / (np.abs(y) + np.abs(yh) + eps)))


def all_metrics(y, yh):
    return {"MAE": mae(y, yh), "RMSE": rmse(y, yh), "sMAPE": smape(y, yh)}



def prepare(p, x=None):
    """Нормализация по train, окна для обучающего (k=p..139) и тестового (k=140..199) участков."""
    if x is None:
        x = generate_series()
    xmin, xmax = fit_minmax(x[:N_TRAIN])
    xn = normalize(x, xmin, xmax)
    Xtr, Ttr, ktr = make_windows(xn, p, p, N_TRAIN)
    Xte, Tte, kte = make_windows(xn, p, N_TRAIN, len(x))
    return {"x": x, "xn": xn, "xmin": xmin, "xmax": xmax,
            "Xtr": Xtr, "Ttr": Ttr, "ktr": ktr, "Xte": Xte, "Tte": Tte, "kte": kte}


def evaluate(w, data):
    """Одношаговый прогноз на обоих участках, метрики в исходной шкале."""
    res = {}
    for name in ("tr", "te"):
        _, yn = forward(w, data["X" + name])
        y_true = denormalize(data["T" + name], data["xmin"], data["xmax"])
        y_pred = denormalize(yn, data["xmin"], data["xmax"])
        res[name] = all_metrics(y_true, y_pred)
        res[name + "_pred"] = y_pred
        res[name + "_true"] = y_true
    return res


def run_config(args):
    """Один запуск (p, N, seed) — вызывается из пула процессов."""
    p, n_hidden, seed, eta = args
    data = prepare(p)
    w, hist, ok = train(data["Xtr"], data["Ttr"], p, n_hidden, seed, eta=eta)
    ev = evaluate(w, data)
    return {"p": p, "N": n_hidden, "seed": seed, "eta": eta, "epochs": len(hist), "Es": hist[-1],
            "converged": ok, "hist": hist, "w": w, "train": ev["tr"], "test": ev["te"]}



def recursive_forecast(w, data, p, start=N_TRAIN, horizon=HORIZON):
    """
    Начиная с k=start: окно из p ИСТИННЫХ значений до k=start-1; каждое предсказание
    подставляется в окно вместо истинного значения. Возвращает прогноз в исходной шкале.
    """
    window = list(data["xn"][start - p:start])
    preds = []
    for _ in range(horizon):
        _, y = forward(w, np.array(window))
        preds.append(float(y))
        window = window[1:] + [float(y)]
    return denormalize(np.array(preds), data["xmin"], data["xmax"])


def predict_mode(w, data, p, window=None, k=None):
    """
    window — p значений в исходной шкале ИЛИ k — номер, с которого берётся окно из ряда
    (окно x[k-p..k-1], прогнозируется x[k]). Печатает нормализованные входы, прогноз в обеих
    шкалах, а для окна из ряда — истинное значение и абсолютную ошибку.
    """
    x, xmin, xmax = data["x"], data["xmin"], data["xmax"]
    true_val = None
    if k is not None:
        window = x[k - p:k]
        true_val = x[k]
    window = np.asarray(window, dtype=float)
    if len(window) != p:
        raise ValueError(f"Нужно ровно p={p} значений, получено {len(window)}")
    xin = normalize(window, xmin, xmax)
    _, yn = forward(w, xin)
    y = float(denormalize(yn, xmin, xmax))
    title = f"окно из ряда, k={k}" if k is not None else "пользовательское окно"
    print(f"--- {title} ---")
    print("  окно (исходная шкала):", np.array2string(window, precision=4, separator=", "))
    print("  нормализованные входы:", np.array2string(xin, precision=4, separator=", "))
    print(f"  прогноз (норм. шкала):  {float(yn):.4f}")
    print(f"  прогноз (исходная шкала): {y:.4f}")
    if true_val is not None:
        print(f"  истинное x({k}) = {true_val:.4f},  |ошибка| = {abs(true_val - y):.4f}")
    return y



def gradient_check():
    """Сравнение аналитических градиентов с численными (центральная разность)."""
    rng = np.random.default_rng(123)
    p, n = 8, 5
    w = init_weights(p, n, rng)
    x, d = rng.uniform(0, 1, p), 0.37
    h, y = forward(w, x)
    g = backward(w, x, h, y, d)

    def loss(ww):
        return 0.5 * (forward(ww, x)[1] - d) ** 2

    worst, eps = 0.0, 1e-6
    for key in w:
        arr = np.atleast_1d(np.array(w[key], dtype=float))
        for idx in np.ndindex(arr.shape):
            wp = {k_: np.array(v_, dtype=float) if np.ndim(v_) else float(v_) for k_, v_ in w.items()}
            wm = {k_: np.array(v_, dtype=float) if np.ndim(v_) else float(v_) for k_, v_ in w.items()}
            if np.ndim(w[key]):
                wp[key][idx] += eps
                wm[key][idx] -= eps
                an = g[key][idx]
            else:
                wp[key] += eps
                wm[key] -= eps
                an = g[key]
            num = (loss(wp) - loss(wm)) / (2 * eps)
            worst = max(worst, abs(num - an))
    return worst


def manual_step(V=10, eta=0.1):
    """Один полный шаг (forward + backward) для сети 3–2–1 из задания; s = Σ w·x + b."""
    X = np.array([0.2, 0.5, 0.8])
    d = 0.6
    W = np.array([[0.05 * V, -0.2],     # W11 W12  (вход 1 -> нейроны 1, 2)
                  [0.3, 0.05 * V],      # W21 W22
                  [-0.1, 0.4]])         # W31 W32
    b = np.array([0.1, -0.1])
    v = np.array([0.5, -0.2 * V])       # Wout1, Wout2
    c = 0.2
    s = X @ W + b
    h = sigmoid(s)
    y = float(h @ v + c)
    E = 0.5 * (y - d) ** 2
    dout = y - d
    gv, gc = dout * h, dout
    dh = dout * v * h * (1 - h)
    gW, gb = np.outer(X, dh), dh
    print(f"Ручной шаг, V={V}, η={eta}")
    print("  W =", W.tolist(), " b =", b.tolist(), " Wout =", v.tolist(), " bout =", c)
    print(f"  s1={s[0]:.5f}  s2={s[1]:.5f}")
    print(f"  h1={h[0]:.5f}  h2={h[1]:.5f}")
    print(f"  y={y:.5f}   E=1/2(y-d)^2={E:.5f}")
    print(f"  δ_out=y-d={dout:.5f}")
    print(f"  dE/dWout1={gv[0]:.5f}  dE/dWout2={gv[1]:.5f}  dE/dbout={gc:.5f}")
    print(f"  σ'(s1)=h1(1-h1)={h[0]*(1-h[0]):.5f}  σ'(s2)={h[1]*(1-h[1]):.5f}")
    print(f"  δ1={dh[0]:.5f}  δ2={dh[1]:.5f}")
    for i in range(3):
        print(f"  dE/dW{i+1}1={gW[i,0]:.5f}  dE/dW{i+1}2={gW[i,1]:.5f}")
    print(f"  dE/db1={gb[0]:.5f}  dE/db2={gb[1]:.5f}")
    W2, b2, v2, c2 = W - eta * gW, b - eta * gb, v - eta * gv, c - eta * gc
    print("  Новые веса скрытого слоя:", np.round(W2, 5).tolist(), " b:", np.round(b2, 5).tolist())
    print("  Новые веса выхода:", np.round(v2, 5).tolist(), " bout:", round(c2, 5))
    h2 = sigmoid(X @ W2 + b2)
    y2 = float(h2 @ v2 + c2)
    print(f"  Проверка: y после шага = {y2:.5f}, E = {0.5*(y2-d)**2:.5f} (было {E:.5f})")


def summarize(runs):
    ep = np.array([r["epochs"] for r in runs], dtype=float)
    mt = np.array([r["test"]["MAE"] for r in runs])
    mtr = np.array([r["train"]["MAE"] for r in runs])
    return {"ep_mean": ep.mean(), "ep_std": ep.std(), "mae_mean": mt.mean(), "mae_std": mt.std(),
            "mae_tr_mean": mtr.mean(), "fails": sum(not r["converged"] for r in runs)}


def print_table(title, key, values, cache):
    print(f"\n{title}")
    print(f"{key:>5} | {'эпохи (ср. ± σ)':>20} | {'тест MAE (ср. ± σ)':>22} | {'train MAE (ср.)':>15} | не сошлось из 5")
    print("-" * 92)
    rows = {}
    for val in values:
        runs = cache[val]
        s = summarize(runs)
        rows[val] = s
        print(f"{val:>5} | {s['ep_mean']:>11.1f} ± {s['ep_std']:<7.1f} | "
              f"{s['mae_mean']:>12.4f} ± {s['mae_std']:<7.4f} | {s['mae_tr_mean']:>15.4f} | {s['fails']}")
    return rows


def noise_experiment(pool, sigma=0.1, epochs=3000, hidden_list=(5, 30)):
    """
    Обучение на ряде с гауссовым шумом (σ в исходной шкале), тест — на чистом ряду.
    Критерий Es<=Ee не используется (при шуме он недостижим) — фиксированное число эпох.
    """
    print(f"\nЭксперимент с шумом: σ={sigma} (исходная шкала), {epochs} эпох без раннего останова, p={P_VAR}")
    x_clean = generate_series()
    out = []
    for n_hidden in hidden_list:
        for noisy in (False, True):
            maes_tr, maes_te = [], []
            for seed in SEEDS[:3]:
                rng = np.random.default_rng(1000 + seed)
                x_tr = x_clean.copy()
                if noisy:
                    x_tr[:N_TRAIN] += rng.normal(0, sigma, N_TRAIN)
                xmin, xmax = fit_minmax(x_tr[:N_TRAIN])
                Xtr, Ttr, _ = make_windows(normalize(x_tr, xmin, xmax), P_VAR, P_VAR, N_TRAIN)
                # тест — на чистом ряду, нормализованном теми же min/max
                xn_clean = normalize(x_clean, xmin, xmax)
                Xte, Tte, _ = make_windows(xn_clean, P_VAR, N_TRAIN, N_POINTS)
                Xtr_c, Ttr_c, _ = make_windows(xn_clean, P_VAR, P_VAR, N_TRAIN)
                w, _, _ = train(Xtr, Ttr, P_VAR, n_hidden, seed, ee=None, max_epochs=epochs)
                for X_, T_, store in ((Xtr_c, Ttr_c, maes_tr), (Xte, Tte, maes_te)):
                    yh = denormalize(forward(w, X_)[1], xmin, xmax)
                    store.append(mae(denormalize(T_, xmin, xmax), yh))
            out.append((n_hidden, noisy, np.mean(maes_tr), np.mean(maes_te)))
            print(f"  N={n_hidden:>2}, шум={'да ' if noisy else 'нет'} | MAE (чистый train) {np.mean(maes_tr):.4f} | "
                  f"MAE (чистый test) {np.mean(maes_te):.4f}")
    return out



def plot_all(base, runs_p, rec, data, best_label):
    os.makedirs(OUT, exist_ok=True)
    x = data["x"]
    k = np.arange(len(x))

   
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(k, x, color="0.25", lw=1.6, label="эталон x(k)")
    ev = base
    ax.plot(data["ktr"], ev["tr_pred"], "o", ms=3, color="tab:blue", label="прогноз (train)")
    ax.plot(data["kte"], ev["te_pred"], "s", ms=3, color="tab:red", label="прогноз (test)")
    ax.axvline(N_TRAIN - 0.5, color="k", ls="--", lw=1)
    ax.text(N_TRAIN + 1, ax.get_ylim()[1] * 0.92, "test", fontsize=10)
    ax.text(N_TRAIN - 18, ax.get_ylim()[1] * 0.92, "train", fontsize=10)
    ax.set_xlabel("k")
    ax.set_ylabel("x(k)")
    ax.set_title(f"Ряд и одношаговый прогноз сети {P_VAR}–{N_VAR}–1 (граница train/test: k=140)")
    ax.legend(loc="lower left", ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_series_forecast.png", dpi=150)
    plt.close(fig)

    
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for r, lab in ((runs_p[P_VAR][0], f"p={P_VAR} (базовая)"), (runs_p[2][0], "p=2"), (runs_p[4][0], "p=4")):
        ax.semilogy(np.arange(1, len(r["hist"]) + 1), r["hist"], label=lab)
    ax.axhline(EE, color="k", ls=":", lw=1, label="Ee = 1e-4")
    ax.set_xscale("log")
    ax.set_xlabel("эпоха (лог. шкала)")
    ax.set_ylabel("Es (лог. шкала)")
    ax.set_title("Сходимость обучения (N=5, seed=0)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_convergence.png", dpi=150)
    plt.close(fig)

  
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    yt, yp = ev["te_true"], ev["te_pred"]
    ax.scatter(yt, yp, s=18, color="tab:red")
    lim = [min(yt.min(), yp.min()) - 0.1, max(yt.max(), yp.max()) + 0.1]
    ax.plot(lim, lim, "k--", lw=1, label="y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("эталон")
    ax.set_ylabel("прогноз")
    ax.set_title("Тестовый участок: эталон – прогноз")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_scatter_test.png", dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    steps = np.arange(1, HORIZON + 1)
    axs[0].plot(steps, rec["true"], "k-o", ms=3, label="эталон")
    axs[0].plot(steps, rec["pred"], "r-s", ms=3, label="рекурсивный прогноз")
    axs[0].set_xlabel("шаг (k = 140 + шаг − 1)")
    axs[0].set_ylabel("x")
    axs[0].set_title(f"Рекурсивный прогноз на {HORIZON} шагов ({best_label})")
    axs[0].legend()
    axs[0].grid(alpha=0.3)
    axs[1].plot(steps, rec["err"], "b-o", ms=3, label="|ошибка|")
    axs[1].axhline(rec["thr"], color="r", ls="--", label=f"5 % размаха = {rec['thr']:.3f}")
    axs[1].set_xlabel("шаг")
    axs[1].set_ylabel("абсолютная ошибка")
    axs[1].set_title("Рост ошибки по шагам")
    axs[1].legend()
    axs[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_recursive.png", dpi=150)
    plt.close(fig)



def train_base(seed=0):
    data = prepare(P_VAR)
    w, hist, ok = train(data["Xtr"], data["Ttr"], P_VAR, N_VAR, seed)
    return w, data, hist, ok


def main_interactive(args):
    w, data, hist, ok = train_base()
    print(f"Сеть {P_VAR}–{N_VAR}–1 обучена за {len(hist)} эпох, Es={hist[-1]:.3e}")
    if args.k is not None:
        predict_mode(w, data, P_VAR, k=args.k)
    if args.window is not None:
        predict_mode(w, data, P_VAR, window=[float(t.replace(",", ".")) for t in args.window.split()])


def main_full():
    os.makedirs(OUT, exist_ok=True)
    x = generate_series()
    data8 = prepare(P_VAR)
    print("=" * 92)
    print(f"Вариант 10: a={A}, b={B}, c={C}, d={D}, p={P_VAR}, N={N_VAR}, η={ETA}, Ee={EE}, макс. эпох={MAX_EPOCHS}")
    print(f"Ряд: min={x.min():.4f}, max={x.max():.4f}, размах={x.max()-x.min():.4f}; "
          f"проходит через ноль -> вместо MAPE используется sMAPE")
    print(f"Нормализация по train (k=0..139): x_min={data8['xmin']:.4f}, x_max={data8['xmax']:.4f}")
    xn_te = data8["xn"][N_TRAIN:]
    print(f"Нормализованный тест лежит в [{xn_te.min():.4f}; {xn_te.max():.4f}]")
    for p in (2, 4, P_VAR, 2 * P_VAR):
        d_ = prepare(p)
        print(f"  p={p:>2}: обучающих примеров {len(d_['Xtr'])} (k={p}..139), тестовых {len(d_['Xte'])} (k=140..199)")
    print(f"Проверка градиентов (аналитические vs численные): max |разность| = {gradient_check():.2e}")

    pool = Pool(2)

    print("\nВыбор скорости обучения (p=8, N=5, 5 запусков):")
    etas = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
    jobs = [(P_VAR, N_VAR, s, e) for e in etas for s in SEEDS]
    res = pool.map(run_config, jobs)
    print(f"{'η':>6} | {'эпохи (ср. ± σ)':>18} | {'тест MAE':>9} | не сошлось")
    for i, e in enumerate(etas):
        rr = res[i * 5:(i + 1) * 5]
        s = summarize(rr)
        print(f"{e:>6} | {s['ep_mean']:>9.1f} ± {s['ep_std']:<6.1f} | {s['mae_mean']:>9.4f} | {s['fails']}")


    p_list = [2, 4, P_VAR, 2 * P_VAR]
    n_list = [1, 2, N_VAR, 2 * N_VAR]
    configs = sorted({(p, N_VAR) for p in p_list} | {(P_VAR, n) for n in n_list})
    jobs = [(p, n, s, ETA) for (p, n) in configs for s in SEEDS]
    results = pool.map(run_config, jobs)
    pool.close()
    by_cfg = {}
    for r in results:
        by_cfg.setdefault((r["p"], r["N"]), []).append(r)
    runs_p = {p: by_cfg[(p, N_VAR)] for p in p_list}
    runs_n = {n: by_cfg[(P_VAR, n)] for n in n_list}

    # ---- базовый эксперимент (seed 0) ----
    base_run = runs_p[P_VAR][0]
    ev = evaluate(base_run["w"], data8)
    print("\n" + "=" * 92)
    print(f"БАЗОВЫЙ ЭКСПЕРИМЕНТ: p={P_VAR}, N={N_VAR}, η={ETA}, seed=0")
    print(f"  эпох: {base_run['epochs']}, итоговая Es = {base_run['Es']:.3e} "
          f"({'критерий Es<=Ee достигнут' if base_run['converged'] else 'критерий НЕ достигнут'})")
    for name, lab in (("tr", "train"), ("te", "test ")):
        m = ev[name]
        print(f"  {lab}: MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  sMAPE={m['sMAPE']:.2f} %")

    tab_p = print_table(f"ИССЛЕДОВАНИЕ ОКНА (N={N_VAR}, 5 запусков)", "p", p_list, runs_p)
    tab_n = print_table(f"ИССЛЕДОВАНИЕ СКРЫТОГО СЛОЯ (p={P_VAR}, 5 запусков)", "N", n_list, runs_n)

    best_cfg = min(by_cfg, key=lambda c: summarize(by_cfg[c])["mae_mean"])
    best_p, best_n = best_cfg
    best_run = by_cfg[best_cfg][0]
    data_b = prepare(best_p)
    rec_pred = recursive_forecast(best_run["w"], data_b, best_p)
    rec_true = x[N_TRAIN:N_TRAIN + HORIZON]
    err = np.abs(rec_pred - rec_true)
    thr = 0.05 * (x.max() - x.min())
    over = np.where(err > thr)[0] + 1
    print("\n" + "=" * 92)
    print(f"МНОГОШАГОВЫЙ ПРОГНОЗ: лучшая по среднему тестовому MAE конфигурация p={best_p}, N={best_n} "
          f"(MAE {summarize(by_cfg[best_cfg])['mae_mean']:.4f}), запуск seed=0")
    print(f"  порог 5 % размаха ряда = {thr:.4f}")
    print(f"  средняя |ошибка| рекурсивного прогноза за {HORIZON} шагов = {err.mean():.4f} "
          f"(одношаговый test MAE этой сети = {best_run['test']['MAE']:.4f})")
    print(f"  макс. ошибка = {err.max():.4f} на шаге {int(err.argmax()) + 1}")
    if len(over):
        print(f"  первое превышение порога на шаге {over[0]} (ошибка {err[over[0]-1]:.4f}); "
              f"шаги с превышением: {over.tolist()}")
    else:
        print("  порог 5 % не превышен ни на одном шаге")
    print("  шаг: эталон | прогноз | |ошибка|")
    for i in range(HORIZON):
        print(f"  {i+1:>3}: {rec_true[i]:>8.4f} | {rec_pred[i]:>8.4f} | {err[i]:.4f}")

    rec = {"true": rec_true, "pred": rec_pred, "err": err, "thr": thr}
    plot_all(ev, runs_p, rec, data8, f"p={best_p}, N={best_n}")
    print(f"\nГрафики сохранены в {OUT}/")


    print("\n" + "=" * 92)
    print(f"РЕЖИМ ФУНКЦИОНИРОВАНИЯ (сеть {P_VAR}–{N_VAR}–1, seed=0)")
    w = base_run["w"]
    for k_ in (20, 60, 120):
        predict_mode(w, data8, P_VAR, k=k_)
    for k_ in (145, 170, 195):
        predict_mode(w, data8, P_VAR, k=k_)
    win = x[100:108].copy()
    print("[пользовательское окно: значения ряда k=100..107 с округлением до 1 знака]")
    predict_mode(w, data8, P_VAR, window=np.round(win, 1))
    print(f"[истинное значение для этого окна x(108) = {x[108]:.4f}]")
    win_out = x[100:108].copy()
    win_out[4] = 6.0
    print("[пользовательское окно с выбросом: 5-е значение заменено на 6.0]")
    predict_mode(w, data8, P_VAR, window=win_out)
    print(f"[истинное значение без выброса x(108) = {x[108]:.4f}]")


    noise_experiment(None)

    return tab_p, tab_n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, help="номер k: окно x[k-p..k-1] берётся из ряда (p<=k<=199)")
    ap.add_argument("--window", type=str, help="p значений ряда в исходной шкале через пробел")
    args = ap.parse_args()
    if args.k is not None or args.window is not None:
        main_interactive(args)
    else:
        manual_step()
        print()
        main_full()