import numpy as np
import matplotlib.pyplot as plt

A_COEF = 2.0
B_COEF = 0.25
C_COEF = 1.0
D_COEF = 0.60
P_VAR = 8
N_VAR = 5
N_POINTS = 200
TRAIN_SIZE = 140
EE = 1e-4
MAX_EPOCHS = 20000
LR = 0.1

np.set_printoptions(precision=4, suppress=True)

def generate_series(a, b, c, d, n=200):
    k = np.arange(n)
    return a * np.cos(b * k) + c * np.sin(d * k)

def normalize_train(x, train_size):
    x_train = x[:train_size]
    xmin, xmax = x_train.min(), x_train.max()
    x_norm = (x - xmin) / (xmax - xmin)
    return x_norm, xmin, xmax

def denormalize(xn, xmin, xmax):
    return xmin + xn * (xmax - xmin)

def make_windows(x_norm, p, train_size):
    X, Y = [], []
    for i in range(p, len(x_norm)):
        X.append(x_norm[i - p:i])
        Y.append(x_norm[i])
    X = np.array(X)
    Y = np.array(Y)
    n_train = train_size - p
    return X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]

class MLP:
    def __init__(self, p, n_hidden, lr=0.1, seed=1):
        self.p = p
        self.N = n_hidden
        self.lr = lr
        self.rng = np.random.RandomState(seed)
        self.W1 = self.rng.normal(0, 0.5, (n_hidden, p))
        self.b1 = self.rng.normal(0, 0.5, n_hidden)
        self.W2 = self.rng.normal(0, 0.5, n_hidden)
        self.b2 = self.rng.normal(0, 0.5)
        self.epochs_trained = 0
        self.final_loss = np.inf
        self.converged = False
        self.loss_history = []

    @staticmethod
    def _sigmoid(s):
        s = np.clip(s, -500, 500)
        return 1.0 / (1.0 + np.exp(-s))

    def forward(self, x):
        s = self.W1 @ x + self.b1
        h = self._sigmoid(s)
        y = self.W2 @ h + self.b2 
        return s, h, y

    def backward_step(self, x, d):
        s, h, y = self.forward(x)
        err = y - d
        loss = 0.5 * err * err

        delta_out = err
        grad_W2 = delta_out * h
        grad_b2 = delta_out
        delta_h = delta_out * self.W2 * h * (1.0 - h)
        grad_W1 = np.outer(delta_h, x)
        grad_b1 = delta_h

        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1
        return loss

    def predict(self, x):
        _, _, y = self.forward(x)
        return y

def train_mlp(X, Y, p, n_hidden, lr=0.1, Ee=1e-4, max_epochs=20000,
              seed=1, verbose=False):
    mlp = MLP(p, n_hidden, lr=lr, seed=seed)
    n = len(X)
    for epoch in range(1, max_epochs + 1):
        idx = mlp.rng.permutation(n)
        Es = 0.0
        for i in idx:
            Es += mlp.backward_step(X[i], Y[i])
        Es /= n
        mlp.loss_history.append(Es)
        mlp.final_loss = Es
        mlp.epochs_trained = epoch
        if Es <= Ee:
            mlp.converged = True
            break
        if not np.isfinite(Es):
            break
    if verbose:
        print(f"  seed={seed}: эпох={mlp.epochs_trained}, "
              f"Es={mlp.final_loss:.3e}, сошлась={mlp.converged}")
    return mlp

def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))

def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def smape(y, yhat):
    denom = np.abs(y) + np.abs(yhat)
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return float(100.0 * np.mean(2.0 * np.abs(y - yhat) / denom))

def one_step_predict(mlp, X):
    return np.array([mlp.predict(x) for x in X])

def recursive_forecast(mlp, last_window, steps):
    window = list(last_window)
    preds = []
    for _ in range(steps):
        x = np.array(window[-mlp.p:])
        y = mlp.predict(x)
        preds.append(y)
        window.append(y)
    return np.array(preds)

def inference_demo(mlp, x, x_norm, xmin, xmax, p):
    print("Режим функционирования")

    def show_window(k, label):
        window_real = x[k - p:k]
        window_norm = x_norm[k - p:k]
        y_norm = mlp.predict(window_norm)
        y_real = denormalize(y_norm, xmin, xmax)
        true_val = x[k]
        err = abs(y_real - true_val)
        print(f"\n[{label}] k={k}")
        print(f"  Вход (исходная шкала): {np.round(window_real, 4)}")
        print(f"  Вход (норм.):         {np.round(window_norm, 4)}")
        print(f"  ŷ (норм.):            {y_norm:.4f}")
        print(f"  ŷ (исходная):         {y_real:.4f}")
        print(f"  x(k) истинное:        {true_val:.4f}")
        print(f"  |ошибка|:             {err:.4f}")

    show_window(20, "train")
    show_window(60, "train")
    show_window(120, "train")

    show_window(150, "test")
    show_window(170, "test")
    show_window(195, "test")

    print("\nПользовательские окна")
    user_windows = [
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]),
    ]
    for i, w in enumerate(user_windows, 1):
        wn = (w - xmin) / (xmax - xmin)
        y_norm = mlp.predict(wn)
        y_real = denormalize(y_norm, xmin, xmax)
        print(f"\n[user {i}] окно (исх.): {np.round(w, 4)}")
        print(f"  Вход (норм.):  {np.round(wn, 4)}")
        print(f"  ŷ (норм.):     {y_norm:.4f}")
        print(f"  ŷ (исходная):  {y_real:.4f}")

def plot_series_with_split(x, train_size):
    plt.figure(figsize=(12, 4))
    k = np.arange(len(x))
    plt.plot(k[:train_size], x[:train_size], 'b-', label='Train')
    plt.plot(k[train_size:], x[train_size:], 'r-', label='Test')
    plt.axvline(train_size - 0.5, color='k', linestyle='--', label='граница')
    plt.title('Временной ряд x(k)')
    plt.xlabel('k'); plt.ylabel('x(k)')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.show()

def plot_one_step(mlp, x, x_norm, X_all, Y_all, xmin, xmax, train_size, p):
    preds_norm = np.array([mlp.predict(xw) for xw in X_all])
    preds_real = denormalize(preds_norm, xmin, xmax)
    true_real = denormalize(Y_all, xmin, xmax)
    ks = np.arange(p, len(x))

    plt.figure(figsize=(12, 4))
    plt.plot(np.arange(len(x)), x, 'k-', alpha=0.4, label='Эталон')
    plt.plot(ks, preds_real, 'g--', label='Одношаговый прогноз')
    plt.axvline(train_size - 0.5, color='k', linestyle=':')
    plt.title('Одношаговый прогноз на train/test')
    plt.xlabel('k'); plt.ylabel('x(k)')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.show()

def plot_convergence(histories, labels, Ee):
    plt.figure(figsize=(10, 5))
    for h, lab in zip(histories, labels):
        plt.plot(range(1, len(h) + 1), h, label=lab, linewidth=1.5)
    plt.axhline(Ee, color='green', linestyle='--', label=f'Ee={Ee}')
    plt.yscale('log')
    plt.xlabel('Эпоха'); plt.ylabel('Es')
    plt.title('Сходимость обучения')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.show()

def plot_scatter(y_true, y_pred, title):
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], 'r--', label='y = x')
    plt.xlabel('Эталон'); plt.ylabel('Прогноз')
    plt.title(title)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.show()

def plot_recursive(x, x_norm, mlp, train_size, steps=30):
    last_window = x_norm[train_size - mlp.p:train_size]
    preds_norm = recursive_forecast(mlp, last_window, steps)
    return preds_norm

def study_window(x_norm, train_size, N, p_list, seeds, lr, Ee, max_epochs):
    rows = []
    print("\nИсследование размера окна p (N=%d)" % N)
    for p in p_list:
        Xtr, Ytr, Xte, Yte = make_windows(x_norm, p, train_size)
        epochs, test_mae = [], []
        not_conv = 0
        for s in seeds:
            m = train_mlp(Xtr, Ytr, p, N, lr, Ee, max_epochs, s)
            epochs.append(m.epochs_trained if m.converged else max_epochs)
            if not m.converged:
                not_conv += 1
            pr = one_step_predict(m, Xte)
            test_mae.append(mae(Yte, pr))
        rows.append(dict(p=p, epochs_mean=np.mean(epochs),
                         epochs_std=np.std(epochs),
                         mae_mean=np.mean(test_mae),
                         mae_std=np.std(test_mae),
                         not_conv=not_conv))
    return rows

def study_hidden(x_norm, train_size, p, n_list, seeds, lr, Ee, max_epochs):
    rows = []
    print("\nИсследование числа скрытых нейронов N (p=%d)" % p)
    for n in n_list:
        Xtr, Ytr, Xte, Yte = make_windows(x_norm, p, train_size)
        epochs, test_mae = [], []
        not_conv = 0
        for s in seeds:
            m = train_mlp(Xtr, Ytr, p, n, lr, Ee, max_epochs, s)
            epochs.append(m.epochs_trained if m.converged else max_epochs)
            if not m.converged:
                not_conv += 1
            pr = one_step_predict(m, Xte)
            test_mae.append(mae(Yte, pr))
        rows.append(dict(N=n, epochs_mean=np.mean(epochs),
                         epochs_std=np.std(epochs),
                         mae_mean=np.mean(test_mae),
                         mae_std=np.std(test_mae),
                         not_conv=not_conv))
    return rows

def print_table(rows, key_name):
    print(f"{key_name:>5} {'эпох(сред)':>12} {'эпох(std)':>10}"
          f"{'MAE(сред)':>11} {'MAE(std)':>9} {'несход.':>8}")
    for r in rows:
        print(f"{r[key_name]:>5} {r['epochs_mean']:>12.1f} {r['epochs_std']:>10.1f}"
              f"{r['mae_mean']:>11.5f} {r['mae_std']:>9.5f} {r['not_conv']:>8}")

def main():
    x = generate_series(A_COEF, B_COEF, C_COEF, D_COEF, N_POINTS)
    plot_series_with_split(x, TRAIN_SIZE)

    x_norm, xmin, xmax = normalize_train(x, TRAIN_SIZE)
    print(f"\nНормировка по train: xmin={xmin:.4f}, xmax={xmax:.4f}")
    print(f"Диапазон train: [{x[:TRAIN_SIZE].min():.4f}, {x[:TRAIN_SIZE].max():.4f}]")
    print(f"Диапазон test:  [{x[TRAIN_SIZE:].min():.4f}, {x[TRAIN_SIZE:].max():.4f}]")
    out_of_range = (x_norm[TRAIN_SIZE:] < 0) | (x_norm[TRAIN_SIZE:] > 1)
    print(f"Точек теста вне [0;1]: {out_of_range.sum()} из {len(out_of_range)}")

    Xtr, Ytr, Xte, Yte = make_windows(x_norm, P_VAR, TRAIN_SIZE)
    print(f"\nОкна: train={len(Xtr)}, test={len(Xte)} (p={P_VAR})")

    print("\nБАЗОВЫЙ ЭКСПЕРИМЕНТ (p=%d, N=%d, lr=%.2f)" % (P_VAR, N_VAR, LR))
    base = train_mlp(Xtr, Ytr, P_VAR, N_VAR, LR, EE, MAX_EPOCHS, seed=1, verbose=True)
    print(f"  Итоговая Es = {base.final_loss:.3e}")

    pred_tr_norm = one_step_predict(base, Xtr)
    pred_te_norm = one_step_predict(base, Xte)
    ytr = denormalize(Ytr, xmin, xmax)
    yte = denormalize(Yte, xmin, xmax)
    p_tr = denormalize(pred_tr_norm, xmin, xmax)
    p_te = denormalize(pred_te_norm, xmin, xmax)

    print(f"\n  Train: MAE={mae(ytr,p_tr):.4f}  RMSE={rmse(ytr,p_tr):.4f}  "
          f"sMAPE={smape(ytr,p_tr):.2f}%")
    print(f"  Test:  MAE={mae(yte,p_te):.4f}  RMSE={rmse(yte,p_te):.4f}  "
          f"sMAPE={smape(yte,p_te):.2f}%")

    p_list = sorted(set([2, 4, P_VAR, 2 * P_VAR]))
    seeds = [1, 2, 3, 4, 5]
    rows_p = study_window(x_norm, TRAIN_SIZE, N_VAR, p_list, seeds, LR, EE, MAX_EPOCHS)
    print_table(rows_p, 'p')

    n_list = sorted(set([1, 2, N_VAR, 2 * N_VAR]))
    rows_n = study_hidden(x_norm, TRAIN_SIZE, P_VAR, n_list, seeds, LR, EE, MAX_EPOCHS)
    print_table(rows_n, 'N')

    best_row = min(rows_p + rows_n, key=lambda r: r['mae_mean'])
    if 'p' in best_row:
        best_p, best_N = best_row['p'], N_VAR
    else:
        best_p, best_N = P_VAR, best_row['N']
    print(f"\nЛучшая конфигурация: p={best_p}, N={best_N}, "
          f"test MAE={best_row['mae_mean']:.5f}")

    Xtr_b, Ytr_b, Xte_b, Yte_b = make_windows(x_norm, best_p, TRAIN_SIZE)
    best_model = train_mlp(Xtr_b, Ytr_b, best_p, best_N, LR, EE, MAX_EPOCHS,
                           seed=1, verbose=True)

    steps = 30
    preds_norm = recursive_forecast(
        best_model, x_norm[TRAIN_SIZE - best_p:TRAIN_SIZE], steps)
    preds_real = denormalize(preds_norm, xmin, xmax)
    true_real = x[TRAIN_SIZE:TRAIN_SIZE + steps]
    ks = np.arange(TRAIN_SIZE, TRAIN_SIZE + steps)

    errors = np.abs(preds_real - true_real)
    span = x.max() - x.min()
    thr = 0.05 * span
    over = np.where(errors > thr)[0]
    print(f"\nРекурсивный прогноз (30 шагов):")
    print(f"  Размах ряда = {span:.3f}, порог 5% = {thr:.3f}")
    if len(over):
        print(f"  Ошибка превысила 5% размаха на шаге {over[0] + 1}")
    else:
        print("  Ошибка не превысила 5% размаха за 30 шагов")

    plt.figure(figsize=(11, 4))
    plt.plot(ks, true_real, 'k-o', label='Эталон')
    plt.plot(ks, preds_real, 'r--s', label='Рекурсивный прогноз')
    plt.title(f'Рекурсивный прогноз на 30 шагов (p={best_p}, N={best_N})')
    plt.xlabel('k'); plt.ylabel('x(k)')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.show()

    plt.figure(figsize=(11, 3.5))
    plt.plot(np.arange(1, steps + 1), errors, 'b-o')
    plt.axhline(thr, color='r', linestyle='--', label='5% размаха')
    plt.title('Абсолютная ошибка по шагам')
    plt.xlabel('Шаг'); plt.ylabel('|ошибка|')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.show()

    # 10. Графики
    plot_one_step(base, x, x_norm, np.vstack([Xtr, Xte]), np.concatenate([Ytr, Yte]),
                  xmin, xmax, TRAIN_SIZE, P_VAR)

    hist = [base.loss_history]
    lbls = [f'база p={P_VAR},N={N_VAR}']
    for p_extra in [2, 2 * P_VAR]:
        Xe, Ye, _, _ = make_windows(x_norm, p_extra, TRAIN_SIZE)
        m_e = train_mlp(Xe, Ye, p_extra, N_VAR, LR, EE, MAX_EPOCHS, seed=1)
        hist.append(m_e.loss_history)
        lbls.append(f'p={p_extra},N={N_VAR}')
    plot_convergence(hist, lbls, EE)

    plot_scatter(yte, p_te, f'Тест: эталон vs прогноз (база p={P_VAR},N={N_VAR})')

    inference_demo(base, x, x_norm, xmin, xmax, P_VAR)


if __name__ == "__main__":
    main()