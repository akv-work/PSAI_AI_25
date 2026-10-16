from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

A, B, C, D = 0.8, 0.20, 0.4, 0.50
P, N = 8, 4
TRAIN_END = 140
LR = 0.05
EE = 0.0001
MAX_EPOCHS = 20000
SEEDS = [1, 2, 3, 4, 5]
OUT = Path(__file__).resolve().parent / "figures"


def generate_series():
    k = np.arange(200)
    return A * np.cos(B * k) + C * np.sin(D * k)


def normalize(x, low, high):
    return (x - low) / (high - low)


def denormalize(x, low, high):
    return low + x * (high - low)


def make_windows(z, p, start, end):
    inputs, targets = [], []
    for k in range(start, end):
        inputs.append(z[k - p:k])
        targets.append(z[k])
    return np.array(inputs), np.array(targets)


def forward(x, model):
    w1, b1, w2, b2 = model
    h = 1 / (1 + np.exp(-np.clip(x @ w1 + b1, -500, 500)))
    return h, h @ w2 + b2[0]


def backward(x, target, model):
    w1, b1, w2, b2 = model
    h, y = forward(x, model)
    delta = 2 * (y - target)
    hidden_delta = delta * w2 * h * (1 - h)
    w2 -= LR * delta * h
    b2[0] -= LR * delta
    w1 -= LR * np.outer(x, hidden_delta)
    b1 -= LR * hidden_delta


def train(inputs, targets, n, seed):
    rng = np.random.default_rng(seed)
    p = inputs.shape[1]
    model = [rng.uniform(-0.1, 0.1, (p, n)), np.zeros(n),
             rng.uniform(-0.1, 0.1, n), np.zeros(1)]
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        for i in rng.permutation(len(inputs)):
            backward(inputs[i], targets[i], model)
        _, y = forward(inputs, model)
        error = float(np.mean((y - targets) ** 2))
        history.append(error)
        if error <= EE:
            break
    return model, history


def metrics(model, inputs, targets, low, high):
    _, y = forward(inputs, model)
    prediction = denormalize(y, low, high)
    true = denormalize(targets, low, high)
    error = np.abs(prediction - true)
    denominator = np.abs(prediction) + np.abs(true)
    smape = 200 * np.mean(error / np.maximum(denominator, 1e-12))
    return np.mean(error), np.sqrt(np.mean(error ** 2)), smape


def experiments(z, low, high):
    results = {}
    configs = [(p, N) for p in [2, 4, P, 2 * P]] + [(P, n) for n in [1, 2, 2 * N]]
    print(f"lr={LR}, Ee={EE}, максимум эпох={MAX_EPOCHS}")
    for p, n in configs:
        xtr, ytr = make_windows(z, p, p, TRAIN_END)
        xte, yte = make_windows(z, p, TRAIN_END, len(z))
        runs = []
        for seed in SEEDS:
            model, history = train(xtr, ytr, n, seed)
            train_metrics = metrics(model, xtr, ytr, low, high)
            test_metrics = metrics(model, xte, yte, low, high)
            runs.append(dict(seed=seed, model=model, history=history,
                             train=train_metrics, test=test_metrics))
            print(f"p={p:2}, N={n}, seed={seed}: эпох={len(history):5}, "
                  f"Es={history[-1]:.7f}, train MAE={train_metrics[0]:.4f}, "
                  f"test MAE={test_metrics[0]:.4f}", flush=True)
        results[p, n] = runs
    print("\np   N    эпохи mean/std     test MAE mean/std    не достигли Ee")
    for (p, n), runs in results.items():
        epochs = [len(r['history']) for r in runs]
        mae = [r['test'][0] for r in runs]
        failed = sum(r['history'][-1] > EE for r in runs)
        print(f"{p:2} {n:3} {np.mean(epochs):9.1f}/{np.std(epochs):6.1f} "
              f"{np.mean(mae):13.4f}/{np.std(mae):.4f} {failed:12}/5")
    return results


def recursive(model, z, p, steps=30):
    window = z[TRAIN_END - p:TRAIN_END].copy()
    predictions = []
    for step in range(steps):
        _, y = forward(window, model)
        predictions.append(y)
        window = np.append(window[1:], y)
    return np.array(predictions)


def plots(x, z, low, high, results, best):
    OUT.mkdir(exist_ok=True)
    p, n = best
    model = results[best][0]['model']
    k = np.arange(len(x))
    xtr, _ = make_windows(z, p, p, TRAIN_END)
    xte, _ = make_windows(z, p, TRAIN_END, len(x))
    _, train_pred = forward(xtr, model)
    _, test_pred = forward(xte, model)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].plot(k, x, color="gray", label="Ряд")
    axes[0, 0].plot(k[p:TRAIN_END], denormalize(train_pred, low, high), label="Прогноз train")
    axes[0, 0].plot(k[TRAIN_END:], denormalize(test_pred, low, high), label="Прогноз test")
    axes[0, 0].axvline(TRAIN_END - 0.5, color="black", linestyle="--", label="Граница train/test")
    axes[0, 0].set(xlabel="k", ylabel="x(k)", title=f"Одношаговый прогноз, p={p}, N={n}, seed=1")
    for config in [(P, N), (4, N), (2 * P, N)]:
        h = results[config][0]['history']
        axes[0, 1].plot(np.arange(1, len(h) + 1), h, label=f"p={config[0]}, N={config[1]}")
    axes[0, 1].axhline(EE, color="black", linestyle=":", label="Ee")
    axes[0, 1].set(xlabel="Эпоха", ylabel="Es", yscale="log", title="Сходимость, seed=1")
    axes[1, 0].scatter(x[TRAIN_END:], denormalize(test_pred, low, high), s=16)
    axes[1, 0].plot([low, high], [low, high], "k--", label="y=x")
    axes[1, 0].set(xlabel="Эталон", ylabel="Прогноз", title="Тестовый участок")
    values = denormalize(recursive(model, z, p), low, high)
    errors = np.abs(values - x[TRAIN_END:TRAIN_END + 30])
    threshold = 0.05 * np.ptp(x)
    exceed = np.flatnonzero(errors > threshold)
    print(f"\nПорог 5% размаха={threshold:.4f}. "
          f"Первое превышение: {int(exceed[0]) + 1 if len(exceed) else 'нет за 30 шагов'}")
    axes[1, 1].plot(np.arange(1, 31), errors, label="Абсолютная ошибка")
    axes[1, 1].axhline(threshold, color="red", linestyle="--", label="5% размаха")
    axes[1, 1].set(xlabel="Шаг", ylabel="Ошибка", title="Ошибка рекурсивного прогноза")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "lab4.png", dpi=150)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(k[TRAIN_END:TRAIN_END + 30], x[TRAIN_END:TRAIN_END + 30], "o-", label="Эталон")
    ax.plot(k[TRAIN_END:TRAIN_END + 30], values, "x-", label="Рекурсивный прогноз")
    ax.set(xlabel="k", ylabel="x(k)", title=f"Прогноз на 30 шагов, p={p}, N={n}, seed=1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "recursive.png", dpi=150)
    plt.show()


def show(window, model, low, high, true=None):
    z = normalize(np.asarray(window), low, high)
    _, y = forward(z, model)
    real = denormalize(y, low, high)
    print("Входы:", np.array2string(z, precision=4, floatmode="fixed"))
    print(f"Прогноз: нормализованный={y:.4f}, исходная шкала={real:.4f}")
    if true is not None:
        print(f"Эталон={true:.4f}, абсолютная ошибка={abs(real - true):.4f}")


def functioning(model, p, x, low, high):
    for k in [20, 70, 130, 150, 170, 195]:
        print(f"\nk={k}, {'train' if k < TRAIN_END else 'test'}")
        show(x[k - p:k], model, low, high, x[k])
    window = x[50:50 + p].copy()
    window[p // 2] += 3
    print("\nОкно с выбросом")
    show(window, model, low, high)
    print("\nПостоянное окно")
    show(np.full(p, 0.3), model, low, high)
    print(f"\nВведите k от {p} до 199 для прогноза x(k) или {p} значений от старого к новому.")
    print("Пустая строка: выход.")
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if not text:
            break
        try:
            parts = text.split()
            if len(parts) == 1:
                k = int(parts[0])
                if not p <= k < len(x):
                    raise ValueError
                show(x[k - p:k], model, low, high, x[k])
            elif len(parts) == p:
                window = np.array(parts, float)
                if not np.all(np.isfinite(window)):
                    raise ValueError
                show(window, model, low, high)
            else:
                raise ValueError
        except ValueError:
            print(f"Нужен номер k от {p} до 199 или {p} конечных чисел.")


def main():
    x = generate_series()
    low, high = x[:TRAIN_END].min(), x[:TRAIN_END].max()
    z = normalize(x, low, high)
    print(f"Train: 0..139, test: 140..199. min={low:.6f}, max={high:.6f}")
    print(f"Базовая сеть {P}-{N}-1: {TRAIN_END - P} обучающих и 60 тестовых окон")
    print(f"Диапазон test после нормализации: {z[TRAIN_END:].min():.6f}..{z[TRAIN_END:].max():.6f}")
    results = experiments(z, low, high)
    run = results[P, N][0]
    print(f"\nБазовая, seed=1: эпох={len(run['history'])}, Es={run['history'][-1]:.8f}")
    print(f"Train: MAE={run['train'][0]:.4f}, RMSE={run['train'][1]:.4f}, sMAPE={run['train'][2]:.2f}%")
    print(f"Test: MAE={run['test'][0]:.4f}, RMSE={run['test'][1]:.4f}, sMAPE={run['test'][2]:.2f}%")
    best = (P, N)
    for key in results:
        if np.mean([r['test'][0] for r in results[key]]) < np.mean([r['test'][0] for r in results[best]]):
            best = key
    print(f"\nЛучшая конфигурация по среднему test MAE: p={best[0]}, N={best[1]}")
    run = results[best][0]
    print(f"Seed=1: эпох={len(run['history'])}, Es={run['history'][-1]:.8f}, "
          f"train MAE={run['train'][0]:.4f}, test MAE={run['test'][0]:.4f}")
    p, n = best
    xtr, ytr = make_windows(z, p, p, TRAIN_END)
    xte, yte = make_windows(z, p, TRAIN_END, len(z))
    noise = np.random.default_rng(10).normal(0, 0.01, xtr.shape)
    noisy_model, _ = train(xtr + noise, ytr, n, 1)
    noisy_mae = metrics(noisy_model, xte, yte, low, high)[0]
    print(f"Шум во входах train, sigma=0.01: test MAE={noisy_mae:.4f}; без шума={run['test'][0]:.4f}")
    plots(x, z, low, high, results, best)
    functioning(run['model'], p, x, low, high)


if __name__ == "__main__":
    main()
