import numpy as np
import matplotlib.pyplot as plt

C0, C1 = -8.0, 3.0
EE_MSE = 0.001
EE_BCE = 0.05
MAX_EPOCHS = 10000
LEARNING_RATE = 0.5
SEEDS = [0, 1, 2, 3, 4]

X_raw = np.array([
    [-8, -8],
    [-8, 3],
    [3, -8],
    [3, 3]
], dtype=float)
T_raw = np.array([-8, 3, 3, -8], dtype=float)

X = (X_raw - C0) / (C1 - C0)
T = ((T_raw - C0) / (C1 - C0)).reshape(4, 1)


def sigmoid(s):
    s = np.clip(s, -60, 60)
    return 1 / (1 + np.exp(-s))


def forward(x, W1, b1, W2, b2):
    x = np.asarray(x).reshape(1, 2)
    h = sigmoid(x @ W1 + b1)
    y = sigmoid(h @ W2 + b2)
    return h, y


def forward_batch(X, W1, b1, W2, b2):
    h = sigmoid(X @ W1 + b1)
    y = sigmoid(h @ W2 + b2)
    return h, y


def train(loss_name, seed, ee):
    rng = np.random.default_rng(seed)
    W1 = rng.uniform(-0.5, 0.5, (2, 2))
    b1 = rng.uniform(-0.5, 0.5, 2)
    W2 = rng.uniform(-0.5, 0.5, (2, 1))
    b2 = rng.uniform(-0.5, 0.5, 1)

    history = []
    epoch = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        for i in rng.permutation(4):
            x = X[i]
            t = T[i]
            h, y = forward(x, W1, b1, W2, b2)

            if loss_name == "MSE":
                delta_out = (y - t) * y * (1 - y)
            else:
                delta_out = y - t

            delta_hidden = (W2[:, 0] * delta_out[0, 0]) * h[0] * (1 - h[0])

            W2 -= LEARNING_RATE * np.outer(h[0], delta_out[0])
            b2 -= LEARNING_RATE * delta_out[0]
            W1 -= LEARNING_RATE * np.outer(x, delta_hidden)
            b1 -= LEARNING_RATE * delta_hidden

        outputs = np.array([forward(x, W1, b1, W2, b2)[1][0, 0] for x in X])

        if loss_name == "MSE":
            error = 0.5 * np.sum((T[:, 0] - outputs) ** 2)
        else:
            eps = 1e-12
            error = -np.sum(
                T[:, 0] * np.log(outputs + eps)
                + (1 - T[:, 0]) * np.log(1 - outputs + eps)
            )

        history.append(float(error))
        if error <= ee:
            break

    outputs = np.array([forward(x, W1, b1, W2, b2)[1][0, 0] for x in X])
    real_outputs = C0 + outputs * (C1 - C0)
    predicted = np.where(outputs >= 0.5, C1, C0)
    accuracy = np.mean(predicted == T_raw)
    mae = np.mean(np.abs(real_outputs - T_raw))

    return {
        "loss": loss_name, "seed": seed, "epochs": epoch, "error": float(error),
        "accuracy": float(accuracy), "mae": float(mae), "history": history,
        "weights": (W1, b1, W2, b2), "converged": error <= ee, "ee": ee
    }


def predict(A, B, model):
    x = np.array([(A - C0) / (C1 - C0), (B - C0) / (C1 - C0)])
    _, y = forward(x, *model["weights"])
    norm = float(y[0, 0])
    real = C0 + norm * (C1 - C0)
    cls = C0 if abs(real - C0) <= abs(real - C1) else C1
    return norm, real, cls


def print_results(results):
    print("\n=== Таблица результатов ===")
    print(f'{"Loss":<5} {"Seed":<5} {"Эпохи":<7} {"Ошибка":<12} {"Accuracy":<10} {"MAE":<9} {"Сошлась":<8}')
    for r in results:
        print(f'{r["loss"]:<5} {r["seed"]:<5} {r["epochs"]:<7} {r["error"]:<12.6f} '
              f'{r["accuracy"]:<10.4f} {r["mae"]:<9.4f} {"да" if r["converged"] else "нет":<8}')

    for name in ("MSE", "BCE"):
        group = [r for r in results if r["loss"] == name]
        conv = [r for r in group if r["converged"]]
        print(f"\n{name}: сходимость {len(conv)}/{len(group)}")
        if conv:
            ep = [r["epochs"] for r in conv]
            print(f"  Разброс эпох: {min(ep)}–{max(ep)}")
        else:
            print("  Критерий остановки не достигнут ни в одном запуске")


def plot_results(mse, bce, results):
    plt.figure(figsize=(8, 5))
    plt.plot(mse["history"], label=f'MSE (Ee={mse["ee"]})', color="blue")
    plt.plot(bce["history"], label=f'BCE (Ee={bce["ee"]})', color="red")
    plt.axhline(mse["ee"], linestyle="--", color="blue", alpha=0.5)
    plt.axhline(bce["ee"], linestyle="--", color="red", alpha=0.5)
    plt.title("Сходимость обучения (Вариант 8: c0=-8, c1=3)")
    plt.xlabel("Эпоха")
    plt.ylabel("Суммарная ошибка Es")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    x = np.arange(5)
    width = 0.35
    plt.figure(figsize=(8, 5))
    for name, shift, color in (("MSE", -width / 2, "steelblue"), ("BCE", width / 2, "indianred")):
        group = sorted([r for r in results if r["loss"] == name], key=lambda r: r["seed"])
        heights = [r["epochs"] for r in group]
        colors = [color if r["converged"] else "white" for r in group]
        edge = [color if r["converged"] else "black" for r in group]
        hatch = ["" if r["converged"] else "//" for r in group]
        plt.bar(x + shift, heights, width, label=name, color=colors,
                edgecolor=edge, hatch=hatch)
    plt.title("Количество эпох по 5 запускам (штриховка — не сошлась)")
    plt.xlabel("Seed")
    plt.ylabel("Эпохи")
    plt.xticks(x, SEEDS)
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

    values = np.linspace(-10, 10, 101)
    aa, bb = np.meshgrid(values, values)
    points = np.column_stack(((aa.ravel() - C0) / (C1 - C0),
                              (bb.ravel() - C0) / (C1 - C0)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (model, title) in zip(axes, [(mse, "Поверхность MSE (в шкале [c0;c1])"),
                                          (bce, "Поверхность BCE (в шкале [c0;c1])")]):
        _, z = forward_batch(points, *model["weights"])
        z = C0 + z[:, 0] * (C1 - C0)
        z = z.reshape(aa.shape)
        im = ax.contourf(aa, bb, z, levels=20, cmap="viridis")
        fig.colorbar(im, ax=ax, label="Выход сети")
        for i, (a, b) in enumerate(X_raw):
            color = "red" if T_raw[i] == C0 else "white"
            ax.scatter(a, b, color=color, edgecolors="black", s=90, zorder=5)
        ax.set_title(title)
        ax.set_xlabel("A")
        ax.set_ylabel("B")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.bar(["MSE", "BCE"], [mse["mae"], bce["mae"]], color=["steelblue", "indianred"])
    plt.title("Средняя абсолютная ошибка в шкале [c0;c1]")
    plt.xlabel("Функция потерь")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.show()


def main():
    results = []
    for name, ee in (("MSE", EE_MSE), ("BCE", EE_BCE)):
        for seed in SEEDS:
            results.append(train(name, seed, ee))

    print_results(results)

    chosen = {}
    for name in ("MSE", "BCE"):
        group = [r for r in results if r["loss"] == name]
        conv = [r for r in group if r["converged"]]
        chosen[name] = min(conv, key=lambda r: r["epochs"]) if conv else min(group, key=lambda r: r["error"])

    plot_results(chosen["MSE"], chosen["BCE"], results)

    print("\n=== Проверочные примеры (конфигурация BCE) ===")
    tests = [(-8, -8), (-8, 3), (3, -8), (3, 3),
             (0, 0), (-3, 2), (2, -3)]
    for a, b in tests:
        norm, real, cls = predict(a, b, chosen["BCE"])
        print(f"A={a:5.1f}, B={b:5.1f}: y_norm={norm:.4f}, y_real={real:.4f}, класс={cls:g}")

    while True:
        raw = input("\nВведите A и B через пробел (q для выхода): ").strip()
        if raw.lower() == "q":
            break
        try:
            A, B = map(float, raw.split())
            if not (-10 <= A <= 10 and -10 <= B <= 10):
                print("Значения должны быть в диапазоне [-10, 10].")
                continue
            norm, real, cls = predict(A, B, chosen["BCE"])
            print(f"Нормализованный выход: {norm:.4f}")
            print(f"Выход в исходной шкале: {real:.4f}")
            print(f"Ближайший класс: {cls:g}")
        except ValueError:
            print("Введите два числа или q.")


if __name__ == "__main__":
    main()
