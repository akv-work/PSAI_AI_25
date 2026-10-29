import numpy as np
import matplotlib.pyplot as plt

c0 = -4
c1 = -9

X_raw = np.array([
    [-4, -4],
    [-4, -9],
    [-9, -4],
    [-9, -9]
], dtype=float)

e_raw = np.array([-4, -9, -9, -4], dtype=float)


def sigmoid(S):
    return 1 / (1 + np.exp(-np.clip(S, -500, 500)))


def sigmoid_derivative_from_h(h):
    return h * (1 - h)


def relu(S):
    return np.maximum(0, S)


def relu_derivative_from_S(S):
    return (S > 0).astype(float)


def normalize_value(v, c0, c1):
    return (v - c0) / (c1 - c0)


def denormalize_value(vn, c0, c1):
    return c0 + vn * (c1 - c0)


def normalize_dataset(X, c0, c1):
    return (X - c0) / (c1 - c0)


def forward(x, W1, T1, W2, T2, hidden_activation="sigmoid"):
    S1 = np.dot(x, W1) - T1
    if hidden_activation == "sigmoid":
        h = sigmoid(S1)
    else:
        h = relu(S1)
    S2 = np.dot(h, W2) - T2
    y = sigmoid(S2)
    return S1, h, S2, y


def predict(x, W1, T1, W2, T2, hidden_activation="sigmoid"):
    _, _, _, y = forward(x, W1, T1, W2, T2, hidden_activation)
    return y


def init_weights(seed, hidden_activation):
    rng = np.random.RandomState(seed)
    if hidden_activation == "sigmoid":
        limit = 0.8
        limit2 = 0.8
    else:
        limit = 0.7
        limit2 = 0.7
    W1 = rng.uniform(-limit, limit, (2, 2))
    T1 = rng.uniform(-0.1, 0.1, 2)
    W2 = rng.uniform(-limit2, limit2, 2)
    T2 = rng.uniform(-0.1, 0.1)
    return W1, T1, W2, T2


def train_network(X, e, alpha, Ee, max_epochs, hidden_activation="sigmoid", seed=1):
    W1, T1, W2, T2 = init_weights(seed, hidden_activation)

    vW1 = np.zeros_like(W1)
    vT1 = np.zeros_like(T1)
    vW2 = np.zeros_like(W2)
    vT2 = 0.0
    mu = 0.9

    errors = []
    epoch = 0

    while True:
        for xi, ei in zip(X, e):
            S1, h, S2, y = forward(xi, W1, T1, W2, T2, hidden_activation)

            delta2 = (y - ei)

            if hidden_activation == "sigmoid":
                d_hidden = sigmoid_derivative_from_h(h)
            else:
                d_hidden = relu_derivative_from_S(S1)

            delta1 = d_hidden * W2 * delta2

            gW2 = h * delta2
            gT2 = -delta2
            gW1 = np.outer(xi, delta1)
            gT1 = -delta1

            vW2 = mu * vW2 - alpha * gW2
            vT2 = mu * vT2 - alpha * gT2
            vW1 = mu * vW1 - alpha * gW1
            vT1 = mu * vT1 - alpha * gT1

            W2 += vW2
            T2 += vT2
            W1 += vW1
            T1 += vT1

        Es = 0.0
        for xi, ei in zip(X, e):
            _, _, _, y = forward(xi, W1, T1, W2, T2, hidden_activation)
            Es += -(ei * np.log(y + 1e-9) + (1 - ei) * np.log(1 - y + 1e-9))
        Es = float(Es)
        errors.append(Es)
        epoch += 1

        if Es <= Ee or epoch >= max_epochs:
            break

    return W1, T1, W2, T2, errors, epoch


def evaluate_network(X_raw, e_raw, W1, T1, W2, T2, hidden_activation="sigmoid"):
    correct = 0
    abs_errors = []

    for xi_raw, ei_raw in zip(X_raw, e_raw):
        xi_n = normalize_dataset(np.array([xi_raw]), c0, c1)[0]
        y_n = predict(xi_n, W1, T1, W2, T2, hidden_activation)
        y_real = denormalize_value(y_n, c0, c1)

        if abs(y_real - c0) < abs(y_real - c1):
            predicted_class = c0
        else:
            predicted_class = c1

        if predicted_class == ei_raw:
            correct += 1

        abs_errors.append(abs(y_real - ei_raw))

    accuracy = correct / len(X_raw)
    mae = float(np.mean(abs_errors))
    return accuracy, mae


def run_experiments(Xn, en, alpha, Ee, max_epochs, seeds):
    results = {"A": [], "B": []}

    for seed in seeds:
        W1, T1, W2, T2, errors, epochs = train_network(
            Xn, en, alpha, Ee, max_epochs, hidden_activation="sigmoid", seed=seed
        )
        acc, mae = evaluate_network(X_raw, e_raw, W1, T1, W2, T2, "sigmoid")
        results["A"].append({
            "seed": seed, "epochs": epochs, "final_error": errors[-1],
            "errors": errors, "accuracy": acc, "mae": mae,
            "W1": W1, "T1": T1, "W2": W2, "T2": T2
        })

        W1, T1, W2, T2, errors, epochs = train_network(
            Xn, en, alpha, Ee, max_epochs, hidden_activation="relu", seed=seed
        )
        acc, mae = evaluate_network(X_raw, e_raw, W1, T1, W2, T2, "relu")
        results["B"].append({
            "seed": seed, "epochs": epochs, "final_error": errors[-1],
            "errors": errors, "accuracy": acc, "mae": mae,
            "W1": W1, "T1": T1, "W2": W2, "T2": T2
        })

    return results


def pick_best(results_key):
    best = max(results_key, key=lambda r: (r["accuracy"], -r["mae"]))
    return best


def plot_convergence(results):
    rep_A = pick_best(results["A"])
    rep_B = pick_best(results["B"])

    plt.figure(figsize=(10, 5))
    plt.plot(rep_A["errors"], label=f"A (Sigmoid, seed={rep_A['seed']}, acc={rep_A['accuracy']:.2f})",
             color="steelblue", linewidth=2)
    plt.plot(rep_B["errors"], label=f"B (ReLU, seed={rep_B['seed']}, acc={rep_B['accuracy']:.2f})",
             color="orange", linewidth=2)
    plt.xlabel("Эпоха")
    plt.ylabel("Суммарная ошибка Es (BCE)")
    plt.title("Сходимость MLP 2-2-1: Sigmoid vs ReLU")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_epochs_scatter(results):
    seeds = [r["seed"] for r in results["A"]]
    epochs_A = [r["epochs"] for r in results["A"]]
    epochs_B = [r["epochs"] for r in results["B"]]
    acc_A = [r["accuracy"] for r in results["A"]]
    acc_B = [r["accuracy"] for r in results["B"]]

    x = np.arange(len(seeds))

    plt.figure(figsize=(12, 5))
    bars_A = plt.bar(x - 0.2, epochs_A, width=0.4, label="Sigmoid (A)", color="steelblue")
    bars_B = plt.bar(x + 0.2, epochs_B, width=0.4, label="ReLU (B)", color="orange")

    for bar, acc in zip(bars_A, acc_A):
        if acc < 1.0:
            bar.set_color("lightgray")
            bar.set_edgecolor("red")
            bar.set_linewidth(2)
    for bar, acc in zip(bars_B, acc_B):
        if acc < 1.0:
            bar.set_color("lightgray")
            bar.set_edgecolor("red")
            bar.set_linewidth(2)

    plt.xticks(x, [f"{s}" for s in seeds])
    plt.xlabel("Запуск (seed)")
    plt.ylabel("Число эпох до остановки")
    plt.title("Разброс числа эпох (серые с красной рамкой = accuracy < 1.0)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_decision_surface(W1, T1, W2, T2, title, hidden_activation):
    A_vals = np.linspace(-10, 10, 250)
    B_vals = np.linspace(-10, 10, 250)

    grid = np.zeros((len(A_vals), len(B_vals)))

    for i, a in enumerate(A_vals):
        for j, b in enumerate(B_vals):
            x_raw = np.array([a, b])
            x_n = normalize_dataset(np.array([x_raw]), c0, c1)[0]
            y_n = predict(x_n, W1, T1, W2, T2, hidden_activation)
            grid[i, j] = denormalize_value(y_n, c0, c1)

    plt.figure(figsize=(8, 6))
    plt.imshow(
        grid.T,
        extent=[-10, 10, -10, 10],
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        vmin=c1, vmax=c0
    )
    plt.colorbar(label="Выход сети (реальная шкала)")
    plt.scatter(X_raw[:, 0], X_raw[:, 1], c=e_raw, cmap="coolwarm",
                edgecolors="k", s=180, vmin=c1, vmax=c0)
    plt.xlabel("A")
    plt.ylabel("B")
    plt.title(title)
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def plot_mae_comparison(results):
    mae_A_all = [r["mae"] for r in results["A"]]
    mae_B_all = [r["mae"] for r in results["B"]]

    mae_A_ok = [r["mae"] for r in results["A"] if r["accuracy"] == 1.0]
    mae_B_ok = [r["mae"] for r in results["B"] if r["accuracy"] == 1.0]

    val_A_all = np.mean(mae_A_all) if mae_A_all else 0
    val_B_all = np.mean(mae_B_all) if mae_B_all else 0
    val_A_ok = np.mean(mae_A_ok) if mae_A_ok else 0
    val_B_ok = np.mean(mae_B_ok) if mae_B_ok else 0

    plt.figure(figsize=(8, 5))
    x = np.arange(2)
    b1 = plt.bar(x - 0.2, [val_A_all, val_B_all], width=0.4,
                 label="Все запуски", color="lightgray", edgecolor="black")
    b2 = plt.bar(x + 0.2, [val_A_ok, val_B_ok], width=0.4,
                 label="Только успешные (acc=1.0)",
                 color=["steelblue", "orange"], edgecolor="black")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                     f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(x, ["Sigmoid (A)", "ReLU (B)"])
    plt.ylabel("Средняя абсолютная ошибка (MAE)")
    plt.title("Сравнение MAE: все запуски vs успешные")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_accuracy_histogram(results):
    acc_A = [r["accuracy"] for r in results["A"]]
    acc_B = [r["accuracy"] for r in results["B"]]

    plt.figure(figsize=(8, 5))
    bins = np.arange(0, 1.1, 0.1)
    plt.hist([acc_A, acc_B], bins=bins, label=["Sigmoid (A)", "ReLU (B)"],
             color=["steelblue", "orange"], edgecolor="black")
    plt.xlabel("Accuracy")
    plt.ylabel("Число запусков")
    plt.title("Распределение accuracy по запускам")
    plt.xticks(bins)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_function_mode(W1, T1, W2, T2, hidden_activation):
    print("\n" + "=" * 60)
    print(f"Режим функционирования (активация: {hidden_activation})")
    print("Введите пару чисел A B из [-10; 10] или 'q' для выхода.")
    print("=" * 60)

    while True:
        user_input = input("A B: ")
        if user_input.lower().strip() == "q":
            break

        try:
            a_str, b_str = user_input.split()
            a, b = float(a_str), float(b_str)
        except ValueError:
            print("Нужно ввести два числа через пробел.")
            continue

        if not (-10 <= a <= 10 and -10 <= b <= 10):
            print("Числа должны быть в [-10; 10].")
            continue

        x_n = normalize_dataset(np.array([[a, b]]), c0, c1)[0]
        y_n = predict(x_n, W1, T1, W2, T2, hidden_activation)
        y_real = denormalize_value(y_n, c0, c1)
        cls = c0 if abs(y_real - c0) < abs(y_real - c1) else c1

        print(f"  Норм. выход: {float(y_n):.4f}")
        print(f"  Реальный выход: {float(y_real):.4f}")
        print(f"  Класс: {cls}\n")


def print_comparison_table(results):
    print("\n" + "=" * 100)
    print("ТАБЛИЦА СРАВНЕНИЯ")
    print("=" * 100)
    print(f"{'Конфигурация':<20}{'Успешных':<12}{'Эпохи (min/mean/max)':<28}"
          f"{'Es (mean)':<15}{'Accuracy (mean)':<18}{'MAE (mean)':<12}")
    print("-" * 100)

    for key, name in [("A", "Sigmoid (A)"), ("B", "ReLU (B)")]:
        epochs = [r["epochs"] for r in results[key]]
        es = [r["final_error"] for r in results[key]]
        acc = [r["accuracy"] for r in results[key]]
        mae = [r["mae"] for r in results[key]]
        n_success = sum(1 for a in acc if a == 1.0)
        print(f"{name:<20}"
              f"{n_success}/{len(acc):<10}"
              f"{min(epochs)}/{np.mean(epochs):.1f}/{max(epochs):<18}"
              f"{np.mean(es):<15.6f}"
              f"{np.mean(acc):<18.3f}"
              f"{np.mean(mae):<12.4f}")
    print("=" * 100)


def main():
    Xn = normalize_dataset(X_raw, c0, c1)
    en = np.array([normalize_value(v, c0, c1) for v in e_raw])

    print("Нормализованные входы:")
    print(Xn)
    print("Нормализованные эталоны:", en)
    print()

    alpha = 0.5
    Ee = 0.01
    max_epochs = 20000
    seeds = list(range(1, 31))

    results = run_experiments(Xn, en, alpha, Ee, max_epochs, seeds)

    for key, name in [("A", "Sigmoid"), ("B", "ReLU")]:
        print(f"\nРезультаты {key} ({name}):")
        for r in results[key]:
            print(f"  seed={r['seed']:>2}, epochs={r['epochs']:>5}, "
                  f"Es={r['final_error']:.6f}, acc={r['accuracy']:.2f}, MAE={r['mae']:.4f}")

    plot_convergence(results)
    plot_epochs_scatter(results)
    plot_accuracy_histogram(results)

    best_A = pick_best(results["A"])
    best_B = pick_best(results["B"])

    print(f"\nЛучший A: seed={best_A['seed']}, accuracy={best_A['accuracy']}, MAE={best_A['mae']:.4f}")
    print(f"Лучший B: seed={best_B['seed']}, accuracy={best_B['accuracy']}, MAE={best_B['mae']:.4f}")

    plot_decision_surface(
        best_A["W1"], best_A["T1"], best_A["W2"], best_A["T2"],
        f"Разделяющая поверхность (A, Sigmoid, seed={best_A['seed']}, acc={best_A['accuracy']:.2f})",
        hidden_activation="sigmoid"
    )

    plot_decision_surface(
        best_B["W1"], best_B["T1"], best_B["W2"], best_B["T2"],
        f"Разделяющая поверхность (B, ReLU, seed={best_B['seed']}, acc={best_B['accuracy']:.2f})",
        hidden_activation="relu"
    )

    plot_mae_comparison(results)
    print_comparison_table(results)

    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ НА ОБУЧАЮЩИХ ПРИМЕРАХ (B, ReLU, лучший seed)")
    print("=" * 60)
    for xi_raw, ei_raw in zip(X_raw, e_raw):
        xi_n = normalize_dataset(np.array([xi_raw]), c0, c1)[0]
        y_n = predict(xi_n, best_B["W1"], best_B["T1"], best_B["W2"], best_B["T2"], "relu")
        y_real = denormalize_value(y_n, c0, c1)
        cls = c0 if abs(y_real - c0) < abs(y_real - c1) else c1
        print(f"  Вход {xi_raw}, эталон {ei_raw}, выход {y_real:.4f}, класс {cls}")

    print("\n" + "=" * 60)
    print("ДОПОЛНИТЕЛЬНЫЕ ПАРЫ (B, ReLU)")
    print("=" * 60)
    extra_pairs = [(-6.5, -7.0), (-8.0, -5.0), (-5.5, -8.5)]
    for xi_raw in extra_pairs:
        xi_n = normalize_dataset(np.array([xi_raw]), c0, c1)[0]
        y_n = predict(xi_n, best_B["W1"], best_B["T1"], best_B["W2"], best_B["T2"], "relu")
        y_real = denormalize_value(y_n, c0, c1)
        cls = c0 if abs(y_real - c0) < abs(y_real - c1) else c1
        print(f"  Вход {xi_raw}, выход {y_real:.4f}, класс {cls}")

    run_function_mode(best_B["W1"], best_B["T1"], best_B["W2"], best_B["T2"], "relu")


if __name__ == "__main__":
    main()