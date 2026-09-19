import numpy as np
import matplotlib.pyplot as plt

c0 = -4
c1 = -9

X_raw = np.array([
    [-4, -4],
    [-4, -9],
    [-9, -4],
    [-9, -9]
])

e_raw = np.array([-4, -9, -9, -4])

def sigmoid(S):
    return 1 / (1 + np.exp(-S))

def sigmoid_derivative(S):
    y = sigmoid(S)
    return y * (1 - y)

def normalize_value(v, c0, c1):
    return (v - c0) / (c1 - c0)

def denormalize_value(vn, c0, c1):
    return c0 + vn * (c1 - c0)


def normalize_dataset(X, c0, c1):
    return (X - c0) / (c1 - c0)

def forward(x, W1, T1, W2, T2):
    S1 = np.dot(x, W1) - T1
    h = sigmoid(S1)
    S2 = np.dot(h, W2) - T2
    y = sigmoid(S2)

    return S1, h, S2, y

def predict(x, W1, T1, W2, T2):
    _, _, _, y = forward(x, W1, T1, W2, T2)
    return y

def mse_delta_output(y, e, S2):
    return (y - e) * sigmoid_derivative(S2)

def bce_delta_output(y, e):

    return (y - e)

def train_network(X, e, alpha, Ee, max_epochs, loss_type, seed):
    np.random.seed(seed)

    W1 = np.random.uniform(-0.1, 0.1, (2, 2))
    T1 = np.random.uniform(-0.1, 0.1, 2)
    W2 = np.random.uniform(-0.1, 0.1, 2)
    T2 = np.random.uniform(-0.1, 0.1, 1)[0]

    errors = []
    epoch = 0

    while True:
        for xi, ei in zip(X, e):
            S1, h, S2, y = forward(xi, W1, T1, W2, T2)

            if loss_type == "MSE":
                delta2 = mse_delta_output(y, ei, S2)
            else:
                delta2 = bce_delta_output(y, ei)

            delta1 = sigmoid_derivative(S1) * W2 * delta2

            W2 -= alpha * h * delta2
            T2 += alpha * delta2

            W1 -= alpha * np.outer(xi, delta1)
            T1 += alpha * delta1

        Es = 0.0
        for xi, ei in zip(X, e):
            _, _, _, y = forward(xi, W1, T1, W2, T2)
            if loss_type == "MSE":
                Es += 0.5 * (y - ei) ** 2
            else:
                Es += -(ei * np.log(y + 1e-9) + (1 - ei) * np.log(1 - y + 1e-9))

        Es = float(Es)
        errors.append(Es)
        epoch += 1

        if Es <= Ee or epoch >= max_epochs:
            break

    return W1, T1, W2, T2, errors, epoch

def evaluate_network(X_raw, e_raw, W1, T1, W2, T2):
    Xn = normalize_dataset(X_raw, c0, c1)
    en = np.array([normalize_value(v, c0, c1) for v in e_raw])

    correct = 0
    abs_errors = []

    for xi_raw, ei_raw, ei_n in zip(X_raw, e_raw, en):
        xi_n = normalize_dataset(np.array([xi_raw]), c0, c1)[0]
        y_n = predict(xi_n, W1, T1, W2, T2)
        y_real = denormalize_value(y_n, c0, c1)

        if abs(y_real - ei_raw) < abs(y_real - (c0 if ei_raw == c1 else c1)):
            correct += 1

        abs_errors.append(abs(y_real - ei_raw))

    accuracy = correct / len(X_raw)
    mae = np.mean(abs_errors)

    return accuracy, mae

def run_experiments(Xn, en, alpha_mse, Ee_mse, alpha_bce, Ee_bce, max_epochs):
    results = {"A": [], "B": []}

    seeds = [1, 2, 3, 4, 5]

    for seed in seeds:
        W1, T1, W2, T2, errors, epochs = train_network(
            Xn, en, alpha_mse, Ee_mse, max_epochs, "MSE", seed
        )
        acc, mae = evaluate_network(X_raw, e_raw, W1, T1, W2, T2)
        results["A"].append({
            "seed": seed,
            "epochs": epochs,
            "final_error": errors[-1],
            "errors": errors,
            "accuracy": acc,
            "mae": mae,
            "W1": W1, "T1": T1, "W2": W2, "T2": T2
        })

        W1, T1, W2, T2, errors, epochs = train_network(
            Xn, en, alpha_bce, Ee_bce, max_epochs, "BCE", seed
        )
        acc, mae = evaluate_network(X_raw, e_raw, W1, T1, W2, T2)
        results["B"].append({
            "seed": seed,
            "epochs": epochs,
            "final_error": errors[-1],
            "errors": errors,
            "accuracy": acc,
            "mae": mae,
            "W1": W1, "T1": T1, "W2": W2, "T2": T2
        })

    return results

def plot_convergence(results):
    rep_A = results["A"][0]
    rep_B = results["B"][0]

    plt.figure(figsize=(8, 5))
    plt.plot(rep_A["errors"], label="Конфигурация A (MSE)")
    plt.plot(rep_B["errors"], label="Конфигурация B (BCE)")
    plt.xlabel("Эпоха")
    plt.ylabel("Суммарная ошибка Es")
    plt.title("Сходимость MLP 2-2-1: MSE vs BCE")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_epochs_scatter(results):
    seeds = [r["seed"] for r in results["A"]]
    epochs_A = [r["epochs"] for r in results["A"]]
    epochs_B = [r["epochs"] for r in results["B"]]

    x = np.arange(len(seeds))

    plt.figure(figsize=(8, 5))
    plt.bar(x - 0.15, epochs_A, width=0.3, label="MSE (A)")
    plt.bar(x + 0.15, epochs_B, width=0.3, label="BCE (B)")
    plt.xticks(x, [f"seed {s}" for s in seeds])
    plt.xlabel("Запуск (seed)")
    plt.ylabel("Число эпох до остановки")
    plt.title("Разброс числа эпох по 5 запускам")
    plt.legend()
    plt.grid(True, axis="y")
    plt.show()

def plot_decision_surface(W1, T1, W2, T2, title, use_real_scale=True):
    A_vals = np.linspace(-10, 10, 200)
    B_vals = np.linspace(-10, 10, 200)

    grid = np.zeros((len(A_vals), len(B_vals)))

    for i, a in enumerate(A_vals):
        for j, b in enumerate(B_vals):
            x_raw = np.array([a, b])
            x_n = normalize_dataset(np.array([x_raw]), c0, c1)[0]
            y_n = predict(x_n, W1, T1, W2, T2)
            if use_real_scale:
                y_real = denormalize_value(y_n, c0, c1)
                grid[i, j] = y_real
            else:
                grid[i, j] = y_n

    plt.figure(figsize=(6, 5))
    plt.imshow(
        grid.T,
        extent=[-10, 10, -10, 10],
        origin="lower",
        aspect="auto",
        cmap="coolwarm"
    )
    plt.colorbar(label="Выход сети" + (" (реальная шкала)" if use_real_scale else " (нормализованная)"))
    plt.scatter(X_raw[:, 0], X_raw[:, 1], c=e_raw, cmap="coolwarm", edgecolors="k")
    plt.xlabel("A")
    plt.ylabel("B")
    plt.title(title)
    plt.grid(False)
    plt.show()

def plot_mae_comparison(results):
    mae_A = [r["mae"] for r in results["A"]]
    mae_B = [r["mae"] for r in results["B"]]

    plt.figure(figsize=(6, 5))
    plt.bar([0, 1], [np.mean(mae_A), np.mean(mae_B)],
            tick_label=["MSE (A)", "BCE (B)"])
    plt.ylabel("Средняя абсолютная ошибка (MAE)")
    plt.title("Сравнение точности восстановления шкалы [c0; c1]")
    plt.grid(True, axis="y")
    plt.show()

def run_function_mode(W1, T1, W2, T2):
    print("\nРежим функционирования сети.")
    print("Введите пару чисел A B из диапазона [-10; 10], или 'q' для выхода.")

    while True:
        user_input = input("A B: ")
        if user_input.lower().strip() == "q":
            break

        try:
            a_str, b_str = user_input.split()
            a = float(a_str)
            b = float(b_str)
        except ValueError:
            print("Нужно ввести два числа через пробел.")
            continue

        if not (-10 <= a <= 10 and -10 <= b <= 10):
            print("Числа должны быть в диапазоне [-10; 10].")
            continue

        x_raw = np.array([a, b])
        x_n = normalize_dataset(np.array([x_raw]), c0, c1)[0]
        y_n = predict(x_n, W1, T1, W2, T2)
        y_real = denormalize_value(y_n, c0, c1)

        if abs(y_real - c0) < abs(y_real - c1):
            class_result = c0
        else:
            class_result = c1

        print(f"Выход в нормализованной шкале: {float(y_n):.4f}")
        print(f"Выход в исходной шкале: {float(y_real):.4f}")
        print(f"Ближайший класс: {class_result}\n")

def main():

    Xn = normalize_dataset(X_raw, c0, c1)
    en = np.array([normalize_value(v, c0, c1) for v in e_raw])


    alpha_mse = 0.9
    Ee_mse = 0.001
    alpha_bce = 0.9
    Ee_bce = 0.01
    max_epochs = 10000


    results = run_experiments(Xn, en, alpha_mse, Ee_mse, alpha_bce, Ee_bce, max_epochs)

    print("Результаты конфигурации A (MSE):")
    for r in results["A"]:
        print(f"seed={r['seed']}, epochs={r['epochs']}, Es={r['final_error']:.6f}, "
              f"accuracy={r['accuracy']:.2f}, MAE={r['mae']:.4f}")

    print("\nРезультаты конфигурации B (BCE):")
    for r in results["B"]:
        print(f"seed={r['seed']}, epochs={r['epochs']}, Es={r['final_error']:.6f}, "
              f"accuracy={r['accuracy']:.2f}, MAE={r['mae']:.4f}")

    plot_convergence(results)
    plot_epochs_scatter(results)

    rep_A = results["A"][0]
    rep_B = results["B"][0]

    plot_decision_surface(
        rep_A["W1"], rep_A["T1"], rep_A["W2"], rep_A["T2"],
        "Разделяющая поверхность (Конфигурация A, MSE)", use_real_scale=True
    )

    plot_decision_surface(
        rep_B["W1"], rep_B["T1"], rep_B["W2"], rep_B["T2"],
        "Разделяющая поверхность (Конфигурация B, BCE)", use_real_scale=True
    )

    plot_mae_comparison(results)

    run_function_mode(rep_B["W1"], rep_B["T1"], rep_B["W2"], rep_B["T2"])

if __name__ == "__main__":
    main()
