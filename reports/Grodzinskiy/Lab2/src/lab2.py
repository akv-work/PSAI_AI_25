import numpy as np
import matplotlib.pyplot as plt

c0 = 7
c1 = -7

X_raw = np.array([
    [7, 7],
    [7, -7],
    [-7, 7],
    [-7, -7]
], dtype=float)

e_raw = np.array([7, -7, -7, 7], dtype=float)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def dsigmoid(y):
    return y * (1 - y)

def normalize_lab1(x):
    return (x + 10) / 20

def denormalize_lab1(y):
    return y * 20 - 10

def normalize_bce(y):
    return (y - c0) / (c1 - c0)

def denormalize_bce(y):
    return c0 + y * (c1 - c0)

X_norm = np.array([[normalize_lab1(v) for v in row] for row in X_raw])
e_mse = np.array([normalize_lab1(v) for v in e_raw])
e_bce = np.array([normalize_bce(v) for v in e_raw])

def get_class(y_real):
    return c0 if abs(y_real - c0) < abs(y_real - c1) else c1

def train_network(X, e_target, alpha, Ee, max_epochs, loss_type, seed):
    np.random.seed(seed)
    W1 = np.random.randn(2, 2) * 0.1
    W2 = np.random.randn(2, 1) * 0.1
    b1 = np.zeros((1, 2))
    b2 = np.zeros((1, 1))

    errors = []
    epoch = 0

    while True:
        error = 0
        for i in range(len(X)):
            x = X[i:i+1]
            y = e_target[i:i+1]

            h = sigmoid(x @ W1 + b1)
            out = sigmoid(h @ W2 + b2)

            if loss_type == "MSE":
                error += np.sum((y - out) ** 2)
                d_out = (y - out) * dsigmoid(out)
            else: 
                error += -np.sum(y * np.log(out + 1e-9) + (1 - y) * np.log(1 - out + 1e-9))
                d_out = y - out

            d_h = d_out @ W2.T * dsigmoid(h)

            W2 += alpha * h.T @ d_out
            b2 += alpha * d_out
            W1 += alpha * x.T @ d_h
            b1 += alpha * d_h

        errors.append(error)
        epoch += 1

        if error <= Ee or epoch >= max_epochs:
            break

    return W1, b1, W2, b2, errors, epoch

def evaluate_network(W1, b1, W2, b2, loss_type):
    correct = 0
    abs_errors = []
    
    for i in range(len(X_raw)):
        x = X_norm[i:i+1]
        h = sigmoid(x @ W1 + b1)
        out = sigmoid(h @ W2 + b2)[0][0]
        
        if loss_type == "MSE":
            y_real = denormalize_lab1(out)
        else:
            y_real = denormalize_bce(out)
            
        target = e_raw[i]
        cls = get_class(y_real)
        
        if cls == target:
            correct += 1
        abs_errors.append(abs(y_real - target))
        
    return correct / len(X_raw), np.mean(abs_errors)

def run_experiments(alpha_mse, Ee_mse, alpha_bce, Ee_bce, max_epochs):
    results = {"A": [], "B": []}
    seeds = [1, 2, 3, 4, 5]

    for seed in seeds:
        # Конфигурация А (MSE)
        W1, b1, W2, b2, errors, epochs = train_network(X_norm, e_mse, alpha_mse, Ee_mse, max_epochs, "MSE", seed)
        acc, mae = evaluate_network(W1, b1, W2, b2, "MSE")
        results["A"].append({"seed": seed, "epochs": epochs, "final_error": errors[-1], 
                             "errors": errors, "accuracy": acc, "mae": mae, 
                             "W1": W1, "b1": b1, "W2": W2, "b2": b2})

        W1, b1, W2, b2, errors, epochs = train_network(X_norm, e_bce, alpha_bce, Ee_bce, max_epochs, "BCE", seed)
        acc, mae = evaluate_network(W1, b1, W2, b2, "BCE")
        results["B"].append({"seed": seed, "epochs": epochs, "final_error": errors[-1], 
                             "errors": errors, "accuracy": acc, "mae": mae, 
                             "W1": W1, "b1": b1, "W2": W2, "b2": b2})
    return results

def plot_convergence(results):
    plt.figure(figsize=(8, 5))
    plt.plot(results["A"][0]["errors"], label="Конфигурация A (MSE)")
    plt.plot(results["B"][0]["errors"], label="Конфигурация B (BCE)")
    plt.xlabel("Эпоха")
    plt.ylabel("Суммарная ошибка Es")
    plt.title("Сходимость MLP 2-2-1: MSE vs BCE (c0=7, c1=-7)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_epochs_bar(results):
    seeds = [r["seed"] for r in results["A"]]
    epochs_A = [r["epochs"] for r in results["A"]]
    epochs_B = [r["epochs"] for r in results["B"]]
    x = np.arange(len(seeds))

    plt.figure(figsize=(8, 5))
    plt.bar(x - 0.15, epochs_A, width=0.3, label="MSE (A)")
    plt.bar(x + 0.15, epochs_B, width=0.3, label="BCE (B)")
    plt.xticks(x, [f"seed {s}" for s in seeds])
    plt.xlabel("Запуск (seed)")
    plt.ylabel("Число эпох")
    plt.title("Разброс числа эпох по 5 запускам")
    plt.legend()
    plt.grid(True, axis="y")
    plt.show()

def plot_decision_surface(W1, b1, W2, b2, title, loss_type):
    A_vals = np.linspace(-10, 10, 200)
    B_vals = np.linspace(-10, 10, 200)
    grid = np.zeros((len(A_vals), len(B_vals)))

    for i, a in enumerate(A_vals):
        for j, b in enumerate(B_vals):
            x_n = np.array([[normalize_lab1(a), normalize_lab1(b)]])
            h = sigmoid(x_n @ W1 + b1)
            out = sigmoid(h @ W2 + b2)[0][0]
            
            if loss_type == "MSE":
                grid[i, j] = denormalize_lab1(out)
            else:
                grid[i, j] = denormalize_bce(out)

    plt.figure(figsize=(6, 5))
    plt.imshow(grid.T, extent=[-10, 10, -10, 10], origin="lower", aspect="auto", cmap="coolwarm")
    plt.colorbar(label="Выход сети (исходная шкала)")
    plt.scatter(X_raw[:, 0], X_raw[:, 1], c=e_raw, cmap="coolwarm", edgecolors="k", s=100)
    plt.xlabel("A")
    plt.ylabel("B")
    plt.title(title)
    plt.grid(False)
    plt.show()

def plot_mae_comparison(results):
    mae_A = [r["mae"] for r in results["A"]]
    mae_B = [r["mae"] for r in results["B"]]
    plt.figure(figsize=(6, 5))
    plt.bar([0, 1], [np.mean(mae_A), np.mean(mae_B)], tick_label=["MSE (A)", "BCE (B)"], color=['tab:blue', 'tab:orange'])
    plt.ylabel("Средняя абсолютная ошибка (MAE)")
    plt.title("Сравнение точности восстановления шкалы [c0; c1]")
    plt.grid(True, axis="y")
    plt.show()

def run_function_mode(W1, b1, W2, b2):
    print("\n=== Режим функционирования сети ===")
    test_pairs = [(7, 7), (7, -7), (-7, 7), (-7, -7), (5, 5), (0, 0), (-8, 9)]
    
    for a, b in test_pairs:
        x = np.array([[normalize_lab1(a), normalize_lab1(b)]])
        h = sigmoid(x @ W1 + b1)
        y_norm = sigmoid(h @ W2 + b2)[0][0]
        y_real = denormalize_bce(y_norm)
        cls = get_class(y_real)

        print(f"Вход: A={a:5.1f}, B={b:5.1f}")
        print(f"  Выход (норм): {float(y_norm):.4f}")
        print(f"  Выход (исх.): {float(y_real):.4f}")
        print(f"  Ближайший класс: {cls}\n")

def main():
    alpha_mse = 0.5
    Ee_mse = 0.01
    alpha_bce = 0.5
    Ee_bce = 0.01
    max_epochs = 10000

    results = run_experiments(alpha_mse, Ee_mse, alpha_bce, Ee_bce, max_epochs)

    print("=== Результаты конфигурации A (MSE) ===")
    for r in results["A"]:
        print(f"seed={r['seed']}, epochs={r['epochs']}, Es={r['final_error']:.6f}, accuracy={r['accuracy']:.2f}, MAE={r['mae']:.4f}")

    print("\n=== Результаты конфигурации B (BCE) ===")
    for r in results["B"]:
        print(f"seed={r['seed']}, epochs={r['epochs']}, Es={r['final_error']:.6f}, accuracy={r['accuracy']:.2f}, MAE={r['mae']:.4f}")

    plot_convergence(results)
    plot_epochs_bar(results)
    
    best_A = min(results["A"], key=lambda x: x["mae"])
    best_B = min(results["B"], key=lambda x: x["mae"])

    plot_decision_surface(best_A["W1"], best_A["b1"], best_A["W2"], best_A["b2"], "Разделяющая поверхность (Конфигурация A, MSE)", "MSE")
    plot_decision_surface(best_B["W1"], best_B["b1"], best_B["W2"], best_B["b2"], "Разделяющая поверхность (Конфигурация B, BCE)", "BCE")
    
    plot_mae_comparison(results)

    run_function_mode(best_B["W1"], best_B["b1"], best_B["W2"], best_B["b2"])

if __name__ == "__main__":
    main()