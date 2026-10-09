import numpy as np
import matplotlib.pyplot as plt

def activation(S):
    return 1 / (1 + np.exp(-S))

def activation_derivative(S):
    y = activation(S)
    return y * (1 - y)

def normalize(value, c0, c1):
    return (value - c0) / (c1 - c0)

def denormalize(value, c0, c1):
    return c0 + value * (c1 - c0)

def forward(x, W1, T1, W2, T2):
    S1 = np.dot(x, W1) - T1
    h = activation(S1)
    S2 = np.dot(h, W2) - T2
    y = activation(S2)
    return S1, h, S2, y

def predict(x, W1, T1, W2, T2):
    _, _, _, y = forward(x, W1, T1, W2, T2)
    return y

def online_train_mse(X, e, alpha=0.9, Ee=0.001, max_epochs=10000, seed=42):

    np.random.seed(seed)

    W1 = np.random.uniform(-0.1, 0.1, (2, 2))
    T1 = np.random.uniform(-0.1, 0.1, 2)

    W2 = np.random.uniform(-0.1, 0.1, 2)
    T2 = np.random.uniform(-0.1, 0.1)

    errors = []
    epoch = 0

    while True:
        for xi, ei in zip(X, e):

            S1, h, S2, y = forward(xi, W1, T1, W2, T2)

            delta2 = ((y - ei) * activation_derivative(S2))
            delta1 = (activation_derivative(S1) * W2 * delta2)

            W2 = W2 - alpha * h * delta2
            T2 = T2 + alpha * delta2
            W1 = W1 - alpha * np.outer(xi, delta1)
            T1 = T1 + alpha * delta1

        Es = 0
        for xi, ei in zip(X, e):
            y = predict(xi, W1, T1, W2, T2)
            Es += 0.5 * (y - ei) ** 2
        errors.append(Es)
        epoch += 1

        if Es <= Ee or epoch >= max_epochs:
            break

    return W1, T1, W2, T2, errors

def online_train_bce(X, e, alpha=0.9, Ee=0.01, max_epochs=10000, seed=42):

    np.random.seed(seed)

    W1 = np.random.uniform(-0.1, 0.1, (2, 2))
    T1 = np.random.uniform(-0.1, 0.1, 2)

    W2 = np.random.uniform(-0.1, 0.1, 2)
    T2 = np.random.uniform(-0.1, 0.1)

    errors = []
    epoch = 0

    while True:
        for xi, ei in zip(X, e):
            S1, h, S2, y = forward(xi, W1, T1, W2, T2)

            delta2 = y - ei
            delta1 = (activation_derivative(S1) * W2 * delta2)

            W2 = W2 - alpha * h * delta2
            T2 = T2 + alpha * delta2
            W1 = W1 - alpha * np.outer(xi, delta1)
            T1 = T1 + alpha * delta1

        Es = 0
        for xi, ei in zip(X, e):
            y = predict(xi, W1, T1, W2, T2)
            y = np.clip(y, 1e-10, 1 - 1e-10) #защита от log(0)
            Es += -(ei * np.log(y) + (1 - ei) * np.log(1 - y))
        errors.append(Es)
        epoch += 1

        if Es <= Ee or epoch >= max_epochs:
            break

    return W1, T1, W2, T2, errors


def calculate_accuracy(X, e, W1, T1, W2, T2):
    correct = 0
    for xi, ei in zip(X, e):
        y = predict(xi, W1, T1, W2, T2)

        predicted_class = 1 if y >= 0.5 else 0

        if predicted_class == ei:
            correct += 1

    return correct / len(e) * 100

def calculate_mae(X, e, W1, T1, W2, T2, c0, c1):
    errors = []
    for xi, ei in zip(X, e):
        y = predict(xi, W1, T1, W2, T2)
        expected = denormalize(ei, c0, c1)
        predicted = denormalize(y, c0, c1)
        errors.append(abs(predicted - expected))
    return np.mean(errors)

def print_results(name, X, e, W1, T1, W2, T2, errors, c0, c1):
    print("\n\n", name)
    print("Эпох обучения:", len(errors))
    print("Финальная ошибка:", errors[-1])

    accuracy = calculate_accuracy(X, e, W1, T1, W2, T2)
    mae = calculate_mae(X, e, W1, T1, W2, T2, c0, c1)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Средняя абсолютная ошибка: {mae:.6f}")

    print("\nОтветы сети:")

    for xi, ei in zip(X, e):
        y = predict(xi, W1, T1, W2, T2)
        expected = denormalize(ei, c0, c1)
        predicted = denormalize(y, c0, c1)
        print(f"Вход: {denormalize(xi, c0, c1)}, " f"Ожидалось: {expected:.4f}, " f"Получено: {predicted:.4f}, " f"Норм. выход: {y:.4f}")

def show_convergence_plot(mse_errors, bce_errors, mse_Ee, bce_Ee):
    plt.figure(figsize=(10, 6))
    plt.plot(mse_errors, label="MSE")

    plt.plot(bce_errors, label="BCE")

    plt.axhline(mse_Ee, linestyle="--", label="Порог MSE")

    plt.axhline(bce_Ee, linestyle=":", color="orange", label="Порог BCE")

    plt.xlabel("Epoch")
    plt.ylabel("Суммарная ошибка Es")

    plt.title("Сходимость MSE и Binary Cross-Entropy")

    plt.legend()
    plt.grid()

    plt.show()


#5 запусков на разных seed
def run_multiple_experiments(X, e, c0, c1):
    seeds = [1, 2, 3, 4, 5]

    mse_results = []
    bce_results = []

    print("\n")
    print("5 запусков MSE")

    for seed in seeds:

        W1, T1, W2, T2, errors = online_train_mse(X, e, alpha=0.9, Ee=0.001, max_epochs=10000, seed=seed)

        accuracy = calculate_accuracy(X, e, W1, T1, W2, T2)

        mae = calculate_mae(X, e, W1, T1, W2, T2, c0, c1)

        reached = errors[-1] <= 0.001

        mse_results.append({ "seed": seed, "epochs": len(errors), "error": errors[-1], "accuracy": accuracy, "mae": mae, "reached": reached })

        print(f"Seed={seed}: " f"эпох={len(errors)}, " f"ошибка={errors[-1]:.6f}, " f"accuracy={accuracy:.2f}%, " f"MAE={mae:.6f}, " f"критерий={'да' if reached else 'нет'}")

    print("\n")
    print("5 запусков BCE")

    for seed in seeds:

        W1, T1, W2, T2, errors = online_train_bce(X, e, alpha=0.9, Ee=0.01, max_epochs=10000, seed=seed)

        accuracy = calculate_accuracy(X, e, W1, T1, W2, T2)

        mae = calculate_mae(X, e, W1, T1, W2, T2, c0, c1)

        reached = errors[-1] <= 0.01

        bce_results.append({ "seed": seed, "epochs": len(errors), "error": errors[-1], "accuracy": accuracy, "mae": mae, "reached": reached })

        print(f"Seed={seed}: " f"эпох={len(errors)}, " f"ошибка={errors[-1]:.6f}, " f"accuracy={accuracy:.2f}%, " f"MAE={mae:.6f}, " f"критерий={'да' if reached else 'нет'}")

    return mse_results, bce_results

#разброс числа эпох
def show_epochs_plot(mse_results, bce_results):
    seeds = [1, 2, 3, 4, 5]

    mse_epochs = [
        result["epochs"]
        for result in mse_results
    ]

    bce_epochs = [
        result["epochs"]
        for result in bce_results
    ]

    x = np.arange(len(seeds))

    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(x - width / 2, mse_epochs, width, label="MSE")

    plt.bar(x + width / 2, bce_epochs, width, label="BCE")

    plt.xlabel("Seed")
    plt.ylabel("Количество эпох")

    plt.title("Разброс числа эпох по 5 запускам")

    plt.xticks(x, seeds)

    plt.legend()
    plt.grid(axis="y")

    plt.show()

#heatmap
def show_surface(W1, T1, W2, T2, X_original, e, c0, c1, title):
    values = np.linspace(-10, 10, 200)
    A, B = np.meshgrid(values, values)

    grid = np.stack([A.ravel(), B.ravel()], axis=1)
    grid_normalized = normalize(grid, c0, c1)
    Y = []

    for x in grid_normalized:
        y = predict(x, W1, T1, W2, T2)
        Y.append(denormalize(y, c0, c1))

    Y = np.array(Y).reshape(A.shape)

    plt.figure(figsize=(8, 7))
    contour = plt.contourf(A, B, Y, levels=50)
    plt.colorbar(contour, label="Выход сети")

    #истинные классы
    for xi, ei in zip(X_original, e):
        class_value = denormalize(ei, c0, c1)

        if class_value == 4.0:
            color = "red"
        else:
            color = "green"

        plt.scatter(xi[0], xi[1], s=100, edgecolors="black", color=color, label=f"Класс {class_value}")

    plt.xlabel("A")
    plt.ylabel("B")

    plt.title(title)

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    plt.grid()

    #убираем повторяющиеся элементы легенды
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    plt.legend(unique.values(), unique.keys())
    plt.show()


#сравнение MAE
def show_mae_plot(mse_mae, bce_mae):
    plt.figure(figsize=(8, 6))
    plt.bar(["MSE", "BCE"], [mse_mae, bce_mae])
    plt.ylabel("Средняя абсолютная ошибка")
    plt.title("Сравнение точности восстановления исходной шкалы")
    plt.grid(axis="y")
    plt.show()


def interactive_mode(W1, T1, W2, T2, c0, c1):
    print("\n")
    while True:
        user_input = input("\nВведите пару чисел (A, B): ")
        if user_input.lower() == "q":
            break
        try:
            a, b = map(float, user_input.split())
        except ValueError:
            print("Введите два числа через пробел.")
            continue
        if not (-10 <= a <= 10 and -10 <= b <= 10):
            print("Числа должны быть " "в диапазоне [-10; 10]")
            continue

        x = np.array([a, b])
        x_normalized = normalize(x, c0, c1)
        y = predict(x_normalized, W1, T1, W2, T2)
        y_real = denormalize(y, c0, c1)
        if abs(y_real - c0) < abs(y_real - c1):
            class_result = c0
        else:
            class_result = c1

        print(f"Нормализованный выход: {y:.4f}")
        print(f"Выход в исходной шкале: " f"{y_real:.4f}")
        print(f"Ближайший класс: " f"{class_result}")

def main():
    c0 = 4
    c1 = -1

    x1 = np.array([ 4, 4, -1, -1 ])
    x2 = np.array([ 4, -1, 4, -1 ])

    e_original = np.array([ 4, -1, -1, 4 ])
    X_original = np.vstack([x1, x2]).T

    X = normalize(X_original, c0, c1)
    e = normalize(e_original, c0, c1)

    print("Исходные входы:")
    print(X_original)

    print("\nНормализованные входы:")
    print(X)

    print("\nИсходные классы:")
    print(e_original)

    print("\nНормализованные классы:")
    print(e)

    W1_mse, T1_mse, W2_mse, T2_mse, errors_mse = (online_train_mse(X, e, alpha=0.9, Ee=0.001, max_epochs=10000, seed=42))
    print_results("КОНФИГУРАЦИЯ A — MSE", X, e, W1_mse, T1_mse, W2_mse, T2_mse, errors_mse, c0, c1)

    W1_bce, T1_bce, W2_bce, T2_bce, errors_bce = (online_train_bce(X, e, alpha=0.4, Ee=0.01, max_epochs=10000, seed=42))
    print_results("КОНФИГУРАЦИЯ Б — BCE", X, e, W1_bce, T1_bce, W2_bce, T2_bce, errors_bce, c0, c1)

    #5 запусков
    mse_results, bce_results = (run_multiple_experiments(X, e, c0, c1))
    show_convergence_plot(errors_mse, errors_bce, 0.001, 0.01) #график схъодимости
    show_epochs_plot(mse_results, bce_results) #разброс эпох
    show_surface(W1_mse, T1_mse, W2_mse, T2_mse, X_original, e, c0, c1, "Выход сети MSE")
    show_surface(W1_bce, T1_bce, W2_bce, T2_bce, X_original, e, c0, c1, "Выход сети BCE")

    mse_mae = calculate_mae(X, e, W1_mse, T1_mse, W2_mse, T2_mse, c0, c1)
    bce_mae = calculate_mae(X, e, W1_bce, T1_bce, W2_bce, T2_bce, c0, c1)
    show_mae_plot(mse_mae, bce_mae)

    print("\n")
    print("Проверка 4 точек XOR")

    for x_original in X_original:
        x = normalize(x_original, c0, c1)
        y = predict(x, W1_bce, T1_bce, W2_bce, T2_bce)
        y_real = denormalize(y, c0, c1)
        if abs(y_real - c0) < abs(y_real - c1):
            class_result = c0
        else:
            class_result = c1
        print(f"A={x_original[0]}, " f"B={x_original[1]} -> " f"y={y:.4f}, " f"y_real={y_real:.4f}, " f"класс={class_result}")

    interactive_mode(W1_bce, T1_bce, W2_bce, T2_bce, c0, c1)


if __name__ == "__main__":
    main()