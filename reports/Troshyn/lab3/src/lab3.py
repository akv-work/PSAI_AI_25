import numpy as np
import matplotlib.pyplot as plt

def sigmoid(S):
    return 1 / (1 + np.exp(-S))

def sigmoid_derivative(S):
    y = sigmoid(S)
    return y * (1 - y)

def ReLU(S):
    return np.maximum(0, S)

def ReLU_derivative(S):
    return (S > 0).astype(float)

def normalize(value, c0, c1):
    return (value - c0) / (c1 - c0)

def denormalize(value, c0, c1):
    return c0 + value * (c1 - c0)

def forward(x, W1, T1, W2, T2, activation):
    S1 = np.dot(x, W1) - T1
    h = activation(S1)
    S2 = np.dot(h, W2) - T2
    y = sigmoid(S2)
    return S1, h, S2, y

def predict(x, W1, T1, W2, T2, activation):
    _, _, _, y = forward(x, W1, T1, W2, T2, activation)
    return y

def train(X, e, activation, activation_derivative, alpha=0.9, Ee=0.01, max_epochs=10000, seed=42):

    np.random.seed(seed)

    W1 = np.random.uniform(-0.1, 0.1, (2, 2))
    T1 = np.random.uniform(-0.1, 0.1, 2)

    W2 = np.random.uniform(-0.1, 0.1, 2)
    T2 = np.random.uniform(-0.1, 0.1)

    errors = []
    epoch = 0

    while True:
        for xi, ei in zip(X, e):
            S1, h, S2, y = forward(xi, W1, T1, W2, T2, activation)

            delta2 = y - ei
            delta1 = (activation_derivative(S1) * W2 * delta2)

            W2 = W2 - alpha * h * delta2
            T2 = T2 + alpha * delta2
            W1 = W1 - alpha * np.outer(xi, delta1)
            T1 = T1 + alpha * delta1

        Es = 0
        for xi, ei in zip(X, e):
            y = predict(xi, W1, T1, W2, T2, activation)
            y = np.clip(y, 1e-10, 1 - 1e-10) #защита от log(0)
            Es += -(ei * np.log(y) + (1 - ei) * np.log(1 - y))
        errors.append(Es)
        epoch += 1

        if Es <= Ee or epoch >= max_epochs:
            break

    return W1, T1, W2, T2, errors


def calculate_accuracy(X, e, W1, T1, W2, T2, activation):
    correct = 0
    for xi, ei in zip(X, e):
        y = predict(xi, W1, T1, W2, T2, activation)

        predicted_class = 1 if y >= 0.5 else 0

        if predicted_class == ei:
            correct += 1

    return correct / len(e) * 100

def calculate_mae(X, e, W1, T1, W2, T2, c0, c1, activation):
    errors = []
    for xi, ei in zip(X, e):
        y = predict(xi, W1, T1, W2, T2, activation)
        expected = denormalize(ei, c0, c1)
        predicted = denormalize(y, c0, c1)
        errors.append(abs(predicted - expected))
    return np.mean(errors)

def print_results(name, X, e, W1, T1, W2, T2, errors, c0, c1, activation):
    print("\n\n", name)
    print("Эпох обучения:", len(errors))
    print("Финальная ошибка:", errors[-1])

    accuracy = calculate_accuracy(X, e, W1, T1, W2, T2, activation)
    mae = calculate_mae(X, e, W1, T1, W2, T2, c0, c1, activation)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Средняя абсолютная ошибка: {mae:.6f}")

    print("\nОтветы сети:")

    for xi, ei in zip(X, e):
        y = predict(xi, W1, T1, W2, T2, activation)
        expected = denormalize(ei, c0, c1)
        predicted = denormalize(y, c0, c1)
        print(f"Вход: {denormalize(xi, c0, c1)}, " f"Ожидалось: {expected:.4f}, " f"Получено: {predicted:.4f}, " f"Норм. выход: {y:.4f}")

def show_convergence_plot(Sigmoid_errors, ReLU_errors, Sigmoid_Ee, ReLU_Ee):
    plt.figure(figsize=(10, 6))
    plt.plot(Sigmoid_errors, label="Sigmoid")

    plt.plot(ReLU_errors, label="ReLU")

    plt.axhline(Sigmoid_Ee, linestyle="--", label="Порог Sigmoid")

    plt.axhline(ReLU_Ee, linestyle=":", color="orange", label="Порог ReLU")

    plt.xlabel("Epoch")
    plt.ylabel("Суммарная ошибка Es")

    plt.title("Сходимость Sigmoid и ReLU")

    plt.legend()
    plt.grid()

    plt.show()


#5 запусков на разных seed
def run_multiple_experiments(X, e, c0, c1):
    seeds = [1, 2, 3, 4, 5]

    Sigmoid_results = []
    ReLU_results = []

    print("\n")
    print("5 запусков Sigmoid")

    for seed in seeds:

        W1, T1, W2, T2, errors = train(X, e, sigmoid, sigmoid_derivative, alpha=0.4, Ee=0.01, max_epochs=10000, seed=seed)

        accuracy = calculate_accuracy(X, e, W1, T1, W2, T2, sigmoid)

        mae = calculate_mae(X, e, W1, T1, W2, T2, c0, c1, sigmoid)

        reached = errors[-1] <= 0.01

        Sigmoid_results.append({ "seed": seed, "epochs": len(errors), "error": errors[-1], "accuracy": accuracy, "mae": mae, "reached": reached })

        print(f"Seed={seed}: " f"эпох={len(errors)}, " f"ошибка={errors[-1]:.6f}, " f"accuracy={accuracy:.2f}%, " f"MAE={mae:.6f}, " f"критерий={'да' if reached else 'нет'}")

    print("\n")
    print("5 запусков ReLU")

    for seed in seeds:

        W1, T1, W2, T2, errors = train(X, e, ReLU, ReLU_derivative, alpha=0.4, Ee=0.01, max_epochs=10000, seed=seed)

        accuracy = calculate_accuracy(X, e, W1, T1, W2, T2, ReLU)

        mae = calculate_mae(X, e, W1, T1, W2, T2, c0, c1, ReLU)

        reached = errors[-1] <= 0.01

        ReLU_results.append({ "seed": seed, "epochs": len(errors), "error": errors[-1], "accuracy": accuracy, "mae": mae, "reached": reached })

        print(f"Seed={seed}: " f"эпох={len(errors)}, " f"ошибка={errors[-1]:.6f}, " f"accuracy={accuracy:.2f}%, " f"MAE={mae:.6f}, " f"критерий={'да' if reached else 'нет'}")

    return Sigmoid_results, ReLU_results

#разброс числа эпох
def show_epochs_plot(Sigmoid_results, ReLU_results):
    seeds = [1, 2, 3, 4, 5]

    Sigmoid_epochs = [
        result["epochs"]
        for result in Sigmoid_results
    ]

    ReLU_epochs = [
        result["epochs"]
        for result in ReLU_results
    ]

    x = np.arange(len(seeds))

    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(x - width / 2, Sigmoid_epochs, width, label="Sigmoid")

    plt.bar(x + width / 2, ReLU_epochs, width, label="ReLU")

    plt.xlabel("Seed")
    plt.ylabel("Количество эпох")

    plt.title("Разброс числа эпох по 5 запускам")

    plt.xticks(x, seeds)

    plt.legend()
    plt.grid(axis="y")

    plt.show()

#heatmap
def show_surface(W1, T1, W2, T2, X_original, e, c0, c1, activation, title):
    values = np.linspace(-10, 10, 200)
    A, B = np.meshgrid(values, values)

    grid = np.stack([A.ravel(), B.ravel()], axis=1)
    grid_normalized = normalize(grid, c0, c1)
    Y = []

    for x in grid_normalized:
        y = predict(x, W1, T1, W2, T2, activation)
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
def show_mae_plot(Sigmoid_mae, ReLU_mae):
    plt.figure(figsize=(8, 6))
    plt.bar(["Sigmoid", "ReLU"], [Sigmoid_mae, ReLU_mae])
    plt.ylabel("Средняя абсолютная ошибка")
    plt.title("Сравнение точности восстановления исходной шкалы")
    plt.grid(axis="y")
    plt.show()


def interactive_mode(W1, T1, W2, T2, c0, c1, activation):
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
        y = predict(x_normalized, W1, T1, W2, T2, activation)
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

    W1_sigm, T1_sigm, W2_sigm, T2_sigm, errors_sigm = (train(X, e, sigmoid, sigmoid_derivative, alpha=0.4, Ee=0.01, max_epochs=10000, seed=4))
    print_results("КОНФИГУРАЦИЯ A — SIGMOID", X, e, W1_sigm, T1_sigm, W2_sigm, T2_sigm, errors_sigm, c0, c1, sigmoid)

    W1_relu, T1_relu, W2_relu, T2_relu, errors_relu = (train(X, e, ReLU, ReLU_derivative, alpha=0.4, Ee=0.01, max_epochs=10000, seed=4))
    print_results("КОНФИГУРАЦИЯ Б — ReLU", X, e, W1_relu, T1_relu, W2_relu, T2_relu, errors_relu, c0, c1, ReLU)

    print("\nСкрытый слой ReLU:")

    for xi, ei in zip(X, e):
        S1, h, S2, y = forward(
            xi,
            W1_relu, T1_relu,
            W2_relu, T2_relu,
            ReLU
        )

        print(
            f"X={xi}, "
            f"S1={S1}, "
            f"h={h}, "
            f"W2={W2_relu}, "
            f"S2={S2:.4f}, "
            f"y={y:.4f}"
        )

    #5 запусков
    sigm_results, relu_results = (run_multiple_experiments(X, e, c0, c1))
    show_convergence_plot(errors_sigm, errors_relu, 0.01, 0.01) #график схъодимости
    show_epochs_plot(sigm_results, relu_results) #разброс эпох
    show_surface(W1_sigm, T1_sigm, W2_sigm, T2_sigm, X_original, e, c0, c1, sigmoid, "Выход сети Sigmoid")
    show_surface(W1_relu, T1_relu, W2_relu, T2_relu, X_original, e, c0, c1, ReLU, "Выход сети ReLU")

    sigm_mae = calculate_mae(X, e, W1_sigm, T1_sigm, W2_sigm, T2_sigm, c0, c1, sigmoid)
    relu_mae = calculate_mae(X, e, W1_relu, T1_relu, W2_relu, T2_relu, c0, c1, ReLU)
    show_mae_plot(sigm_mae, relu_mae)

    print("\n")
    print("Проверка 4 точек XOR")

    for x_original in X_original:
        x = normalize(x_original, c0, c1)
        y = predict(x, W1_relu, T1_relu, W2_relu, T2_relu, ReLU)
        y_real = denormalize(y, c0, c1)
        if abs(y_real - c0) < abs(y_real - c1):
            class_result = c0
        else:
            class_result = c1
        print(f"A={x_original[0]}, " f"B={x_original[1]} -> " f"y={y:.4f}, " f"y_real={y_real:.4f}, " f"класс={class_result}")

    interactive_mode(W1_relu, T1_relu, W2_relu, T2_relu, c0, c1, ReLU)


if __name__ == "__main__":
    main()