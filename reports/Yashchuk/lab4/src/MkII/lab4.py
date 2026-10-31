
import numpy as np
import matplotlib.pyplot as plt
from numba import njit


# Вариант 14
A, B, C, D = 2.4, 0.31, 1.2, 0.72
P, N = 7, 6
LR, EE, MAX_EPOCHS = 0.1, 1e-4, 20000
SEEDS = range(5)


def make_series():
    k = np.arange(200)
    return k, A * np.cos(B * k) + C * np.sin(D * k)


def normalize(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)


def denormalize(x, xmin, xmax):
    return x * (xmax - xmin) + xmin


def windows(data, p, start, end):
    X, y = [], []
    for i in range(start, end):
        X.append(data[i-p:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


@njit
def train_fast(X, y, n_hidden, lr, ee, max_epochs, seed):
    np.random.seed(seed)

    n_samples, n_inputs = X.shape
    W1 = np.random.uniform(-0.5, 0.5, (n_inputs, n_hidden))
    b1 = np.zeros(n_hidden)
    W2 = np.random.uniform(-0.5, 0.5, n_hidden)
    b2 = 0.0

    history = np.empty(max_epochs)

    for epoch in range(max_epochs):
        order = np.random.permutation(n_samples)

        for idx in order:
            hidden = np.empty(n_hidden)

            for j in range(n_hidden):
                z = b1[j]
                for i in range(n_inputs):
                    z += X[idx, i] * W1[i, j]
                hidden[j] = 1.0 / (1.0 + np.exp(-max(-500.0, min(500.0, z))))

            output = b2
            for j in range(n_hidden):
                output += hidden[j] * W2[j]

            delta_out = y[idx] - output
            delta_hidden = np.empty(n_hidden)

            for j in range(n_hidden):
                delta_hidden[j] = (
                    delta_out * W2[j] * hidden[j] * (1.0 - hidden[j])
                )

            for j in range(n_hidden):
                W2[j] += lr * delta_out * hidden[j]
            b2 += lr * delta_out

            for i in range(n_inputs):
                for j in range(n_hidden):
                    W1[i, j] += lr * delta_hidden[j] * X[idx, i]

            for j in range(n_hidden):
                b1[j] += lr * delta_hidden[j]

        error = 0.0
        for idx in range(n_samples):
            output = b2
            for j in range(n_hidden):
                z = b1[j]
                for i in range(n_inputs):
                    z += X[idx, i] * W1[i, j]
                h = 1.0 / (1.0 + np.exp(-max(-500.0, min(500.0, z))))
                output += h * W2[j]
            error += (y[idx] - output) ** 2

        error /= n_samples
        history[epoch] = error

        if error <= ee:
            return W1, b1, W2, b2, history[:epoch+1]

    return W1, b1, W2, b2, history


def predict(X, model):
    W1, b1, W2, b2 = model
    hidden = sigmoid(X @ W1 + b1)
    return hidden @ W2 + b2


def metrics(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    smape = np.mean(
        2 * np.abs(actual - predicted) /
        (np.abs(actual) + np.abs(predicted) + 1e-8)
    ) * 100
    return mae, rmse, smape


def print_metrics(name, actual, predicted):
    mae, rmse, smape = metrics(actual, predicted)
    print(f"{name}: MAE={mae:.4f}, RMSE={rmse:.4f}, sMAPE={smape:.2f}%")


def train_model(X, y, n_hidden, seed):
    return train_fast(X, y, n_hidden, LR, EE, MAX_EPOCHS, seed)


def show_prediction(window, model, xmin, xmax, actual=None):
    x_norm = normalize(np.array(window), xmin, xmax).reshape(1, -1)
    pred_norm = predict(x_norm, model)[0]
    pred = denormalize(pred_norm, xmin, xmax)

    print("Входные значения:", np.round(x_norm[0], 4))
    print(f"Прогноз (норм.): {pred_norm:.4f}")
    print(f"Прогноз (исходный масштаб): {pred:.4f}")

    if actual is not None:
        print(f"Истинное значение: {actual:.4f}")
        print(f"Абсолютная ошибка: {abs(actual - pred):.4f}")
    print()


def recursive_forecast(model, last_window, xmin, xmax, steps=30):
    current = last_window.copy()
    result = []

    for _ in range(steps):
        pred = predict(current.reshape(1, -1), model)[0]
        result.append(pred)
        current = np.append(current[1:], pred)

    return denormalize(np.array(result), xmin, xmax)


def main():
    k, series = make_series()

    split = int(len(series) * 0.7)
    train = series[:split]

    xmin, xmax = train.min(), train.max()
    data = normalize(series, xmin, xmax)

    # Базовая сеть 7-6-1
    X_train, y_train = windows(data, P, P, split)
    X_test, y_test = windows(data, P, split, len(data))

    print("Обучение базовой сети...")
    baseline = train_model(X_train, y_train, N, 0)
    model = baseline[:4]
    baseline_history = baseline[4]

    pred_train = denormalize(predict(X_train, model), xmin, xmax)
    pred_test = denormalize(predict(X_test, model), xmin, xmax)

    print(f"Эпох базовой сети: {len(baseline_history)}")
    print_metrics("Обучающая выборка", series[P:split], pred_train)
    print_metrics("Тестовая выборка", series[split:], pred_test)

    # Исследование размера окна p
    print("\nИсследование размера окна p")
    p_results = {}
    p_histories = {}

    for p in [2, 4, 7, 14]:
        epochs, errors, not_converged = [], [], 0

        for seed in SEEDS:
            if p == 7 and seed == 0:
                result = baseline
            else:
                X, y = windows(data, p, p, split)
                result = train_model(X, y, N, seed)

            m = result[:4]
            history = result[4]

            if p not in p_histories and seed == 0:
                p_histories[p] = history

            Xt, yt = windows(data, p, split, len(data))
            pred = denormalize(predict(Xt, m), xmin, xmax)
            actual = denormalize(yt, xmin, xmax)

            epochs.append(len(history))
            errors.append(metrics(actual, pred)[0])

            if history[-1] > EE:
                not_converged += 1

        p_results[p] = (np.mean(epochs), np.std(epochs),
                        np.mean(errors), np.std(errors), not_converged)

        print(
            f"p={p:2}: эпохи {np.mean(epochs):.0f} ± {np.std(epochs):.0f}; "
            f"MAE {np.mean(errors):.4f} ± {np.std(errors):.4f}; "
            f"без сходимости: {not_converged}/5"
        )

    # Исследование числа скрытых нейронов
    print("\nИсследование числа скрытых нейронов")
    n_results = {}

    for n in [1, 2, 6, 12]:
        epochs, errors, not_converged = [], [], 0

        for seed in SEEDS:
            if n == 6 and seed == 0:
                result = baseline
            else:
                result = train_model(X_train, y_train, n, seed)

            m = result[:4]
            history = result[4]
            pred = denormalize(predict(X_test, m), xmin, xmax)

            epochs.append(len(history))
            errors.append(metrics(series[split:], pred)[0])

            if history[-1] > EE:
                not_converged += 1

        n_results[n] = (np.mean(epochs), np.std(epochs),
                        np.mean(errors), np.std(errors), not_converged)

        print(
            f"N={n:2}: эпохи {np.mean(epochs):.0f} ± {np.std(epochs):.0f}; "
            f"MAE {np.mean(errors):.4f} ± {np.std(errors):.4f}; "
            f"без сходимости: {not_converged}/5"
        )

    # График 1. Исходный ряд и одношаговый прогноз
    plt.figure(figsize=(11, 5))
    plt.plot(k, series, label="Исходный ряд")
    plt.plot(np.arange(P, split), pred_train, label="Прогноз на обучении")
    plt.plot(np.arange(split, len(series)), pred_test, label="Прогноз на тесте")
    plt.axvline(split, color="black", linestyle="--", label="Граница выборок")
    plt.xlabel("k")
    plt.ylabel("x(k)")
    plt.title("Одношаговое прогнозирование временного ряда")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("plot1_forecast.png")

    # График 2. Сходимость
    plt.figure(figsize=(9, 5))
    plt.plot(baseline_history, label="p=7, N=6")
    for p in [4, 14]:
        plt.plot(p_histories[p], label=f"p={p}, N=6")
    plt.yscale("log")
    plt.xlabel("Эпоха")
    plt.ylabel("MSE")
    plt.title("Сходимость нейронной сети")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("plot2_convergence.png")

    # График 3. Истинные и предсказанные значения
    plt.figure(figsize=(6, 6))
    plt.scatter(series[split:], pred_test)
    low = min(series[split:].min(), pred_test.min())
    high = max(series[split:].max(), pred_test.max())
    plt.plot([low, high], [low, high], "r--", label="y=x")
    plt.xlabel("Истинные значения")
    plt.ylabel("Предсказанные значения")
    plt.title("Сравнение на тестовой выборке")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("plot3_scatter.png")

    # Рекурсивный прогноз на 30 шагов
    future = recursive_forecast(
        model, data[split-P:split], xmin, xmax, 30
    )
    true_future = series[split:split+30]
    errors = np.abs(true_future - future)
    threshold = 0.05 * (series.max() - series.min())

    exceeded = np.where(errors > threshold)[0]
    if len(exceeded):
        print(f"\nОшибка превысила 5% диапазона на шаге {exceeded[0]+1}")
    else:
        print("\nОшибка не превысила 5% диапазона за 30 шагов")

    # График 4. Рекурсивный прогноз и ошибка
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    steps = np.arange(1, 31)

    axes[0].plot(steps, true_future, label="Истинные значения")
    axes[0].plot(steps, future, label="Рекурсивный прогноз")
    axes[0].set_title("Рекурсивный прогноз на 30 шагов")
    axes[0].set_xlabel("Шаг")
    axes[0].set_ylabel("x(k)")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(steps, errors, label="Абсолютная ошибка")
    axes[1].axhline(threshold, color="red", linestyle="--",
                    label="5% диапазона ряда")
    axes[1].set_xlabel("Шаг")
    axes[1].set_ylabel("Ошибка")
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.savefig("plot4_recursive.png")

    # Примеры прогнозирования
    print("\nПримеры на обучающей выборке")
    for i in [10, 40, 80]:
        show_prediction(series[i-P:i], model, xmin, xmax, series[i])

    print("Примеры на тестовой выборке")
    for i in [145, 160, 180]:
        show_prediction(series[i-P:i], model, xmin, xmax, series[i])

    print("Пользовательские окна")
    custom = series[50:57].copy()
    show_prediction(custom, model, xmin, xmax)

    custom[3] += 0.5 * (xmax - xmin)
    show_prediction(custom, model, xmin, xmax)

    print("Графики сохранены в текущей папке.")
    plt.show()

    # Ручной режим
    print("\nРучной режим. Введите индекс k или 7 значений через пробел.")
    print("Для выхода введите q.")

    while True:
        value = input("> ").strip()

        if value.lower() == "q":
            break

        try:
            parts = value.split()

            if len(parts) == 1:
                index = int(parts[0])
                if index < P or index >= len(series):
                    print(f"Индекс должен быть от {P} до 199")
                    continue

                show_prediction(
                    series[index-P:index], model, xmin, xmax, series[index]
                )

            elif len(parts) == P:
                custom = np.array([float(v) for v in parts])
                show_prediction(custom, model, xmin, xmax)

            else:
                print("Введите индекс или ровно 7 чисел.")

        except ValueError:
            print("Некорректный ввод.")


if __name__ == "__main__":
    main()
