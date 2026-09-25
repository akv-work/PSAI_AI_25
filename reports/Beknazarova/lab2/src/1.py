import numpy as np
import matplotlib.pyplot as plt


class MLP:
    def __init__(self, lr, c0, c1, loss_type='mse', seed=None):
        self.lr = lr
        self.c0 = c0
        self.c1 = c1
        self.loss_type = loss_type
        self.trained = False
        self.epochs_trained = 0
        self.final_error = 0.0
        self.momentum = 0.9

        self.x_min = min(c0, c1)
        self.x_max = max(c0, c1)

        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()

        if loss_type == 'bce':
            scale = np.sqrt(1.0 / 2.0)
            w = self.rng.normal(0, scale, (2, 2))
            self.W1 = np.array([
                w[0],
                -w[0] + self.rng.normal(0, 0.1, 2)
            ])
            self.T1 = self.rng.normal(0, scale, 2)
            self.W2 = np.array([
                abs(self.rng.normal(0, scale)),
                -abs(self.rng.normal(0, scale))
            ])
            self.T2 = self.rng.normal(0, scale)
        else:
            self.W1 = self.rng.uniform(-0.1, 0.1, (2, 2))
            self.T1 = self.rng.uniform(-0.1, 0.1, 2)
            self.W2 = np.array([
                abs(self.rng.uniform(-0.1, 0.1)),
                -abs(self.rng.uniform(-0.1, 0.1))
            ])
            self.T2 = self.rng.uniform(-0.1, 0.1)

        self.dW1 = np.zeros((2, 2))
        self.dT1 = np.zeros(2)
        self.dW2 = np.zeros(2)
        self.dT2 = 0.0

    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def _sigmoid_deriv(self, y):
        return y * (1.0 - y)

    def normalize_input(self, x):
        if self.x_max == self.x_min:
            return 0.0
        return 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0

    def normalize_target(self, y):
        if self.c1 == self.c0:
            return 0.0
        return (y - self.c0) / (self.c1 - self.c0)

    def denormalize_output(self, y_norm):
        return self.c0 + y_norm * (self.c1 - self.c0)

    def forward(self, x1n, x2n):
        s1 = self.W1[0, 0] * x1n + self.W1[0, 1] * x2n - self.T1[0]
        s2 = self.W1[1, 0] * x1n + self.W1[1, 1] * x2n - self.T1[1]
        h1 = self._sigmoid(s1)
        h2 = self._sigmoid(s2)

        so = self.W2[0] * h1 + self.W2[1] * h2 - self.T2
        y = self._sigmoid(so)
        return h1, h2, y

    def epoch(self, X1, X2, Y):
        Es = 0.0
        indices = self.rng.permutation(len(X1))

        for i in indices:
            x1n = self.normalize_input(X1[i])
            x2n = self.normalize_input(X2[i])
            target = self.normalize_target(Y[i])

            h1, h2, y = self.forward(x1n, x2n)

            if self.loss_type == 'bce':
                eps = 1e-15
                y_safe = np.clip(y, eps, 1 - eps)
                error = -(target * np.log(y_safe) + (1 - target) * np.log(1 - y_safe))
                Es += error
                delta_out = target - y
            else:
                error = 0.5 * (target - y) ** 2
                Es += error
                delta_out = (target - y) * self._sigmoid_deriv(y)

            delta_h1 = delta_out * self.W2[0] * self._sigmoid_deriv(h1)
            delta_h2 = delta_out * self.W2[1] * self._sigmoid_deriv(h2)

            self.dW2[0] = self.momentum * self.dW2[0] + self.lr * delta_out * h1
            self.dW2[1] = self.momentum * self.dW2[1] + self.lr * delta_out * h2
            self.dT2 = self.momentum * self.dT2 + self.lr * delta_out

            self.W2[0] += self.dW2[0]
            self.W2[1] += self.dW2[1]
            self.T2 -= self.dT2

            self.dW1[0, 0] = self.momentum * self.dW1[0, 0] + self.lr * delta_h1 * x1n
            self.dW1[0, 1] = self.momentum * self.dW1[0, 1] + self.lr * delta_h1 * x2n
            self.dT1[0] = self.momentum * self.dT1[0] + self.lr * delta_h1

            self.dW1[1, 0] = self.momentum * self.dW1[1, 0] + self.lr * delta_h2 * x1n
            self.dW1[1, 1] = self.momentum * self.dW1[1, 1] + self.lr * delta_h2 * x2n
            self.dT1[1] = self.momentum * self.dT1[1] + self.lr * delta_h2

            self.W1[0, 0] += self.dW1[0, 0]
            self.W1[0, 1] += self.dW1[0, 1]
            self.T1[0] -= self.dT1[0]

            self.W1[1, 0] += self.dW1[1, 0]
            self.W1[1, 1] += self.dW1[1, 1]
            self.T1[1] -= self.dT1[1]

        self.final_error = Es
        return Es

    def train(self, X1, X2, Y, max_epochs, Ee):
        for epoch_num in range(1, max_epochs + 1):
            Es = self.epoch(X1, X2, Y)
            self.epochs_trained = epoch_num
            if Es <= Ee:
                self.trained = True
                return True
        return False

    def predict_norm(self, x1, x2):
        x1n = self.normalize_input(x1)
        x2n = self.normalize_input(x2)
        _, _, y = self.forward(x1n, x2n)
        return y

    def predict(self, x1, x2):
        return self.denormalize_output(self.predict_norm(x1, x2))

    def print_weights(self):
        print("Скрытый слой:")
        print(f"  Нейрон 1: W1[0][0]={self.W1[0,0]:.4f}, W1[0][1]={self.W1[0,1]:.4f}, T1[0]={self.T1[0]:.4f}")
        print(f"  Нейрон 2: W1[1][0]={self.W1[1,0]:.4f}, W1[1][1]={self.W1[1,1]:.4f}, T1[1]={self.T1[1]:.4f}")
        print("Выходной слой:")
        print(f"  W2[0]={self.W2[0]:.4f}, W2[1]={self.W2[1]:.4f}, T2={self.T2:.4f}")


def calculate_mae(mlp, X1, X2, Y):
    preds = np.array([mlp.predict(x1, x2) for x1, x2 in zip(X1, X2)])
    return float(np.mean(np.abs(preds - Y)))


def calculate_accuracy(mlp, X1, X2, Y, tol=0.2):
    correct = 0
    for x1, x2, y_true in zip(X1, X2, Y):
        y_norm = mlp.predict_norm(x1, x2)
        t_norm = mlp.normalize_target(y_true)
        if abs(y_norm - t_norm) < tol:
            correct += 1
    return correct / len(X1)


def plot_convergence(histories, Ee, title_suffix=""):
    plt.figure(figsize=(10, 6))
    colors = {'mse': 'blue', 'bce': 'red'}
    labels = {'mse': 'Конфигурация А (MSE)', 'bce': 'Конфигурация Б (BCE)'}

    for loss_type, history in histories.items():
        if history:
            plt.plot(range(1, len(history) + 1), history,
                     color=colors[loss_type], label=labels[loss_type], linewidth=2)

    plt.axhline(y=Ee, color='green', linestyle='--', label=f'Порог Ee = {Ee}')
    plt.xlabel('Эпоха')
    plt.ylabel('Суммарная ошибка Es')
    plt.title(f'Сходимость обучения{title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()


def plot_epochs_scatter(results, max_epochs):
    fig, ax = plt.subplots(figsize=(10, 6))
    seeds = list(range(1, 6))
    width = 0.35
    x = np.arange(len(seeds))

    mse_epochs = [e if e is not None else max_epochs for e in results['mse']]
    bce_epochs = [e if e is not None else max_epochs for e in results['bce']]

    mse_colors = ['blue' if e is not None else 'gray' for e in results['mse']]
    bce_colors = ['red' if e is not None else 'gray' for e in results['bce']]

    bars1 = ax.bar(x - width/2, mse_epochs, width, label='Конфигурация А (MSE)',
                   color=mse_colors, edgecolor='black')
    bars2 = ax.bar(x + width/2, bce_epochs, width, label='Конфигурация Б (BCE)',
                   color=bce_colors, edgecolor='black')

    for i, (b1, b2) in enumerate(zip(bars1, bars2)):
        if results['mse'][i] is None:
            b1.set_hatch('//')
        if results['bce'][i] is None:
            b2.set_hatch('//')

    ax.set_xlabel('Номер запуска (seed)')
    ax.set_ylabel('Количество эпох')
    ax.set_title('Разброс числа эпох по 5 запускам')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Seed {s}' for s in seeds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_decision_surfaces(mlp_mse, mlp_bce, X1, X2, Y, c0, c1):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    resolution = 80
    x_range = np.linspace(-10, 10, resolution)
    y_range = np.linspace(-10, 10, resolution)
    XX, YY = np.meshgrid(x_range, y_range)

    for ax, mlp, title in zip(axes, [mlp_mse, mlp_bce],
                               ['Конфигурация А (MSE)', 'Конфигурация Б (BCE)']):
        Z = np.zeros_like(XX)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = mlp.predict_norm(XX[i, j], YY[i, j])

        contour = ax.contourf(XX, YY, Z, levels=20, cmap='RdBu_r', alpha=0.85)
        fig.colorbar(contour, ax=ax, label='Нормализованный выход ŷ')

        for x1, x2, y_true in zip(X1, X2, Y):
            y_norm = mlp.normalize_target(y_true)
            color = 'blue' if y_norm < 0.5 else 'red'
            ax.scatter(x1, x2, c=color, s=200, edgecolors='black',
                       linewidths=2, zorder=5)

        ax.set_xlabel('A')
        ax.set_ylabel('B')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Разделяющая поверхность на плоскости (A, B)', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_mae_comparison(mae_mse, mae_bce):
    fig, ax = plt.subplots(figsize=(8, 6))
    configs = ['Конфигурация А\n(MSE)', 'Конфигурация Б\n(BCE)']
    values = [mae_mse, mae_bce]
    colors = ['blue', 'red']

    bars = ax.bar(configs, values, color=colors, edgecolor='black', width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=12)

    ax.set_ylabel('Средняя абсолютная ошибка (MAE)')
    ax.set_title('Сравнение точности восстановления шкалы [c0; c1]')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def input_data_manual():
    print("\nРучной ввод данных")
    n = int(input("Введите количество примеров: "))

    print("Введите x1 (через пробел): ")
    x1 = list(map(float, input().split()))
    print("Введите x2 (через пробел): ")
    x2 = list(map(float, input().split()))
    print("Введите эталонные значения (через пробел): ")
    y = list(map(float, input().split()))

    if not (len(x1) == len(x2) == len(y) == n):
        print("Ошибка: количество значений не совпадает с n.")
        return None

    all_vals = sorted(set(x1 + x2 + y))
    if len(all_vals) < 2:
        print("Ошибка: менее двух различных значений.")
        return None

    c0, c1 = all_vals[0], all_vals[1]
    print(f"\nАвтоматически определены классы: c0 = {c0}, c1 = {c1}")

    return np.array(x1), np.array(x2), np.array(y), c0, c1


def run_inference_mode(mlp, c0, c1, X1, X2, Y):
    print("Режим функционирования")

    print("Обучающая выборка")
    for i, (x1, x2, y_true) in enumerate(zip(X1, X2, Y), 1):
        y_norm = mlp.predict_norm(x1, x2)
        y_real = mlp.predict(x1, x2)
        cls = "c0" if abs(y_norm - 0.0) <= abs(y_norm - 1.0) else "c1"
        print(f"Пример {i}: (A={x1}, B={x2})")
        print(f"  ŷ = {y_norm:.4f}, y_real = {y_real:.4f}  [эталон: {y_true}]  класс: {cls}")

    print("\nДополнительные примеры")
    extra = [(-5.0, 5.0), (3.0, -3.0), (1.0, -1.0)]
    for x1, x2 in extra:
        y_norm = mlp.predict_norm(x1, x2)
        y_real = mlp.predict(x1, x2)
        cls = "c0" if abs(y_norm - 0.0) <= abs(y_norm - 1.0) else "c1"
        print(f"Пример: (A={x1}, B={x2})")
        print(f"  ŷ = {y_norm:.4f}, y_real = {y_real:.4f}  класс: {cls}")

    print("\nИнтерактивный ввод (q — выход)")
    while True:
        try:
            raw = input("\nA B: ").strip()
            if raw.lower() == 'q':
                break
            parts = raw.split()
            if len(parts) != 2:
                print("Ошибка: нужно два числа.")
                continue
            a, b = float(parts[0]), float(parts[1])
            y_norm = mlp.predict_norm(a, b)
            y_real = mlp.predict(a, b)
            cls = "c0" if abs(y_norm - 0.0) <= abs(y_norm - 1.0) else "c1"
            print(f"  ŷ = {y_norm:.4f}, y_real = {y_real:.4f}, класс: {cls}")
        except ValueError:
            print("Ошибка ввода.")
        except EOFError:
            break


def main():
    data = input_data_manual()
    if data is None:
        return
    X1, X2, Y, c0, c1 = data

    print("\nПараметры обучения")
    alpha_mse = float(input("alpha для MSE (рекомендую 0.5): "))
    alpha_bce = float(input("alpha для BCE (рекомендую 0.1): "))
    Ee = float(input("Порог остановки Ee (рекомендую 0.01): "))
    max_epochs = int(input("Максимум эпох (рекомендую 100000): "))

    print("Конфигурация А")

    results_mse = {'epochs': [], 'final_error': [], 'accuracy': [], 'mae': []}
    best_mlp_mse, best_history_mse = None, None

    for seed in range(1, 6):
        print(f"\nЗапуск {seed}/5 (seed={seed})...")
        mlp = MLP(lr=alpha_mse, c0=c0, c1=c1, loss_type='mse', seed=seed)

        history = []
        for ep in range(1, max_epochs + 1):
            Es = mlp.epoch(X1, X2, Y)
            history.append(Es)
            if Es <= Ee:
                mlp.trained = True
                mlp.epochs_trained = ep
                break
        if not mlp.trained:
            mlp.epochs_trained = max_epochs

        mae = calculate_mae(mlp, X1, X2, Y)
        acc = calculate_accuracy(mlp, X1, X2, Y)

        results_mse['epochs'].append(mlp.epochs_trained if mlp.trained else None)
        results_mse['final_error'].append(mlp.final_error)
        results_mse['accuracy'].append(acc)
        results_mse['mae'].append(mae)

        print(f"  Эпох: {mlp.epochs_trained} (сошлась: {mlp.trained})")
        print(f"  Es = {mlp.final_error:.6f}, Accuracy = {acc:.2%}, MAE = {mae:.4f}")

        if best_mlp_mse is None or (mlp.trained and not best_mlp_mse.trained):
            best_mlp_mse = mlp
            best_history_mse = history.copy()

    print("Конфигурация Б")

    results_bce = {'epochs': [], 'final_error': [], 'accuracy': [], 'mae': []}
    best_mlp_bce, best_history_bce = None, None

    for seed in range(1, 6):
        print(f"\nЗапуск {seed}/5 (seed={seed})...")
        mlp = MLP(lr=alpha_bce, c0=c0, c1=c1, loss_type='bce', seed=seed)

        history = []
        for ep in range(1, max_epochs + 1):
            Es = mlp.epoch(X1, X2, Y)
            history.append(Es)
            if Es <= Ee:
                mlp.trained = True
                mlp.epochs_trained = ep
                break
        if not mlp.trained:
            mlp.epochs_trained = max_epochs

        mae = calculate_mae(mlp, X1, X2, Y)
        acc = calculate_accuracy(mlp, X1, X2, Y)

        results_bce['epochs'].append(mlp.epochs_trained if mlp.trained else None)
        results_bce['final_error'].append(mlp.final_error)
        results_bce['accuracy'].append(acc)
        results_bce['mae'].append(mae)

        print(f"  Эпох: {mlp.epochs_trained} (сошлась: {mlp.trained})")
        print(f"  Es = {mlp.final_error:.6f}, Accuracy = {acc:.2%}, MAE = {mae:.4f}")

        if best_mlp_bce is None or (mlp.trained and not best_mlp_bce.trained):
            best_mlp_bce = mlp
            best_history_bce = history.copy()

    def fmt_list(lst):
        return ", ".join(f"{x}" if x is not None else "н/д" for x in lst)

    print(f"{'Параметр':<35} {'MSE':<20} {'BCE':<20}")
    print(f"{'Эпохи (5 seed)':<35} {fmt_list(results_mse['epochs']):<20} {fmt_list(results_bce['epochs']):<20}")
    print(f"{'Es (среднее)':<35} {np.mean(results_mse['final_error']):<20.6f} {np.mean(results_bce['final_error']):<20.6f}")
    print(f"{'Accuracy (среднее)':<35} {np.mean(results_mse['accuracy']):<20.2%} {np.mean(results_bce['accuracy']):<20.2%}")
    print(f"{'MAE (среднее)':<35} {np.mean(results_mse['mae']):<20.4f} {np.mean(results_bce['mae']):<20.4f}")
    print(f"{'Сошлось из 5':<35} {sum(1 for e in results_mse['epochs'] if e is not None):<20} {sum(1 for e in results_bce['epochs'] if e is not None):<20}")

    print("\nПостроение графиков...")
    plot_convergence({'mse': best_history_mse, 'bce': best_history_bce}, Ee,
                     title_suffix=f" (c0={c0}, c1={c1})")
    plot_epochs_scatter({'mse': results_mse['epochs'], 'bce': results_bce['epochs']}, max_epochs)
    plot_decision_surfaces(best_mlp_mse, best_mlp_bce, X1, X2, Y, c0, c1)
    plot_mae_comparison(np.mean(results_mse['mae']), np.mean(results_bce['mae']))

    run_inference_mode(best_mlp_bce, c0, c1, X1, X2, Y)


if __name__ == "__main__":
    main()