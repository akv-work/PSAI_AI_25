import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, lr, c0, c1, hidden_activation='sigmoid', seed=None):
        self.lr = lr
        self.c0 = c0
        self.c1 = c1
        self.hidden_activation = hidden_activation
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

        self.dW1 = np.zeros((2, 2))
        self.dT1 = np.zeros(2)
        self.dW2 = np.zeros(2)
        self.dT2 = 0.0

    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def _sigmoid_deriv(self, y):
        return y * (1.0 - y)

    def _relu(self, x):
        return np.maximum(0.0, x)

    def _relu_deriv(self, x):
        return (x > 0).astype(float)

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

        if self.hidden_activation == 'relu':
            h1 = self._relu(s1)
            h2 = self._relu(s2)
        else:
            h1 = self._sigmoid(s1)
            h2 = self._sigmoid(s2)

        so = self.W2[0] * h1 + self.W2[1] * h2 - self.T2
        y = self._sigmoid(so)
        return s1, s2, h1, h2, y

    def epoch(self, X1, X2, Y):
        Es = 0.0
        indices = self.rng.permutation(len(X1))

        for i in indices:
            x1n = self.normalize_input(X1[i])
            x2n = self.normalize_input(X2[i])
            target = self.normalize_target(Y[i])

            s1, s2, h1, h2, y = self.forward(x1n, x2n)

            eps = 1e-15
            y_safe = np.clip(y, eps, 1 - eps)
            error = -(target * np.log(y_safe) + (1 - target) * np.log(1 - y_safe))
            Es += error
            delta_out = target - y
            delta_out = np.clip(delta_out, -1.0, 1.0)

            if self.hidden_activation == 'relu':
                dh1 = self._relu_deriv(s1)
                dh2 = self._relu_deriv(s2)
            else:
                dh1 = self._sigmoid_deriv(h1)
                dh2 = self._sigmoid_deriv(h2)

            delta_h1 = delta_out * self.W2[0] * dh1
            delta_h2 = delta_out * self.W2[1] * dh2
            delta_h1 = np.clip(delta_h1, -1.0, 1.0)
            delta_h2 = np.clip(delta_h2, -1.0, 1.0)

            # Выходной слой
            self.dW2[0] = self.momentum * self.dW2[0] + self.lr * delta_out * h1
            self.dW2[1] = self.momentum * self.dW2[1] + self.lr * delta_out * h2
            self.dT2 = self.momentum * self.dT2 + self.lr * delta_out

            self.W2[0] += self.dW2[0]
            self.W2[1] += self.dW2[1]
            self.T2 -= self.dT2

            # Скрытый слой
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
        _, _, _, _, y = self.forward(x1n, x2n)
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

def count_dead_neurons(mlp, X1, X2):
    if mlp.hidden_activation != 'relu':
        return 0
    dead = 0
    for j in range(2):
        active = False
        for x1, x2 in zip(X1, X2):
            x1n = mlp.normalize_input(x1)
            x2n = mlp.normalize_input(x2)
            s1, s2, h1, h2, _ = mlp.forward(x1n, x2n)
            h = h1 if j == 0 else h2
            if h > 1e-6:
                active = True
                break
        if not active:
            dead += 1
    return dead

def plot_convergence(histories, Ee, title_suffix=""):
    plt.figure(figsize=(10, 6))
    colors = {'sigmoid': 'blue', 'relu': 'red'}
    labels = {'sigmoid': 'Конфигурация А (Sigmoid)', 'relu': 'Конфигурация В (ReLU)'}

    for key, history in histories.items():
        if history:
            plt.plot(range(1, len(history) + 1), history,
                     color=colors[key], label=labels[key], linewidth=2)

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

    sig_epochs = [e if e is not None else max_epochs for e in results['sigmoid']]
    relu_epochs = [e if e is not None else max_epochs for e in results['relu']]

    sig_colors = ['blue' if e is not None else 'gray' for e in results['sigmoid']]
    relu_colors = ['red' if e is not None else 'gray' for e in results['relu']]

    bars1 = ax.bar(x - width/2, sig_epochs, width, label='Конфигурация А (Sigmoid)',
                   color=sig_colors, edgecolor='black')
    bars2 = ax.bar(x + width/2, relu_epochs, width, label='Конфигурация В (ReLU)',
                   color=relu_colors, edgecolor='black')

    for i, (b1, b2) in enumerate(zip(bars1, bars2)):
        if results['sigmoid'][i] is None:
            b1.set_hatch('//')
        if results['relu'][i] is None:
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

def plot_decision_surfaces(mlp_sig, mlp_relu, X1, X2, Y, c0, c1):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    resolution = 80
    x_range = np.linspace(-10, 10, resolution)
    y_range = np.linspace(-10, 10, resolution)
    XX, YY = np.meshgrid(x_range, y_range)

    for ax, mlp, title in zip(axes, [mlp_sig, mlp_relu],
                               ['Конфигурация А (Sigmoid)', 'Конфигурация В (ReLU)']):
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

def plot_mae_comparison(mae_sig, mae_relu):
    fig, ax = plt.subplots(figsize=(8, 6))
    configs = ['Конфигурация А\n(Sigmoid)', 'Конфигурация В\n(ReLU)']
    values = [mae_sig, mae_relu]
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

def run_configuration(name, hidden_activation, alpha, X1, X2, Y, c0, c1, Ee, max_epochs):
    print(f"{name}")

    results = {'epochs': [], 'final_error': [], 'accuracy': [], 'mae': [], 'dead': []}
    best_mlp, best_history = None, None

    for seed in range(1, 6):
        print(f"\nЗапуск {seed}/5 (seed={seed})...")
        mlp = MLP(lr=alpha, c0=c0, c1=c1, hidden_activation=hidden_activation, seed=seed)

        history = []
        for ep in range(1, max_epochs + 1):
            Es = mlp.epoch(X1, X2, Y)

            if not np.isfinite(Es):
                print(f"  NaN на эпохе {ep} — обучение прервано")
                mlp.trained = False
                mlp.epochs_trained = ep
                mlp.final_error = float('nan')
                break

            history.append(Es)
            if Es <= Ee:
                mlp.trained = True
                mlp.epochs_trained = ep
                break

        if not mlp.trained and np.isfinite(mlp.final_error):
            mlp.epochs_trained = max_epochs

        mae = calculate_mae(mlp, X1, X2, Y) if np.isfinite(mlp.final_error) else float('nan')
        acc = calculate_accuracy(mlp, X1, X2, Y) if np.isfinite(mlp.final_error) else 0.0
        dead = count_dead_neurons(mlp, X1, X2)

        results['epochs'].append(mlp.epochs_trained if mlp.trained else None)
        results['final_error'].append(mlp.final_error)
        results['accuracy'].append(acc)
        results['mae'].append(mae)
        results['dead'].append(dead)

        print(f"  Эпох: {mlp.epochs_trained} (сошлась: {mlp.trained})")
        print(f"  Es = {mlp.final_error:.6f}, Accuracy = {acc:.2%}, MAE = {mae:.4f}")
        print(f"  Мёртвых нейронов: {dead}")

        if best_mlp is None or (mlp.trained and not best_mlp.trained):
            best_mlp = mlp
            best_history = history.copy()

    return results, best_mlp, best_history

def main():
    data = input_data_manual()
    if data is None:
        return
    X1, X2, Y, c0, c1 = data

    print("\nПараметры обучения")
    alpha_sig = float(input("alpha для Sigmoid (рекомендую 0.1): "))
    alpha_relu = float(input("alpha для ReLU (рекомендую 0.05): "))
    Ee = float(input("Порог остановки Ee (рекомендую 0.01): "))
    max_epochs = int(input("Максимум эпох (рекомендую 100000): "))

    results_sig, best_mlp_sig, best_history_sig = run_configuration(
        "Конфигурация А (Sigmoid на скрытом слое)",
        'sigmoid', alpha_sig, X1, X2, Y, c0, c1, Ee, max_epochs
    )

    results_relu, best_mlp_relu, best_history_relu = run_configuration(
        "Конфигурация В (ReLU на скрытом слое)",
        'relu', alpha_relu, X1, X2, Y, c0, c1, Ee, max_epochs
    )

    def fmt_list(lst):
        return ", ".join(f"{x}" if x is not None else "н/д" for x in lst)

    print(f"{'Параметр':<35} {'Sigmoid':<22} {'ReLU':<22}")
    print(f"{'Эпохи (5 seed)':<35} {fmt_list(results_sig['epochs']):<22} {fmt_list(results_relu['epochs']):<22}")
    print(f"{'Es (среднее)':<35} {np.nanmean(results_sig['final_error']):<22.6f} {np.nanmean(results_relu['final_error']):<22.6f}")
    print(f"{'Accuracy (среднее)':<35} {np.mean(results_sig['accuracy']):<22.2%} {np.mean(results_relu['accuracy']):<22.2%}")
    print(f"{'MAE (среднее)':<35} {np.nanmean(results_sig['mae']):<22.4f} {np.nanmean(results_relu['mae']):<22.4f}")
    print(f"{'Сошлось из 5':<35} {sum(1 for e in results_sig['epochs'] if e is not None):<22} {sum(1 for e in results_relu['epochs'] if e is not None):<22}")
    print(f"{'Мёртвых нейронов (всего)':<35} {sum(results_sig['dead']):<22} {sum(results_relu['dead']):<22}")

    print("\nПостроение графиков...")
    plot_convergence({'sigmoid': best_history_sig, 'relu': best_history_relu}, Ee,
                     title_suffix=f" (c0={c0}, c1={c1})")
    plot_epochs_scatter({'sigmoid': results_sig['epochs'], 'relu': results_relu['epochs']}, max_epochs)
    plot_decision_surfaces(best_mlp_sig, best_mlp_relu, X1, X2, Y, c0, c1)
    plot_mae_comparison(np.nanmean(results_sig['mae']), np.nanmean(results_relu['mae']))

    run_inference_mode(best_mlp_relu, c0, c1, X1, X2, Y)

if __name__ == "__main__":
    main()