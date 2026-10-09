import numpy as np
import matplotlib.pyplot as plt

c0 = 7
c1 = -7
LR_BCE = {'sigmoid': 0.3, 'relu': 0.13}
EE_BCE = 0.01
MAX_EPOCHS = 10000
SEEDS = [1, 2, 3, 4, 5]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(y):
    return y * (1.0 - y)

def relu(x):
    return np.maximum(0.0, x)

def relu_derivative(S):
    return (S > 0).astype(float)

def normalize(value):
    return (value - c0) / (c1 - c0)

def denormalize(value):
    return c0 + value * (c1 - c0)

def get_class(value):
    if abs(value - c0) <= abs(value - c1):
        return c0
    return c1

data = np.array([
    [c0, c0, c0],
    [c0, c1, c1],
    [c1, c0, c1],
    [c1, c1, c0]
], dtype=float)

X = data[:, :2]
Y = data[:, 2]

Xn = normalize(X)    
Yn = normalize(Y)

class MLP:
    def __init__(self, seed=1, hidden_activation='sigmoid'):
        rng = np.random.default_rng(seed)
        if hidden_activation == 'relu':
            self.W1 = rng.normal(0, 0.3, (2, 2))
            self.b1 = rng.uniform(0, 0.5, 2)
            self.W2 = rng.uniform(-0.2, 0.2, 2)
            self.b2 = float(rng.uniform(-0.2, 0.2))
        else:
            self.W1 = rng.uniform(-0.1, 0.1, (2, 2))
            self.b1 = rng.uniform(-0.1, 0.1, 2)
            self.W2 = rng.uniform(-0.1, 0.1, 2)
            self.b2 = float(rng.uniform(-0.1, 0.1))
        self.hidden_activation = hidden_activation
        self.history = []

    def forward(self, x):
        S1 = np.dot(x, self.W1) + self.b1
        if self.hidden_activation == 'relu':
            h = relu(S1)
        else:
            h = sigmoid(S1)
        S2 = np.dot(h, self.W2) + self.b2
        y = sigmoid(S2)
        return S1, h, S2, y

    def _hidden_deriv(self, S1, h):
        if self.hidden_activation == 'relu':
            return relu_derivative(S1)
        return sigmoid_derivative(h)

    def train_bce(self, X, Y, Ee=EE_BCE, lr=None):
        if lr is None:
            lr = LR_BCE[self.hidden_activation]
        epoch = 0
        self.history = []
        while epoch < MAX_EPOCHS:
            for x, target in zip(X, Y):
                S1, h, S2, y = self.forward(x)
                y_clip = np.clip(y, 1e-12, 1 - 1e-12)
                delta2 = (y_clip - target)
                old_W2 = self.W2.copy()
                self.W2 -= lr * h * delta2
                self.b2 -= lr * delta2
                delta1 = old_W2 * delta2 * self._hidden_deriv(S1, h)
                self.W1 -= lr * np.outer(x, delta1)
                self.b1 -= lr * delta1
            epoch += 1
            Es = self.calculate_bce(X, Y)
            self.history.append(Es)
            if Es <= Ee:
                return epoch, Es, True
        Es = self.calculate_bce(X, Y)
        return epoch, Es, False

    def calculate_bce(self, X, Y):
        error = 0.0
        for x, target in zip(X, Y):
            _, _, _, y = self.forward(x)
            y = np.clip(y, 1e-12, 1 - 1e-12)
            error += -(target * np.log(y) + (1 - target) * np.log(1 - y))
        return error

    def predict_norm(self, x):
        xn = normalize(np.asarray(x, dtype=float))
        _, _, _, y = self.forward(xn)
        return float(y)

    def predict(self, x):
        return float(denormalize(self.predict_norm(x)))

def calculate_accuracy(network, X, Y):
    correct = 0
    for x, target in zip(X, Y):
        predicted = network.predict(x)
        if get_class(predicted) == target:
            correct += 1
    return correct / len(Y) * 100

def calculate_mae(network, X, Y):
    errors = []
    for x, target in zip(X, Y):
        predicted = network.predict(x)
        errors.append(abs(predicted - target))
    return float(np.mean(errors))

def show_convergence(histories, Ee, title_suffix=""):
    plt.figure(figsize=(10, 6))
    colors = {'sigmoid': 'gold', 'relu': 'green'}
    labels = {'sigmoid': 'Sigmoid', 'relu': 'ReLU'}
    for key, history in histories.items():
        if history:
            plt.plot(range(1, len(history) + 1), history,
                     color=colors[key],
                     label=labels[key], linewidth=2)
    plt.axhline(y=Ee, color='red', linestyle='--', label=f'Ee = {Ee}')
    plt.xlabel('Эпоха')
    plt.ylabel('Es')
    plt.title(f'Зависимость Es от номера эпохи{title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

def show_epochs_bars(results, max_epochs):
    fig, ax = plt.subplots(figsize=(10, 6))
    seeds = SEEDS
    width = 0.3
    x = np.arange(len(seeds))
    sig_epochs  = [e if e is not None else max_epochs for e in results['sigmoid']]
    relu_epochs = [e if e is not None else max_epochs for e in results['relu']]
    sig_colors  = ['gold' if e is not None else 'grey' for e in results['sigmoid']]
    relu_colors = ['green' if e is not None else 'grey' for e in results['relu']]
    ax.bar(x - width/2, sig_epochs,  width, label='Sigmoid',
           color=sig_colors, edgecolor='black')
    ax.bar(x + width/2, relu_epochs, width, label='ReLU',
           color=relu_colors, edgecolor='black')
    ax.set_xlabel('Seed')
    ax.set_ylabel('Количество эпох')
    ax.set_title('Число эпох для 5 запусков')
    ax.set_xticks(x)
    ax.set_xticklabels([f'seed {s}' for s in seeds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def show_decision_map(mlp_sig, mlp_relu):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    resolution = 80
    x_range = np.linspace(-10, 10, resolution)
    y_range = np.linspace(-10, 10, resolution)
    XX, YY = np.meshgrid(x_range, y_range)

    for ax, mlp, title in zip(axes, [mlp_sig, mlp_relu], ['Sigmoid', 'ReLU']):
        Z = np.zeros_like(XX)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = mlp.predict_norm([XX[i, j], YY[i, j]])

        contour = ax.contourf(XX, YY, Z, levels=20, cmap='YlGn_r', alpha=0.85)
        fig.colorbar(contour, ax=ax, label='нормализованный y')

        for x, target in zip(X, Y):
            color = 'green' if target == c0 else 'gold'
            ax.scatter(x[0], x[1], c=color, s=200,
                       edgecolors='black', linewidths=2, zorder=5)

        ax.set_xlabel('A')
        ax.set_ylabel('B')
        ax.set_title(title)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.grid(True, alpha=0.3)
    plt.suptitle('Разделяющая поверхность на плоскости (A, B)', fontsize=14)
    plt.tight_layout()
    plt.show()

def show_mae(mae_sig, mae_relu):
    fig, ax = plt.subplots(figsize=(8, 6))
    configs = ['Sigmoid', 'ReLU']
    values = [mae_sig, mae_relu]
    colors = ['gold', 'green']
    bars = ax.bar(configs, values, color=colors, edgecolor='black', width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=12)
    ax.set_ylabel('MAE')
    ax.set_title('Сравнение точности восстановления шкалы')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def run_multiple_experiments():
    results = {'sigmoid': [], 'relu': []}
    stats = {
        'sigmoid': {'epochs': [], 'errors': [], 'accs': [], 'maes': [], 'reached': 0},
        'relu':    {'epochs': [], 'errors': [], 'accs': [], 'maes': [], 'reached': 0},
    }

    print("Конфигурация А (Sigmoid)")
    for seed in SEEDS:
        net = MLP(seed=seed, hidden_activation='sigmoid')
        epochs, error, reached = net.train_bce(Xn, Yn, Ee=EE_BCE)
        acc = calculate_accuracy(net, X, Y)
        mae = calculate_mae(net, X, Y)
        results['sigmoid'].append(epochs if reached else None)
        stats['sigmoid']['epochs'].append(epochs)
        stats['sigmoid']['errors'].append(error)
        stats['sigmoid']['accs'].append(acc)
        stats['sigmoid']['maes'].append(mae)
        if reached:
            stats['sigmoid']['reached'] += 1
        print(f"Запуск {seed}")
        print(f"Эпох: {epochs} | Es={error:.6f} | Acc={acc:.1f}% | MAE={mae:.4f}")
    print()

    print("Конфигурация В (ReLU)")
    for seed in SEEDS:
        net = MLP(seed=seed, hidden_activation='relu')
        epochs, error, reached = net.train_bce(Xn, Yn, Ee=EE_BCE)
        acc = calculate_accuracy(net, X, Y)
        mae = calculate_mae(net, X, Y)
        results['relu'].append(epochs if reached else None)
        stats['relu']['epochs'].append(epochs)
        stats['relu']['errors'].append(error)
        stats['relu']['accs'].append(acc)
        stats['relu']['maes'].append(mae)
        if reached:
            stats['relu']['reached'] += 1
        print(f"Запуск {seed}")
        print(f"Эпох: {epochs} | Es={error:.6f} | Acc={acc:.1f}% | MAE={mae:.4f}")
    print()

    print("Результаты:")
    for name, key in [("Sigmoid", 'sigmoid'), ("ReLU", 'relu')]:
        s = stats[key]
        ep = ", ".join(str(e) for e in s['epochs'])
        avg_err = sum(s['errors']) / len(s['errors'])
        avg_acc = sum(s['accs']) / len(s['accs'])
        avg_mae = sum(s['maes']) / len(s['maes'])
        print(f"{name}: эпохи: {ep} | Es={avg_err:.6f} | "
              f"Acc={avg_acc:.1f}% | MAE={avg_mae:.4f} | "
              f"сошлось: {s['reached']}")
    return results, stats

def run_examples(mlp):
    print("\nВведите A и B из диапазона [-10; 10], для выхода введите q")
    while True:
        a_str = input("A = ").strip()
        if a_str.lower() == "q":
            break
        b_str = input("B = ").strip()
        if b_str.lower() == "q":
            break

        try:
            a = float(a_str)
            b = float(b_str)
        except ValueError:
            print("Ошибка ввода. Введите числа.")
            continue

        if not (-10 <= a <= 10 and -10 <= b <= 10):
            print("Значения должны быть в диапазоне [-10; 10]")
            continue

        y_norm = mlp.predict_norm([a, b])
        y_real = denormalize(y_norm)
        cls = get_class(y_real)
        print(f"y = {y_norm:.4f} | y_real = {y_real:.4f} | Класс: {cls}")

def main():
    results, stats = run_multiple_experiments()

    mlp_sig = MLP(seed=1, hidden_activation='sigmoid')
    mlp_sig.train_bce(Xn, Yn, Ee=EE_BCE)
    mlp_sig_mae = calculate_mae(mlp_sig, X, Y)

    mlp_relu = MLP(seed=1, hidden_activation='relu')
    mlp_relu.train_bce(Xn, Yn, Ee=EE_BCE)
    mlp_relu_mae = calculate_mae(mlp_relu, X, Y)

    show_convergence({'sigmoid': mlp_sig.history, 'relu': mlp_relu.history},
                     EE_BCE, title_suffix=' (seed=1)')
    show_epochs_bars(results, MAX_EPOCHS)
    show_decision_map(mlp_sig, mlp_relu)
    show_mae(mlp_sig_mae, mlp_relu_mae)
    run_examples(mlp_relu)

if __name__ == "__main__":
    main()