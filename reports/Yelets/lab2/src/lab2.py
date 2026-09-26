import numpy as np
import matplotlib.pyplot as plt

c0 = 7
c1 = -7
LR_MSE = 0.5     
LR_BCE = 0.3    
EE_MSE = 0.01
EE_BCE = 0.01
MAX_EPOCHS = 10000
SEEDS = [1, 2, 3, 4, 5]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(y):
    return y * (1.0 - y)

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
    def __init__(self, seed=1):
        rng = np.random.default_rng(seed)
        self.W1 = rng.uniform(-0.1, 0.1, (2, 2))
        self.b1 = rng.uniform(-0.1, 0.1, 2)
        self.W2 = rng.uniform(-0.1, 0.1, 2)
        self.b2 = float(rng.uniform(-0.1, 0.1))
        self.history = []

    def forward(self, x):
        S1 = np.dot(x, self.W1) + self.b1
        h = sigmoid(S1)
        S2 = np.dot(h, self.W2) + self.b2
        y = sigmoid(S2)
        return S1, h, S2, y

    def train_mse(self, X, Y, Ee=EE_MSE, lr=LR_MSE):
        epoch = 0
        self.history = []
        while epoch < MAX_EPOCHS:
            for x, target in zip(X, Y):
                S1, h, S2, y = self.forward(x)
                delta2 = (y - target) * sigmoid_derivative(y)
                old_W2 = self.W2.copy()
                self.W2 -= lr * h * delta2
                self.b2 -= lr * delta2
                delta1 = old_W2 * delta2 * sigmoid_derivative(h)
                self.W1 -= lr * np.outer(x, delta1)
                self.b1 -= lr * delta1
            epoch += 1
            Es = self.calculate_mse(X, Y)
            self.history.append(Es)
            if Es <= Ee:
                return epoch, Es, True
        Es = self.calculate_mse(X, Y)
        return epoch, Es, False

    def train_bce(self, X, Y, Ee=EE_BCE, lr=LR_BCE):
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
                delta1 = old_W2 * delta2 * sigmoid_derivative(h)
                self.W1 -= lr * np.outer(x, delta1)
                self.b1 -= lr * delta1
            epoch += 1
            Es = self.calculate_bce(X, Y)
            self.history.append(Es)
            if Es <= Ee:
                return epoch, Es, True
        Es = self.calculate_bce(X, Y)
        return epoch, Es, False

    def calculate_mse(self, X, Y):
        error = 0.0
        for x, target in zip(X, Y):
            _, _, _, y = self.forward(x)
            error += 0.5 * (y - target) ** 2
        return error

    def calculate_bce(self, X, Y):
        error = 0.0
        for x, target in zip(X, Y):
            _, _, _, y = self.forward(x)
            y = np.clip(y, 1e-12, 1 - 1e-12)
            error += -(target * np.log(y) + (1 - target) * np.log(1 - y))
        return error

    def predict_norm(self, x):
        _, _, _, y = self.forward(normalize(np.asarray(x, dtype=float)))
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
    colors = {'mse': 'green', 'bce': 'gold'}
    labels = {'mse': 'MSE', 'bce': 'BCE'}
    for loss_type, history in histories.items():
        if history:
            plt.plot(range(1, len(history) + 1), history,
                     color=colors[loss_type],
                     label=labels[loss_type], linewidth=2)
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
    mse_epochs = [e if e is not None else max_epochs for e in results['mse']]
    bce_epochs = [e if e is not None else max_epochs for e in results['bce']]
    mse_colors = ['green' if e is not None else 'grey' for e in results['mse']]
    bce_colors = ['gold' if e is not None else 'grey' for e in results['bce']]
    ax.bar(x - width/2, mse_epochs, width, label='MSE',
           color=mse_colors, edgecolor='black')
    ax.bar(x + width/2, bce_epochs, width, label='BCE',
           color=bce_colors, edgecolor='black')
    ax.set_xlabel('Seed')
    ax.set_ylabel('Количество эпох')
    ax.set_title('Число эпох для 5 запусков')
    ax.set_xticks(x)
    ax.set_xticklabels([f'seed {s}' for s in seeds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def show_decision_map(mlp_mse, mlp_bce):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    resolution = 80
    x_range = np.linspace(-10, 10, resolution)
    y_range = np.linspace(-10, 10, resolution)
    XX, YY = np.meshgrid(x_range, y_range)

    for ax, mlp, title in zip(axes, [mlp_mse, mlp_bce], ['MSE', 'BCE']):
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

def show_mae(mae_mse, mae_bce):
    fig, ax = plt.subplots(figsize=(8, 6))
    configs = ['MSE', 'BCE']
    values = [mae_mse, mae_bce]
    colors = ['green', 'gold']
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
    results = {'mse': [], 'bce': []}
    stats = {'mse': {'epochs': [], 'errors': [], 'accs': [], 'maes': [], 'reached': 0},
             'bce': {'epochs': [], 'errors': [], 'accs': [], 'maes': [], 'reached': 0}}

    print("Конфигурация А (MSE)")
    for seed in SEEDS:
        net = MLP(seed=seed)
        epochs, error, reached = net.train_mse(Xn, Yn, Ee=EE_MSE)
        acc = calculate_accuracy(net, X, Y)
        mae = calculate_mae(net, X, Y)
        results['mse'].append(epochs if reached else None)
        stats['mse']['epochs'].append(epochs)
        stats['mse']['errors'].append(error)
        stats['mse']['accs'].append(acc)
        stats['mse']['maes'].append(mae)
        if reached:
            stats['mse']['reached'] += 1
        print(f"Запуск {seed}")
        print(f"Эпох: {epochs} | Es={error:.6f} | Acc={acc:.1f}% | MAE={mae:.4f}")
    print()

    print("Конфигурация Б (BCE)")
    for seed in SEEDS:
        net = MLP(seed=seed)
        epochs, error, reached = net.train_bce(Xn, Yn, Ee=EE_BCE)
        acc = calculate_accuracy(net, X, Y)
        mae = calculate_mae(net, X, Y)
        results['bce'].append(epochs if reached else None)
        stats['bce']['epochs'].append(epochs)
        stats['bce']['errors'].append(error)
        stats['bce']['accs'].append(acc)
        stats['bce']['maes'].append(mae)
        if reached:
            stats['bce']['reached'] += 1
        print(f"Запуск {seed}")
        print(f"Эпох: {epochs} | Es={error:.6f} | Acc={acc:.1f}% | MAE={mae:.4f}")
    print()

    print("Результаты:")
    for name, key in [("MSE", 'mse'), ("BCE", 'bce')]:
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
    mlp_mse = MLP(seed=1)
    mlp_mse.train_mse(Xn, Yn, Ee=EE_MSE)
    mlp_mse_mae = calculate_mae(mlp_mse, X, Y)
    mlp_bce = MLP(seed=1)
    mlp_bce.train_bce(Xn, Yn, Ee=EE_BCE)
    mlp_bce_mae = calculate_mae(mlp_bce, X, Y)

    show_convergence({'mse': mlp_mse.history, 'bce': mlp_bce.history}, EE_MSE, title_suffix=' (seed=1)')
    show_epochs_bars(results, MAX_EPOCHS)
    show_decision_map(mlp_mse, mlp_bce)
    show_mae(mlp_mse_mae, mlp_bce_mae)
    run_examples(mlp_bce)

if __name__ == "__main__":
    main()