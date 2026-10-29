import numpy as np
import matplotlib.pyplot as plt

c_zero, c_one = -4, -9

data = np.array([
    [c_zero, c_zero, c_zero],
    [c_zero, c_one,  c_one],
    [c_one,  c_zero, c_one],
    [c_one,  c_one,  c_zero]
], dtype=float)

X_raw = data[:, :2]
Y_raw = data[:, 2]

lr_rate = {'sigmoid': 0.3, 'relu': 0.13}
err_limit = 0.01
max_epochs = 10000
seeds = [1, 2, 3, 4, 5]
show_seed = 1

def phi(z):
    """Сигмоида."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def dphi(a):
    """Производная сигмоиды по её выходу."""
    return a * (1.0 - a)

def relu(z):
    return np.maximum(0.0, z)

def drelu(z):
    """Производная ReLU по входу нейрона."""
    return (z > 0).astype(float)

def scale(v):
    """Из [c0; c1] в [0; 1]."""
    return (v - c_zero) / (c_one - c_zero)

def unscale(v):
    """Из [0; 1] обратно в [c0; c1]."""
    return c_zero + v * (c_one - c_zero)

def nearest_class(v):
    return c_zero if abs(v - c_zero) <= abs(v - c_one) else c_one

X_in = scale(X_raw)
T_target = scale(Y_raw)

class Net:
    def __init__(self, seed, hidden):
        rng = np.random.default_rng(seed)
        if hidden == 'relu':
            self.Vh = rng.normal(0, 0.3, (2, 2))
            self.Bh = rng.uniform(0, 0.5, 2)
            self.Vo = rng.uniform(-0.2, 0.2, 2)
            self.Bo = float(rng.uniform(-0.2, 0.2))
        else:
            self.Vh = rng.uniform(-0.1, 0.1, (2, 2))
            self.Bh = rng.uniform(-0.1, 0.1, 2)
            self.Vo = rng.uniform(-0.1, 0.1, 2)
            self.Bo = float(rng.uniform(-0.1, 0.1))
        self.hidden = hidden
        self.hist = []

    def feedforward(self, x):
        zh = x @ self.Vh + self.Bh
        ah = relu(zh) if self.hidden == 'relu' else phi(zh)
        ao = phi(ah @ self.Vo + self.Bo)
        return zh, ah, ao

    def hidden_grad(self, zh, ah):
        return drelu(zh) if self.hidden == 'relu' else dphi(ah)

    def total_error(self):
        """Суммарная BCE по всей выборке."""
        err = 0.0
        for x, y in zip(X_in, T_target):
            ao = self.feedforward(x)[2]
            p = np.clip(ao, 1e-12, 1 - 1e-12)
            err += -(y * np.log(p) + (1 - y) * np.log(1 - p))
        return err

    def train(self):
        lr = lr_rate[self.hidden]
        self.hist = []
        for epoch in range(1, max_epochs + 1):
            for x, y in zip(X_in, T_target):
                zh, ah, ao = self.feedforward(x)

                d_out = ao - y
                d_hid = self.Vo * d_out * self.hidden_grad(zh, ah)

                self.Vo -= lr * ah * d_out
                self.Bo -= lr * d_out
                self.Vh -= lr * np.outer(x, d_hid)
                self.Bh -= lr * d_hid

            err = self.total_error()
            self.hist.append(err)
            if err <= err_limit:
                return epoch, err, True
        return max_epochs, self.hist[-1], False

    def predict_norm(self, a, b):
        x = scale(np.array([a, b], dtype=float))
        return float(self.feedforward(x)[2])

    def predict_real(self, a, b):
        return float(unscale(self.predict_norm(a, b)))

    def dead_neurons(self):
        if self.hidden != 'relu':
            return 0
        zh = X_in @ self.Vh + self.Bh
        return int(np.sum(np.all(zh <= 0, axis=0)))

    def active_counts(self):
        zh = X_in @ self.Vh + self.Bh
        return [int(c) for c in np.sum(zh > 0, axis=0)]

def compute_metrics(net):
    preds = np.array([net.predict_real(a, b) for a, b in X_raw])
    classes = np.array([nearest_class(v) for v in preds])
    accuracy = np.mean(classes == Y_raw) * 100
    mae = np.mean(np.abs(preds - Y_raw))
    return accuracy, mae

def run_config(hidden, title):
    print(title)
    out = {'nets': [], 'epochs': [], 'errors': [], 'accs': [],
           'maes': [], 'reached': [], 'dead': []}
    for s in seeds:
        net = Net(s, hidden)
        epochs, err, reached = net.train()
        acc, mae = compute_metrics(net)
        for key, val in zip(out, [net, epochs, err, acc, mae, reached,
                                  net.dead_neurons()]):
            out[key].append(val)
        print(f"Запуск {s}")
        print(f"Эпох: {epochs} | Es={err:.6f} | Acc={acc:.1f}% | MAE={mae:.4f}"
              f" | Ee достигнут: {'да' if reached else 'нет'}")
    print()
    return out

def plot_convergence(net_s, net_r):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(net_s.hist) + 1), net_s.hist,
             color='gold', linewidth=2, label='Sigmoid')
    plt.plot(range(1, len(net_r.hist) + 1), net_r.hist,
             color='green', linewidth=2, label='ReLU')
    plt.axhline(err_limit, color='red', linestyle='--', label=f'Ee = {err_limit}')
    plt.yscale('log')
    plt.xlabel('Номер эпохи')
    plt.ylabel('Суммарная ошибка Es')
    plt.title(f'Сходимость Sigmoid и ReLU (seed={show_seed}, вариант 7)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_epochs(res_s, res_r):
    plt.figure(figsize=(10, 6))
    pos = np.arange(len(seeds))
    w = 0.35
    for i in range(len(seeds)):
        plt.bar(pos[i] - w / 2, res_s['epochs'][i], w, color='gold',
                edgecolor='black', hatch='' if res_s['reached'][i] else '//',
                label='Sigmoid' if i == 0 else '')
        plt.bar(pos[i] + w / 2, res_r['epochs'][i], w, color='green',
                edgecolor='black', hatch='' if res_r['reached'][i] else '//',
                label='ReLU' if i == 0 else '')
    plt.xlabel('Seed')
    plt.ylabel('Количество эпох')
    plt.title('Число эпох по 5 запускам (штриховка — Ee не достигнут)')
    plt.xticks(pos, seeds)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def plot_surfaces(net_s, net_r):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    grid = np.linspace(-10, 10, 100)
    A, B = np.meshgrid(grid, grid)

    for ax, net, title in zip(axes, [net_s, net_r], ['Sigmoid', 'ReLU']):
        Z = np.array([[net.predict_norm(a, b) for a in grid] for b in grid])
        cs = ax.contourf(A, B, Z, levels=20, cmap='YlGn_r', alpha=0.85)
        fig.colorbar(cs, ax=ax, label='нормализованный выход y')
        ax.contour(A, B, Z, levels=[0.5], colors='black', linewidths=2)

        for a, b, t in data:
            ax.scatter(a, b, s=200, edgecolors='black', linewidths=2, zorder=5,
                       color='blue' if t == c_zero else 'red')

        ax.set_xlabel('A')
        ax.set_ylabel('B')
        ax.set_title(title)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f'Выход сети на плоскости (A, B): синие — класс {c_zero}, '
                 f'красные — класс {c_one}', fontsize=13)
    plt.tight_layout()
    plt.show()

def plot_mae(mae_s, mae_r):
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(['Sigmoid', 'ReLU'], [mae_s, mae_r],
                  color=['gold', 'green'], edgecolor='black', width=0.6)
    for bar, val in zip(bars, [mae_s, mae_r]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=12)
    ax.set_ylabel('MAE в шкале [c0; c1]')
    ax.set_title('Точность восстановления шкалы')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def print_summary(res_s, res_r):
    print("Результаты (среднее по 5 запускам):")
    for name, r in [("Sigmoid", res_s), ("ReLU", res_r)]:
        ep = ", ".join(str(e) for e in r['epochs'])
        print(f"{name}: эпохи: {ep} | Es={np.mean(r['errors']):.6f} | "
              f"Acc={np.mean(r['accs']):.1f}% | MAE={np.mean(r['maes']):.4f} | "
              f"сошлось: {sum(r['reached'])}/{len(seeds)}")
    print(f"Полностью мёртвых нейронов ReLU по запускам: {res_r['dead']}")
    print("Активность скрытых нейронов ReLU (число входов из 4, где нейрон != 0):")
    for s, net, ok in zip(seeds, res_r['nets'], res_r['reached']):
        print(f"  seed={s}: {net.active_counts()} | Ee достигнут: {'да' if ok else 'нет'}")

def print_predictions(net, title):
    print(f"\nПредсказания {title}")
    for a, b, t in data:
        norm = net.predict_norm(a, b)
        real = unscale(norm)
        print(f"({a:5.0f},{b:5.0f}) | y_norm={norm:.4f} | y={real:7.3f} | "
              f"class={nearest_class(real):5.0f} | expected={t:5.0f}")
    print("Дополнительные пары:")
    for a, b in [(-5, -8), (-8, -5), (-4.5, -4.5)]:
        norm = net.predict_norm(a, b)
        real = unscale(norm)
        print(f"({a:5.1f},{b:5.1f}) | y_norm={norm:.4f} | y={real:7.3f} | "
              f"class={nearest_class(real):5.0f}")

def run_interactive(net):
    print("\n=== Режим функционирования (ReLU) ===")
    print("Введите A B из диапазона [-10; 10] через пробел, для выхода — q")
    while True:
        raw = input("A B: ").strip()
        if raw.lower() == "q":
            break
        try:
            a, b = map(float, raw.split())
        except ValueError:
            print("Нужно ввести два числа через пробел.")
            continue
        if not (-10 <= a <= 10 and -10 <= b <= 10):
            print("A и B должны быть в диапазоне [-10; 10].")
            continue
        norm = net.predict_norm(a, b)
        real = unscale(norm)
        print(f"Нормализованный выход: {norm:.4f}")
        print(f"Выход в шкале [c0; c1]: {real:.4f}")
        print(f"Класс: {nearest_class(real):g}")

def main():
    print("Лабораторная работа №3")
    print(f"Вариант 7: c0 = {c_zero}, c1 = {c_one}\n")

    res_s = run_config('sigmoid', "Конфигурация А (Sigmoid)")
    res_r = run_config('relu', "Конфигурация В (ReLU)")
    print_summary(res_s, res_r)

    i = seeds.index(show_seed)
    net_s, net_r = res_s['nets'][i], res_r['nets'][i]

    print_predictions(net_s, "Sigmoid")
    print_predictions(net_r, "ReLU")

    plot_convergence(net_s, net_r)
    plot_epochs(res_s, res_r)
    plot_surfaces(net_s, net_r)
    plot_mae(res_s['maes'][i], res_r['maes'][i])
    run_interactive(net_r)

if __name__ == "__main__":
    main()