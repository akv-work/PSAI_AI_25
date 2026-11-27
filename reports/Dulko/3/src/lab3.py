from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

C0, C1 = 7, -7
RAW = np.array([[7, 7], [7, -7], [-7, 7], [-7, -7]], float)
X = RAW / 10
T = np.array([0, 1, 1, 0], float)
Y = C0 + T * (C1 - C0)
LR = 0.3
EE = 0.01
MAX_EPOCHS = 10000
SEEDS = [0, 4, 5, 32, 60]
OUT = Path(__file__).resolve().parent / "figures"


def sigmoid(s):
    return 1 / (1 + np.exp(-np.clip(s, -500, 500)))


class MLP:
    def __init__(self, rng, activation):
        self.activation = activation
        self.W1 = rng.uniform(-0.5, 0.5, (2, 2))
        self.b1 = rng.uniform(-0.5, 0.5, 2)
        self.W2 = rng.uniform(-0.5, 0.5, 2)
        self.b2 = rng.uniform(-0.5, 0.5)

    def forward(self, x):
        s = x @ self.W1 + self.b1
        h = np.maximum(s, 0) if self.activation == "ReLU" else sigmoid(s)
        return h, sigmoid(h @ self.W2 + self.b2)

    def step(self, x, target):
        h, y = self.forward(x)
        delta = y - target
        derivative = (h > 0) if self.activation == "ReLU" else h * (1 - h)
        hidden_delta = delta * self.W2 * derivative
        self.W2 -= LR * delta * h
        self.b2 -= LR * delta
        self.W1 -= LR * np.outer(x, hidden_delta)
        self.b1 -= LR * hidden_delta


def train(activation, seed):
    rng = np.random.default_rng(seed)
    net = MLP(rng, activation)
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        for i in rng.permutation(4):
            net.step(X[i], T[i])
        _, y = net.forward(X)
        y = np.clip(y, 1e-12, 1 - 1e-12)
        error = float(-np.sum(T * np.log(y) + (1 - T) * np.log(1 - y)))
        history.append(error)
        if error <= EE:
            break
    h, y = net.forward(X)
    real = C0 + y * (C1 - C0)
    return dict(seed=seed, net=net, history=history, epochs=epoch, error=error,
                ok=error <= EE, accuracy=np.mean((y >= 0.5) == T),
                mae=np.mean(np.abs(real - Y)), dead=int(np.sum(np.all(h == 0, axis=0))))


def statistics(name, runs):
    print(f"\n{name}, шаг {LR}, Ee {EE}, максимум {MAX_EPOCHS} эпох")
    print("seed   эпохи      Es     accuracy     MAE    сошлось  нулевых нейронов")
    for r in runs:
        print(f"{r['seed']:4} {r['epochs']:7} {r['error']:9.6f} "
              f"{r['accuracy']:10.0%} {r['mae']:8.4f} "
              f"{'да' if r['ok'] else 'нет':>8} {r['dead']:8}")
    epochs = [r['epochs'] for r in runs]
    print(f"Сошлось {sum(r['ok'] for r in runs)}/5. Эпохи min/max: {min(epochs)}/{max(epochs)}")


def representative(runs):
    successful = [r for r in runs if r['ok']]
    return (successful or runs)[0]


def plots(results):
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    grid = np.linspace(-10, 10, 180)
    a, b = np.meshgrid(grid, grid)
    points = np.stack([a, b], axis=-1) / 10
    for j, (name, runs) in enumerate(results.items()):
        r = representative(runs)
        axes[0, 0].plot(np.arange(1, r['epochs'] + 1), r['history'],
                        label=f"{name}, seed {r['seed']}")
        bars = axes[0, 1].bar(np.arange(5) + (j - 0.5) * 0.36,
                              [v['epochs'] for v in runs], 0.36, label=name)
        for bar, run in zip(bars, runs):
            if not run['ok']:
                bar.set_hatch("///")
        _, y = r['net'].forward(points)
        real = C0 + y * (C1 - C0)
        ax = axes[1, j]
        image = ax.contourf(a, b, real, np.linspace(-7, 7, 25), cmap="RdYlBu")
        ax.contour(a, b, real, levels=[0], colors="black", linewidths=1)
        ax.scatter(RAW[T == 0, 0], RAW[T == 0, 1], c="green", edgecolors="black", label="класс 7")
        ax.scatter(RAW[T == 1, 0], RAW[T == 1, 1], c="purple", edgecolors="black", label="класс -7")
        ax.set(xlabel="A", ylabel="B", title=f"{name}, seed {r['seed']}", aspect="equal")
        ax.legend(fontsize=10)
        fig.colorbar(image, ax=ax, label="Выход в исходной шкале")
    axes[0, 0].axhline(EE, color="black", linestyle=":", label="Ee")
    axes[0, 0].set(xlabel="Эпоха", ylabel="Es", yscale="log", title="Сходимость")
    axes[0, 1].set(xlabel="seed", ylabel="Эпохи", title="Штриховка: Ee не достигнута",
                    xticks=np.arange(5), xticklabels=SEEDS)
    for ax in axes[0]:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "lab3.png", dpi=150)
    plt.show()


def show(net, a, b):
    _, y = net.forward(np.array([a, b]) / 10)
    real = C0 + y * (C1 - C0)
    closest = C0 if abs(real - C0) <= abs(real - C1) else C1
    print(f"A {a:g}, B {b:g}: y {y:.4f}, исходная шкала {real:.4f}, класс {closest}")


def functioning(results):
    for name, runs in results.items():
        run = representative(runs)
        print(f"\n{name}, seed {run['seed']}")
        for a, b in np.vstack([RAW, [[6, -6], [-6, 6], [0, 0]]]):
            show(run['net'], a, b)
    print("\nВведите A B из [-10; 10]. Пустая строка: выход.")
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if not text:
            break
        try:
            a, b = map(float, text.split())
            if not (-10 <= a <= 10 and -10 <= b <= 10):
                raise ValueError
        except ValueError:
            print("Нужны два числа из [-10; 10].")
            continue
        for name, runs in results.items():
            print(name, end=": ")
            show(representative(runs)['net'], a, b)


def main():
    results = {}
    for name in ["Сигмоида", "ReLU"]:
        results[name] = [train(name, seed) for seed in SEEDS]
        statistics(name, results[name])
    plots(results)
    functioning(results)


if __name__ == "__main__":
    main()
