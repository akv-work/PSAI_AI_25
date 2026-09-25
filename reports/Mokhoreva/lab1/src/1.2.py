import numpy as np

CLASS_ZERO = 3.0
CLASS_ONE = -8.0
LEARNING_RATE = 0.5
ERROR_THRESHOLD = 0.01
EPOCH_LIMIT = 10_000


def sigma(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigma_prime(output):
    return output * (1.0 - output)


def to_unit(value):
    return (value - CLASS_ZERO) / (CLASS_ONE - CLASS_ZERO)


def from_unit(value):
    return CLASS_ZERO + value * (CLASS_ONE - CLASS_ZERO)


def nearest_class(value):
    return CLASS_ZERO if abs(value - CLASS_ZERO) <= abs(value - CLASS_ONE) else CLASS_ONE


def build_dataset():
    raw = np.array([
        [CLASS_ZERO, CLASS_ZERO, CLASS_ZERO],
        [CLASS_ZERO, CLASS_ONE,  CLASS_ONE],
        [CLASS_ONE,  CLASS_ZERO, CLASS_ONE],
        [CLASS_ONE,  CLASS_ONE,  CLASS_ZERO],
    ], dtype=float)
    return raw[:, :2], raw[:, 2]


class SingleNeuron:
    def __init__(self, rng_seed=1):
        rng = np.random.default_rng(rng_seed)
        self.weights = rng.uniform(-0.1, 0.1, size=2)
        self.bias = float(rng.uniform(-0.1, 0.1))

    def forward(self, sample):
        z = sample @ self.weights + self.bias
        return z, sigma(z)

    def fit(self, inputs, targets):
        for epoch in range(1, EPOCH_LIMIT + 1):
            for sample, target in zip(inputs, targets):
                _, out = self.forward(sample)
                delta = (out - target) * sigma_prime(out)
                self.weights -= LEARNING_RATE * sample * delta
                self.bias -= LEARNING_RATE * delta

            total = self.total_error(inputs, targets)
            if total <= ERROR_THRESHOLD:
                return epoch, total, True

        return EPOCH_LIMIT, self.total_error(inputs, targets), False

    def total_error(self, inputs, targets):
        err = 0.0
        for sample, target in zip(inputs, targets):
            _, out = self.forward(sample)
            err += 0.5 * (out - target) ** 2
        return float(err)

    def predict(self, sample):
        _, out = self.forward(to_unit(sample))
        return float(from_unit(out))


def accuracy(net, inputs, targets):
    hits = sum(
        nearest_class(net.predict(x)) == t
        for x, t in zip(inputs, targets)
    )
    return hits / len(targets) * 100


def show_table(net, inputs, targets):
    print(f"\n{'A':>7} {'B':>7} {'target':>9} {'y':>12} {'class':>8}")
    print("-" * 50)
    for x, t in zip(inputs, targets):
        y = net.predict(x)
        print(f"{x[0]:7.2f} {x[1]:7.2f} {t:9.2f} {y:12.6f} {nearest_class(y):8.2f}")


def main():
    inputs_raw, targets_raw = build_dataset()
    inputs = to_unit(inputs_raw)
    targets = to_unit(targets_raw)

    net = SingleNeuron(rng_seed=1)
    epochs, err, converged = net.fit(inputs, targets)
    acc = accuracy(net, inputs_raw, targets_raw)

    print("=" * 55)
    print(f"SLP | XOR | c0 = {CLASS_ZERO}, c1 = {CLASS_ONE}")
    print("=" * 55)
    print(f"Эпох до остановки : {epochs}")
    print(f"Суммарная ошибка  : {err:.8f}")
    print(f"Точность          : {acc:.2f}%")
    print(f"Сошлась           : {'да' if converged else 'нет'}")

    show_table(net, inputs_raw, targets_raw)

    print("\nВведите A и B из [-10; 10] (q — выход)")
    while True:
        raw_a = input("A = ").strip()
        if raw_a.lower() == "q":
            break
        raw_b = input("B = ").strip()
        if raw_b.lower() == "q":
            break
        try:
            a, b = float(raw_a), float(raw_b)
        except ValueError:
            print("Нужны числа.")
            continue
        if not (-10 <= a <= 10 and -10 <= b <= 10):
            print("Диапазон: [-10; 10].")
            continue
        y = net.predict(np.array([a, b]))
        print(f"  y = {y:.6f} | класс = {nearest_class(y)}")


if __name__ == "__main__":
    main()