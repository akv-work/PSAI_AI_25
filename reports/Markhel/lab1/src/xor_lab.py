"""
Лабораторная работа 1
Многослойный персептрон (2-2-1) для задачи XOR с числовыми классами.

Вариант 6: c0 = -9, c1 = 6
Таблица истинности:
    A    B    A xor B
   -9   -9      -9
   -9    6       6
    6   -9       6
    6    6      -9

Допустимый диапазон входов/выходов: [-10; 10].
Активация - сигмоида на всех слоях. Обучение - backprop, онлайн-режим.
Критерий остановки: суммарная ошибка Es <= Ee, либо max_epochs.
для запуска -
cd "C:\Users\user\Desktop\4 курс 1\MRZVS\1LAB"
.venv\Scripts\python xor_lab.py
"""

import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

C0, C1 = -9.0, 6.0
LO, HI = -10.0, 10.0

X_RAW = np.array([[-9.0, -9.0],
                   [-9.0,  6.0],
                   [ 6.0, -9.0],
                   [ 6.0,  6.0]])
T_RAW = np.array([-9.0, 6.0, 6.0, -9.0])

def normalize(x):
    return (np.asarray(x, dtype=float) - LO) / (HI - LO)


def denormalize(y):
    return np.asarray(y, dtype=float) * (HI - LO) + LO


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid_from_output(y):
    """Производная сигмоиды y=sigmoid(x)."""
    return y * (1.0 - y)


X = normalize(X_RAW)
T = normalize(T_RAW)

class MLP221:
    def __init__(self, lr=0.5, seed=None):
        rng = np.random.default_rng(seed)
        self.W1 = rng.uniform(-0.5, 0.5, (2, 2))
        self.b1 = rng.uniform(-0.5, 0.5, (2,))
        self.W2 = rng.uniform(-0.5, 0.5, (2, 1))
        self.b2 = rng.uniform(-0.5, 0.5, (1,))
        self.lr = lr

    def forward(self, x):
        self.h_in = x @ self.W1 + self.b1
        self.h_out = sigmoid(self.h_in)
        self.o_in = self.h_out @ self.W2 + self.b2
        self.o_out = sigmoid(self.o_in)
        return self.o_out

    def train_step(self, x, t):
        o = self.forward(x)
        err = t - o
        delta_out = err * dsigmoid_from_output(o)
        delta_hidden = (delta_out @ self.W2.T) * dsigmoid_from_output(self.h_out)

        self.W2 += self.lr * np.outer(self.h_out, delta_out)
        self.b2 += self.lr * delta_out
        self.W1 += self.lr * np.outer(x, delta_hidden)
        self.b1 += self.lr * delta_hidden
        return float(np.sum(err ** 2))

    def train(self, X, T, Ee=0.001, max_epochs=50000, rng=None):
        rng = rng or np.random.default_rng(0)
        n = len(X)
        Es = float("inf")
        for epoch in range(1, max_epochs + 1):
            order = rng.permutation(n)
            Es = 0.0
            for i in order:
                Es += self.train_step(X[i], T[i:i + 1])
            if Es <= Ee:
                return epoch, Es
        return max_epochs, Es

    def predict(self, x):
        return float(self.forward(np.asarray(x, dtype=float)).item())

class SingleLayerPerceptron:
    def __init__(self, lr=0.5, seed=None):
        rng = np.random.default_rng(seed)
        self.W = rng.uniform(-0.5, 0.5, (2, 1))
        self.b = rng.uniform(-0.5, 0.5, (1,))
        self.lr = lr

    def forward(self, x):
        self.o_in = x @ self.W + self.b
        self.o_out = sigmoid(self.o_in)
        return self.o_out

    def train_step(self, x, t):
        o = self.forward(x)
        err = t - o
        delta = err * dsigmoid_from_output(o)
        self.W += self.lr * np.outer(x, delta)
        self.b += self.lr * delta
        return float(np.sum(err ** 2))

    def train(self, X, T, Ee=0.001, max_epochs=50000, rng=None):
        rng = rng or np.random.default_rng(0)
        n = len(X)
        Es = float("inf")
        for epoch in range(1, max_epochs + 1):
            order = rng.permutation(n)
            Es = 0.0
            for i in order:
                Es += self.train_step(X[i], T[i:i + 1])
            if Es <= Ee:
                return epoch, Es
        return max_epochs, Es

    def predict(self, x):
        return float(self.forward(np.asarray(x, dtype=float)).item())

def classify(y_real):

    return C0 if abs(y_real - C0) <= abs(y_real - C1) else C1


def evaluate(model, X, T_RAW):
    outputs_real = []
    correct = 0
    for x, t_raw in zip(X, T_RAW):
        y_norm = model.predict(x)
        y_real = denormalize(y_norm)
        outputs_real.append(y_real)
        if classify(y_real) == t_raw:
            correct += 1
    accuracy = correct / len(X)
    return outputs_real, accuracy


def run_experiment(name, model, Ee, max_epochs, seed):
    rng = np.random.default_rng(seed)
    epochs, Es = model.train(X, T, Ee=Ee, max_epochs=max_epochs, rng=rng)
    outputs_real, accuracy = evaluate(model, X, T_RAW)
    converged = Es <= Ee

    print(f"=== {name} ===")
    print(f"Критерий остановки Ee = {Ee}, максимум эпох = {max_epochs}")
    print(f"Эпох затрачено: {epochs}  |  Сходимость достигнута: {'да' if converged else 'нет'}")
    print(f"Итоговая суммарная ошибка Es = {Es:.6f}")
    print(f"{'A':>6}{'B':>6}{'A xor B (цель)':>16}{'выход сети':>14}{'класс':>8}")
    for (a, b), t_raw, y_real in zip(X_RAW, T_RAW, outputs_real):
        print(f"{a:6.0f}{b:6.0f}{t_raw:16.0f}{y_real:14.3f}{classify(y_real):8.0f}")
    print(f"Accuracy = {accuracy * 100:.1f}%")
    print()
    return {"name": name, "epochs": epochs, "Es": Es, "converged": converged,
            "accuracy": accuracy, "outputs_real": outputs_real}


def demo_predictions(mlp, slp, examples):
    print("=== Режим функционирования (проверочные примеры) ===")
    print(f"{'A':>6}{'B':>6}{'MLP y':>10}{'MLP класс':>12}{'SLP y':>10}{'SLP класс':>12}")
    for a, b in examples:
        x = normalize(np.array([a, b]))
        y_mlp = denormalize(mlp.predict(x))
        y_slp = denormalize(slp.predict(x))
        print(f"{a:6.1f}{b:6.1f}{y_mlp:10.3f}{classify(y_mlp):12.1f}"
              f"{y_slp:10.3f}{classify(y_slp):12.1f}")
    print()


def interactive_loop(mlp, slp):
    print("=== Интерактивный режим ===")
    print(f"Введите пару чисел A, B из диапазона [{LO:.0f}; {HI:.0f}] через пробел")
    print("(например: 3 -5). Для выхода введите 'q'.")
    while True:
        try:
            raw = input("A B > ").strip()
        except EOFError:
            break
        if raw.lower() in ("q", "quit", "exit"):
            break
        parts = raw.replace(",", " ").split()
        if len(parts) != 2:
            print("Нужно ввести ровно два числа.")
            continue
        try:
            a, b = float(parts[0]), float(parts[1])
        except ValueError:
            print("Не удалось распознать числа.")
            continue
        if not (LO <= a <= HI and LO <= b <= HI):
            print(f"Значения должны быть в диапазоне [{LO:.0f}; {HI:.0f}].")
            continue

        x = normalize(np.array([a, b]))
        y_mlp = denormalize(mlp.predict(x))
        y_slp = denormalize(slp.predict(x))
        print(f"  МСП (2-2-1): y = {y_mlp:.3f}  -> ближе к классу {classify(y_mlp):.0f} "
              f"({'c0' if classify(y_mlp) == C0 else 'c1'})")
        print(f"  Однослойный: y = {y_slp:.3f}  -> ближе к классу {classify(y_slp):.0f} "
              f"({'c0' if classify(y_slp) == C0 else 'c1'})")


if __name__ == "__main__":
    Ee = 0.001
    max_epochs = 50000

    mlp = MLP221(lr=0.5, seed=1)
    mlp_result = run_experiment("Многослойный персептрон 2-2-1", mlp, Ee, max_epochs, seed=1)

    slp = SingleLayerPerceptron(lr=0.5, seed=1)
    slp_result = run_experiment("Однослойный персептрон (без скрытого слоя)", slp, Ee, max_epochs, seed=1)

    print("=== Сравнение ===")
    print(f"{'Модель':35}{'Эпохи':>10}{'Es':>12}{'Accuracy':>12}")
    for r in (mlp_result, slp_result):
        epochs_str = str(r["epochs"]) if r["converged"] else f">{r['epochs']} (не сошёлся)"
        print(f"{r['name']:35}{epochs_str:>10}{r['Es']:>12.5f}{r['accuracy'] * 100:>11.1f}%")
    print()

    demo_examples = [(-9, -9), (-9, 6), (6, -9), (6, 6), (0, 0), (-10, 10), (3, -7), (10, 10)]
    demo_predictions(mlp, slp, demo_examples)

    if sys.stdin.isatty():
        interactive_loop(mlp, slp)
    else:
        print("(stdin не является интерактивным терминалом - интерактивный режим пропущен;"
              " запустите `python xor_lab.py` в консоли, чтобы ввести свои A, B)")
